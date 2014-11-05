
#
# PolyAffineCL: poly affine transform on CL managed data
#
# Transform is described as a set of local affine transforms that have
# regions of influence modeled as Gaussians
#
# This class provides CL implementations of parameter optimization and
# deformation
#
# Requires: ImageCL and DeformationCL
#
# Author: Marcel Prastawa (marcel.prastawa@gmail.com)
#


from ImageCL import ImageCL
from DeformationCL import DeformationCL

import numpy


class PolyAffineCL:

  def __init__(self, fixedCL, movingCL):
    self.centers = []
    self.radii = []
    self.affines = []
    self.translations = []

    self.fixedCL = fixedCL

    self.origMovingCL = movingCL
    self.movingCL = movingCL

  def create_identity(self, number_per_axis=3):
    """Identity transform with equal number of affines at each axis"""
    self.centers = []
    self.radii = []
    self.affines = []
    self.translations = []

    shape = self.fixedCL.shape
    spacing = self.fixedCL.spacing

    rad = numpy.array([1.0,1.0,1.0], dtype=numpy.float32)
    for d in range(3):
      rad[d] =  (shape[d]-1) * spacing[d] / (number_per_axis+1)

    A0 = numpy.zeros((3,3), dtype=numpy.float32)
    T0 = numpy.zeros((3,1), dtype=numpy.float32)

    for i in range(number_per_axis):
      cx = (i+1) * (shape[0]-1) * spacing[0] / (number_per_axis+1)
      for j in range(number_per_axis):
        cy = (j+1) * (shape[1]-1) * spacing[1] / (number_per_axis+1)
        for k in range(number_per_axis):
          cz = (k+1) * (shape[2]-1) * spacing[2] / (number_per_axis+1)

          C = numpy.array([cx, cy, cz], dtype=numpy.float32)

          print "Adding affine at center", C, "radius", rad

          self.centers.append(C)
          self.radii.append(rad)
          self.affines.append(A0)
          self.translations.append(T0)

    print "Created identity with", len(self.affines), "affine transforms"

  def optimize(self, maxIters=10):
    """Optimize polyaffine parameters using adaptive step gradient descent"""

    AList = self.affines
    TList = self.translations

    numTransforms = len(AList)

    stepA = 1.0
    stepT = 1.0

    prevErrorL2 = float('Inf')

    DiffFM = self.fixedCL.subtract(self.movingCL)
    DiffFMSq = DiffFM.multiply(DiffFM)
    errorL2 = DiffFMSq.sum()

    refErrorL2 = errorL2
    print "Ref diff", refErrorL2

    for iter in range(maxIters):
      dAList, dTList = self.gradient()

      if iter == 0:
        max_dA = 1e-10
        max_dT = 1e-10
        for q in range(numTransforms):
          max_dA = max(numpy.max(numpy.abs(dAList[q])), max_dA)
          max_dT = max(numpy.max(numpy.abs(dTList[q])), max_dT)
        stepA = 0.1 / max_dA
        stepT = 5.0 / max_dT

      print "opt iter", iter, "steps", stepA, stepT

      for lineIter in range(20):
        print "opt line iter", lineIter

        #AList = AList - dAList * stepA
        #TList = TList - dTList * stepT
        ATestList = list(AList)
        TTestList = list(TList)
        for q in range(numTransforms):
          ATestList[q] = AList[q] + dAList[q]*stepA
          TTestList[q] = TList[q] + dTList[q]*stepT
          #print "Test T", q, TTestList[q]

        M = self.warp_moving(ATestList, TTestList)

        DiffFM = self.fixedCL.subtract(M)
        DiffFMSq = DiffFM.multiply(DiffFM)
        errorL2Test = DiffFMSq.sum()

        print "Test diff", errorL2Test

        if errorL2Test < errorL2:
          AList = ATestList
          TList = TTestList

          stepA *= 1.2
          stepT *= 1.2

          prevErrorL2 = errorL2
          errorL2 = errorL2Test

          self.movingCL = M

          print "PolyAffine error=", errorL2

          break
        else:
          stepA *= 0.5
          stepT *= 0.5

      if abs(errorL2 - prevErrorL2) < 1e-4 * abs(errorL2 - refErrorL2):
        break

    self.affines = AList
    self.translations = TList

  def gradient(self):
    """Gradient of L2 norm"""

    numTransforms = len(self.centers)

    gradA_list = []
    gradT_list = []

    sx = self.fixedCL.shape[0]
    sy = self.fixedCL.shape[1]
    sz = self.fixedCL.shape[2]

    """
    Coord = numpy.mgrid[0:sx, 0:sy, 0:sz]

    CoordCL = []
    for d in range(3):
      cc = ImageCL(self.fixedCL.clqueue, self.fixedCL.clprogram)
      cc.fromArray(Coord[d], self.fixedCL.spacing)
      CoordCL.append(cc)
    """

    Phi = DeformationCL(self.fixedCL)
    Phi.set_identity()

    CoordCL = [Phi.hx, Phi.hy, Phi.hz]

    for q in range(numTransforms):
      C = self.centers[q]
      r = self.radii[q]
      A = self.affines[q]
      T = self.translations[q]

      F = self.fixedCL.getROI(C, r)
      M = self.movingCL.getROI(C, r)

      XList = []
      for d in range(3):
        XList.append(CoordCL[d].getROI(C, r))

      DiffFM = F.subtract(M)

      GList = M.gradient()

      CF = numpy.array(F.shape, dtype=numpy.float32) / 2.0
      W = self._gaussian(F.shape, CF, r/2.0)

      WD = W.multiply(DiffFM)

      gradA = numpy.zeros((3,3), dtype=numpy.float32)
      for i in range(3):
        for j in range(3):
          GX = GList[i].multiply(XList[j])
          gradA[i,j] = WD.multiply(GX).sum()

      gradT = numpy.zeros((3,1), dtype=numpy.float32)
      for d in range(3):
        gradT[d] = WD.multiply(GList[d]).sum()

      gradA_list.append(gradA)
      gradT_list.append(gradT)

    return [gradA_list, gradT_list]
      

  def warp_moving(self, AList, TList):
    """
    Compute deformation field and update moving image.
    Returns warped version of origMovingCL with current poly-affine parameters.
    """

    numTransforms = len(self.affines)

    shape = self.fixedCL.shape

    """
    WSum = None
    # TODO: compute sum of W for division
    # save computed W's in a list, avoid it to save GPU memory?
    #WList = []
    for q in range(numTransforms):
      C = self.centers[q]
      r = self.radii[q]

      W = self._gaussian(shape, C, r/2.0)
      if q == 0:
        WSum = W
      else:
        WSum = WSum.add(W)
    WSum.shift(1e-8)
    """

    """
    Coord = numpy.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    CoordCL = []
    for d in range(3):
      cc = ImageCL(self.fixedCL.clqueue, self.fixedCL.clprogram)
      cc.fromArray(Coord[d], self.fixedCL.spacing)
      CoordCL.append(cc)
    """

    Phi = DeformationCL(self.fixedCL)
    Phi.set_identity()

    CoordCL = [Phi.hx, Phi.hy, Phi.hz]

    CoordCL0 = list(CoordCL)

    for q in range(numTransforms):
      #A = self.affines[q]
      #T = self.translations[q]
      A = AList[q]
      T = TList[q]

      C = self.centers[q]
      r = self.radii[q]

      W = self._gaussian(shape, C, r/2.0)
      # W = W.divide(WSum)

      WX = []
      for d in range(3):
        WX.append(W.multiply(CoordCL0[d]))

      for d in range(3):
        for j in range(3):
          AWX = WX[j].clone()
          AWX.scale(A[d,j])
          CoordCL[d] = CoordCL[d].add(AWX)
        WT = W.clone()
        WT.scale(T[d])
        CoordCL[d] = CoordCL[d].add(WT)

    warpCL = DeformationCL(self.fixedCL)
    warpCL.set_mapping(CoordCL[0], CoordCL[1], CoordCL[2])

    return warpCL.applyTo(self.origMovingCL)

  def _gaussian(self, shape, center, radii):
    """Returns ImageCL object of Gaussian"""

    """
    sx = shape[0]
    sy = shape[1]
    sz = shape[2]

    Coord = numpy.mgrid[0:sx, 0:sy, 0:sz]
    for d in range(3):
      Coord[d] -= center[d]

    #G = numpy.ones(shape, dtype=numpy.float32)
    G = numpy.zeros(shape, dtype=numpy.float32)
    for d in range(3):
      G += (Coord[d] / radii[d]) ** 2.0
    G = numpy.exp(-0.5 * G)
    #G /= numpy.max(G)
    G /= numpy.sum(G)

    gaussianCL = ImageCL(self.fixedCL.clqueue, self.fixedCL.clprogram)
    gaussianCL.fromArray(G, self.fixedCL.spacing)
    """

    temp = ImageCL(self.fixedCL.clqueue, self.fixedCL.clprogram)
    temparr = numpy.zeros(shape, dtype=numpy.float32)
    temp.fromArray(temparr, self.fixedCL.spacing) 

    Phi = DeformationCL(temp)

    Phi.hx.shift(-center[0])
    Phi.hx.scale(1.0 / radii[0])
    G = Phi.hx.multiply(Phi.hx)

    Phi.hy.shift(-center[1])
    Phi.hy.scale(1.0 / radii[1])
    G = G.add(Phi.hy.multiply(Phi.hy))

    Phi.hz.shift(-center[2])
    Phi.hz.scale(1.0 / radii[2])
    G = G.add(Phi.hz.multiply(Phi.hz))

    G.scale(-0.5)
    gaussianCL = G.exp()
    gaussianCL.scale(1.0 / gaussianCL.max())

    return gaussianCL
