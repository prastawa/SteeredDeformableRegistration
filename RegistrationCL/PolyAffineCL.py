
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


#TODO:
# use volume node / vtkimage as inputs, only convert blocks to ImageCL
# as needed

from ImageCL import ImageCL
from DeformationCL import DeformationCL

import numpy

import random


class PolyAffineCL:

  def __init__(self, fixedCL, movingCL):
    self.centers = []
    self.radii = []
    self.affines = []
    self.translations = []

    self.fixedCL = fixedCL

    self.origMovingCL = movingCL
    self.movingCL = movingCL

    self.convergenceRatio = 1e-5

  def create_identity(self, number_per_axis=3):
    """Identity transform with equal number of affines at each axis"""
    self.centers = []
    self.radii = []
    self.affines = []
    self.translations = []

    shape = self.fixedCL.shape
    spacing = self.fixedCL.spacing

    rad = numpy.ones((3,1), dtype=numpy.float32)
    for d in range(3):
      rad[d] =  (shape[d]-1) * spacing[d] / (number_per_axis+1)

    A0 = numpy.zeros((3,3), dtype=numpy.float32)
    #A0 = numpy.eye(3, dtype=numpy.float32)
    T0 = numpy.zeros((3,1), dtype=numpy.float32)

    for i in range(number_per_axis):
      cx = (i+1) * (shape[0]-1) * spacing[0] / (number_per_axis+1)
      for j in range(number_per_axis):
        cy = (j+1) * (shape[1]-1) * spacing[1] / (number_per_axis+1)
        for k in range(number_per_axis):
          cz = (k+1) * (shape[2]-1) * spacing[2] / (number_per_axis+1)

          C = numpy.array([cx, cy, cz], dtype=numpy.float32)
          C = C.reshape(3, 1)

          #print "Adding affine at center", C, "radius", rad

          self.centers.append(C)
          self.radii.append(rad)
          self.affines.append(A0)
          self.translations.append(T0)

    print "Created identity with", len(self.affines), "affine transforms"

  def optimize(self, maxIters=10):
    """Optimize polyaffine parameters using adaptive step gradient descent"""

    #TODO: separate one iteration as a method for feedback updates
    # separate by params as well?
    # Keep track of step sizes and iteration number
    # method for init, if iter == 0 set up variables (ref error, steps, etc)

    AList = self.affines
    TList = self.translations

    CList = self.centers

    numTransforms = len(AList)

    stepA = 1.0
    stepT = 1.0
    stepC = 1.0

    prevErrorL2 = float('Inf')

    DiffFM = self.fixedCL.subtract(self.movingCL)
    DiffFMSq = DiffFM.multiply(DiffFM)
    errorL2 = DiffFMSq.sum()

    refErrorL2 = errorL2
    print "Ref diff", refErrorL2

    # Alternating gradient descent with adaptive step sizes
    for iter in range(maxIters):

      # Translation
      dTList = self.gradient_translation()

      if iter == 0:
        max_dT = 1e-10
        for q in range(numTransforms):
          max_dT = max(numpy.max(numpy.abs(dTList[q])), max_dT)
        stepT = 5.0 / max_dT

      print "Line search trans"
      for lineIter in range(20):
        print "opt line iter", lineIter

        TTestList = list(TList)
        for q in range(numTransforms):
          TTestList[q] = TList[q] - dTList[q]*stepT

        M = self.warp_moving(self.affines, TTestList, self.centers)

        DiffFM = self.fixedCL.subtract(M)
        DiffFMSq = DiffFM.multiply(DiffFM)
        errorL2Test = DiffFMSq.sum()

        print "Test diff", errorL2Test

        if errorL2Test < errorL2:
          TList = TTestList

          # TODO: figure out book-keeping, shouldn't have duplicates like this
          # gradient and warp_moving should be aware of which to use
          self.translations = TList

          stepT *= 1.2

          prevErrorL2 = errorL2
          errorL2 = errorL2Test

          self.movingCL = M

          print "PolyAffine error=", errorL2

          break
        else:
          stepT *= 0.5

      if abs(errorL2 - prevErrorL2) < self.convergenceRatio * abs(errorL2 - refErrorL2):
        break

      # Affine
      dAList = self.gradient_affine()

      if iter == 0:
        max_dA = 1e-10
        for q in range(numTransforms):
          max_dA = max(numpy.max(numpy.abs(dAList[q])), max_dA)
        stepA = 0.05 / max_dA

      print "Line search affine"
      for lineIter in range(20):
        print "opt line iter", lineIter

        ATestList = list(AList)
        for q in range(numTransforms):
          ATestList[q] = AList[q] - dAList[q]*stepA

        M = self.warp_moving(ATestList, self.translations, self.centers)

        DiffFM = self.fixedCL.subtract(M)
        DiffFMSq = DiffFM.multiply(DiffFM)
        errorL2Test = DiffFMSq.sum()

        print "Test diff", errorL2Test

        if errorL2Test < errorL2:
          AList = ATestList

          # TODO: figure out book-keeping, shouldn't have duplicates like this
          # gradient and warp_moving should be aware of which to use
          self.affines = AList

          stepA *= 1.2

          prevErrorL2 = errorL2
          errorL2 = errorL2Test

          self.movingCL = M

          print "PolyAffine error=", errorL2

          break
        else:
          stepA *= 0.5

      if abs(errorL2 - prevErrorL2) < self.convergenceRatio * abs(errorL2 - refErrorL2):
        break

      # Anchor positions
      dCList = self.gradient_anchor()

      if iter == 0:
        max_dC = 1e-10
        for q in range(numTransforms):
          max_dC = max(numpy.max(numpy.abs(dCList[q])), max_dC)
        stepC = 5.0 / max_dC

      print "Line search anchor"
      for lineIter in range(20):
        print "opt line iter", lineIter

        CTestList = list(CList)
        for q in range(numTransforms):
          CTestList[q] = CList[q] - dCList[q]*stepC

        M = self.warp_moving(self.affines, self.translations, CTestList)

        DiffFM = self.fixedCL.subtract(M)
        DiffFMSq = DiffFM.multiply(DiffFM)
        errorL2Test = DiffFMSq.sum()

        print "Test diff", errorL2Test

        if errorL2Test < errorL2:
          CList = CTestList

          # TODO: figure out book-keeping, shouldn't have duplicates like this
          # gradient and warp_moving should be aware of which to use
          self.centers = CTestList

          stepC *= 1.2

          prevErrorL2 = errorL2
          errorL2 = errorL2Test

          self.movingCL = M

          print "PolyAffine error=", errorL2

          break
        else:
          stepC *= 0.5

      if abs(errorL2 - prevErrorL2) < self.convergenceRatio * abs(errorL2 - refErrorL2):
        break

      print "opt iter", iter, "steps", stepA, stepT, stepC

    self.affines = AList
    self.translations = TList
    self.centers = CList

  def gradient(self):
    """Gradient of L2 norm"""

    numTransforms = len(self.centers)

    gradA_list = []
    gradT_list = []

    gradC_list = []
    gradR_list = []

    """
    sx = self.fixedCL.shape[0]
    sy = self.fixedCL.shape[1]
    sz = self.fixedCL.shape[2]

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
          gradA[i,j] = -2.0 * WD.multiply(GX).sum()

      gradT = numpy.zeros((3,1), dtype=numpy.float32)
      for d in range(3):
        gradT[d] = -2.0 * WD.multiply(GList[d]).sum()

      gradC = numpy.zeros((3,1), dtype=numpy.float32)
      gradR = numpy.zeros((3,1), dtype=numpy.float32)

      dot_AT_XC = F.clone()
      dot_AT_XC.fill(0.0)

      dot_AT_XR = F.clone()
      dot_AT_XR.fill(0.0)

      for d in range(3):
        AT = F.clone()
        AT.fill(0.0)
        for j in range(3):
          Y = XList[d].clone()
          Y.scale(A[d,j])
          AT.add_inplace(Y)
        AT.shift(T[d])

        XC = XList[d].clone()
        XC.shift(-C[d])
        XC.scale(2.0 / r[d]**2)

        dot_AT_XC.add_inplace(AT.multiply(XC))

        XR = XList[d].clone()
        XR.shift(-C[d])
        XR.scale(4.0 / r[d]**3)

        dot_AT_XR.add_inplace(AT.multiply(XR))

      for d in range(3):
        gradC[d] = -WD.multiply(GList[d].multiply(dot_AT_XC)).sum()
        gradR[d] = WD.multiply(GList[d].multiply(dot_AT_XR)).sum()

      gradA_list.append(gradA)
      gradT_list.append(gradT)

      gradC_list.append(gradC)
      gradR_list.append(gradR)

    return gradA_list, gradT_list, gradC_list, gradR_list

  def gradient_affine(self):
    """Gradient of L2 norm for affine matrices only"""

    numTransforms = len(self.centers)

    gradA_list = []

    """
    sx = self.fixedCL.shape[0]
    sy = self.fixedCL.shape[1]
    sz = self.fixedCL.shape[2]

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
          gradA[i,j] = -2.0 * WD.multiply(GX).sum()

      gradA_list.append(gradA)

    return gradA_list
      
  def gradient_translation(self):
    """Gradient of L2 norm for translations only"""

    numTransforms = len(self.centers)

    gradT_list = []

    """
    sx = self.fixedCL.shape[0]
    sy = self.fixedCL.shape[1]
    sz = self.fixedCL.shape[2]

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

      gradT = numpy.zeros((3,1), dtype=numpy.float32)
      for d in range(3):
        gradT[d] = -2.0 * WD.multiply(GList[d]).sum()

      gradT_list.append(gradT)

    return gradT_list

  def gradient_anchor(self):
    """Gradient of L2 norm for anchor positions only"""

    numTransforms = len(self.centers)

    gradC_list = []

    """
    sx = self.fixedCL.shape[0]
    sy = self.fixedCL.shape[1]
    sz = self.fixedCL.shape[2]

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

      gradC = numpy.zeros((3,1), dtype=numpy.float32)

      dot_AT_XC = F.clone()
      dot_AT_XC.fill(0.0)

      for d in range(3):
        AT = F.clone()
        AT.fill(0.0)
        for j in range(3):
          Y = XList[d].clone()
          Y.scale(A[d,j])
          AT.add_inplace(Y)
        AT.shift(T[d])

        XC = XList[d].clone()
        XC.shift(-C[d])
        XC.scale(2.0 / r[d]**2)

        dot_AT_XC.add_inplace(AT.multiply(XC))

      for d in range(3):
        gradC[d] = -WD.multiply(GList[d].multiply(dot_AT_XC)).sum()

      gradC_list.append(gradC)

    return gradC_list

  def warp_moving(self, AList, TList, CList):
    """
    Compute deformation field and update moving image.
    Returns warped version of origMovingCL with current poly-affine parameters.
    """

    numTransforms = len(self.affines)

    shape = self.fixedCL.shape

    # TODO: use applyPolyAffine kernel in ImageFunctions.cl

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
      #C = self.centers[q]

      A = AList[q]
      T = TList[q]
      C = CList[q]

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
          CoordCL[d].add_inplace(AWX)
          #CoordCL[d] = AWX
        WT = W.clone()
        WT.scale(T[d])
        CoordCL[d].add_inplace(WT)

    warpCL = DeformationCL(self.fixedCL)
    warpCL.set_mapping(CoordCL[0], CoordCL[1], CoordCL[2])

    return warpCL.applyTo(self.origMovingCL)

  # TODO: switch to CL kernels that compute weights
  def _gaussian(self, shape, center, radii):
    """Returns ImageCL object of Gaussian"""

    """
    sx = shape[0]
    sy = shape[1]
    sz = shape[2]

    Coord = numpy.mgrid[0:sx, 0:sy, 0:sz]
    #TODO: sync order with CL?
    #Coord = numpy.mgrid[0:sz, 0:sy, 0:sx]
    #Coord.reverse()
    for d in range(3):
      Coord[d] -= center[d]

    #G = numpy.ones(shape, dtype=numpy.float32)
    G = numpy.zeros(shape, dtype=numpy.float32)
    for d in range(3):
      G += (Coord[d] / radii[d]) ** 2.0
    G = numpy.exp(-0.5 * G)
    G /= numpy.max(G)
    #G /= numpy.sum(G)

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
    #gaussianCL.scale(1.0 / gaussianCL.sum())

    return gaussianCL

"""
  # Returns patches containing image momenta update
  # (fixed - moving) .* weight .* gradient(moving)
  def extract_momenta_patches(self, i, fixedImage, movingImage):
    F = self.extract_patch(i, fixedImage)
    M = self.extract_patch(i, movingImage)

    Wdiff = F.clone()

    numAffines = n.uint32(len(self.centers))

    # TODO: how?
    # apply only affines that are close to center i to get M?
    # apply all poly affines then extract M?
    # then multiply by weight of i / sum_i(w) ?
    fixedImage.clprogram.getWeightedDiffPolyAffine(
      F.clqueue, F.shape, None,
      center_array.data, width_array.data,
      numAffines,
      F.clarray.data,
      M.clarray.data,
      Wdiff.clarray.data
      ).wait()

    moms = M.gradient()
    for dim in xrange(3):
      moms[dim].multiply_inplace(Wdiff)

    return moms
    

  def apply(self, image):
    # Create CL array containing centers, weights, matrices, translations

    numAffines = n.uint32(len(self.centers))

    centers_array = cl.array.zeros(image.clqueue, (numAffines,3), n.float32)
    widths_array = cl.array.zeros(image.clqueue, (numAffines,1), n.float32)
    matrices_array = cl.array.zeros(image.clqueue, (numAffines,9), n.float32)
    trans_array = cl.array.zeros(image.clqueue, (numAffines,3), n.float32)

    for i in xrange(numAffines):
      centers_array[i,:] = self.centers[i]
      widths_array[i,0] = self.widths[i]
      # Matrix as list of list, convert to column vector
      matrices_array[i,:] = \
        [item for sublist in self.matrices[i] for item in sublist]
      trans_array[i,:] = self.translations[i]

    outimage = image.clone()

    # Pass to kernel that handles deformation
    image.clprogram.applyPolyAffine(
      image.clqueue, image.shape, None,
      centers_array.data, widths_array.data,
      matrices_array.data, trans_array.data,
      numAffines,
      image.clarray.data,
      outimage.clarray.data
      ).wait()

    return outimage

  # Move outside?
  def update_element(self, i, learning_rates):
    # Search radius
    radius = ceil(2*self.widths[i]) + 1

    # Extract patches at center i within search radius
    fixed_patch = polyafffine.extract_patch(fixed_image, i, center)
    moving_patch = polyafffine.extract_patch(output_image, i, center)

    # Gradient descent update on GPU

    # Update the transform parameters on CPU
    pass
"""
