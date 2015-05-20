
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

# TODO:
# - use volume node / vtkimage as inputs, only convert blocks to ImageCL
#   as needed, make sure it can handle big images
# - anchor merging, blend two affines if anchors are close together
#   define objective function or use heuristic?

from ImageCL import ImageCL
from DeformationCL import DeformationCL

import pyopencl.array as cla

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

    self.origin = numpy.array(self.fixedCL.origin, dtype=numpy.single)

    self.sum_weights = None
    self.weights = []

    self.convergenceRatio = 1e-4

    self.lineSearchIterations = 5

    # Optimizer state
    self.optimIter = 0
    self.optimMode = 0

    self.stepA = -1.0
    self.stepT = -1.0
    self.stepC = -1.0

    self.refErrorL2 = 0.0
    self.currErrorL2 = 0.0
    self.prevErrorL2 = 0.0

  def create_identity(self, number_per_axis=3):
    """Identity transform with equal number of affines at each axis"""
    self.centers = []
    self.radii = []
    self.affines = []
    self.translations = []

    shape = self.fixedCL.shape
    spacing = self.fixedCL.spacing

    rad = numpy.ones((3,), dtype=numpy.single)
    for d in range(3):
      rad[d] =  (shape[d]-1) * spacing[d] / (number_per_axis+1)
      #rad[d] =  (shape[d]-1) * spacing[d] / (number_per_axis+1) * 1.5

    A0 = numpy.zeros((3,3), dtype=numpy.single)
    #A0 = numpy.eye(3, dtype=numpy.single)

    T0 = numpy.zeros((3,), dtype=numpy.single)

    for i in range(number_per_axis):
      cx = (i+1) * (shape[0]-1) * spacing[0] / (number_per_axis+1) + self.origin[0]
      for j in range(number_per_axis):
        cy = (j+1) * (shape[1]-1) * spacing[1] / (number_per_axis+1) + self.origin[1]
        for k in range(number_per_axis):
          cz = (k+1) * (shape[2]-1) * spacing[2] / (number_per_axis+1) + self.origin[2]

          C = numpy.array([cx, cy, cz], dtype=numpy.single)

          #print "Adding affine at center", C, "radius", rad

          self.centers.append(C)
          self.radii.append(rad)
          self.affines.append(A0)
          self.translations.append(T0)

    print "Created identity with", len(self.affines), "affine transforms"

  def add_affine(self, A, T, C, r):
    self.centers.append(C)
    self.radii.append(r)
    self.affines.append(A)
    self.translations.append(T)

    self.movingCL = self.warp(self.origMovingCL,
      self.affines, self.translations, self.centers)

    # NOTE: need to reinitialize optimizer, either in here or outside
    #self.optimize_setup()

  def optimize_setup(self):
    """Optimization setup, needs to be called before iterative calls to
    optimize_step."""
    self.optimIter = 0
    self.optimMode = 0

    numTransforms = len(self.affines)

    self.stepA = -1.0
    self.stepT = -1.0
    self.stepC = -1.0

    self.prevErrorL2 = float('Inf')

    self.compute_weights_and_sum()

    DiffFM = self.fixedCL.subtract(self.movingCL)
    DiffFMSq = DiffFM.multiply(DiffFM)
    errorL2 = DiffFMSq.sum()

    self.currErrorL2 = errorL2

    self.refErrorL2 = errorL2
    print "Ref diff", self.refErrorL2

  def compute_weights_and_sum(self):

    numTransforms = len(self.affines)

    for w in self.weights:
      del w
    self.weights = []

    del self.sum_weights

    self.sum_weights = self.fixedCL.clone()
    self.sum_weights.fill(1e-10)

    for q in range(numTransforms):
      C = self.centers[q]
      r = self.radii[q]

      W = self._get_weights(self.fixedCL.shape, C, r)

      # Storing list of W will take up too much memory, store only ROI
      #W_ROI = W.getROI(C, r)
      #self.weights.append(W_ROI)

      self.sum_weights.add_inplace(W)

  def optimize_step(self):
    """
    Gradient descent update step that alternates between parameters,
    NOT thread safe.
    """

    self.prevErrorL2 = self.currErrorL2

    print "Mode", self.optimMode

    # Alternating gradient descent with adaptive step sizes

    if self.optimIter > 1 and (self.optimIter % 5) == 0:
      self.optimize_anchors()
      self.compute_weights_and_sum()
    #TODO
    #  self.optimize_radius()
    #  self.compute_weights_and_sum()
    else:
      if self.optimMode == 0:
        self.optimize_translations()
      #elif self.optimMode == 1:
      else:
        self.optimize_affines()

      self.optimMode = (self.optimMode + 1) % 3

    self.optimIter += 1

  def optimize_translations(self):

    numTransforms = len(self.affines)

    TList = self.translations

    dTList = self.gradient_translation()

    if self.stepT < 0.0:
      max_dT = 1e-10
      for q in range(numTransforms):
        max_dT = max(numpy.max(numpy.abs(dTList[q])), max_dT)
      self.stepT = 5.0 / max_dT

    print "Line search trans"
    for lineIter in range(self.lineSearchIterations):
      print "opt line iter", lineIter

      TTestList = list(TList)
      for q in range(numTransforms):
        TTestList[q] = TList[q] - dTList[q]*self.stepT

      M = self.warp(self.origMovingCL, self.affines, TTestList, self.centers)

      DiffFM = self.fixedCL.subtract(M)
      DiffFMSq = DiffFM.multiply(DiffFM)
      errorL2Test = DiffFMSq.sum()

      print "Test diff", errorL2Test

      if errorL2Test < self.currErrorL2:
        TList = TTestList

        # TODO: figure out book-keeping, shouldn't have duplicates like this
        # gradient and warp should be aware of which to use
        self.translations = TList

        self.stepT *= 1.2

        self.currErrorL2 = errorL2Test

        self.movingCL = M

        print "PolyAffine error=", self.currErrorL2

        break
      else:
        self.stepT *= 0.5

  def optimize_affines(self):

    numTransforms = len(self.affines)

    AList = self.affines

    dAList = self.gradient_affine()

    if self.stepA < 0.0:
      max_dA = 1e-10
      for q in range(numTransforms):
        max_dA = max(numpy.max(numpy.abs(dAList[q])), max_dA)
      self.stepA = 0.5 / max_dA

    print "Line search affine"
    for lineIter in range(self.lineSearchIterations):
      print "opt line iter", lineIter

      ATestList = list(AList)
      for q in range(numTransforms):
        ATestList[q] = AList[q] - dAList[q]*self.stepA

      M = self.warp(self.origMovingCL, ATestList, self.translations, self.centers)

      DiffFM = self.fixedCL.subtract(M)
      DiffFMSq = DiffFM.multiply(DiffFM)
      errorL2Test = DiffFMSq.sum()

      del DiffFM
      del DiffFMSq

      print "Test diff", errorL2Test

      if errorL2Test < self.currErrorL2:
        AList = ATestList

        # TODO: figure out book-keeping, shouldn't have duplicates like this
          # gradient and warp should be aware of which to use
        self.affines = AList

        self.stepA *= 1.2

        self.currErrorL2 = errorL2Test

        self.movingCL = M

        print "PolyAffine error=", self.currErrorL2

        break
      else:
        self.stepA *= 0.5

  def optimize_anchors(self):

    numTransforms = len(self.affines)

    CList = self.centers

    dCList = self.gradient_anchor()

    if self.stepC < 0.0:
      max_dC = 1e-10
      for q in range(numTransforms):
        max_dC = max(numpy.max(numpy.abs(dCList[q])), max_dC)
      self.stepC = 5.0 / max_dC

    print "Line search anchor"
    for lineIter in range(self.lineSearchIterations):
      print "opt line iter", lineIter

      CTestList = list(CList)
      for q in range(numTransforms):
        CTestList[q] = CList[q] - dCList[q]*self.stepC

      M = self.warp(self.origMovingCL, self.affines, self.translations, CTestList)

      DiffFM = self.fixedCL.subtract(M)
      DiffFMSq = DiffFM.multiply(DiffFM)
      errorL2Test = DiffFMSq.sum()

      del DiffFM
      del DiffFMSq

      print "Test diff", errorL2Test

      if errorL2Test < self.currErrorL2:
        CList = CTestList

        # TODO: figure out book-keeping, shouldn't have duplicates like this
        # gradient and warp should be aware of which to use
        self.centers = CTestList

        self.stepC *= 1.2

        self.currErrorL2 = errorL2Test

        self.movingCL = M

        print "PolyAffine error=", self.currErrorL2

        break
      else:
        self.stepC *= 0.5


  def optimize(self, maxIters=10):
    """Offline optimization of polyaffine parameters using adaptive step
    gradient descent."""

    self.optimize_setup()

    for iter in range(maxIters):
      self.optimize_step()

      if abs(self.currErrorL2 - self.prevErrorL2) < self.convergenceRatio * abs(self.currErrorL2 - self.refErrorL2):
        if self.optimMode == 0 and self.optimIter > 1:
          break

      print "opt iter", iter, "steps", self.stepA, self.stepT, self.stepC

  def gradient(self):
    """Gradient of L2 norm"""

    numTransforms = len(self.centers)

    gradA_list = []
    gradT_list = []

    gradC_list = []
    gradR_list = []

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

      CF = numpy.array(F.shape, dtype=numpy.single) / 2.0

      #W = self.weights[q].divide(self.sum_weights.getROI(C, r))
      #W = self.weights[q]

      W = self._get_weights(F.shape, CF, r)
      #W = self._get_weights(F.shape, C, r)

      WD = W.multiply(DiffFM)

      gradA = numpy.zeros((3,3), dtype=numpy.single)
      for i in range(3):
        for j in range(3):
          GX = GList[i].multiply(XList[j])
          gradA[i,j] = -2.0 * WD.multiply(GX).sum()

      gradT = numpy.zeros((3,), dtype=numpy.single)
      for d in range(3):
        gradT[d] = -2.0 * WD.multiply(GList[d]).sum()

      gradC = numpy.zeros((3,), dtype=numpy.single)
      gradR = numpy.zeros((3,), dtype=numpy.single)

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

      CF = numpy.array(F.shape, dtype=numpy.single) / 2.0

      #W = self.weights[q].divide(self.sum_weights.getROI(C, r))
      #W = self.weights[q]

      W = self._get_weights(F.shape, CF, r)
      #W = self._get_weights(F.shape, C, r)

      WD = W.multiply(DiffFM)

      gradA = numpy.zeros((3,3), dtype=numpy.single)
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

      CF = numpy.array(F.shape, dtype=numpy.single) / 2.0

      #W = self.weights[q].divide(self.sum_weights.getROI(C, r))
      #W = self.weights[q]

      W = self._get_weights(F.shape, CF, r)
      #W = self._get_weights(F.shape, C, r)

      WD = W.multiply(DiffFM)

      gradT = numpy.zeros((3,), dtype=numpy.single)
      for d in range(3):
        gradT[d] = -2.0 * WD.multiply(GList[d]).sum()

      gradT_list.append(gradT)

    return gradT_list

  def gradient_anchor(self):
    """Gradient of L2 norm for anchor positions only"""

    numTransforms = len(self.centers)

    gradC_list = []

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

      CF = numpy.array(F.shape, dtype=numpy.single) / 2.0

      #W = self.weights[q].divide(self.sum_weights.getROI(C, r))
      #W = self.weights[q]

      W = self._get_weights(F.shape, CF, r)
      #W = self._get_weights(F.shape, C, r)

      WD = W.multiply(DiffFM)

      gradC = numpy.zeros((3,), dtype=numpy.single)

      dot_G_XC = F.clone()
      dot_G_XC.fill(0.0)

      ATList = []

      for d in range(3):
        AT = F.clone()
        AT.fill(0.0)
        for j in range(3):
          Y = XList[d].clone()
          Y.scale(A[d,j])
          AT.add_inplace(Y)
        AT.shift(T[d])

        ATList.append(AT)

        XC = XList[d].clone()
        XC.shift(-C[d])
        XC.scale(2.0 / r[d]**2)

        dot_G_XC.add_inplace(GList[d].multiply(XC))

      for d in range(3):
        gradC[d] = -WD.multiply(ATList[d].multiply(dot_G_XC)).sum()

      gradC_list.append(gradC)

    return gradC_list

  def applyTo(self, image):
    """
    Apply poly-affine transform to an image.
    """
    return self.warp(image, self.affines, self.translations, self.centers)

  def warp(self, image, AList, TList, CList):
    """
    Compute deformation field and update moving image.
    Returns warped version of origMovingCL with current poly-affine parameters.
    """

    numTransforms = len(AList)

    shape = self.fixedCL.shape

    A = numpy.zeros((numTransforms, 3*3), numpy.single)
    C = numpy.zeros((numTransforms, 3), numpy.single)
    T = numpy.zeros((numTransforms, 3), numpy.single)
    R = numpy.zeros((numTransforms, 3), numpy.single)

    for i in range(numTransforms):
      A[i,:] = AList[i].ravel()
      C[i,:] = CList[i].ravel()
      T[i,:] = TList[i] .ravel()
      R[i,:] = self.radii[i].ravel()

    clmatrices = cla.to_device(image.clqueue, A)
    clcenters = cla.to_device(image.clqueue, C)
    cltrans = cla.to_device(image.clqueue, T)
    clradii = cla.to_device(image.clqueue, R)

    clorigin = cla.to_device(image.clqueue, numpy.array(image.origin))

    warpedImage = image.clone()

    image.clprogram.applyPolyAffine(image.clqueue, image.shape, None,
      clcenters.data, clradii.data, clmatrices.data, cltrans.data,
      numpy.uint32(numTransforms),
      image.clarray.data, image.clspacing.data, clorigin.data,
      warpedImage.clarray.data)

    return warpedImage

  def _get_weights(self, shape, center, radii):
    """Returns ImageCL object of Gaussian weights"""

    weightsCL = ImageCL(self.fixedCL.preferredDeviceType)
    temparr = numpy.zeros(shape, dtype=numpy.single)
    weightsCL.fromArray(temparr, self.fixedCL.origin, self.fixedCL.spacing) 

    clcenter = cla.to_device(weightsCL.clqueue,
      numpy.array(center) )

    clorigin = cla.to_device(weightsCL.clqueue,
      numpy.array(self.fixedCL.origin) )

    clradii = cla.to_device(weightsCL.clqueue, radii)

    weightsCL.clprogram.weightsPolyAffine(
      weightsCL.clqueue, shape, None,
      clcenter.data, clradii.data,
      weightsCL.clspacing.data, clorigin.data,
      weightsCL.clarray.data)

    return weightsCL

    """
    origin = numpy.array(center, dtype=numpy.single)
    for dim in range(3):
      origin[dim] -= radii[dim]

    temp = ImageCL(self.fixedCL.preferredDeviceType)
    temparr = numpy.zeros(shape, dtype=numpy.single)
    temp.fromArray(temparr, origin, self.fixedCL.spacing) 

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
