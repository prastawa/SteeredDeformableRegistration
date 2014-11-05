
#
# ImageCL: an image class with CL arrays and functionalities
#
# Adds support for image registration operations on Slicer image data
# using PyOpenCL
#
# Requires the program in ImageFunctions.cl
#
# Author: Marcel Prastawa (marcel.prastawa@gmail.com)
#

import vtk

import pyopencl as cl
import pyopencl.array as cla
import pyopencl.clmath as clmath

import numpy as n

class ImageCL:
  def __init__(self, clqueue=None, clprogram=None):
    self.clqueue = clqueue

    # TODO: initialize clprogram with ImageFunctions.cl here?
    # need to pass a preferred device type (CPU, GPU) to this class
    self.clprogram = clprogram

    self.origin = [0.0, 0.0, 0.0]
    self.shape = [0, 0, 0]
    self.spacing = [1.0, 1.0, 1.0]

    self.clarray = None
    self.clsize = None
    self.clspacing = None

  def __del__(self):
    self.clarray = None
    self.clsize = None
    self.clspacing = None

  def clone_empty(self):
    """Clone self without filling in CL array data"""
    outimgcl = ImageCL(self.clqueue, self.clprogram)
    outimgcl.origin = list(self.origin)
    outimgcl.shape = list(self.shape)
    outimgcl.spacing = list(self.spacing)

    outimgcl.clarray = None
    if self.clsize is not None:
      outimgcl.clsize = self.clsize.copy()
    if self.clspacing is not None:
      outimgcl.clspacing = self.clspacing.copy()

    return outimgcl

  def clone(self):
    """Clone self with deep copy of CL array"""
    outimgcl = self.clone_empty()
    if self.clarray is not None:
      outimgcl.clarray = self.clarray.copy()
    return outimgcl

  def fromVolume(self, volume):
    """Fill data using a MRML volume node"""

    # Cast to float
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToFloat()
    castf.SetInput( volume.GetImageData() )
    castf.Update()

    vtkimage = castf.GetOutput()

    self.shape = list(vtkimage.GetDimensions())

    self.origin = list(volume.GetOrigin())

    self.spacing = list(volume.GetSpacing())

    # Store original orientation in case need to map back
    self.originalIJKToRAS = vtk.vtkMatrix4x4()
    volume.GetIJKToRASDirectionMatrix(self.originalIJKToRAS)

    nspacing = n.zeros((3,), dtype=n.float32)
    for dim in xrange(3):
      nspacing[dim] = self.spacing[dim]
    self.clspacing = cla.to_device(self.clqueue, nspacing)

    nsize = n.zeros((3,), dtype=n.uint32)
    for dim in xrange(3):
      nsize[dim] = self.shape[dim]
    self.clsize = cla.to_device(self.clqueue, nsize)

    # VTK image data is stored in reverse order
    reverse_shape = list(vtkimage.GetDimensions())
    reverse_shape.reverse()
    narray = vtk.util.numpy_support.vtk_to_numpy(
        vtkimage.GetPointData().GetScalars()).reshape(reverse_shape)
        #vtkimage.GetPointData().GetScalars()).reshape(self.shape)
    narray = n.asfortranarray(narray)
    narray = narray.transpose(2, 1, 0)
    #narray = narray.astype('float32')
    self.clarray = cl.array.to_device(self.clqueue, narray)

    vtkimage = None

  def fromArray(self, imarray, spacing):
    """Fill data using a numpy array along with spacing information"""

    self.shape = imarray.shape
    self.origin = [0, 0, 0]
    self.spacing = spacing

    nspacing = n.zeros((3,), dtype=n.float32)
    for dim in xrange(3):
      nspacing[dim] = self.spacing[dim]
    self.clspacing = cla.to_device(self.clqueue, nspacing)

    nsize = n.zeros((3,), dtype=n.uint32)
    for dim in xrange(3):
      nsize[dim] = self.shape[dim]
    self.clsize = cla.to_device(self.clqueue, nsize)

    narray = imarray.astype('float32')
    self.clarray = cl.array.to_device(self.clqueue, narray)

  def getROI(self, center, radius):
    """Extract ImageCL object at an ROI defined by center and radius"""

    # Convert radius to voxels
    voxrad = [1,1,1]
    for d in range(3):
      voxrad[d] = radius[d] / self.spacing[d]

    X0 = [0,0,0]
    X1 = [self.shape[0]-1, self.shape[1]-1, self.shape[2]-1]

    for d in range(3):
      p0 = int(center[d]/self.spacing[d] - voxrad[d])
      p0 = max(0, p0)
      p0 = min(self.shape[d]-1, p0)

      p1 = int(center[d]/self.spacing[d] + voxrad[d])
      p1 = max(0, p1)
      p1 = min(self.shape[d]-1, p1)

      X0[d] = p0
      X1[d] = p1

    #print "C", center
    #print "r", radius
    #print "X0", X0
    #print "X1", X1

    arr = self.clarray.get()
    subarr = arr[X0[0]:X1[0], X0[1]:X1[1], X0[2]:X1[2]]

    #subclarray = self.clarray[X0[0]:X1[0], X0[1]:X1[1], X0[2]:X1[2]]
    #subarr = subclarray.get()

    #print "subarr shape", subarr.shape

    outimgcl = ImageCL(self.clqueue, self.clprogram)
    outimgcl.fromArray(subarr, self.spacing)
    return outimgcl

  def toVTKImage(self):
    """Returns vtkImageData containing image from GPU memory"""
    narray = self.clarray.get().astype('float32')
    narray = narray.transpose(2, 1, 0)

    vtkarray = vtk.util.numpy_support.numpy_to_vtk(narray.flatten(), deep=True)
 
    # NOTE: vtk image does not contain image and spacing, all info in volume
    vtkimage = vtk.vtkImageData()
    vtkimage.SetScalarTypeToFloat()
    vtkimage.SetNumberOfScalarComponents(1)
    vtkimage.SetExtent(
      0, self.shape[0]-1, 0, self.shape[1]-1, 0, self.shape[2]-1)
      #0, self.shape[2]-1, 0, self.shape[1]-1, 0, self.shape[0]-1)
    vtkimage.AllocateScalars()
    vtkimage.GetPointData().SetScalars(vtkarray)
    vtkimage.GetPointData().GetScalars().Modified()
    vtkimage.GetPointData().Modified()
    vtkimage.Modified()

    return vtkimage

  def copyToVolume(self, volume):
    """Copy GPU data to an existing MRML volume node"""
    #volume.SetOrigin(self.origin[2], self.origin[1], self.origin[0])
    #volume.SetSpacing(self.spacing[2], self.spacing[1], self.spacing[0])
    volume.SetOrigin(self.origin[0], self.origin[1], self.origin[2])
    volume.SetSpacing(self.spacing[0], self.spacing[1], self.spacing[2])

    vtkimage = self.toVTKImage()
    volume.SetAndObserveImageData(vtkimage)

  def fill(self, value):
    """Fill GPU data with scalar"""
    self.clarray.fill(value)

  def normalize(self):
    """Normalize intensities in GPU memory to range in [0,1]"""

    # Brute force CPU approach
    #narray = self.clarray.get()
    #minp = narray.min()
    #maxp = narray.max()

    # NOTE: this may not work on Nvidia and Windows, due to space in header
    # include location
    # WORKAROUND: edit Slicer/lib/Python/Lib/site-packages/pyopencl/reduction.py
    # and insert code from cl/pyopencl-complex.h manually
    clminp = cl.array.min(self.clarray, self.clqueue)
    clmaxp = cl.array.max(self.clarray, self.clqueue)
    # Convert reductions to scalars
    minp = clminp.get()[()]
    maxp = clmaxp.get()[()]

    range = maxp - minp
    if range > 0.0:
      self.clarray -= minp
      self.clarray /= range

  def scale(self, v):
    self.clarray *= v

  def shift(self, v):
    self.clarray += v
    
  def add(self, otherimgcl):
    outimgcl = self.clone_empty()
    outimgcl.clarray = self.clarray + otherimgcl.clarray
    return outimgcl
    
  def subtract(self, otherimgcl):
    outimgcl = self.clone_empty()
    outimgcl.clarray = self.clarray - otherimgcl.clarray
    return outimgcl
    
  def multiply(self, otherimgcl):
    outimgcl = self.clone_empty()
    outimgcl.clarray = self.clarray * otherimgcl.clarray
    return outimgcl

  def divide(self, otherimgcl):
    outimgcl = self.clone_empty()
    outimgcl.clarray = self.clarray / otherimgcl.clarray
    return outimgcl
    
  def exp(self):
    outimgcl = self.clone_empty()
    outimgcl.clarray = clmath.exp(self.clarray)
    return outimgcl

  def add_inplace(self, otherimgcl):
    self.clarray += otherimgcl.clarray
    return self

  def subtract_inplace(self, otherimgcl):
    self.clarray -= otherimgcl.clarray
    return self

  def multiply_inplace(self, otherimgcl):
    self.clarray *= otherimgcl.clarray
    return self

  def divide_inplace(self, otherimgcl):
    self.clarray /= otherimgcl.clarray
    return self

  def minmax(self):
    #TODO
    #use cl.reduction.ReductionKernel
    #https://github.com/pyopencl/pyopencl/blob/master/examples/demo-struct-reduce.py
    pass

  def min(self):
    #return self.clarray.get().min()

    # NOTE: see note on normalize()
    clminp = cl.array.min(self.clarray, self.clqueue)
    minp = clminp.get()[()]
    return minp

  def max(self):
    #return self.clarray.get().max()

    # NOTE: see note on normalize()
    clmaxp = cl.array.max(self.clarray, self.clqueue)
    maxp = clmaxp.get()[()]
    return maxp

  def sum(self):
    return self.clarray.get().sum()
    #clsump = cl.array.sum(self.clarray, self.clqueue)
    #sump = clsump.get()[()]
    #return sump

  def gradient(self):
    """Returns list of gradients in x, y, and z"""
    gradx = self.clone()
    grady = self.clone()
    gradz = self.clone()

    self.clprogram.gradient(self.clqueue, self.shape, None,
      self.clarray.data,
      self.clsize.data, self.clspacing.data,
      gradx.clarray.data, grady.clarray.data, gradz.clarray.data).wait()

    return [gradx, grady, gradz]

  def gradient_magnitude(self):
    [gx, gy, gz] = self.gradient()
    mag = gx.multiply(gx)
    mag.add_inplace(gy.multiply(gy))
    mag.add_inplace(gz.multiply(gz))

    return mag

  def discrete_gaussian(self, sigma):
    """Discrete / convolutional Gaussian smoothing on GPU"""
    outimgcl = self.clone()

    tempclarray = outimgcl.clarray.copy()

    sx = sigma / self.spacing[0]
    varx = n.float32(sx * sx)
    widthx = n.int32(3 * sx)
    self.clprogram.gaussian_x(self.clqueue, self.shape, None,
      tempclarray.data, self.clsize.data,
      varx, widthx, outimgcl.clarray.data).wait()

    sy = sigma / self.spacing[1]
    vary = n.float32(sx * sy)
    widthy = n.int32(3 * sy)
    self.clprogram.gaussian_y(self.clqueue, self.shape, None,
      outimgcl.clarray.data, self.clsize.data,
      vary, widthy,
      tempclarray.data).wait()

    sz = sigma / self.spacing[2]
    varz = n.float32(sx * sz)
    widthz = n.int32(3 * sz)
    self.clprogram.gaussian_z(self.clqueue, self.shape, None,
      tempclarray.data, self.clsize.data,
      varz, widthz,
      outimgcl.clarray.data).wait()

    return outimgcl

  def recursive_gaussian(self, sigma):
    """Recursive Gaussian smoothing on GPU"""
    outimgcl = self.clone()

    sizeX = self.shape[0]
    sizeY = self.shape[1]
    sizeZ = self.shape[2]

    sz = n.float32(sigma / self.spacing[2])
    outimgcl.clprogram.recursive_gaussian_z(outimgcl.clqueue,
      (sizeX, sizeY), None,
      outimgcl.clarray.data, outimgcl.clsize.data, sz).wait()

    sy = n.float32(sigma / self.spacing[1])
    outimgcl.clprogram.recursive_gaussian_y(outimgcl.clqueue,
      (sizeX, sizeZ), None,
      outimgcl.clarray.data, outimgcl.clsize.data, sy).wait()

    sx = n.float32(sigma / self.spacing[0])
    outimgcl.clprogram.recursive_gaussian_x(outimgcl.clqueue,
      (sizeY, sizeZ), None,
      outimgcl.clarray.data, outimgcl.clsize.data, sx).wait()

    return outimgcl

  def get_resampled_spacing(self, targetShape):
    re_spacing = [1.0, 1.0, 1.0]
    for dim in xrange(3):
      re_spacing[dim] = (self.spacing[dim] * self.shape[dim]) / targetShape[dim]
    return re_spacing

  def resample(self, targetShape):
    """Resample GPU data to specified size"""

    # TODO do some filtering if downsampling?
    """
    isDownsampling = False
    for dim in xrange(3):
      if targetShape[dim] <= self.shape[dim] / 2:
        isDownsampling = True
        break
    minspacing = min(self.spacing)
    if isDownsampling:
      smoothimgcl = self.discrete_gaussian(minspacing)
    """

    outimgcl = self.clone_empty()

    outimgcl.shape = list(targetShape)
    outimgcl.spacing = self.get_resampled_spacing(targetShape)

    for dim in xrange(3):
      outimgcl.clsize[dim] = targetShape[dim]
      outimgcl.clspacing[dim] = outimgcl.spacing[dim]

    outimgcl.clarray = cl.array.zeros(self.clqueue, targetShape, n.float32)

    hxclarray = cl.array.zeros_like(outimgcl.clarray)
    hyclarray = cl.array.zeros_like(outimgcl.clarray)
    hzclarray = cl.array.zeros_like(outimgcl.clarray)

    self.clprogram.identity(self.clqueue, targetShape, None,
      outimgcl.clsize.data, outimgcl.clspacing.data,
      hxclarray.data,  hyclarray.data, hzclarray.data).wait()
    
    self.clprogram.interpolate(self.clqueue, targetShape, None,
      self.clarray.data,
      self.clsize.data, self.clspacing.data,
      hxclarray.data, hyclarray.data, hzclarray.data,
      outimgcl.clarray.data,
      outimgcl.clsize.data).wait()

    hxclarray = None
    hyclarray = None
    hzclarray = None

    return outimgcl

  @staticmethod
  def add_splat3(outimages, posM, valueM, sigmaM):
    """
    Special purpose function for fluid deformation:
    adds splatted forces fx,fy,fz to hx,hy,hz
    """
    shape = outimages[0].shape

    clqueue = outimages[0].clqueue
    clsize = outimages[0].clsize
    clspacing = outimages[0].clspacing
    clprogram = outimages[0].clprogram

    clposM = cl.array.to_device(clqueue, posM)
    clvalueM = cl.array.to_device(clqueue, valueM)
    clsigmaM = cl.array.to_device(clqueue, sigmaM)

    numV = n.uint32(posM.shape[0])

    clprogram.add_splat3(clqueue, shape, None,
      clposM.data,
      clvalueM.data,
      clsigmaM.data,
      numV,
      outimages[0].clarray.data,
      outimages[1].clarray.data,
      outimages[2].clarray.data,
      clsize.data,
      clspacing.data).wait()

