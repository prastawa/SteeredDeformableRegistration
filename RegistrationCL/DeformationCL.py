
#
# DeformationCL
#
# Represent deformation map h using a list of 3 ImageCL objects
# with support for composition (h1 \circ h2), and warping of scalar volumes
#
# Warping is applied as Iwarped = I(h)
#
# Requires: ImageCL
#
# Author: Marcel Prastawa (marcel.prastawa@gmail.com)
#

import ImageCL

class DeformationCL:

  def __init__(self, imgcl, hlist=None):

    self.clgrid = imgcl

    self.clqueue = imgcl.clqueue
    self.clprogram = imgcl.clprogram

    if hlist is None:
      # Assign identity mapping if mapping not specified at init
      self.hx = imgcl.clone()
      self.hy = imgcl.clone()
      self.hz = imgcl.clone()
      self.set_identity()
    else:
      self.hx = hlist[0]
      self.hy = hlist[1]
      self.hz = hlist[2]

  def __del__(self):
    self.clgrid = None
    self.hx = None
    self.hy = None
    self.hz = None

  def clone(self):
    hcopies = [self.hx.clone(), self.hy.clone(), self.hz.clone()]
    return DeformationCL(self.clgrid, hcopies)

  def set_mapping(self, hx, hy, hz):
    self.hx = hx
    self.hy = hy
    self.hz = hz
    self.clqueue = self.hx.clqueue
    self.clprogram = self.hx.clprogram

  def set_identity(self):

    clspacing = self.hx.clspacing

    self.clprogram.identity(self.clqueue, self.hx.shape, None,
      clspacing.data,
      self.hx.clarray.data,  self.hy.clarray.data, self.hz.clarray.data).wait()

  def add_velocity(self, velocList):
    self.hx.add_inplace(velocList[0])
    self.hy.add_inplace(velocList[1])
    self.hz.add_inplace(velocList[2])

  def maxMagnitude(self):
    magimg = self.hx.multiply(self.hx)
    magimg.add_inplace( self.hy.multiply(self.hy) )
    magimg.add_inplace( self.hz.multiply(self.hz) )
    return math.sqrt( magimg.max() )

  def resample(self, targetShape):
    hx_new = self.hx.resample(targetShape)
    hy_new = self.hy.resample(targetShape)
    hz_new = self.hz.resample(targetShape)

    H_new = [hx_new, hy_new, hz_new]

    outdef = DeformationCL(hx_new, H_new)

    H_new = None

    return outdef

  def applyTo(self, vol):
    # Output image is in the same grid as h
    outimgcl = self.hx.clone()

    outimgcl.clprogram.interpolate(outimgcl.clqueue, self.hx.shape, None,
      vol.clarray.data,
      vol.clsize.data, vol.clspacing.data,
      self.hx.clarray.data, self.hy.clarray.data, self.hz.clarray.data,
      outimgcl.clarray.data).wait()

    return outimgcl

  def compose(self, otherdef):
    hx_new = otherdef.applyTo(self.hx)
    hy_new = otherdef.applyTo(self.hy)
    hz_new = otherdef.applyTo(self.hz)

    H_new = [hx_new, hy_new, hz_new]

    outdef = DeformationCL(otherdef.clgrid, H_new)

    H_new = None

    return outdef
