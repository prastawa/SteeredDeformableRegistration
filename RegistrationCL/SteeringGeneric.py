
from ImageCL import ImageCL
from ImageQuery import ImageQuery

import numpy as np

class SteeringGeneric:

  def __init__(self, fixedCL, movingCL, center, radius):
    self.fixedCL = fixedCL
    self.movingCL = movingCL

    self.fixedSlice = None
    self.movingSlice = None

    self.fixedAxis = None
    self.movingAxis = None

    self.center = center
    self.radius = radius


  def update_slices(self):
    """
    Display images at certain axes and query user on likely rotation plane.
    """

    fixedROI = self.fixedCL.getROI(self.center, self.radius)
    movingROI = self.movingCL.getROI(self.center, self.radius)

    query2d = ImageQuery(fixedROI, movingROI)

    self.fixedSlice, self.movingSlice, self.fixedAxis, self.movingAxis = \
      query2d.display_and_query()

  def get_slices(self):
    return self.fixedSlice, self.movingSlice

  def get_slice_axes(self):
    return self.fixedAxis, self.movingAxis

  def get_frame_matrix(self, a, b):
    """Get orthonormal frame from two vectors a and b"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)

    #print "Frame from "
    #print a
    #print b

    f1 = a
    f2 = np.cross(a, c)
    f3 = c

    f2 = f2 / np.linalg.norm(f2)
    f3 = f3 / np.linalg.norm(f3)

    F = np.zeros((3,3), np.float32)
    F[:,0] = f1
    F[:,1] = f2
    F[:,2] = f3

    return F

  def gaussian2(self, shape):
    """Gaussian for 2D steering"""
    sx = shape[0]
    sy = shape[1]

    x0 = -sx / 2
    x1 = x0 + sx
    y0 = -sy / 2
    y1 = y0 + sy

    X, Y = np.mgrid[x0:x1, y0:y1]
    G = np.exp( -(X**2/np.float(sx) + Y**2/np.float(sy)) )
    #G /= G.sum()
    G /= np.max(G)

    return G

  
  def interp2(self, x, y, I):
    """Interpolation for 2D steering"""
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, I.shape[0]-1);
    x1 = np.clip(x1, 0, I.shape[0]-1);
    y0 = np.clip(y0, 0, I.shape[1]-1);
    y1 = np.clip(y1, 0, I.shape[1]-1);

    i00 = I[x0, y0]
    i01 = I[x0, y1]
    i10 = I[x1, y0]
    i11 = I[x1, y1]

    w00 = (x1-x) * (y1-y)
    w01 = (x1-x) * (y-y0)
    w10 = (x-x0) * (y1-y)
    w11 = (x-x0) * (y-y0)

    return w00*i00 + w01*i01 + w10*i10 + w11*i11

