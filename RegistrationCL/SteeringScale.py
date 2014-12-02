
import ImageCL

import numpy as np

from SteeringGeneric import SteeringGeneric

class SteeringScale(SteeringGeneric):

  def __init__(self, fixedCL, movingCL, center, radius):
    SteeringGeneric.__init__(self, fixedCL, movingCL, center, radius)

  def steer(self):

    N = self.fixedCL.shape

    # Obtain 2D view
    SteeringGeneric.update_slices(self)

    fixed2D, moving2D = self.get_slices()

    fixedV, movingV = self.get_slice_axes()

    F = SteeringGeneric.get_frame_matrix(self, fixedV, movingV)

    # Perform 2D registration
    scale, T2 = self.register_scale2d(fixed2D, moving2D)

    print "S = ", scale

    S = np.identity(3, np.single) * scale

    T = F[:,0] * T2[0]  + F[:,1] * T2[1]
    C = np.array([N[0]/2, N[1]/2, N[2]/2], np.single)
    T += C - np.dot(S,C)

    # Subtract identitiy from matrix to match convention for PolyAffineCL
    for dim in range(3):
      S[dim, dim] -= 1.0

    # Return 3D transformation that will be folded into an existing polyaffine
    return S, T

  def register_scale2d(self, fixed, moving):
    sizex, sizey = moving.shape

    x0, y0 = np.mgrid[0:sizex,0:sizey]

    xc = float(sizex) / 2.0
    yc = float(sizey) / 2.0

    F = fixed
    M = moving

    G = SteeringGeneric.gaussian2(self, fixed.shape)

    D = G * (F - M)
    Dnorm = np.linalg.norm(D.ravel())

    Dnorm_ref = Dnorm

    I = np.identity(2, np.single)
    T = np.zeros((2,1), np.single)

    scale = 1.0

    for iter in xrange(20):

      print "2D diff", Dnorm

      S = I * scale
      dS = I

      Dnorm_prev = Dnorm

      D = G * (F - M)

      Mx, My = np.gradient(M)

      dScale = np.sum(D * (x0*Mx + y0*My))

      dT = np.zeros((2,1), np.single)
      dT[0] = np.sum(D * Mx)
      dT[1] = np.sum(D * My)

      if iter == 0:
        stepScale = 0.2 / max(abs(dScale), 1e-5)
        stepT = 1.0 / max(np.max(np.abs(dT)), 1e-5)

      for sub_iter in xrange(20):
        scale_test = scale + dScale * stepScale

        T_test = T + dT * stepT

        Cx = (1.0 - scale_test) * xc
        Cy = (1.0 - scale_test) * yc

        x = x0*scale_test + T_test[0] + Cx
        y = y0*scale_test + T_test[1] + Cy

        M_test = SteeringGeneric.interp2(self, x, y, moving)

        D_test = G * (F - M_test)
        Dnorm_test = np.linalg.norm(D_test.ravel())
        if Dnorm_test < Dnorm_prev:
          stepScale *= 1.2
          stepT *= 1.2
          Dnorm = Dnorm_test
          scale = scale_test
          T = T_test
          M = M_test
          break
        else:
          stepScale *= 0.5
          stepT *= 0.5

      if abs(Dnorm-Dnorm_prev) < 1e-4 * abs(Dnorm-Dnorm_ref):
        break

    return scale, T
