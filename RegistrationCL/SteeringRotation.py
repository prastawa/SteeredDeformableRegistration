
import ImageCL

import numpy as np

import math

from SteeringGeneric import SteeringGeneric

class SteeringRotation(SteeringGeneric):

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
    R2, T2 = self.register_rigid2d(fixed2D, moving2D)

    print "R2 = ", R2
    print "T2 = ", T2

    R = np.identity(3, np.float32)
    R[:2,:2] = R2

    # Project trafo to 3D, from 2D transform in u,v axes from frame F = [u v w]
    # [u v w] [R2 0; 0 0 1] [x; y; 0] = R [u v w] [x; y; 0]
    # R = F [R2 0; 0 0 1] inv(F)
    R = np.dot(F, np.dot(R, np.linalg.inv(F)))

    T = F[:,0] * T2[0]  + F[:,1] * T2[1]
    C = np.array([N[0]/2, N[1]/2, N[2]/2], np.float32)
    T += C - np.dot(R,C)

    # Subtract identity from R to follow convention in PolyAffineCL
    for dim in range(3):
      R[dim, dim] -= 1.0

    # Return 3D transformation that will be folded into an existing polyaffine
    return R, T

  def register_rigid2d(self, fixed, moving):
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

    R = np.identity(2, np.float32)
    T = np.zeros((2,1), np.float32)

    angle = 0

    for iter in range(20):

      Dnorm_prev = Dnorm

      ca = math.cos(angle)
      sa = math.sin(angle)

      R = np.array([[ca, -sa], [sa, ca]], np.float32)
      dR = np.array([[-sa, -ca], [ca, -sa]], np.float32)

      D = G * (F - M)

      Mx, My = np.gradient(M)

      dRx = dR[0,0]*(x0-xc) + dR[0,1]*(y0-yc)
      dRy = dR[1,0]*(x0-xc) + dR[1,1]*(y0-yc)
      dAngle = np.sum(G * D * (dRx*Mx + dRy*My))

      dT = np.zeros((2,1), np.float32)
      dT[0] = np.sum(D * Mx)
      dT[1] = np.sum(D * My)

      if iter == 0:
        stepAngle = 0.5 / max(abs(dAngle),1e-5)
        stepT = 1.0 / max(np.max(np.abs(dT)), 1e-5)


      for sub_iter in range(20):
        angle_test = angle + dAngle * stepAngle

        ca = math.cos(angle_test)
        sa = math.sin(angle_test)

        R = np.array([[ca, -sa], [sa, ca]], np.float32)

        Cx = xc - (R[0,0]*xc + R[0,1]*yc)
        Cy = yc - (R[1,0]*xc + R[1,1]*yc)

        T_test = T + dT * stepT

        x = R[0,0] * x0 + R[0,1] * y0 + T_test[0] + Cx
        y = R[1,0] * x0 + R[1,1] * y0 + T_test[1] + Cy
        M_test = SteeringGeneric.interp2(self, x, y, moving)

        D_test = G * (F - M_test)
        Dnorm_test = np.linalg.norm(D_test.ravel())
        if Dnorm_test < Dnorm_prev:
          stepAngle *= 1.2
          stepT *= 1.2
          M = M_test
          Dnorm = Dnorm_test
          angle = angle_test
          T = T_test
          break
        else:
          stepAngle *= 0.5
          stepT *= 0.5

      if abs(Dnorm-Dnorm_prev) < 1e-4 * abs(Dnorm-Dnorm_ref):
        break

    return R, T
