
import ImageCL

from SteeringGeneric import SteeringGeneric

class SteeringRotation(SteeringGeneric):

  def __init__(self, fixedCL, movingCL):
    SteeringGeneric.__init__(self, fixedCL, movingCL)

  def steer(self):
    # TODO 2d rotation reg on slices 
    # followed by 3d correction
