
import ImageCL
from SteeringGeneric import SteeringGeneric

class SteeringScale(SteeringGeneric):
  def __init__(self):
    SteeringGeneric.__init__(self, fixedCL, movingCL)

  def steer(self):
    # TODO 2d isotropic scaling reg on slices 
    # followed by 3d correction

