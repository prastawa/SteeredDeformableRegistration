
class SteeringGeneric:

  def __init__(self, fixedCL, movingCL, center, radius):
    self.fixedCL = fixedCL
    self.movingCL = movingCL

    self.fixedSlice = None
    self.movingSlice = None

    self.center = center
    self.radius = radius


  def display_and_query(self):
    """
    Display images at certain axes and query user on likely rotation plane.
    """

    fixedROI = self.fixedCL.getROI(self.center, self.radius)
    movingROI = self.movingCL.getROI(self.center, self.radius)

    query2d = ImageQuery(fixedROI, movingROI)
    query2d.show()

    self.fixedSlice, self.movingSlice = query2d.get_slices()

