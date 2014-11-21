"""
Divide image into blocks for GPU operations to get around memory limitations

TODO
use sitk image

lower priority?
"""

import SimpleITK as sitk

class ImageComposite:

  def __init__(self, image, level=2):
    # Divide each axis by level
    self.volume = volume
    self.level = level
    self.current_block = 0

    # TODO: generate blocks by subdividing
    self.blocks = []

  def getROI(self, center, radius):

  def getCLROI(self, center, radius):

  def assignROI(self, center, radius, image):
  # or?
  def assignROI(self, X0, X1 image):

  def addROI(self, center, radius, image):

  def get_roi_from_sphere(self, center, radius):
    return X0, X1

  def next_block(self):
    # Move to next block on the list
    # Returns None if done

  def get_block(self):
    # Return numpy array for current block

  def update_block(self, imarray):

    self.volume[block_indices] = imarray

  def get_composite(self):
    # Combine all blocks and return the whole image
