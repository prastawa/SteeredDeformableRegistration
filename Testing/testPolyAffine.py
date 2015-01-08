
import numpy

import SimpleITK as sitk

#from ..RegistrationCL import ImageCL, DeformationCL, PolyAffineCL

import os, sys

if len(sys.argv) != 3:
  print "Usage", sys.argv[0], " fixed moving"
  sys.exit(-1)

sys.path.append("..")
from RegistrationCL import *

# Which CL device?
#preferredDeviceType = "CPU"
preferredDeviceType = "GPU"

fixedImage = sitk.ReadImage(sys.argv[1])

movingImage = sitk.ReadImage(sys.argv[2])

fixedArray = sitk.GetArrayFromImage(fixedImage).astype('float32')
movingArray = sitk.GetArrayFromImage(movingImage).astype('float32')

print fixedArray.shape
print movingArray.shape

fixedCL = ImageCL(preferredDeviceType)
fixedCL.fromArray(fixedArray, fixedImage.GetOrigin(), fixedImage.GetSpacing())

movingCL = ImageCL(preferredDeviceType)
movingCL.fromArray(movingArray, fixedImage.GetOrigin(), fixedImage.GetSpacing())

polyAffine = PolyAffineCL(fixedCL, movingCL)
polyAffine.create_identity(5)
polyAffine.optimize(20)

warpedCL = polyAffine.movingCL
warpedArray = warpedCL.clarray.get().astype('float32')

warpedImage = sitk.GetImageFromArray(warpedArray)
warpedImage.SetOrigin(fixedImage.GetOrigin())
warpedImage.SetSpacing(fixedImage.GetSpacing())
sitk.WriteImage(warpedImage, "warped.nrrd")
