
import numpy

import SimpleITK as sitk

import pyopencl as cl

#from ..RegistrationCL import ImageCL, DeformationCL, PolyAffineCL

import os, sys

if len(sys.argv) != 3:
  print "Usage", sys.argv[0], " fixed moving"
  sys.exit(-1)

sys.path.append("..")
from RegistrationCL import *

# Create cl context and queue
#preferredDeviceType = "CPU"
preferredDeviceType = "GPU"

clcontext = None
for platform in cl.get_platforms():
  for device in platform.get_devices():
    if cl.device_type.to_string(device.type) == preferredDeviceType:
      clcontext = cl.Context([device])
      print ("using: %s" % cl.device_type.to_string(device.type))
      break;
if clcontext is None:
  clcontext = cl.create_some_context()

clqueue = cl.CommandQueue(clcontext)

# Compile OpenCL code and create program object
# TODO: compile different code versions for different params and sizes
inPath = os.path.join( "..", "ImageFunctions.cl")

fp = open(inPath)
source = fp.read()
fp.close()

clprogram = cl.Program(clcontext, source).build()

fixedImage = sitk.ReadImage(sys.argv[1])

movingImage = sitk.ReadImage(sys.argv[2])

fixedArray = sitk.GetArrayFromImage(fixedImage).astype('float32')
movingArray = sitk.GetArrayFromImage(movingImage).astype('float32')

fixedCL = ImageCL(clqueue, clprogram)
fixedCL.fromArray(fixedArray, fixedImage.GetSpacing())

movingCL = ImageCL(clqueue, clprogram)
#movingCL.fromArray(movingArray, movingImage.GetSpacing())
movingCL.fromArray(movingArray, fixedImage.GetSpacing())

polyAffine = PolyAffineCL(fixedCL, movingCL)
polyAffine.create_identity(4)
polyAffine.optimize(20)

warpedCL = polyAffine.movingCL
warpedArray = warpedCL.clarray.get().astype('float32')

warpedImage = sitk.GetImageFromArray(warpedArray)
warpedImage.SetSpacing(fixedImage.GetSpacing())
sitk.WriteImage(warpedImage, "warped.nrrd")
