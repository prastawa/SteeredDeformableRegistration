#!/usr/bin/python

#TODO: generate collection of images with known local trafo, display rotation panel, pick moments, associate moments (+ data) with trafo using learning
#TODO 2: application of learning, user pick region, select moments, then apply best trafo

import SimpleITK as sitk
import vtk
import vtk.util.numpy_support

import numpy as np
import scipy

import math
import sys

import nibabel as nib

def normalize(arr):
  minv = arr.min()
  maxv = arr.max()

  rangev = maxv - minv
  if rangev <= 0.0:
    return arr

  return (arr - minv) / rangev

def gaussian(shape):

  # TODO: center and width
  # TODO: make this be a part of polyaffine update instead?

  sx = shape[0]
  sy = shape[1]
  sz = shape[2]

  x0 = -sx / 2
  x1 = x0 + sx
  y0 = -sy / 2
  y1 = y0 + sy
  z0 = -sz / 2
  z1 = z0 + sz

  X, Y, Z = np.mgrid[x0:x1, y0:y1, z0:z1]
  G = np.exp(-(X**2/np.float(sx) + Y**2/np.float(sy) + Z**2/np.float(sz)) )
  #G /= G.sum()
  G /= np.max(G)

  return G

if len(sys.argv) != 2:
  print "Usage : " + sys.argv[0] + " <image>"
  sys.exit(-1)

sys.path.append("..")
from RegistrationCL import *

reader = sitk.ImageFileReader()
reader.SetFileName ( sys.argv[1] )
image = reader.Execute()

print "Read " + sys.argv[1]

# Convert data to VTK
image_array = np.float64( sitk.GetArrayFromImage(image) )

N = image_array.shape
print N

image_array = image_array[::2, ::2,::2]
N = image_array.shape
print "Down shape", N

#wait = input("PRESS ENTER TO CONTINUE.")

#minI = np.min(image_array)
#maxI = np.max(image_array)
#image_array = (image_array - minI) / (maxI - minI + 1e-4)

image_array_vtkorder = np.asfortranarray(image_array)
image_array_vtkorder = image_array_vtkorder.transpose(2,1,0)

vtkimport = vtk.vtkImageImport()
#image_array_string = image_array_vtkorder.tostring()
#vtkimport.CopyImportVoidPointer(image_array_string, len(image_array_string))
vtkimport.CopyImportVoidPointer(image_array_vtkorder, image_array_vtkorder.nbytes)
vtkimport.SetDataScalarTypeToDouble()
vtkimport.SetNumberOfScalarComponents(1)
vtkimport.SetDataExtent(0, N[0]-1, 0, N[1]-1, 0, N[2]-1)
vtkimport.SetWholeExtent(0, N[0]-1, 0, N[1]-1, 0, N[2]-1)
vtkimport.Update()

image_vtk = vtkimport.GetOutput()

#print image_array[40,60,40], " ?=? ",  image_vtk.GetScalarComponentAsDouble(40,60,40,0)
#print image_array[25,70,50], " ?=? ", image_vtk.GetScalarComponentAsDouble(25,70,50,0)

print image_vtk.GetDimensions()
print image_vtk.GetOrigin()
print image_vtk.GetSpacing()

fixed_image_vtk = image_vtk
fixed_image_array = image_array

# Create another image as a rotated version
cx = N[0]/2
cy = N[1]/2
cz = N[2]/2

trafo = vtk.vtkTransform()
#trafo.PostMultiply()
trafo.Translate(cx, cy, cz)
#trafo.RotateZ(8)
#trafo.RotateY(-15)
#trafo.RotateX(-40)
trafo.RotateZ(-30)
trafo.RotateY(20)
#trafo.Scale(0.9, 0.9, 0.9)
trafo.Translate(-cx, -cy, -cz)

print trafo.GetMatrix()

rotatef = vtk.vtkImageReslice()
rotatef.SetInput(image_vtk)
rotatef.SetInformationInput(image_vtk)
rotatef.SetResliceTransform(trafo)
rotatef.SetOutputExtent(0, N[0]-1, 0, N[1]-1, 0, N[2]-1)
rotatef.SetInterpolationModeToLinear()
rotatef.Update()

moving_image_vtk = rotatef.GetOutput()
print "moving after rot dims " , moving_image_vtk.GetDimensions()

# TODO: vtk grad?
#Ix, Iy, Iz = np.gradient(moving_image_array)
#moving_image_array = Ix * Ix + Iy * Iy + Iz * Iz
#gradf = vtk.vtkImageGradientMagnitude()
#gradf.SetInput(moving_image_vtk)
#gradf.HandleBoundariesOn()
#gradf.SetDimensionality(3)
#gradf.Update()

#moving_image_vtk = gradf.GetOutput()

moving_image_array = vtk.util.numpy_support.vtk_to_numpy(
  moving_image_vtk.GetPointData().GetScalars() ).reshape(N[2], N[1], N[0])
moving_image_array = moving_image_array.transpose(2, 1, 0)

#print moving_image_array[40,60,40], " ?=? ", moving_image_vtk.GetScalarComponentAsDouble(40,60,40,0)
#print moving_image_array[25,70,50], " ?=? ", moving_image_vtk.GetScalarComponentAsDouble(25,70,50,0)

print fixed_image_vtk.GetScalarRange()
print moving_image_vtk.GetScalarRange()

fixedCL = ImageCL("CPU")
fixedCL.fromArray(fixed_image_array)

movingCL = ImageCL("CPU")
movingCL.fromArray(moving_image_array)

center = np.array([cx, cy, cz], np.float32)
radius = np.array(N, np.float32) / 2

rotsteer = SteeringRotation(fixedCL, movingCL, center, radius)

R, T = rotsteer.steer()

for dim in range(3):
  R[dim, dim] += 1.0

print "R = ", R
print "T = ", T

# Get 3D registration output
A = np.zeros((4,4))
A[0:3,0:3] = R
A[0,3] = T[0]
A[1,3] = T[1]
A[2,3] = T[2]
A[3,3] = 1

print "A = ", A

print "moving dims before testtrafo ", moving_image_vtk.GetDimensions()

testmat = vtk.vtkMatrix4x4()
testmat.DeepCopy(A.ravel().tolist())
#testmat.Invert()

foo = vtk.vtkImageData()
foo.DeepCopy(moving_image_vtk)

#testtrafo = vtk.vtkMatrixToHomogeneousTransform()
#testtrafo.SetInput(testmat)
testtrafo = vtk.vtkTransform()
#trafo.Translate(cx, cy, cz)
testtrafo.SetMatrix(testmat)
#trafo.Translate(-cx, -cy, -cz)

testrotatef = vtk.vtkImageReslice()
testrotatef.SetInput(foo)
#testrotatef.SetInformationInput(moving_image_vtk)
#testrotatef.SetResliceTransform(testtrafo.MakeTransform())
testrotatef.SetResliceTransform(testtrafo)
testrotatef.SetInterpolationModeToLinear()
testrotatef.Update()

testimage_vtk = testrotatef.GetOutput()

testimage_array = vtk.util.numpy_support.vtk_to_numpy( testimage_vtk.GetPointData().GetScalars() ).reshape(N[2], N[1], N[0])
testimage_array = testimage_array.transpose(2, 1, 0)

print "fixed dims ", fixed_image_vtk.GetDimensions()
print "moving dims ", moving_image_vtk.GetDimensions()
print "test dims ", testimage_vtk.GetDimensions()

print "moving shape ", moving_image_array.shape

"""
# Get a slice after 3D registration
A = np.zeros((4,4))
A[0:3,0] = F[:,0]
A[0:3,1] = F[:,1]
A[0:3,2] = F[:,2]
A[0,3] = cx
A[1,3] = cy
A[2,3] = cz
A[3,3] = 1

resliceAxes = vtk.vtkMatrix4x4()
resliceAxes.DeepCopy(A.ravel().tolist())

reslicef = vtk.vtkImageReslice()
reslicef.SetInput(testimage_vtk)
reslicef.SetInformationInput(testimage_vtk)
reslicef.SetOutputExtent(0, panelSize-1, 0, panelSize-1, 0, 0)
reslicef.SetOutputDimensionality(2)
reslicef.SetResliceAxes(resliceAxes)
reslicef.SetInterpolationModeToLinear()
reslicef.Update()

sliceDims = reslicef.GetOutput().GetDimensions()
testimage_display = vtk.util.numpy_support.vtk_to_numpy( reslicef.GetOutput().GetPointData().GetScalars() ).reshape(sliceDims[1], sliceDims[0])
testimage_display = np.transpose(testimage_display)

#fixed_image_display = np.squeeze( fixed_image_array[:,:,cz] )
#moving_image_display = np.squeeze( moving_image_array[:,:,cz] )
#testimage_display = np.squeeze( testimage_array[:,:,cz] )
fixed_image_display = fixed2D
moving_image_display = moving2D

print "2D |fixed - moving| = ", np.linalg.norm((fixed_image_display - moving_image_display).ravel())
print "2D |fixed - test| = ", np.linalg.norm((fixed_image_display - testimage_display).ravel())
#print "2D |fixed - moving| = ", np.sum(np.power(fixed_image_display - moving_image_display, 2))
#print "2D |fixed - test| = ", np.sum(np.power(fixed_image_display - testimage_display, 2))
"""

print "3D |fixed - moving| = ", np.linalg.norm((fixed_image_array - moving_image_array).ravel())
print "3D |fixed - test| = ", np.linalg.norm((fixed_image_array - testimage_array).ravel())
#print "3D |fixed - moving| = ", np.sum(np.power(fixed_image_array - moving_image_array, 2))
#print "3D |fixed - test| = ", np.sum(np.power(fixed_image_array - testimage_array, 2))

outim = nib.Nifti1Image(testimage_array, np.identity(4))
outim.to_filename("testrot.nii")

