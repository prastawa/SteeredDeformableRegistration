
import vtk, qt, ctk, slicer
import math
#import SimpleITK as sitk
#import sitkUtils
import Queue
import time

import cProfile, pstats, StringIO

################################################################################
#
# CL support classes
#
################################################################################

import pyopencl as cl
import pyopencl.array as cla
import numpy as n

#
# ImageCL
#
# Adds support for image registration operations to Slicer image data
# using PyOpenCL
#

class ImageCL:
  def __init__(self, clqueue=None, clprogram=None):
    self.clqueue = clqueue
    self.clprogram = clprogram

    self.origin = [0.0, 0.0, 0.0]
    self.shape = [0, 0, 0]
    self.spacing = [1.0, 1.0, 1.0]

    self.clarray = None
    self.clsize = None
    self.clspacing = None

  def __del__(self):
    self.clarray = None
    self.clsize = None
    self.clspacing = None

  def clone_empty(self):
    outimgcl = ImageCL(self.clqueue, self.clprogram)
    outimgcl.origin = list(self.origin)
    outimgcl.shape = list(self.shape)
    outimgcl.spacing = list(self.spacing)

    outimgcl.clarray = None
    if self.clsize is not None:
      outimgcl.clsize = self.clsize.copy()
    if self.clspacing is not None:
      outimgcl.clspacing = self.clspacing.copy()

    return outimgcl

  def clone(self):
    outimgcl = self.clone_empty()
    if self.clarray is not None:
      outimgcl.clarray = self.clarray.copy()
    return outimgcl

  def fromVolume(self, volume):

    # Cast to float
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToFloat()
    castf.SetInput( volume.GetImageData() )
    castf.Update()

    vtkimage = castf.GetOutput()

    # VTK image data are stored in reverse order
    self.shape = list(vtkimage.GetDimensions())
    self.shape.reverse()

    self.origin = list(volume.GetOrigin())
    self.origin.reverse()

    self.spacing = list(volume.GetSpacing())
    self.spacing.reverse()

    # Store original orientation in case need to map back
    self.originalIJKToRAS = vtk.vtkMatrix4x4()
    volume.GetIJKToRASDirectionMatrix(self.originalIJKToRAS)

    nspacing = n.zeros((3,), dtype=n.float32)
    for dim in xrange(3):
      nspacing[dim] = self.spacing[dim]
    self.clspacing = cla.to_device(self.clqueue, nspacing)

    nsize = n.zeros((3,), dtype=n.uint32)
    for dim in xrange(3):
      nsize[dim] = self.shape[dim]
    self.clsize = cla.to_device(self.clqueue, nsize)

    narray = vtk.util.numpy_support.vtk_to_numpy(
        vtkimage.GetPointData().GetScalars()).reshape(self.shape)
    #narray = narray.astype('float32')
    self.clarray = cl.array.to_device(self.clqueue, narray)

  def toVTKImage(self):
    narray = self.clarray.get().astype('float32')
    #narray = narray.transpose(2, 1, 0)

    vtkarray = vtk.util.numpy_support.numpy_to_vtk(narray.flatten(), deep=True)
 
    # NOTE: vtk image does not contain image and spacing, all info in volume
    vtkimage = vtk.vtkImageData()
    vtkimage.SetScalarTypeToFloat()
    vtkimage.SetNumberOfScalarComponents(1)
    vtkimage.SetExtent(
      0, self.shape[2]-1, 0, self.shape[1]-1, 0, self.shape[0]-1)
    vtkimage.AllocateScalars()
    vtkimage.GetPointData().SetScalars(vtkarray)
    vtkimage.GetPointData().GetScalars().Modified()
    vtkimage.GetPointData().Modified()
    vtkimage.Modified()

    return vtkimage

  def fill(self, value):
    self.clarray.fill(value)

  def normalize(self):

    # Brute force CPU approach
    #narray = self.clarray.get()
    #minp = narray.min()
    #maxp = narray.max()

    # NOTE: this may not work on Nvidia and Windows, due to space in header
    # include location
    # WORKAROUND: edit Slicer/lib/Python/Lib/site-packages/pyopencl/reduction.py
    # and insert code from cl/pyopencl-complex.h manually
    clminp = cl.array.min(self.clarray, self.clqueue)
    clmaxp = cl.array.max(self.clarray, self.clqueue)
    # Convert reductions to scalars
    minp = clminp.get()[()]
    maxp = clmaxp.get()[()]

    range = maxp - minp
    if range > 0.0:
      self.clarray -= minp
      self.clarray /= range

  def scale(self, v):
    self.clarray *= v
    
  def add(self, otherimgcl):
    outimgcl = self.clone_empty()
    outimgcl.clarray = self.clarray + otherimgcl.clarray
    return outimgcl
    
  def subtract(self, otherimgcl):
    outimgcl = self.clone_empty()
    outimgcl.clarray = self.clarray - otherimgcl.clarray
    return outimgcl
    
  def multiply(self, otherimgcl):
    outimgcl = self.clone_empty()
    outimgcl.clarray = self.clarray * otherimgcl.clarray
    return outimgcl

  def add_inplace(self, otherimgcl):
    self.clarray += otherimgcl.clarray
    return self

  def subtract_inplace(self, otherimgcl):
    self.clarray -= otherimgcl.clarray
    return self

  def multiply_inplace(self, otherimgcl):
    self.clarray *= otherimgcl.clarray
    return self

  def minmax(self):
    #TODO
    #use cl.reduction.ReductionKernel
    #https://github.com/pyopencl/pyopencl/blob/master/examples/demo-struct-reduce.py
    pass

  def min(self):
    #return self.clarray.get().min()

    # NOTE: see note on normalize()
    clminp = cl.array.min(self.clarray, self.clqueue)
    minp = clminp.get()[()]
    return minp

  def max(self):
    #return self.clarray.get().max()

    # NOTE: see note on normalize()
    clmaxp = cl.array.max(self.clarray, self.clqueue)
    maxp = clmaxp.get()[()]
    return maxp

  def gradient(self):
    gradx = self.clone()
    grady = self.clone()
    gradz = self.clone()

    self.clprogram.gradient(self.clqueue, self.shape, None,
      self.clarray.data,
      self.clsize.data, self.clspacing.data,
      gradx.clarray.data, grady.clarray.data, gradz.clarray.data).wait()

    return [gradx, grady, gradz]

  def gradient_magnitude(self):
    [gx, gy, gz] = self.gradient()
    mag = gx.multiply(gx)
    mag.add_inplace(gy.multiply(gy))
    mag.add_inplace(gz.multiply(gz))

    return mag

  def discrete_gaussian(self, sigma):
    outimgcl = self.clone()

    var = n.zeros((1,), dtype=n.float32)
    var_array = cla.to_device(self.clqueue, var)
    width = n.zeros((1,), dtype=n.int32)
    width_array = cla.to_device(self.clqueue, width)

    tempclarray = outimgcl.clarray.copy()

    var_array[0] = (sigma / self.spacing[0]) ** 2
    width_array[0] = math.ceil(3*sigma)
    self.clprogram.gaussian_x(self.clqueue, self.shape, None,
      tempclarray.data, self.clsize.data,
      var_array.data, width_array.data, outimgcl.clarray.data).wait()

    var_array[0] = (sigma / self.spacing[1]) ** 2
    width_array[0] = math.ceil(3*sigma)
    self.clprogram.gaussian_y(self.clqueue, self.shape, None,
      outimgcl.clarray.data, self.clsize.data,
      var_array.data, width_array.data,
      tempclarray.data).wait()

    var_array[0] = (sigma / self.spacing[2]) ** 2
    width_array[0] = math.ceil(3*sigma)
    self.clprogram.gaussian_z(self.clqueue, self.shape, None,
      tempclarray.data, self.clsize.data,
      var_array.data, width_array.data,
      outimgcl.clarray.data).wait()

    return outimgcl

  def recursive_gaussian(self, sigma):
    outimgcl = self.clone()

    sigma_array = cl.array.zeros(self.clqueue, (1,), n.float32)

    sizeX = self.shape[0]
    sizeY = self.shape[1]
    sizeZ = self.shape[2]

    sigma_array[0] = sigma / self.spacing[2]
    outimgcl.clprogram.recursive_gaussian_z(outimgcl.clqueue,
      (sizeX, sizeY), None,
      outimgcl.clarray.data, outimgcl.clsize.data, sigma_array.data).wait()
    sigma_array[0] = sigma / self.spacing[1]
    outimgcl.clprogram.recursive_gaussian_y(outimgcl.clqueue,
      (sizeX, sizeZ), None,
      outimgcl.clarray.data, outimgcl.clsize.data, sigma_array.data).wait()
    sigma_array[0] = sigma / self.spacing[0]
    outimgcl.clprogram.recursive_gaussian_x(outimgcl.clqueue,
      (sizeY, sizeZ), None,
      outimgcl.clarray.data, outimgcl.clsize.data, sigma_array.data).wait()

    return outimgcl

  def resample(self, targetShape):

    # TODO do some filtering if downsampling?
    """
    isDownsampling = False
    for dim in xrange(3):
      if targetShape[dim] <= self.shape[dim] / 2:
        isDownsampling = True
        break
    minspacing = min(self.spacing)
    if isDownsampling:
      smoothimgcl = self.discrete_gaussian(minspacing)
    """

    outimgcl = self.clone_empty()

    for dim in xrange(3):
      outimgcl.shape[dim] = targetShape[dim]
      outimgcl.spacing[dim] = \
        (self.spacing[dim] * self.shape[dim]) / targetShape[dim]

      outimgcl.clsize[dim] = targetShape[dim]
      outimgcl.clspacing[dim] = outimgcl.spacing[dim]


    # Resize cl array
    outimgcl.clarray = cl.array.zeros(self.clqueue, targetShape, n.float32)

    hxclarray = cl.array.zeros_like(outimgcl.clarray)
    hyclarray = cl.array.zeros_like(outimgcl.clarray)
    hzclarray = cl.array.zeros_like(outimgcl.clarray)

    self.clprogram.identity(self.clqueue, targetShape, None,
      outimgcl.clsize.data, outimgcl.clspacing.data,
      hxclarray.data,  hyclarray.data, hzclarray.data).wait()
    
    self.clprogram.interpolate(self.clqueue, targetShape, None,
      self.clarray.data,
      self.clsize.data, self.clspacing.data,
      hxclarray.data, hyclarray.data, hzclarray.data,
      outimgcl.clarray.data,
      outimgcl.clsize.data).wait()

    hxclarray = None
    hyclarray = None
    hzclarray = None

    """

    # CPU
    outimgcl = self.clone()

    vtkimage = self.toVTKImage()

    outspacing = list(self.spacing)
    for dim in xrange(3):
      outspacing[dim] = (self.spacing[dim] * self.shape[dim]) / targetShape[dim]

    outimgcl.shape = targetShape
    outimgcl.spacing = outspacing

    resizef = vtk.vtkImageResize()
    resizef.SetResizeMethodToOutputDimensions()
    resizef.SetOutputDimensions(targetShape[2], targetShape[1], targetShape[0])
    resizef.SetInput(vtkimage)
    resizef.Update()
    """

    return outimgcl

#
# DeformationCL
#
# Represent deformation map h using a list of 3 ImageCL objects
# with support for composition (h1 \circ h2), and warping of scalar volumes
#
# Warping is applied as Iwarped = I(h)
#
class DeformationCL:

  def __init__(self, imgcl, hlist=None):

    self.clgrid = imgcl

    self.clqueue = imgcl.clqueue
    self.clprogram = imgcl.clprogram

    if hlist is None:
      # Assign identity mapping if mapping not specified at init
      self.hx = imgcl.clone()
      self.hy = imgcl.clone()
      self.hz = imgcl.clone()
      self.set_identity()
    else:
      self.hx = hlist[0]
      self.hy = hlist[1]
      self.hz = hlist[2]

  def __del__(self):
    self.clgrid = None
    self.hx = None
    self.hy = None
    self.hz = None

  def clone(self):
    hcopies = [self.hx.clone(), self.hy.clone(), self.hz.clone()]
    return DeformationCL(self.clgrid, hcopies)

  def set_mapping(self, hx, hy, hz):
    self.hx = hx
    self.hy = hy
    self.hz = hz
    self.clqueue = self.hx.clqueue
    self.clprogram = self.hx.clprogram

  def set_identity(self):

    clsize = self.hx.clsize
    clspacing = self.hx.clspacing

    self.clprogram.identity(self.clqueue, self.hx.shape, None,
      clsize.data, clspacing.data,
      self.hx.clarray.data,  self.hy.clarray.data, self.hz.clarray.data).wait()

  def add_velocity(self, velocList):
    self.hx.add_inplace(velocList[0])
    self.hy.add_inplace(velocList[1])
    self.hz.add_inplace(velocList[2])

  def maxMagnitude(self):
    magimg = self.hx.multiply(self.hx)
    magimg.add_inplace( self.hy.multiply(self.hy) )
    magimg.add_inplace( self.hz.multiply(self.hz) )
    return math.sqrt( magimg.max() )

  def resample(self, targetShape):
    hx_new = self.hx.resample(targetShape)
    hy_new = self.hy.resample(targetShape)
    hz_new = self.hz.resample(targetShape)

    H_new = [hx_new, hy_new, hz_new]

    return DeformationCL(hx_new, H_new)

  def applyTo(self, vol):
    # Output image is in the same grid as h
    outimgcl = self.hx.clone()

    self.clprogram.interpolate(vol.clqueue, self.hx.shape, None,
      vol.clarray.data,
      vol.clsize.data, vol.clspacing.data,
      self.hx.clarray.data, self.hy.clarray.data, self.hz.clarray.data,
      outimgcl.clarray.data,
      outimgcl.clsize.data).wait()

    return outimgcl

  def compose(self, otherdef):
    hx_new = otherdef.applyTo(self.hx)
    hy_new = otherdef.applyTo(self.hy)
    hz_new = otherdef.applyTo(self.hz)

    H_new = [hx_new, hy_new, hz_new]

    return DeformationCL(otherdef.clgrid, H_new)

  def splat_force(self, forces):
    # TODO: CL kernel that splats forces fx,fy,fz to hx,hy,hz
    # splat(fx, fy, fz, fPositions, numF, hx, hy, hz, hsize, hspacing)
    # at each pos, check proximity to forces then add weighted values
    pass

# TODO add support for downsampling and upsampling
# first in ImageCL then here as well

# TODO ???
# At initialization, invoke affine registration using CLI module:
# self.parameters = {}
# self.parameters['InitialTransform'] = initialTransform.GetID()
# self.parameters['FixedImageFileName'] = fixedVolume.GetID()
# self.parameters['MovingImageFileName'] = movingVolume.GetID()
# self.parameters['OutputTransform'] = outputTransform.GetID()
# #self.parameters['ResampledImageFileName'] = outputVolume.GetID()
# self.parameters['Iterations']=self.regIterationSlider.value
# slicer.cli.run(slicer.modules.affineregistration, None,
#                            self.parameters, wait_for_completion=True)

#
# SteeredFluidReg
#

class SteeredFluidReg:
  def __init__(self, parent):
    parent.title = "SteeredFluidReg"
    parent.categories = ["Registration"]
    parent.dependencies = []
    parent.contributors = ["Marcel Prastawa (GE), James Miller (GE), Steve Pieper (Isomics)"] # replace with "Firstname Lastname (Org)"
    parent.helpText = """
    Steerable fluid registration example as a scripted loadable extension.
    """
    parent.acknowledgementText = """
    Funded by NIH grant 3P41RR013218.
""" # replace with organization, grant and thanks.
    self.parent = parent

################################################################################
#
# SteeredFluidRegWidget
#
################################################################################

class SteeredFluidRegWidget:
  def __init__(self, parent = None):
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout = self.parent.layout()
    if not parent:
      self.setup()
      self.parent.show()

    self.logic = SteeredFluidRegLogic()

    self.interaction = False
    
##    self.parameterNode = None
##    self.parameterNodeTag = None
##
##    # connections is a list of widget/signal/slot tripples
##    # for the options gui that can be connected/disconnected
##    # as needed to prevent triggering mrml updates while
##    # updating the state of the gui
##    # - each level of the inheritance tree can add entries
##    #   to this list for use by the connectWidgets
##    #   and disconnectWidgets methods
##    self.connections = []
##    self.connectionsConnected = False
##
##    # 1) find the parameter node in the scene and observe it
##    # 2) set the defaults (will only set them if they are not
##    # already set)
##    self.updateParameterNode(self.parameterNode, vtk.vtkCommand.ModifiedEvent)
##    self.setMRMLDefaults()
##
##    # TODO: change this to look for specfic events (added, removed...)
##    # but this requires being able to access events by number from wrapped code
##    tag = slicer.mrmlScene.AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateParameterNode)
##    self.observerTags.append( (slicer.mrmlScene, tag) )

  def __del__(self):
    self.destroy()
    if self.parameterNode:
      self.parameterNode.RemoveObserver(self.parameterNodeTag)
    for tagpair in self.observerTags:
      tagpair[0].RemoveObserver(tagpair[1])

  def connectWidgets(self):
    if self.connectionsConnected: return
    for widget,signal,slot in self.connections:
      success = widget.connect(signal,slot)
      if not success:
        print("Could not connect {signal} to {slot} for {widget}".format(
          signal = signal, slot = slot, widget = widget))
    self.connectionsConnected = True

  def disconnectWidgets(self):
    if not self.connectionsConnected: return
    for widget,signal,slot in self.connections:
      success = widget.disconnect(signal,slot)
      if not success:
        print("Could not disconnect {signal} to {slot} for {widget}".format(
          signal = signal, slot = slot, widget = widget))
    self.connectionsConnected = False

  def setup(self):
    # Instantiate and connect widgets ...

    # reload button
    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    self.reloadButton.name = "SteeredFluidReg Reload"
    self.layout.addWidget(self.reloadButton)
    self.reloadButton.connect('clicked()', self.onReload)

    #
    # IO Collapsible button
    #
    ioCollapsibleButton = ctk.ctkCollapsibleButton()
    ioCollapsibleButton.text = "Volume and Transform Parameters"
    self.layout.addWidget(ioCollapsibleButton)

    # Layout within the parameter collapsible button
    ioFormLayout = qt.QFormLayout(ioCollapsibleButton)

    # # InitialTransform node selector
    # self.initialTransformSelector = slicer.qMRMLNodeComboBox()
    # self.initialTransformSelector.objectName = 'initialTransformSelector'
    # self.initialTransformSelector.toolTip = "The initial transform volume."
    # self.initialTransformSelector.nodeTypes = ['vtkMRMLGridTransformNode']
    # self.initialTransformSelector.noneEnabled = True
    # self.initialTransformSelector.addEnabled = True
    # self.initialTransformSelector.removeEnabled = True
    # ioFormLayout.addRow("Initial Transform:", self.initialTransformSelector)
    # self.initialTransformSelector.setMRMLScene(slicer.mrmlScene)
    # self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        # self.initialTransformSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Fixed Volume node selector
    self.fixedSelector = slicer.qMRMLNodeComboBox()
    self.fixedSelector.objectName = 'fixedSelector'
    self.fixedSelector.toolTip = "The fixed volume."
    self.fixedSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.fixedSelector.noneEnabled = False
    self.fixedSelector.addEnabled = False
    self.fixedSelector.removeEnabled = False
    ioFormLayout.addRow("Fixed Volume:", self.fixedSelector)
    self.fixedSelector.setMRMLScene(slicer.mrmlScene)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.fixedSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Moving Volume node selector
    self.movingSelector = slicer.qMRMLNodeComboBox()
    self.movingSelector.objectName = 'movingSelector'
    self.movingSelector.toolTip = "The moving volume."
    self.movingSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.movingSelector.noneEnabled = False
    self.movingSelector.addEnabled = False
    self.movingSelector.removeEnabled = False
    ioFormLayout.addRow("Moving Volume:", self.movingSelector)
    self.movingSelector.setMRMLScene(slicer.mrmlScene)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.movingSelector, 'setMRMLScene(vtkMRMLScene*)')

    # # Transform node selector
    # self.transformSelector = slicer.qMRMLNodeComboBox()
    # self.transformSelector.objectName = 'transformSelector'
    # self.transformSelector.toolTip = "The transform volume."
    # self.transformSelector.nodeTypes = ['vtkMRMLGridTransformNode']
    # self.transformSelector.noneEnabled = True
    # self.transformSelector.addEnabled = True
    # self.transformSelector.removeEnabled = True
    # ioFormLayout.addRow("Moving To Fixed Transform:", self.transformSelector)
    # self.transformSelector.setMRMLScene(slicer.mrmlScene)
    # self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        # self.transformSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Output Volume node selector
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.objectName = 'outputSelector'
    self.outputSelector.toolTip = "The output volume."
    self.outputSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.outputSelector.noneEnabled = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    ioFormLayout.addRow("Output Volume:", self.outputSelector)
    self.outputSelector.setMRMLScene(slicer.mrmlScene)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.outputSelector, 'setMRMLScene(vtkMRMLScene*)')
    
    selectors = (self.fixedSelector, self.movingSelector, self.outputSelector)
    for selector in selectors:
      selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateLogicFromGUI)
    #
    # Interaction options collapsible button
    #

    # TODO: pull, expand, shrink checkbuttons -- default is pull

    uiOptCollapsibleButton = ctk.ctkCollapsibleButton()
    uiOptCollapsibleButton.text = "Steering UI Parameters"
    self.layout.addWidget(uiOptCollapsibleButton)

    # Layout within the parameter collapsible button
    uiOptFormLayout = qt.QFormLayout(uiOptCollapsibleButton)

    # Interaction mode
    steerModeLayout = qt.QGridLayout()
    self.pullModeRadio = qt.QRadioButton("Pull")
    self.expandModeRadio = qt.QRadioButton("Expand")
    self.shrinkModeRadio = qt.QRadioButton("Shrink")
    steerModeLayout.addWidget(self.pullModeRadio, 0, 0)
    steerModeLayout.addWidget(self.expandModeRadio, 0, 1)
    steerModeLayout.addWidget(self.shrinkModeRadio, 0, 2)

    steerModeRadios = (self.pullModeRadio, self.expandModeRadio, self.shrinkModeRadio)
    for r in steerModeRadios:
      r.connect('clicked(bool)', self.updateLogicFromGUI)

    self.pullModeRadio.checked = True

    uiOptFormLayout.addRow("Steering Mode: ", steerModeLayout)

    # Floating image opacity
    self.opacitySlider = ctk.ctkSliderWidget()
    self.opacitySlider.decimals = 2
    self.opacitySlider.singleStep = 0.05
    self.opacitySlider.minimum = 0.0
    self.opacitySlider.maximum = 1.0
    self.opacitySlider.toolTip = "Transparency of floating moving image"
    uiOptFormLayout.addRow("Floating Image Opacity:", self.opacitySlider)

    # Draw iterations
    self.drawIterationSlider = ctk.ctkSliderWidget()
    self.drawIterationSlider.decimals = 0
    self.drawIterationSlider.singleStep = 1
    self.drawIterationSlider.minimum = 1
    self.drawIterationSlider.maximum = 100
    self.drawIterationSlider.toolTip = "Update and draw every N iterations"
    uiOptFormLayout.addRow("Redraw Iterations:", self.drawIterationSlider)

    self.drawIterationSlider.value = self.logic.drawIterations
    self.opacitySlider.value = self.logic.opacity

    #
    # Registration regOptions collapsible button
    #
    regOptCollapsibleButton = ctk.ctkCollapsibleButton()
    regOptCollapsibleButton.text = "Registration Parameters"
    self.layout.addWidget(regOptCollapsibleButton)

    # Layout within the parameter collapsible button
    regOptFormLayout = qt.QFormLayout(regOptCollapsibleButton)

    spinBoxLayout = qt.QGridLayout()
    self.gridSpinBoxes = [qt.QSpinBox(), qt.QSpinBox(), qt.QSpinBox()]
    for dim in xrange(3):
      self.gridSpinBoxes[dim].setValue(64)
      spinBoxLayout.addWidget(self.gridSpinBoxes[dim], 0, dim)

    regOptFormLayout.addRow("Deformation Grid: ", spinBoxLayout)
    # TODO: regridding callback in logic

    # Fluid kernel width
    self.fluidKernelWidth = ctk.ctkSliderWidget()
    self.fluidKernelWidth.decimals = 1
    self.fluidKernelWidth.singleStep = 0.5
    self.fluidKernelWidth.minimum = 0.5
    self.fluidKernelWidth.maximum = 50.0
    self.fluidKernelWidth.toolTip = "Area of effect for deformation forces."
    regOptFormLayout.addRow("Deformation Stiffness:", self.fluidKernelWidth)

    self.fluidKernelWidth.value = self.logic.fluidKernelWidth

    sliders = (self.drawIterationSlider, self.fluidKernelWidth, self.opacitySlider)
    for slider in sliders:
      slider.connect('valueChanged(double)', self.updateLogicFromGUI)

    #
    # Developer collapsible button
    #

    devCollapsibleButton = ctk.ctkCollapsibleButton()
    devCollapsibleButton.text = "Developer Parameters"
    self.layout.addWidget(devCollapsibleButton)

    # Layout within the parameter collapsible button
    devFormLayout = qt.QFormLayout(devCollapsibleButton)

    self.profilingButton = qt.QCheckBox("Code Profiling")
    self.profilingButton.toolTip = "Obtain statistics of code execution."
    self.profilingButton.name = "SteeredFluidReg Profiling"
    self.profilingButton.connect('toggled(bool)', self.toggleProfiling)
    devFormLayout.addWidget(self.profilingButton)

    self.debugButton = qt.QCheckBox("Print Debug Messages")
    self.debugButton.toolTip = "Display extra messages in Python console."
    self.debugButton.name = "SteeredFluidReg Debug"
    self.debugButton.connect('toggled(bool)', self.updateLogicFromGUI)
    devFormLayout.addWidget(self.debugButton)

    #
    # Execution triggers
    #

    # Start button
    self.regButton = qt.QPushButton("Start")
    self.regButton.toolTip = "Run steered registration"
    self.regButton.checkable = True
    self.layout.addWidget(self.regButton)
    self.regButton.connect('toggled(bool)', self.onStart)

    # Add vertical spacer
    #self.layout.addStretch(1)

    self.profiler = cProfile.Profile()

    # to support quicker development:
    import os
    if (os.getenv('USERNAME') == '212357326') or (os.getenv('USER') == 'prastawa'):
    #if False:
      self.logic.testingData()
      self.fixedSelector.setCurrentNode(slicer.util.getNode('testbrain1'))
      # Disable anyway, in case checkbox clicked during execution
      self.movingSelector.setCurrentNode(slicer.util.getNode('testbrain2'))
      # self.transformSelector.setCurrentNode(slicer.util.getNode('movingToFixed'))
      # self.initialTransformSelector.setCurrentNode(slicer.util.getNode('movingToFixed'))


  def toggleProfiling(self, checked):
    if checked:
      self.profiler.enable()
    else:
      self.profiler.disable()

  def updateLogicFromGUI(self, args):

    # Handled by logic on start
    #self.logic.useFixedVolume(self.fixedSelector.currentNode())
    #self.logic.useMovingVolume(self.movingSelector.currentNode())

    #TODO
    # self.logic.transform = self.transformSelector.currentNode()

    # TODO: hook with grid transform, for now just update background image from GPU
    # KEY PP: allow motion, but crashes???
    # if(self.logic.transform is not None):
    #   self.logic.moving.SetAndObserveTransformNodeID(self.logic.transform.GetID())
    
    self.logic.drawIterations = self.drawIterationSlider.value
    self.logic.fluidKernelWidth = self.fluidKernelWidth.value
    self.logic.opacity = self.opacitySlider.value

    self.logic.debugMessages = self.debugButton.checked

    if self.pullModeRadio.checked:
      self.logic.steerMode = "pull"
    if self.expandModeRadio.checked:
      self.logic.steerMode = "expand"
    if self.shrinkModeRadio.checked:
      self.logic.steerMode = "shrink"

    # TODO: signal logic that objective function may have changed
    # trigger appropriate behaviors (ex. delta adjust)
 
  def onResetButtonToggled(self):
    self.logic.actionState = "reset"
    #TODO
    #self.logic.reset()
   
  def onStart(self, checked):

    if checked:
      self.regButton.text = "Stop"

      self.logic.automaticRegistration = True

      layoutManager = slicer.app.layoutManager()
      layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutThreeOverThreeView)
      
      # Set up display of volumes as composites, and create output node if not specified
      fixedVolume = self.fixedSelector.currentNode()
      movingVolume = self.movingSelector.currentNode()
      outputVolume = self.outputSelector.currentNode()

      self.logic.useFixedVolume(fixedVolume)
      self.logic.useMovingVolume(movingVolume)

      # Initialize output using moving volume
      self.logic.initOutputVolume(outputVolume)

      #cool1 = slicer.vtkMRMLColorTableNode()
      #cool1.SetTypeToCool1()
      #fixedVolume.GetScene().AddNode(cool1)

      #warm1 = slicer.vtkMRMLColorTableNode()
      #warm1.SetTypeToWarm1()
      #movingVolume.GetScene().AddNode(warm1)

      #fixedDisplay = fixedVolume.GetDisplayNode()
      #fixedDisplay.SetAndObserveColorNodeID(cool1.GetID())

      #movingDisplay = movingVolume.GetDisplayNode()
      #movingDisplay.SetAndObserveColorNodeID(warm1.GetID())
      
      # TODO: move parts to startDeformableReg()

      #orig = movingVolume.GetImageData().GetOrigin()
      #sp = movingVolume.GetImageData().GetSpacing()
      #print "Call build id with orig = " + str(orig) + " sp = " + str(sp)
      
      # vl = slicer.modules.volumes.logic()
      # self.gridVolume = vl.CloneVolume(slicer.mrmlScene, movingVolume, "warped-grid")
      # self.gridVolume = slicer.vtkMRMLScalarVolumeNode()
      # self.gridVolume.CopyWithScene(movingVolume)
      # self.gridVolume.SetAndObserveStorageNodeID(None)
      # self.gridVolume.Modified()
      # self.gridVolume.LabelMapOn()
      # # self.gridVolume.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels")
      # self.gridVolume.SetName("warped-grid")
      
      # print "Grid volume id = " + str(self.gridVolume.GetID())
  
      # gridImage = self.buildGrid(movingVolume.GetImageData())
      # #self.gridVolume.GetImageData().GetPointData().SetScalars( gridImage.GetPointData().GetScalars() )
      # self.gridVolume.SetAndObserveImageData(gridImage)

      self.logic.interaction = True
      self.logic.automaticRegistration = True

      print('Automatic registration = %d' %(self.logic.automaticRegistration))
      
      self.logic.startSteeredRegistration()

    else:

      self.regButton.text = "Start"

      self.logic.automaticRegistration = False
      
      self.logic.interaction=False
      self.logic.stopSteeredRegistration()

      # TODO: clear out temporary variables

      # TODO: Use a grid transform and make it observable?
      #if(self.logic.transform is not None): 
      #  self.logic.moving.SetAndObserveTransformNodeID(self.logic.transform.GetID())
      
      print('Automatic registration = %d' %(self.logic.automaticRegistration))

      self.profiler.disable()
      if self.profilingButton.isChecked():
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
     
  def buildGrid(self, inputImage):
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToShort()
    castf.SetInput(inputImage)
    castf.Update()

    #gridImage = castf.GetOutput()
    gridImage = vtk.vtkImageData()
    gridImage.DeepCopy(castf.GetOutput())
    gridImage.GetPointData().GetScalars().FillComponent(0, 0)
    
    size = gridImage.GetDimensions()
    
    for i in xrange(size[0]):
      for j in xrange(size[1]):
        for k in xrange(size[2]):
          if i > 0 and (i % 20) < 3:
            gridImage.SetScalarComponentFromDouble(i, j, k, 0, 1.0)

    for j in xrange(size[1]):
      for i in xrange(size[0]):
        for k in xrange(size[2]):
          if j > 0 and (j % 20) < 3:
            gridImage.SetScalarComponentFromDouble(i, j, k, 0, 1.0)
            
    for k in xrange(size[2]):
      for i in xrange(size[0]):
        for j in xrange(size[1]):
          if k > 0 and (k % 20) < 3:
            gridImage.SetScalarComponentFromDouble(i, j, k, 0, 1.0)
            
    return gridImage
    
  
  #########################################################################
  
  def onReload(self,moduleName="SteeredFluidReg"):
    """Generic reload method for any scripted module.
    ModuleWizard will subsitute correct default moduleName.
    """
    import imp, sys, os, slicer

    widgetName = moduleName + "Widget"

    # reload the source code
    # - set source file path
    # - load the module to the global space
    filePath = eval('slicer.modules.%s.path' % moduleName.lower())
    p = os.path.dirname(filePath)
    if not sys.path.__contains__(p):
      sys.path.insert(0,p)
    fp = open(filePath, "r")
    globals()[moduleName] = imp.load_module(
        moduleName, fp, filePath, ('.py', 'r', imp.PY_SOURCE))
    fp.close()

    # rebuild the widget
    # - find and hide the existing widget
    # - create a new widget in the existing parent
    parent = slicer.util.findChildren(name='%s Reload' % moduleName)[0].parent()
    for child in parent.children():
      try:
        child.hide()
      except AttributeError:
        pass
    globals()[widgetName.lower()] = eval(
        'globals()["%s"].%s(parent)' % (moduleName, widgetName))
    globals()[widgetName.lower()].setup()


################################################################################
#
# SteeredFluidReg logic
#
################################################################################

class SteeredFluidRegLogic(object):
#TODO
  """ Implement a template matching optimizer that is
  integrated with the slicer main loop.
  """

  def __init__(self):
    self.interval = 1000
    self.timer = None

    # parameter defaults
    self.drawIterations = 5
    self.fluidKernelWidth = 15.0
    self.opacity = 0.5

    # TODO
    #self.transform = ?

    # optimizer state variables
    self.iteration = 0
    self.interaction = False

    self.steerMode = "pull"

    self.position = []
    self.paintCoordinates = []

    self.lastEventPosition = [0.0, 0.0, 0.0]
    self.startEventPosition = [0.0, 0.0, 0.0]
    
    # Queue containing info on arrow draw events, tuples of (Mtime, xy0, RAS0, xy1, RAS1, sliceWidget)
    self.arrowQueue = Queue.Queue()

    self.arrowStartXY = (0, 0, 0)
    self.arrowEndXY = (0, 0, 0)
    
    self.arrowStartRAS = [0.0, 0.0, 0.0]
    self.arrowEndRAS = [0.0, 0.0, 0.0]

    print("Reload")

    self.actionState = "idle"
    self.interactorObserverTags = []
    
    self.styleObserverTags = []
    self.sliceWidgetsPerStyle = {}

    self.nodeIndexPerStyle = {}
    self.sliceNodePerStyle = {}
    
    self.lastDrawMTime = 0
    self.lastDrawSliceWidget = None

    self.expandShrinkVectors = []
    
    self.arrowsActor = vtk.vtkActor()

    self.movingArrowActor = vtk.vtkActor()

    self.movingContourActor = vtk.vtkActor2D()
    #self.movingContourActor = vtk.vtkImageActor()
    
    self.lastHoveredGradMag = 0

    preferredDeviceType = "GPU"

    # Create cl context and queue
    self.clcontext = None
    for platform in cl.get_platforms():
      for device in platform.get_devices():
        if cl.device_type.to_string(device.type) == preferredDeviceType:
          self.clcontext = cl.Context([device])
          print ("using: %s" % cl.device_type.to_string(device.type))
          break;
    if self.clcontext is None:
      self.clcontext = cl.create_some_context()

    self.clqueue = cl.CommandQueue(self.clcontext)

    # Compile OpenCL code and create program object
    inPath = os.path.dirname(slicer.modules.steeredfluidreg.path) + "/ImageFunctions.cl"

    fp = open(inPath)
    source = fp.read()
    fp.close()

    self.clprogram = cl.Program(self.clcontext, source).build()

    
  def __del__(self):
  
    # TODO
    # Clean up, delete line in render window
    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in xrange(sliceNodeCount):
      # find the widget for each node in scene
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      
      if sliceWidget:
        renwin = sliceWidget.sliceView().renderWindow()
        rencol = renwin.GetRenderers()
        if rencol.GetNumberOfItems() == 2:
          rencol.GetItemAsObject(1).RemoveActor(self.arrowsActor)

  def reorientVolumeToAxial(self, volume):
    nodeName = slicer.mrmlScene.GetUniqueNameByString(volume.GetName())

    vl = slicer.modules.volumes.logic()
    axialVolume = vl.CloneVolume(slicer.mrmlScene, volume, nodeName)
    axialVolume.SetName(nodeName)

    slicer.mrmlScene.AddNode(axialVolume)

    cliparams = {}
    cliparams["orientation"] = "Axial"
    cliparams["inputVolume1"] = volume.GetID()
    cliparams["outputVolume"] = axialVolume.GetID()

    print "Axial reorientation of " + volume.GetID() + " to " + axialVolume.GetID()

    slicer.cli.run(slicer.modules.orientscalarvolume, None,
      cliparams, wait_for_completion=True)

    return axialVolume

  def useFixedVolume(self, volume):

    # TODO store original orientation?
    axialVolume = self.reorientVolumeToAxial(volume)
    self.axialFixedVolume = axialVolume

    widget = slicer.modules.SteeredFluidRegWidget

    self.fixedImageCL = ImageCL(self.clqueue, self.clprogram)
    self.fixedImageCL.fromVolume(axialVolume)
    self.fixedImageCL.normalize()

    fixedShape_down = list(self.fixedImageCL.shape)
    for dim in xrange(3):
      #fixedShape_down[dim] = fixedShape_down[dim] / 2
      fixedShape_down[dim] = \
        min(fixedShape_down[dim], widget.gridSpinBoxes[dim].value)
    self.fixedImageCL_down = self.fixedImageCL.resample(fixedShape_down)

    self.ratios_down = [1.0, 1.0, 1.0]
    for dim in xrange(3):
      self.ratios_down[dim] = self.fixedImageCL.spacing[dim] / self.fixedImageCL_down.spacing[dim]

    if self.debugMessages:
      print "Using deformation grid " + str(fixedShape_down)

  def useMovingVolume(self, volume):

    # TODO store original orientation?
    axialVolume = self.reorientVolumeToAxial(volume)
    self.axialMovingVolume = axialVolume

    self.movingImageCL = ImageCL(self.clqueue, self.clprogram)
    self.movingImageCL.fromVolume(axialVolume)
    self.movingImageCL.normalize()

  def initOutputVolume(self, outputVolume):
    # NOTE: Reuse old result?
    # TODO: need to store old deformation for this to work, for now reset everything
    widget = slicer.modules.SteeredFluidRegWidget

    if outputVolume is None:
      vl = slicer.modules.volumes.logic()
      # Use reoriented moving volume
      outputVolume = vl.CloneVolume(slicer.mrmlScene, self.axialMovingVolume, "steered-warped")
      widget.outputSelector.setCurrentNode(outputVolume)
    else:
      outputImage = vtk.vtkImageData()
      outputImage.DeepCopy(self.axialMovingVolume.GetImageData())
      outputVolume.SetAndObserveImageData(outputImage)
      #TODO reuse deformation

    self.outputImageCL = ImageCL(self.clqueue, self.clprogram)
    self.outputImageCL.fromVolume(outputVolume)
    self.outputImageCL.normalize()
        
    # Force update of gradient magnitude image
    self.updateOutputVolume( self.outputImageCL )

  def updateOutputVolume(self, imgcl):

    widget = slicer.modules.SteeredFluidRegWidget

    outputVolume = widget.outputSelector.currentNode()  

    vtkimage = imgcl.toVTKImage()
  
    #castf = vtk.vtkImageCast()
    #castf.SetOutputScalarTypeToFloat()
    #castf.SetInput(vtkimage)
    #castf.Update()
  
    #outputVolume.GetImageData().GetPointData().SetScalars( castf.GetOutput().GetPointData().GetScalars() )

    outputVolume.GetImageData().GetPointData().SetScalars( vtkimage.GetPointData().GetScalars() )
    outputVolume.GetImageData().GetPointData().GetScalars().Modified()
    outputVolume.GetImageData().Modified()
    outputVolume.Modified()
    
    gradimgcl = imgcl.gradient_magnitude()
    gradimgcl.normalize()

    vtkgradimage = gradimgcl.toVTKImage()

    self.outputGradientMag = vtkgradimage

    # NOTE: may need vtk deep copy
    #self.outputGradientMag = vtk.vtkImageData()
    #self.outputGradientMag.DeepCopy(vtkgradimage)

    #self.outputGradientMagMax = self.outputGradientMag.GetScalarRange()[1]

  def startSteeredRegistration(self):

    widget= slicer.modules.SteeredFluidRegWidget

    fixedVolume = widget.fixedSelector.currentNode()  
    movingVolume = widget.movingSelector.currentNode()  
    outputVolume = widget.outputSelector.currentNode()  

    #print(self.identity)            
    self.removeObservers()
    # get new slice nodes
    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in xrange(sliceNodeCount):
      # find the widget for each node in scene
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      
      if sliceWidget:
          
        # add observers and keep track of tags
        style = sliceWidget.sliceView().interactorStyle()
        self.interactor = style.GetInteractor()
        self.sliceWidgetsPerStyle[self.interactor] = sliceWidget
        self.nodeIndexPerStyle[self.interactor] = nodeIndex
        self.sliceNodePerStyle[self.interactor] = sliceNode
        
        events = ( "LeftButtonPressEvent","LeftButtonReleaseEvent","MouseMoveEvent", "KeyPressEvent","EnterEvent", "LeaveEvent" )
        for event in events:
          tag = self.interactor.AddObserver(event, self.processEvent, 1.0)
          self.interactorObserverTags.append(tag)
      
    compositeNodeDict = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')

    sortedKeys = sorted(compositeNodeDict.iterkeys())

    for i in xrange(3):
      compositeNode = compositeNodeDict[sortedKeys[i]]
      compositeNode.SetBackgroundVolumeID(fixedVolume.GetID())
      compositeNode.SetForegroundVolumeID(outputVolume.GetID())
      compositeNode.SetForegroundOpacity(0.0)
      compositeNode.LinkedControlOn()
    for i in xrange(3,6):
      compositeNode = compositeNodeDict[sortedKeys[i]]
      compositeNode.SetBackgroundVolumeID(fixedVolume.GetID())
      compositeNode.SetForegroundVolumeID(outputVolume.GetID())
      compositeNode.SetForegroundOpacity(1.0)
      compositeNode.LinkedControlOn()

    #ii = 0
    #for compositeNode in compositeNodes.values():
    #  print "Composite node " + compositeNode.GetName()
    #  #compositeNode.SetLabelVolumeID(self.gridVolume.GetID())
    #  compositeNode.SetBackgroundVolumeID(fixedVolume.GetID())
    #  compositeNode.SetForegroundVolumeID(outputVolume.GetID())
    #  
    #  compositeNode.SetForegroundOpacity(0.5)
    #  if ii == 1:
    #    compositeNode.SetForegroundOpacity(0.0)
    #  if ii == 0:
    #    compositeNode.SetForegroundOpacity(1.0)
    #  compositeNode.SetLabelOpacity(0.5)
    #  
    #  ii += 1

    #compositeNodes.values()[0].LinkedControlOn()
    yellowWidget = layoutManager.sliceWidget('Yellow')
      
    # Set all view orientation to axial
    #sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    #for nodeIndex in xrange(sliceNodeCount):
    #  sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
    #  sliceNode.SetOrientationToAxial()
      
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.FitSliceToAll()
    
    self.outputImageCL_down = self.outputImageCL.resample(
      self.fixedImageCL_down.shape)

    self.identityCL_down = DeformationCL(self.fixedImageCL_down)
    self.deformationCL_down = self.identityCL_down

    self.identityCL = DeformationCL(self.fixedImageCL)
    self.deformationCL = self.identityCL
    
    self.fluidDelta = 0.0
    
    self.registrationIterationNumber = 0;
    qt.QTimer.singleShot(self.interval, self.updateStep)       
          
  def stopSteeredRegistration(self):
    slicer.mrmlScene.RemoveNode(self.axialFixedVolume)
    slicer.mrmlScene.RemoveNode(self.axialMovingVolume)

    self.actionState = "idle"
    self.removeObservers()

  def updateStep(self):
  
    self.registrationIterationNumber = self.registrationIterationNumber + 1
    #print('Registration iteration %d' %(self.registrationIterationNumber))
    
    # TODO: break down into computeImageMomenta, addUserControl, deformImages?
    self.fluidUpdate()
   
    # Only upsample and redraw updated image every N iterations
    if self.registrationIterationNumber % self.drawIterations == 0:
      self.deformationCL = self.deformationCL_down.compose(self.identityCL)
      #self.deformationCL = self.deformationCL_down.resample(
      #  self.fixedImageCL.shape)
      self.outputImageCL = self.deformationCL.applyTo(self.movingImageCL)

      #TODO: display downsampled output image instead?

      self.updateOutputVolume(self.outputImageCL)
      self.redrawSlices()

    # Initiate another iteration of the registration algorithm.
    if self.interaction:
      qt.QTimer.singleShot(self.interval, self.updateStep)
      
  def fluidUpdate(self):
    
    # Gradient descent: grad of output image * (fixed - output)
    diffImageCL_down = self.fixedImageCL_down.subtract(self.outputImageCL_down)

    gradientsCL_down = self.outputImageCL_down.gradient()

    momentasCL_down = [None, None, None]
    for dim in xrange(3):
      momentasCL_down[dim] = gradientsCL_down[dim].multiply(diffImageCL_down)

    isArrowUsed = False
    
    # Add user inputs to momentum vectors
    # User defined impulses are in arrow queue containing xy, RAS, slice widget
    if not self.arrowQueue.empty():
#TODO convert arrow queue to cl matrix
# write GPU kernel to insert them into mom images
# need rastoijk matrix

      # TODO use axial reoriented fixed volume, skip using RAS matrix?
      widget= slicer.modules.SteeredFluidRegWidget
              
      #fixedVolume = widget.fixedSelector.currentNode()
      #movingVolume = widget.movingSelector.currentNode()
      fixedVolume = self.axialFixedVolume
      movingVolume = self.axialMovingVolume

      imageSize = fixedVolume.GetImageData().GetDimensions()

      shape = momentasCL_down[0].shape

      origin = movingVolume.GetOrigin()

      isArrowUsed = True

      # Only do 10 arrows per iteration, TODO: allow user adjustment
      numArrowsToProcess = min(self.arrowQueue.qsize(), 10)

      if self.debugMessages:
        print "Folding in %d arrows" % numArrowsToProcess

      # for mapping drawn force to image grid
      # TODO use reoriented volume with identity matrix?, skip using RAS matrix?
      # issue with VTK negative coord in x,y ?
      movingRAStoIJK = vtk.vtkMatrix4x4()
      movingVolume.GetRASToIJKMatrix(movingRAStoIJK)
    
      for count in xrange(numArrowsToProcess):
        arrowTuple = self.arrowQueue.get()
      
        #print "arrowTuple = " + str(arrowTuple)
      
        Mtime = arrowTuple[0]
        sliceWidget = arrowTuple[1]
        startXY = arrowTuple[2]
        endXY = arrowTuple[3]
        startRAS = arrowTuple[4]
        endRAS = arrowTuple[5]

        #TODO add arrow display to proper widget/view (in ???)
      
        #startXYZ = sliceWidget.sliceView().convertDeviceToXYZ(startXY)
        #startRAS = sliceWidget.sliceView().convertXYZToRAS(startXYZ)
      
        #endXYZ = sliceWidget.sliceView().convertDeviceToXYZ(endXY)
        #endRAS = sliceWidget.sliceView().convertXYZToRAS(endXYZ)
  
        startIJK = movingRAStoIJK.MultiplyPoint(startRAS + (1,))
        endIJK = movingRAStoIJK.MultiplyPoint(endRAS + (1,))
        #startIJK = startRAS
        #endIJK = endRAS

        # Scale according to downsampling ratio
        startIJK = list(startIJK)
        endIJK = list(endIJK)
        for dim in xrange(3):
          startIJK[dim] *= self.ratios_down[dim]
          endIJK[dim] *= self.ratios_down[dim]
          #startIJK[dim] -= origin[dim]
          #endIJK[dim] -= origin[dim]
          #startIJK[dim] /= self.fixedImageCL_down.spacing[dim]
          #endIJK[dim] /= self.fixedImageCL_down.spacing[dim]
                
        forceVector = [0, 0, 0]
        forceMag = 0.0

        pos = [0, 0, 0]
      
        for dim in xrange(3):
          # TODO automatically determine magnitude from the gradient update (balanced?)
          # TODO need orientation adjustment when using RAS
          #forceVector[dim] = (endRAS[dim] - startRAS[dim])

          if self.steerMode == "pull":
            forceVector[dim] = (startIJK[dim] - endIJK[dim])
            pos[dim] = int( round(startIJK[dim]) )
          elif self.steerMode == "expand":
            forceVector[dim] = (startIJK[dim] - endIJK[dim])
            pos[dim] = int( round(endIJK[dim]) )
          elif self.steerMode == "shrink":
            forceVector[dim] = (startIJK[dim] - endIJK[dim])
            pos[dim] = int( round(startIJK[dim]) )

          forceMag += forceVector[dim] ** 2
        
        forceMag = math.sqrt(forceMag)

        # CL array index is reverse of VTK image index
        forceVector.reverse()
        pos.reverse()

        for dim in xrange(3):
          if pos[dim] < 0:
            pos[dim] = 0
          if pos[dim] >= shape[dim]:
            pos[dim] = shape[dim]-1

        if self.debugMessages:
          print "pos = " + str(pos)
          print "forceVector = " + str(forceVector)

        # TODO: do CL array access and updates in groups? more efficient? create index matrix and then do subset ops?
            
        # Find vector along grad that projects to the force vector described on the plane
        if self.steerMode == "pull":
        # TODO: do we do this when doing expand/shrink?
          gvec = [0.0, 0.0, 0.0]

          gmag = 0.0
          for dim in xrange(3):
            grad_array = gradientsCL_down[dim].clarray
            g = grad_array[ pos[0], pos[1], pos[2] ]
            gvec[dim] = g.get()[()] # Convert to scalar
            gmag += gvec[dim] ** 2
          gmag = math.sqrt(gmag)
          if gmag == 0.0:
            continue
            
          gdotf = 0
          for dim in xrange(3):
            gvec[dim] = gvec[dim] / gmag
            gdotf += gvec[dim] * forceVector[dim]
          if gdotf == 0.0:
            continue

          for dim in xrange(3):
            forceVector[dim] = gvec[dim] * forceMag**2.0 / gdotf

        for dim in xrange(3):
          mom_array = momentasCL_down[dim].clarray
          mom_array[ pos[0], pos[1], pos[2] ] += \
            forceVector[dim]

    velocitiesCL_down = [None, None, None]
    for dim in xrange(3):
      velocitiesCL_down[dim] = \
        momentasCL_down[dim].recursive_gaussian(self.fluidKernelWidth)
      
    # Compute max velocity
    velocMagCL = velocitiesCL_down[0].multiply(velocitiesCL_down[0])
    for dim in xrange(1,3):
      velocMagCL.add_inplace(
        velocitiesCL_down[dim].multiply(velocitiesCL_down[dim]) )
      
    maxVeloc = velocMagCL.max()

    if self.debugMessages:
      print "maxVeloc = %f" % maxVeloc
    
    if maxVeloc <= 0.0:
      return
      
    maxVeloc = math.sqrt(maxVeloc)

    #if self.fluidDelta == 0.0 or (self.fluidDelta*maxVeloc) > 2.0:
    if isArrowUsed or self.fluidDelta == 0.0 or (self.fluidDelta*maxVeloc) > 2.0:
      self.fluidDelta = 2.0 / maxVeloc

    for dim in xrange(3):
      velocitiesCL_down[dim].scale(self.fluidDelta)

    # Reset delta for next iteration if we used a user-defined impulse
    # TODO: control
    if isArrowUsed:
      self.fluidDelta = 0.0

    smallDeformationCL_down = self.identityCL_down.clone()
    smallDeformationCL_down.add_velocity(velocitiesCL_down)

    self.deformationCL_down = self.deformationCL_down.compose(
      smallDeformationCL_down)

    self.outputImageCL_down = self.deformationCL_down.applyTo(
      self.movingImageCL)

  def processEvent(self,observee,event=None):

    eventProcessed = False
  
    from slicer import app

    layoutManager = slicer.app.layoutManager()

    if self.sliceWidgetsPerStyle.has_key(observee):

      eventProcessed = True

      sliceWidget = self.sliceWidgetsPerStyle[observee]
      style = sliceWidget.sliceView().interactorStyle()
      self.interactor = style.GetInteractor()
      nodeIndex = self.nodeIndexPerStyle[observee]
      sliceNode = self.sliceNodePerStyle[observee]

      windowSize = sliceNode.GetDimensions()
      windowW = float(windowSize[0])
      windowH = float(windowSize[1])

      aspectRatio = windowH/windowW

      self.lastDrawnSliceWidget = sliceWidget
  
      if event == "EnterEvent":
        # TODO check interaction mode (eg tugging vs squishing)
        cursor = qt.QCursor(qt.Qt.OpenHandCursor)
        app.setOverrideCursor(cursor)
        
        self.actionState = "interacting"
        
        self.abortEvent(event)

      elif event == "LeaveEvent":
        
        cursor = qt.QCursor(qt.Qt.ArrowCursor)
        app.setOverrideCursor(cursor)
        #app.restoreOverrideCursor()
        
        self.actionState = "idle"
        
        self.abortEvent(event)
      
      elif event == "LeftButtonPressEvent":
      
        if self.steerMode == "pull" and nodeIndex > 2 and self.lastHoveredGradMag > 0.01:
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)
        
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.startEventPosition = ras
        
          self.actionState = "pullStart"
        
          self.arrowStartXY = xy
          self.arrowStartRAS = ras

          # Create a patch containing moving image information
          contourimg = vtk.vtkImageData()
          contourimg.SetDimensions(50,50,1) # TODO automatically determine from window sizes
          contourimg.SetSpacing(1.0 / windowW, 1.0 / windowW, 1.0)
          #contourimg.SetSpacing(1.0 / (100*windowW), 1.0 / (100*windowW), 1.0)
          contourimg.SetNumberOfScalarComponents(4)
          contourimg.SetScalarTypeToFloat()
          contourimg.AllocateScalars()

          contourimg.GetPointData().GetScalars().FillComponent(1, 0.0)
          contourimg.GetPointData().GetScalars().FillComponent(2, 0.0)
          #contourimg.GetPointData().GetScalars().FillComponent(3, 0.5)
          contourimg.GetPointData().GetScalars().FillComponent(3, self.opacity)

          w = slicer.modules.SteeredFluidRegWidget
          
          movingRAStoIJK = vtk.vtkMatrix4x4()
          w.movingSelector.currentNode().GetRASToIJKMatrix(movingRAStoIJK)

          outputImage = w.outputSelector.currentNode().GetImageData()

          for xshift in xrange(50):
            for yshift in xrange(50):
              xy_p = (round(xy[0] +  xshift-25), round(xy[1] + yshift-25))
              xyz_p = sliceWidget.sliceView().convertDeviceToXYZ(xy_p)
              ras_p = sliceWidget.sliceView().convertXYZToRAS(xyz_p)
          
              ijk_p = movingRAStoIJK.MultiplyPoint(ras_p + (1,))

              #TODO verify coord is inside buffer
              #g = self.outputGradientMag.GetScalarComponentAsDouble(round(ijk_p[0]), round(ijk_p[1]), round(ijk_p[2]), 0)
              g = outputImage.GetScalarComponentAsDouble(round(ijk_p[0]), round(ijk_p[1]), round(ijk_p[2]), 0)
              contourimg.SetScalarComponentFromDouble(xshift, yshift, 0, 0, g)

          imagemapper = vtk.vtkImageMapper()
          imagemapper.SetInput(contourimg)
          imagemapper.SetColorLevel(0.5)
          imagemapper.SetColorWindow(0.5)

          self.movingContourActor.SetMapper(imagemapper)
          self.movingContourActor.SetPosition(xy[0]-25, xy[1]-25)

          #self.movingContourActor.SetInput(contourimg)
          #self.movingContourActor.SetOpacity(0.5)
          #self.movingContourActor.GetProperty().SetOpacity(0.5)

          if self.debugMessages:
            print "Init contour pos = " + str(self.movingContourActor.GetPosition())

          # Add image to slice view in row above
          otherSliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex-3, 'vtkMRMLSliceNode')
          otherSliceWidget = layoutManager.sliceWidget(otherSliceNode.GetLayoutName())
          otherSliceView = otherSliceWidget.sliceView()
          otherSliceStyle = otherSliceWidget.interactorStyle()
          otherSliceStyle.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor2D(self.movingContourActor)

        elif self.steerMode == "expand" and nodeIndex > 2:

          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)
        
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.startEventPosition = ras
        
          self.actionState = "expandStart"

          self.expandStartTime = time.clock()
        
          self.arrowStartXY = xy
          self.arrowStartRAS = ras

        elif self.steerMode == "shrink" and nodeIndex > 2:

          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)
        
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.startEventPosition = ras
        
          self.actionState = "shrinkStart"

          self.arrowStartXY = xy
          self.arrowStartRAS = ras
      
        else:
          self.actionState = "clickReject"

        self.abortEvent(event)

      elif event == "LeftButtonReleaseEvent":
      
        if self.actionState == "pullStart":
          cursor = qt.QCursor(qt.Qt.OpenHandCursor)
          app.setOverrideCursor(cursor)

          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.lastEventPosition = ras

          self.arrowEndXY = xy
          self.arrowEndRAS = ras

          # TODO: only draw arrow within ??? seconds
          self.lastDrawMTime = sliceNode.GetMTime()
          
          self.lastDrawSliceWidget = sliceWidget

          self.arrowQueue.put(
            (sliceNode.GetMTime(), sliceWidget, self.arrowStartXY, self.arrowEndXY, self.arrowStartRAS, self.arrowEndRAS) )

          style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.movingArrowActor)

          otherSliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex-3, 'vtkMRMLSliceNode')
          otherSliceWidget = layoutManager.sliceWidget(otherSliceNode.GetLayoutName())
          otherSliceView = otherSliceWidget.sliceView()
          otherSliceStyle = otherSliceWidget.interactorStyle()
          otherSliceStyle.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor2D(self.movingContourActor)

        if self.actionState == "expandStart":
          cursor = qt.QCursor(qt.Qt.OpenHandCursor)
          app.setOverrideCursor(cursor)

          # TODO: only draw arrow within ??? seconds
          self.lastDrawMTime = sliceNode.GetMTime()
          
          self.lastDrawSliceWidget = sliceWidget

          a = self.arrowStartXY
          for i in xrange(len(self.expandShrinkVectors)):
            v = self.expandShrinkVectors[i]
            xy = (a[0]+v[0], a[1]+v[1], 0.0)

            xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
            ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

            arrowEndXY = xy
            arrowEndRAS = ras

            self.arrowQueue.put(
              (sliceNode.GetMTime(), sliceWidget, self.arrowStartXY, arrowEndXY, self.arrowStartRAS, arrowEndRAS) )

          style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.movingArrowActor)

        if self.actionState == "shrinkStart":
          cursor = qt.QCursor(qt.Qt.OpenHandCursor)
          app.setOverrideCursor(cursor)

          # TODO: only draw arrow within ??? seconds
          self.lastDrawMTime = sliceNode.GetMTime()
          
          self.lastDrawSliceWidget = sliceWidget

          a = self.arrowStartXY
          for i in xrange(len(self.expandShrinkVectors)):
            v = self.expandShrinkVectors[i]
            xy = (a[0]+v[0], a[1]+v[1], 0.0)

            xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
            ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

            arrowEndXY = xy
            arrowEndRAS = ras

            self.arrowQueue.put(
              (sliceNode.GetMTime(), sliceWidget, arrowEndXY, self.arrowStartXY, arrowEndRAS, self.arrowStartRAS) )

          style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.movingArrowActor)

        self.actionState = "interacting"
        
        self.abortEvent(event)

      elif event == "MouseMoveEvent":

        if self.actionState == "interacting" and self.steerMode == "pull":

          # Hovering when pulling -> change cursors based on gradient
          
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)
          
          w = slicer.modules.SteeredFluidRegWidget
          
          movingRAStoIJK = vtk.vtkMatrix4x4()
          w.movingSelector.currentNode().GetRASToIJKMatrix(movingRAStoIJK)
     
          ijk = movingRAStoIJK.MultiplyPoint(ras + (1,))
          
          g = self.outputGradientMag.GetScalarComponentAsDouble(round(ijk[0]), round(ijk[1]), round(ijk[2]), 0)
          if nodeIndex > 2 and (g > 0.05):
          #if nodeIndex > 2 and (g > 0.05*self.outputGradientMagMax):
            cursor = qt.QCursor(qt.Qt.OpenHandCursor)
            app.setOverrideCursor(cursor)
          else:
            cursor = qt.QCursor(qt.Qt.ForbiddenCursor)
            app.setOverrideCursor(cursor)
            
          self.lastHoveredGradMag = g

        elif self.actionState == "pullStart":

          # Drawing an arrow that pulls/drags voxels
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)

          xy = style.GetInteractor().GetEventPosition()

          coord = vtk.vtkCoordinate()
          coord.SetCoordinateSystemToDisplay()

          coord.SetValue(self.arrowStartXY[0], self.arrowStartXY[1], 0.0)
          worldStartXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          coord.SetValue(xy[0], xy[1], 0.0)
          worldXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          # TODO refactor code, one method for drawing collection of arrows, one actor for all arrows (moving + static ones)?

          pts = vtk.vtkPoints()
          pts.InsertNextPoint(worldStartXY)

          vectors = vtk.vtkDoubleArray()
          vectors.SetNumberOfComponents(3)
          vectors.SetNumberOfTuples(1)
          vectors.SetTuple3(0, worldXY[0]-worldStartXY[0], worldXY[1]-worldStartXY[1], worldXY[2]-worldStartXY[2]) 

          pd = vtk.vtkPolyData()
          pd.SetPoints(pts)
          pd.GetPointData().SetVectors(vectors)

          arrowSource = vtk.vtkArrowSource()

          glyphArrow = vtk.vtkGlyph3D()
          glyphArrow.SetInput(pd)
          glyphArrow.SetSource(arrowSource.GetOutput())
          glyphArrow.ScalingOn()
          glyphArrow.OrientOn()
          glyphArrow.SetScaleFactor(1.0)
          glyphArrow.SetVectorModeToUseVector()
          glyphArrow.SetScaleModeToScaleByVector()
          glyphArrow.Update()
      
          mapper = vtk.vtkPolyDataMapper()
          mapper.SetInput(glyphArrow.GetOutput())
      
          self.movingArrowActor.SetMapper(mapper)
#TODO
          #self.movingArrowActor.GetProperty().SetColor([0.1, 0.1, 0.9])
          #self.movingArrowActor.GetProperty().SetOpacity(0.5)

          style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.movingArrowActor)

          #self.movingContourActor.SetPosition(worldXY)
          self.movingContourActor.SetPosition(xy[0]-25, xy[1]-25)

        elif self.actionState == "expandStart":

          # Drawing arrows for local expansion
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)

          xy = style.GetInteractor().GetEventPosition()

          coord = vtk.vtkCoordinate()
          coord.SetCoordinateSystemToDisplay()

          coord.SetValue(self.arrowStartXY[0], self.arrowStartXY[1], 0.0)
          worldStartXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          coord.SetValue(xy[0], xy[1], 0.0)
          worldXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          # TODO: click and hold mode? requires a timer for detection
          #arrowSize = (time.clock() - self.expandStartTime) * 0.1
          arrowSize = 0.0
          for dim in xrange(2):
            arrowSize += (xy[dim] - self.arrowStartXY[dim]) ** 2.0
          arrowSize = math.sqrt(arrowSize)
          dArrowSize = 0.7 * arrowSize

          worldArrowSize = 0.0
          for dim in xrange(3):
            worldArrowSize += (worldXY[dim] - worldStartXY[dim]) ** 2.0
          worldArrowSize = math.sqrt(worldArrowSize)
          dWorldArrowSize = 0.7*worldArrowSize

          pts = vtk.vtkPoints()
          for count in xrange(8):
            pts.InsertNextPoint(worldStartXY)

          self.expandShrinkVectors = []
          self.expandShrinkVectors.append( (arrowSize, 0) )
          self.expandShrinkVectors.append( (-arrowSize, 0) )
          self.expandShrinkVectors.append( (0, arrowSize) )
          self.expandShrinkVectors.append( (0, -arrowSize) )
          self.expandShrinkVectors.append( (dArrowSize, dArrowSize) )
          self.expandShrinkVectors.append( (dArrowSize, -dArrowSize) )
          self.expandShrinkVectors.append( (-dArrowSize, dArrowSize) )
          self.expandShrinkVectors.append( (-dArrowSize, -dArrowSize) )

          worldExpandShrinkVectors = []
          worldExpandShrinkVectors.append( (worldArrowSize, 0) )
          worldExpandShrinkVectors.append( (-worldArrowSize, 0) )
          worldExpandShrinkVectors.append( (0, worldArrowSize) )
          worldExpandShrinkVectors.append( (0, -worldArrowSize) )
          worldExpandShrinkVectors.append( (dWorldArrowSize, dWorldArrowSize) )
          worldExpandShrinkVectors.append( (dWorldArrowSize, -dWorldArrowSize) )
          worldExpandShrinkVectors.append( (-dWorldArrowSize, dWorldArrowSize) )
          worldExpandShrinkVectors.append( (-dWorldArrowSize, -dWorldArrowSize) )

          vectors = vtk.vtkDoubleArray()
          vectors.SetNumberOfComponents(3)
          vectors.SetNumberOfTuples(8)
          for count in xrange(8):
            v = worldExpandShrinkVectors[count]
            vectors.SetTuple3(count, v[0], v[1], 0.0)

          pd = vtk.vtkPolyData()
          pd.SetPoints(pts)
          pd.GetPointData().SetVectors(vectors)

          arrowSource = vtk.vtkArrowSource()

          glyphArrow = vtk.vtkGlyph3D()
          glyphArrow.SetInput(pd)
          glyphArrow.SetSource(arrowSource.GetOutput())
          glyphArrow.ScalingOn()
          glyphArrow.OrientOn()
          glyphArrow.SetScaleFactor(1.0)
          glyphArrow.SetVectorModeToUseVector()
          glyphArrow.SetScaleModeToScaleByVector()
          glyphArrow.Update()
      
          mapper = vtk.vtkPolyDataMapper()
          mapper.SetInput(glyphArrow.GetOutput())
      
          self.movingArrowActor.SetMapper(mapper)
# TODO
          #self.movingArrowActor.GetProperty().SetColor([0.1, 0.1, 0.9])
          #self.movingArrowActor.GetProperty().SetOpacity(0.5)

          style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.movingArrowActor)

        elif self.actionState == "shrinkStart":

          # Drawing arrows for local shrinkage
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)

          xy = style.GetInteractor().GetEventPosition()

          coord = vtk.vtkCoordinate()
          coord.SetCoordinateSystemToDisplay()

          coord.SetValue(self.arrowStartXY[0], self.arrowStartXY[1], 0.0)
          worldStartXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          coord.SetValue(xy[0], xy[1], 0.0)
          worldXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          # TODO: click and hold mode? requires a timer for detection
          #arrowSize = (time.clock() - self.expandStartTime) * 0.1
          arrowSize = 0.0
          for dim in xrange(2):
            arrowSize += (xy[dim] - self.arrowStartXY[dim]) ** 2.0
          arrowSize = math.sqrt(arrowSize)
          dArrowSize = 0.7 * arrowSize

          worldArrowSize = 0.0
          for dim in xrange(3):
            worldArrowSize += (worldXY[dim] - worldStartXY[dim]) ** 2.0
          worldArrowSize = math.sqrt(worldArrowSize)
          dWorldArrowSize = 0.7*worldArrowSize

          self.expandShrinkVectors = []
          self.expandShrinkVectors.append( (arrowSize, 0) )
          self.expandShrinkVectors.append( (-arrowSize, 0) )
          self.expandShrinkVectors.append( (0, arrowSize) )
          self.expandShrinkVectors.append( (0, -arrowSize) )
          self.expandShrinkVectors.append( (dArrowSize, dArrowSize) )
          self.expandShrinkVectors.append( (dArrowSize, -dArrowSize) )
          self.expandShrinkVectors.append( (-dArrowSize, dArrowSize) )
          self.expandShrinkVectors.append( (-dArrowSize, -dArrowSize) )

          wa = worldStartXY

          pts = vtk.vtkPoints()
          pts.InsertNextPoint(wa[0]+worldArrowSize, wa[1], wa[2])
          pts.InsertNextPoint(wa[0]-worldArrowSize, wa[1], wa[2])
          pts.InsertNextPoint(wa[0], wa[1]+worldArrowSize, wa[2])
          pts.InsertNextPoint(wa[0], wa[1]-worldArrowSize, wa[2])
          pts.InsertNextPoint(wa[0]+dWorldArrowSize, wa[1]+dWorldArrowSize, wa[2])
          pts.InsertNextPoint(wa[0]+dWorldArrowSize, wa[1]-dWorldArrowSize, wa[2])
          pts.InsertNextPoint(wa[0]-dWorldArrowSize, wa[1]+dWorldArrowSize, wa[2])
          pts.InsertNextPoint(wa[0]-dWorldArrowSize, wa[1]-dWorldArrowSize, wa[2])

          worldExpandShrinkVectors = []
          worldExpandShrinkVectors.append( (-worldArrowSize, 0) )
          worldExpandShrinkVectors.append( (worldArrowSize, 0) )
          worldExpandShrinkVectors.append( (0, -worldArrowSize) )
          worldExpandShrinkVectors.append( (0, worldArrowSize) )
          worldExpandShrinkVectors.append( (-dWorldArrowSize, -dWorldArrowSize) )
          worldExpandShrinkVectors.append( (-dWorldArrowSize, dWorldArrowSize) )
          worldExpandShrinkVectors.append( (dWorldArrowSize, -dWorldArrowSize) )
          worldExpandShrinkVectors.append( (dWorldArrowSize, dWorldArrowSize) )

          vectors = vtk.vtkDoubleArray()
          vectors.SetNumberOfComponents(3)
          vectors.SetNumberOfTuples(8)
          for count in xrange(8):
            v = worldExpandShrinkVectors[count]
            vectors.SetTuple3(count, v[0], v[1], 0.0)

          pd = vtk.vtkPolyData()
          pd.SetPoints(pts)
          pd.GetPointData().SetVectors(vectors)

          arrowSource = vtk.vtkArrowSource()

          glyphArrow = vtk.vtkGlyph3D()
          glyphArrow.SetInput(pd)
          glyphArrow.SetSource(arrowSource.GetOutput())
          glyphArrow.ScalingOn()
          glyphArrow.OrientOn()
          glyphArrow.SetScaleFactor(1.0)
          glyphArrow.SetVectorModeToUseVector()
          glyphArrow.SetScaleModeToScaleByVector()
          glyphArrow.Update()
      
          mapper = vtk.vtkPolyDataMapper()
          mapper.SetInput(glyphArrow.GetOutput())
      
          self.movingArrowActor.SetMapper(mapper)
#TODO
          #self.movingArrowActor.GetProperty().SetColor([0.1, 0.1, 0.9])
          #self.movingArrowActor.GetProperty().SetOpacity(0.5)

          style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.movingArrowActor)
          
        else:
          pass
        
        
        self.abortEvent(event)
          
      else:
        eventProcessed = False

    if eventProcessed:
      self.redrawSlices()

  def removeObservers(self):
    # remove observers and reset
    for tag in self.interactorObserverTags:
      self.interactor.RemoveObserver(tag)
    self.interactorObserverTags = []
    self.sliceWidgetsPerStyle = {}
    

  def abortEvent(self,event):
    """Set the AbortFlag on the vtkCommand associated
    with the event - causes other things listening to the
    interactor not to receive the events"""
    # TODO: make interactorObserverTags a map to we can
    # explicitly abort just the event we handled - it will
    # be slightly more efficient
    for tag in self.interactorObserverTags:
      cmd = self.interactor.GetCommand(tag)
      cmd.SetAbortFlag(1)

  def redrawSlices(self):
    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in xrange(sliceNodeCount):
      # find the widget for each node in scene
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      
      if sliceWidget:
        renwin = sliceWidget.sliceView().renderWindow()
        rencol = renwin.GetRenderers()
        if rencol.GetNumberOfItems() == 2:
          rencol.GetItemAsObject(1).RemoveActor(self.arrowsActor)
          
        renwin.Render()
        
    if not self.arrowQueue.empty():
    
      numArrows = self.arrowQueue.qsize()

      # TODO renwin = arrowTuple[1], need to maintain actors for each renwin?
      renwin = self.lastDrawnSliceWidget.sliceView().renderWindow()

      winsize = renwin.GetSize()
      winsize = (float(winsize[0]), float(winsize[1]))

      ren = renwin.GetRenderers().GetFirstRenderer()

      coord = vtk.vtkCoordinate()
      coord.SetCoordinateSystemToDisplay()
    
      pts = vtk.vtkPoints()
      
      vectors = vtk.vtkDoubleArray()
      vectors.SetNumberOfComponents(3)
      vectors.SetNumberOfTuples(numArrows)
      
      for i in xrange(numArrows):
        arrowTuple = self.arrowQueue.queue[i]
        sliceWidget = arrowTuple[1]
        startXY = arrowTuple[2]
        endXY = arrowTuple[3]
        startRAS = arrowTuple[4]
        endRAS = arrowTuple[5]

        coord.SetValue(startXY[0], startXY[1], 0.0)
        worldStartXY = coord.GetComputedWorldValue(ren)
        pts.InsertNextPoint(worldStartXY)

        coord.SetValue(endXY[0], endXY[1], 0.0)
        worldEndXY = coord.GetComputedWorldValue(ren)

        if self.debugMessages:
          print "startXY = " + str(startXY)
          print "worldStartXY = " + str(worldStartXY)
        
        vectors.SetTuple3(i, (worldEndXY[0] - worldStartXY[0]), (worldEndXY[1] - worldStartXY[1]), 0.0)
      
      pd = vtk.vtkPolyData()
      pd.SetPoints(pts)
      pd.GetPointData().SetVectors(vectors)
      
      arrowSource = vtk.vtkArrowSource()
      #arrowSource.SetTipLength(1.0 / winsize[0])
      #arrowSource.SetTipRadius(2.0 / winsize[0])
      #arrowSource.SetShaftRadius(1.0 / winsize[0])

      glyphArrow = vtk.vtkGlyph3D()
      glyphArrow.SetInput(pd)
      glyphArrow.SetSource(arrowSource.GetOutput())
      glyphArrow.ScalingOn()
      glyphArrow.OrientOn()
      glyphArrow.SetScaleFactor(1.0)
      glyphArrow.SetVectorModeToUseVector()
      glyphArrow.SetScaleModeToScaleByVector()
      glyphArrow.Update()
      
      mapper = vtk.vtkPolyDataMapper()
      mapper.SetInput(glyphArrow.GetOutput())
      
      self.arrowsActor.SetMapper(mapper)
      self.arrowsActor.GetProperty().SetColor([0.1, 0.1, 0.9])
      self.arrowsActor.GetProperty().SetOpacity(0.5)

      # TODO add actors to the appropriate widgets (or all?)
      # TODO make each renwin have two ren's from beginning?
      rencol = renwin.GetRenderers()
      
      if rencol.GetNumberOfItems() == 2:
        renOverlay = rencol.GetItemAsObject(1)
      else:
        renOverlay = vtk.vtkRenderer()
        renwin.SetNumberOfLayers(2)
        renwin.AddRenderer(renOverlay)

      renOverlay.AddActor(self.arrowsActor)
      renOverlay.SetInteractive(0)
      #renOverlay.SetLayer(1)
      #renOverlay.ResetCamera()

      renwin.Render()

  def testingData(self):
    """Load some default data for development
    and set up a transform and viewing scenario for it.
    """

    #import SampleData
    #sampleDataLogic = SampleData.SampleDataLogic()
    #mrHead = sampleDataLogic.downloadMRHead()
    #dtiBrain = sampleDataLogic.downloadDTIBrain()
    
    # w = slicer.modules.SteeredFluidRegWidget
    # w.fixedSelector.setCurrentNode(mrHead)
    # w.movingSelector.setCurrentNode(dtiBrain)
    
    if not slicer.util.getNodes('testbrain1*'):
      import os
      fileName = "C:\\Work\\testbrain1.nrrd"
      vl = slicer.modules.volumes.logic()
      brain1Node = vl.AddArchetypeVolume(fileName, "testbrain1", 0)
    else:
      nodes = slicer.util.getNodes('testbrain1.nrrd')
      brain1Node = nodes[0]

    if not slicer.util.getNodes('testbrain2*'):
      import os
      fileName = "C:\\Work\\testbrain2.nrrd"
      vl = slicer.modules.volumes.logic()
      brain2Node = vl.AddArchetypeVolume(fileName, "testbrain2", 0)
    #TODO else assign from list

    # if not slicer.util.getNodes('movingToFixed*'):
      # # Create transform node
      # transform = slicer.vtkMRMLLinearTransformNode()
      # transform.SetName('movingToFixed')
      # slicer.mrmlScene.AddNode(transform)

    # transform = slicer.util.getNode('movingToFixed')
    
    # ###
    # # neutral.SetAndObserveTransformNodeID(transform.GetID())
    # ###
    
    compositeNodes = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')
    for compositeNode in compositeNodes.values():
      compositeNode.SetBackgroundVolumeID(brain1Node.GetID())
      compositeNode.SetForegroundVolumeID(brain2Node.GetID())
      compositeNode.SetForegroundOpacity(0.5)
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.FitSliceToAll()


    


