
#from __main__ import vtk, qt, ctk, slicer
import vtk, qt, ctk, slicer
import math
import threading
import SimpleITK as sitk
import sitkUtils
import Queue

#
# CL classes
#

import pyopencl as cl
import pyopencl.array as cla
import numpy as n

#
# VolumeCL
#
# Wrapper for Slicer volume node, adding support for basic image operations
# using PyOpenCL
#
# Supports: convolve, grad, add, mul, sub, etc
#

class VolumeCL:
  def __init__(self, clqueue, volume, makeAxial=False):

    # TODO: only have image info (origin, spacing, orient) and VTK volume?
    # do not keep copies of volume nodes -> issues, clashing names???
    # rename to ImageCL convert from VolumeNode -> CL then can get vtkImageData

    # DEBUG PP
    #self.volumeNode = slicer.vtkMRMLScalarVolumeNode()
    #self.volumeNode.Copy(volume)
    #self.volumeNode.SetName(volume.GetName() + " clone")

    #image = vtk.vtkImageData()
    #image.DeepCopy(volume.GetImageData())
    #self.volumeNode.SetAndObserveImageData(image)

    #vl = slicer.modules.volumes.logic()
    #self.volumeNode = vl.CloneVolume(slicer.mrmlScene, volume, volume.GetName() + " clone")

    # TODO need orientation matrix->name?
    #self.originalOrientation = volume.GetOrientation()
    self.originalOrientation = "Axial"

    print "Input ID: " + str(volume.GetID())
    print "Copy ID: " + str(self.volumeNode.GetID())

    # Reorient volume to axial if requested
    if makeAxial:
      cliparams = {}
      cliparams["orientation"] = "Axial"
      cliparams["inputVolume1"] = volume.GetID()
      cliparams["outputVolume"] = self.volumeNode.GetID()

      print "Axial reorientation of " + volume.GetID() + " to " + self.volumeNode.GetID()

      slicer.cli.run(slicer.modules.orientscalarvolume, None,
	cliparams, wait_for_completion=True)

    # Cast to float
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToFloat()
    castf.SetInput(self.volumeNode.GetImageData())
    castf.Update()

    # Avoid pointer issue with VTK filter output
    img = vtk.vtkImageData()
    img.DeepCopy( castf.GetOutput() )

    self.volumeNode.SetAndObserveImageData(img)

    # CL queue created in an initialization function outside of class
    self.clqueue = clqueue

    # Create array on device from host data
    self.sync_dev()

    # Compile program
    inPath = os.path.dirname(slicer.modules.steeredfluidreg.path) + "/ImageFunctions.cl.in"
    fp = open(inPath)
    sourceIn = fp.read()
    fp.close()

    slices, rows, columns = self.shape
    source = sourceIn % {
      'slices' : slices,
      'rows' : rows,
      'columns' : columns,
      'xspacing' : 1.0,
      'yspacing' : 1.0,
      'zspacing' : 1.0,
      'kernelSize' : 1.0,
      'kernelWidth' : 1.0
    }

    self.clprogram = cl.Program(self.clqueue.context, source).build()

    # TODO: maintain modification flags for host and dev memory
    # use them to sync as needed

  def __del__(self):
    # Remove volume node
    # TODO
    pass


  def clone(self):
    copyImage = vtk.vtkImageData()
    copyImage.DeepCopy(self.volumeNode.GetImageData())

    copyVolumeNode = slicer.vtkMRMLScalarVolumeNode()
    copyVolumeNode.Copy(self.volumeNode)
    copyVolumeNode.SetName(self.volumeNode.GetName() + " clone")
    copyVolumeNode.SetAndObserveImageData(copyImage)

    return VolumeCL(self.clqueue, copyVolumeNode)

  def fill(self, value):
    self.volumeNode.GetImageData().GetPointData().GetScalars().FillComponent(
      0, value)
    self.clarray.fill(value)

  def sync_dev(self):
    self.shape = list(self.volumeNode.GetImageData().GetDimensions())
    self.shape.reverse()
    array = vtk.util.numpy_support.vtk_to_numpy(
        self.volumeNode.GetImageData().GetPointData().GetScalars()).reshape(self.shape)
    array = array.astype('float32')
    self.clarray = cl.array.to_device(self.clqueue, array)

  def sync_host(self):
     narray = self.clarray.get().astype('float32')
     #narray = n.transpose(narray, (2, 1,0)) # Already handled at input

     vtkarray = vtk.util.numpy_support.numpy_to_vtk(narray.flatten(), deep=True)

     self.volumeNode.GetImageData().GetPointData().SetScalars(vtkarray)
     self.volumeNode.GetImageData().GetPointData().GetScalars().Modified()
     self.volumeNode.GetImageData().GetPointData().Modified()
     self.volumeNode.GetImageData().Modified()
     self.volumeNode.Modified()

  def normalize(self):
     [minp, maxp] =  self.volumeNode.GetImageData().GetScalarRange()
     range = maxp - minp
     if range > 0.0:
      self.clarray = (self.clarray - minp) / range
      #self.sync_host()

  def scale(self, v):
     self.clarray = self.clarray * v
     #self.sync_host()
    
  def add(self, othervolcl):
    outvol = self.clone()
    outvol.clarray = self.clarray + othervolcl.clarray
    #outvol.sync_host()
    return outvol
    
  def subtract(self, othervolcl):
    outvol = self.clone()
    outvol.clarray = self.clarray - othervolcl.clarray
    #outvol.sync_host()
    return outvol
    
  def multiply(self, othervolcl):
    outvol = self.clone()
    outvol.clarray = self.clarray * othervolcl.clarray
    #outvol.sync_host()
    return outvol

  def max(self):
    return self.clarray.get().max()

  def getVolumeInOriginalOrientation(self):
    # Get data in original input orientation (for display purposes?)
    # NOTE: may not be necessary if Slicer reslices everything properly

    outVolumeNode = slicer.vtkMRMLScalarVolumeNode()
    outVolumeNode.Copy(self.volumeNode)

    cliparams = {}
    cliparams["orientation"] = self.originalOrientation
    cliparams["inputVolume1"] = self.volumeNode.GetID()
    cliparams["outputVolume"] = outVolumeNode.GetID()

    slicer.cli.run(slicer.modules.orientscalarvolume, None,
      cliparams, wait_for_completion=True)

    return outVolumeNode

  def gradient(self):
    gradx = self.clone()
    grady = self.clone()
    gradz = self.clone()

    self.clprogram.gradient(self.clqueue, self.shape, None, self.clarray.data,  gradx.clarray.data, grady.clarray.data, gradz.clarray.data).wait()

    # TODO: skip memory transfer until needed
    # restrict access using getVolumeNode()?
    #gradx.sync_host()
    #grady.sync_host()
    #gradz.sync_host()

    return [gradx, grady, gradz]

  def gradient_magnitude(self):
    [gx, gy, gz] = self.gradient()
    mag = gx.multiply(gx)
    mag = mag.add(gy.multiply(gy))
    mag = mag.add(gz.multiply(gz))
    return mag

  def gaussian(self, kernelwidth, kernelsize):
    outvol = self.clone()

    var = n.zeros((1,), dtype=n.float32)
    var[0] = kernelwidth * kernelwidth
    var_array = cla.to_device(self.clqueue, var)
    width = n.zeros((1,), dtype=n.int32)
    width[0] = kernelwidth
    width_array = cla.to_device(self.clqueue, width)

    self.clprogram.gaussian(self.clqueue, self.shape, None, self.clarray.data, var_array.data,  width_array.data, outvol.clarray.data).wait()

    # Keep memory in CL device and host synchronized
    #outvol.sync_host()
    
    return outvol

#
# DeformationCL
#
# Wrapper for a list of 3 VolumeCL objects containing deformation maps h
# with support for composition (h1 \circ h2), and warping of scalar volumes
#
# Warping is applied as Iwarped = I(h)
#
class DeformationCL:

  def __init__(self, samplevolcl):

    self.hx = samplevolcl.clone()
    self.hy = samplevolcl.clone()
    self.hz = samplevolcl.clone()

    self.clqueue = samplevolcl.clqueue
    self.clprogram = samplevolcl.clprogram

    self.set_identity()

  def set_mapping(self, hx, hy, hz):
    self.hx = hx
    self.hy = hy
    self.hz = hz
    self.clprogram = self.hx.clprogram

  def __del__(self):
    pass

  def set_identity(self):
    self.clprogram.identity(self.clqueue, self.hx.shape, None, self.hx.clarray.data,  self.hy.clarray.data, self.hz.clarray.data).wait()

    #self.hx.sync_host()
    #self.hy.sync_host()
    #self.hz.sync_host()

  def add_velocity(self, velocList):
    self.hx = self.hx.add(velocList[0])
    self.hy = self.hy.add(velocList[1])
    self.hz = self.hz.add(velocList[2])

  def maxMagnitude(self):
    magVol = self.hx.clone()
    magVol.fill(0)

    magVol = magVol.add( self.hx.multiply(self.hx) )
    magVol = magVol.add( self.hy.multiply(self.hy) )
    magVol = magVol.add( self.hz.multiply(self.hz) )

    return math.sqrt( magVol.max() )

  def applyTo(self, vol):
    outvol = vol.clone()
    
    self.clprogram.interpolate(self.hx.clqueue, self.hx.shape, None, vol.clarray.data, self.hx.clarray.data, self.hy.clarray.data, self.hz.clarray.data, outvol.clarray.data).wait()

    # Keep memory in CL and host synchronized
    #outvol.sync_host()
    
    return outvol

  def compose(self, otherdef):
    newhx = otherdef.applyTo(self.hx)
    newhy = otherdef.applyTo(self.hy)
    newhz = otherdef.applyTo(self.hz)

    outdef = DeformationCL(self.hx)
    outdef.set_mapping(newhx, newhy, newhz)
    return outdef

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

#
# SteeredFluidRegWidget
#

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
    
    self.momentas = []
    
    self.threadLock = threading.Lock()
    
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
    # io Collapsible button
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
    # opt Collapsible button
    #
    optCollapsibleButton = ctk.ctkCollapsibleButton()
    optCollapsibleButton.text = "Registration Parameters"
    self.layout.addWidget(optCollapsibleButton)

    # Layout within the parameter collapsible button
    optFormLayout = qt.QFormLayout(optCollapsibleButton)

    # Floating image opacity
    self.opacitySlider = ctk.ctkSliderWidget()
    self.opacitySlider.decimals = 2
    self.opacitySlider.singleStep = 0.05
    self.opacitySlider.minimum = 0.0
    self.opacitySlider.maximum = 1.0
    self.opacitySlider.toolTip = "Transparency of floating moving image ."
    optFormLayout.addRow("Floating image opacity:", self.opacitySlider)

    # iteration slider
    self.regIterationSlider = ctk.ctkSliderWidget()
    self.regIterationSlider.decimals = 2
    self.regIterationSlider.singleStep = 10
    self.regIterationSlider.minimum = 10
    self.regIterationSlider.maximum = 10000
    self.regIterationSlider.toolTip = "Number of iterations"
    optFormLayout.addRow("Iterations:", self.regIterationSlider)

    # Fluid viscosity
    self.viscositySlider = ctk.ctkSliderWidget()
    self.viscositySlider.decimals = 2
    self.viscositySlider.singleStep = 1.0
    self.viscositySlider.minimum = 0.0
    self.viscositySlider.maximum = 100.0
    self.viscositySlider.toolTip = "Area of effect for deformation forces."
    optFormLayout.addRow("Deformation stiffness:", self.viscositySlider)

    # get default values from logic
    self.regIterationSlider.value = self.logic.regIteration
    self.viscositySlider.value = self.logic.viscosity
    self.opacitySlider.value = self.logic.opacity

    #print(self.logic.regIteration)

    sliders = (self.regIterationSlider, self.viscositySlider, self.opacitySlider)
    for slider in sliders:
      slider.connect('valueChanged(double)', self.updateLogicFromGUI)


    # Start button
    self.regButton = qt.QPushButton("Start")
    self.regButton.toolTip = "Run registration."
    self.regButton.checkable = True
    self.layout.addWidget(self.regButton)
    self.regButton.connect('toggled(bool)', self.onStart)

    # Add vertical spacer
    #self.layout.addStretch(1)

    # to support quicker development:
    import os
    if (os.getenv('USERNAME') == '212357326') or (os.getenv('USER') == 'prastawa'):
    #if False:
      self.logic.testingData()
      self.fixedSelector.setCurrentNode(slicer.util.getNode('testbrain1'))
      self.movingSelector.setCurrentNode(slicer.util.getNode('testbrain2'))
      # self.transformSelector.setCurrentNode(slicer.util.getNode('movingToFixed'))
      # self.initialTransformSelector.setCurrentNode(slicer.util.getNode('movingToFixed'))


  def updateLogicFromGUI(self,args):
    self.logic.fixed = self.fixedSelector.currentNode()
    self.logic.moving = self.movingSelector.currentNode()
    # self.logic.transform = self.transformSelector.currentNode()

    # TODO: hook with grid transform, for now just update background image from GPU
    # KEY PP: allow motion, but crashes???
    # if(self.logic.transform is not None):
    #   self.logic.moving.SetAndObserveTransformNodeID(self.logic.transform.GetID())
    
    self.logic.regIteration = self.regIterationSlider.value
    self.logic.viscosity = self.viscositySlider.value
    self.logic.opacity = self.opacitySlider.value
 
          
  def onResetButtonToggled(self):
    self.logic.actionState = "reset"

    # Set momentas to zero, copy moving to output
    for dim in xrange(3):
      self.displacement[dim].GetImageData().GetPointData().GetScalars().FillComponent(0, 0.0)
    if outputVolume is not None:
      outputVolume.GetImageData().GetPointData().SetScalars(movingVolume.GetImageData().GetPointData().GetScalars())
   
  def onStart(self,checked):
    
    if checked:
      self.regButton.text = "Stop"

      layoutManager = slicer.app.layoutManager()
      layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutThreeOverThreeView)
      
      # Set up display of volumes as composites, and create output node if not specified
      fixedVolume = self.fixedSelector.currentNode()
      movingVolume = self.movingSelector.currentNode()
      outputVolume = self.outputSelector.currentNode()

      self.fixedVolumeCL = VolumeCL(self.logic.clQueue, fixedVolume, True)
      self.fixedVolumeCL.normalize()

      self.movingVolumeCL = VolumeCL(self.logic.clQueue, movingVolume, True)
      self.movingVolumeCL.normalize()

      #cool1 = slicer.vtkMRMLColorTableNode()
      #cool1.SetTypeToCool1()
      #fixedVolume.GetScene().AddNode(cool1)

      #warm1 = slicer.vtkMRMLColorTableNode()
      #warm1.SetTypeToWarm1()
      #movingVolume.GetScene().AddNode(warm1)

      fixedDisplay = fixedVolume.GetDisplayNode()
      #fixedDisplay.SetAndObserveColorNodeID(cool1.GetID())

      movingDisplay = movingVolume.GetDisplayNode()
      #movingDisplay.SetAndObserveColorNodeID(warm1.GetID())
      
      # NOTE: Reuse old result?
      # TODO: need to store old deformation for this to work, for now reset everything
      # if outputVolume is None:
        # vl = slicer.modules.volumes.logic()
        # outputVolume = vl.CloneVolume(slicer.mrmlScene, movingVolume, "steered-warped")
        # self.outputSelector.setCurrentNode(outputVolume)
      # else:
        # # Disabled to allow pausing and unpausing
        # # TODO DEBUG
        # #movingArray = slicer.util.array(movingVolume.GetName())
        # #outputArray = slicer.util.array(outputVolume.GetName())
        # #outputArray[:] = movingArray[:]
        # pass
      
      if outputVolume is None:
        vl = slicer.modules.volumes.logic()
        outputVolume = vl.CloneVolume(slicer.mrmlScene, movingVolume, "steered-warped")
        self.outputSelector.setCurrentNode(outputVolume)
      else:
	outputImage = vtk.vtkImageData()
	outputImage.DeepCopy(movingVolume.GetImageData())
        #outputImage.GetPointData().SetScalars(movingVolume.GetImageData().GetPointData().GetScalars())
        outputVolume.SetAndObserveImageData(outputImage)

      self.outputVolumeCL = VolumeCL(self.logic.clQueue, outputVolume, True)
      self.outputVolumeCL.normalize()
        
      # Force update of gradient magnitude image
      self.updateOutputVolume( self.outputVolumeCL )

      # TODO DEBUG
      # Propagate image information to image data structures?
      #fixedVolume.GetImageData().SetOrigin( fixedVolume.GetOrigin() )
      #fixedVolume.GetImageData().SetSpacing( fixedVolume.GetSpacing() )
      #movingVolume.GetImageData().SetOrigin( movingVolume.GetOrigin() )
      #movingVolume.GetImageData().SetSpacing( movingVolume.GetSpacing() )
      #outputVolume.GetImageData().SetOrigin( outputVolume.GetOrigin() )
      #outputVolume.GetImageData().SetSpacing( outputVolume.GetSpacing() )
      
      orig = movingVolume.GetImageData().GetOrigin()
      sp = movingVolume.GetImageData().GetSpacing()
      print "Call build id with orig = " + str(orig) + " sp = " + str(sp)

      # TODO: move parts to startDeformableReg()
      
      # Zero displacement
      self.displacement = [None, None, None]
      for dim in xrange(3):
        castf = vtk.vtkImageCast()
        castf.SetOutputScalarTypeToDouble()
        castf.SetInput(movingVolume.GetImageData())
        castf.Update()

        disp = vtk.vtkImageData()
        disp.DeepCopy( castf.GetOutput() )
        disp.GetPointData().GetScalars().FillComponent(0, 0.0)
        self.displacement[dim] = disp

      
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
        
      ii = 0
      
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

      self.logic.interaction = True
      self.logic.start()

      self.logic.automaticRegistration = True
      print('Automatic registration = %d' %(self.logic.automaticRegistration))
      
      self.startDeformableRegistration()
    else:
      self.regButton.text = "Start"
      
      self.logic.interaction=False
      self.logic.stop()

      # TODO: Use a grid transform and make it observable?
      #if(self.logic.transform is not None): 
      #  self.logic.moving.SetAndObserveTransformNodeID(self.logic.transform.GetID())
      
      self.logic.automaticRegistration = False
      print('Automatic registration = %d' %(self.logic.automaticRegistration))

  def updateOutputVolume(self, volcl):

    volcl.sync_host()

    outputVolume = self.outputSelector.currentNode()  
    outputVolume.Copy(volcl.volumeNode)
  
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToFloat()
    castf.SetInput(volcl.volumeNode.GetImageData())
    castf.Update()
  
    outputVolume.GetImageData().GetPointData().SetScalars( castf.GetOutput().GetPointData().GetScalars() )
    outputVolume.GetImageData().GetPointData().GetScalars().Modified()
    outputVolume.GetImageData().Modified()
    outputVolume.Modified()
    
    gradvolcl = volcl.gradient_magnitude()
    gradvolcl.normalize()
    gradvolcl.sync_host()
    # NOTE: may need vtk deep copy
    self.outputGradientMag = gradvolcl.volumeNode.GetImageData()

#TODO DEBUG for some reason grad mag takes over moving vol????

    
  def startDeformableRegistration(self):     
    fixedVolume = self.fixedSelector.currentNode()
    movingVolume = self.movingSelector.currentNode()
    outputVolume = self.outputSelector.currentNode()
    # initialTransform = self.initialTransformSelector.currentNode()
    # outputTransform = self.transformSelector.currentNode()

    self.deformationCL = DeformationCL(self.outputVolumeCL)
 
    vl = slicer.modules.volumes.logic()
    
    #TODO: clone output to moving

    self.parameters = {}
    # self.parameters['InitialTransform'] = initialTransform.GetID()
    self.parameters['FixedImageFileName'] = fixedVolume.GetID()
    self.parameters['MovingImageFileName'] = movingVolume.GetID()
    # self.parameters['OutputTransform'] = outputTransform.GetID()
    #self.parameters['ResampledImageFileName'] = outputVolume.GetID()

    self.parameters['Iterations']=self.regIterationSlider.value

    # TODO put fluid stuff in logic?
    
    self.fluidDelta = 0.0

    print('registration begin')
    print "return result every %d iterations" %(self.regIterationSlider.value)
    
    self.interval = 1000
    self.registrationIterationNumber = 0;
    qt.QTimer.singleShot(self.interval, self.updateStep)       

  def updateStep(self):
  
    #TODO can updates clash?
    self.threadLock.acquire()
    
    self.registrationIterationNumber = self.registrationIterationNumber + 1
    #print('Registering iteration %d' %(self.registrationIterationNumber))
    
    self.fluidUpdate()
   
    self.logic.redrawSlices()

    # Initiate another iteration of the registration algorithm.
    if self.logic.interaction:
      qt.QTimer.singleShot(self.interval, self.updateStep)
      
    self.threadLock.release()

  def fluidUpdate(self):
    #print('fluid update')
    
    # Gradient descent: grad of output image * (fixed - output)
    diffVolumeCL = self.fixedVolumeCL.subtract(self.outputVolumeCL)

    gradientsCL = self.outputVolumeCL.gradient()

    momentasCL = [None, None, None]
    momentaImages = [None, None, None]
    for dim in xrange(3):
      momentasCL[dim] = gradientsCL[dim].multiply(diffVolumeCL)
      momentaImages[dim] = momentasCL[dim].volumeNode.GetImageData()
      #TODO: momentaImages[dim] = self.momentasCL[dim].GetVolumeNode().GetImageData()
      momentasCL[dim].sync_host()

    isArrowUsed = False
    
    # Add user inputs to momentum vectors
    # User defined impulses are in arrow queue containing xy, RAS, slice widget
    if not self.logic.arrowQueue.empty():

      fixedVolume = self.fixedSelector.currentNode()

      imageSize = fixedVolume.GetImageData().GetDimensions()

      isArrowUsed = True
    
      arrowTuple = self.logic.arrowQueue.get()
      
      print "arrowTuple = " + str(arrowTuple)
      
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

      # for mapping drawn force to image grid
      movingRAStoIJK = vtk.vtkMatrix4x4()
      self.movingSelector.currentNode().GetRASToIJKMatrix(movingRAStoIJK)
  
      startIJK = movingRAStoIJK.MultiplyPoint(startRAS + (1,))
      endIJK = movingRAStoIJK.MultiplyPoint(endRAS + (1,))
                
      forceVector = [0, 0, 0]
      forceCenter = [0, 0, 0]
      forceMag = 0
      
      for dim in xrange(3):
        forceCenter[dim] = round(startIJK[dim])
        # TODO automatically determine magnitude from the gradient update (balanced?)
	# TODO need orientation adjustment when using RAS
        #forceVector[dim] = (endRAS[dim] - startRAS[dim])
	# TODO need inversion for IJK ?
        #forceVector[dim] = (endIJK[dim] - startIJK[dim])
        forceVector[dim] = (startIJK[dim] - endIJK[dim])
        forceMag += forceVector[dim] ** 2
        
      forceMag = math.sqrt(forceMag)

               
      # TODO: real splatting? need an area of effect instead of point impulse
      pos = [0, 0, 0]
      for ti in xrange(-1,2):
        pos[0] = forceCenter[0] + ti
        if pos[0] < 0 or pos[0] >= imageSize[0]:
          continue
        for tj in xrange(-1,2):
          pos[1] = forceCenter[1] + tj
          if pos[1] < 0 or pos[1] >= imageSize[1]:
            continue
          for tk in xrange(-1,2):
            pos[2] = forceCenter[2] + tk
            if pos[2] < 0 or pos[2] >= imageSize[2]:
              continue
            
            # Find vector along grad that projects to the force vector described on the plane
            gvec = [0, 0, 0]
            gmag = 0
            for dim in xrange(3):
              gvec[dim] = momentaImages[dim].GetScalarComponentAsDouble(pos[0], pos[1], pos[2], 0)
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
              a = momentaImages[dim].GetScalarComponentAsDouble(pos[0], pos[1], pos[2], 0)
              momentaImages[dim].SetScalarComponentFromDouble(pos[0], pos[1], pos[2], 0,
                a + gvec[dim] * forceMag**2.0 / gdotf)
             
      # for dim in xrange(3):
        # a = self.momentas[dim].GetScalarComponentAsDouble(forceCenter[0], forceCenter[1], forceCenter[2], 0)
        # self.momentas[dim].SetScalarComponentFromDouble(forceCenter[0], forceCenter[1], forceCenter[2], 0,
          # a + forceVector[dim])

      # Sync data modified by arrows to the device
      # TODO: more efficient integration one call to sync dev and add
      # rather than sync_host, add, sync_dev
      for dim in xrange(3):
        momentasCL[dim].sync_dev()
              
    velocitiesCL = [None, None, None]
    for dim in xrange(3):
      velocitiesCL[dim] = momentasCL[dim].gaussian(1.0, 3)
      
    # Compute max velocity
    velocMagCL = velocitiesCL[0].multiply(velocitiesCL[0])
    for dim in xrange(1,3):
      velocMagCL = velocMagCL.add(
	velocitiesCL[dim].multiply(velocitiesCL[dim]) )
      
    maxVeloc = velocMagCL.max()
    print "maxVeloc squared = %f" % maxVeloc
    
    if maxVeloc <= 0.0:
      return
      
    maxVeloc = math.sqrt(maxVeloc)

    print "delta = %f" % self.fluidDelta
    
    if self.fluidDelta == 0.0 or (self.fluidDelta*maxVeloc) > 2.0:
      self.fluidDelta = 2.0 / maxVeloc
      print "new delta = %f" % self.fluidDelta

    print "maxVeloc*delta = %f" % (maxVeloc*self.fluidDelta)

    for dim in xrange(3):
      velocitiesCL[dim].scale(self.fluidDelta)

    # Reset delta for next iteration if we used an impulse
    if isArrowUsed:
      self.fluidDelta = 0.0

    smallDeformationCL = DeformationCL(self.outputVolumeCL)
    smallDeformationCL.add_velocity(velocitiesCL)

    #self.deformationCL = self.deformationCL.compose(smallDeformationCL)
    self.deformationCL = smallDeformationCL

    #tempVolumeCL = self.deformationCL.applyTo(self.movingVolumeCL)
    #self.outputVolumeCL.volumeNode.GetImageData().DeepCopy(
      #tempVolumeCL.volumeNode.GetImageData() )
    #self.updateOutputVolume(tempVolumeCL)

    self.outputVolumeCL = self.deformationCL.applyTo(self.movingVolumeCL)
#TODO: need to retag as output?
    self.updateOutputVolume(self.outputVolumeCL)
     

  def getMinMax(self, inputImage):
    histf = vtk.vtkImageHistogramStatistics()
    histf.SetInput(inputImage)
    histf.Update()
    
    imin = histf.GetMinimum()
    imax = histf.GetMaximum()
    
    return (imin, imax)
    
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
    
  def warpImage(self, image, H):
    # Returns I \circ H
    
    fixedVolume = self.fixedSelector.currentNode()
    fixedImage = fixedVolume.GetImageData()
    
    appendf = vtk.vtkImageAppendComponents()
    appendf.SetInput(H[0])
    appendf.AddInput(H[1])
    appendf.AddInput(H[2])
    appendf.Update()
    
    gridTrafo = vtk.vtkGridTransform()
    gridTrafo.SetDisplacementGrid(appendf.GetOutput())
    gridTrafo.SetDisplacementScale(1.0)
    gridTrafo.SetDisplacementShift(0.0)
    gridTrafo.SetInterpolationModeToCubic()
    gridTrafo.Update()
    
    reslice = vtk.vtkImageReslice()
    reslice.SetInput(image)
    reslice.SetResliceTransform(gridTrafo)
    reslice.SetInterpolationModeToCubic()
    reslice.SetOutputDimensionality(3)
    reslice.SetOutputOrigin(fixedImage.GetOrigin())
    reslice.SetOutputSpacing(fixedImage.GetSpacing())
    reslice.SetOutputExtent(fixedImage.GetWholeExtent())
    reslice.SetBackgroundLevel(0)
    reslice.SetNumberOfThreads(8)
    reslice.Update()
    
    #return reslice.GetOutput()
    warpedImage = vtk.vtkImageData()
    warpedImage.DeepCopy(reslice.GetOutput())
    return warpedImage
    
  def buildIdentity(self, image):
    # Returns list of vtkImageData containing the identity transform (idx, idy, idz)
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToDouble()
    castf.SetInput(image)
    castf.Update()
    
    size = image.GetDimensions()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    
    print "Build id orig = " + str(origin) + " size = " + str(size) + " spacing = " + str(spacing)
    
    # NOTE: assume axial orientation
    # TODO: reorient-flip input images when starting? How to overwrite input volume nodes?

    idList = [None, None, None]
    for dim in xrange(3):
      copyImage = vtk.vtkImageData()
      copyImage.DeepCopy(castf.GetOutput())
      copyImage.GetPointData().GetScalars().FillComponent(0, 0.0)
      idList[dim] = copyImage

    # TODO: coordinate of MRML volume or vtkImageData?
    # TODO: use IJK to RAS?

    for i in xrange(size[0]):
      x = i*spacing[0] + origin[0]
      for j in xrange(size[1]):
        y = j*spacing[1] + origin[1]
        for k in xrange(size[2]):
          z = k*spacing[2] + origin[2]
          idList[0].SetScalarComponentFromDouble(i, j, k, 0, x)
          idList[1].SetScalarComponentFromDouble(i, j, k, 0, y)
          idList[2].SetScalarComponentFromDouble(i, j, k, 0, z)
      
    return idList
    

  def composeDeformations(self, F, G):
    # Returns H = F \circ G
    # F and G are lists of deformation maps (hx, hy, hz)
    
    fixedVolume = self.fixedSelector.currentNode()
    fixedImage = fixedVolume.GetImageData()
    
    appendf = vtk.vtkImageAppendComponents()
    appendf.SetInput(G[0])
    appendf.AddInput(G[1])
    appendf.AddInput(G[2])
    appendf.Update()
    
    gridTrafo = vtk.vtkGridTransform()
    gridTrafo.SetDisplacementGrid(appendf.GetOutput())
    gridTrafo.SetDisplacementScale(1.0)
    gridTrafo.SetDisplacementShift(0.0)
    gridTrafo.SetInterpolationModeToLinear()
    gridTrafo.Update()
    
    H = [None, None, None]
    
    for dim in xrange(3):
      # H[dim] = self.warpImage(F[dim], G)
      reslice = vtk.vtkImageReslice()
      reslice.SetInput(F[dim])
      reslice.SetResliceTransform(gridTrafo)
      reslice.SetInterpolationModeToLinear()
      reslice.SetOutputDimensionality(3)
      reslice.SetOutputOrigin(fixedImage.GetOrigin())
      reslice.SetOutputSpacing(fixedImage.GetSpacing())
      reslice.SetOutputExtent(fixedImage.GetWholeExtent())
      reslice.SetNumberOfThreads(8)
      reslice.Update()
      
      #H[dim] = reslice.GetOutput()
      H[dim] = vtk.vtkImageData()
      H[dim].DeepCopy(reslice.GetOutput())

    return H

  def composeDisplacements(self, F, G):
    # Returns H = F \circ G
    # F and G are lists of displacement maps [vx, vy, vz]
    
    fixedVolume = self.fixedSelector.currentNode()
    fixedImage = fixedVolume.GetImageData()

    mapF = [None, None, None]
    for dim in xrange(3):
      addf = vtk.vtkImageMathematics()
      addf.SetOperationToAdd()
      addf.SetInput1(F[dim])
      addf.SetInput2(self.identityMap[dim])
      addf.Update()

      mapF[dim] = vtk.vtkImageData()
      mapF[dim].DeepCopy( addf.GetOutput() )

    appendf = vtk.vtkImageAppendComponents()
    appendf.SetInput(G[0])
    appendf.AddInput(G[1])
    appendf.AddInput(G[2])
    appendf.Update()
    
    gridTrafo = vtk.vtkGridTransform()
    gridTrafo.SetDisplacementGrid(appendf.GetOutput())
    gridTrafo.SetDisplacementScale(1.0)
    gridTrafo.SetDisplacementShift(0.0)
    gridTrafo.SetInterpolationModeToLinear()
    gridTrafo.Update()
    
    H = [None, None, None]
    
    for dim in xrange(3):
      # H[dim] = self.warpImage(F[dim], G)
      reslice = vtk.vtkImageReslice()
      reslice.SetInput(mapF[dim])
      reslice.SetResliceTransform(gridTrafo)
      reslice.SetInterpolationModeToLinear()
      reslice.SetOutputDimensionality(3)
      #reslice.SetOutputOrigin(fixedImage.GetOrigin())
      #reslice.SetOutputSpacing(fixedImage.GetSpacing())
      #reslice.SetOutputExtent(fixedImage.GetWholeExtent())
      # TODO: SetBackgroundLevel(inf) then detect then replace with id
      # TODO: for now just map it outside
      reslice.SetBackgroundLevel(999999)
      reslice.SetNumberOfThreads(8)
      reslice.Update()
      
      #H[dim] = reslice.GetOutput()
      H[dim] = vtk.vtkImageData()
      H[dim].DeepCopy( reslice.GetOutput() )

    # NOTE: VTK requires displacement field, need to subtract identity from H
    for dim in xrange(3):
      subf = vtk.vtkImageMathematics()
      subf.SetOperationToSubtract()
      subf.SetInput1(H[dim])
      subf.SetInput2(self.identityMap[dim])
      subf.Update()

      #H[dim] = subf.GetOutput()
      H[dim] = vtk.vtkImageData()
      H[dim].DeepCopy(subf.GetOutput())

    return H
    
  
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



#
# SteeredFluidReg logic
#

class SteeredFluidRegLogic(object):
#TODO
  """ Implement a template matching optimizer that is
  integrated with the slicer main loop.
  """

  def __init__(self,fixed=None,moving=None,transform=None):
    self.interval = 1000
    self.timer = None

    # parameter defaults
    self.regIteration = 10
    self.viscosity = 50.0
    self.opacity = 0.5

    # slicer nodes set by the GUI
    self.fixed = fixed
    self.moving = moving
    #self.transform = transform

    # optimizer state variables
    self.iteration = 0
    self.interaction = False

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
    
    self.arrowsActor = vtk.vtkActor()

    self.movingArrowActor = vtk.vtkActor()
    self.movingContourActor = vtk.vtkActor2D()
    #self.movingContourActor = vtk.vtkImageActor()
    
    self.lastHoveredGradMag = 0

    preferredDeviceType = "GPU"

    # TODO create cl context and queue
    self.clContext = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if cl.device_type.to_string(device.type) == preferredDeviceType:
               self.clContext = cl.Context([device])
               print ("using: %s" % cl.device_type.to_string(device.type))
               break;

    self.clQueue = cl.CommandQueue(self.clContext)

    
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

  def start(self):

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
          


  def processEvent(self,observee,event=None):
  
    from slicer import app

    layoutManager = slicer.app.layoutManager()

    if self.sliceWidgetsPerStyle.has_key(observee):
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
      
        if nodeIndex > 2 and self.lastHoveredGradMag > 0.01:
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)
        
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.startEventPosition = ras
        
          self.actionState = "arrowStart"
        
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
              #g = w.outputGradientMag.GetScalarComponentAsDouble(round(ijk_p[0]), round(ijk_p[1]), round(ijk_p[2]), 0)
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

          print "Init contour pos = " + str(self.movingContourActor.GetPosition())

          # Add image to slice view in row above
          otherSliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex-3, 'vtkMRMLSliceNode')
          otherSliceWidget = layoutManager.sliceWidget(otherSliceNode.GetLayoutName())
          otherSliceView = otherSliceWidget.sliceView()
          otherSliceStyle = otherSliceWidget.interactorStyle()
          otherSliceStyle.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor2D(self.movingContourActor)
      
        else:
          self.actionState = "arrowReject"

        self.abortEvent(event)

      elif event == "LeftButtonReleaseEvent":
      
        if self.actionState == "arrowStart":
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

        self.actionState = "interacting"
        
        self.abortEvent(event)

      elif event == "MouseMoveEvent":

        if self.actionState == "interacting":

          # Hovering
          
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)
          
          w = slicer.modules.SteeredFluidRegWidget
          
          movingRAStoIJK = vtk.vtkMatrix4x4()
          w.movingSelector.currentNode().GetRASToIJKMatrix(movingRAStoIJK)
     
          ijk = movingRAStoIJK.MultiplyPoint(ras + (1,))
          
          g = w.outputGradientMag.GetScalarComponentAsDouble(round(ijk[0]), round(ijk[1]), round(ijk[2]), 0)
          if nodeIndex > 2 and (g > 0.01):
            cursor = qt.QCursor(qt.Qt.OpenHandCursor)
            app.setOverrideCursor(cursor)
          else:
            cursor = qt.QCursor(qt.Qt.ForbiddenCursor)
            app.setOverrideCursor(cursor)
            
          self.lastHoveredGradMag = g

        elif self.actionState == "arrowStart":

          # Dragging / drawing an arrow
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
          # TODO figure out proper scaling factor or arrow source size
          #glyphArrow.SetScaleFactor(0.001)
          glyphArrow.SetScaleFactor(2.0)
          glyphArrow.SetVectorModeToUseVector()
          glyphArrow.SetScaleModeToScaleByVector()
          glyphArrow.Update()
      
          mapper = vtk.vtkPolyDataMapper()
          mapper.SetInput(glyphArrow.GetOutput())
      
          self.movingArrowActor.SetMapper(mapper)

          style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.movingArrowActor)

          #self.movingContourActor.SetPosition(worldXY)
          self.movingContourActor.SetPosition(xy[0]-25, xy[1]-25)
          
        else:
          pass
        
        
        self.abortEvent(event)
          
      else:
        pass

    self.redrawSlices()

  def stop(self):
    self.actionState = "idle"
    self.removeObservers()
    

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
      print "Window size = " + str(winsize)

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

        # DEBUG
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
      # TODO figure out proper scaling factor or arrow source size
      #glyphArrow.SetScaleFactor(0.001)
      glyphArrow.SetScaleFactor(2.0)
      glyphArrow.SetVectorModeToUseVector()
      glyphArrow.SetScaleModeToScaleByVector()
      glyphArrow.Update()
      
      mapper = vtk.vtkPolyDataMapper()
      mapper.SetInput(glyphArrow.GetOutput())
      
      self.arrowsActor.SetMapper(mapper)
      #self.arrowsActor.GetProperty().SetColor([1.0, 0.0, 0.0])

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


    


