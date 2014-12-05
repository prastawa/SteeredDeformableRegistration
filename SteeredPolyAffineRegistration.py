
import vtk, qt, ctk, slicer
import math
#import SimpleITK as sitk
#import sitkUtils
import Queue
import time

import cProfile, pstats, StringIO

import numpy as n

import pyopencl as cl
import pyopencl.array as cla

from RegistrationCL import ImageCL, DeformationCL, PolyAffineCL, \
  SteeringRotation, SteeringScale

# TODO add support for downsampling and upsampling?
# TODO use image patch / compositing instead

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
# SteeredPolyAffineRegistration
#

class SteeredPolyAffineRegistration:
  def __init__(self, parent):
    parent.title = "SteeredPolyAffineRegistration"
    parent.categories = ["Registration"]
    parent.dependencies = []
    parent.contributors = ["Marcel Prastawa (GE), James Miller (GE), Steve Pieper (Isomics)"] # replace with "Firstname Lastname (Org)"
    parent.helpText = """
    Steerable poly-affine registration example as a scripted loadable extension.
    """
    parent.acknowledgementText = """
    Funded by NIH grant P41RR013218 (NAC).
""" # replace with organization, grant and thanks.
    self.parent = parent

################################################################################
#
# SteeredPolyAffineRegistrationWidget
#
################################################################################

class SteeredPolyAffineRegistrationWidget:
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

    self.logic = SteeredPolyAffineRegistrationLogic()

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
    self.reloadButton.name = "SteeredPolyAffineRegistration Reload"
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
    self.rotateModeRadio = qt.QRadioButton("Rotate")
    self.scaleModeRadio = qt.QRadioButton("Scale")
    steerModeLayout.addWidget(self.rotateModeRadio, 0, 0)
    steerModeLayout.addWidget(self.scaleModeRadio, 0, 1)

    steerModeRadios = (self.scaleModeRadio, self.rotateModeRadio)
    for r in steerModeRadios:
      r.connect('clicked(bool)', self.updateLogicFromGUI)

    self.rotateModeRadio.checked = True

    uiOptFormLayout.addRow("Steering Mode: ", steerModeLayout)

    # Number of polyaffine transforms at each dimension
    self.numberAffinesSlider = ctk.ctkSliderWidget()
    self.numberAffinesSlider.decimals = 0
    self.numberAffinesSlider.singleStep = 1
    self.numberAffinesSlider.minimum = 1
    self.numberAffinesSlider.maximum = 100
    self.numberAffinesSlider.toolTip = "Number of affines in each dim"
    uiOptFormLayout.addRow("Number of affines per dim:", self.numberAffinesSlider)

    self.numberAffinesSlider.value = self.logic.numberAffines


    # Draw iterations
    self.drawIterationSlider = ctk.ctkSliderWidget()
    self.drawIterationSlider.decimals = 0
    self.drawIterationSlider.singleStep = 1
    self.drawIterationSlider.minimum = 1
    self.drawIterationSlider.maximum = 100
    self.drawIterationSlider.toolTip = "Update and draw every N iterations"
    uiOptFormLayout.addRow("Redraw Iterations:", self.drawIterationSlider)

    self.drawIterationSlider.value = self.logic.drawIterations

    #
    # Registration regOptions collapsible button
    #
    regOptCollapsibleButton = ctk.ctkCollapsibleButton()
    regOptCollapsibleButton.text = "Registration Parameters"
    self.layout.addWidget(regOptCollapsibleButton)

    # Layout within the parameter collapsible button
    regOptFormLayout = qt.QFormLayout(regOptCollapsibleButton)

    # TODO: button for adding transform at current slice positions?

    self.polyAffineRadius = ctk.ctkSliderWidget()
    self.polyAffineRadius.decimals = 1
    self.polyAffineRadius.singleStep = 0.5
    self.polyAffineRadius.minimum = 1.0
    self.polyAffineRadius.maximum = 100.0
    self.polyAffineRadius.toolTip = "Area of effect for new transform."
    regOptFormLayout.addRow("Polyaffine radius: ", self.polyAffineRadius)

    self.polyAffineRadius.value = self.logic.polyAffineRadius

    sliders = (self.drawIterationSlider, self.polyAffineRadius)
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
    self.profilingButton.name = "SteeredPolyAffineRegistration Profiling"
    self.profilingButton.connect('toggled(bool)', self.toggleProfiling)
    devFormLayout.addWidget(self.profilingButton)

    self.debugButton = qt.QCheckBox("Print Debug Messages")
    self.debugButton.toolTip = "Display extra messages in Python console."
    self.debugButton.name = "SteeredPolyAffineRegistration Debug"
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
    
    self.logic.numberAffines = self.numberAfffinesSlider.value
    self.logic.drawIterations = self.drawIterationSlider.value
    self.logic.polyAffineRadius = self.polyAffineRadius.value

    self.logic.debugMessages = self.debugButton.checked

    if self.rotateModeRadio.checked:
      self.logic.steerMode = "rotate"
    if self.scaleModeRadio.checked:
      self.logic.steerMode = "scale"

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
      
      # TODO: move parts to startPolyAffineReg()

      #orig = movingVolume.GetImageData().GetOrigin()
      #sp = movingVolume.GetImageData().GetSpacing()
      #print "Call build id with orig = " + str(orig) + " sp = " + str(sp)
      
  
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
    
  
  #########################################################################
  
  def onReload(self,moduleName="SteeredPolyAffineRegistration"):
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
# SteeredPolyAffineRegistration logic
#
################################################################################

class SteeredPolyAffineRegistrationLogic(object):
#TODO
  """ Implement a template matching optimizer that is
  integrated with the slicer main loop.
  """

  def __init__(self):
    self.interval = 1000
    self.timer = None

    # parameter defaults
    self.numberAffines = 4
    self.drawIterations = 2
    self.polyAffineRadius = 10.0

    # TODO
    #self.transform = ?

    # optimizer state variables
    self.iteration = 0
    self.interaction = False

    self.steerMode = "rotate"

    self.position = []
    self.paintCoordinates = []

    self.lastEventPosition = [0.0, 0.0, 0.0]
    self.startEventPosition = [0.0, 0.0, 0.0]
    
    # Queue containing info on line draw events, tuples of
    # (Mtime, xy0, RAS0, xy1, RAS1, sliceWidget)
    self.lineQueue = Queue.Queue()

    self.lineStartXY = (0, 0, 0)
    self.lineEndXY = (0, 0, 0)
    
    self.lineStartRAS = [0.0, 0.0, 0.0]
    self.lineEndRAS = [0.0, 0.0, 0.0]

    print("Reload")

    self.actionState = "idle"
    self.interactorObserverTags = []
    
    self.styleObserverTags = []
    self.sliceWidgetsPerStyle = {}

    self.nodeIndexPerStyle = {}
    self.sliceNodePerStyle = {}
    
    self.lastDrawMTime = 0
    self.lastDrawSliceWidget = None

    self.linesActor = vtk.vtkActor()
    self.linesMapper = vtk.vtkPolyDataMapper()
    self.linesGlyph = vtk.vtkGlyph3D()

    self.movingArrowActor = vtk.vtkActor()
    self.movingArrowMapper = vtk.vtkPolyDataMapper()
    self.movingArrowGlyph = vtk.vtkGlyph3D()

    self.linesActor.GetProperty().SetOpacity(0.5)
    self.linesActor.GetProperty().SetColor([0.1, 0.8, 0.1])

    self.movingArrowActor.GetProperty().SetOpacity(0.5)
    self.movingArrowActor.GetProperty().SetColor([0.1, 0.1, 0.9])

    self.preferredDeviceType = "GPU"
    
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
          rencol.GetItemAsObject(1).RemoveActor(self.linesActor)

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

    widget = slicer.modules.SteeredPolyAffineRegistrationWidget

    self.fixedImageCL = ImageCL(self.preferredDeviceType)
    self.fixedImageCL.fromVolume(axialVolume)
    self.fixedImageCL.normalize()

  def useMovingVolume(self, volume):

    # TODO store original orientation?
    axialVolume = self.reorientVolumeToAxial(volume)
    self.axialMovingVolume = axialVolume

    self.movingImageCL = ImageCL(self.preferredDeviceType)
    self.movingImageCL.fromVolume(axialVolume)
    self.movingImageCL.normalize()

  def initOutputVolume(self, outputVolume):
    # NOTE: Reuse old result?
    # TODO: need to store old deformation for this to work, for now reset everything
    widget = slicer.modules.SteeredPolyAffineRegistrationWidget

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

    self.outputImageCL = ImageCL(self.preferredDeviceType)
    self.outputImageCL.fromVolume(outputVolume)
    self.outputImageCL.normalize()
        
    # Force update of gradient magnitude image
    self.updateOutputVolume( self.outputImageCL )

  def updateOutputVolume(self, imgcl):

    widget = slicer.modules.SteeredPolyAffineRegistrationWidget

    outputVolume = widget.outputSelector.currentNode()  

    """
    displayShape = self.fixedImageCL.shape
    for dim in xrange(3):
      displayShape[dim] = min(
        displayShape[dim], widget.displayGridSpinBoxes[dim].value)

    imgcl_gridded = imgcl.resample(displayShape)
    imgcl_gridded.copyToVolume(outputVolume)

    # TODO: need ratios / mapping between screen and grad mag image
    # displayGridRatios, warpGridRatios
    """

    vtkimage = imgcl.toVTKImage()

    #TODO sync origin
  
    #castf = vtk.vtkImageCast()
    #castf.SetOutputScalarTypeToFloat()
    #castf.SetInput(vtkimage)
    #castf.Update()
  
    #outputVolume.GetImageData().GetPointData().SetScalars( castf.GetOutput().GetPointData().GetScalars() )

    oldimage = outputVolume.GetImageData()

    outputVolume.SetAndObserveImageData(vtkimage)
    #outputVolume.GetImageData().GetPointData().SetScalars( vtkimage.GetPointData().GetScalars() )
    #outputVolume.GetImageData().GetPointData().GetScalars().Modified()
    #outputVolume.GetImageData().Modified()
    #outputVolume.Modified()

  def startSteeredRegistration(self):

    widget= slicer.modules.SteeredPolyAffineRegistrationWidget

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

    #compositeNodes.values()[0].LinkedControlOn()
    yellowWidget = layoutManager.sliceWidget('Yellow')
      
    # Set all view orientation to axial
    #sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    #for nodeIndex in xrange(sliceNodeCount):
    #  sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
    #  sliceNode.SetOrientationToAxial()
      
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.FitSliceToAll()
    
    # TODO: initialize poly affine
    # One big affine?
    # Eight affine blocks?

    self.polyAffine = PolyAffineCL(self.fixedImageCL, self.movingImageCL)
    self.polyAffine.create_identity(self.numberAffines)
    self.polyAffine.setup_optimizer()

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

    isLineUsed = False

    self.polyAffineCL.step()
    
    if not self.lineQueue.empty():
      self.invoke_correction()
      isLineUsed = True

    # Only upsample and redraw updated image every N iterations
    if self.registrationIterationNumber % self.drawIterations == 0:
      #self.outputImageCL = self.polyAffineCL.applyTo(self.origMovingImageCL)
      self.outputImageCL = self.polyAffineCL.movingImageCL

      self.updateOutputVolume(self.outputImageCL)
      self.redrawSlices()

    # Initiate another iteration of the registration algorithm.
    if self.interaction:
      qt.QTimer.singleShot(self.interval, self.updateStep)

  def invoke_correction(self):
    
    widget= slicer.modules.SteeredPolyAffineRegistrationWidget

    # Determine region center and radius
              
    #fixedVolume = widget.fixedSelector.currentNode()
    #movingVolume = widget.movingSelector.currentNode()
    fixedVolume = self.axialFixedVolume
    movingVolume = self.axialMovingVolume

    imageSize = fixedVolume.GetImageData().GetDimensions()

    shape = self.fixedImageCL.shape
    spacing = self.fixedImageCL.spacing

    origin = movingVolume.GetOrigin()

    # TODO

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
        
          self.lineStartXY = xy
          self.lineStartRAS = ras

        elif self.steerMode == "expand" and nodeIndex > 2:

          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)
        
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.startEventPosition = ras
        
          self.actionState = "expandStart"

          self.expandStartTime = time.clock()
        
          self.lineStartXY = xy
          self.lineStartRAS = ras

        elif self.steerMode == "shrink" and nodeIndex > 2:

          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)
        
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.startEventPosition = ras
        
          self.actionState = "shrinkStart"

          self.lineStartXY = xy
          self.lineStartRAS = ras
      
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

          self.lineEndXY = xy
          self.lineEndRAS = ras

          # TODO: only draw line within ??? seconds
          self.lastDrawMTime = sliceNode.GetMTime()
          
          self.lastDrawSliceWidget = sliceWidget

          self.lineQueue.put(
            (sliceNode.GetMTime(), sliceWidget, self.lineStartXY, self.lineEndXY, self.lineStartRAS, self.lineEndRAS) )

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.RemoveActor(self.movingArrowActor)

        if self.actionState == "expandStart":
          cursor = qt.QCursor(qt.Qt.OpenHandCursor)
          app.setOverrideCursor(cursor)

          # TODO: only draw line within ??? seconds
          self.lastDrawMTime = sliceNode.GetMTime()
          
          self.lastDrawSliceWidget = sliceWidget

          a = self.lineStartXY
          for i in xrange(len(self.expandShrinkVectors)):
            v = self.expandShrinkVectors[i]
            xy = (a[0]+v[0], a[1]+v[1], 0.0)

            xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
            ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

            lineEndXY = xy
            lineEndRAS = ras

            self.lineQueue.put(
              (sliceNode.GetMTime(), sliceWidget, self.lineStartXY, lineEndXY, self.lineStartRAS, lineEndRAS) )

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.RemoveActor(self.movingArrowActor)

        if self.actionState == "shrinkStart":
          cursor = qt.QCursor(qt.Qt.OpenHandCursor)
          app.setOverrideCursor(cursor)

          # TODO: only draw line within ??? seconds
          self.lastDrawMTime = sliceNode.GetMTime()
          
          self.lastDrawSliceWidget = sliceWidget

          a = self.lineStartXY
          for i in xrange(len(self.expandShrinkVectors)):
            v = self.expandShrinkVectors[i]
            xy = (a[0]+v[0], a[1]+v[1], 0.0)

            xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
            ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

            lineEndXY = xy
            lineEndRAS = ras

            self.lineQueue.put(
              (sliceNode.GetMTime(), sliceWidget, lineEndXY, self.lineStartXY, lineEndRAS, self.lineStartRAS) )

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.RemoveActor(self.movingArrowActor)

        self.actionState = "interacting"
        
        self.abortEvent(event)

      elif event == "MouseMoveEvent":

        if self.actionState == "interacting" and self.steerMode == "pull":

          # Hovering when pulling -> change cursors based on gradient
          
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)
          
          w = slicer.modules.SteeredPolyAffineRegistrationWidget
          
          movingRAStoIJK = vtk.vtkMatrix4x4()
          w.movingSelector.currentNode().GetRASToIJKMatrix(movingRAStoIJK)
     
          ijk = movingRAStoIJK.MultiplyPoint(ras + (1,))
          
          if nodeIndex > 2:
            cursor = qt.QCursor(qt.Qt.OpenHandCursor)
            app.setOverrideCursor(cursor)
          else:
            cursor = qt.QCursor(qt.Qt.ForbiddenCursor)
            app.setOverrideCursor(cursor)
            
          self.lastHoveredGradMag = g

        elif self.actionState == "pullStart":

          # Drawing an line that pulls/drags voxels
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)

          xy = style.GetInteractor().GetEventPosition()

          coord = vtk.vtkCoordinate()
          coord.SetCoordinateSystemToDisplay()

          coord.SetValue(self.lineStartXY[0], self.lineStartXY[1], 0.0)
          worldStartXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          coord.SetValue(xy[0], xy[1], 0.0)
          worldXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          # TODO refactor code, one method for drawing collection of lines, one actor for all lines (moving + static ones)?

          pts = vtk.vtkPoints()
          pts.InsertNextPoint(worldStartXY)

          vectors = vtk.vtkDoubleArray()
          vectors.SetNumberOfComponents(3)
          vectors.SetNumberOfTuples(1)
          vectors.SetTuple3(0, worldXY[0]-worldStartXY[0], worldXY[1]-worldStartXY[1], worldXY[2]-worldStartXY[2]) 

          pd = vtk.vtkPolyData()
          pd.SetPoints(pts)
          pd.GetPointData().SetVectors(vectors)

          lineSource = vtk.vtkArrowSource()

          self.movingArrowGlyph.SetInput(pd)
          self.movingArrowGlyph.SetSource(lineSource.GetOutput())
          self.movingArrowGlyph.ScalingOn()
          self.movingArrowGlyph.OrientOn()
          self.movingArrowGlyph.SetScaleFactor(1.0)
          self.movingArrowGlyph.SetVectorModeToUseVector()
          self.movingArrowGlyph.SetScaleModeToScaleByVector()
          self.movingArrowGlyph.Update()
      
          self.movingArrowMapper.SetInput(self.movingArrowGlyph.GetOutput())
      
          self.movingArrowActor.SetMapper(self.movingArrowMapper)

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.AddActor(self.movingArrowActor)

        elif self.actionState == "expandStart":

          # Drawing lines for local expansion
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)

          xy = style.GetInteractor().GetEventPosition()

          coord = vtk.vtkCoordinate()
          coord.SetCoordinateSystemToDisplay()

          coord.SetValue(self.lineStartXY[0], self.lineStartXY[1], 0.0)
          worldStartXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          coord.SetValue(xy[0], xy[1], 0.0)
          worldXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          # TODO: click and hold mode? requires a timer for detection
          #lineSize = (time.clock() - self.expandStartTime) * 0.1
          lineSize = 0.0
          for dim in xrange(2):
            lineSize += (xy[dim] - self.lineStartXY[dim]) ** 2.0
          lineSize = math.sqrt(lineSize)
          dArrowSize = 0.7 * lineSize

          worldArrowSize = 0.0
          for dim in xrange(3):
            worldArrowSize += (worldXY[dim] - worldStartXY[dim]) ** 2.0
          worldArrowSize = math.sqrt(worldArrowSize)
          dWorldArrowSize = 0.7*worldArrowSize

          pts = vtk.vtkPoints()
          for count in xrange(8):
            pts.InsertNextPoint(worldStartXY)

          self.expandShrinkVectors = []
          self.expandShrinkVectors.append( (lineSize, 0) )
          self.expandShrinkVectors.append( (-lineSize, 0) )
          self.expandShrinkVectors.append( (0, lineSize) )
          self.expandShrinkVectors.append( (0, -lineSize) )
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

          lineSource = vtk.vtkArrowSource()

          self.movingArrowGlyph.SetInput(pd)
          self.movingArrowGlyph.SetSource(lineSource.GetOutput())
          self.movingArrowGlyph.ScalingOn()
          self.movingArrowGlyph.OrientOn()
          self.movingArrowGlyph.SetScaleFactor(1.0)
          self.movingArrowGlyph.SetVectorModeToUseVector()
          self.movingArrowGlyph.SetScaleModeToScaleByVector()
          self.movingArrowGlyph.Update()
      
          self.movingArrowMapper.SetInput(self.movingArrowGlyph.GetOutput())
      
          self.movingArrowActor.SetMapper(self.movingArrowMapper)

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.AddActor(self.movingArrowActor)

        elif self.actionState == "shrinkStart":

          # Drawing lines for local shrinkage
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)

          xy = style.GetInteractor().GetEventPosition()

          coord = vtk.vtkCoordinate()
          coord.SetCoordinateSystemToDisplay()

          coord.SetValue(self.lineStartXY[0], self.lineStartXY[1], 0.0)
          worldStartXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          coord.SetValue(xy[0], xy[1], 0.0)
          worldXY = coord.GetComputedWorldValue(style.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

          # TODO: click and hold mode? requires a timer for detection
          #lineSize = (time.clock() - self.expandStartTime) * 0.1
          lineSize = 0.0
          for dim in xrange(2):
            lineSize += (xy[dim] - self.lineStartXY[dim]) ** 2.0
          lineSize = math.sqrt(lineSize)
          dArrowSize = 0.7 * lineSize

          worldArrowSize = 0.0
          for dim in xrange(3):
            worldArrowSize += (worldXY[dim] - worldStartXY[dim]) ** 2.0
          worldArrowSize = math.sqrt(worldArrowSize)
          dWorldArrowSize = 0.7*worldArrowSize

          self.expandShrinkVectors = []
          self.expandShrinkVectors.append( (lineSize, 0) )
          self.expandShrinkVectors.append( (-lineSize, 0) )
          self.expandShrinkVectors.append( (0, lineSize) )
          self.expandShrinkVectors.append( (0, -lineSize) )
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

          lineSource = vtk.vtkArrowSource()

          self.movingArrowGlyph.SetInput(pd)
          self.movingArrowGlyph.SetSource(lineSource.GetOutput())
          self.movingArrowGlyph.ScalingOn()
          self.movingArrowGlyph.OrientOn()
          self.movingArrowGlyph.SetScaleFactor(1.0)
          self.movingArrowGlyph.SetVectorModeToUseVector()
          self.movingArrowGlyph.SetScaleModeToScaleByVector()
          self.movingArrowGlyph.Update()
      
          self.movingArrowMapper.SetInput(self.movingArrowGlyph.GetOutput())
      
          self.movingArrowActor.SetMapper(self.movingArrowMapper)

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.AddActor(self.movingArrowActor)
          
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
      if cmd is not None:
        cmd.SetAbortFlag(1)

  def redrawSlices(self):
    # TODO: memory leak

    layoutManager = slicer.app.layoutManager()
    sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
    for nodeIndex in xrange(sliceNodeCount):
      # find the widget for each node in scene
      sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
      sliceWidget = layoutManager.sliceWidget(sliceNode.GetLayoutName())
      
      if sliceWidget:
        renwin = sliceWidget.sliceView().renderWindow()
        rencol = renwin.GetRenderers()
        if rencol.GetNumberOfItems() >= 2:
          rencol.GetItemAsObject(1).RemoveActor(self.linesActor)
          
        renwin.Render()
        
    if not self.lineQueue.empty():
    
      numArrows = self.lineQueue.qsize()

      # TODO renwin = lineTuple[1], need to maintain actors for each renwin?
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
        lineTuple = self.lineQueue.queue[i]
        sliceWidget = lineTuple[1]
        startXY = lineTuple[2]
        endXY = lineTuple[3]
        startRAS = lineTuple[4]
        endRAS = lineTuple[5]

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
      
      lineSource = vtk.vtkArrowSource()
      #lineSource.SetTipLength(1.0 / winsize[0])
      #lineSource.SetTipRadius(2.0 / winsize[0])
      #lineSource.SetShaftRadius(1.0 / winsize[0])

      self.linesGlyph.SetInput(pd)
      self.linesGlyph.SetSource(lineSource.GetOutput())
      self.linesGlyph.ScalingOn()
      self.linesGlyph.OrientOn()
      self.linesGlyph.SetScaleFactor(1.0)
      self.linesGlyph.SetVectorModeToUseVector()
      self.linesGlyph.SetScaleModeToScaleByVector()
      self.linesGlyph.Update()
      
      self.linesMapper.SetInput(self.linesGlyph.GetOutput())
      
      self.linesActor.SetMapper(self.linesMapper)

      # TODO add actors to the appropriate widgets (or all?)
      # TODO make each renwin have two ren's from beginning?

      renOverlay = self.getOverlayRenderer(renwin)

      renOverlay.AddActor(self.linesActor)

      renwin.Render()

  def getOverlayRenderer(self, renwin):
    rencol = renwin.GetRenderers()
      
    renOverlay = None
    if rencol.GetNumberOfItems() >= 2:
      renOverlay = rencol.GetItemAsObject(1)
    else:
      renOverlay = vtk.vtkRenderer()
      renwin.SetNumberOfLayers(2)
      renwin.AddRenderer(renOverlay)

    #renOverlay.SetInteractive(0)
    #renOverlay.SetLayer(1)

    return renOverlay

  def testingData(self):
    """Load some default data for development
    and set up a transform and viewing scenario for it.
    """

    #import SampleData
    #sampleDataLogic = SampleData.SampleDataLogic()
    #mrHead = sampleDataLogic.downloadMRHead()
    #dtiBrain = sampleDataLogic.downloadDTIBrain()
    
    # w = slicer.modules.SteeredPolyAffineRegistrationWidget
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


    


