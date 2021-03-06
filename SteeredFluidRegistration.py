
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

from RegistrationCL import ImageCL, DeformationCL

# TODO add support for downsampling and upsampling in ImageCL and DeformationCL?

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
# SteeredFluidRegistration
#

class SteeredFluidRegistration:
  def __init__(self, parent):
    parent.title = "SteeredFluidRegistration"
    parent.categories = ["Registration"]
    parent.dependencies = []
    parent.contributors = ["Marcel Prastawa (GE), James Miller (GE), Steve Pieper (Isomics)"] # replace with "Firstname Lastname (Org)"
    parent.helpText = """
    Steerable fluid registration example as a scripted loadable extension.
    """
    parent.acknowledgementText = """
    Funded by NIH grant P41RR013218 (NAC).
""" # replace with organization, grant and thanks.
    self.parent = parent

################################################################################
#
# SteeredFluidRegistrationWidget
#
################################################################################

class SteeredFluidRegistrationWidget:
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

    self.logic = SteeredFluidRegistrationLogic()

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
    self.reloadButton.name = "SteeredFluidRegistration Reload"
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

    # TODO:
    # Grid for image display
    # use QSpinBoxes like deformation grid
    # Determines output volume size, so we can accomodate large images
    # like 500x500x500, that needs CPU resampling at the end
    # use this smaller image grid for steering, when started do
    # downsampling with CPU, when stopped upsample to original grid with CPU
    # TODO: needs ImageCL.copyToVolume(outputVolume)

    display_spinBoxLayout = qt.QGridLayout()
    self.displayGridSpinBoxes = [qt.QSpinBox(), qt.QSpinBox(), qt.QSpinBox()]
    for dim in xrange(3):
      self.displayGridSpinBoxes[dim].setRange(16, 256)
      self.displayGridSpinBoxes[dim].setSingleStep(2)
      self.displayGridSpinBoxes[dim].setValue(256)
      display_spinBoxLayout.addWidget(self.displayGridSpinBoxes[dim], 0, dim)

    uiOptFormLayout.addRow("Display Grid: ", display_spinBoxLayout)

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

    warp_spinBoxLayout = qt.QGridLayout()
    self.warpGridSpinBoxes = [qt.QSpinBox(), qt.QSpinBox(), qt.QSpinBox()]
    for dim in xrange(3):
      self.warpGridSpinBoxes[dim].setRange(16, 128)
      self.warpGridSpinBoxes[dim].setSingleStep(2)
      self.warpGridSpinBoxes[dim].setValue(64)
      warp_spinBoxLayout.addWidget(self.warpGridSpinBoxes[dim], 0, dim)

    regOptFormLayout.addRow("Deformation Grid: ", warp_spinBoxLayout)
    # TODO: regridding callback in logic

    # Fluid kernel width
    self.fluidKernelWidth = ctk.ctkSliderWidget()
    self.fluidKernelWidth.decimals = 1
    self.fluidKernelWidth.singleStep = 0.5
    self.fluidKernelWidth.minimum = 0.5
    self.fluidKernelWidth.maximum = 50.0
    self.fluidKernelWidth.toolTip = "Area of effect for deformation forces."
    regOptFormLayout.addRow("Deformation Stiffness: ", self.fluidKernelWidth)

    self.fluidKernelWidth.value = self.logic.fluidKernelWidth

    self.userInputWeight = ctk.ctkSliderWidget()
    self.userInputWeight.decimals = 1
    self.userInputWeight.singleStep = 0.1
    self.userInputWeight.minimum = 0.0
    self.userInputWeight.maximum = 100.0
    self.userInputWeight.toolTip = "Weight for user input."
    regOptFormLayout.addRow("User Steer Weight: ", self.userInputWeight)

    self.userInputWeight.value = self.logic.userInputWeight

    sliders = (self.drawIterationSlider, self.fluidKernelWidth,
      self.opacitySlider, self.userInputWeight)
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
    self.profilingButton.name = "SteeredFluidRegistration Profiling"
    self.profilingButton.connect('toggled(bool)', self.toggleProfiling)
    devFormLayout.addWidget(self.profilingButton)

    self.debugButton = qt.QCheckBox("Print Debug Messages")
    self.debugButton.toolTip = "Display extra messages in Python console."
    self.debugButton.name = "SteeredFluidRegistration Debug"
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
    self.logic.userInputWeight = self.userInputWeight.value
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
      
      # TODO: move parts to startFluidReg()

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
  
  def onReload(self,moduleName="SteeredFluidRegistration"):
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
# SteeredFluidRegistration logic
#
################################################################################

class SteeredFluidRegistrationLogic(object):
#TODO
  """ Implement a template matching optimizer that is
  integrated with the slicer main loop.
  """

  def __init__(self):
    self.interval = 1000
    self.timer = None

    # parameter defaults
    self.drawIterations = 2
    self.fluidKernelWidth = 15.0
    self.userInputWeight = 1.0
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
    self.arrowsMapper = vtk.vtkPolyDataMapper()
    self.arrowsGlyph = vtk.vtkGlyph3D()

    self.movingArrowActor = vtk.vtkActor()
    self.movingArrowMapper = vtk.vtkPolyDataMapper()
    self.movingArrowGlyph = vtk.vtkGlyph3D()

    self.arrowsActor.GetProperty().SetOpacity(0.5)
    self.arrowsActor.GetProperty().SetColor([0.1, 0.8, 0.1])

    self.movingArrowActor.GetProperty().SetOpacity(0.5)
    self.movingArrowActor.GetProperty().SetColor([0.1, 0.1, 0.9])

    self.movingContourActor = vtk.vtkActor2D()
    #self.movingContourActor = vtk.vtkImageActor()
    
    self.lastHoveredGradMag = 0

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

    widget = slicer.modules.SteeredFluidRegistrationWidget

    self.fixedImageCL = ImageCL(self.preferredDeviceType)
    self.fixedImageCL.fromVolume(axialVolume)
    self.fixedImageCL.normalize()

    fixedShape_down = list(self.fixedImageCL.shape)
    for dim in xrange(3):
      #fixedShape_down[dim] = fixedShape_down[dim] / 2
      fixedShape_down[dim] = \
        min(fixedShape_down[dim], widget.warpGridSpinBoxes[dim].value)
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

    self.movingImageCL = ImageCL(self.preferredDeviceType)
    self.movingImageCL.fromVolume(axialVolume)
    self.movingImageCL.normalize()

  def initOutputVolume(self, outputVolume):
    # NOTE: Reuse old result?
    # TODO: need to store old deformation for this to work, for now reset everything
    widget = slicer.modules.SteeredFluidRegistrationWidget

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

    widget = slicer.modules.SteeredFluidRegistrationWidget

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

    del oldimage

    gradimgcl = imgcl.gradient_magnitude()
    gradimgcl.normalize()

    vtkgradimage = gradimgcl.toVTKImage()

    self.outputGradientMag = vtkgradimage

    # NOTE: may need vtk deep copy
    #self.outputGradientMag = vtk.vtkImageData()
    #self.outputGradientMag.DeepCopy(vtkgradimage)

    #self.outputGradientMagMax = self.outputGradientMag.GetScalarRange()[1]

  def startSteeredRegistration(self):

    widget= slicer.modules.SteeredFluidRegistrationWidget

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

# TODO:
# resample output volume to display grid using CPU
# set identityCL and deformationCL to be this size
    self.identityCL = DeformationCL(self.outputImageCL)
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

    isArrowUsed = False
    
    [gradientsCL, momentasCL] = self.computeImageForces()

    # TODO: store short history of momentas, and user momentas
    # do statistics on interaction
    if not self.arrowQueue.empty():
      self.addUserControl(gradientsCL, momentasCL)
      isArrowUsed = True

    gradientsCL = None

    self.updateDeformation(momentasCL, isArrowUsed)

    momentasCL = None
   
    # Only upsample and redraw updated image every N iterations
    if self.registrationIterationNumber % self.drawIterations == 0:
      self.deformationCL = self.deformationCL_down.compose(self.identityCL)
      #self.deformationCL = self.deformationCL_down.resample(
      #  self.fixedImageCL.shape)
      self.outputImageCL = self.deformationCL.applyTo(self.movingImageCL)

      #TODO: deformationCL and outputImageCL need to be in display grid

      self.updateOutputVolume(self.outputImageCL)
      self.redrawSlices()

    # Initiate another iteration of the registration algorithm.
    if self.interaction:
      qt.QTimer.singleShot(self.interval, self.updateStep)
      
  def computeImageForces(self):
    
    # Gradient descent: grad of output image * (fixed - output)
    diffImageCL_down = self.fixedImageCL_down.subtract(self.outputImageCL_down)

    gradientsCL_down = self.outputImageCL_down.gradient()

    momentasCL_down = [None, None, None]
    for dim in xrange(3):
      imagef = gradientsCL_down[dim].multiply(diffImageCL_down)
      momentasCL_down[dim] = \
        imagef.recursive_gaussian(self.fluidKernelWidth)
      imagef = None

    return [gradientsCL_down, momentasCL_down]

  def addUserControl(self, gradientsCL_down, momentasCL_down):
    
    # Add user inputs to momentum vectors
    # User defined impulses are in arrow queue containing xy, RAS, slice widget
#TODO convert arrow queue to cl matrix
# write GPU kernel to insert them into mom images
# need rastoijk matrix

    # TODO use axial reoriented fixed volume, skip using RAS matrix?
    widget= slicer.modules.SteeredFluidRegistrationWidget
              
    #fixedVolume = widget.fixedSelector.currentNode()
    #movingVolume = widget.movingSelector.currentNode()
    fixedVolume = self.axialFixedVolume
    movingVolume = self.axialMovingVolume

    imageSize = fixedVolume.GetImageData().GetDimensions()

    shape = momentasCL_down[0].shape
    spacing = momentasCL_down[0].spacing

    origin = movingVolume.GetOrigin()

    # Only do 10 arrows per iteration, TODO: allow user adjustment
    numArrowsToProcess = min(self.arrowQueue.qsize(), 10)

    # for mapping drawn force to image grid
    # TODO use reoriented volume with identity matrix?, skip using RAS matrix?
    # issue with VTK negative coord in x,y ?
    movingRAStoIJK = vtk.vtkMatrix4x4()
    movingVolume.GetRASToIJKMatrix(movingRAStoIJK)

    if self.debugMessages:
      print "Folding in %d arrows" % numArrowsToProcess
      print "movingRAStoIJK = " + str(movingRAStoIJK)

    forceX = n.zeros((numArrowsToProcess, 3), n.float32)
    forceV = n.zeros((numArrowsToProcess, 3), n.float32)

    # Splat size depends on amount of motion defined by user
    sigmaM = n.zeros((numArrowsToProcess, 1), n.float32)
    
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

      # NOTE: CL array index is reverse of VTK image index
      startIJK = list(startIJK)
      endIJK = list(endIJK)

      # TODO: DEBUG: need it in VTK order?
      #startIJK.reverse()
      #endIJK.reverse()

      # Scale according to downsampling ratio
      for dim in xrange(3):
        startIJK[dim] *= self.ratios_down[dim]
        endIJK[dim] *= self.ratios_down[dim]

      # TODO: use RAS (with origin and orient)?

      sigma = 0.0
      forceMag = 0.0
      for dim in xrange(3):
        d = 0
        if self.steerMode == "pull":
          d = (startIJK[dim] - endIJK[dim]) * spacing[dim]
          forceX[count, dim] = startIJK[dim] * spacing[dim]
        if self.steerMode == "expand":
          d = (startIJK[dim] - endIJK[dim]) * spacing[dim]
          forceX[count, dim] = endIJK[dim] * spacing[dim]
        if self.steerMode == "shrink":
          d = (startIJK[dim] - endIJK[dim]) * spacing[dim]
          forceX[count, dim] = startIJK[dim] * spacing[dim]

        #d = (startIJK[dim] - endIJK[dim]) * spacing[dim]
        #d = (endIJK[dim] - startIJK[dim]) * spacing[dim]
        #forceX[count, dim] = startIJK[dim] * spacing[dim]

        forceV[count, dim] = d * self.userInputWeight

        sigma += d*d
        forceMag += forceV[count, dim] ** 2

      #sigmaM[count, 0] = sigma
      sigmaM[count, 0] = 1.0

      forceMag = math.sqrt(forceMag)

      # Find vector along grad at start position that projects to the force
      # vector described on the plane
      if self.steerMode == "pull":

        for dim in xrange(3):
          startIJK[dim] = int( round(startIJK[dim]) )
          endIJK[dim] = int( round(endIJK[dim]) )

          if startIJK[dim] < 0:
            startIJK[dim] = 0
          if startIJK[dim] >= shape[dim]:
            startIJK[dim] = shape[dim]-1

        gvec = [0.0, 0.0, 0.0]

        gmag = 0.0
        for dim in xrange(3):
          grad_array = gradientsCL_down[dim].clarray
          g = grad_array[ startIJK[0], startIJK[1], startIJK[2] ]
          gvec[dim] = g.get()[()] # Convert to scalar
          gmag += gvec[dim] ** 2
        gmag = math.sqrt(gmag)
        if gmag == 0.0:
          continue
          
        gdotf = 0
        for dim in xrange(3):
          gvec[dim] = gvec[dim] / gmag
          gdotf += gvec[dim] * forceV[count, dim]
        if gdotf == 0.0:
          continue

        for dim in xrange(3):
          forceV[count, dim] = gvec[dim] * forceMag**2.0 / gdotf

    ImageCL.add_splat3(momentasCL_down, forceX, forceV, sigmaM)

  def updateDeformation(self, momentasCL_down, isArrowUsed):

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

          w = slicer.modules.SteeredFluidRegistrationWidget
          
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

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.RemoveActor(self.movingArrowActor)

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

          renOverlay = self.getOverlayRenderer(
            style.GetInteractor().GetRenderWindow() )
          renOverlay.RemoveActor(self.movingArrowActor)

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
          
          w = slicer.modules.SteeredFluidRegistrationWidget
          
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

          self.movingArrowGlyph.SetInput(pd)
          self.movingArrowGlyph.SetSource(arrowSource.GetOutput())
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

          self.movingArrowGlyph.SetInput(pd)
          self.movingArrowGlyph.SetSource(arrowSource.GetOutput())
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

          self.movingArrowGlyph.SetInput(pd)
          self.movingArrowGlyph.SetSource(arrowSource.GetOutput())
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

      self.arrowsGlyph.SetInput(pd)
      self.arrowsGlyph.SetSource(arrowSource.GetOutput())
      self.arrowsGlyph.ScalingOn()
      self.arrowsGlyph.OrientOn()
      self.arrowsGlyph.SetScaleFactor(1.0)
      self.arrowsGlyph.SetVectorModeToUseVector()
      self.arrowsGlyph.SetScaleModeToScaleByVector()
      self.arrowsGlyph.Update()
      
      self.arrowsMapper.SetInput(self.arrowsGlyph.GetOutput())
      
      self.arrowsActor.SetMapper(self.arrowsMapper)

      # TODO add actors to the appropriate widgets (or all?)
      # TODO make each renwin have two ren's from beginning?

      renOverlay = self.getOverlayRenderer(renwin)

      renOverlay.AddActor(self.arrowsActor)

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
    
    # w = slicer.modules.SteeredFluidRegistrationWidget
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


    


