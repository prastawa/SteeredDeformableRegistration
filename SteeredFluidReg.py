# Based on work by Steve Pieper, Kunlin Cao, Dirk Padfield

#from __main__ import vtk, qt, ctk, slicer
import vtk, qt, ctk, slicer
import math
import threading
import SimpleITK as sitk
import sitkUtils
import Queue

# TODO
#
# CL classes
#

# ImageCL
# convolve, grad, add, mul, sub

# DeformationCL
# compose, warp/apply

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
    self.viscositySlider.toolTip = "Viscosity of fluid deformation."
    optFormLayout.addRow("Fluid viscosity:", self.viscositySlider)

    # get default values from logic
    self.regIterationSlider.value = self.logic.regIteration
    self.viscositySlider.value = self.logic.viscosity

    #print(self.logic.regIteration)

    sliders = (self.regIterationSlider, self.viscositySlider)
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
      
      # Set up display of volumes as composites, and create output node if not specified
      fixedVolume = self.fixedSelector.currentNode()
      movingVolume = self.movingSelector.currentNode()
      outputVolume = self.outputSelector.currentNode()

      cool1 = slicer.vtkMRMLColorTableNode()
      cool1.SetTypeToCool1()
      fixedVolume.GetScene().AddNode(cool1)

      warm1 = slicer.vtkMRMLColorTableNode()
      warm1.SetTypeToWarm1()
      movingVolume.GetScene().AddNode(warm1)

      fixedDisplay = fixedVolume.GetDisplayNode()
      fixedDisplay.SetAndObserveColorNodeID(cool1.GetID())

      movingDisplay = movingVolume.GetDisplayNode()
      movingDisplay.SetAndObserveColorNodeID(warm1.GetID())

      # TODO: DEBUG: cast images before processing?
      # TODO: crashes, how to cast volume nodes?
      #fixcastf = vtk.vtkImageCast()
      #fixcastf.SetOutputScalarTypeToDouble()
      #fixcastf.SetInput(fixedVolume.GetImageData())
      #fixcastf.Update()
      #fixedVolume.SetAndObserveImageData(self.normalizeImage( fixcastf.GetOutput() ) )
      #fixedVolume.Modified()

      #movcastf = vtk.vtkImageCast()
      #movcastf.SetOutputScalarTypeToDouble()
      #movcastf.SetInput(movingVolume.GetImageData())
      #movcastf.Update()
      #movingVolume.SetAndObserveImageData(self.normalizeImage( movcastf.GetOutput() ) )
      #movingVolume.Modified()
      
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
        outputVolume.GetImageData().GetPointData().SetScalars(movingVolume.GetImageData().GetPointData().GetScalars())

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
      
      self.identityMap = self.buildIdentity( movingVolume.GetImageData() )
      
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

        
      # Force update of gradient magnitude image
      self.updateOutputVolume( outputVolume.GetImageData() )
      
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
      
      compositeNodes = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')
      for compositeNode in compositeNodes.values():
        print "Composite node " + compositeNode.GetName()

        #compositeNode.SetLabelVolumeID(self.gridVolume.GetID())
        compositeNode.SetBackgroundVolumeID(fixedVolume.GetID())
        compositeNode.SetForegroundVolumeID(outputVolume.GetID())
        
        compositeNode.SetForegroundOpacity(0.5)
        if ii == 1:
          compositeNode.SetForegroundOpacity(0.0)
        if ii == 0:
          compositeNode.SetForegroundOpacity(1.0)
        compositeNode.SetLabelOpacity(0.5)
        
        ii += 1

      compositeNodes.values()[0].LinkedControlOn()
        
      # Set all view orientation to axial
      sliceNodeCount = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLSliceNode')
      for nodeIndex in xrange(sliceNodeCount):
        sliceNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, 'vtkMRMLSliceNode')
        sliceNode.SetOrientationToAxial()
        
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
    
  def updateOutputVolume(self, outputImage):
  
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToFloat()
    castf.SetInput(outputImage)
    castf.Update()
  
    outputVolume = self.outputSelector.currentNode()  
    outputVolume.GetImageData().GetPointData().SetScalars( castf.GetOutput().GetPointData().GetScalars() )
    outputVolume.GetImageData().GetPointData().GetScalars().Modified()
    outputVolume.GetImageData().Modified()
    outputVolume.Modified()
    
    gmagf = vtk.vtkImageGradientMagnitude()
    gmagf.SetDimensionality(3)
    gmagf.HandleBoundariesOn()
    gmagf.SetInput(outputImage)
    gmagf.Update()

    gradImage = vtk.vtkImageData()
    gradImage.DeepCopy( gmagf.GetOutput() )
    
    self.outputGradientMag = self.normalizeImage(gradImage)

    
  def startDeformableRegistration(self):     
    fixedVolume = self.fixedSelector.currentNode()
    movingVolume = self.movingSelector.currentNode()
    #outputVolume = self.outputSelector.currentNode()
    # initialTransform = self.initialTransformSelector.currentNode()
    # outputTransform = self.transformSelector.currentNode()
 
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
    
    # Create floating point images of zeros for momenta
    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToDouble()
    castf.SetInput(movingVolume.GetImageData())
    castf.Update()
    
    momentaX = vtk.vtkImageData()
    momentaX.DeepCopy(castf.GetOutput())
    momentaX.GetPointData().GetScalars().FillComponent(0, 0.0)
    
    momentaY = vtk.vtkImageData()
    momentaY.DeepCopy(momentaX)
    
    momentaZ = vtk.vtkImageData()
    momentaZ.DeepCopy(momentaX)
    
    self.momentas = [momentaX, momentaY, momentaZ]
    
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
    
    # TODO get images
    
    # SITK
    #Ifixed = sitkUtils.PullFromSlicer(self.fixedSelector.currentNode())
    #Imoving = sitkUtils.PullFromSlicer(self.movingSelector.currentNode())
    #
    #s = Ifixed.GetSpacing()
    #print "Ifixed spacing (%.2f, %.2f, %.2f)" % (s[0], s[1], s[2])
    
    # VTK
    fixedVolume = self.fixedSelector.currentNode()
    movingVolume = self.movingSelector.currentNode()
    outputVolume = self.outputSelector.currentNode()

    fixedImage = fixedVolume.GetImageData()
    movingImage = movingVolume.GetImageData()
    outputImage = outputVolume.GetImageData()
    
    # HACK
    # TODO VTK filters sometimes do not work when inputs are not of double type
    # TODO should do this in setup phase, need to override MRML volume node
    fixcastf = vtk.vtkImageCast()
    fixcastf.SetOutputScalarTypeToDouble()
    fixcastf.SetInput(fixedImage)
    fixcastf.Update()
    fixedImage = self.normalizeImage( fixcastf.GetOutput() )
    
    movcastf = vtk.vtkImageCast()
    movcastf.SetOutputScalarTypeToDouble()
    movcastf.SetInput(movingImage)
    movcastf.Update()
    movingImage = movcastf.GetOutput()
    scaledMovingImage = self.normalizeImage( movcastf.GetOutput() )

    outcastf = vtk.vtkImageCast()
    outcastf.SetOutputScalarTypeToDouble()
    outcastf.SetInput(outputImage)
    outcastf.Update()
    outputImage = self.normalizeImage( outcastf.GetOutput() )
    
    imageSize = fixedImage.GetDimensions()
    
    # Reset momentas to zero
    # TODO need to keep track of compounded deformation fields for final transform
    # TODO recompute output image as moving warped by this compounded deformation
    # for dim in xrange(3):
      # castf = vtk.vtkImageCast()
      # castf.SetOutputScalarTypeToDouble()
      # castf.SetInput(outputVolume.GetImageData())
      # castf.Update()
    
      # momenta = vtk.vtkImageData()
      # momenta.DeepCopy(castf.GetOutput())
      # momenta.GetPointData().GetScalars().FillComponent(0, 0.0)
      
      # #self.momentas[dim].GetPointData().GetScalars().FillComponent(0, 0.0)
      # self.momentas[dim] = momenta
    
    # Gradient descent: grad of output image * diff(fixed, output)
    gradf = vtk.vtkImageGradient()
    gradf.SetDimensionality(3)
    gradf.SetInput(outputImage)
    gradf.Update()
    
    gradImages = [None, None, None]
    for dim in xrange(3):
      extractf = vtk.vtkImageExtractComponents()
      extractf.SetComponents(dim)
      extractf.SetInput(gradf.GetOutput())
      extractf.Update()

      #gradImages[dim] = extractf.GetOutput()
      gradImages[dim] = vtk.vtkImageData()
      gradImages[dim].DeepCopy(extractf.GetOutput())
      
    subf = vtk.vtkImageMathematics()
    subf.SetOperationToSubtract()
    subf.SetInput1(fixedImage)
    subf.SetInput2(outputImage)
    subf.Update()
    
    diffImage = subf.GetOutput()
    
    for dim in xrange(3):
      mulf = vtk.vtkImageMathematics()
      mulf.SetOperationToMultiply()
      mulf.SetInput1(diffImage)
      mulf.SetInput2(gradImages[dim])
      mulf.Update()

      self.momentas[dim] = vtk.vtkImageData()
      self.momentas[dim].DeepCopy( mulf.GetOutput() )

    isArrowUsed = False
    
    # Add user inputs to momentum vectors
    # User defined impulses are in arrow queue containing xy, RAS, slice widget
    if not self.logic.arrowQueue.empty():

      isArrowUsed = True
    
      arrowTuple = self.logic.arrowQueue.get()
      
      print "arrowTuple = " + str(arrowTuple)
      
      Mtime = arrowTuple[0]
      sliceWidget = arrowTuple[1]
      startXY = arrowTuple[2]
      endXY = arrowTuple[3]
      startRAS = arrowTuple[4]
      endRAS = arrowTuple[5]
      
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
        forceVector[dim] = (endRAS[dim] - startRAS[dim])
        #forceVector[dim] = (endIJK[dim] - startIJK[dim])
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
              gvec[dim] = gradImages[dim].GetScalarComponentAsDouble(pos[0], pos[1], pos[2], 0)
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
              a = self.momentas[dim].GetScalarComponentAsDouble(pos[0], pos[1], pos[2], 0)
              self.momentas[dim].SetScalarComponentFromDouble(pos[0], pos[1], pos[2], 0,
                a + gvec[dim] * forceMag**2.0 / gdotf)
             
      # for dim in xrange(3):
        # a = self.momentas[dim].GetScalarComponentAsDouble(forceCenter[0], forceCenter[1], forceCenter[2], 0)
        # self.momentas[dim].SetScalarComponentFromDouble(forceCenter[0], forceCenter[1], forceCenter[2], 0,
          # a + forceVector[dim])
              
    velocList = []
    for dim in xrange(3):
      velocList.append( self.applyImageKernel(self.momentas[dim]) )
      
    # Compute max velocity
    sqf = vtk.vtkImageMathematics()
    sqf.SetOperationToSquare()
    sqf.SetInput1(velocList[0])
    sqf.Update()
    
    velocMagImage = sqf.GetOutput()
    
    for dim in xrange(1,3):
      mulf2 = vtk.vtkImageMathematics()
      mulf2.SetOperationToSquare()
      mulf2.SetInput1(velocList[dim])
      mulf2.Update()
      
      addf = vtk.vtkImageMathematics()
      addf.SetOperationToAdd()
      addf.SetInput1(mulf2.GetOutput())
      addf.SetInput2(velocMagImage)
      addf.Update()
      
      velocMagImage = addf.GetOutput()
      
    maxVeloc = velocMagImage.GetScalarRange()[1]
    print "maxVeloc = %f" % maxVeloc
    
    if maxVeloc <= 0.0:
      return
      
    maxVeloc = math.sqrt(maxVeloc)

    print "delta = %f" % self.fluidDelta
    
    if self.fluidDelta == 0.0 or (self.fluidDelta*maxVeloc) > 2.0:
      self.fluidDelta = 2.0 / maxVeloc
      print "new delta = %f" % self.fluidDelta

    print "maxVeloc*delta = %f" % (maxVeloc*self.fluidDelta)

    for dim in xrange(3):
      scalf = vtk.vtkImageMathematics()
      scalf.SetOperationToMultiplyByK()
      scalf.SetInput1(velocList[dim])
      scalf.SetConstantK(self.fluidDelta)
      scalf.Update()
      velocList[dim] = scalf.GetOutput()

    # Reset delta for next iteration if we used an impulse
    if isArrowUsed:
      self.fluidDelta = 0.0
        
    #smallDef = [None, None, None]
    #for dim in xrange(3):
    #  addf = vtk.vtkImageMathematics()
    #  addf.SetOperationToAdd()
    #  addf.SetInput1(self.identityMap[dim])
    #  addf.SetInput2(velocList[dim])
    #  addf.Update()
    #  smallDef[dim] = addf.GetOutput()

    self.displacement = self.composeDisplacements(self.displacement, velocList)
    
    warpedImage = self.warpImage(scaledMovingImage, self.displacement)
    #warpedImage = movingImage
    
    #warpedImage = self.warpImage(outputVolume.GetImageData(), velocList)
    
    self.updateOutputVolume(warpedImage)
        
    # appendf = vtk.vtkImageAppendComponents()
    # appendf.SetInput(velocList[0])
    # appendf.AddInput(velocList[1])
    # appendf.AddInput(velocList[2])
    # appendf.Update()
    
    # # TODO compound deformation H = H \circ V
    
    # gridTrafo = vtk.vtkGridTransform()
    # gridTrafo.SetDisplacementGrid(appendf.GetOutput())
    # gridTrafo.SetDisplacementScale(1.0)
    # gridTrafo.SetDisplacementShift(0.0)
    # gridTrafo.SetInterpolationModeToCubic()
    # gridTrafo.Update()
    
    # # TODO transform from original moving or current warped version?
    # reslice = vtk.vtkImageReslice()
    # reslice.SetInput(outputVolume.GetImageData())
    # reslice.SetResliceTransform(gridTrafo)
    # reslice.SetInterpolationModeToCubic()
    # reslice.SetOutputDimensionality(3)
    # reslice.SetOutputOrigin(fixedImage.GetOrigin())
    # reslice.SetOutputSpacing(fixedImage.GetSpacing())
    # reslice.SetOutputExtent(fixedImage.GetWholeExtent())
    # reslice.SetNumberOfThreads(8)
    # reslice.Update()
    
    # # greslice = vtk.vtkImageReslice()
    # # greslice.SetInput(self.gridVolume.GetImageData())
    # # greslice.SetResliceTransform(gridTrafo)
    # # greslice.SetInterpolationModeToNearestNeighbor()
    # # greslice.SetOutputDimensionality(3)
    # # greslice.SetOutputOrigin(fixedImage.GetOrigin())
    # # greslice.SetOutputSpacing(fixedImage.GetSpacing())
    # # greslice.SetOutputExtent(fixedImage.GetWholeExtent())
    # # greslice.SetNumberOfThreads(8)
    # # greslice.Update()
    
    # print "Resliced"
    
    # self.updateOutputVolume( reslice.GetOutput() )
    
    # # self.gridVolume.GetImageData().GetPointData().SetScalars(
      # # greslice.GetOutput().GetPointData().GetScalars() )
    # # self.gridVolume.GetImageData().GetPointData().GetScalars().Modified()
    # # self.gridVolume.GetImageData().Modified()
    # # self.gridVolume.Modified()

     

  def getMinMax(self, inputImage):
    histf = vtk.vtkImageHistogramStatistics()
    histf.SetInput(inputImage)
    histf.Update()
    
    imin = histf.GetMinimum()
    imax = histf.GetMaximum()
    
    return (imin, imax)

  def applyImageKernel(self, inputImage):

    # Force recomputation of range
    inputImage.Modified()
    inputImage.GetPointData().GetScalars().Modified()
  
    minmax0 = inputImage.GetScalarRange()
    min0 = minmax0[0]
    max0 = minmax0[1]
    range0 = max0 - min0
    
    if range0 <= 0.0:
      outImage = vtk.vtkImageData()
      outImage.SetScalarTypeToDouble()
      outImage.DeepCopy(inputImage)
      outImage.GetPointData().GetScalars().FillComponent(0, 0.0)
      return outImage
    
    spacing = inputImage.GetSpacing()
  
    gaussfLarge = vtk.vtkImageGaussianSmooth()
    gaussfLarge.SetInput(inputImage)
    gaussfLarge.SetNumberOfThreads(8)
    gaussfLarge.SetDimensionality(3)
    gaussfLarge.SetStandardDeviations(5.0 / spacing[0], 5.0 / spacing[1], 5.0 / spacing[2])
    gaussfLarge.SetRadiusFactor(3.0)
    gaussfLarge.Update()

    gaussfSmall = vtk.vtkImageGaussianSmooth()
    gaussfSmall.SetInput(inputImage)
    gaussfSmall.SetNumberOfThreads(8)
    gaussfSmall.SetDimensionality(3)
    gaussfSmall.SetStandardDeviations(1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2])
    gaussfSmall.SetRadiusFactor(3.0)
    gaussfSmall.Update()

    scalf = vtk.vtkImageMathematics()
    scalf.SetOperationToMultiplyByK()
    scalf.SetInput1(gaussfLarge.GetOutput())
    scalf.SetConstantK(self.logic.viscosity)
    scalf.Update()

    addf = vtk.vtkImageMathematics()
    addf.SetOperationToAdd()
    addf.SetInput1(gaussfSmall.GetOutput())
    addf.SetInput2(scalf.GetOutput())
    addf.Update()
    
    #DEBUG: avoid memory leaks when returning filter output
    #filteredImage = addf.GetOutput()
    filteredImage = vtk.vtkImageData()
    filteredImage.DeepCopy(addf.GetOutput())
    
    return filteredImage
    
  def normalizeImage(self, inputImage):

    castf = vtk.vtkImageCast()
    castf.SetOutputScalarTypeToDouble()
    castf.SetInput(inputImage)
    castf.Update()

    castImage = castf.GetOutput()
    
    minmax = castImage.GetScalarRange()
    range = minmax[1] - minmax[0]
    
    if range <= 0.0:
      outImage = vtk.vtkImageData()
      outImage.DeepCopy(castImage)
      outImage.GetPointData().GetScalars().FillComponent(0, 0.0)
      return outImage
    
    normf = vtk.vtkImageShiftScale()
    normf.SetInput(castImage)
    normf.SetShift(-minmax[0])
    normf.SetScale(1.0 / range)
    normf.Update()
    
    #return normf.GetOutput()
    scaledImage = vtk.vtkImageData()
    scaledImage.DeepCopy(normf.GetOutput())

    return scaledImage
    
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
  """ Implement a template matching optimizer that is
  integrated with the slicer main loop.
  """

  def __init__(self,fixed=None,moving=None,transform=None):
    self.interval = 1000
    self.timer = None

    # parameter defaults
    self.regIteration = 10
    self.viscosity = 50.0

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

#TODO
    self.movingArrowActor = vtk.vtkActor()
    
    self.lastHoveredGradMag = 0
    
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

    if self.sliceWidgetsPerStyle.has_key(observee):
      sliceWidget = self.sliceWidgetsPerStyle[observee]
      style = sliceWidget.sliceView().interactorStyle()
      self.interactor = style.GetInteractor()
      nodeIndex = self.nodeIndexPerStyle[observee]
      sliceNode = self.sliceNodePerStyle[observee]

      windowSize = sliceNode.GetDimensions()
      windowW = windowSize[0]
      windowH = windowSize[1]

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
      
        if self.lastHoveredGradMag > 0.1:
          cursor = qt.QCursor(qt.Qt.ClosedHandCursor)
          app.setOverrideCursor(cursor)
        
          xy = style.GetInteractor().GetEventPosition()
          xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
          ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

          self.startEventPosition = ras
        
          self.actionState = "arrowStart"
        
          self.arrowStartXY = xy
          self.arrowStartRAS = ras
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
          if (g > 0.1):
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


    


