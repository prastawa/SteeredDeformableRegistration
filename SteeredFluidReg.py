# Based on work by Steve Pieper, Kunlin Cao, Dirk Padfield

#from __main__ import vtk, qt, ctk, slicer
import vtk, qt, ctk, slicer
import math
import threading
import SimpleITK as sitk
import sitkUtils

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

    # kernel size
    self.kernelSizeSlider = ctk.ctkSliderWidget()
    self.kernelSizeSlider.decimals = 2
    self.kernelSizeSlider.singleStep = 0.5
    self.kernelSizeSlider.minimum = 0.5
    self.kernelSizeSlider.maximum = 50.0
    self.kernelSizeSlider.toolTip = "Scale of deformation."
    optFormLayout.addRow("Fluid kernel size:", self.kernelSizeSlider)

    # get default values from logic
    self.regIterationSlider.value = self.logic.regIteration
    self.kernelSizeSlider.value = self.logic.kernelSize

    #print(self.logic.regIteration)

    sliders = (self.regIterationSlider, self.kernelSizeSlider)
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
    if (os.getenv('USERNAME') == '200009249') or (os.getenv('USER') == 'dirkpadfield'):
      self.logic.testingData()
      self.fixedSelector.setCurrentNode(slicer.util.getNode('MRHead'))
      self.movingSelector.setCurrentNode(slicer.util.getNode('neutral'))
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
    self.logic.kernelSize = self.kernelSizeSlider.value
 
          
  def onResetButtonToggled(self):
    self.logic.actionState = "identity"
    # TODO: set momentas to zero, copy moving to output
   
  def onStart(self,checked):
    if checked:
      self.regButton.text = "Stop"
      
      # Set up display of volumes as composites, and create output node if not specified
      fixedVolume = self.fixedSelector.currentNode()
      movingVolume = self.movingSelector.currentNode()
      outputVolume = self.outputSelector.currentNode()
      
      if outputVolume is None:
        vl = slicer.modules.volumes.logic()
        outputVolume = vl.CloneVolume(slicer.mrmlScene, movingVolume, "steered-warped")
        self.outputSelector.setCurrentNode(outputVolume)
      else:
        # TODO DEBUG
        movingArray = slicer.util.array(movingVolume.GetName())
        outputArray = slicer.util.array(outputVolume.GetName())
        outputArray[:] = movingArray[:]
        
      compositeNodes = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')
      for compositeNode in compositeNodes.values():
        compositeNode.SetBackgroundVolumeID(fixedVolume.GetID())
        compositeNode.SetForegroundVolumeID(outputVolume.GetID())
        # TODO DEBUG
        #compositeNode.SetForegroundOpacity(1.0)
        compositeNode.SetForegroundOpacity(0.5)
        
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
    
    
  def startDeformableRegistration(self):     
    fixedVolume = self.fixedSelector.currentNode()
    movingVolume = self.movingSelector.currentNode()
    #outputVolume = self.outputSelector.currentNode()
    # initialTransform = self.initialTransformSelector.currentNode()
    # outputTransform = self.transformSelector.currentNode()
    
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

    print('registration begin')
    print "return result every %d iterations" %(self.regIterationSlider.value)
    
    self.interval = 1000
    self.registrationIterationNumber = 0;
    qt.QTimer.singleShot(self.interval, self.updateStep)       

  def updateStep(self):
    self.registrationIterationNumber = self.registrationIterationNumber + 1
    #print('Registering iteration %d' %(self.registrationIterationNumber))
    
    self.threadLock.acquire()
    self.fluidUpdate()
    self.threadLock.release()

    self.logic.redrawSlices()

    # Initiate another iteration of the registration algorithm.
    if self.logic.interaction:
      qt.QTimer.singleShot(self.interval, self.updateStep)    

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
    # TODO should do this in setup phase
    fixcastf = vtk.vtkImageCast()
    fixcastf.SetOutputScalarTypeToDouble()
    fixcastf.SetInput(fixedImage)
    fixcastf.Update()
    fixedImage = self.normalizeImage( fixcastf.GetOutput() )

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
    
    for dim in xrange(3):
      extractf = vtk.vtkImageExtractComponents()
      extractf.SetComponents(dim)
      extractf.SetInput(gradf.GetOutput())
      extractf.Update()
      self.momentas[dim] = extractf.GetOutput()
      
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
      mulf.SetInput2(self.momentas[dim])
      mulf.Update()
      self.momentas[dim] = mulf.GetOutput()
    
    # Add user inputs to momentum vectors
    # User defined impulses are in arrow queue containing xy, RAS, slice widget
    if not self.logic.arrowQueue.empty():
    
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
  
      print "Add to momentas"
      
      startIJK = movingRAStoIJK.MultiplyPoint(startRAS + (1,))
      endIJK = movingRAStoIJK.MultiplyPoint(endRAS + (1,))
                
      forceVector = [0, 0, 0]
      forceCenter = [0, 0, 0]
      
      for dim in xrange(3):
        forceCenter[dim] = round(startIJK[dim])
        # TODO automatically determine magnitude from the gradient update (balanced?)
        forceVector[dim] = (endRAS[dim] - startRAS[dim]) * 0.2

      print "forceVector = " + str(forceVector)
               
      #TODO real splatting? need an area of effect instead of point impulse
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
            print "set pos"
            print pos
            for dim in xrange(3):
              a = self.momentas[dim].GetScalarComponentAsDouble(pos[0], pos[1], pos[2], 0)
              self.momentas[dim].SetScalarComponentFromDouble(pos[0], pos[1], pos[2], 0,
                a + forceVector[dim])
             
      # for dim in xrange(3):
        # a = self.momentas[dim].GetScalarComponentAsDouble(forceCenter[0], forceCenter[1], forceCenter[2], 0)
        # self.momentas[dim].SetScalarComponentFromDouble(forceCenter[0], forceCenter[1], forceCenter[2], 0,
          # a + forceVector[dim])
              
    velocList = []
    for dim in xrange(3):
      velocList.append( self.applyImageKernel(self.momentas[dim]) )
      
      print "v range = " + str(velocList[dim].GetScalarRange())
      
    # Compute max velocity
    sqf = vtk.vtkImageMathematics()
    sqf.SetOperationToSquare()
    sqf.SetInput1(velocList[0])
    sqf.Update()
    
    velocMagImage = sqf.GetOutput()
    
    for dim in xrange(1,3):
      print " v sq dim " + str(dim)
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
    
    if maxVeloc > 2.0:
      for dim in xrange(3):
        scalf = vtk.vtkImageMathematics()
        scalf.SetOperationToMultiplyByK()
        scalf.SetInput1(velocList[dim])
        scalf.SetConstantK(2.0 / maxVeloc)
        scalf.Update()
        velocList[dim] = scalf.GetOutput()
        
    appendf = vtk.vtkImageAppendComponents()
    appendf.SetInput(velocList[0])
    appendf.AddInput(velocList[1])
    appendf.AddInput(velocList[2])
    appendf.Update()
    
    # TODO compound deformation H = H \circ V
    
    gridTrafo = vtk.vtkGridTransform()
    gridTrafo.SetDisplacementGrid(appendf.GetOutput())
    gridTrafo.SetDisplacementScale(1.0)
    gridTrafo.SetDisplacementShift(0.0)
    gridTrafo.SetInterpolationModeToCubic()
    gridTrafo.Update()
    
    # TODO transform from original moving or current warped version?
    reslice = vtk.vtkImageReslice()
    reslice.SetInput(outputVolume.GetImageData())
    reslice.SetResliceTransform(gridTrafo)
    reslice.SetInterpolationModeToCubic()
    reslice.SetOutputDimensionality(3)
    reslice.SetOutputOrigin(fixedImage.GetOrigin())
    reslice.SetOutputSpacing(fixedImage.GetSpacing())
    reslice.SetOutputExtent(fixedImage.GetWholeExtent())
    reslice.SetNumberOfThreads(8)
    reslice.Update()
    
    print "Resliced"
    
    outputVolume.GetImageData().GetPointData().GetScalars().DeepCopy(
      reslice.GetOutput().GetPointData().GetScalars() )
    #outputVolume.GetImageData().GetPointData().SetScalars( reslice.GetOutput().GetPointData().GetScalars() )
    outputVolume.GetImageData().GetPointData().GetScalars().Modified()
    outputVolume.GetImageData().Modified()
    outputVolume.Modified()
     

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
    
    print "min0 = %f, max0 = %f" % (min0, max0)
    
    spacing = inputImage.GetSpacing()
  
    gaussf = vtk.vtkImageGaussianSmooth()
    gaussf.SetInput(inputImage)
    gaussf.SetNumberOfThreads(8)
    gaussf.SetDimensionality(3)
    gaussf.SetStandardDeviations(4.0 / spacing[0], 4.0 / spacing[1], 4.0 / spacing[2])
    gaussf.SetRadiusFactor(3.0)
    gaussf.Update()
    
    filteredImage = self.normalizeImage(gaussf.GetOutput())
    
    mulf = vtk.vtkImageMathematics()
    mulf.SetOperationToMultiplyByK()
    mulf.SetInput1(filteredImage)
    mulf.SetConstantK(range0)
    mulf.Update()
    
    addf = vtk.vtkImageMathematics()
    addf.SetOperationToAddConstant()
    addf.SetInput1(mulf.GetOutput())
    addf.SetConstantC(min0)
    addf.Update()
    
    filteredImage = addf.GetOutput()
    
    minmax1 = filteredImage.GetScalarRange()
    min1 = minmax1[0]
    max1 = minmax1[1]
    print "min1 = %f, max1 = %f" % (min1, max1)
    
    return filteredImage
    
    # fft = vtk.vtkImageFFT()
    # fft.SetInput( self.normalizeImage(inputImage) )
    # #fft.SetInput(inputImage)
    # fft.Update()
    
    # bwf = vtk.vtkImageButterworthLowPass()
    # bwf.SetInput(fft.GetOutput())
    # bwf.SetCutOff(0.001) # TODO determine from image info
    # bwf.Update()
    
    # rfft = vtk.vtkImageRFFT()
    # rfft.SetInput(bwf.GetOutput())
    # rfft.Update()
    
    # realf = vtk.vtkImageExtractComponents()
    # realf.SetInput(rfft.GetOutput())
    # realf.SetComponents(0)
    # realf.Update()
     
    # filteredImage = self.normalizeImage(realf.GetOutput())
    
    # mulf = vtk.vtkImageMathematics()
    # mulf.SetOperationToMultiplyByK()
    # mulf.SetInput1(filteredImage)
    # mulf.SetConstantK(range0)
    # mulf.Update()
    
    # addf = vtk.vtkImageMathematics()
    # addf.SetOperationToAddConstant()
    # addf.SetInput1(mulf.GetOutput())
    # addf.SetConstantC(min0)
    # addf.Update()
    
    # filteredImage = addf.GetOutput()
    
    # filteredImage.Modified()
    # filteredImage.GetPointData().GetScalars().Modified()
    
    # minmax1 = filteredImage.GetScalarRange()
    # min1 = minmax1[0]
    # max1 = minmax1[1]
    # print "min1 = %f, max1 = %f" % (min1, max1)

    # return filteredImage
    
  def normalizeImage(self, inputImage):
    # Force recomputation of range
    inputImage.Modified()
    inputImage.GetPointData().GetScalars().Modified()
    
    minmax = inputImage.GetScalarRange()
    range = minmax[1] - minmax[0]
    
    if range <= 0.0:
      outImage = vtk.vtkImageData()
      outImage.DeepCopy(inputImage)
      outImage.GetPointData().GetScalars().FillComponent(0, 0.0)
      return outImage
    
    normf = vtk.vtkImageShiftScale()
    normf.SetInput(inputImage)
    normf.SetShift(-minmax[0])
    normf.SetScale(1.0 / range)
    normf.Update()
    
    return normf.GetOutput()

  def composeDeformation(self, F, G):
    # TODO return F \circ G
    pass
  
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
  Note: currently depends on numpy/scipy installation in mac system
  """

  def __init__(self,fixed=None,moving=None,transform=None):
    self.interval = 10
    self.timer = None

    # parameter defaults
    self.regIteration = 10
    self.kernelSize = 5

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
    
    self.lineActor = vtk.vtkActor2D()
    
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
          rencol.GetItemAsObject(1).RemoveActor(self.lineActor)

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
        
        events = ("LeftButtonPressEvent","LeftButtonReleaseEvent","MouseMoveEvent", "KeyPressEvent","EnterEvent", "LeaveEvent")
        for event in events:
          tag = self.interactor.AddObserver(event, self.processEvent, 1.0)
          self.interactorObserverTags.append(tag)
          


  def processEvent(self,observee,event=None):

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
  
      if event == "LeftButtonPressEvent":
        xy = style.GetInteractor().GetEventPosition()
        xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
        ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

        self.startEventPosition = ras
        
        self.actionState = "arrowStart"
        
        self.arrowStartXY = xy
        self.arrowStartRAS = ras

        #print(self.actionState)
        self.abortEvent(event)

      elif event == "LeftButtonReleaseEvent":

        xy = style.GetInteractor().GetEventPosition()
        xyz = sliceWidget.sliceView().convertDeviceToXYZ(xy)
        ras = sliceWidget.sliceView().convertXYZToRAS(xyz)

        self.lastEventPosition = ras
        
        self.actionState = "arrowEnd"

        self.arrowEndXY = xy
        self.arrowEndRAS = ras

        # TODO: only draw arrow within ??? seconds
        self.lastDrawMTime = sliceNode.GetMTime()
        
        self.lastDrawSliceWidget = sliceWidget

        self.arrowQueue.put(
          (sliceNode.GetMTime(), sliceWidget, self.arrowStartXY, self.arrowEndXY, self.arrowStartRAS, self.arrowEndRAS) )
        
        
        #print(self.actionState)
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
          rencol.GetItemAsObject(1).RemoveActor(self.lineActor)
          
        renwin.Render()
        
    if not self.arrowQueue.empty():
    
      pts = vtk.vtkPoints()
      
      lines = vtk.vtkCellArray()
      
      for i in xrange(self.arrowQueue.qsize()):
        arrowTuple = self.arrowQueue.queue[i]
        sliceWidget = arrowTuple[1]
        startXY = arrowTuple[2]
        endXY = arrowTuple[3]

        # TODO assume ortographic projection, no need for inv(camera)
        p = pts.InsertNextPoint(startXY + (0,))
        q = pts.InsertNextPoint(endXY + (0,))
      
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, p);
        line.GetPointIds().SetId(1, q);
        
        lines.InsertNextCell(line)
      
      pd = vtk.vtkPolyData()
      pd.SetPoints(pts)
      pd.SetLines(lines)
      
      mapper = vtk.vtkPolyDataMapper2D()
      mapper.SetInput(pd)
      
      self.lineActor.SetMapper(mapper)
      self.lineActor.GetProperty().SetColor([1.0, 0.0, 0.0])    
      
      renwin = self.lastDrawnSliceWidget.sliceView().renderWindow()
      rencol = renwin.GetRenderers()
      
      if rencol.GetNumberOfItems() == 2:
        print "Old r2"
        renOverlay = rencol.GetItemAsObject(1)
      else:
        print "New r2"
        renOverlay = vtk.vtkRenderer()
        renwin.SetNumberOfLayers(2)
        renwin.AddRenderer(renOverlay)

      renOverlay.AddActor(self.lineActor)
      renOverlay.SetInteractive(0)
      renOverlay.SetLayer(1)
      #renOverlay.ResetCamera()

      renwin.Render()

  def testingData(self):
    """Load some default data for development
    and set up a transform and viewing scenario for it.
    """
    if not slicer.util.getNodes('MRHead*'):
      import os
      fileName = "/home/src/NAMIC/MR-head.nrrd"
      vl = slicer.modules.volumes.logic()
      volumeNode = vl.AddArchetypeVolume(fileName, "MRHead", 0)
    if not slicer.util.getNodes('neutral*'):
      import os
      fileName = "/home/src/NAMIC/helloPython/data/spgr.nhdr"
      vl = slicer.modules.volumes.logic()
      volumeNode = vl.AddArchetypeVolume(fileName, "neutral", 0)
    if not slicer.util.getNodes('movingToFixed*'):
      # Create transform node
      transform = slicer.vtkMRMLLinearTransformNode()
      transform.SetName('movingToFixed')
      slicer.mrmlScene.AddNode(transform)
    head = slicer.util.getNode('MRHead')
    neutral = slicer.util.getNode('neutral')
    transform = slicer.util.getNode('movingToFixed')
    ###
    # neutral.SetAndObserveTransformNodeID(transform.GetID())
    ###
    compositeNodes = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')
    for compositeNode in compositeNodes.values():
      compositeNode.SetBackgroundVolumeID(head.GetID())
      compositeNode.SetForegroundVolumeID(neutral.GetID())
      compositeNode.SetForegroundOpacity(0.5)
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.FitSliceToAll()


    


