
import ImageCl

import numpy as np
import vtk

class ImageQuery:

  def __init__(self, fixedCL, movingCL):
    self.fixedCL = fixedCL
    self.movingCL = movingCL

    self.panelSize = 128

  def get_frame(self, image):

    minI = np.min(image)
    maxI = np.max(image)
    rangeI = maxI - minI
    if rangeI > 0.0:
      image = (image - minI) / rangeI

    image = image * gaussian(image.shape)

    # TODO: gradient at scale?
    #Ix, Iy, Iz = np.gradient(image)
    #image = Ix * Ix + Iy * Iy + Iz * Iz

    """
    simage = sitk.GetImageFromArray(image)
  
    fimage = sitk.SmoothingRecursiveGaussian(simage, sigma=1.0)
    #fimage = sitk.DiscreteGaussian(image, 1.0)

    Ix = sitk.Derivative(fimage, 0, 1)
    Iy = sitk.Derivative(fimage, 1, 1)
    Iz = sitk.Derivative(fimage, 2, 1)

    image = Ix * Ix + Iy * Iy + Iz * Iz
    image = sitk.GetArrayFromImage(image)
    minI = np.min(image)
    maxI = np.max(image)
    image = (image - minI) / (maxI - minI + 1e-4)
    """

    Xtuple = np.where(np.abs(image) > 0)
    #Xtuple = np.where(image > 0.5)
    X = np.vstack(Xtuple).T

    Y = image[Xtuple]
  
    Z = np.array(X)
    for j in xrange(X.shape[1]):
      Z[:,j] = X[:,j] * Y
      #Z[:,j] = X[:,j]
    # TODO: robust mean, cov from MCD?
    #Z = Z - np.mean(Z, axis=0)
    #Z = Z / np.std(Z, axis=0)

    print "Z shape", Z.shape

    S = np.dot(Z.T, Z)
    #mu, S = fastmcd(Z)

    #U, s, V = np.linalg.svd(M, full_matrices=False)
    #U, s, V = np.linalg.svd(Z.T, full_matrices=False)
    U, s, V = np.linalg.svd(S, full_matrices=False)

    """
    print "U = "
    print U
    print "V = "
    print V
    """

    #W, U = np.linalg.eig(M)

    rankU = np.linalg.matrix_rank(U)
    if rankU < 3:
      U = np.eye(3)

    e = U[:,0]
    f = U[:,1]

    Ix, Iy, Iz = np.gradient(image)

    for iter in range(20):
      obj = np.sum( (e[0]*Ix) ** 2 + (e[1]*Iy) ** 2 + (e[2]*Iz) ** 2 +
        (f[0]*Ix) ** 2 + (f[1]*Iy) ** 2 + (f[2]*Iz) ** 2 )

      print "View obj", obj

      grad_e = np.array([np.sum(e[0]*Ix), np.sum(e[1]*Iy), np.sum(e[2]*Iz)]) * 2
      grad_f = np.array([np.sum(f[0]*Ix), np.sum(f[1]*Iy), np.sum(f[2]*Iz)]) * 2

      if iter == 0:
        stepE = 0.1 / np.max(np.abs(grad_e))
        stepF = 0.1 / np.max(np.abs(grad_f))

      e = e + stepE * grad_e
      f = f + stepF * grad_f

      e /= np.linalg.norm(e)
      f /= np.linalg.norm(f)

    U[:,0] = e
    U[:,1] = f
    U[:,2] = np.cross(e,f)
  
    return U

  def get_slicing_matrix(self, a, b):
    """Get slicing matrix from two frame vectors a and b"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)

    print "Frame from "
    print a
    print b

    f1 = a
    f2 = np.cross(a, c)
    f3 = c

    f2 = f2 / np.linalg.norm(f2)
    f3 = f3 / np.linalg.norm(f3)

    F = np.zeros((3,3))
    F[:,0] = f1
    F[:,1] = f2
    F[:,2] = f3

    rankF = np.linalg.matrix_rank(F)
    if rankF < 3:
      F = np.eye(3)

    return F

  def display_and_query(self):

    # Convert images to VTK data structures
    fixedVTKImage = self.fixedCL.toVTKImage()
    movingVTKImage = self.movingCL.toVTKImage()

    # Viewing parameters
    fixedFrame = self.get_frame(self.fixedCL.get())
    movingFrame = self.get_frame(self.movingCL.get())

    N = self.panelSize

    fixedShape = self.fixedCL.shape

    C = [0, 0, 0]
    for d in range(3):
      C[d] = fixedShape[d] / 2.0

    #
    # Create panel of views for fixed and moving image, with likely optimal
    # orientations
    #
    fixedArray = np.zeros((N*3, N*3), np.float32)
    movingArray = np.zeros((N*3, N*3), np.float32)

    for r in range(3):
      for c in range(3):
        V = self.get_slicing_matrix(fixedFrame[:,r], movingFrame[:,c])

        A = np.zeros((4,4))
        A[0:3,0] = V[:,0]
        A[0:3,1] = V[:,1]
        A[0:3,2] = V[:,2]
        A[0,3] = C[0]
        A[1,3] = C[1]
        A[2,3] = C[2]
        A[3,3] = 1.0

        resliceAxes = vtk.vtkMatrix4x4()
        resliceAxes.DeepCopy(A.ravel().tolist())

        reslicef = vtk.vtkImageReslice()
        reslicef.SetInput(fixedVTKImage)
        reslicef.SetInformationInput(fixedVTKImage)
        reslicef.SetOutputExtent(0, N-1, 0, N-1, 0, 0)
        reslicef.SetOutputDimensionality(2)
        reslicef.SetResliceAxes(resliceAxes)
        reslicef.SetInterpolationModeToLinear()
        reslicef.Update()

        fixedSlice = vtk.util.numpy_support.vtk_to_numpy(
          reslicef.GetOutput().GetPointData().GetScalars() ).reshape(N, N)
        fixedSlice = np.transpose(fixedSlice)

        fixedArray[r*N:(r+1)*N, c*N:(c+1)*N] = fixedSlice

        resliceAxes = vtk.vtkMatrix4x4()
        resliceAxes.DeepCopy(A.ravel().tolist())

        reslicef = vtk.vtkImageReslice()
        reslicef.SetInput(movingVTKImage)
        reslicef.SetInformationInput(movingVTKImage)
        reslicef.SetOutputExtent(0, N-1, 0, N-1, 0, 0)
        reslicef.SetOutputDimensionality(2)
        reslicef.SetResliceAxes(resliceAxes)
        reslicef.SetInterpolationModeToLinear()
        reslicef.Update()

        movingSlice = vtk.util.numpy_support.vtk_to_numpy(
          reslicef.GetOutput().GetPointData().GetScalars() ).reshape(N, N)
        movingSlice = np.transpose(movingSlice)

        movingArray[r*N:(r+1)*N, c*N:(c+1)*N] = movingSlice

    #
    # Display panel of views with blending slider
    #
    fixedArray = normalize(fixedArray)
    movingArray = normalize(movingArray)

    ren = vtk.vtkRenderer()

    renWin = vtk.vtkRenderWindow()
    renWin.SetWindowName("Image Alignment Query")
    renWin.SetSize(800, 800)

    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(iren)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.AddObserver("InteractionEvent", slider_callback)
    sliderWidget.EnabledOn()

    dataImporter = vtk.vtkImageImport()
    displayArray_vtkorder = np.asfortranarray(displayArray)
    displayArray_vtkorder = displayArray.transpose()
    ##dataImporter.CopyImportVoidPointer(displayArray_vtkorder, displayArray_vtkorder.nbytes)
    #dataImporter.CopyImportVoidPointer(displayArray, displayArray.nbytes)
    displayArray = np.uint8(displayArray * 255)
    dataImporter.CopyImportVoidPointer(displayArray, displayArray.nbytes)
    #dataImporter.SetDataScalarTypeToFloat()
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, panelSize*3-1, 0, panelSize*3-1, 0, 0)
    dataImporter.SetWholeExtent(0, panelSize*3-1, 0, panelSize*3-1, 0, 0)
    dataImporter.Update()

    imageActor = vtk.vtkImageActor()
    imageActor.SetInput(dataImporter.GetOutput())
    imageActor.SetPosition(100.0, 100.0, 0.0)
    imageActor.SetZSlice(0)
    imageActor.PickableOn()

    imageStyle = vtk.vtkInteractorStyleImage()
    iren.SetInteractorStyle(imageStyle)

    ren.AddActor(imageActor)

    #ren.InteractiveOff()
    renWin.AddRenderer(ren)

    picker = vtk.vtkPropPicker()
    #picker = vtk.vtkWorldPointPicker()
    picker.PickFromListOn()
    picker.AddPickList(imageActor)

    def pick_callback(obj, event):
      global picker
      mousePos = obj.GetEventPosition()
      print "Mouse", mousePos
      picker.PickProp(mousePos[0], mousePos[1], ren)
      #picker.Pick(mousePos[0], mousePos[1], 0, ren)
      #print "Path", picker.GetPath()
      pickedActor = picker.GetActor()
      print "picked actor type=", type(pickedActor)
      #print "Image bounds", imageActor.GetPosition(), imageActor.GetPosition2()
      print "Select", picker.GetSelectionPoint()
      print "Pick", picker.GetPickPosition()

      p = picker.GetPickPosition()
      c = round( (p[0]-100) / (3*panelSize) * 2 )
      r = round( (p[1]-100) / (3*panelSize) * 2 )

      if r < 0 or r >= 3 or c < 0 or c >= 3:
        print "Outside"
      else:
        print "Image row", r, "col", c

      self.queryRowColumn = (r, c)

      iren.GetRenderWindow.Finalize()
      iren.TerminateApp()


    iren.SetPicker(picker)
    iren.AddObserver("LeftButtonPressEvent", pick_callback)

    renWin.Render()
    iren.Start()


    # Return selection of a view in panel
    return fixedSlice[], movingSlice, fixedFrame[:,r], movingFrame[:,c]
