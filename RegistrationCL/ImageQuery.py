
from ImageCL import ImageCL

import numpy as np
import vtk

class ImageQuery:

  def __init__(self, fixedCL, movingCL):
    self.fixedCL = fixedCL
    self.movingCL = movingCL

    self.panelSize = 128

  def gaussian(self, shape):

    # TODO: sigma
    # TODO: consolidate with the one in polyaffine class?

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

  def get_frame(self, image):

    minI = np.min(image)
    maxI = np.max(image)
    rangeI = maxI - minI
    if rangeI > 0.0:
      image = (image - minI) / rangeI

    #image = image * self.gaussian(image.shape)

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

    # Match directions to image gradients
    # Maximize \sum_x dot(e, grad_I(x))^2 + dot(f, grad_I(x))^2

    Ix, Iy, Iz = np.gradient(image)

    sum_Ix = np.sum(Ix)
    sum_Iy = np.sum(Iy)
    sum_Iz = np.sum(Iz)

    obj_ref = np.sum( (e[0]*Ix + e[1]*Iy + e[2]*Iz) ** 2 +
      (f[0]*Ix + f[1]*Iy + f[2]*Iz) ** 2 )

    obj = obj_ref

    for iter in range(20):
      print "View obj", obj

      obj_prev = obj

      dot_e = np.sum( e[0]*Ix + e[1]*Iy + e[2]*Iz )
      grad_e = np.array(
        [np.sum(dot_e*Ix), np.sum(dot_e*Iy), np.sum(dot_e*Iz)], np.single )

      dot_f = np.sum( f[0]*Ix + f[1]*Iy + f[2]*Iz )
      grad_f = np.array(
        [np.sum(dot_f*Ix), np.sum(dot_f*Iy), np.sum(dot_f*Iz)], np.single )

      if iter == 0:
        stepE = 0.1 / np.max(np.abs(grad_e))
        stepF = 0.1 / np.max(np.abs(grad_f))

      for lineIter in range(20):
        e_test = e + stepE * grad_e
        f_test = f + stepF * grad_f

        e_test /= np.linalg.norm(e_test)
        f_test /= np.linalg.norm(f_test)

        obj_test = np.sum( (e_test[0]*Ix + e_test[1]*Iy + e_test[2]*Iz) ** 2 +
          (f_test[0]*Ix + f_test[1]*Iy + f_test[2]*Iz) ** 2 )

        if obj_test > obj:
          obj = obj_test
          e = e_test
          f = f_test
          stepE *= 1.2
          stepF *= 1.2
          break
        else:
          stepE *= 0.5
          stepF *= 0.5

      if abs(obj-obj_prev) < 1e-4 * abs(obj-obj_ref):
        break

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
    fixedFrame = self.get_frame(self.fixedCL.clarray.get())
    movingFrame = self.get_frame(self.movingCL.clarray.get())

    N = self.panelSize

    fixedShape = self.fixedCL.shape

    XC = [0, 0, 0]
    for d in range(3):
      XC[d] = fixedShape[d] / 2.0

    #
    # Create panel of views for fixed and moving image, with likely optimal
    # orientations
    #
    fixedArray = np.zeros((N*3, N*3), np.single)
    movingArray = np.zeros((N*3, N*3), np.single)

    for r in range(3):
      for c in range(3):
        V = self.get_slicing_matrix(fixedFrame[:,r], movingFrame[:,c])

        A = np.zeros((4,4))
        A[0:3,0] = V[:,0]
        A[0:3,1] = V[:,1]
        A[0:3,2] = V[:,2]
        A[0,3] = XC[0]
        A[1,3] = XC[1]
        A[2,3] = XC[2]
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

    def normalize(arr):
      minv = arr.min()
      maxv = arr.max()
      rangev = maxv - minv
      if rangev <= 0.0:
        return arr
      return (arr - minv) / rangev

    fixedArray = normalize(fixedArray)
    movingArray = normalize(movingArray)

    ren = vtk.vtkRenderer()

    renWin = vtk.vtkRenderWindow()
    renWin.SetWindowName("Image Alignment Query")
    renWin.SetSize(800, 800)

    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    dataImporter = vtk.vtkImageImport()
    fixedArray_vtkorder = np.asfortranarray(fixedArray)
    fixedArray_vtkorder = fixedArray.transpose()
    ##dataImporter.CopyImportVoidPointer(fixedArray_vtkorder, fixedArray_vtkorder.nbytes)
    #dataImporter.CopyImportVoidPointer(fixedArray, fixedArray.nbytes)
    displayArray = np.uint8(fixedArray * 255)
    dataImporter.CopyImportVoidPointer(displayArray, displayArray.nbytes)
    #dataImporter.SetDataScalarTypeToFloat()
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, N*3-1, 0, N*3-1, 0, 0)
    dataImporter.SetWholeExtent(0, N*3-1, 0, N*3-1, 0, 0)
    dataImporter.Update()

    imageActor = vtk.vtkImageActor()
    imageActor.SetInput(dataImporter.GetOutput())
    imageActor.SetPosition(100.0, 100.0, 0.0)
    imageActor.SetZSlice(0)
    imageActor.PickableOn()

    imageStyle = vtk.vtkInteractorStyleImage()
    iren.SetInteractorStyle(imageStyle)

    ren.AddActor(imageActor)

    def slider_callback(obj, event):
      # Get slider value
      alpha = obj.GetRepresentation().GetValue()

      print "Slider value", alpha

      displayArray = fixedArray * alpha + movingArray * (1.0 - alpha)

      #minI = displayArray.min()
      #maxI = displayArray.max()
      #displayArray = (displayArray - minI) / (maxI - minI) * 255
      #displayArray = np.uint8(displayArray)
      #displayArray = np.uint8(normalize(displayArray) * 255)
      displayArray = np.uint8(displayArray * 255)

      dataImporter.CopyImportVoidPointer(displayArray, displayArray.nbytes)

    sliderRep  = vtk.vtkSliderRepresentation2D()
    sliderRep.SetMinimumValue(0.0)
    sliderRep.SetMaximumValue(1.0)
    sliderRep.SetValue(1.0)
    #sliderRep.SetTitleText("Fixed vs moving")
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint1Coordinate().SetValue(0.2, 0.1)
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetValue(0.8, 0.1)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(iren)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.AddObserver("InteractionEvent", slider_callback)
    sliderWidget.EnabledOn()

    #ren.InteractiveOff()
    renWin.AddRenderer(ren)

    picker = vtk.vtkPropPicker()
    #picker = vtk.vtkWorldPointPicker()
    picker.PickFromListOn()
    picker.AddPickList(imageActor)

    self.queryRowColumn = (0, 0)

    def pick_callback(obj, event):
      mousePos = obj.GetEventPosition()

      picker.PickProp(mousePos[0], mousePos[1], ren)

      p = picker.GetPickPosition()
      c = round( (p[0]-100) / (3*N) * 2 )
      r = round( (p[1]-100) / (3*N) * 2 )

      if r < 0 or r >= 3 or c < 0 or c >= 3:
        print "Outside"
        return

      print "Image row", r, "col", c

      self.queryRowColumn = (r, c)

      iren.GetRenderWindow().Finalize()
      iren.TerminateApp()

    iren.SetPicker(picker)
    iren.AddObserver("LeftButtonPressEvent", pick_callback)

    renWin.Render()
    iren.Start()

    # Return selection of a view in panel
    r, c = self.queryRowColumn

    fixedSlice = fixedArray[r*N:(r+1)*N, c*N:(c+1)*N]
    movingSlice = movingArray[r*N:(r+1)*N, c*N:(c+1)*N]

    return fixedSlice, movingSlice, fixedFrame[:,r], movingFrame[:,c]
