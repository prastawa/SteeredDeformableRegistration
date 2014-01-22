import pyopencl as cl
import pyopencl.array as cla
import numpy as n

for platform in cl.get_platforms():
  for device in platform.get_devices():
   print "OpenCL platform %s device: %s" % (str(platform), cl.device_type.to_string(device.type))

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

shape = (4, 5, 3)

a = n.zeros(shape, dtype=n.float32)
a_array = cla.to_device(queue, a)

b = n.zeros((1,), dtype=n.float32)
b[0] = 22.5;
b_array = cla.to_device(queue, b)

prg = cl.Program(ctx, """
    __kernel void test1(__global float* a, __global float* q) {
      size_t column = get_global_id(2);
      size_t row = get_global_id(1);
      size_t slice = get_global_id(0);

      size_t sizeX = 4;
      size_t sizeY = 5;
      size_t sizeZ = 3;

      if (slice < sizeX && row < sizeY && column < sizeZ)
      {
        size_t offset = slice*sizeY*sizeZ + row*sizeZ + column;
        a[offset] = offset + q[0];
      }
    }
    """).build()

event = prg.test1(queue, shape, None, a_array.data, b_array.data)
event.wait()

a = a_array.get()
print(a)

a = n.random.normal(0, 1.0, size=shape)
a = n.float32(a)
a_array = cla.to_device(queue, a)

"""
# Original SlicerCL code
    inPath = os.path.dirname(slicer.modules.growcutcl.path) + "/GrowCutCL.cl.in"
    fp = open(inPath)
    sourceIn = fp.read()
    fp.close()

    slices, rows, columns = self.shape
    source = sourceIn % {
        'slices' : slices,
        'rows' : rows,
        'columns' : columns,
        }
    self.clProgram = pyopencl.Program(self.clContext, source).build()
    self.pendingEvent = None
"""

fp = open("../ImageFunctions.cl.in")
sourceIn = fp.read()
fp.close()

slices, rows, columns = shape
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

volumeCLProgram = cl.Program(ctx, source).build()

out1 = n.zeros(shape, dtype=n.float32)
out1_array = cla.to_device(queue, out1)
out2 = n.zeros(shape, dtype=n.float32)
out2_array = cla.to_device(queue, out2)
out3 = n.zeros(shape, dtype=n.float32)
out3_array = cla.to_device(queue, out3)


event = volumeCLProgram.gradient(queue, shape, None, a_array.data, out1_array.data, out2_array.data, out3_array.data)
event.wait()

print("Grad x")
out1 = out1_array.get()
print(out1)
print("Grad y")
out2 = out2_array.get()
print(out2)
print("Grad z")
out3 = out3_array.get()
print(out3)

print("CPU a+a")
print(a+a)

volumeCLProgram.add(queue, shape, None, a_array.data, a_array.data, out1_array.data).wait()
print("CL a+a")
out1 = out1_array.get()
print(out1)

print "Copy array"
q_array = a_array.copy()
volumeCLProgram.add(queue, shape, None, q_array.data, q_array.data, out2_array.data).wait()
print("CL q+q")
out2 = out2_array.get()
print(out2)

var = n.zeros((1,), dtype=n.float32)
var[0] = 0.5
var_array = cla.to_device(queue, var)
width = n.zeros((1,), dtype=n.int32)
width[0] = 3
width_array = cla.to_device(queue, width)

print("a")
print(a)

volumeCLProgram.gaussian(queue, shape, None, a_array.data, var_array.data, width_array.data, out1_array.data).wait()
print("gaussian a")
out1 = out1_array.get()
print(out1)


