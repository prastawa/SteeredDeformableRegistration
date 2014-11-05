
// TODO image origin spacing and orientation
// How to store orientation matrix? pass it through device array? or pass it
// through #def constants? more recompilations? is this OK?

#define NUM_GAUSSIAN_STEPS 4

//
// Gaussian filtering
//

// Use float array to store var and width due to PyOpenCL issue (???)
__kernel void gaussian_x(
  __global float* src,
  __global uint* size,
  float var, int width,
  __global float* dst)
{
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1] || column >= size[2])
    return;

  size_t offset = slice*size[1]*size[2] + row*size[2] + column;

  size_t kernelSize = convert_uint(width);
  size_t halfKernelSize = kernelSize / 2;

  size_t slice0 = slice - halfKernelSize;
  if (slice0 >= size[0]) slice0 = 0;

  size_t slice1 = slice + halfKernelSize;
  if (slice1 >= size[0]) slice1 = size[0] - 1;

  float fslice = convert_float(slice);

  float wv = 0.0;
  float n_weight = 0.0;

  for (size_t pslice = slice0; pslice <= slice1; pslice++)
  {
    float d = convert_float(pslice) - fslice;
    float g = exp(-0.5 * d*d / var);

    size_t n_offset = pslice*size[1]*size[2] + row*size[2] + column;
    wv += src[n_offset] * g;

    n_weight += g;
  }

  if (n_weight > 0.0)
    wv /= n_weight;

  dst[offset] = wv;
}

__kernel void gaussian_y(
  __global float* src,
  __global uint* size,
  float var, int width,
  __global float* dst)
{
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1] || column >= size[2])
    return;

  size_t offset = slice*size[1]*size[2] + row*size[2] + column;

  size_t kernelSize = convert_uint(width);
  size_t halfKernelSize = kernelSize / 2;

  size_t row0 = row - halfKernelSize;
  if (row0 >= size[1]) row0 = 0;

  size_t row1 = row + halfKernelSize;
  if (row1 >= size[1]) row1 = size[1] - 1;

  float frow = convert_float(row);

  float wv = 0.0;
  float n_weight = 0.0;

  for (size_t prow = row0; prow <= row1; prow++)
  {
    float d = convert_float(prow) - frow;
    float g = exp(-0.5 * d*d / var);

    size_t n_offset = slice*size[1]*size[2] + prow*size[2] + column;
    wv += src[n_offset] * g;

    n_weight += g;
  }

  if (n_weight > 0.0)
    wv /= n_weight;

  dst[offset] = wv;
}

__kernel void gaussian_z(
  __global float* src,
  __global uint* size,
  float var, int width,
  __global float* dst)
{
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1] || column >= size[2])
    return;

  size_t offset = slice*size[1]*size[2] + row*size[2] + column;

  size_t kernelSize = convert_uint(width);
  size_t halfKernelSize = kernelSize / 2;

  size_t column0 = column - halfKernelSize;
  if (column0 >= size[2]) column0 = 0;

  size_t column1 = column + halfKernelSize;
  if (column1 >= size[2]) column1 = size[2] - 1;

  float fcolumn = convert_float(column);

  float wv = 0.0;
  float n_weight = 0.0;

  for (size_t pcolumn = column0; pcolumn <= column1; pcolumn++)
  {
    float d = convert_float(pcolumn) - fcolumn;
    float g = exp(-0.5 * d*d / var);

    size_t n_offset = slice*size[1]*size[2] + row*size[2] + pcolumn;
    wv += src[n_offset] * g;

    n_weight += g;
  }

  if (n_weight > 0.0)
    wv /= n_weight;

  dst[offset] = wv;
}

// Gaussian filtering in x direction, in-place, sigma in voxels
__kernel void recursive_gaussian_x(
  __global float* img,
  __global uint* size,
  float sigma)
{
  size_t column = get_global_id(1);
  size_t row = get_global_id(0);

  if (row >= size[1] || column >= size[2])
    return;

  size_t sliceStart = row*size[2] + column;
  size_t sliceEnd = (size[0]-1)*size[1]*size[2] + row*size[2] + column;

  // NOTE: assume this is done outside
  //float ssigma = sigma / spacing[0];

  float lambda = (sigma*sigma) / (2.0 * convert_float(NUM_GAUSSIAN_STEPS));
  float nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda)) / (2.0*lambda);

  float boundary = (1.0 / (1.0 - nu));

  for (int step = 0; step < NUM_GAUSSIAN_STEPS; step++)
  {
    img[sliceStart] *= boundary;

    for (size_t slice = 1; slice < size[0]; slice++)
    {
      size_t pos = slice*size[1]*size[2] + sliceStart;
      size_t pos_prev = (slice-1)*size[1]*size[2] + sliceStart;
      img[pos] += img[pos_prev] * nu;
    }

    img[sliceEnd] *= boundary;

    for (size_t slice = (size[0]-1); slice > 0; slice--)
    {
      size_t pos = slice*size[1]*size[2] + sliceStart;
      size_t pos_prev = (slice-1)*size[1]*size[2] + sliceStart;
      img[pos_prev] += img[pos] * nu;
    }
  }

  // NOTE: do this outside to get full 3D parallelization
  // Ignore since scaling is not important?
  for (size_t slice = 0; slice < size[0]; slice++)
  {
    size_t pos = slice*size[1]*size[2] + sliceStart;
    img[pos] *= pow(nu / lambda, convert_float(NUM_GAUSSIAN_STEPS));
  }
}

// Gaussian filtering in y direction, in-place
__kernel void recursive_gaussian_y(
  __global float* img,
  __global uint* size,
  float sigma)
{
  size_t column = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || column >= size[2])
    return;

  // NOTE: assume this is done outside
  //float ssigma = sigma / spacing[1];

  size_t rowStart = slice*size[1]*size[2] + column;
  size_t rowEnd = slice*size[1]*size[2] + (size[1]-1)*size[2] + column;

  float lambda = (sigma*sigma) / (2.0 * convert_float(NUM_GAUSSIAN_STEPS));
  float nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda)) / (2.0*lambda);

  float boundary = (1.0 / (1.0 - nu));

  for (int step = 0; step < NUM_GAUSSIAN_STEPS; step++)
  {
    img[rowStart] *= boundary;

    for (size_t row = 1; row < size[1]; row++)
    {
      size_t pos = row*size[2] + rowStart;
      size_t pos_prev = (row-1)*size[2] + rowStart;
      img[pos] += img[pos_prev] * nu;
    }

    img[rowEnd] *= boundary;

    for (size_t row = (size[1]-1); row > 0; row--)
    {
      size_t pos = row*size[2] + rowStart;
      size_t pos_prev = (row-1)*size[2] + rowStart;
      img[pos_prev] += img[pos] * nu;
    }
  }

  // NOTE: do this outside to get full 3D parallelization
  for (size_t row = 0; row < size[1]; row++)
  {
    size_t pos = row*size[2] + rowStart;
    img[pos] *= pow(nu / lambda, convert_float(NUM_GAUSSIAN_STEPS));
  }
}

// Gaussian filtering in z direction, in-place
__kernel void recursive_gaussian_z(
  __global float* img,
  __global uint* size,
  float sigma)
{
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1])
    return;

  size_t columnStart = slice*size[1]*size[2] + row*size[2];
  size_t columnEnd = slice*size[1]*size[2] + row*size[2] + size[2]-1;

  float lambda = (sigma*sigma) / (2.0 * convert_float(NUM_GAUSSIAN_STEPS));
  float nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda)) / (2.0*lambda);

  float boundary = (1.0 / (1.0 - nu));

  for (int step = 0; step < NUM_GAUSSIAN_STEPS; step++)
  {
    img[columnStart] *= boundary;

    for (size_t column = 1; column < size[2]; column++)
    {
      size_t pos = column + columnStart;
      size_t pos_prev = (column-1) + columnStart;
      img[pos] += img[pos_prev] * nu;
    }

    img[columnEnd] *= boundary;

    for (size_t column = (size[2]-1); column > 0; column--)
    {
      size_t pos = column + columnStart;
      size_t pos_prev = (column-1) + columnStart;
      img[pos_prev] += img[pos] * nu;
    }
  }

  // NOTE: do this outside to get full 3D parallelization
  for (size_t column = 0; column < size[2]; column++)
  {
    size_t pos = column + columnStart;
    img[pos] *= pow(nu / lambda, convert_float(NUM_GAUSSIAN_STEPS));
  }
}

//
// Gradient using central finite difference
//

__kernel void gradient(
  __global float* src,
  __global uint* size,
  __global float* spacing,
  __global float* dst_x,
  __global float* dst_y,
  __global float* dst_z)
{
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1] || column >= size[2])
    return;

  size_t slice_a = slice - 1;
  size_t slice_b = slice + 1;
  size_t row_a = row - 1;
  size_t row_b = row + 1;
  size_t column_a = column - 1;
  size_t column_b = column + 1;

  if (slice_a >= size[0]) slice_a = 0;
  if (slice_b >= size[0]) slice_b = size[0] - 1;
  if (row_a >= size[1]) row_a = 0;
  if (row_b >= size[1]) row_b = size[1] - 1;
  if (column_a >= size[2]) column_a = 0;
  if (column_b >= size[2]) column_b = size[2] - 1;

  size_t offset = slice*size[1]*size[2] + row*size[2] + column;

  size_t offset_a_x = slice_a*size[1]*size[2] + row*size[2] + column;
  size_t offset_b_x = slice_b*size[1]*size[2] + row*size[2] + column;
  size_t offset_a_y = slice*size[1]*size[2] + row_a*size[2] + column;
  size_t offset_b_y = slice*size[1]*size[2] + row_b*size[2] + column;
  size_t offset_a_z = slice*size[1]*size[2] + row*size[2] + column_a;
  size_t offset_b_z = slice*size[1]*size[2] + row*size[2] + column_b;

  dst_x[offset] = 0.5 * (src[offset_b_x] - src[offset_a_x]) / spacing[0];
  dst_y[offset] = 0.5 * (src[offset_b_y] - src[offset_a_y]) / spacing[1];
  dst_z[offset] = 0.5 * (src[offset_b_z] - src[offset_a_z]) / spacing[2];
}

//
// Interpolation
//

__kernel void interpolate(
  __global float* src,
  __global uint* srcsize,
  __global float* srcspacing,
  __global float* hx,
  __global float* hy,
  __global float* hz,
  __global float* dst,
  __global uint* dstsize)
{
/*
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= dstsize[0] || row >= dstsize[1] || column >= dstsize[2])
    return;

  size_t dstpos = slice*dstsize[1]*dstsize[2] + row*dstsize[2] + column;
*/

  size_t ix = get_global_id(0);
  size_t iy = get_global_id(1);
  size_t iz = get_global_id(2);

  if (ix >= dstsize[0] || iy >= dstsize[1] || iz >= dstsize[2])
    return;

  //size_t dstpos = iz*dstsize[0]*dstsize[1] + iy*dstsize[0] + ix;
  size_t dstpos = ix*dstsize[1]*dstsize[2] + iy*dstsize[2] + iz;

  // Assume axial with zero origin for now
  float x = hx[dstpos] / srcspacing[0];
  float y = hy[dstpos] / srcspacing[1];
  float z = hz[dstpos] / srcspacing[2];

  int x0 = convert_int(x);
  int y0 = convert_int(y);
  int z0 = convert_int(z);

  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (z0 < 0) z0 = 0;
  if (x0 >= srcsize[0]) x0 = srcsize[0]-1;
  if (y0 >= srcsize[1]) y0 = srcsize[1]-1;
  if (z0 >= srcsize[2]) z0 = srcsize[2]-1;

  int x1 = x0 + 1;
  int y1 = y0 + 1;
  int z1 = z0 + 1;

  if (x1 < 0) x1 = 0;
  if (y1 < 0) y1 = 0;
  if (z1 < 0) z1 = 0;
  if (x1 >= srcsize[0]) x1 = srcsize[0]-1;
  if (y1 >= srcsize[1]) y1 = srcsize[1]-1;
  if (z1 >= srcsize[2]) z1 = srcsize[2]-1;

  float fx1 = x - floor(x);
  float fy1 = y - floor(y);
  float fz1 = z - floor(z);

  float fx0 = 1.0 - fx1;
  float fy0 = 1.0 - fy1;
  float fz0 = 1.0 - fz1;

  float pix000 = src[x0*srcsize[1]*srcsize[2] + y0*srcsize[2] + z0];
  float pix001 = src[x0*srcsize[1]*srcsize[2] + y0*srcsize[2] + z1];
  float pix010 = src[x0*srcsize[1]*srcsize[2] + y1*srcsize[2] + z0];
  float pix011 = src[x0*srcsize[1]*srcsize[2] + y1*srcsize[2] + z1];
  float pix100 = src[x1*srcsize[1]*srcsize[2] + y0*srcsize[2] + z0];
  float pix101 = src[x1*srcsize[1]*srcsize[2] + y0*srcsize[2] + z1];
  float pix110 = src[x1*srcsize[1]*srcsize[2] + y1*srcsize[2] + z0];
  float pix111 = src[x1*srcsize[1]*srcsize[2] + y1*srcsize[2] + z1];
/*
  float pix000 = src[z0*srcsize[0]*srcsize[1] + y0*srcsize[0] + x0];
  float pix001 = src[z1*srcsize[0]*srcsize[1] + y0*srcsize[0] + x0];
  float pix010 = src[z0*srcsize[0]*srcsize[1] + y1*srcsize[0] + x0];
  float pix011 = src[z1*srcsize[0]*srcsize[1] + y1*srcsize[0] + x0];
  float pix100 = src[z0*srcsize[0]*srcsize[1] + y0*srcsize[0] + x1];
  float pix101 = src[z1*srcsize[0]*srcsize[1] + y0*srcsize[0] + x1];
  float pix110 = src[z0*srcsize[0]*srcsize[1] + y1*srcsize[0] + x1];
  float pix111 = src[z1*srcsize[0]*srcsize[1] + y1*srcsize[0] + x1];
*/

  dst[dstpos] =
    fx0*fy0*fz0*pix000
    + fx0*fy0*fz1*pix001
    + fx0*fy1*fz0*pix010
    + fx0*fy1*fz1*pix011
    + fx1*fy0*fz0*pix100
    + fx1*fy0*fz1*pix101
    + fx1*fy1*fz0*pix110
    + fx1*fy1*fz1*pix111;
}

//
// Identity map
//

__kernel void identity(
  __global uint* size,
  __global float* spacing,
  __global float* hx,
  __global float* hy,
  __global float* hz)
{
#if 0
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1] || column >= size[2])
    return;

  size_t offset = slice*size[1]*size[2] + row*size[2] + column;

  // TODO: use image spacing and origin
  // orient? assume already axial
  //hx[offset] = slice * spacing[0] + origin[0];
  //hy[offset] = row * spacing[1] + origin[1];
  //hz[offset] = column * spacing[2] + origin[2];

  hx[offset] = slice * spacing[0];
  hy[offset] = row * spacing[1];
  hz[offset] = column * spacing[2];
#else
  size_t ix = get_global_id(0);
  size_t iy = get_global_id(1);
  size_t iz = get_global_id(2);

  if (ix >= size[0] || iy >= size[1] || iz >= size[2])
    return;

  //size_t offset = iz*size[0]*size[1] + iy*size[0] + ix;
  size_t offset = ix*size[1]*size[2] + iy*size[2] + iz;

  // TODO: use image spacing and origin
  // orient? assume already axial
  //hx[offset] = slice * spacing[0] + origin[0];
  //hy[offset] = row * spacing[1] + origin[1];
  //hz[offset] = column * spacing[2] + origin[2];

  hx[offset] = ix * spacing[0];
  hy[offset] = iy * spacing[1];
  hz[offset] = iz * spacing[2];
#endif

}

//
// Splatting and adding 3D user inputs into an image
//

__kernel void add_splat3(
  __global float* posM,
  __global float* valueM,
  __global float* sigmaM,
  uint numV,
  __global float* outx,
  __global float* outy,
  __global float* outz,
  __global uint* size,
  __global float* spacing)
{
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1] || column >= size[2])
    return;

  size_t offset = slice*size[1]*size[2] + row*size[2] + column;

  float x = convert_float(slice) * spacing[0];
  float y = convert_float(row) * spacing[1];
  float z = convert_float(column) * spacing[2];

  float sumV = 0.0;
  for (uint i = 0; i < numV; i++)
  {
    //float dx = (x - posM[i]);
    //float dy = (y - posM[i+n]);
    //float dz = (z - posM[i+n*2]);
    float dx = (x - posM[i*3]);
    float dy = (y - posM[i*3+1]);
    float dz = (z - posM[i*3+2]);

    float u = (dx*dx + dy*dy + dz*dz) / sigmaM[i];

    float weight = 1.0 / (1.0 + u);
    if (u < 0.5)
      weight = 0.0;

    if (weight > 0.01)
    {
      //outx[offset] += weight * valueM[i];
      //outy[offset] += weight * valueM[i+n];
      //outz[offset] += weight * valueM[i+n*2];
      outx[offset] += weight * valueM[i*3];
      outy[offset] += weight * valueM[i*3+1];
      outz[offset] += weight * valueM[i*3+2];
    }
  }
}

__kernel void applyPolyAffine(
  __global float* centers,
  __global float* widths,
  __global float* matrices,
  __global float* translations,
  uint numAffines,
  __global float* src, __global uint* size, __global float* spacing,
  __global float* dst)
{
  size_t column = get_global_id(2);
  size_t row = get_global_id(1);
  size_t slice = get_global_id(0);

  if (slice >= size[0] || row >= size[1] || column >= size[2])
    return;

  size_t offset = slice*size[1]*size[2] + row*size[2] + column;

  float p[3];
  p[0] = convert_float(slice) * spacing[0];
  p[1] = convert_float(row) * spacing[1];
  p[2] = convert_float(column) * spacing[2];

  float tp[3];
  for (uint dim = 0; dim < 3; dim++)
    tp[dim] = 0;

  float sumw = 0;

  for (uint i = 0; i < numAffines; i++)
  {
    float dist = 0;
    for (uint dim = 0; dim < 3; dim++)
    {
      float d = p[dim] - centers[i*3 + dim];
      dist += d*d;
    }

    float w = exp(-0.5 * dist / (widths[i]*widths[i]));

    sumw += w;

    __global float* M = matrices + (i*9);

    for (uint r = 0; r < 3; r++)
      for (uint c = 0; c < 3; c++)
        tp[r] += w * M[r*3 + c] * p[c];

    for (uint dim = 0; dim < 3; dim++)
      tp[dim] += w * translations[i*3 + dim];
  }

  float x = tp[0] / sumw / spacing[0];
  float y = tp[1] / sumw / spacing[1];
  float z = tp[2] / sumw / spacing[2];

  int x0 = convert_int(x);
  int y0 = convert_int(y);
  int z0 = convert_int(z);

  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (z0 < 0) z0 = 0;
  if (x0 >= size[0]) x0 = size[0]-1;
  if (y0 >= size[1]) y0 = size[1]-1;
  if (z0 >= size[2]) z0 = size[2]-1;

  int x1 = x0 + 1;
  int y1 = y0 + 1;
  int z1 = z0 + 1;

  if (x1 < 0) x1 = 0;
  if (y1 < 0) y1 = 0;
  if (z1 < 0) z1 = 0;
  if (x1 >= size[0]) x1 = size[0]-1;
  if (y1 >= size[1]) y1 = size[1]-1;
  if (z1 >= size[2]) z1 = size[2]-1;

  float fx1 = x - floor(x);
  float fy1 = y - floor(y);
  float fz1 = z - floor(z);

  float fx0 = 1.0 - fx1;
  float fy0 = 1.0 - fy1;
  float fz0 = 1.0 - fz1;

  float pix000 = src[x0*size[1]*size[2] + y0*size[2] + z0];
  float pix001 = src[x0*size[1]*size[2] + y0*size[2] + z1];
  float pix010 = src[x0*size[1]*size[2] + y1*size[2] + z0];
  float pix011 = src[x0*size[1]*size[2] + y1*size[2] + z1];
  float pix100 = src[x1*size[1]*size[2] + y0*size[2] + z0];
  float pix101 = src[x1*size[1]*size[2] + y0*size[2] + z1];
  float pix110 = src[x1*size[1]*size[2] + y1*size[2] + z0];
  float pix111 = src[x1*size[1]*size[2] + y1*size[2] + z1];

  dst[offset] =
    fx0*fy0*fz0*pix000
    + fx0*fy0*fz1*pix001
    + fx0*fy1*fz0*pix010
    + fx0*fy1*fz1*pix011
    + fx1*fy0*fz0*pix100
    + fx1*fy0*fz1*pix101
    + fx1*fy1*fz0*pix110
    + fx1*fy1*fz1*pix111;
}

// vim: filetype=C
