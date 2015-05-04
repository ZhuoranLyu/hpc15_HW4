/* Convolution example; originally written by Lucas Wilcox.
 * Minor modifications by Georg Stadler.
 * The function expects a bitmap image (*.ppm) as input, as
 * well as a number of blurring loops to be performed.
 */

#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include "cl-helper.h"
#include "ppma_io.h"

#define FILTER_WIDTH 7
#define HALF_FILTER_WIDTH 3

// local size of work group
#define WGX 8
#define WGY 8


void print_kernel_info(cl_command_queue queue, cl_kernel knl)
{
  // get device associated with the queue
  cl_device_id dev;
  CALL_CL_SAFE(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
        sizeof(dev), &dev, NULL));

  char kernel_name[4096];
  CALL_CL_SAFE(clGetKernelInfo(knl, CL_KERNEL_FUNCTION_NAME,
        sizeof(kernel_name), &kernel_name, NULL));
  kernel_name[4095] = '\0';
  printf("Info for kernel %s:\n", kernel_name);

  size_t kernel_work_group_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(kernel_work_group_size), &kernel_work_group_size, NULL));
  printf("  CL_KERNEL_WORK_GROUP_SIZE=%zd\n", kernel_work_group_size);

  size_t preferred_work_group_size_multiple;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(preferred_work_group_size_multiple),
        &preferred_work_group_size_multiple, NULL));
  printf("  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=%zd\n",
      preferred_work_group_size_multiple);

  cl_ulong kernel_local_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(kernel_local_mem_size), &kernel_local_mem_size, NULL));
  printf("  CL_KERNEL_LOCAL_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_local_mem_size);

  cl_ulong kernel_private_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
        sizeof(kernel_private_mem_size), &kernel_private_mem_size, NULL));
  printf("  CL_KERNEL_PRIVATE_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_private_mem_size);
}


int main(int argc, char *argv[])
{
  int error, xsize, ysize, rgb_max;
  int *r, *b, *g;

  float *gray, *congray, *congray_cl;

  // identity kernel
  // float filter[] = {
  //   0,0,0,0,0,0,0,
  //   0,0,0,0,0,0,0,
  //   0,0,0,0,0,0,0,
  //   0,0,0,1,0,0,0,
  //   0,0,0,0,0,0,0,
  //   0,0,0,0,0,0,0,
  //   0,0,0,0,0,0,0,
  // };

  // 45 degree motion blur
  float filter[] =
    {0,      0,      0,      0,      0, 0.0145,      0,
     0,      0,      0,      0, 0.0376, 0.1283, 0.0145,
     0,      0,      0, 0.0376, 0.1283, 0.0376,      0,
     0,      0, 0.0376, 0.1283, 0.0376,      0,      0,
     0, 0.0376, 0.1283, 0.0376,      0,      0,      0,
0.0145, 0.1283, 0.0376,      0,      0,      0,      0,
     0, 0.0145,      0,      0,      0,      0,      0};

  // mexican hat kernel
  // float filter[] = {
  //   0, 0,-1,-1,-1, 0, 0,
  //   0,-1,-3,-3,-3,-1, 0,
  //  -1,-3, 0, 7, 0,-3,-1,
  //  -1,-3, 7,24, 7,-3,-1,
  //  -1,-3, 0, 7, 0,-3,-1,
  //   0,-1,-3,-3,-3,-1, 0,
  //   0, 0,-1,-1,-1, 0, 0
  // };


  if(argc != 3)
  {
    fprintf(stderr, "Usage: %s image.ppm num_loops\n", argv[0]);
    abort();
  }

  const char* filename = argv[1];
  const int num_loops = atoi(argv[2]);


  // --------------------------------------------------------------------------
  // load image
  // --------------------------------------------------------------------------
  printf("Reading ``%s''\n", filename);
  ppma_read(filename, &xsize, &ysize, &rgb_max, &r, &g, &b);
  printf("Done reading ``%s'' of size %dx%d\n", filename, xsize, ysize);

  // --------------------------------------------------------------------------
  // allocate CPU buffers
  // --------------------------------------------------------------------------
  posix_memalign((void**)&gray, 32, xsize*ysize*sizeof(float));
  if(!gray) { fprintf(stderr, "alloc gray"); abort(); }
  posix_memalign((void**)&congray, 32, xsize*ysize*sizeof(float));
  if(!congray) { fprintf(stderr, "alloc gray"); abort(); }
  posix_memalign((void**)&congray_cl, 32, xsize*ysize*sizeof(float));
  if(!congray_cl) { fprintf(stderr, "alloc gray"); abort(); }

  // --------------------------------------------------------------------------
  // convert image to grayscale
  // --------------------------------------------------------------------------
  for(int n = 0; n < xsize*ysize; ++n)
    gray[n] = (0.21f*r[n])/rgb_max + (0.72f*g[n])/rgb_max + (0.07f*b[n])/rgb_max;

  // --------------------------------------------------------------------------
  // execute filter on cpu
  // --------------------------------------------------------------------------
  for(int i = HALF_FILTER_WIDTH; i < ysize - HALF_FILTER_WIDTH; ++i)
  {
    for(int j = HALF_FILTER_WIDTH; j < xsize - HALF_FILTER_WIDTH; ++j)
    {
      float sum = 0;
      for(int k = -HALF_FILTER_WIDTH; k <= HALF_FILTER_WIDTH; ++k)
      {
        for(int l = -HALF_FILTER_WIDTH; l <= HALF_FILTER_WIDTH; ++l)
        {
          sum += gray[(i+k)*xsize + (j+l)] *
            filter[(k+HALF_FILTER_WIDTH)*FILTER_WIDTH + (l+HALF_FILTER_WIDTH)];
        }
      }
      congray[i*xsize + j] = sum;
    }
  }

  // --------------------------------------------------------------------------
  // output cpu filtered image
  // --------------------------------------------------------------------------
  printf("Writing cpu filtered image\n");
  for(int n = 0; n < xsize*ysize; ++n)
    r[n] = g[n] = b[n] = (int)(congray[n] * rgb_max);
  error = ppma_write("output_cpu.ppm", xsize, ysize, r, g, b);
  if(error) { fprintf(stderr, "error writing image"); abort(); }

  // --------------------------------------------------------------------------
  // get an OpenCL context and queue
  // --------------------------------------------------------------------------
  cl_context ctx;
  cl_command_queue queue;
  create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);
  print_device_info_from_queue(queue);

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  char *knl_text = read_file("convolution.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "convolution", NULL);
  free(knl_text);

#ifdef NON_OPTIMIZED
  int deviceWidth = xsize;
#else
  int deviceWidth = ((xsize + WGX - 1)/WGX)* WGX;
#endif
  int deviceHeight = ysize;
  size_t deviceDataSize = deviceHeight*deviceWidth*sizeof(float);

  // --------------------------------------------------------------------------
  // allocate device memory
  // --------------------------------------------------------------------------
  cl_int status;
  cl_mem buf_gray = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
     deviceDataSize, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem buf_congray = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
      deviceDataSize, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem buf_filter = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
     FILTER_WIDTH*FILTER_WIDTH*sizeof(float), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // --------------------------------------------------------------------------
  // transfer to device
  // --------------------------------------------------------------------------
#ifdef NON_OPTIMIZED
  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, buf_gray, /*blocking*/ CL_TRUE, /*offset*/ 0,
        deviceDataSize, gray, 0, NULL, NULL));
#else
  size_t buffer_origin[3] = {0,0,0};
  size_t host_origin[3] = {0,0,0};
  size_t region[3] = {deviceWidth*sizeof(float), ysize, 1};
  clEnqueueWriteBufferRect(queue, buf_gray, CL_TRUE,
                           buffer_origin, host_origin, region,
                           deviceWidth*sizeof(float), 0, xsize*sizeof(float), 0,
                           gray, 0, NULL, NULL);
#endif

  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, buf_filter, /*blocking*/ CL_TRUE, /*offset*/ 0,
        FILTER_WIDTH*FILTER_WIDTH*sizeof(float), filter, 0, NULL, NULL));

  // --------------------------------------------------------------------------
  // run code on device
  // --------------------------------------------------------------------------

  cl_int rows = ysize;
  cl_int cols = xsize;
  cl_int filterWidth = FILTER_WIDTH;
  cl_int paddingPixels = 2*HALF_FILTER_WIDTH;

  size_t local_size[] = { WGX, WGY };
  size_t global_size[] = {
    ((xsize-paddingPixels + local_size[0] - 1)/local_size[0])* local_size[0],
    ((ysize-paddingPixels + local_size[1] - 1)/local_size[1])* local_size[1],
  };

  cl_int localWidth = local_size[0] + paddingPixels;
  cl_int localHeight = local_size[1] + paddingPixels;
  size_t localMemSize = localWidth * localHeight * sizeof(float);

  CALL_CL_SAFE(clSetKernelArg(knl, 0, sizeof(buf_gray), &buf_gray));
  CALL_CL_SAFE(clSetKernelArg(knl, 1, sizeof(buf_congray), &buf_congray));
  CALL_CL_SAFE(clSetKernelArg(knl, 2, sizeof(buf_filter), &buf_filter));
  CALL_CL_SAFE(clSetKernelArg(knl, 3, sizeof(rows), &rows));
  CALL_CL_SAFE(clSetKernelArg(knl, 4, sizeof(cols), &cols));
  CALL_CL_SAFE(clSetKernelArg(knl, 5, sizeof(filterWidth), &filterWidth));
  CALL_CL_SAFE(clSetKernelArg(knl, 6, localMemSize, NULL));
  CALL_CL_SAFE(clSetKernelArg(knl, 7, sizeof(localHeight), &localHeight));
  CALL_CL_SAFE(clSetKernelArg(knl, 8, sizeof(localWidth), &localWidth));

  // --------------------------------------------------------------------------
  // print kernel info
  // --------------------------------------------------------------------------
  print_kernel_info(queue, knl);

  CALL_CL_SAFE(clFinish(queue));
  timestamp_type tic, toc;
  get_timestamp(&tic);
  for(int loop = 0; loop < num_loops; ++loop)
  {
    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl, 2, NULL,
          global_size, local_size, 0, NULL, NULL));
  }
  CALL_CL_SAFE(clFinish(queue));
  get_timestamp(&toc);

  double elapsed = timestamp_diff_in_seconds(tic,toc)/num_loops;
  printf("%f s\n", elapsed);
  printf("%f MPixels/s\n", xsize*ysize/1e6/elapsed);
  printf("%f GBit/s\n", 2*xsize*ysize*sizeof(float)/1e9/elapsed);
  printf("%f GFlop/s\n", (xsize-HALF_FILTER_WIDTH)*(ysize-HALF_FILTER_WIDTH)
	 *FILTER_WIDTH*FILTER_WIDTH/1e9/elapsed);

  // --------------------------------------------------------------------------
  // transfer back & check
  // --------------------------------------------------------------------------
#ifdef NON_OPTIMIZED
  CALL_CL_SAFE(clEnqueueReadBuffer(
        queue, buf_congray, /*blocking*/ CL_TRUE, /*offset*/ 0,
        xsize * ysize * sizeof(float), congray_cl,
        0, NULL, NULL));
#else
  buffer_origin[0] = 3*sizeof(float);
  buffer_origin[1] = 3;
  buffer_origin[2] = 0;

  host_origin[0] = 3*sizeof(float);
  host_origin[1] = 3;
  host_origin[2] = 0;

  region[0] = (xsize-paddingPixels)*sizeof(float);
  region[1] = (ysize-paddingPixels);
  region[2] = 1;

  clEnqueueReadBufferRect(queue, buf_congray, CL_TRUE,
      buffer_origin, host_origin, region,
      deviceWidth*sizeof(float), 0, xsize*sizeof(float), 0,
      congray_cl, 0, NULL, NULL);
#endif

  // --------------------------------------------------------------------------
  // output OpenCL filtered image
  // --------------------------------------------------------------------------
  printf("Writing OpenCL filtered image\n");
  for(int n = 0; n < xsize*ysize; ++n)
    r[n] = g[n] = b[n] = (int)(congray_cl[n] * rgb_max);
  error = ppma_write("output_cl.ppm", xsize, ysize, r, g, b);
  if(error) { fprintf(stderr, "error writing image"); abort(); }

  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_SAFE(clReleaseMemObject(buf_congray));
  CALL_CL_SAFE(clReleaseMemObject(buf_gray));
  CALL_CL_SAFE(clReleaseMemObject(buf_filter));
  CALL_CL_SAFE(clReleaseKernel(knl));
  CALL_CL_SAFE(clReleaseCommandQueue(queue));
  CALL_CL_SAFE(clReleaseContext(ctx));
  free(gray);
  free(congray);
  free(congray_cl);
  free(r);
  free(b);
  free(g);
}
