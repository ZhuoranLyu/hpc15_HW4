// Convolution kernel from Heterogeneous Computing with OpenCL
// by Gaster, Howes, Kaeli, Mistry, and Schaa

kernel void convolution(
      global float* imageIn,
      global float* imageOut,
    constant float* filter,
               int  rows,
               int  cols,
               int  filterWidth,
       local float* localImage,
               int  localHeight,
               int  localWidth)
{
  // Determine the amount of padding for this filter
  int filterRadius = filterWidth/2;
  int padding = filterRadius * 2;

  // Determine where each workgroup begins reading
  int groupStartCol = get_group_id(0)*get_local_size(0);
  int groupStartRow = get_group_id(1)*get_local_size(1);

  // Determine the local ID of each work-item
  int localCol = get_local_id(0);
  int localRow = get_local_id(1);

  // Determine the global ID of each work-item.  Work-items
  // representing the output region will have a unique global ID
  int globalCol = groupStartCol + localCol;
  int globalRow = groupStartRow + localRow;

#if 0
  if(globalRow < rows && globalCol < cols)
    imageOut[globalRow*cols + globalCol] = imageIn[globalRow*cols + globalCol];
#endif
  // Cache the data to local memory

  // Step down rows
  for(int i = localRow; i < localHeight; i += get_local_size(1)) {

    int curRow = groupStartRow+i;

    // Step across columns
    for(int j = localCol; j < localWidth; j += get_local_size(0)) {

      int curCol = groupStartCol+j;

      // Perform the read if it is in bounds
      if(curRow < rows && curCol < cols) {
        localImage[i*localWidth + j] = imageIn[curRow*cols+curCol];
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Perform the convolution
  if(globalRow < rows-padding && globalCol < cols-padding) {

    // Each work item will filter around its start location 
    //(starting from the filter radius left and up)
    float sum = 0.0f;
    int filterIdx = 0;

    // Not unrolled
    for(int i = localRow; i < localRow+filterWidth; i++) {
      int offset = i*localWidth;
      for(int j = localCol; j < localCol+filterWidth; j++){
        sum += localImage[offset+j] * filter[filterIdx++];
      }
    }

    /*
    // Inner loop unrolled
    for(int i = localRow; i < localRow+filterWidth; i++) {
    int offset = i*localWidth+localCol;
    sum += localImage[offset++] * filter[filterIdx++];
    sum += localImage[offset++] * filter[filterIdx++];
    sum += localImage[offset++] * filter[filterIdx++];
    sum += localImage[offset++] * filter[filterIdx++];
    sum += localImage[offset++] * filter[filterIdx++];
    sum += localImage[offset++] * filter[filterIdx++];
    sum += localImage[offset++] * filter[filterIdx++];
    }
    */

    // Write the data out
    imageOut[(globalRow+filterRadius)*cols + (globalCol+filterRadius)] = sum;
    imageIn[(globalRow+filterRadius)*cols + (globalCol+filterRadius)] = sum;
  }

  return;
}

