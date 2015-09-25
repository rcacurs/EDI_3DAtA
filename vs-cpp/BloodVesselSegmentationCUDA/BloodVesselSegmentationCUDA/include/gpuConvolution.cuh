#ifndef __GPU_CONVOLUTION_CUH
#define __GPU_CONVOLUTION_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/imageMatrix.cuh"

// Performs 2D seperable convolution
//		inpuMat - input matrix on which to performe convolution
//      d_filterKernel - filter kernel alocated on gpu
//		kernelSize - size of the kernel
//      outputMat - output matrix where results are stored

void sepConvolve2D(ImMatG * inputMat, double *d_filterKernel, size_t kernelSize, ImMatG * outputMat);

// computes 1D gaussian kernel with specified size and sigma value (zero mean)
//		kernelSize - size of the kernel
//		sigma - standard deviation of gaussian distribution
// returns:
//		pointer to filter kernel data (allocated on Host memory)

double* gaussianKernel1D(int kernelSize, double sigma);


__global__ void row_convolve(double * d_inputData, int ROWS, int COLS, double * d_filterKernel, size_t kernelSize, double * d_outputData);

__global__ void col_convolve(double * d_inputData, int ROWS, int COLS, double * d_filterKernel, size_t kernelSize, double * d_outputData);
#endif