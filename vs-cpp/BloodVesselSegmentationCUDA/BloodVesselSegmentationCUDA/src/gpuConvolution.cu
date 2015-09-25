#include"../include/gpuConvolution.cuh"
#include<malloc.h>
#define _USE_MATH_DEFINES
#include<math.h>
#include<stdio.h>

// performs seperable convolution with specified filter (same filter applied to rows and columns)
void sepConvolve2D(ImMatG * inputMat, double *d_filterKernel, size_t kernelSize, ImMatG * outputMat){
	int patchSize = 16;
	double * d_temp;
	cudaMalloc(&d_temp, inputMat->getLength()*sizeof(double));
	row_convolve << < dim3(inputMat->cols / patchSize, inputMat->rows / patchSize, 1), dim3(patchSize, patchSize, 1), (patchSize + kernelSize / 2 * 2)*patchSize*sizeof(double) >> >(inputMat->data_d, inputMat->rows, inputMat->cols, d_filterKernel, kernelSize, d_temp);
	col_convolve << < dim3(inputMat->cols / patchSize, inputMat->rows / patchSize, 1), dim3(patchSize, patchSize, 1), (patchSize + kernelSize / 2 * 2)*patchSize*sizeof(double) >> >(d_temp, inputMat->rows, inputMat->cols, d_filterKernel, kernelSize, outputMat->data_d);
	cudaFree(d_temp);
}


// return gaussian filter 1D kernel. return pointer to array
//		kernelSize size of filter elements
//		sigma vriance of filter
double * gaussianKernel1D(int kernelSize, double sigma){
	double *kernel = (double*)malloc(kernelSize*sizeof(double));
	double sum = 0;
	// compute elements
	for (int i = 0; i < kernelSize; i++){
		double elem = 1 / (sqrt(2 * M_PI)*sigma)*exp(-pow((double)i - kernelSize / 2, 2) / (2 * pow(sigma, 2)));
		sum += elem;
		kernel[i] = elem;
	}

	// normalize
	for (int i = 0; i < kernelSize; i++){
		kernel[i]=kernel[i] / sum;
	}
	return kernel;
}

// ====================== KERNEL FUNCTIONS =================================
/*performs row wise convolution on image block*/
__global__ void row_convolve(double * d_inputData, int ROWS, int COLS, double * d_filterKernel, size_t kernelSize, double * d_outputData){
	extern __shared__ double buffer[];

	int kernelRadius = kernelSize / 2;
	// load data in shared memory
	// indexes for pixel coordinates to be loades in to shared memory.
	int colPixIndex = threadIdx.x + blockDim.x*blockIdx.x;
	int rowPixIndex = threadIdx.y + blockDim.y* blockIdx.y;
	int linPixIndex = colPixIndex + rowPixIndex*COLS;
	// load patch to be processed

	int linBufIdx = threadIdx.x + kernelRadius + threadIdx.y*(blockDim.x + kernelRadius * 2);
	buffer[linBufIdx] = d_inputData[linPixIndex];

	// load apron

	if (threadIdx.x < kernelRadius){

		int idxBuf = linBufIdx-kernelRadius;
		if ( colPixIndex-kernelRadius>= 0){
			buffer[idxBuf] = d_inputData[linPixIndex-kernelRadius];
		}
		else{
			buffer[idxBuf] = 0;
		}
	}

	if (threadIdx.x >= blockDim.x - kernelRadius){
		int idxBuf = linBufIdx + kernelRadius;
		if ( colPixIndex + kernelRadius< COLS){
			buffer[idxBuf] = d_inputData[linPixIndex+kernelRadius];
		}
		else{
			buffer[idxBuf] = 0;
		}
	}
	__syncthreads();
	//// convolve
	//
	double sum = 0;
	for (int i = 0; i < kernelSize; i++){
		sum += buffer[linBufIdx+(i-kernelRadius)] * d_filterKernel[kernelSize - i - 1];
	}
	d_outputData[linPixIndex] = sum;
}

// performs columnwise convolution
__global__ void col_convolve(double * d_inputData, int ROWS, int COLS, double * d_filterKernel, size_t kernelSize, double * d_outputData){
	extern __shared__ double buffer[];

	int kernelRadius = kernelSize / 2;
	// load data in shared memory
	// indexes for pixel coordinates to be loades in to shared memory.
	int colPixIndex = threadIdx.x + blockDim.x*blockIdx.x;
	int rowPixIndex = threadIdx.y + blockDim.y* blockIdx.y;
	int linPixIndex = colPixIndex + rowPixIndex*COLS;
	// load patch to be processed

	int linBufIdx = threadIdx.x+(threadIdx.y+kernelRadius)*blockDim.x;
	buffer[linBufIdx] = d_inputData[linPixIndex];

	// load apron

	if (threadIdx.y < kernelRadius){

		int idxBuf = linBufIdx - kernelRadius*blockDim.x;
		if (rowPixIndex - kernelRadius >= 0){
			buffer[idxBuf] = d_inputData[linPixIndex - kernelRadius*COLS];
		}
		else{
			buffer[idxBuf] = 0;
		}
	}

	if (threadIdx.y >= blockDim.y - kernelRadius){
		int idxBuf = linBufIdx + kernelRadius*blockDim.x;
		if (rowPixIndex + kernelRadius< ROWS){
			buffer[idxBuf] = d_inputData[linPixIndex + kernelRadius*COLS];
		}
	else{
			buffer[idxBuf] = 0;
		}
	}
	__syncthreads();
	////// convolve
	////
	double sum = 0;
	for (int i = 0; i < kernelSize; i++){
		sum += buffer[linBufIdx + (i - kernelRadius)*blockDim.x] * d_filterKernel[kernelSize - i - 1];
	}
	d_outputData[linPixIndex] = sum;
}


