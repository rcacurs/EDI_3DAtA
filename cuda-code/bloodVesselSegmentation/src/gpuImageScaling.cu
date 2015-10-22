#include"../include/gpuImageScaling.cuh"
#include<iostream>

/*
	downsamples input two times. 
*/

__global__ void bicubicInterpolationKernel(double *inputImage, size_t rows, size_t cols, size_t scaling, double *scalingCoeffs, double*outputImage){
	
	int colsUp = cols*scaling;
	int colIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int rowIdx = threadIdx.y + blockIdx.y*blockDim.y;

	int threadLinIndex = threadIdx.x + blockDim.x*threadIdx.y;

	extern __shared__ double coeffs[];
	
	if (threadLinIndex < 4 * scaling){ // copy coeffs to shared memory
		coeffs[threadLinIndex] = scalingCoeffs[threadLinIndex];
	}

	__syncthreads();

	if ((colIdx < (cols*scaling)) && (rowIdx < rows)){ 
		int Aidx = colIdx / scaling-1; // compute indexes for input
		int Bidx = Aidx + 1;
		int Cidx = Bidx + 1;
		int Didx = Cidx + 1;

		Aidx = (Aidx >0) ? Aidx : 0;
		Bidx = (Bidx >0) ? Bidx : 0;
		Cidx = (Cidx >0) ? Cidx : 0;
		Didx = (Didx >0) ? Didx : 0;

		Aidx = (Aidx < cols) ? Aidx : cols - 1;
		Bidx = (Bidx < cols) ? Bidx : cols - 1;
		Cidx = (Cidx < cols) ? Cidx : cols - 1;
		Didx = (Didx < cols) ? Didx : cols - 1;

		int coeffIdx = colIdx%scaling;
		outputImage[colIdx + rowIdx*cols*scaling] =  inputImage[Aidx + rowIdx*cols] * scalingCoeffs[0 + coeffIdx * 4] +
													inputImage[Bidx + rowIdx*cols] * scalingCoeffs[1 + coeffIdx * 4] +
													inputImage[Cidx + rowIdx*cols] * scalingCoeffs[2 + coeffIdx * 4] +
													inputImage[Didx + rowIdx*cols] * scalingCoeffs[3 + coeffIdx * 4];

	}
}


__global__ void downsample2Kernel(double * inputImage, double * outputImage){
	int xOutput = (threadIdx.x + blockIdx.x*blockDim.x);
	int yOutput = (threadIdx.y + blockIdx.y*blockDim.y);
	int xInput = xOutput * 2;
	int yInput = yOutput * 2;
	int COLSOUT = gridDim.x*blockDim.x;
	int COLSIN = COLSOUT * 2;
	int linIn = xInput + yInput*COLSIN;
	int linOut = xOutput + yOutput*COLSOUT;
	outputImage[linOut] = inputImage[linIn];
}

__global__ void meansOfPatchesKernel(double *integralImage, int rows, int cols, int patchSize, double *means){
	int xInd = threadIdx.x + blockIdx.x*blockDim.x;
	int yInd = threadIdx.y + blockIdx.y*blockDim.y;
	int offsetx = patchSize-1;
	int offsety = patchSize-1;

	if (xInd  < (gridDim.x*blockDim.x-1)){
		offsetx = patchSize;
	}
	if (yInd < (gridDim.y*blockDim.y-1)){
		offsety = patchSize;
	}
	int Ax = 0;
	int Ay = 0;
	int Bx = offsetx;
	int By = 0;
	int Cx = 0;
	int Cy = offsety;
	int Dx = offsetx;
	int Dy = offsety;
	means[xInd + yInd*(cols - (patchSize / 2) * 2)] = (integralImage[xInd + Dx + (yInd + Dy)*cols]// +
		+ integralImage[xInd + Ax + (yInd + Ay)*cols]
		- integralImage[xInd + Bx + (yInd + By)*cols]
		- integralImage[xInd + Cx + (yInd + Cy)*cols])/(patchSize*patchSize);

}

// copmutes normalized sqrt(var+10)
__global__ void variancesOfPatchesKernel(double *integralImage, double *integralImageSq, int rows, int cols, int patchSize, double *variances){
	int xInd = threadIdx.x + blockIdx.x*blockDim.x;
	int yInd = threadIdx.y + blockIdx.y*blockDim.y;
	int offsetx = patchSize - 1;
	int offsety = patchSize - 1;

	if (xInd  < (gridDim.x*blockDim.x - 1)){
		offsetx = patchSize;
	}
	if (yInd < (gridDim.y*blockDim.y - 1)){
		offsety = patchSize;
	}
	int Ax = 0;
	int Ay = 0;
	int Bx = offsetx;
	int By = 0;
	int Cx = 0;
	int Cy = offsety;
	int Dx = offsetx;
	int Dy = offsety;
	double S1 = integralImage[xInd + Dx + (yInd + Dy)*cols]// +
		+ integralImage[xInd + Ax + (yInd + Ay)*cols]
		- integralImage[xInd + Bx + (yInd + By)*cols]
		- integralImage[xInd + Cx + (yInd + Cy)*cols];
	double S2 = integralImageSq[xInd + Dx + (yInd + Dy)*cols]// +
		+ integralImageSq[xInd + Ax + (yInd + Ay)*cols]
		- integralImageSq[xInd + Bx + (yInd + By)*cols]
		- integralImageSq[xInd + Cx + (yInd + Cy)*cols];
	double var = (S2 - S1*S1 / (patchSize*patchSize)) / (patchSize*patchSize - 1);
	variances[xInd + yInd*(cols - (patchSize / 2) * 2)] = sqrt(var + 10);
}
__global__ void rowScan(double *input, int rows, int cols, double *output, bool sqared){

	extern __shared__ double temp[];

	int offset = 1;
	int pixRow = threadIdx.y + blockIdx.y*blockDim.y;
	int colsPadded = blockDim.x * 2;
	if (threadIdx.x*2 < cols){
		temp[2 * threadIdx.x + threadIdx.y*colsPadded] = ((sqared) ? pow(input[2 * threadIdx.x + pixRow*cols], 2) : input[2 * threadIdx.x + pixRow*cols]);
		
	} else{
		temp[2 * threadIdx.x + threadIdx.y*colsPadded] = 0;
		
	}

	if (threadIdx.x * 2 + 1 < cols){
		temp[2 * threadIdx.x + threadIdx.y*colsPadded + 1] = ((sqared) ? pow(input[2 * threadIdx.x + 1 + pixRow*cols], 2) : input[2 * threadIdx.x + 1 + pixRow*cols]);
	}
	else{
		temp[2 * threadIdx.x + threadIdx.y*colsPadded + 1] = 0;
	}

	for (int d = colsPadded >> 1; d > 0; d >>= 1){
		__syncthreads();
		if (threadIdx.x < d){
			int ai = offset*(2 * threadIdx.x + 1) - 1 + threadIdx.y*colsPadded;
			int bi = offset*(2 * threadIdx.x + 2) - 1 + threadIdx.y*colsPadded;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadIdx.x == 0) temp[colsPadded - 1 + threadIdx.y*colsPadded] = 0;

	for (int d = 1; d < colsPadded; d *= 2){
		offset >>= 1; 
		__syncthreads();
		if (threadIdx.x < d){
			int ai = offset*(2 * threadIdx.x + 1) - 1 + threadIdx.y*colsPadded;
			int bi = offset*(2 * threadIdx.x + 2) - 1 + threadIdx.y*colsPadded;

			double t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	
	__syncthreads();

	if (threadIdx.x*2 < cols){
		output[2 * threadIdx.x + pixRow*cols] = temp[2 * threadIdx.x + threadIdx.y*colsPadded];
		
	}
	if (threadIdx.x * 2 + 1 < cols){
		output[2 * threadIdx.x + pixRow*cols + 1] = temp[2 * threadIdx.x + 1 + threadIdx.y*colsPadded];
	}

}


__global__ void bsxfunminusColKernel(double *input, int rows, int cols, double *vector, double *output){
	extern __shared__ double buffer[];
	int rowPix = threadIdx.y + blockIdx.y*blockDim.y;
	int colPix = threadIdx.x + blockIdx.x*blockDim.x;

	if ((rowPix < rows)&&(colPix<cols)){
		if (threadIdx.x == 0){
			buffer[threadIdx.y] = vector[rowPix];
		}
		__syncthreads();

		int xInd = colPix;
		int yInd = rowPix;

		output[xInd + cols*yInd] = input[xInd+cols*yInd]-buffer[threadIdx.y];// input[xInd + cols*yInd] - buffer[threadIdx.y];
	}

}

__global__ void bsxfunminuxRowKernel(double *input, int rows, int cols, double *vector, double *output){
	extern __shared__ double buffer[];
	int rowPix = threadIdx.y + blockIdx.y*blockDim.y;
	int colPix = threadIdx.x + blockIdx.x*blockDim.x;

	if ((rowPix < rows) && (colPix < cols)){
		if (threadIdx.y == 0){
			buffer[threadIdx.x] = vector[colPix];
		}

		__syncthreads();
		int xInd = colPix;
		int yInd = rowPix;

		output[xInd + cols*yInd] = input[xInd + cols*yInd] - buffer[threadIdx.x];

	}
}

__global__ void bsxfunmultRowKernel(double *input, int rows, int cols, double *vector, double *output){
	extern __shared__ double buffer[];
	int rowPix = threadIdx.y + blockIdx.y*blockDim.y;
	int colPix = threadIdx.x + blockIdx.x*blockDim.x;

	if ((rowPix < rows)&&(colPix<cols)){
		if (threadIdx.y == 0){
			buffer[threadIdx.x] = vector[colPix];
		}

		__syncthreads();
		int xInd = colPix;
		int yInd = rowPix;

		output[xInd + cols*yInd] = input[xInd + cols*yInd]*buffer[threadIdx.x];
	}
}

__global__ void bsxfundivideRowKernel(double *input, int rows, int cols, double *vector, double *output){
	extern __shared__ double buffer[];
	int rowPix = threadIdx.y + blockIdx.y*blockDim.y;
	int colPix = threadIdx.x + blockIdx.x*blockDim.x;

	if ((rowPix < rows) && (colPix<cols)){
		if (threadIdx.y == 0){
			buffer[threadIdx.x] = vector[colPix];
		}
		__syncthreads();

		int xInd = colPix;
		int yInd = rowPix;

		output[xInd + cols*yInd] = input[xInd + cols*yInd] / buffer[threadIdx.x];// input[xInd + cols*yInd] - buffer[threadIdx.y];
	}
}
// divies columwise
__global__ void bsxfundivideKernel(double *input, int rows, int cols, double *vector, double *output){
	extern __shared__ double buffer[];
	int rowPix = threadIdx.y + blockIdx.y*blockDim.y;
	int colPix = threadIdx.x + blockIdx.x*blockDim.x;

	if ((rowPix < rows) && (colPix<cols)){
		if (threadIdx.x == 0){
			buffer[threadIdx.y] = vector[rowPix];
		}
		__syncthreads();

		int xInd = colPix;
		int yInd = rowPix;

		output[xInd + cols*yInd] = input[xInd + cols*yInd] / buffer[threadIdx.y];// input[xInd + cols*yInd] - buffer[threadIdx.y];
	}

}

// works only if two rows!!!
__global__ void maxKernel(double *input, size_t rows, size_t cols, double *output){
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;

	if (xIdx < cols){
		double val1 = input[xIdx];
		double val2 = input[xIdx + cols];
		output[xIdx] = (val1 > val2) ? val1 : val2;
	}
}

__global__ void sumkernel(double *input, size_t rows, size_t cols, double *output){
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;

	if (xIdx < cols){
		output[xIdx] = input[xIdx] + input[xIdx + cols];
	}
}

__global__ void expKernel(double *input, size_t rows, size_t cols, double *output){
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;

	if ((xIdx < cols) && (yIdx < rows)){
		output[xIdx + yIdx*cols] = exp(input[xIdx + yIdx*cols]);
	}
}

void bsxfunminusCol(const ImMatG *inputImage, ImMatG *outputImage, double *vector){
	outputImage->rows = inputImage->rows;
	outputImage->cols = inputImage->cols;
	cudaMalloc(&(outputImage->data_d), (outputImage->rows)*(outputImage->cols)*sizeof(double));

	int threadsX = 16;
	int threadsY = 16;
	int blocksY = ceil(((float)(inputImage-> rows)) / threadsY);
	int blocksX = ceil(((float)(inputImage->cols)) / threadsY);
	bsxfunminusColKernel<<<dim3(blocksX, blocksY, 1), dim3(threadsX, threadsY, 1), threadsY*sizeof(double)>>>(inputImage->data_d, inputImage->rows, inputImage->cols, vector, outputImage->data_d);
}

void bsxfunminusRow(const ImMatG *inputImage, ImMatG *outputImage, double *vector){
	outputImage->rows = inputImage->rows;
	outputImage->cols = inputImage->cols;
	cudaMalloc(&(outputImage->data_d), (outputImage->rows)*(outputImage->cols)*sizeof(double));
	int threadsX = 16;
	int threadsY = 16;
	int blocksY = ceil(((float)(inputImage->rows)) / threadsY);
	int blocksX = ceil(((float)(inputImage->cols)) / threadsX);

	bsxfunminuxRowKernel << <dim3(blocksX, blocksY, 1), dim3(threadsX, threadsY, 1), threadsX*sizeof(double) >> >(inputImage->data_d, inputImage->rows, inputImage->cols, vector, outputImage->data_d);
}

// works only with two rowed ImMatG
void maxColumnVal(const ImMatG * input, ImMatG * output){
	output->rows = 1;
	output->cols = input->cols;
	cudaMalloc(&(output->data_d), sizeof(double)*output->cols);

	int threadsX = 32;
	int blocksX = ceil(((float)(input->cols)) / threadsX);
	maxKernel << <dim3(blocksX, 1, 1), dim3(threadsX, 1, 1) >> >(input->data_d, input->rows, input->cols, output->data_d);

}
// works only with two rowed ImMatG
void columnSum(const ImMatG *input, ImMatG *output){
	output->rows = 1;
	output->cols = input->cols;

	cudaMalloc(&(output->data_d), sizeof(double)*output->cols);

	int threadsX = 32;

	int blocksX = ceil(((float)(input->cols)) / threadsX);
	sumkernel << <dim3(blocksX, 1, 1), dim3(threadsX, 1, 1) >> >(input->data_d, input->rows, input->cols, output->data_d);

}


void expElem(const ImMatG *input, ImMatG *output){
	output->rows = input->rows;
	output->cols = input->cols;

	cudaMalloc(&(output->data_d), output->getLength()*sizeof(double));

	int threadsX = 16;
	int threadsY = 16;

	int blocksX = ceil(((double)(input->cols)) / threadsX);
	int blocksY = ceil(((double)(input->rows)) / threadsY);

	expKernel << < dim3(blocksX, blocksY, 1), dim3(threadsX, threadsY, 1) >> >(input->data_d, input->rows, input->cols, output->data_d);


}
void bsxfunmultRow(const ImMatG *inputImage, ImMatG*outputImage, double *vector){
	outputImage->rows = inputImage->rows;
	outputImage->cols = inputImage->cols;
	cudaMalloc(&(outputImage->data_d), (outputImage->rows)*(outputImage->cols)*sizeof(double));

	int threadsY = 16;
	int threadsX = 32;
	int blocksX = ceil(((float)(inputImage->cols)) / threadsX);
	int blocksY = ceil(((float)(inputImage->rows)) / threadsY);
	bsxfunmultRowKernel << <dim3(blocksX, blocksY, 1), dim3(threadsX, threadsY, 1), threadsX*sizeof(double) >> >(inputImage->data_d, inputImage->rows, inputImage->cols, vector, outputImage->data_d);
}

void bsxfundivide(const ImMatG *inputImage, ImMatG *outputImage, double *vector){
	outputImage->rows = inputImage->rows;
	outputImage->cols = inputImage->cols;
	cudaMalloc(&(outputImage->data_d), (outputImage->rows)*(outputImage->cols)*sizeof(double));

	int threadsX = 16;
	int threadsY = 16;
	int blocksY = ceil(((double)(inputImage->rows)) / threadsY);
	int blocksX = ceil(((double)(inputImage->cols)) / threadsX);
	bsxfundivideKernel << <dim3(blocksX, blocksY, 1), dim3(threadsX, threadsY, 1), threadsY*sizeof(double) >> >(inputImage->data_d, inputImage->rows, inputImage->cols, vector, outputImage->data_d);
}

void bsxfunRowDivide(const ImMatG *inputImage, ImMatG *outputImage, double *vector){
	outputImage->rows = inputImage->rows;
	outputImage->cols = inputImage->cols;

	cudaMalloc(&(outputImage->data_d), (outputImage->rows)*(outputImage->cols)*sizeof(double));

	int threadsX = 16;
	int threadsY = 16;

	int blocksY = ceil(((double)(inputImage->rows)) / threadsY);
	int blocksX = ceil(((double)(inputImage->cols)) / threadsX);

	bsxfundivideRowKernel << <dim3(blocksX, blocksY, 1), dim3(threadsX, threadsY, 1), threadsX*sizeof(double) >> >(inputImage->data_d, inputImage->rows, inputImage->cols, vector, outputImage->data_d);

}
// computes means of pathex from integral image
void meansOfPatches(const ImMatG *inputImage, int patchSize, ImMatG *meanVector){
	int rows = inputImage->rows - (patchSize / 2) * 2;
	int cols = inputImage->cols - (patchSize / 2) * 2;
	meanVector->rows = 1;
	meanVector->cols = rows*cols;
	cudaMalloc(&(meanVector->data_d), sizeof(double)*rows*cols);
	int threads = 8;

	meansOfPatchesKernel << <dim3((cols) / threads, rows / threads, 1), dim3(threads, threads, 1) >> >(inputImage->data_d, inputImage->rows, inputImage->cols, patchSize, meanVector->data_d);
}

void variancesOfPatches(const ImMatG *integralImage, const ImMatG *integralImageSq, int patchSize, ImMatG *varVector){
	int rows = integralImage->rows - (patchSize / 2) * 2;
	int cols = integralImage->cols - (patchSize / 2) * 2;
	varVector->rows = 1;
	varVector->cols = rows*cols;
	cudaMalloc(&(varVector->data_d), sizeof(double)*rows*cols);
	int threads = 8;

	variancesOfPatchesKernel << <dim3((cols) / threads, rows / threads, 1), dim3(threads, threads, 1) >> >(integralImage->data_d, integralImageSq->data_d, integralImage->rows, integralImage->cols, patchSize, varVector->data_d);
}

/*
	inserts one two dimensional array into other, top left corner of input image is inserted into
	row and column specified by rowOffset and colOffset
*/
__global__ void insertArray(const double *inputImage, int inRows, int inCols,  double *outputImage, int outRows, int outCols, int rowOffset, int colOffset){
	int colIdx = threadIdx.x + blockDim.x*blockIdx.x;
	int rowIdx = threadIdx.y + blockDim.y*blockIdx.y;

	int linInIdx = colIdx + rowIdx*inCols;
	int linOutIdx = colIdx + colOffset + (rowIdx + rowOffset)*outCols;

	outputImage[linOutIdx] = inputImage[linInIdx];

}


__global__ void im2colKernel(const double *inputImage, int inRows, int inCols, int patchSize, double *outputArray){
	
	int xPos = threadIdx.x + blockDim.x*blockIdx.x;
	int yPos = threadIdx.y + blockDim.y*blockIdx.y;

	for (int i = 0; i < patchSize; i++){
		for (int j = 0; j < patchSize; j++){
			outputArray[(xPos + yPos*inCols)*patchSize*patchSize+j+patchSize*i] = inputImage[xPos+j+(yPos+i)*(inCols+(patchSize/2)*2)];
		}
	}

}
// must remember to transpose image before
ImMatG * im2col(const ImMatG *inputImage, int patchSize){
	int inCols = (inputImage->cols) - (patchSize / 2) * 2;
	int inRows = (inputImage->rows) - (patchSize / 2) * 2;
	ImMatG *patches = new ImMatG(inRows*inCols, patchSize*patchSize);
	int numThreads = 16;
	im2colKernel<<<dim3(inCols/numThreads, inRows/numThreads,1), dim3(numThreads, numThreads, 1)>>>(inputImage->data_d, inRows, inCols, patchSize, patches->data_d);
	return patches;
}
void padArray(const ImMatG *inputImage, ImMatG *paddedImage, int paddingPixels){
	int numThreads = 16;
	paddedImage->rows = inputImage->rows + 2 * paddingPixels;
	paddedImage->cols = inputImage->cols + 2 * paddingPixels;
	cudaMalloc(&(paddedImage->data_d), paddedImage->rows*paddedImage->cols*sizeof(double));
	cudaMemset(paddedImage->data_d, 0, paddedImage->rows*paddedImage->cols*sizeof(double));

	insertArray << <dim3(inputImage->cols / numThreads, inputImage->rows / numThreads, 1), dim3(numThreads, numThreads) >> >(inputImage->data_d,
		inputImage->rows,
		inputImage->cols,
		paddedImage->data_d,
		paddedImage->rows,
		paddedImage->cols,
		paddingPixels,
		paddingPixels);
}

void downsample2(ImMatG * inputImage, ImMatG * outputImage){

	int threads=8;

	downsample2Kernel << <dim3( (outputImage->cols) / threads, (outputImage->rows) / ( threads), 1), dim3(threads, threads, 1) >> > (inputImage->data_d, outputImage->data_d);
}

ImMatG * computeIntegralImage(const ImMatG *inputImage){
	ImMatG *temp = new ImMatG(inputImage->rows, inputImage->cols);
	int threadsY=1;
	// compute number of colusmns
	int p = pow(2.0,ceil(log2((double)(inputImage->cols))));
	
	rowScan <<< dim3(1, inputImage->rows/threadsY, 1), dim3(p/2, threadsY, 1), sizeof(double)*threadsY*p>>>(inputImage->data_d, inputImage->rows, inputImage->cols, temp->data_d, false);
	ImMatG *tempTr = temp->transpose();
	delete temp;
	ImMatG *tempTrSc = new ImMatG(tempTr->rows, tempTr->cols);
	p = pow(2.0,ceil(log2((double)(tempTrSc->cols))));
	rowScan << < dim3(1, tempTr->rows / threadsY, 1), dim3(p/2, threadsY, 1), sizeof(double)*threadsY*p >> >(tempTr->data_d, tempTr->rows, tempTr->cols, tempTrSc->data_d, false);
	ImMatG *result = tempTrSc->transpose();
	delete tempTr;
	delete tempTrSc;
	return result;
}

ImMatG * computeIntegralImageSq(const ImMatG *inputImage){
	ImMatG *temp = new ImMatG(inputImage->rows, inputImage->cols);
	int threadsY = 1;
	// compute number of colusmns
	int p = pow(2.0, ceil(log2((double)(inputImage->cols))));

	rowScan << < dim3(1, inputImage->rows / threadsY, 1), dim3(p / 2, threadsY, 1), sizeof(double)*threadsY*p >> >(inputImage->data_d, inputImage->rows, inputImage->cols, temp->data_d, true);
	ImMatG *tempTr = temp->transpose();
	delete temp;
	ImMatG *tempTrSc = new ImMatG(tempTr->rows, tempTr->cols);
	p = pow(2.0, ceil(log2((double)(tempTrSc->cols))));
	rowScan << < dim3(1, tempTr->rows / threadsY, 1), dim3(p / 2, threadsY, 1), sizeof(double)*threadsY*p >> >(tempTr->data_d, tempTr->rows, tempTr->cols, tempTrSc->data_d, false);
	ImMatG *result = tempTrSc->transpose();
	delete tempTr;
	delete tempTrSc;
	return result;
}

std::vector<ImMatG> computePyramid(ImMatG & input, int layers){
	std::vector<ImMatG> pyramid;

	pyramid.push_back(input);
	for (int i = 1; i < layers; i++){

	}

	return pyramid;
}

ImagePyramidCreator::ImagePyramidCreator(int layers, int kernelSize, double sigma){
	this->layers = layers;
	this->kernelSize = kernelSize;
	this->sigma = sigma;
	cudaMalloc(&(this->kernel_d), kernelSize*sizeof(double));
	double * kernel = gaussianKernel1D(kernelSize, sigma);
	cudaMemcpy(kernel_d, kernel, kernelSize*sizeof(double), cudaMemcpyHostToDevice);
	free(kernel);
}

// returns bicubic coefficient table
double *bicubicCoeffs(size_t scaling){
	double *result = (double *)malloc(scaling * 4 * sizeof(double));

	for (int i = 0; i < scaling; i++){
		double alfa = i*(1/((double)scaling));
		result[0 + i * 4] = -0.5 * pow(alfa + 1, 3) + 2.5 * pow(alfa + 1, 2) - 4 * (alfa + 1) + 2;
		result[1 + i * 4] = 1.5 * pow(alfa, 3) - 2.5 * pow(alfa, 2) + 1;
		result[2 + i * 4] = 1.5 * pow(1 - alfa, 3) - 2.5 * pow(1 - alfa, 2) + 1;
		result[3 + i * 4] = -0.5 * pow(2 - alfa, 3) + 2.5 * pow(2 - alfa, 2) - 4 * (2 - alfa) + 2;
	}
	return result;
}

// result is transposed
ImMatG * bicubicResize(ImMatG *inputImage, size_t scaling){
	ImMatG * rowUpscaled = new ImMatG(inputImage->rows, (inputImage->cols)*scaling);

	double * scalingCoeffs = bicubicCoeffs(scaling);
	double * scalingCoeffs_d;
	cudaMalloc(&scalingCoeffs_d, sizeof(double)*scaling * 4);
	cudaMemcpy(scalingCoeffs_d, scalingCoeffs, sizeof(double)*scaling * 4, cudaMemcpyHostToDevice);
	int threadsx = 16;
	int threadsy = 16;

	bicubicInterpolationKernel << <dim3(rowUpscaled->cols / threadsx,
		(rowUpscaled->rows) / threadsy, 1),
		dim3(threadsx, threadsy, 1), sizeof(double) * 4 * scaling >> >(
		inputImage->data_d, inputImage->rows, inputImage->cols, scaling, scalingCoeffs_d, rowUpscaled->data_d);

	ImMatG * rowUpscaledT = rowUpscaled->transpose();
	ImMatG * resultT = new ImMatG(inputImage->rows*scaling, inputImage->cols*scaling);

	bicubicInterpolationKernel << <dim3(scaling*rowUpscaledT->cols / threadsx,
		rowUpscaledT->rows / threadsy, 1),
		dim3(threadsx, threadsy, 1), sizeof(double) * 4 * scaling >> >(
		rowUpscaledT->data_d, rowUpscaledT->rows, rowUpscaledT->cols, scaling, scalingCoeffs_d, resultT->data_d);
	
	free(scalingCoeffs);
	cudaFree(scalingCoeffs_d);
	delete rowUpscaled;
	delete rowUpscaledT;
	return resultT;// ->transpose();
}

ImagePyramidCreator::~ImagePyramidCreator(){
	cudaFree(kernel_d);

}


std::vector<ImMatG *> ImagePyramidCreator::createPyramid(ImMatG *input){
	std::vector<ImMatG *> pyramid;

	ImMatG * blurLayer = new ImMatG(input->rows, input->cols);
	sepConvolve2D(input, kernel_d, kernelSize, blurLayer);
	pyramid.push_back(blurLayer);
	for (int i = 1; i < this->layers; i++){
		ImMatG *bluredLayer = new ImMatG(pyramid[i - 1]->rows, pyramid[i - 1]->cols);
		ImMatG *downScaled = new ImMatG(pyramid[i - 1]->rows/2, pyramid[i - 1]->cols/2);
		sepConvolve2D(pyramid[i - 1], kernel_d, kernelSize, bluredLayer);
		downsample2(bluredLayer, downScaled);
		pyramid.push_back(downScaled);
		delete bluredLayer;
	}
	return pyramid;
}

 


