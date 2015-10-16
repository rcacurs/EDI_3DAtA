#include "../include/imageMatrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
//ImMatG function definition

ImMatG::ImMatG(){
	rows = 0;
	cols = 0;
}
ImMatG::ImMatG(size_t rows, size_t cols, double * data, bool onDeviceMemory){
	this->rows = rows;
	this->cols = cols;
	if (onDeviceMemory){
		this->data_d = data;
	} else{
		cudaMalloc(&(this->data_d), rows*cols*sizeof(double));
		cudaError_t cuerror = cudaMemcpy(this->data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
	}
}
ImMatG::ImMatG(size_t rows, size_t cols){
	this->rows = rows;
	this->cols = cols;
	cudaMalloc(&(this->data_d), rows*cols*sizeof(double));
}
ImMatG::~ImMatG(){
	cudaFree((this->data_d));
}
size_t ImMatG::getLength(void){
	return rows*cols;
}

// GPU KERNELS
__global__ void transposeKernel(const double *input, double *output, int height, int width){

	extern __shared__ double temp[];
	int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y*blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)){
		int id_in = yIndex*width + xIndex;
		temp[threadIdx.x+threadIdx.y*(blockDim.x)] = input[id_in];
	}

	__syncthreads();

	int tempXIndex = xIndex;
	xIndex = yIndex;
	yIndex = tempXIndex;

	if ((xIndex < height) && (yIndex < width)){
		int id_out = xIndex+yIndex*height;
		output[id_out] = temp[threadIdx.x+threadIdx.y*(blockDim.x)];
	}
}

ImMatG* ImMatG::transpose(){
	ImMatG *result= new ImMatG(cols, rows);
	int numThreads = 16;
	int blocksX = ceil(((float)cols) / numThreads);
	int blocksY = ceil(((float)rows) / numThreads);

	transposeKernel<<<dim3(blocksX, blocksY, 1), dim3(numThreads, numThreads, 1), (numThreads)*(numThreads)*sizeof(double)>>>(data_d, result->data_d, rows, cols);
	return result;
}

__global__ void fillRowKernel(double *data, size_t cols, size_t row, double value){
	int Xidx = threadIdx.x + blockIdx.x*blockDim.x;
	if (Xidx < cols){
		data[Xidx + row*cols] = value;
	}
}

void ImMatG::fillRow(size_t row, double value){
	if ((row >= this->rows) || (row < 0)){
		std::cout << "Index doesn't agree with image size" << std::endl;
		return;
	}

	int threadNum = 128;
	fillRowKernel << <dim3(ceil(cols / threadNum), 1, 1), dim3(threadNum, 1, 1) >> >(data_d, cols, row, value);
}

// creates im mat object from csv file
// parameter:
//		filename - filename of the files
// returns: image matrix allocated on gpu
ImMatG* readCSV(std::string fileName){
	std::ifstream fileStream(fileName);
	std::string line;
	double val;
	std::vector<double> values;
	int rows = 0, cols = 0;
	while (getline(fileStream, line)){
		std::stringstream ss(line);
		cols = 0;
		while (ss >> val){
			values.push_back(val);
			cols++;
			if (ss.peek() == ','){
				ss.ignore();
			}

		}
		rows++;
	}
	ImMatG * result = new ImMatG(rows, cols, values.data(), false);
	return result;

}

double *ImMatG::getData(){
	double * data = new double[getLength()];
	cudaMemcpy(data, data_d, sizeof(double)*getLength(), cudaMemcpyDeviceToHost);
	return data;
}

__global__ void getColumnKernel(double *image, size_t rows, size_t cols, double *column){

	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;


	if ((xIdx > cols) && (yIdx < cols)){

	}
}

