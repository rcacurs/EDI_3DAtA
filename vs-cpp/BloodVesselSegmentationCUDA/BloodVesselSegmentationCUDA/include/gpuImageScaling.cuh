#ifndef __GPU_IMAGE_SCALING_H
#define __GPU_IMAGE_SCALING_H
#include<math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"../include/imageMatrix.h"
#include "../include/gpuConvolution.cuh"
#include<vector>



// computes integral image of for given image
//		inputImage - inputimage
//returns:
//		ImMatG * - object where integral image is stored

ImMatG * computeIntegralImage(const ImMatG* inputImage);

// computes integral image of input image squared
//		inputImage - inputimage
//returns:
//		ImMatG * - object where integral image is stored
ImMatG * computeIntegralImageSq(const ImMatG *inputImage);

// downsamples input matrix by factor 2
//		inputImage - inputimage to be downsample
//		outputImage - pointer to object where result will be stored
void downsample2(ImMatG * inputImage, ImMatG * outputImage);

// pads border of matrix with zeros
// parameters:
//		inputImage - input matrix to be padded
//		paddedImage - padededmatrix
//		paddingPixel - number of border pixels to be added (per side)
void padArray(const ImMatG *inputImage, ImMatG *paddedImage, int paddingPixels);


// converts patches of image in column format (MATLAB functions)
// PARAMETERS:
//		inputImage - input matrix
//		patchSize - size of each patch
// returns:
//		*ImMatG - pointer to matrix containing result
ImMatG * im2col(const ImMatG *inputImage, int patchSize);

// substracts column vector from each column of matrix
//parameters:
//		inputImage - input matrix
//		outputImage - output matrix
//      vector - vector to be subtracted(must have the same element count as matrix columns)
void bsxfunminusCol(const ImMatG *inputImage, ImMatG *outputImage, double *vector);

// subtracts row vector from each row of matrix
//parameters:
//		inputImage - input matrix
//		outputImage - output matrix
//		vector - vector to be subtracted (must have the same element count as matrix rows)
void bsxfunminusRow(const ImMatG *inputImage, ImMatG *outputImage, double *vector);

// multiplies each element in each row with corresponding element from input vector
//parameters:
//		inputImage - input matrix
//		outputImage - output matrix
//		vector - pointer to vector with wich multiply
void bsxfunmultRow(const ImMatG *inputImage, ImMatG*outputImage, double *vector);

// divides each element in each column with corresponding element from input vector
//parameters:
//		inputImage - input matrix
//		outputImage - output image
//		vector - pointer to vector with values to be divides
void bsxfundivide(const ImMatG *inputImage, ImMatG *outputImage, double *vector);

// divides each element in each row with corresponding element from input vector
// parameters:
//		inputImage - input matrix
//		outputImage - outputImage
//		vector - vector for where values are stored

void bsxfunRowDivide(const ImMatG *inputImage, ImMatG *outputImage, double *vector);

// computes exp() of values stored in matrix
// parameters:
//		input - input matrix
//		output - output matrix
void expElem(const ImMatG *input, ImMatG *output);

// computes maximum value of each colum. works only on two row matrixes
// parameters:
//		input - input matrix
//		output - output matrix
void maxColumnVal(const ImMatG * input, ImMatG * output);

// computes colum sum for each column. work only on two row matrices
// parameters:
//		input - input matrix
//		output - output matrix
void columnSum(const ImMatG *input, ImMatG *output);

// computes means of patches from computed integral image. means are stored in matrix corresponding
// to data formeat returne by im2col
// parameters:
//		inptuImage - integral Image
//		int patchSize - size of square patch patchSize X patchSize
//		meanVector - pointer where result is stored
void meansOfPatches(const ImMatG *inputImage, int patchSize, ImMatG *meanVector);

//computes variances of patches from computed integral image.
//parameters:
//		integralImage - integral image
//		integralImageSq - integral image of squared elements
//		int patchSize - size of the patch 
//		varVec - pointer to where result is stored
//
void variancesOfPatches(const ImMatG *integralImage, const ImMatG *integralImageSq, int patchSize, ImMatG *varVector);

// functions performs bicubic resize of image. currently supports only integer number resize
// parameters:
//		inputImage - image to be resized
//		scaling - scale factor (integer number)
ImMatG * bicubicResize(ImMatG *inputImage, size_t scaling);


// returns bicubic coefficients
// parameters:
//		scaling - scaling factor (integer)
//returns:
//		double * - pointer to scaling coefficients (allocated on host)
double *bicubicCoeffs(size_t scaling);

// rowScan kernel
// parameters:
//		input - input array allocated on device
//		rows - rows of input matrix
//		cols - cols of input matrix
//		output - pointer for output on device matrix
//		sqared - boolean value to indicate if input values should be squared
__global__ void rowScan(double *input, int rows, int cols, double *output, bool sqared);

// class for creation of image pyramid

class ImagePyramidCreator{
	double * kernel_d;	// pointer to kernel allocated on host
	int layers;			// layers number of pyramid layers (scales)
	int kernelSize;		// filter kernel size for anti aliasing filter
	double sigma;		// sigma for gaussian filter
	
	// constructors:
	public:
		// creates pyramid creator
		//		layers - specifies number of layers
		//		kernelSize - specifies kernel size
		//		sigma - specified gaussian filter parameter
		ImagePyramidCreator(int layers, int kernelSize, double sigma);

		// destructor for this class. dealocated gpu memory
		~ImagePyramidCreator();

		// function that creeates pyramid
		//		input - input matrix from which pyramid is built
		// returns:
		//		vector<ImMatG *> where each vector element represents each pyramid layer
		std::vector<ImMatG *> createPyramid(ImMatG * input);

};

#endif