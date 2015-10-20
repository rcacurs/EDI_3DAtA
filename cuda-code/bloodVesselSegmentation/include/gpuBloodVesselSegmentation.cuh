#ifndef __GPU_BLOOD_VESSEL_SEGMENTATION_H
#define __GPU_BLOOD_VESSEL_SEGMENTATION_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <vector>
#include"../include/imageMatrix.h"
#include"../include/gpuImageScaling.cuh"

// performs bloodvessel segmentation from CT scan
// paramgeters:
//		inputLayerG - grayscale tomography scan
//		dCodes - filter patameter codes
//		dMeans - filter parameter means
//		handle - handle for cublas 

ImMatG * extractFeatures(ImMatG *inputLayerG, ImMatG *dCodes, ImMatG *dMeans, cublasHandle_t handle);

// classifies features min max classifier
// parameters:
//		features - matrix containing features
//		mask - block containing image mask
//		scaleMean - classifier parameters
//		model - classifier parameter
//		scalesSd - classifier parameters
//		handle - cublas context handle

ImMatG * classify(ImMatG * features, int rows, int cols, ImMatG *scaleMean, ImMatG *model, ImMatG *scalesSd, cublasHandle_t handle);


ImMatG *segmentBloodVessels(ImMatG *inputLayerG, ImMatG *dCodes, ImMatG *dMeans, ImMatG *scaleMean, ImMatG *model, ImMatG *scalesSd);

#endif