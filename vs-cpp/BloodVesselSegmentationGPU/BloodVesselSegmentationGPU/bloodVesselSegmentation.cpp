#include"bloodVesselSegmentation.h"
#include<stdlib.h>
#include<opencv2/core/core.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/imgproc.hpp>
#include<Math.h>
#include<iostream>
#include<vector>

using namespace std;
using std::vector;
using namespace cv;
using namespace cuda;
int tick1, tick2;
JNIEXPORT jdoubleArray JNICALL Java_lv_edi_EDI_13DAtA_opencvcudainterface_Compute_gaussianBlur(JNIEnv * env, jobject obj, jdoubleArray input, jint rows, jint cols){

	tick1 = getTickCount();
	jsize len = env->GetArrayLength(input);
	jdouble * body = env->GetDoubleArrayElements(input, 0);
	jdoubleArray result = (env)->NewDoubleArray(len);
	tick2 = getTickCount();

	tick1 = getTickCount();
	Mat inputImage(rows, cols, CV_32FC1), filteredImage;
	for (int i = 0; i < rows*cols; i++){
		((float*)inputImage.data)[i] = (float)body[i];
	}
	tick2 = getTickCount();
	tick1 = getTickCount();
	GpuMat inputImageG, filteredImageG;
	inputImageG.upload(inputImage);
	tick2 = getTickCount();
	Ptr<Filter> gaussianFilterP = cuda::createGaussianFilter(CV_32FC1, CV_32FC1, Size(5, 5), 2, 2);
	gaussianFilterP->apply(inputImageG, filteredImageG);

	filteredImageG.download(filteredImage);

	double *buffer = (double *)malloc(sizeof(double)*(rows*cols));

	float * ptr = (float *)filteredImage.data;
	for (int i = 0; i < rows*cols; i++){
		buffer[i] = (double)ptr[i];
	}

	(env)->SetDoubleArrayRegion(result, 0, len, buffer);
	(env)->ReleaseDoubleArrayElements(input, body, 0);
	free(buffer);
	return result;
}

void gaussianPyramid(cv::cuda::GpuMat & input, std::vector<cv::cuda::GpuMat> & result,  int numLayers){
	result.clear();
	GpuMat filtered;
	Ptr<Filter> gaussianFilterP = cuda::createGaussianFilter(CV_32FC1, CV_32FC1, Size(5, 5), 1, 1);
	gaussianFilterP->apply(input, filtered);
	result.push_back(filtered);
	for (int i = 1; i < numLayers; i++){
		GpuMat mat;
		cuda::pyrDown(result[i - 1], mat);
		result.push_back(mat);
	}
};

void im2col(GpuMat & input, GpuMat & result, int patchSize){

	int pixNum = patchSize*patchSize;
	int count = 0;
	int colMax = input.cols - 2 * (patchSize / 2);
	int rowMax = input.rows - 2 * (patchSize / 2);
	int patchesNum = colMax*rowMax;

	GpuMat patch;
	Mat inputMat;

	input.download(inputMat);

	Mat patchMat;
	Mat allPatches(patchesNum, patchSize*patchSize, CV_32FC1);
	Mat rowMat;

	
	for (int col = 0; col < colMax; col++){
		for (int row = 0; row < rowMax; row++){

			float * pAll = (float *)allPatches.ptr(count);

			for (int i = 0; i < patchSize; i++){
				for (int j = 0; j < patchSize; j++){

					float * p = (float *)inputMat.ptr(row + j);
					pAll[i*patchSize + j] = p[col + i];
				}
			}
			count++;
		}
	}

	result.upload(allPatches);
}

void filterSMF(GpuMat & input, GpuMat & output, int patchSize, GpuMat & codesD, GpuMat &  meanD){
	GpuMat borderedInput;
	cuda::copyMakeBorder(input, borderedInput, patchSize/2, patchSize/2, patchSize/2, patchSize/2, BORDER_CONSTANT, 0);
	GpuMat allPatches;
	im2col(borderedInput, allPatches, patchSize);
	Mat allPatchesMat;
	Mat tempMatMat;
	Mat codesDMat;
	codesD.download(codesDMat);
	allPatches.download(allPatchesMat);
	Mat inputMat, integralMat(1, 1, CV_32FC1), integralSqMat(1, 1, CV_32FC1);
	borderedInput.download(inputMat);
	cv::integral(inputMat, integralMat, integralSqMat, CV_32F);
	tick2 = getTickCount();
	
	GpuMat tempMat;

	float mean, var;
	float sumI, sumIsq;
	Mat meansMat;
	meanD.download(meansMat);
	int patchPixels = patchSize*patchSize;
	int colindex, rowindex, Ar, Ac, Br, Bc, Cr, Cc, Dr, Dc;

	
	for (int i = 0; i < allPatches.rows; i++){
		
		tempMatMat = allPatchesMat.row(i);//(Range(i, i + 1), Range(0, allPatches.cols));

		colindex = i / input.cols;
		rowindex = i % input.cols;
		Ar = rowindex;
		Ac = colindex;
		Br = Ar;
		Bc = Ac + patchSize;
		Cr = Ar + patchSize;
		Cc = Ac;
		Dr = Ar + patchSize;
		Dc = Ac + patchSize;
		sumI = ((float*)integralMat.ptr(Ar))[Ac] + ((float*)integralMat.ptr(Dr))[Dc] - ((float*)integralMat.ptr(Cr))[Cc] - ((float*)integralMat.ptr(Br))[Bc];
		sumIsq = float(((double*)integralSqMat.ptr(Ar))[Ac] + ((double*)integralSqMat.ptr(Dr))[Dc] - ((double*)integralSqMat.ptr(Cr))[Cc] - ((double*)integralSqMat.ptr(Br))[Bc]);
		mean = sumI / (patchPixels);
		//var = (sumIsq - (2 * sumI*mean) + patchPixels*(mean*mean)) / (patchPixels-1);
		var = (sumIsq - sumI*sumI/patchPixels) / (patchPixels - 1);
		var = sqrt(var + 10);
		if (var == 0) var = 1 / 1000;
		cv::subtract(tempMatMat, mean, tempMatMat);
		cv::divide(tempMatMat, var, tempMatMat);
		//float meanf = ((float *)meansMat.data)[i];
		cv::subtract(tempMatMat,meansMat, tempMatMat);

	}
	GpuMat  empty;
	Mat filteredMat;
	allPatches.upload(allPatchesMat);
	cv::cuda::gemm(allPatches, codesD, 1, empty, 0, output);
	
}

void extractFeatures(GpuMat & input, GpuMat & output, int patchSize, GpuMat & codesD, GpuMat &  meanD){

	int ticki1 = getCPUTickCount();
	vector<GpuMat> pyramid;
	gaussianPyramid(input, pyramid, 6);

	Mat features;// (pyramid.size()*codesD.cols, input.rows*input.cols, CV_32FC1);
	int featureCount = 0;
	Mat temp1, tempT;
	GpuMat upscaledImage, oneFilter;
	int ticki2 = getCPUTickCount();
	
	for (int i = 0; i < pyramid.size(); i++){
		ticki1 = getCPUTickCount();
		GpuMat filteredLayer;
		Mat filteredLayerMat;
		filterSMF(pyramid[i], filteredLayer, 5, codesD, meanD);
		filteredLayer.download(filteredLayerMat);
		ticki2 = getCPUTickCount();

		for (int j = 0; j < filteredLayer.cols; j++){
			temp1 = filteredLayerMat.col(j).clone();
			
			temp1=temp1.reshape(0, pyramid[i].rows);
			cv::transpose(temp1, tempT);
			oneFilter.upload(tempT);
			if (i > 0){
				cuda::resize(oneFilter, upscaledImage, Size(0, 0), pow(2, i), pow(2, i), INTER_CUBIC);
			}
			else{
				upscaledImage = oneFilter;
			}
			
			upscaledImage.download(temp1);
			
			ticki1 = getCPUTickCount();
			features.push_back(temp1.reshape(1, 1));
			ticki2 = getCPUTickCount();
			//cout << "C++ one feature copy time: " << (ticki2 - ticki1) / getTickFrequency() << endl;
			featureCount++;
			oneFilter.release();
			temp1.release();
			upscaledImage.release();
			temp1.release();
			tempT.release();
		}
	}
	features.push_back(Mat(1, features.cols, CV_32FC1, 1));
	output.upload(features);
}

void classify(GpuMat & inputFeatures, GpuMat & output, GpuMat & maskImage, GpuMat & model, Mat & mean, Mat & Sd){
	//masking image
	GpuMat submat;
	for (int i = 0; i < inputFeatures.rows-1; i++){
		submat = inputFeatures(Range(i, i + 1), Range(0, inputFeatures.cols));
		cuda::multiply(submat, maskImage, submat);
	};

	for (int i = 0; i < inputFeatures.rows-1; i++){
		submat = inputFeatures(Range(i, i + 1), Range(0, inputFeatures.cols));
		cuda::subtract(submat, ((float *)mean.data)[i], submat);
		cuda::divide(submat, ((float *)Sd.data)[i], submat);
	}
	GpuMat featuresNorm, dummy;

	cuda::gemm(model, inputFeatures, 1, dummy, 0, featuresNorm);
	// find maximum values of featuresNorm
	GpuMat maxVec;
	cuda::reduce(featuresNorm, maxVec, 0, REDUCE_MAX);

	for (int i = 0; i < featuresNorm.rows; i++){
		submat = featuresNorm(Range(i, i + 1), Range(0, featuresNorm.cols));
		cuda::subtract(submat, maxVec, submat);
	}
	cuda::exp(featuresNorm, featuresNorm);
	GpuMat sumVec;
	cuda::reduce(featuresNorm, sumVec, 0, REDUCE_SUM);
	submat = featuresNorm(Range(1, 2), Range(0, featuresNorm.cols));
	GpuMat result;
	cuda::divide(submat, sumVec, output);
}

JNIEXPORT jdoubleArray JNICALL Java_lv_edi_EDI_13DAtA_opencvcudainterface_Compute_segmentBloodVessels(JNIEnv * env, jobject obj, jdoubleArray input, jint rows, jint cols, jdoubleArray codes, jdoubleArray means, jint patchSize, jint numberOfFilters, jdoubleArray model, jdoubleArray scaleparamsMean, jdoubleArray scaleparamsSd, jdoubleArray imageMask){
	jsize len = env->GetArrayLength(input);
	jsize lenCodes = env->GetArrayLength(codes);
	jsize lenMeans = env->GetArrayLength(means);
	jsize lenModel = env->GetArrayLength(model);
	jsize lenScaleParamsMean = env->GetArrayLength(scaleparamsMean);
	jsize lenScaleParamsSd = env->GetArrayLength(scaleparamsSd);
	jsize lenMask = env->GetArrayLength(imageMask);
	jdouble * codesBody = env->GetDoubleArrayElements(codes, 0);
	jdouble * meansBody = env->GetDoubleArrayElements(means, 0);
	jdouble * modelBody = env->GetDoubleArrayElements(model, 0);
	jdouble * maskBody = env->GetDoubleArrayElements(imageMask, 0);
	jdouble * scaleParamsMeanBody = env->GetDoubleArrayElements(scaleparamsMean, 0);
	jdouble * scaleParamsSdBody = env->GetDoubleArrayElements(scaleparamsSd, 0);
	jdouble * body = env->GetDoubleArrayElements(input, 0);

	tick1 = getCPUTickCount();
	Mat inputImage(rows, cols, CV_32FC1);

	GpuMat inputImageG;
	for (int i = 0; i < rows*cols; i++){ // fill Mat object
		((float*)inputImage.data)[i] = (float)body[i];
	}
	Mat codesMat(patchSize*patchSize, numberOfFilters, CV_32FC1);
	for (int i = 0; i < lenCodes; i++){
		((float*)codesMat.data)[i] = (float)codesBody[i];
	}
	Mat meansMat(1, patchSize*patchSize, CV_32FC1);
	for (int i = 0; i < lenMeans; i++){
		((float*)meansMat.data)[i] = (float)meansBody[i];
	}
	Mat modelMat(2, lenModel / 2, CV_32FC1);
	for (int i = 0; i < lenModel; i++){
		((float*)modelMat.data)[i] = (float)modelBody[i];
	}
	Mat scaleParamsMean(1, lenScaleParamsMean, CV_32FC1);
	for (int i = 0; i < lenScaleParamsMean; i++){
		((float*)scaleParamsMean.data)[i] = (float)scaleParamsMeanBody[i];
	}
	Mat scaleParamsSd(1, lenScaleParamsSd, CV_32FC1);
	for (int i = 0; i < lenScaleParamsSd; i++){
		((float*)scaleParamsSd.data)[i] = (float)scaleParamsSdBody[i];
	}
	Mat imageMaskMat(1, lenMask, CV_32FC1);
	for (int i = 0; i < lenMask; i++){
		((float*)imageMaskMat.data)[i] = (float)maskBody[i];
	}
	
	GpuMat codesG, meansG, modelG, imageMaskG;
	codesG.upload(codesMat);
	meansG.upload(meansMat);
	modelG.upload(modelMat);
	imageMaskG.upload(imageMaskMat);
	//scaleParamsMeanG.upload(scaleParamsMean);
	//scaleParamsSdG.upload(scaleParamsSdG);

	Mat tempMat;
	tick2 = getCPUTickCount();
	inputImageG.upload(inputImage);
	GpuMat features;
	tick2 = getCPUTickCount();
	tick1 = getCPUTickCount();
	extractFeatures(inputImageG, features, 5, codesG, meansG);
	tick2 = getCPUTickCount();
	
	GpuMat segmentationResult;
	Mat segmentationResMat;

	classify(features, segmentationResult, imageMaskG, modelG, scaleParamsMean, scaleParamsSd);
	segmentationResult.download(segmentationResMat);

	//Mat segmentated;
	segmentationResult.download(segmentationResMat);
	//segmentationResMat=segmentationResMat.reshape(1, inputImage.rows);
	tick1 = getCPUTickCount();
	GpuMat feature = features(Range(33, 34), Range(0, features.cols));
	Mat featureMat;
	Mat featureMat2;
	Mat converted;
	feature.download(featureMat);

	double *buffer = (double *)malloc(sizeof(double)*(segmentationResMat.total()));

	float * ptr = (float *)segmentationResMat.data;
	for (int i = 0; i < segmentationResMat.total(); i++){
		buffer[i] = (double)(ptr[i]);
	}

	jdoubleArray result = (env)->NewDoubleArray(segmentationResMat.total());
	(env)->SetDoubleArrayRegion(result, 0, segmentationResMat.total(), buffer);
	(env)->ReleaseDoubleArrayElements(input, body, 0);
	tick2 = getCPUTickCount();
	free(buffer);
	tick2 = getCPUTickCount();

	return result;
}
