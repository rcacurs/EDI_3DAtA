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
	cout << "Preparing function parameters for c++ time: " << (tick2 - tick1) / (getTickFrequency()) <<" [s]"<< endl;

	tick1 = getTickCount();
	Mat inputImage(rows, cols, CV_32FC1), filteredImage;
	for (int i = 0; i < rows*cols; i++){
		((float*)inputImage.data)[i] = (float)body[i];
	}
	tick2 = getTickCount();
	cout << "Copying data to mat type" << (tick2-tick1)/getTickFrequency()<<" [s]"<<endl;
	tick1 = getTickCount();
	GpuMat inputImageG, filteredImageG;
	inputImageG.upload(inputImage);
	tick2 = getTickCount();
	cout << "Copying data to gpu matrix type: " << (tick2 - tick1) / getTickFrequency() << endl;
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

		//cout << "Submatrix extraction time: " << (tick22 - tick12)/getTickFrequency() << endl;

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
		var = (sumIsq - (2 * sumI*mean) + patchPixels*(mean*mean)) / (patchPixels - 1);
		var = sqrt(var + 10);
		if (var == 0) var = 1 / 1000;
		cv::subtract(tempMatMat, mean, tempMatMat);
		//cout << "normalisation1: " << (tick22 - tick12) / getTickFrequency() << endl;
		cv::divide(tempMatMat, var, tempMatMat);
		//cout << "normalisation2: " << (tick22 - tick12) / getTickFrequency() << endl;
		//float meanf = ((float *)meansMat.data)[i];
		cv::subtract(tempMatMat,meansMat, tempMatMat);
		//cout << "normalisation3: " << (tick22 - tick12) / getTickFrequency() << endl;
	}
	//cout << "normalisation time: " << (tick2 - tick1) / getTickFrequency() << endl;
	GpuMat  empty;
	Mat filteredMat;
	//cout << "preparing to multiply matrices" << endl;
	allPatches.upload(allPatchesMat);
	cv::cuda::gemm(allPatches, codesD, 1, empty, 0, output);
	


}

void extractFeatures(GpuMat & input, GpuMat & output, int patchSize, GpuMat & codesD, GpuMat &  meanD){

	int ticki1 = getCPUTickCount();
	vector<GpuMat> pyramid;
	gaussianPyramid(input, pyramid, 6);

	Mat features;// (pyramid.size()*codesD.cols, input.rows*input.cols, CV_32FC1);
	cout << "features size: " << features.cols << " " << features.rows << endl;
	int featureCount = 0;
	Mat temp1, tempT;
	GpuMat upscaledImage, oneFilter;
	int ticki2 = getCPUTickCount();
	
	cout << "C++ pyramid computation time: " << (ticki2 - ticki1)/getTickFrequency() << endl;
	for (int i = 0; i < pyramid.size(); i++){
		ticki1 = getCPUTickCount();
		GpuMat filteredLayer;
		Mat filteredLayerMat;
		filterSMF(pyramid[i], filteredLayer, 5, codesD, meanD);
		filteredLayer.download(filteredLayerMat);
		ticki2 = getCPUTickCount();

		cout << "C++ one scale filter time " << (ticki2 - ticki1) / getTickFrequency() << endl;
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
			//float * p = (float *)(features.ptr(featureCount));
			//for (int z = 0; z < temp1.total(); z++){
			//	p[z] = temp1.at<float>(z/input.cols, z%input.cols);
			//}
			features.push_back(temp1.reshape(1, 1));
			ticki2 = getCPUTickCount();
			cout << "C++ one feature copy time: " << (ticki2 - ticki1) / getTickFrequency() << endl;
			featureCount++;
			oneFilter.release();
			temp1.release();
			upscaledImage.release();
			temp1.release();
			tempT.release();
		}
	}
	output.upload(features);
}

JNIEXPORT jdoubleArray JNICALL Java_lv_edi_EDI_13DAtA_opencvcudainterface_Compute_segmentBloodVessels(JNIEnv * env, jobject obj, jdoubleArray input, jint rows, jint cols, jdoubleArray codes, jdoubleArray means, jint patchSize, jint numberOfFilters){
	jsize len = env->GetArrayLength(input);
	jsize lenCodes = env->GetArrayLength(codes);
	jsize lenMeans = env->GetArrayLength(means);
	jdouble * codesBody = env->GetDoubleArrayElements(codes, 0);
	jdouble * meansBody = env->GetDoubleArrayElements(means, 0);
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
	GpuMat codesG, meansG;
	codesG.upload(codesMat);
	meansG.upload(meansMat);

	Mat tempMat;
	tick2 = getCPUTickCount();
	//cout << "Data prep time: " << (tick2 - tick1) / getTickFrequency() << endl;
	inputImageG.upload(inputImage);
	GpuMat features;
	tick2 = getCPUTickCount();
	cout << "C++ data preperation time: " << (tick2-tick1)/getTickFrequency()<<endl;
	tick1 = getCPUTickCount();
	extractFeatures(inputImageG, features, 5, codesG, meansG);
	tick2 = getCPUTickCount();
	cout << "C++ feature extractor time: " << (tick2 - tick1) / getTickFrequency() << endl;
	
	tick1 = getCPUTickCount();
	GpuMat feature = features(Range(32, 33), Range(0, features.cols));
	Mat featureMat;
	Mat featureMat2;
	Mat converted;
	feature.download(featureMat);
	featureMat2 = featureMat.reshape(0, inputImage.rows);

	double *buffer = (double *)malloc(sizeof(double)*(featureMat.total()));

	float * ptr = (float *)featureMat.data;
	for (int i = 0; i < featureMat.total(); i++){
		buffer[i] = (double)(ptr[i]);
	}

	jdoubleArray result = (env)->NewDoubleArray(featureMat.total());
	(env)->SetDoubleArrayRegion(result, 0, featureMat.total(), buffer);
	(env)->ReleaseDoubleArrayElements(input, body, 0);
	tick2 = getCPUTickCount();
	//cout << "OutputPreperation time: " << (tick2 - tick1) / getTickFrequency() << endl;
	free(buffer);
	tick2 = getCPUTickCount();

	cout << "C++ result data preperation time: " << (tick2 - tick1) / getTickFrequency() << endl;
	return result;
}
