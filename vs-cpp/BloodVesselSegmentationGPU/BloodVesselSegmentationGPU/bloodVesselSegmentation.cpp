#include"bloodVesselSegmentation.h"
#include<stdlib.h>
#include<opencv2/core/core.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/cudafilters.hpp>
#include<iostream>

using namespace std;
using namespace cv;
using namespace cuda;

JNIEXPORT jdoubleArray JNICALL Java_lv_edi_EDI_13DAtA_opencvcudainterface_Compute_gaussianBlur(JNIEnv * env, jobject obj, jdoubleArray input, jint rows, jint cols){
	jsize len = env->GetArrayLength(input);
	jdouble * body = env->GetDoubleArrayElements(input, 0);
	jdoubleArray result = (env)->NewDoubleArray(len);
	
	Mat inputImage(rows, cols, CV_64FC1), filteredImage;
	GpuMat inputImageG, filteredImageG;
	inputImage.data = (uchar *)body;
	cout << inputImage << endl;
	inputImageG.upload(inputImage);

	Ptr<Filter> gaussianFilterP = cuda::createGaussianFilter(CV_64FC1, CV_64FC1, Size(5, 5), 2, 2);
	gaussianFilterP->apply(inputImageG, filteredImageG);

	filteredImageG.download(filteredImage);

	double *buffer = (double *)malloc(sizeof(double)*(rows*cols));
	double * ptr = (double *)filteredImageG.data;
	for (int i = 0; i < len; i++){
		buffer[i] = ptr[i];
	}

	(env)->SetDoubleArrayRegion(result, 0, len, buffer);
	(env)->ReleaseDoubleArrayElements(input, body, 0);
	free(buffer);
	return result;
}