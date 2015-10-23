#include<iostream>
#include"../include/computeInterface.h"
#include<gpuBloodVesselSegmentation.cuh>
#include<imageMatrix.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
extern "C"{
JNIEXPORT jdoubleArray JNICALL Java_lv_edi_EDI_13DAtA_opencvcudainterface_Compute_segmentBloodVessels(JNIEnv *env, jobject thisObj,
 jdoubleArray input, jint r1, jint c1,
 jdoubleArray dCodes, jint r2, jint c2,
 jdoubleArray dMeans, jint r3, jint c3,
 jdoubleArray scalesMean, jint r4, jint c4,
 jdoubleArray  model, jint r5, jint c5,
 jdoubleArray scalesSd, jint r6, jint c6){
		std::cout << "Looks like it works!" << std::endl;
		
		// get pointers to data
		jdouble *inputPtr = env->GetDoubleArrayElements(input, 0);
		jdouble *dCodesPtr = env->GetDoubleArrayElements(dCodes, 0);
		jdouble *dMeansPtr = env->GetDoubleArrayElements(dMeans, 0);
		jdouble *scalesMeanPtr = env->GetDoubleArrayElements(scalesMean, 0);
		jdouble *modelPtr = env->GetDoubleArrayElements(model, 0);
		jdouble *scalesSdPtr = env->GetDoubleArrayElements(scalesSd, 0);
		// for(int i=0; i<env->GetArrayLength(scalesSd); i++){
			// std::cout<<scalesSdPtr[i]<<std::endl;
		// }
		// move data to GPU by using ImMatG *
		ImMatG *inputG = new ImMatG(r1, c1, inputPtr, false);
		ImMatG *dCodesG = new ImMatG(r2, c2, dCodesPtr, false);
		ImMatG *dMeansG = new ImMatG(r3, c3, dMeansPtr, false);
		ImMatG *scalesMeanG = new ImMatG(r4, c4, scalesMeanPtr, false);
		ImMatG *modelG = new ImMatG(r5, c5, modelPtr, false);
		ImMatG *scalesSdG = new ImMatG(r6, c6, scalesSdPtr, false);
		
		ImMatG *segmRes = segmentBloodVessels(inputG, dCodesG, dMeansG, scalesMeanG, modelG, scalesSdG);
		double *dt = segmRes->getData();
		Mat inputImage(r1,c1,CV_64FC1,(void *)inputPtr);
		imshow("input image", inputImage/255);

		std::cout<<"Segm Res Length "<<segmRes->getLength()<<std::endl;
		
		for(int i=0; i<segmRes->getLength(); i++){
			if(dt[i]>=0.85){
				dt[i]=1.0;
			} else{
				dt[i]=0;
			}
		}
		
		Mat image(segmRes->rows, segmRes->cols, CV_64FC1, (void *)dt);
		imshow("test", image);
		waitKey(0);
		jdoubleArray result = (env)->NewDoubleArray(segmRes->getLength());
		(env)->SetDoubleArrayRegion(result, 0, segmRes->getLength(), dt);
		
		(env)->ReleaseDoubleArrayElements(input, inputPtr, 0);
		(env)->ReleaseDoubleArrayElements(dCodes, dCodesPtr, 0);
		(env)->ReleaseDoubleArrayElements(dMeans, dMeansPtr, 0);
		(env)->ReleaseDoubleArrayElements(scalesMean, scalesMeanPtr, 0);
		(env)->ReleaseDoubleArrayElements(model, modelPtr, 0);
		(env)->ReleaseDoubleArrayElements(scalesSd, scalesSdPtr, 0);
		delete segmRes;
		delete[] dt;
		delete inputG;
		delete dCodesG;
		delete dMeansG;
		delete scalesMeanG;
		delete modelG;
		delete scalesSdG;
		return result;
}
}