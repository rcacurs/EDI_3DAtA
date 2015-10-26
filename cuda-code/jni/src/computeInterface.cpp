#include<iostream>
#include"../include/computeInterface.h"
#include<gpuBloodVesselSegmentation.cuh>
#include<imageMatrix.h>

using namespace std;
extern "C"{
JNIEXPORT jdoubleArray JNICALL Java_lv_edi_EDI_13DAtA_opencvcudainterface_Compute_segmentBloodVessels(JNIEnv *env, jobject thisObj,
 jdoubleArray input, jint r1, jint c1,
 jdoubleArray dCodes, jint r2, jint c2,
 jdoubleArray dMeans, jint r3, jint c3,
 jdoubleArray scalesMean, jint r4, jint c4,
 jdoubleArray  model, jint r5, jint c5,
 jdoubleArray scalesSd, jint r6, jint c6){
		
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
		ImMatG *inputG = new ImMatG(r1, c1, (double *)inputPtr, false);
		ImMatG *dCodesG = new ImMatG(r2, c2, (double *)dCodesPtr, false);
		ImMatG *dMeansG = new ImMatG(r3, c3, (double *)dMeansPtr, false);
		ImMatG *scalesMeanG = new ImMatG(r4, c4, (double *)scalesMeanPtr, false);
		ImMatG *modelG = new ImMatG(r5, c5, (double *)modelPtr, false);
		ImMatG *scalesSdG = new ImMatG(r6, c6, (double *)scalesSdPtr, false);

												
		ImMatG *segmRes = segmentBloodVessels(inputG, dCodesG, dMeansG, scalesMeanG, modelG, scalesSdG);
		double *dt = segmRes->getData();

		
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