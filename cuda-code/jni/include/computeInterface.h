#include <jni.h>
 
#ifndef _COMPUTE_INTERFACE_H
#define _COMPUTE_INTERFACE_H

extern "C" {

/*
 * Class:   
 * Method: 
 * Signature:
 */

JNIEXPORT jdoubleArray JNICALL Java_lv_edi_EDI_13DAtA_opencvcudainterface_Compute_segmentBloodVessels(JNIEnv *env, jobject thisObj,
 jdoubleArray input, jint r1, jint c1,
 jdoubleArray dCodes, jint r2, jint c2,
 jdoubleArray dMeans, jint r3, jint c3,
 jdoubleArray scalesMean, jint r4, jint c4,
 jdoubleArray  model, jint r5, jint c5,
 jdoubleArray scalesSd, jint r6, jint c6);
 
}

#endif