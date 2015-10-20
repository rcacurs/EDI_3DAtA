#include<iostream>
#include<imageMatrix.h>
#include<gpuBloodVesselSegmentation.cuh>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

//convert to cv matrix

int main(int argc, char *argv[]){
	if(argc<3){
		cout<<"Input image and lung mask should be specified!"<<endl;
		return 1;
	}
	int tick1=0, tick2=0; // FOR TIME MEASUREMENTS
	// === READING TEST IMAGE USING OPENCV =============
	Mat image = imread(argv[1]);
	Mat imageMask = imread(argv[2]);
	Mat imageGS;
	Mat maskGS;
	double threshold = 0.8;
	int iterations = 10;
	cvtColor(imageMask, maskGS, CV_RGB2GRAY);
	cvtColor(image, imageGS, CV_RGB2GRAY);
	maskGS=maskGS/255;
		
	maskGS.convertTo(maskGS, CV_64FC1);
	imageGS.convertTo(imageGS, CV_64FC1);
	// === READING FILTER PARAMETERS =========
	
	ImMatG *dMeans = readCSV("./dMean.csv");
	ImMatG *dCodes = readCSV("./dCodes.csv");
	ImMatG *scaleMean = readCSV("./scaleparamsMean.csv");
	ImMatG *scalesSd = readCSV("./scaleparamsSd.csv");
	ImMatG *model = readCSV("./model.csv");
	// ==== convert cv images to ImMatG *
	ImMatG * imageGPU = new ImMatG(imageGS.rows, imageGS.cols, (double *)(imageGS.data), false);
	ImMatG * maskGPU = new ImMatG(maskGS.rows, maskGS.cols, (double *)(maskGS.data), false);
	
	// ==== PERFORMING BLOOD VESSEL SEGMENTATION ==========
	
	tick1 = getTickCount();
	for(int i=0; i<iterations; i++){
		ImMatG *segmRest = segmentBloodVessels(imageGPU, maskGPU, dCodes, dMeans, scaleMean, model, scalesSd);
		double *dtt = segmRest->getData();
		delete segmRest;
		delete[] dtt;
	}
	tick2 = getTickCount();
	
	cout <<"segmentation average time for" <<iterations<<" iterations: "<<1000*((double)tick2-tick1)/(getTickFrequency()*iterations)<<"[ms]"<<endl;
	
	ImMatG * segmRes = segmentBloodVessels(imageGPU, maskGPU, dCodes, dMeans, scaleMean, model, scalesSd);
	double *dt = segmRes->getData();
	
	cout<<"segm res size: "<<segmRes->rows<<" "<<segmRes->cols<<endl;
	Mat resultMat = Mat(segmRes->rows, segmRes->cols, CV_64FC1, (void*)dt);
	
	Mat output = (imageGS.mul(maskGS))/255;
	Mat outputRGB;
	output.convertTo(output, CV_32FC1);
	cvtColor(output, outputRGB, CV_GRAY2RGB);
	cout<<outputRGB.rows<<" "<<outputRGB.cols<<" "<<outputRGB.total()<<" "<<outputRGB.channels()<<endl;
	for(int i=0; i<(outputRGB.rows)*(outputRGB.cols); i++){
		double val = ((double *)(resultMat.data))[i];
		 if(val>=threshold){
			 ((float *)(outputRGB.data))[i*3]=0;
			 ((float *)(outputRGB.data))[i*3+1]=1.0f;
			 ((float *)(outputRGB.data))[i*3+2]=0;			
		 }
	}
	imshow("Input image ", outputRGB);
	imshow("segmRes", resultMat);
	waitKey(0);
	return 0;
}