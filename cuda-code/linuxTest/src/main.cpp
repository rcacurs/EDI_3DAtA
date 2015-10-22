#include<iostream>
#include<math.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<imageMatrix.h>
#include<gpuBloodVesselSegmentation.cuh>
#include<string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
	long int iterations = 100;
	if(argc<3){
		cout <<"Input image and mask must be specified!"<<endl;	
		return 1;
	}
	// LOAD FILTER PARAMETERS
	long int tick1,tick2;
	string parameterFolder = "../../../filter-classifier-parameters/";
	ImMatG *dMeans = readCSV(parameterFolder+"dMean.csv");
	ImMatG *dCodes = readCSV(parameterFolder+"dCodes.csv");
	ImMatG *scaleMean = readCSV(parameterFolder+"scaleparamsMean.csv");
	ImMatG *scalesSd = readCSV(parameterFolder+"scaleparamsSd.csv");
	ImMatG *model = readCSV(parameterFolder+"model.csv");

//	cout<<"means "<<dMeans->rows<<" "<<dMeans->cols<<endl;
//	cout<<"codes "<<dCodes->rows<<" "<<dCodes->cols<<endl;
//	cout<<"scaleMean "<<scaleMean->rows<<" "<<scaleMean->cols<<endl;
//	cout<<"scale sd "<<scalesSd->rows<<" "<<scalesSd->cols<<endl;
//	cout<<"model "<<model->rows<<" "<<model->cols<<endl;
	
	// LOAD IMAGE AND MASK
	Mat image=imread(argv[1]);
	Mat mask=imread(argv[2]);
	Mat maskGS, imageGS;
	cvtColor(mask, maskGS, CV_RGB2GRAY);
	cvtColor(image, imageGS, CV_RGB2GRAY);
	maskGS=maskGS/255;

		
	maskGS.convertTo(maskGS, CV_64FC1);
	imageGS.convertTo(imageGS, CV_64FC1);

	// LOAD DATA TO ImMatG
	ImMatG * imageGPU = new ImMatG(imageGS.rows, imageGS.cols, (double *)(imageGS.data), false);
	ImMatG * maskGPU = new ImMatG(maskGS.rows, maskGS.cols, (double *)(maskGS.data), false);
	
	ImMatG *segmRes;
	double *dt;
	tick1 = getTickCount();
	for(int i=0; i<iterations; i++){
		segmRes = segmentBloodVessels(imageGPU, dCodes, dMeans, scaleMean, model, scalesSd);
		dt = segmRes->getData();
		if(i<(iterations-1)){
			delete[] dt;	
			delete segmRes;	
		}
	}
	tick2 = getTickCount();
	double time = (1000*((double)tick2-tick1)/getTickFrequency())/iterations;
	cout <<"One layer segmentation time: "<<time<<" [ms]"<<endl;

	Mat resultMat = Mat(segmRes->rows, segmRes->cols, CV_64FC1, (void*)dt);
	resultMat = resultMat.mul(maskGS);
	for(int i=0; i<image.total(); i++){
		double val = ((double *)resultMat.data)[i];
		if(val>=0.95){
			image.data[i*3]=0;
			image.data[i*3+1]=255;
			image.data[i*3+2]=0;
		}
	}

	imshow("Test window",image);
	imshow("Mask image", mask);
	imshow("Segmentation result: ",resultMat.mul(maskGS));

	delete[] dt;
	delete segmRes;
	waitKey(0);
	return 0;
}
