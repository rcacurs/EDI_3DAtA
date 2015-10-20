#include<iostream>
#include<imageMatrix.h>
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
	
	// === READING TEST IMAGE USING OPENCV =============
	Mat image = imread(argv[1]);
	Mat imageMask = imread(argv[2]);
	Mat imageGS;
	Mat maskGS;
	
	cvtColor(imageMask, maskGS, CV_RGB2GRAY);
	cvtColor(image, imageGS, CV_RGB2GRAY);
	maskGS=maskGS/255;
		
	maskGS.convertTo(maskGS, CV_64FC1);
	imageGS.convertTo(imageGS, CV_64FC1);
	// === READING FILTER PARAMETERS =========
	
	ImMatG * dMeans = readCSV("./dMean.csv");
	double *dt = dMeans->getData();
	cout<<dMeans->getLength()<<endl;
	for(int i=0; i<dMeans->getLength(); i++){
		cout<<dt[i]<<" "<<endl;
	}
	
	Mat output = (imageGS.mul(maskGS))/255;
	imshow("Input image ", output);
	waitKey(0);
	return 0;
}