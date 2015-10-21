#include<iostream>
#include<math.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<imageMatrix.h>
#include<string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
	if(argc<3){
		cout <<"Input image and mask must be specified!"<<endl;	
		return 1;
	}
	// LOAD FILTER PARAMETERS
	
	string parameterFolder = "../../../filter-classifier-parameters/";
	ImMatG *dMeans = readCSV(parameterFolder+"dMean.csv");
	ImMatG *dCodes = readCSV(parameterFolder+"dCodes.csv");
	ImMatG *scaleMean = readCSV(parameterFolder+"scaleparamsMean.csv");
	ImMatG *scalesSd = readCSV(parameterFolder+"scaleparamsSd.csv");
	ImMatG *model = readCSV(parameterFolder+"model.csv");

	cout<<"means "<<dMeans->rows<<" "<<dMeans->cols<<endl;
	cout<<"codes "<<dCodes->rows<<" "<<dCodes->cols<<endl;
	cout<<"scaleMean "<<scaleMean->rows<<" "<<scaleMean->cols<<endl;
	cout<<"scale sd "<<scalesSd->rows<<" "<<scalesSd->cols<<endl;
	cout<<"model "<<model->rows<<" "<<model->cols<<endl;

	Mat image=imread(argv[1]);
	Mat mask=imread(argv[2]);

	Mat test(5, 5, CV_32FC1);
	cout<<"works!"<<endl;
	imshow("Test window",image);
	imshow("Mask image", mask);
	cout<<test<<endl;
	waitKey(0);
	return 0;
}
