#include<iostream>
#include<imageMatrix.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
int main(){
	double data[6]={1, 2, 3, 4, 5, 6};
	
	ImMatG *im = new ImMatG(3, 2, data, false);
	double *resData = im->getData();
	for(int i=0; i<im->getLength(); i++){
		cout<<resData[i]<<endl;
	}
	Mat test=Mat::eye(5, 5, CV_32FC1);
	cout<<test<<endl;
	delete im;
	return 0;
}