#include<iostream>
#include<math.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
	if(argc<3){
		cout <<"Input image and mask must be specified!"<<endl;	
		return 1;
	}
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
