#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void main(){
	Mat matrix(5, 5, CV_8UC1);
	cout << "Works" << endl;
	cout << matrix<<endl;
	system("pause");
}