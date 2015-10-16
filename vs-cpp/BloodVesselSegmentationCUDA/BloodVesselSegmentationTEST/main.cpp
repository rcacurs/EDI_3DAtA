#include<iostream>
#include<imageMatrix.h>

using namespace std;
int main(){

	cout << "Program for blood vessel segmentaiton test." << endl;
	
	double data[] = { 1, 2, 3, 4, 5, 6 };
	ImMatG * testImage = new ImMatG(2, 3, data, false);
	double *resultData = testImage->getData();

	for (int i = 0; i < testImage->getLength(); i++){
		cout << resultData[i] << endl;
	}

	delete testImage;
	system("pause");
	return 0;
}