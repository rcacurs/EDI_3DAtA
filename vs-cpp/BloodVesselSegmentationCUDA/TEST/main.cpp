#include<iostream>
#include<imageMatrix.h>

using namespace std;

int main(){
	double data[] = { 1, 2, 3, 4, 5, 6 };
	ImMatG * testMat = new ImMatG(3, 2, data, false);

	double *dataRes = testMat->getData();
	for (int i = 0; i < testMat->getLength(); i++){
		cout << dataRes[i] << endl;
	}
	delete testMat;
	return 0;
}