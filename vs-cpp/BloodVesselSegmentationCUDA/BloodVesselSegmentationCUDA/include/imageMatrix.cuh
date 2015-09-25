#ifndef __IMAGE_MATRIX_CUH
#define __IMAGE_MATRIX_CUH

#include<Math.h>
//  class that describes grayscale image matrix for computation on GPU using CUDA

class ImMatG{
public:
	int rows;
	int cols;
	double * data_d; // data allocated on CPU memory
	
	// CONSTRUCTORS
	//==========================================================
	ImMatG();  // default constructor (nothing is allocated)
	// constructs image with specified rows, columns and data array
	//		rows: number of rows for image object
	//		cols: number of columnt for image object
	//		data: pointer to memory block containing grayscale image data
	//		bool: flag that indicates if data pointer points to device or host memory
	
	ImMatG(size_t rows, size_t cols, double* data, bool onDeviceMemory);

	// constructs image with specified rows, columns and data array, and allocates memory on GPU
	//		rows: number of rows for image object
	//		cols: number of cols for image object
	//		
	ImMatG(size_t rows, size_t cols);

	// destrucrot of ImMatG object releases GPU memory
	~ImMatG();
	
	// return length of pixell buffer
	size_t getLength(void);

	// returns transposed version of this matrix (new object is created)
	ImMatG * transpose();

	// returns column of matrix
	ImMatG * getColumn();

	// fills matrix row with value
	//		row - row which is filled
	//		value - value which is filled
	void fillRow(size_t row, double value);
};
#endif