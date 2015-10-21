if [ ! -d "./bin" ]; then
	mkdir bin
fi
nvcc --include-path\
  ../bloodVesselSegmentation/include,$OPENCV_INCLUDE\
 --library-path ../bloodVesselSegmentation/bin,$OPENCV_LIB,$CUDA_PATH/lib64\
 --library cudadevrt,cudart,cublas,bvsegmentation,opencv_core,opencv_imgcodecs,opencv_highgui\
 --output-file ./bin/mainTest ./src/main.cpp

