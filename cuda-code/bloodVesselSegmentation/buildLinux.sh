#!/bin/bash
if [ ! -d "./bin" ]; then
	mkdir bin
fi

nvcc --device-c ./src/gpuBloodVesselSegmentation.cu --output-file ./bin/gpuBloodVesselSegmentation.o
nvcc --device-c ./src/gpuConvolution.cu --output-file ./bin/gpuConvolution.o
nvcc --device-c ./src/gpuImageScaling.cu --output-file ./bin/gpuImageScaling.o
nvcc --device-c ./src/imageMatrix.cu --output-file ./bin/imageMatrix.o
nvcc --lib --output-file ./bin/bvsegmentation.a\
 ./bin/imageMatrix.o\
 ./bin/gpuConvolution.o\
 ./bin/gpuImageScaling.o\
 ./gin/gpuBloodVesselSegmentation.o
