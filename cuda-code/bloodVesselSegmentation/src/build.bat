call vcvars64.bat
nvcc --device-c gpuBloodVesselSegmentation.cu gpuConvolution.cu gpuImageScaling.cu imageMatrix.cu
nvcc --lib gpuBloodVesselSegmentation.obj gpuConvolution.obj gpuImageScaling.obj imageMatrix.obj --output-file segmentation.lib
