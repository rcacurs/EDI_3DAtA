
nvcc --device-c .\src\gpuBloodVesselSegmentation.cu \src\gpuConvolution.cu .\srcgpuImageScaling.cu .\src\imageMatrix.cu
nvcc --lib gpuBloodVesselSegmentation.obj gpuConvolution.obj gpuImageScaling.obj imageMatrix.obj --output-file segmentation.lib
