nvcc --device-c .\src\gpuBloodVesselSegmentation.cu --output-file .\bin\gpuBloodVesselSegmentation.obj
nvcc --device-c .\src\gpuConvolution.cu --output-file .\bin\gpuConvolution.obj
nvcc --device-c .\src\gpuImageScaling.cu --output-file .\bin\gpuImageScaling.obj
nvcc --device-c .\src\imageMatrix.cu --output-file .\bin\imageMatrix.obj
nvcc --lib --output-file .\bin\bvsegmentation.lib^
 .\bin\imageMatrix.obj^
 .\bin\gpuConvolution.obj^
 .\bin\gpuImageScaling.obj^
 .\bin\gpuBloodVesselSegmentation.obj
 
