IF NOT EXIST .\bin mkdir bin

nvcc -Xcompiler /wd4819 --device-c .\src\gpuBloodVesselSegmentation.cu --output-file .\bin\gpuBloodVesselSegmentation.obj
nvcc -Xcompiler /wd4819 --device-c .\src\gpuConvolution.cu --output-file .\bin\gpuConvolution.obj
nvcc -Xcompiler /wd4819 --device-c .\src\gpuImageScaling.cu --output-file .\bin\gpuImageScaling.obj
nvcc -Xcompiler /wd4819 --device-c .\src\imageMatrix.cu --output-file .\bin\imageMatrix.obj
nvcc -Xcompiler /wd4819 --lib --output-file .\bin\bvsegmentation.lib^
 .\bin\imageMatrix.obj^
 .\bin\gpuConvolution.obj^
 .\bin\gpuImageScaling.obj^
 .\bin\gpuBloodVesselSegmentation.obj
 
 
