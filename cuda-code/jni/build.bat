IF NOT EXIST .\bin mkdir bin

nvcc --include-path "%JAVA_HOME%\include","%JAVA_HOME%\include\win32",^
..\bloodVesselSegmentation\include^
 --library-path ..\bloodVesselSegmentation\bin\,"%CUDA_PATH%\lib\x64"^
 --library cublas,bvsegmentation,cudadevrt,cudart^
 .\src\computeInterface.cpp^
 --shared^
 --machine 64^
 --output-file bin\computeCudaInterface.dll^
 