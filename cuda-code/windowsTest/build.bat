nvcc --include-path^
 ..\bloodVesselSegmentation\include,"%OPENCV_INCLUDE%"^
 --library-path ..\bloodVesselSegmentation\bin\,"%CUDA_PATH%\lib\x64","%OPENCV_LIB%"^
 --library cublas,bvsegmentation,cudadevrt,cudart,opencv_core300,opencv_highgui300,opencv_imgproc300,opencv_imgcodecs300^
 .\src\main.cpp --output-file .\bin\mainTest.exe
