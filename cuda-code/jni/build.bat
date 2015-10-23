nvcc --include-path "%JAVA_HOME%\include","%JAVA_HOME%\include\win32","%OPENCV_INCLUDE%",^
..\bloodVesselSegmentation\include^
 --library-path ..\bloodVesselSegmentation\bin\,"%CUDA_PATH%\lib\x64","%OPENCV_LIB%"^
 --library cublas,bvsegmentation,cudadevrt,cudart,opencv_core300,opencv_highgui300^
 .\src\computeInterface.cpp^
 --shared^
 --machine 64^
 --output-file ..\..\apps\Vessel-OBJ-Creator\bin\lv\edi\EDI_3DAtA\vessel2objapp\computeCudaInterface.dll^
 