THIS FOLDER CONTAINS CUDA SOURCE FILES.
 structure:
 \bloodVesselSegmentation (contains code and inaries for blood vessel segmentation code)
	\src (contains source files)
	\include (contains include files)
	\bin (contains compiled code bvsegmentation.lib is the main library)
	build.bat (batch file for windows, that build program)
 \windowsTest
	\src(source files)
	\bin contains test executable
\linuxTest	(requires $OPENCV_INCLUDE, $OPENCV_LIB environmen vars)
	\src
	\bin
	
==== Building bloodVesselSegmentation library =======
 folder \bloodVesselSegmentation contains build script for windows (build.bat) and linux (buildLinux.sh). 

 Dependencies:
  - CUDA TOOLKIT (Tested on 6.5 and 7.5), and CUDA_PATH environment should be set
  
  
  

