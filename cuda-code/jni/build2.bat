call vcvarsa32.bat
cl .\src\computeInterface.cpp /I "%JAVA_HOME%\include" /I "%JAVA_HOME%\include\win32" -FascomputeInteCudaInterface.dll -MD -LD