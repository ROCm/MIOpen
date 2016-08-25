# MLOpen

AMD's library for machine learning primitives

##Linux

To build, create a build directory and run cmake.
```
mkdir build; cd build;
cmake ..
make 
```

To execute/use MLOpen, use the MLOpenDriver app: https://github.com/AMDComputeLibraries/MLOpen/tree/develop/driver

##WINDOWS

## General prerequisites

* GCN-based GPU architecture or later (in particular, check your inegrated graphics HW).
* Latest AMD display driver.
* AMD APP SDK
* CMake for WINDOWS
* MS VS15 full installation

##Build:
```
cd .../MLOpen
mkdir build
cd ./build
cmake .. -G "Visual Studio 14 2015 Win64"
```
* Fire VS15
* Open SuperBuild.MLOpen.sln
* Right click on MLOpenDriver
* Click "Set up as Startup Project"
* Build solution

##Run:
##From inside VS15
* Right click on MLOpenDriver
* Click on Properties
* Click on Debugging
* Working directory: $(ProjectDir)../
* Environment: PATH=./src\Debug;%PATH%
* Command arguments (example):conv -n 10 -c 13 -k 13 -x 3 -y 3 -H 32 -W 64 -p 3 -q 3 -u 1 -v 1 -V 1 -F 1

##From command line
```
cd .../MLOpen/build
PATH=.\src\Debug;%PATH%
(example)
.\driver\Debug\MLOpenDriver.exe -n 100 -c 3 -k 32 -x 5 -y 5 -H 32 -W 32 -F 1 -p 2 -q 2
```

##Install Library for use in external projects

To install the library, type `sudo make install`. This installs the library in `/usr/local/`. 
If `/usr/local/` is not the desired installation location, use `CMAKE_INSTALL_PREFIX="path"` for the desired location. 

Currently, MLOpen requires a macro to be set at the time of compiling the application which uses it. The macros chooses the appropriate backend, e.g., `-DMLOPEN_BACKEND_OPENCL`. This will requirement will be removed in the future.

To execute/use MLOpen, use the MLOpenDriver app: https://github.com/AMDComputeLibraries/MLOpen/tree/develop/driver

MLOpen also comes with a unit-test framework. To run the unit-tests, type `make check`.
For running specific tests, first the compile and the tests and then execute the specific test:
```
make tests
./test/test_tensor
```
