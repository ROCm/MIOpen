# MLOpen

AMD's library for machine learning primitives

##Linux

To build, create a build director and run cmake.
```
mkdir build; cd build;
cmake ..
make 
```

To execute/use MLOpen, use the MLOpenDriver app: https://github.com/AMDComputeLibraries/MLOpen/tree/develop/driver

##WINDOWS

## General prerequsites

GCN-based GPU architecture or later (in particular, check your inegrated graphics HW).

Latest AMD display driver.

AMD APP SDK

CMake for WINDOWS

MS VS15 full installation

##Build:
cd .../MLOpen

mkdir build

cd ./build

cmake .. -G "Visual Studio 14 2015 Win64"

fire VS15

open SuperBuild.MLOpen.sln

right click on MLOpenDriver

click "Set up as Startup Project"

build solution

##Run:
##From inside VS15
right click on MLOpenDriver

click on Properties

click on Debugging

Working directory: $(ProjectDir)../

Environment: PATH=./src\Debug;%PATH%

Command arguments (example):conv -n 10 -c 13 -k 13 -x 3 -y 3 -H 32 -W 64 -p 3 -q 3 -u 1 -v 1 -V 1 -F 1

##From command line
cd .../MLOpen/build

PATH=.\src\Debug;%PATH%

(eaxmple)

.\driver\Debug\MLOpenDriver.exe -n 100 -c 3 -k 32 -x 5 -y 5 -H 32 -W 32 -F 1 -p 2 -q 2