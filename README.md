# MIOpen

AMD's library for high peformance machine learning primitives. MIOpen supports two programming models - 
1. OpenCL 
2. [HIP](https://github.com/RadeonOpenCompute/HIP).


## Prerequisites
* A ROCm enabled platform, more info [here](https://rocm.github.io/install.html)
* Base software stack, which includes
  * OpenCL - OpenCL libraries and header files
  * HIP - 
    * HIP and HCC libraries and header files
    * [clang-ocl](https://github.com/RadeonOpenCompute/clang-ocl) -- **required**
* MIOpen relies on the [tinygemm](https://github.com/RadeonOpenCompute/tinygemm) library to enable several functionalities that require GEMM. tinygemm is recommended but *not* required.

Please find the install instructions on the above dependencies on their respective repositories.

## Configure with cmake

First create a build directory:

```
mkdir build; cd build;
```

Next configure cmake. The preferred backend for MIOpen can be set using the `-DMLOPEN_BACKEND` cmake variable. 

#### For OpenCL, run:

```
cmake -DMLOPEN_BACKEND=OpenCL ..
```

The above assumes that OpenCL is installed in one of the standard locations. If not, then manually set these two cmake variables: 

```
cmake -DMLOPEN_BACKEND=OpenCL -DOPENCL_LIBRARIES=<opencl-library-path> -DOPENCL_INCLUDE_DIRS<opencl-headers-path> ..
```

#### For HIP, run:

Set the C++ compiler to `hcc`.
```
cmake -DMLOPEN_BACKEND=HIPOC -DCMAKE_PREFIX_PATH="<hip-installed-path>;<hcc-installed-path>;<clang-ocl-installed-path>" ..
```
An example cmake step can be:
```
CXX=/opt/rocm/hcc/bin/hcc cmake -DMLOPEN_BACKEND_HIPOC -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hcc;/opt/rocm/hip" ..
```

By default the install location is set to '/opt/rocm', this can be set by using `CMAKE_INSTALL_PREFIX`:

```
cmake -DMLOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=<mlopen-installed-path> ..
```

Also, the path to database for network configs can be set using the `MLOPEN_DB_PATH` variable. By default it is set to where the database files would be installed. For development purposes, setting `BUILD_DEV` will set the path to the database files stored in the source directory:

```
cmake -DMLOPEN_BACKEND=OpenCL -DBUILD_DEV=On ..
```

The configuration can be changed after running cmake by using `ccmake`:

```
ccmake ..
```

or `cmake-gui`:

```
cmake-gui ..
```

The `ccmake` program is not available on windows.

## Building the library

The library can be built, from the `build` directory using the 'Release' configuration:

` cmake --build . --config Release ` **OR** ` make `

And can be installed by using the 'install' target:

` cmake --build . --config Release --target install ` **OR** ` make install `

This will install the library to the `CMAKE_INSTALL_PREFIX` path that was set. 

## Building the driver

MIOpen provides an [application-driver](https://github.com/AMDComputeLibraries/MLOpen/tree/develop/driver) which can be used to execute any one particular layer in isolation and measure performance and verification of the library. 

Documentation on the driver is [here](https://github.com/AMDComputeLibraries/MLOpen/blob/develop/driver/README.md) 

The driver can be built using the `MLOpenDriver` target:

` cmake --build . --config Release --target MLOpenDriver ` **OR** ` make MLOpenDriver `

Then it can be ran using, like so:

```
./driver/MLOpenDriver --help
```

## Running the tests

The tests can be run by using the 'check' target:

` cmake --build . --config Release --target check ` **OR** ` make tests `

A single test can be built and ran, by doing:

```
cmake --build . --config Release --target test_tensor
./test/test_tensor
```

## Windows (Not supported)

Only OpenCL backend is functional.

### General prerequisites

* GCN-based GPU architecture or later (in particular, check your inegrated graphics HW).
* Latest AMD display driver.
* AMD APP SDK
* CMake for WINDOWS
* MS VS15 full installation

### Build:

For windows you may need to specify the MSVC generator, like so:

```
cd .../MLOpen
mkdir build
cd ./build
cmake .. -G "Visual Studio 14 2015 Win64 -DMLOPEN_BACKEND=OpenCL"
```
* Open VS15
* Open SuperBuild.MLOpen.sln
* Right click on MLOpenDriver
* Click "Set up as Startup Project"
* Build solution

### Run 
#### (From inside VS15)
* Right click on MLOpenDriver
* Click on Properties
* Click on Debugging
* Working directory: $(ProjectDir)../
* Environment: PATH=./src\Debug;%PATH%
* Command arguments (example):conv -n 10 -c 13 -k 13 -x 3 -y 3 -H 32 -W 64 -p 3 -q 3 -u 1 -v 1 -V 1 -F 1

#### From command line
```
cd .../MLOpen/build
PATH=.\src\Debug;%PATH%
(example)
.\driver\Debug\MLOpenDriver.exe conv -n 100 -c 3 -k 32 -x 5 -y 5 -H 32 -W 32 -F 1 -p 2 -q 2
```
