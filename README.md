# MLOpen

AMD's library for machine learning primitives

To build, create a build directory and run cmake.
```
mkdir build; cd build;
cmake ..
make 
```

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
