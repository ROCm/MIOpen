# MIOpen

AMD's library for high peformance machine learning primitives. MIOpen supports two programming models - 
1. OpenCL 
2. [HIP](https://github.com/ROCm-Developer-Tools/HIP)

## Prerequisites
* A ROCm enabled platform, more info [here](https://rocm.github.io/install.html)
* Base software stack, which includes
  * OpenCL - OpenCL libraries and header files
  * HIP - 
    * HIP and HCC libraries and header files
    * [clang-ocl](https://github.com/RadeonOpenCompute/clang-ocl) -- **required**
* [MIOpenGEMM](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM) to enable various functionalities including transposed and dilated convolutions
* ROCm cmake modules can be installed from [here](https://github.com/RadeonOpenCompute/rocm-cmake)
* [OpenSSL](https://www.openssl.org/) or [libressl](https://www.libressl.org/)
* [Boost](http://www.boost.org/) at least version 1.58
  * MIOpen uses `boost-system` and `boost-filesystem` packages to enable persistent [kernel cache](https://github.com/ROCmSoftwarePlatform/MIOpen/blob/master/doc/src/cache.md)

Instructions to install the above dependencies are present in this [section](#installing-the-dependencies).

## Installing MIOpen with pre-built packages

MIOpen can be installed on Ubuntu using `apt-get`.

For OpenCL backend: `apt-get install miopen-opencl`

For HIP backend: `apt-get install miopen-hip`

## Building MIOpen from source

## Configuring with cmake

First create a build directory:

```
mkdir build; cd build;
```

Next configure cmake. The preferred backend for MIOpen can be set using the `-DMIOPEN_BACKEND` cmake variable. 

#### For OpenCL, run:

```
cmake -DMIOPEN_BACKEND=OpenCL ..
```

The above assumes that OpenCL is installed in one of the standard locations. If not, then manually set these two cmake variables: 

```
cmake -DMIOPEN_BACKEND=OpenCL -DOPENCL_LIBRARIES=<opencl-library-path> -DOPENCL_INCLUDE_DIRS=<opencl-headers-path> ..
```

#### For HIP, run:

Set the C++ compiler to `hcc`.
```
cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="<hip-installed-path>;<hcc-installed-path>;<clang-ocl-installed-path>" ..
```
An example cmake step can be:
* `OpenSSL` installed using `apt-get` on Ubuntu v16? **Yes**.
```
CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip" -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" ..
```
* `OpenSSL` installed using `apt-get` on Ubuntu v16? **No**.
```
CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip" ..
```


By default the install location is set to '/opt/rocm', this can be set by using `CMAKE_INSTALL_PREFIX`:

```
cmake -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..
```

Also, the path to database for network configs can be set using the `MIOPEN_DB_PATH` variable. By default it is set to where the database files would be installed. For development purposes, setting `BUILD_DEV` will set the path to the database files stored in the source directory:

```
cmake -DMIOPEN_BACKEND=OpenCL -DBUILD_DEV=On ..
```

The configuration can be changed after running cmake by using `ccmake`:

` ccmake .. ` **OR** `cmake-gui`: ` cmake-gui ..`

The `ccmake` program is not available on windows.

## Building the library

The library can be built, from the `build` directory using the 'Release' configuration:

` cmake --build . --config Release ` **OR** ` make `

And can be installed by using the 'install' target:

` cmake --build . --config Release --target install ` **OR** ` make install `

This will install the library to the `CMAKE_INSTALL_PREFIX` path that was set. 

## Building the driver

MIOpen provides an [application-driver](https://github.com/ROCmSoftwarePlatform/MIOpen/tree/master/driver) which can be used to execute any one particular layer in isolation and measure performance and verification of the library. 

The driver can be built using the `MIOpenDriver` target:

` cmake --build . --config Release --target MIOpenDriver ` **OR** ` make MIOpenDriver `

Documentation on how to run the driver is [here](https://github.com/ROCmSoftwarePlatform/MIOpen/blob/master/driver/README.md). 


If building for HIP and `boost` was installed via `apt-get` in Ubuntu v16, add the following to the cmake line [above](#configure-with-cmake):
```
-DMIOPEN_MAKE_BOOST_PUBLIC=ON
```

## Running the tests

The tests can be run by using the 'check' target:

` cmake --build . --config Release --target check ` **OR** ` make check `

A single test can be built and ran, by doing:

```
cmake --build . --config Release --target test_tensor
./bin/test_tensor
```

If `boost` was installed via `apt-get` in Ubuntu v16, add the following to the cmake line [above](#configure-with-cmake):
```
-DMIOPEN_MAKE_BOOST_PUBLIC=ON
```

## Building the documentation

HTML and PDF documentation can be built using:

`cmake --build . --config Release --target doc` **OR** `make doc`

This will build a local searchable web site inside the ./MIOpen/doc/html folder and a PDF document inside the ./MIOpen/doc/pdf folder.

Documentation is built using generated using [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html) and should be installed separately.

HTML and PDFs are generated using [Sphinx](http://www.sphinx-doc.org/en/stable/index.html) and [Breathe](https://breathe.readthedocs.io/en/latest/), with the [ReadTheDocs theme](https://github.com/rtfd/sphinx_rtd_theme).

Requirements for both Sphinx, Breathe, and the ReadTheDocs theme can be filled for these in the MIOpen/doc folder:

`pip install -r ./requirements.txt`

Depending on your setup `sudo` may be required for the pip install.

## Formatting the code

All the code is formatted using clang-format. To format a file, use:

```
clang-format-3.8 -style=file -i <path-to-source-file>
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```

## Installing the dependencies

The dependencies can be installed with the `install_deps.cmake`, script:

```
cmake -P install_deps.cmake
```

This will install by default to `/usr/local` but it can be installed in another location with `--prefix` argument:

```
cmake -P install_deps.cmake --prefix /some/local/dir
```

If Ubuntu v16 is used then the `OpenSSL` and `Boost` packages can also be installed by:
```
sudo apt-get install libssl-dev
sudo apt-get install libboost-dev
sudo apt-get install libboost-system-dev
sudo apt-get install libboost-filesystem-dev
```
