# MIOpen

AMD's library for high performance machine learning primitives. 
Sources and binaries can be found at [MIOpen's GitHub site](https://github.com/ROCmSoftwarePlatform/MIOpen).
The latest released documentation can be read online [here](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/index.html).

MIOpen supports two programming models - 
1. [HIP](https://github.com/ROCm-Developer-Tools/HIP) (Primary Support).
2. OpenCL.

## Prerequisites
* More information about ROCm stack via [ROCm Information Portal](https://docs.amd.com/).
* A ROCm enabled platform, more info [here](https://rocm.github.io/install.html).
* Base software stack, which includes:
  * HIP - 
    * HIP and HCC libraries and header files.
    * [clang-ocl](https://github.com/RadeonOpenCompute/clang-ocl) -- **required**
  * OpenCL - OpenCL libraries and header files.
* [MIOpenGEMM](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM) - enable various functionalities including transposed and dilated convolutions. 
  * This is optional on the HIP backend, and required on the OpenCL backend.
  * Users can enable this library using the cmake configuration flag `-DMIOPEN_USE_MIOPENGEMM=On`, which is enabled by default when OpenCL backend is chosen.
* [ROCm cmake](https://github.com/RadeonOpenCompute/rocm-cmake) - provide cmake modules for common build tasks needed for the ROCM software stack.
* [Half](http://half.sourceforge.net/) - IEEE 754-based half-precision floating point library
* [Boost](http://www.boost.org/) 
  * MIOpen uses `boost-system` and `boost-filesystem` packages to enable persistent [kernel cache](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html)
  * Version 1.79 is recommended, older version may need patches to work on newer systems, e.g. boost1{69,70,72} w/glibc-2.34
* [SQLite3](https://sqlite.org/index.html) - reading and writing performance database
* [MIOpenTENSILE](https://github.com/ROCmSoftwarePlatform/MIOpenTensile) - users can enable this library using the cmake configuration flag`-DMIOPEN_USE_MIOPENTENSILE=On`. (deprecated after ROCm 5.1.1)
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) - AMD library for Basic Linear Algebra Subprograms (BLAS) on the ROCm platform.
  * Minimum version branch for pre-ROCm 3.5 [master-rocm-2.10](https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/master-rocm-2.10)
  * Minimum version branch for post-ROCm 3.5 [master-rocm-3.5](https://github.com/ROCmSoftwarePlatform/rocBLAS/releases/tag/rocm-3.5.0)
* [MLIR](https://github.com/ROCmSoftwarePlatform/llvm-project-mlir) - (Multi-Level Intermediate Representation) with its MIOpen dialect to support and complement kernel development.
* [Comopsable Kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) - C++ templated device library for GEMM-like and reduction-like operators.

## Installing MIOpen with pre-built packages

MIOpen can be installed on Ubuntu using `apt-get`.

For OpenCL backend: `apt-get install miopen-opencl`

For HIP backend: `apt-get install miopen-hip`

Currently both the backends cannot be installed on the same system simultaneously. If a different backend other than what currently exists on the system is desired, please uninstall the existing backend completely and then install the new backend.

## Installing MIOpen kernels package

MIOpen provides an optional pre-compiled kernels package to reduce the startup latency. These precompiled kernels comprise a select set of popular input configurations and will expand in future release to contain additional coverage.

To install the kernels package for your GPU architecture, use the following command:

```
apt-get install miopenkernels-<arch>-<num cu>
```

Where `<arch>` is the GPU architecture ( for example, `gfx900`, `gfx906`, `gfx1030` ) and `<num cu>` is the number of CUs available in the GPU (for example 56 or 64 etc). 

Not installing these packages would not impact the functioning of MIOpen, since MIOpen will compile these kernels on the target machine once the kernel is run. However, the compilation step may significantly increase the startup time for different operations.

The script `utils/install_precompiled_kernels.sh` provided as part of MIOpen automates the above process, it queries the user machine for the GPU architecture and then installs the appropriate package. It may be invoked as: 

```
./utils/install_precompiled_kernels.sh
```

The above script depends on the __rocminfo__ package to query the GPU architecture.

## Installing the dependencies

The dependencies can be installed with the `install_deps.cmake`, script: `cmake -P install_deps.cmake`

This will install by default to `/usr/local` but it can be installed in another location with `--prefix` argument:
```
cmake -P install_deps.cmake --prefix <miopen-dependency-path>
```
An example cmake step can be:
```
cmake -P install_deps.cmake --minimum --prefix /root/MIOpen/install_dir
```
This prefix can used to specify the dependency path during the configuration phase using the `CMAKE_PREFIX_PATH`.

* MIOpen's HIP backend uses [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) by default. Users can install rocBLAS minimum release by using `apt-get install rocblas`. To disable using rocBLAS set the configuration flag `-DMIOPEN_USE_ROCBLAS=Off`. rocBLAS is *not* available for the OpenCL backend.

* MIOpen's OpenCL backend uses [MIOpenGEMM](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM) by default. Users can install MIOpenGEMM minimum release by using `apt-get install miopengemm`.

## Building MIOpen from source

### Configuring with cmake

First create a build directory:

```
mkdir build; cd build;
```

Next configure cmake. The preferred backend for MIOpen can be set using the `-DMIOPEN_BACKEND` cmake variable. 

### For the HIP backend (ROCm 3.5 and later), run:
Set the C++ compiler to `clang++`.
```
export CXX=<location-of-clang++-compiler>
cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="<hip-installed-path>;<rocm-installed-path>;<miopen-dependency-path>" ..
```

An example cmake step can be:
```
export CXX=/opt/rocm/llvm/bin/clang++ && \
cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..
```

Note: When specifying the path for the `CMAKE_PREFIX_PATH` variable, **do not** use the `~` shorthand for the user home directory.

### For OpenCL, run:

```
cmake -DMIOPEN_BACKEND=OpenCL ..
```

The above assumes that OpenCL is installed in one of the standard locations. If not, then manually set these cmake variables: 

```
cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=<hip-compiler-path> -DOPENCL_LIBRARIES=<opencl-library-path> -DOPENCL_INCLUDE_DIRS=<opencl-headers-path> ..
```

And an example setting the dependency path for an envirnment in ROCm 3.5 and later:
```
cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..
```
### Setting Up Locations

By default the install location is set to '/opt/rocm', this can be set by using `CMAKE_INSTALL_PREFIX`:

```
cmake -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..
```

### System Performance Database and User Database

The default path to the System PerfDb is `miopen/share/miopen/db/` within install location. The default path to the User PerfDb is `~/.config/miopen/`. For development purposes, setting `BUILD_DEV` will change default path to both database files to the source directory:

```
cmake -DMIOPEN_BACKEND=OpenCL -DBUILD_DEV=On ..
```

Database paths can be explicitly customized by means of `MIOPEN_SYSTEM_DB_PATH` (System PerfDb) and `MIOPEN_USER_DB_PATH` (User PerfDb) cmake variables.

More information about the performance database can be found [here](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html).


### Persistent Program Cache

MIOpen by default caches the device programs in the location `~/.cache/miopen/`. In the cache directory there exists a directory for each version of MIOpen. Users can change the location of the cache directory during configuration using the flag `-DMIOPEN_CACHE_DIR=<cache-directory-path>`. 

Users can also disable the cache during runtime using the environmental variable set as `MIOPEN_DISABLE_CACHE=1`. 

#### For MIOpen version 2.3 and earlier
If the compiler changes, or the user modifies the kernels then the cache must be deleted for the MIOpen version in use; e.g., `rm -rf ~/.cache/miopen/<miopen-version-number>`. More information about the cache can be found [here](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html).

#### For MIOpen version 2.4 and later
MIOpen's kernel cache directory is versioned so that users' cached kernels will not collide when upgrading from earlier version.

### Changing the cmake configuration

The configuration can be changed after running cmake by using `ccmake`:

` ccmake .. ` **OR** `cmake-gui`: ` cmake-gui ..`

The `ccmake` program can be downloaded as the Linux package `cmake-curses-gui`, but is not available on windows.

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

Documentation on how to run the driver is [here](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/driver.html). 

## Running the tests

The tests can be run by using the 'check' target:

` cmake --build . --config Release --target check ` **OR** ` make check `

A single test can be built and ran, by doing:

```
cmake --build . --config Release --target test_tensor
./bin/test_tensor
```

## Building the documentation

HTML and PDF documentation can be built using:

`cmake --build . --config Release --target doc` **OR** `make doc`

This will build a local searchable web site inside the ./MIOpen/doc/html folder and a PDF document inside the ./MIOpen/doc/pdf folder.

Documentation is built using generated using [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html) and should be installed separately.

HTML and PDFs are generated using [Sphinx](http://www.sphinx-doc.org/en/stable/index.html) and [Breathe](https://breathe.readthedocs.io/en/latest/), with the [ReadTheDocs theme](https://github.com/rtfd/sphinx_rtd_theme).

Requirements for both Sphinx, Breathe, and the ReadTheDocs theme can be filled for these in the MIOpen/doc folder:

```
pip install -r ./requirements.txt
```


Depending on your setup `sudo` may be required for the pip install.

## Formatting the code

All the code is formatted using clang-format. To format a file, use:

```
clang-format-10 -style=file -i <path-to-source-file>
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```

## Installing the dependencies manually

If Ubuntu v16 is used then the `Boost` packages can also be installed by:
```
sudo apt-get install libboost-dev
sudo apt-get install libboost-system-dev
sudo apt-get install libboost-filesystem-dev
```

*Note:* MIOpen by default will attempt to build with Boost statically linked libraries. If it is needed, the user can build with dynamically linked Boost libraries by using this flag during the configruation stage:
```
-DBoost_USE_STATIC_LIBS=Off
```
however, this is not recommended.

The `half` header needs to be installed from [here](http://half.sourceforge.net/). 


## Using docker

The easiest way is to use docker. You can build the top-level docker file:
```
docker build -t miopen-image .
```

Then to enter the development environment use `docker run`, for example:
```
docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device /dev/dri:/dev/dri:rw  --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined miopen-image
```

Prebuilt docker images can be found on [ROCm's public docker hub here](https://hub.docker.com/r/rocm/miopen/tags).

## Citing MIOpen


MIOpen's paper is freely available and can be accessed on arXiv:  
[MIOpen: An Open Source Library For Deep Learning Primitives](https://arxiv.org/abs/1910.00078)


### Citation BibTeX
```
@misc{jeh2019miopen,
    title={MIOpen: An Open Source Library For Deep Learning Primitives},
    author={Jehandad Khan and Paul Fultz and Artem Tamazov and Daniel Lowell and Chao Liu and Michael Melesse and Murali Nandhimandalam and Kamil Nasyrov and Ilya Perminov and Tejash Shah and Vasilii Filippov and Jing Zhang and Jing Zhou and Bragadeesh Natarajan and Mayank Daga},
    year={2019},
    eprint={1910.00078},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

