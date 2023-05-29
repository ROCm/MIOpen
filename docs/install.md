## Prerequisites

* More information about ROCm stack via [ROCm Information Portal](https://docs.amd.com/).
* A ROCm enabled platform, more info [here](https://rocm.github.io/install.html).
* Base software stack, which includes:
  * HIP - 
    * HIP and HCC libraries and header files.
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
* [Composable Kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) - C++ templated device library for GEMM-like and reduction-like operators.

## Installing MIOpen with pre-built packages

MIOpen can be installed on Ubuntu using `apt-get`.

For OpenCL backend: `apt-get install miopen-opencl`

For HIP backend: `apt-get install miopen-hip`

Currently both the backends cannot be installed on the same system simultaneously. If a different backend other than what currently exists on the system is desired, please uninstall the existing backend completely and then install the new backend.

## Installing MIOpen kernels package

MIOpen provides an optional pre-compiled kernels package to reduce the startup latency. These precompiled kernels comprise a select set of popular input configurations and will expand in future release to contain additional coverage.

Note that all compiled kernels are locally cached in the folder `$HOME/.cache/miopen/`, so precompiled kernels reduce the startup latency only for the first execution of a neural network. Precompiled kernels do not reduce startup time on subsequent runs.

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

More info can be found [here](https://github.com/ROCmSoftwarePlatform/MIOpen/blob/develop/doc/src/cache.md#installing-pre-compiled-kernels).

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
