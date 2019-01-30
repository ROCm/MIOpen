

# Scripts for building MIOpen packages

Important: Users should run these scripts from the `tools` directory.

## Preparing the ToolChain
The first set creates "prepares" the cget environment. You give it what directory you want to install the dependencies. If that directory does not exist, then it will create it. The scripts also installs cget if you don't already have it. 

You should have installed on your system already:
* ROCm-Developer-Tools/HIP
* RadeonOpenCompute/rocm-cmake
* ROCmSoftwarePlatform/rocBLAS
* RadeonOpenCompute/clang-ocl
* ROCmSoftwarePlatform/MIOpenGEMM


Example usage of the prepare script for the OpenCL backend:
```
./prepare-miopen-ocl.sh <dependency_ocl_dir>
```

and for preparing the HIP backend:
```
./prepare-miopen-hip.sh <dependency_hip_dir>
```

## Building the Packages
The second script you use to build the packages. The first argument is the directory path you used for one of the above hip, or OpenCL dependencies. The second argument is the build directory. If the build directory does not exist the script will try to create it.

Example usage of the build script for the HIP backend:
```
./build.sh <dependency_hip_dir> <hip_build_dir>
```

Or this for the OpenCL backend:
```
./build.sh <dependency_ocl_dir> <ocl_build_dir>
```

Users should run the prepare script first then the build script.


