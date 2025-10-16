# MIOpen Samples

This directory contains sample code for using [MIOpen](https://github.com/ROCm/MIOpen).

## Available Samples

- **cba_fused_infer**: Demonstrates a forward pass of fused convolution + bias + activation.

More samples will be added in the future.

## Building and Running

**Prerequisites**: [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest/install/install.html) must be installed on your system.

To build and run a sample:

```bash
cd <sample_directory>
mkdir build
cd build
cmake ..
make
./<sample_executable>
```
