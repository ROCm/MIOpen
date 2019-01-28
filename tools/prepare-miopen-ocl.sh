#!/bin/bash

################################################################################
#
# MIT License
#
# Copyright (c) 2019 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################


if [ "${1}" = "" ]; then 
    echo "Missing dependencies directory as first argument."
    exit 1
fi

if [ "${1}" = "-h" ]; then 
    echo "Usage: ./prepare-miopen-hip.sh <dependencies directory>"
    exit 0
fi

#Install cget
echo "Installing cget"
pip install cget

mkdir -p ${1}
cg="cget -p ${1}"

cd ../
# For the OpenCL backend
echo "Init OpenCL cget"
$cg init

echo "Setting up dependencies"
$cg ignore ROCm-Developer-Tools/HIP
$cg ignore RadeonOpenCompute/rocm-cmake
$cg ignore ROCmSoftwarePlatform/rocBLAS
$cg ignore ROCmSoftwarePlatform/MIOpen
$cg ignore RadeonOpenCompute/clang-ocl
$cg ignore ROCmSoftwarePlatform/MIOpenGEMM


 
# Skip cppcheck since we wont run analysis
$cg ignore danmar/cppcheck

echo "Installing toolchain"
$cg install 



