################################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

# if the number of samples grows, its likely better to organize them in subdirectories
message(STATUS "Building MIOpen samples")
cmake_minimum_required(VERSION 3.15)
project(MIOpenSample LANGUAGES CXX HIP)
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to HIP")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${HIP_PATH}/cmake)

list(
        APPEND
        CMAKE_PREFIX_PATH
        ${ROCM_PATH}/llvm
        ${ROCM_PATH}
        ${ROCM_PATH}/hip
        /opt/rocm/llvm
        /opt/rocm
        /opt/rocm/hip
        /opt/rocm/lib
    )

find_package(MIOpen REQUIRED)
find_package(HIP REQUIRED)

add_executable(sample_cba_fused_fwd sample_cba_fused_fwd.cpp)

target_include_directories(sample_cba_fused_fwd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HIP_INCLUDE_DIRS}
    ${MIOPEN_INCLUDE_DIRS}
)

target_link_libraries(sample_cba_fused_fwd PRIVATE MIOpen hip::host)
