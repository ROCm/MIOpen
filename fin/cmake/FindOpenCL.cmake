################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 Advanced Micro Devices, Inc.
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
find_path(OPENCL_INCLUDE_DIRS
    NAMES OpenCL/cl.h CL/cl.h
    HINTS
    ${OPENCL_ROOT}/include
    $ENV{AMDAPPSDKROOT}/include
    $ENV{CUDA_PATH}/include
    PATHS
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
    /opt/cuda/include
    /opt/rocm/opencl/include
    ${CMAKE_INSTALL_PREFIX}/opencl/include
    DOC "OpenCL header file path"
    )
mark_as_advanced( OPENCL_INCLUDE_DIRS )

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    find_library( OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
        ${OPENCL_ROOT}/lib
        $ENV{AMDAPPSDKROOT}/lib
        $ENV{CUDA_PATH}/lib
        DOC "OpenCL dynamic library path"
        PATH_SUFFIXES x86_64 x64 x86_64/sdk
        PATHS
        /usr/lib
        /usr/local/cuda/lib
        /opt/cuda/lib
        /opt/rocm/opencl/lib
        ${CMAKE_INSTALL_PREFIX}/opencl/lib
        )
else( )
    find_library( OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
        ${OPENCL_ROOT}/lib
        $ENV{AMDAPPSDKROOT}/lib
        $ENV{CUDA_PATH}/lib
        DOC "OpenCL dynamic library path"
        PATH_SUFFIXES x86 Win32

        PATHS
        /usr/lib
        /usr/local/cuda/lib
        /opt/cuda/lib
        )
endif( )
mark_as_advanced( OPENCL_LIBRARIES )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( OPENCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS )

set(OpenCL_FOUND ${OPENCL_FOUND} CACHE INTERNAL "")
set(OpenCL_LIBRARIES ${OPENCL_LIBRARIES} CACHE INTERNAL "")
set(OpenCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS} CACHE INTERNAL "")

if( NOT OPENCL_FOUND )
    message( STATUS "FindOpenCL looked for libraries named: OpenCL" )
endif()
