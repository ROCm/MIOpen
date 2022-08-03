################################################################################
# 
# MIT License
# 
# Copyright (c) 2020 Advanced Micro Devices, Inc.
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
cmake_minimum_required( VERSION 2.8.12 )
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc /opt/rocm/opencl)

configure_file("${PROJECT_SOURCE_DIR}/src/include/config.h.in" "${PROJECT_BINARY_DIR}/src/include/config.h")

include_directories(include "${PROJECT_BINARY_DIR}/src/include")
add_executable(fin main.cpp fin.cpp base64.cpp)
target_compile_definitions( fin PRIVATE -D__HIP_PLATFORM_HCC__=1 )
target_link_libraries(fin MIOpen ${Boost_LIBRARIES} hip::host)
target_link_libraries(fin ${CMAKE_THREAD_LIBS_INIT})
if(rocblas_FOUND)
    target_link_libraries( fin $<BUILD_INTERFACE:roc::rocblas> )
    # target_link_libraries( fin PRIVATE roc::rocblas )
endif()

# Cmake does not add flags correctly for gcc
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
    set_target_properties(fin PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
endif()
install(TARGETS fin
    OPTIONAL 
    RUNTIME DESTINATION bin)
