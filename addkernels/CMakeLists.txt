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

set(ADD_KERNELS_SOURCE include_inliner.cpp addkernels.cpp)

add_executable(addkernels EXCLUDE_FROM_ALL ${ADD_KERNELS_SOURCE})
target_include_directories(addkernels PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
if(HAS_LIB_STD_FILESYSTEM)
    target_link_libraries(addkernels PRIVATE stdc++fs)
endif()

# This hack is for MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM.
# We can't use config.h because it is generated after addkernels.
target_compile_definitions(addkernels PRIVATE -DMIOPEN_HACK_DO_NOT_INCLUDE_CONFIG_H=1)
if(MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM)
    target_compile_definitions(addkernels PRIVATE -DMIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM=1)
    target_link_libraries(addkernels PRIVATE Boost::filesystem)
else()
    target_compile_definitions(addkernels PRIVATE -DMIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM=0)
endif()

clang_tidy_check(addkernels)
