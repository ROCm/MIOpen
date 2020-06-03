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

set(ADD_KERNELS_SOURCE addDbs.cpp)
add_executable(addDbs EXCLUDE_FROM_ALL ${ADD_KERNELS_SOURCE})
target_link_libraries(addDbs Boost::filesystem ${SQLITE3_STATIC_LDFLAGS})
# target_include_directories(addDbs PUBLIC ../src/include ../include ${CMAKE_BINARY_DIR}/include )
target_include_directories(addDbs PUBLIC ${SQLITE3_STATIC_INCLUDE_DIRS})
#     target_include_directories(MIOpen SYSTEM INTERFACE $<BUILD_INTERFACE:${SQLITE3_STATIC_INCLUDE_DIRS}>)
target_compile_options(addDbs PUBLIC ${SQLITE3_STATIC_CFLAGS})
#     target_compile_options(MIOpen INTERFACE $<BUILD_INTERFACE:${SQLITE3_STATIC_CFLAGS}>)
#     target_link_libraries(MIOpen INTERFACE $<BUILD_INTERFACE:${SQLITE3_STATIC_LDFLAGS}>)
clang_tidy_check(addDbs)
