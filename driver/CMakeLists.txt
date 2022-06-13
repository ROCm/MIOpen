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
cmake_minimum_required( VERSION 3.5)

find_package(Threads REQUIRED)

add_executable(MIOpenDriver main.cpp InputFlags.cpp)
target_link_libraries(MIOpenDriver MIOpen)
target_link_libraries(MIOpenDriver ${CMAKE_THREAD_LIBS_INIT})
if(NOT MIOPEN_EMBED_DB STREQUAL "")
target_link_libraries(MIOpenDriver $<BUILD_INTERFACE:miopen_data> )
endif()
# Cmake does not add flags correctly for gcc
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
    set_target_properties(MIOpenDriver PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
endif()
install(TARGETS MIOpenDriver 
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})
