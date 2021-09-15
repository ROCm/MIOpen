#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=../../..
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                                                                          \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                                                                                  \
-D HALF_INCLUDE_DIR="/root/workspace/external/half/include"                                                                                    \
-D BUILD_DEV=ON                                                                                                                                \
-D CMAKE_BUILD_TYPE=Release                                                                                                                    \
-D CMAKE_CXX_FLAGS="-DCK_AMD_GPU_GFX908 -O3 --amdgpu-target=gfx908 -mllvm --amdgpu-spill-vgpr-to-agpr=0 -gline-tables-only -save-temps=$PWD"   \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                                                                      \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                                                                 \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                                                              \
${MY_PROJECT_SOURCE}
