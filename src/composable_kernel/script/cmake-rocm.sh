#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=../../../
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                                                              \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                                                                      \
-D CMAKE_BUILD_TYPE=Release                                                                                                        \
-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx908 -mllvm --amdgpu-spill-vgpr-to-agpr=0 -gline-tables-only -save-temps=$PWD"           \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                                                          \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                                                     \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                                                  \
${MY_PROJECT_SOURCE}

#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-spill-vgpr-to-agpr=0"                                               \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-spill-vgpr-to-agpr=0 -save-temps=$CWD"                              \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0"       \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0 -save-temps=$CWD"       \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0 -v -gline-tables-only -save-temps=$CWD"       \

#CXX_FLAG_TMP=-Weverything
#            -Wno-c++98-compat \
#            -Wno-c++98-compat-pedantic \
#            -Wno-conversion \
#            -Wno-double-promotion \
#            -Wno-exit-time-destructors \
#            -Wno-extra-semi \
#            -Wno-float-conversion \
#            -Wno-gnu-anonymous-struct \
#            -Wno-gnu-zero-variadic-macro-arguments \
#            -Wno-missing-noreturn \
#            -Wno-missing-prototypes \
#            -Wno-nested-anon-types \
#            -Wno-padded \
#            -Wno-return-std-move-in-c++11 \
#            -Wno-shorten-64-to-32 \
#            -Wno-sign-conversion \
#            -Wno-unknown-warning-option \
#            -Wno-unused-command-line-argument \
#            -Wno-weak-vtables \
#            -Wno-covered-switch-default \
#            -Wno-disabled-macro-expansion \
#            -Wno-undefined-reinterpret-cast

