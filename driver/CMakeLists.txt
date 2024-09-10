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

find_package(Threads REQUIRED)

add_executable(MIOpenDriver 
    InputFlags.cpp
    conv_common.cpp
    dm_activ.cpp
    dm_adam.cpp
    dm_addlayernorm.cpp
    dm_bnorm.cpp
    dm_cat.cpp
    dm_conv.cpp
    dm_convbfp16.cpp
    dm_convbfp8.cpp
    dm_convfp16.cpp
    dm_convfp8.cpp
    dm_convint8.cpp
    dm_dropout.cpp
    dm_fusion.cpp
    dm_gemm.cpp
    dm_getitem.cpp
    dm_glu.cpp
    dm_groupnorm.cpp
    dm_layernorm.cpp
    dm_lrn.cpp
    dm_pool.cpp
    dm_prelu.cpp
    dm_reduce.cpp
    dm_reduceextreme.cpp
    dm_reducecalculation.cpp
    dm_rnn.cpp
    dm_rope.cpp
    dm_softmax.cpp
    dm_t5layernorm.cpp
    dm_tensorop.cpp
    dm_transformers_adam_w.cpp
    main.cpp
    registry_driver_maker.cpp
    rocrand_wrapper.cpp)
if(WIN32)
    # Refer to https://en.cppreference.com/w/cpp/language/types for details.
    target_compile_options(MIOpenDriver PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
endif()
add_dependencies(MIOpenDriver generate_kernels)
target_include_directories(MIOpenDriver PRIVATE ../src/kernels)
target_link_libraries(MIOpenDriver MIOpen Threads::Threads roc::rocrand)
if(NOT MIOPEN_EMBED_DB STREQUAL "")
target_link_libraries(MIOpenDriver $<BUILD_INTERFACE:miopen_data> )
endif()
# Cmake does not add flags correctly for gcc
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
    set_target_properties(MIOpenDriver PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
endif()

if( NOT ENABLE_ASAN_PACKAGING )
  install(TARGETS MIOpenDriver 
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
      DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
