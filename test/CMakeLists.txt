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

include(CTest)

option( MIOPEN_TEST_ALL "Run the full test suite" OFF )
option(MIOPEN_TEST_CONV Off)
option(MIOPEN_TEST_DEEPBENCH Off)

set(MIOPEN_TEST_FLAGS "" CACHE STRING "")

find_package(Threads REQUIRED)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

function(add_test_command NAME EXE)
    if(WIN32)
        set(WINPATH)
        foreach(PATH ${CMAKE_FIND_ROOT_PATH})
            list(APPEND WINPATH ${PATH}/bin)
        endforeach()
        file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmd"
            CONTENT "set PATH=${WINPATH};%PATH%
                    %1 ${ARGN}")
        add_test(NAME ${NAME} COMMAND ${WINE_CMD} cmd /c "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmd" $<TARGET_FILE:${EXE}>)
    else()
        add_test(NAME ${NAME} COMMAND ${EXE} ${ARGN})
    endif()
endfunction()

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
    # Cmake does not add flags correctly for gcc
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
    endif()
    separate_arguments(MIOPEN_TEST_FLAGS_ARGS UNIX_COMMAND ${MIOPEN_TEST_FLAGS})
    if(MIOPEN_TEST_ALL)
        set(TEST_COMMAND ${TEST_NAME} --all ${MIOPEN_TEST_FLAGS_ARGS})
    else()
        set(TEST_COMMAND ${TEST_NAME} ${MIOPEN_TEST_FLAGS_ARGS})
    endif()
    add_test_command(${TEST_NAME} ${TEST_COMMAND})
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
endfunction(add_test_executable)

file(GLOB TESTS *.cpp)

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST})
    target_link_libraries(test_${BASE_NAME} MIOpen)
endforeach()

function(add_custom_test NAME)
    add_custom_target(${NAME} ${ARGN})
    add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${NAME})
endfunction()

if(MIOPEN_TEST_ALL)
    add_custom_test(test_conv_extra
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 1 1 1 1 --weights 1 1 2 2 --filter 0 0 3 3
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 1 161 700  --weights 4 1 5 20 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 1 161 700  --weights 4 1 5 20 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 32 79 341  --weights 4 32 5 10 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 32 79 341  --weights 4 32 5 10 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 3 227 227  --weights 4 3 11 11 --filter 0 0 4 4 --disable-backward-data
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 3 224 224  --weights 4 3 11 11 --filter 2 2 4 4
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 1 48 480 --weights 16 1 3 3 --filter 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 32 64 27 27 --weights 192 64 5 5 --filter 2 2 1 1
    )
endif()

if(MIOPEN_TEST_DEEPBENCH)
    add_custom_test(test_deepbench
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 1 161 700  --weights 32 1 5 20 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 1 161 700  --weights 32 1 5 20 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 1 161 700  --weights 32 1 5 20 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 32 1 161 700  --weights 32 1 5 20 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 4 32 79 341  --weights 32 32 5 10 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 32 79 341  --weights 32 32 5 10 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 32 79 341  --weights 32 32 5 10 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 32 32 79 341  --weights 32 32 5 10 --filter 0 0 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 1 48 480 --weights 16 1 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 16 24 240 --weights 32 16 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 32 12 120 --weights 64 32 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 64 6 60 --weights 128 64 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 3 108 108 --weights 64 3 3 3 --filter 1 1 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 64 54 54 --weights 64 64 3 3 --filter 1 1 1 1 
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 128 27 27 --weights 128 128 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 128 14 14 --weights 256 128 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 256 7 7 --weights 512 256 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 3 224 224 --weights 64 3 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 64 112 112 128 --weights 64 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 128 56 56 --weights 256 128 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 256 28 28 --weights 512 256 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 512 14 14 --weights 512 512 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 8 512 7 7 --weights 512 512 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 3 224 224 --weights 64 3 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 64 112 112 128 --weights 64 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 128 56 56 --weights 256 128 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 256 28 28 --weights 512 256 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 512 14 14 --weights 512 512 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 512 7 7 --weights 512 512 3 3 --filter 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 3 224 224 --weights 64 3 7 7 --filter 3 3 2 2
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 192 28 28 --weights 32 192 5 5 --filter 2 2 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 512 14 14 --weights 48 512 5 5 --filter 2 2 1 1 
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 832 7 7 --weights 128 832 5 5 --filter 2 2 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 192 28 28 --weights 32 192 1 1 --filter 0 0 1 1 
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 512 14 14 --weights 48 512 1 1 --filter 0 0 1 1
    COMMAND $<TARGET_FILE:test_conv> --verbose --input 16 832 7 7 --weights 128 832 1 1 --filter 0 0 1 1
    )
endif()

if(MIOPEN_TEST_CONV)
    add_test(NAME test_conv_input_1x3x32x32_weights_1x3x7x7_filter_1_1_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 32, 32 --weights 1, 3, 7, 7 --filter 1, 1, 1, 1)
    add_test(NAME test_conv_input_1x3x227x227_weights_1x3x7x7_filter_1_1_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 227, 227 --weights 1, 3, 7, 7 --filter 1, 1, 1, 1)
    add_test(NAME test_conv_input_1x64x56x56_weights_1x64x1x1_filter_0_0_2_2 COMMAND $<TARGET_FILE:test_conv> --input 1, 64, 56, 56 --weights 1, 64, 1, 1 --filter 0, 0, 2, 2)
    add_test(NAME test_conv_input_1x3x32x32_weights_1x3x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 32, 32 --weights 1, 3, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x224x224_weights_1x3x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 224, 224 --weights 1, 3, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x227x227_weights_1x3x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 227, 227 --weights 1, 3, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x231x231_weights_1x3x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 231, 231 --weights 1, 3, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x224x224_weights_1x3x5x5_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 224, 224 --weights 1, 3, 5, 5 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x227x227_weights_1x3x5x5_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 227, 227 --weights 1, 3, 5, 5 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x231x231_weights_1x3x5x5_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 231, 231 --weights 1, 3, 5, 5 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x32x32_weights_1x3x7x7_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 32, 32 --weights 1, 3, 7, 7 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x224x224_weights_1x3x7x7_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 224, 224 --weights 1, 3, 7, 7 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x227x227_weights_1x3x7x7_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 227, 227 --weights 1, 3, 7, 7 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x231x231_weights_1x3x7x7_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 231, 231 --weights 1, 3, 7, 7 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x64x56x56_weights_1x64x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 64, 56, 56 --weights 1, 64, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x64x112x112_weights_1x64x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 64, 112, 112 --weights 1, 64, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x64x512x1024_weights_1x64x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 64, 512, 1024 --weights 1, 64, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x96x27x27_weights_1x96x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 96, 27, 27 --weights 1, 96, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x96x28x28_weights_1x96x3x3_filter_2_2_1_1 COMMAND $<TARGET_FILE:test_conv> --input 1, 96, 28, 28 --weights 1, 96, 3, 3 --filter 2, 2, 1, 1)
    add_test(NAME test_conv_input_1x3x32x32_weights_1x3x3x3_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 32, 32 --weights 1, 3, 3, 3 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x224x224_weights_1x3x3x3_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 224, 224 --weights 1, 3, 3, 3 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x227x227_weights_1x3x3x3_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 227, 227 --weights 1, 3, 3, 3 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x231x231_weights_1x3x3x3_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 231, 231 --weights 1, 3, 3, 3 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x32x32_weights_1x3x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 32, 32 --weights 1, 3, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x224x224_weights_1x3x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 224, 224 --weights 1, 3, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x227x227_weights_1x3x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 227, 227 --weights 1, 3, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x231x231_weights_1x3x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 231, 231 --weights 1, 3, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x32x32_weights_1x3x7x7_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 32, 32 --weights 1, 3, 7, 7 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x224x224_weights_1x3x7x7_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 224, 224 --weights 1, 3, 7, 7 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x227x227_weights_1x3x7x7_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 227, 227 --weights 1, 3, 7, 7 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x3x231x231_weights_1x3x7x7_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 3, 231, 231 --weights 1, 3, 7, 7 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x16x14x14_weights_1x16x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 16, 14, 14 --weights 1, 16, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x16x28x28_weights_1x16x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 16, 28, 28 --weights 1, 16, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x24x14x14_weights_1x24x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 24, 14, 14 --weights 1, 24, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x32x7x7_weights_1x32x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 32, 7, 7 --weights 1, 32, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x32x8x8_weights_1x32x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 32, 8, 8 --weights 1, 32, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x32x14x14_weights_1x32x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 32, 14, 14 --weights 1, 32, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x32x16x16_weights_1x32x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 32, 16, 16 --weights 1, 32, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x32x28x28_weights_1x32x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 32, 28, 28 --weights 1, 32, 5, 5 --filter 0 0 4 4)
    add_test(NAME test_conv_input_1x48x7x7_weights_1x48x5x5_filter_0_0_4_4 COMMAND $<TARGET_FILE:test_conv> --input 1, 48, 7, 7 --weights 1, 48, 5, 5 --filter 0 0 4 4)
endif()
