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

cmake_policy(SET CMP0057 NEW)

include(CTest)
include(CheckCXXCompilerFlag)

option( MIOPEN_TEST_ALL "Run the full test suite" OFF )
option( MIOPEN_TEST_HALF "Test in half mode" OFF )
option( MIOPEN_TEST_INT8 "Test in int8 mode" OFF )
option( MIOPEN_TEST_BFLOAT16 "Test in bfloat16 mode" OFF )
option(MIOPEN_TEST_CONV Off)
option(MIOPEN_TEST_DEEPBENCH Off)
option(MIOPEN_TEST_DRIVER_ITER_MODE Off)

set(MIOPEN_TEST_FLAGS "" CACHE STRING "")
set(MIOPEN_TEST_GDB On CACHE BOOL "")


if(MIOPEN_TEST_DRIVER_ITER_MODE)
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=2)
else()
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=1)   
endif()

find_package(Threads REQUIRED)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

set(SKIP_TESTS)
set(MIOPEN_TEST_FLOAT_ARG)

if(MIOPEN_TEST_HALF)   
    if(MIOPEN_BACKEND_OPENCL)
        set(SKIP_TESTS test_gru test_rnn_vanilla test_lstm)
    endif()
    set(MIOPEN_TEST_FLOAT_ARG --half)
endif()

if(MIOPEN_TEST_INT8)
    set(SKIP_ALL_EXCEPT_TESTS test_tensor_vec test_tensor_cast test_tensor_trans test_tensor_copy test_tensor_set test_tensor_transform test_conv test_immed_conv)
    set(MIOPEN_TEST_FLOAT_ARG --int8)
endif()

if(MIOPEN_TEST_BFLOAT16)
    set(SKIP_ALL_EXCEPT_TESTS test_conv test_tensor_copy test_tensor_set test_tensor_vec test_immed_conv)
    set(MIOPEN_TEST_FLOAT_ARG --bfloat16)
endif()

function(add_test_command NAME EXE)
    if((NOT (NAME IN_LIST SKIP_ALL_EXCEPT_TESTS)) AND (MIOPEN_TEST_INT8 OR MIOPEN_TEST_BFLOAT16))
        add_test(NAME ${NAME} COMMAND echo skipped)
        set_tests_properties(${NAME} PROPERTIES DISABLED On)
    elseif(NAME IN_LIST SKIP_TESTS)
        add_test(NAME ${NAME} COMMAND echo skipped)
        set_tests_properties(${NAME} PROPERTIES DISABLED On)
    elseif(WIN32)
        set(WINPATH)
        foreach(PATH ${CMAKE_FIND_ROOT_PATH})
            list(APPEND WINPATH ${PATH}/bin)
        endforeach()
        file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmd"
            CONTENT "set PATH=${WINPATH};%PATH%
                    %1 ${ARGN}")
        add_test(NAME ${NAME} COMMAND ${WINE_CMD} cmd /c "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmd" $<TARGET_FILE:${EXE}>)
    else()
        if(MIOPEN_TEST_GDB)
            file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmake"
                CONTENT "
                execute_process(COMMAND $<TARGET_FILE:${EXE}> ${ARGN} RESULT_VARIABLE RESULT)
                if(NOT RESULT EQUAL 0)
                    if(EXISTS core)
                        execute_process(COMMAND gdb $<TARGET_FILE:${EXE}> core -batch -ex bt)
                    endif()
                    message(FATAL_ERROR \"Test failed\")
                endif()
            ")
            add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmake")
        else()
            add_test(NAME ${NAME} COMMAND ${EXE} ${ARGN})
        endif()
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
        set(TEST_COMMAND ${TEST_NAME} ${MIOPEN_TEST_FLOAT_ARG} --all ${MIOPEN_TEST_FLAGS_ARGS})
    else()
        set(TEST_COMMAND ${TEST_NAME} ${MIOPEN_TEST_FLOAT_ARG} ${MIOPEN_TEST_FLAGS_ARGS})
    endif()
    add_test_command(${TEST_NAME} ${TEST_COMMAND})
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
    target_link_libraries(${TEST_NAME} MIOpen)
endfunction(add_test_executable)

set(MIOPEN_TEST_SANITIZERS)
foreach(SANTIZER address thread)
    check_cxx_compiler_flag("-fsanitize=${SANTIZER} -fno-sanitize-recover=${SANTIZER}" MIOPEN_HAS_${SANTIZER})
    if(MIOPEN_HAS_${SANTIZER})
        list(APPEND MIOPEN_TEST_SANITIZERS ${SANTIZER})
    endif()
endforeach()

function(add_sanitize_test TEST_SOURCE)
    get_filename_component(BASE_NAME ${TEST_SOURCE} NAME_WE)
    foreach(SANTIZER ${MIOPEN_TEST_SANITIZERS})
        add_test_executable(test_${BASE_NAME}_${SANTIZER} ${TEST_SOURCE})
        target_compile_options(test_${BASE_NAME}_${SANTIZER} PUBLIC -fsanitize=${SANTIZER} -fno-sanitize-recover=${SANTIZER})
        target_link_libraries(test_${BASE_NAME}_${SANTIZER} -fsanitize=${SANTIZER} -fno-sanitize-recover=${SANTIZER})
    endforeach()
endfunction()

file(GLOB TESTS *.cpp)

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST})
endforeach()

# add_sanitize_test(perfdb.cpp)
# add_sanitize_test(cache.cpp)
# add_sanitize_test(tensor_test.cpp)
# add_sanitize_test(type_name.cpp)

function(add_custom_test NAME)
    if(NOT (MIOPEN_TEST_INT8 OR MIOPEN_TEST_BFLOAT16))
        set(options ALL)
        set(oneValueArgs)
        set(multiValueArgs)

        cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    
        add_custom_target(${NAME} ${PARSE_UNPARSED_ARGUMENTS})
        if(NOT PARSE_ALL OR MIOPEN_TEST_ALL)
            add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${NAME})
        endif()
    endif()
endfunction()




if(MIOPEN_TEST_DEEPBENCH)
    add_custom_test(test_deepbench_rnn
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 16 --seq-len 50 --vector-len 1760 --hidden-size 1760 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 50 --vector-len 1760 --hidden-size 1760 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 64 --seq-len 50 --vector-len 1760 --hidden-size 1760 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 128 --seq-len 50 --vector-len 1760 --hidden-size 1760 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 16 --seq-len 50 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 50 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 64 --seq-len 50 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 128 --seq-len 50 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 16 --seq-len 50 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 50 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 64 --seq-len 50 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 128 --seq-len 50 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 16 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 64 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 128 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 16 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 64 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 128 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 16 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 64 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 128 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 16 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 64 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 128 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 8 --seq-len 50 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 16 --seq-len 50 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 50 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 16 --seq-len 150 --vector-len 256 --hidden-size 256 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 150 --vector-len 256 --hidden-size 256 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 64 --seq-len 150 --vector-len 256 --hidden-size 256 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 1500 --vector-len 2816 --hidden-size 2816 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 750 --vector-len 2816 --hidden-size 2816 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 375 --vector-len 2816 --hidden-size 2816 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 187 --vector-len 2816 --hidden-size 2816 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 1500 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 750 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 375 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 187 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 1500 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 750 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 375 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 187 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 1500 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 750 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 375 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 187 --vector-len 2560 --hidden-size 2560 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 1 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 1500 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 64 --seq-len 1500 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill
    )
endif()


add_custom_test(test_rnn_extra ALL
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --no-hx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 1 --no-hx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 1 --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 1 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 0 --no-hx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 0 --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 0 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 1 --no-hx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 1 --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 1 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --no-hy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 1 --no-hy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 1 --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 1 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 0 --no-hy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 0 --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 0 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 1 --no-hy
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 1 --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 1 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --no-hx --no-dhy --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --rnn-mode 1 --no-hx --no-dhy --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 0 --no-hx --no-dhy --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_rnn_vanilla> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --rnn-mode 1 --no-hx --no-dhy --no-hy --no-dhx
)

add_custom_test(test_gru_extra ALL
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dhy
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dhy
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dhx
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dhx
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-dhy --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_gru> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-dhy --no-hy --no-dhx
)

add_custom_test(test_lstm_extra ALL
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dhy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-cx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dcy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cx --no-dcy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dhy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-dhy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-cx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dcy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cx --no-dcy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dhx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy --no-cy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dcx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cy --no-dcx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dhx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy --no-dhx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy --no-cy
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dcx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cy --no-dcx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx
COMMAND $<TARGET_FILE:test_lstm> --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx
)


add_custom_test(test_conv_extra ALL
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	1	1	1	--weights	1	1	2	2	--pads_strides_dilations	0	0	3	3	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	3	227	227	--weights	4	3	11	11	--pads_strides_dilations	0	0	4	4	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	3	224	224	--weights	4	3	11	11	--pads_strides_dilations	2	2	4	4	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	1	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	64	27	27	--weights	192	64	5	5	--pads_strides_dilations	2	2	1	1	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	64	14	14	--weights	24	64	5	5	--pads_strides_dilations	2	2	1	1	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	96	14	14	--weights	32	96	5	5	--pads_strides_dilations	2	2	1	1	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	16	14	14	--weights	4	16	5	5	--pads_strides_dilations	2	2	1	1	1	1						
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	32	14	14	--weights	4	32	5	5	--pads_strides_dilations	2	2	1	1	1	1						
)


add_custom_test(test_conv_trans ALL
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	128	28	28	--weights	128	128	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	default		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	256	28	28	--weights	256	256	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	same		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	32	28	28	--weights	32	32	5	5	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	default		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	512	14	14	--weights	512	512	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	same		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	512	4	4	--weights	512	512	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	valid		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	64	56	56	--weights	64	64	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	valid		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	100	3	64	64	--weights	3	3	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	100	6	4	4	--weights	6	4	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default		
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	128	28	28	--weights	128	16	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	default	--group-count	8
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	256	28	28	--weights	256	64	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	same	--group-count	4
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	32	28	28	--weights	32	1	5	5	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	default	--group-count	32
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	512	14	14	--weights	512	16	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	same	--group-count	32
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	512	4	4	--weights	512	16	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	valid	--group-count	32
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	64	56	56	--weights	64	2	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	valid	--group-count	32
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	100	3	64	64	--weights	3	3	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default	--group-count	3
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	100	6	4	4	--weights	6	4	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default	--group-count	2
)


add_custom_test(test_conv_group ALL
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	4	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	16	24	240	--weights	32	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	16				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	32	12	120	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	64	6	60	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	64	54	54	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	128	27	27	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	64	112	112	--weights	128	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	2				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	9	224	224	--weights	63	3	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	64	112	112	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	3	224	224	--weights	63	1	7	7	--pads_strides_dilations	3	3	2	2	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	192	28	28	--weights	32	12	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	832	7	7	--weights	128	52	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	192	28	28	--weights	32	24	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	832	7	7	--weights	128	104	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	11	23	161	700	--weights	46	1	7	7	--pads_strides_dilations	1	1	2	2	1	1	--group-count	23				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	7				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	7				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	7				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	7				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	7				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3				
COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	3				
)

add_custom_test(test_conv_3d ALL
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input 16    32   4    9     9  --weights    64    32   3  3    3  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  4     3   4  227   227  --weights     4     3   3 11   11  --pads_strides_dilations  0  0  0    1  1   1    1   1   1  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input 16   128   4   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  1  1    1  1   1    1   1   1  --group-count   32  --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input 16   128  56   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  2  3    1  1   1    1   2   3  --group-count   32  --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  4     4   4  161   700  --weights    32     1   3  5   20  --pads_strides_dilations  1  1  1    2  2   2    1   1   1  --group-count   4   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  8   512   4   28    28  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  1   1    1   1   1  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  8   512   4   56    56  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  8   512   3   14    14  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --trans_output_pads 0 0 0 --group-count   1   --cmode trans  --pmode   same
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input 16    64   3    4     4  --weights    64    32   1  3    3  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --trans_output_pads 0 0 0 --group-count   4   --cmode trans  --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input 16    32   4    9     9  --weights    64    32   3  3    3  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  4     3   4  227   227  --weights     4     3   3 11   11  --pads_strides_dilations  0  0  0    1  1   1    1   2   3  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input 16   128   4   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  2  3    1  1   1    1   2   3  --group-count   32  --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  4     4   4  161   700  --weights    32     1   3  5   20  --pads_strides_dilations  1  2  3    1  2   3    1   2   3  --group-count   4   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  8   512   4   28    28  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  1   1    1   2   3  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  8   512   4   56    56  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input  8   512   3   14    14  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --trans_output_pads 0 0 0 --group-count   1   --cmode trans  --pmode   same
COMMAND $<TARGET_FILE:test_conv> --verbose --conv_dim_type conv3d --input 16    64   3    4     4  --weights    64    32   1  3    3  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --trans_output_pads 0 0 0 --group-count   4   --cmode trans  --pmode   default
)

add_custom_test(test_immed_conv_for_implicit_gemm ALL
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	 16 	28	28	--weights	192  16 	3	3	--pads_strides_dilations	0	0	2	2	2	2	--disable-backward-data   --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	 16 	14	14	--weights	160  16 	3	3	--pads_strides_dilations	0	0	2	2	2	2	--disable-backward-data   --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	 16 	 7	 7	--weights	128  16 	3	3	--pads_strides_dilations	0	0	2	2	2	2	--disable-backward-data   --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	 16 	55	55	--weights	 96  16 	1	7	--pads_strides_dilations	0	0	2	2	1	2	--disable-backward-data   --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	 16 	28	28	--weights	 64  16 	1	7	--pads_strides_dilations	0	0	2	2	1	2	--disable-backward-data   --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	 16 	14	14	--weights	 32  16 	1	7	--pads_strides_dilations	0	0	2	2	1	2	--disable-backward-data   --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	55	55	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	1	1	1	2	--disable-forward         --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	28	28	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	1	1	1	2	--disable-forward         --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	14	14	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	1	1	1	2	--disable-forward         --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	 7	 7	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	1	1	1	2	--disable-forward         --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	55	55	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	2	2	1	2	--disable-forward         --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	28	28	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	2	2	1	2	--disable-forward         --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	14	14	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	2	2	1	2	--disable-forward         --disable-backward-weights
COMMAND	$<TARGET_FILE:test_immed_conv>	--verbose	--input	128	128 	 7	 7	--weights	 16 128 	1	1	--pads_strides_dilations	0	0	2	2	1	2	--disable-forward         --disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose   --input 128  32     28  28  --weights   192  32     3   3   --pads_strides_dilations    0   0   2   2   2   2   --disable-backward-data   --disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose   --input 128  32     14  14  --weights   160  32     3   3   --pads_strides_dilations    0   0   2   2   2   2   --disable-backward-data   --disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose   --input 128  32     7   7   --weights   128  32     3   3   --pads_strides_dilations    0   0   2   2   2   2   --disable-backward-data   --disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose   --input 128  32     55  55  --weights    96  32     1   7   --pads_strides_dilations    0   0   2   2   1   2   --disable-backward-data   --disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose   --input 128  32     28  28  --weights    64  32     1   7   --pads_strides_dilations    0   0   2   2   1   2   --disable-backward-data   --disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose   --input 128  32     14  14  --weights    32  32     1   7   --pads_strides_dilations    0   0   2   2   1   2   --disable-backward-data   --disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	64	    56	56	--weights	256	 64	    1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	64	    56	56	--weights	64	 64	    1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	64	    73	73	--weights	80	 64     1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	64      56	56  --weights	64	 64     1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	128	    28	28	--weights	512	128	    1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	160	    73	73	--weights	64	160   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	192	    35	35	--weights	32	192   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	192	    35	35	--weights	48	192   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	192	    35	35	--weights	64	192   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	192 	28	28	--weights	16	192   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	192 	28	28	--weights	32	192   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	192 	28	28	--weights	64	192   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	192 	28	28	--weights	96	192   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256	    35	35	--weights	48	256   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256	    35	35	--weights	64	256   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256	    56	56	--weights	128	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256	    56	56	--weights	512	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256	    56	56	--weights	64	256	    1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256 	28	28	--weights	128	256   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256 	28	28	--weights	32	256   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	256 	28	28	--weights	64	256   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	288	    35	35	--weights	48	288   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	288	    35	35	--weights	64	288   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	384	    35	35	--weights	192	384   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	384	    35	35	--weights	64	384   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	384	    35	35	--weights	96	384   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	480 	14	14	--weights	16	480   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	480 	14	14	--weights	192	480   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	480 	14	14	--weights	64	480   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	480 	14	14	--weights	96	480   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512	    28	28	--weights	128	512	    1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512	    28	28	--weights	256	512	    1	1	--pads_strides_dilations	0	0	2	2	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512 	14	14	--weights	112	512   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512 	14	14	--weights	128	512   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512 	14	14	--weights	144	512   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512 	14	14	--weights	160	512   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512 	14	14	--weights	24	512   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512 	14	14	--weights	32	512   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
COMMAND $<TARGET_FILE:test_immed_conv>  --verbose	--input	64	512 	14	14	--weights	64	512   	1	1	--pads_strides_dilations	0	0	1	1	1	1	--disable-backward-weights
)

if(MIOPEN_TEST_DEEPBENCH)
    add_custom_test(test_deepbench_conv
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	4	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	32	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	1	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	16	24	240	--weights	32	16	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	32	12	120	--weights	64	32	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	64	6	60	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	108	108	--weights	64	3	3	3	--pads_strides_dilations	1	1	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	64	54	54	--weights	64	64	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	128	27	27	--weights	128	128	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	128	14	14	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	256	7	7	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	3	224	224	--weights	64	3	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	64	112	112	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	128	56	56	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	256	28	28	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	512	14	14	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	8	512	7	7	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	3	224	224	--weights	64	3	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	64	112	112	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	128	56	56	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	256	28	28	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	512	14	14	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	512	7	7	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	3	224	224	--weights	64	3	7	7	--pads_strides_dilations	3	3	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	192	28	28	--weights	32	192	5	5	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	512	14	14	--weights	48	512	5	5	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	832	7	7	--weights	128	832	5	5	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	192	28	28	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	512	14	14	--weights	48	512	1	1	--pads_strides_dilations	0	0	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	16	832	7	7	--weights	128	832	1	1	--pads_strides_dilations	0	0	1	1	1	1						
)
endif()

if(MIOPEN_TEST_CONV)
    add_custom_test(test_miopen_conv
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	1	1	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	64	56	56	--weights	1	64	1	1	--pads_strides_dilations	0	0	2	2	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	32	32	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	224	224	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	227	227	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	231	231	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	224	224	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	227	227	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	231	231	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	224	224	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	231	231	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	64	56	56	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	64	112	112	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	64	512	1024	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	96	27	27	--weights	1	96	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	96	28	28	--weights	1	96	3	3	--pads_strides_dilations	2	2	1	1	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	32	32	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	224	224	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	227	227	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	231	231	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	32	32	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	224	224	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	227	227	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	231	231	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	224	224	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	3	231	231	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	16	14	14	--weights	1	16	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	16	28	28	--weights	1	16	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	24	14	14	--weights	1	24	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	32	7	7	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	32	8	8	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	32	14	14	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	32	16	16	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	32	28	28	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1						
    COMMAND	$<TARGET_FILE:test_conv>	--verbose	--input	1	48	7	7	--weights	1	48	5	5	--pads_strides_dilations	0	0	4	4	1	1						
)
endif()
