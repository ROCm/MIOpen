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
option( MIOPEN_TEST_GFX908 "Test arch gfx908 mode" OFF )
option( MIOPEN_TEST_CONV Off)
option( MIOPEN_TEST_DEEPBENCH Off)
option( MIOPEN_TEST_DRIVER_ITER_MODE Off)
option( MIOPEN_TEST_MIOTENSILE "Test MIOpenTensile path" OFF )
option( MIOPEN_TEST_MLIR "Add tests for MLIR -- EXPERIMENTAL" ${MIOPEN_USE_MLIR} )

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

set(MIOPEN_TEST_NONXDLOPS TRUE)
if(MIOPEN_TEST_GFX908)
    set(MIOPEN_TEST_NONXDLOPS FALSE)
endif()

# The usage is non-trivial, see function add_test_command.
set(SKIP_ALL_EXCEPT_TESTS dummy)
set(SKIP_TESTS dummy) # dummy is for REMOVE_DUPLICATES
set(MIOPEN_TEST_FLOAT_ARG)


set(MIOPEN_TEST_FLOAT Off)
if(MIOPEN_TEST_HALF)
    set(MIOPEN_TEST_FLOAT_ARG --half)
elseif(MIOPEN_TEST_INT8)
    set(MIOPEN_TEST_FLOAT_ARG --int8)
elseif(MIOPEN_TEST_BFLOAT16)
    set(MIOPEN_TEST_FLOAT_ARG --bfloat16)
else()
    set(MIOPEN_TEST_FLOAT On)
endif()

if(NOT MIOPEN_TEST_MIOTENSILE)
	if(MIOPEN_TEST_HALF)
	    if(MIOPEN_BACKEND_OPENCL)
		set(SKIP_TESTS test_gru test_rnn_vanilla test_lstm test_conv_igemm_dynamic)
	    endif()
	elseif(MIOPEN_TEST_INT8)
	    set(SKIP_ALL_EXCEPT_TESTS test_tensor_vec test_tensor_cast test_tensor_trans test_tensor_copy test_tensor_set test_tensor_transform test_conv2d)
	elseif(MIOPEN_TEST_BFLOAT16)
	    set(SKIP_ALL_EXCEPT_TESTS test_conv2d test_tensor_copy test_tensor_set test_tensor_vec test_immed_conv2d test_check_numerics_test)
	    if(MIOPEN_TEST_GFX908)
		    list(APPEND SKIP_ALL_EXCEPT_TESTS test_conv_extra test_conv_for_implicit_gemm test_miopen_conv test_deepbench_conv)
	    endif()
	endif()
else()
	if(MIOPEN_TEST_HALF)
	    set(SKIP_ALL_EXCEPT_TESTS test_conv2d test_conv3d test_immed_conv2d test_immed_conv3d test_gru test_rnn_vanilla test_lstm test_gru_extra test_rnn_extra test_lstm_extra )
	elseif(MIOPEN_TEST_INT8)
	    set(SKIP_ALL_EXCEPT_TESTS test_conv2d)
	elseif(MIOPEN_TEST_BFLOAT16)
	    set(SKIP_ALL_EXCEPT_TESTS test_conv2d test_immed_conv2d)
    else()
        set(SKIP_ALL_EXCEPT_TESTS test_conv2d test_conv3d test_immed_conv2d test_immed_conv3d test_gru test_rnn_vanilla test_lstm test_gru_extra test_rnn_extra test_lstm_extra )
	endif()
endif()

if(MIOPEN_TEST_GFX908)
    list(APPEND SKIP_TESTS test_main test_tensor_scale test_tensor_set test_tensor_transform test_tensor_vec test_w_supertensor test_dropout test_immed_conv3d test_conv3d test_soft_max test_fusion_aux test_activation test_lrn_test test_ctc test_conv2d_bias test_conv3d_bias test_cba_inference test_cbna_inference test_pooling2d test_na_train test_na_inference test_bn_aux test_conv_igemm_dynamic)
endif()

list(REMOVE_DUPLICATES SKIP_TESTS)
list(REMOVE_DUPLICATES SKIP_ALL_EXCEPT_TESTS)

function(add_test_command NAME EXE)
    # Restrict the use of SKIP_ALL_EXCEPT_TESTS list in the Int8, BF16 and MIOpenTensile tests
    if(((NOT (NAME IN_LIST SKIP_ALL_EXCEPT_TESTS)) AND (MIOPEN_TEST_INT8 OR MIOPEN_TEST_BFLOAT16 OR MIOPEN_TEST_MIOTENSILE)) OR (NAME IN_LIST SKIP_TESTS))
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
    if(NOT MIOPEN_EMBED_DB STREQUAL "")
        target_link_libraries(${TEST_NAME} MIOpen miopen_data)
    else()
        target_link_libraries(${TEST_NAME} MIOpen)
    endif()
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
# All the tests are manually sorted in descending order of durations taken from
# Jenkins, "Full long tests"/"HIP Release All". This is to exploit ctest's
# parallelizm as much as possible. Before this change, there were some long
# jobs at the end, utilizing only one CPU core. This order can potentially
# save up to ~23 minutes (20%) of total time of "HIP Release All".
#
# We use two lists, LONG_TESTS and SHORT_TESTS to implement the sorting
# mentioned above. If you would like to add a new test, insert it into the
# LONG_TESTS if it takes more than 800 seconds, or to SHORT_TESTS otherwise.
# Please notice that each list is also sorted and it is highly recommended
# to keep this sorting when adding new tests.

set( LONG_TESTS
    pooling2d.cpp
    dropout.cpp
    conv2d.cpp
    pooling3d.cpp
    soft_max.cpp
    lrn_test.cpp
    reduce_test.cpp
    )

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST})
endforeach()

set_tests_properties(
    test_pooling2d
    test_dropout
    test_conv2d
    test_pooling3d
    test_soft_max
    test_lrn_test
    PROPERTIES COST 800)

set_tests_properties(test_sqlite_perfdb test_perfdb
    PROPERTIES RUN_SERIAL On)

# add_sanitize_test(perfdb.cpp)
# add_sanitize_test(cache.cpp)
# add_sanitize_test(tensor_test.cpp)
# add_sanitize_test(type_name.cpp)

function(add_custom_test NAME)
    set(options SKIP_UNLESS_ALL ALLOW_BFLOAT16 ALLOW_HALF ALLOW_INT8 ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE)
    set(oneValueArgs)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Many custom tests do test only FP32 data type and therefore
    # added only if none of MIOPEN_TEST_HALF, MIOPEN_TEST_INT8, MIOPEN_TEST_BFLOAT16
    # are set, except the test is allowed explicitly.
    # Some tests are xDLOPs specific and should not run on gfx900/906 targets.
    if((NOT (MIOPEN_TEST_INT8 OR MIOPEN_TEST_BFLOAT16 OR MIOPEN_TEST_HALF OR MIOPEN_TEST_MIOTENSILE OR MIOPEN_TEST_NONXDLOPS))
    OR (MIOPEN_TEST_BFLOAT16 AND ${PARSE_ALLOW_BFLOAT16} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_HALF AND ${PARSE_ALLOW_HALF} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_INT8 AND ${PARSE_ALLOW_INT8} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_NONXDLOPS AND ${PARSE_ALLOW_NONXDLOPS} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_FLOAT AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE})))
        add_custom_target(${NAME} ${PARSE_UNPARSED_ARGUMENTS})
        if(NOT PARSE_SKIP_UNLESS_ALL OR MIOPEN_TEST_ALL)
            add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${NAME})
            set_tests_properties(${NAME} PROPERTIES COST 600)
        endif()
    endif()
endfunction()

function(add_perf_test NAME)
    set(options SKIP_UNLESS_ALL ALLOW_BFLOAT16 ALLOW_HALF ALLOW_INT8 ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE)
    set(oneValueArgs)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Many custom tests do test only FP32 data type and therefore
    # added only if none of MIOPEN_TEST_HALF, MIOPEN_TEST_INT8, MIOPEN_TEST_BFLOAT16
    # are set, except the test is allowed explicitly.
    # Some tests are xDLOPs specific and should not run on gfx900/906 targets.
    if((NOT (MIOPEN_TEST_INT8 OR MIOPEN_TEST_BFLOAT16 OR MIOPEN_TEST_HALF OR MIOPEN_TEST_MIOTENSILE OR MIOPEN_TEST_NONXDLOPS))
    OR (MIOPEN_TEST_BFLOAT16 AND ${PARSE_ALLOW_BFLOAT16} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_HALF AND ${PARSE_ALLOW_HALF} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_INT8 AND ${PARSE_ALLOW_INT8} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_NONXDLOPS AND ${PARSE_ALLOW_NONXDLOPS} AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE}))
    OR (MIOPEN_TEST_FLOAT AND (NOT MIOPEN_TEST_MIOTENSILE OR ${PARSE_ALLOW_GEMM_FEATURE})))
        add_custom_target(${NAME} ${PARSE_UNPARSED_ARGUMENTS})
        if(NOT PARSE_SKIP_UNLESS_ALL OR MIOPEN_TEST_ALL)
            add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${NAME})
            set_tests_properties(${NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "(FAILED)|(Perf Db: record not found)")
            set_tests_properties(${NAME} PROPERTIES COST 600)
        endif()
    endif()
endfunction()

set(IMPLICITGEMM_ARGS ${MIOPEN_TEST_FLOAT_ARG})

# ./bin/MIOpenDriver conv -n 128 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
# MIOPEN_DEBUG_CONV_IMMED_FALLBACK=0
if(MIOPEN_EMBED_DB)
    set(MIOPEN_EMBED_TEST_ARG ${MIOPEN_TEST_FLOAT_ARG} --disable-validation --verbose)
    # WORKAROUND for issue #874
    set(MIOPEN_WA_ISSUE_874_F  MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0)
    set(MIOPEN_WA_ISSUE_874_W  MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0)
    set(MIOPEN_WA_ISSUE_874_FW MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0)
add_perf_test(test_conv_embed_db
    COMMAND ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_F}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND                           $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 128 28 28 --weights 128 128 3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 128 28 28 --weights 512 128 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 2048 7 7 --weights 512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_F}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 14 14 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 14 14 --weights 256 256 3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 128 256 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND                           $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND                           $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 3 230 230   --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 1024 512 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 128 512 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 7 7   --weights 2048 512 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 7 7   --weights 512 512 3 3  --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 256 64 1 1  --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 64 64 1 1  --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 64 64 3 3   --pads_strides_dilations 1 1 1 1 1 1
)
endif()

if(MIOPEN_TEST_MLIR)
    set(IMPLICITGEMM_MLIR_ENV_BASE MIOPEN_FIND_MODE=normal)
    set(IMPLICITGEMM_MLIR_ENV_F_CPP ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmMlirCppFwd)
    set(IMPLICITGEMM_MLIR_ENV_F_BIN ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmMlirBinFwd)
    set(IMPLICITGEMM_MLIR_ENV_B ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmMlirCppBwd)
    set(IMPLICITGEMM_MLIR_ENV_W ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmMlirCppWrW)

    set(IMPLICITGEMM_MLIR_ARGS_F ${IMPLICITGEMM_ARGS} --verbose --disable-backward-data --disable-backward-weights)
    set(IMPLICITGEMM_MLIR_ARGS_B ${IMPLICITGEMM_ARGS} --verbose --disable-forward --disable-backward-weights)
    set(IMPLICITGEMM_MLIR_ARGS_W ${IMPLICITGEMM_ARGS} --verbose --disable-forward --disable-backward-data)
    
    add_custom_test(test_conv_igemm_mlir ALLOW_NONXDLOPS
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F_CPP} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F_CPP} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 64 256 56 56  --weights  256  256 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F_CPP} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 64 128 58 58  --weights  128  128 3 3 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F_BIN} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 64 128 58 58  --weights  128  128 3 3 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F_BIN} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 8 32 32  --weights  128  8 3 3 --pads_strides_dilations 0 0 1 1 1 1

        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 64 256 56 56  --weights  256  256 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 64 128 58 58  --weights  128  128 3 3 --pads_strides_dilations 0 0 1 1 1 1

        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 64 256 56 56  --weights  256  256 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 64 128 58 58  --weights  128  128 3 3 --pads_strides_dilations 0 0 1 1 1 1
    )
endif()

add_custom_test(test_conv_for_implicit_gemm SKIP_UNLESS_ALL ALLOW_BFLOAT16 ALLOW_HALF ALLOW_NONXDLOPS
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 16	28	28	--weights	192  16		3	3	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 16	14	14	--weights	160  16		3	3	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 16	 7	 7	--weights	128  16		3	3	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 16	55	55	--weights	 96  16		1	7	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 16	28	28	--weights	 64  16		1	7	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 16	14	14	--weights	 32  16		1	7	--pads_strides_dilations	0	0	2	2	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose --input 64   32     28  28  --weights   192  32     3   3   --pads_strides_dilations    0   0   2   2   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose --input 64   32     14  14  --weights   160  32     3   3   --pads_strides_dilations    0   0   2   2   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose --input 64   32     7   7   --weights   128  32     3   3   --pads_strides_dilations    0   0   2   2   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose --input 64   32     55  55  --weights    96  32     1   7   --pads_strides_dilations    0   0   2   2   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose --input 64   32     28  28  --weights    64  32     1   7   --pads_strides_dilations    0   0   2   2   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose --input 64   32     14  14  --weights   32   32     1   7   --pads_strides_dilations    0   0   2   2   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 64	    56	56	--weights	256	 64	    1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 64	    56	56	--weights	64	 64	    1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 64	    73	73	--weights	80	 64     1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	 64     56	56  --weights	64	 64     1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	55	55	--weights	16  128		1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	28	28	--weights	16  128		1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	14	14	--weights	16  128		1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	 7	 7	--weights	16  128		1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64     56  56  --weights   256  64     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64     56  56  --weights   64   64     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64     73  73  --weights   80   64     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64     56  56  --weights   64   64     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128     55  55  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128     28  28  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1
# COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128     14  14  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128      7   7  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	55	55	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	28	28	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	14	14	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	 7	 7	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	    28	28	--weights	512	128	    1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	160	    73	73	--weights	64	160	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	    35	35	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	    35	35	--weights	48	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	    35	35	--weights	64	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	16	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	64	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	96	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    35	35	--weights	48	256	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    35	35	--weights	64	256	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    56	56	--weights	128	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    56	56	--weights	512	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    56	56	--weights	64	256	    1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	28	28	--weights	128	256	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	28	28	--weights	32	256	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	28	28	--weights	64	256	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	288	    35	35	--weights	48	288	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	288	    35	35	--weights	64	288	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	384	    35	35	--weights	192	384	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	384	    35	35	--weights	64	384	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	384	    35	35	--weights	96	384	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	16	480	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	192	480	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	64	480	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	96	480	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	    28	28	--weights	128	512	    1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	    28	28	--weights	256	512	    1	1	--pads_strides_dilations	0	0	2	2	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	112	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	128	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	144	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	160	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	24	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	32	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	64	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   32  832  1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   192  832  1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   128  832  1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   32  832  1   1   --pads_strides_dilations    0   0   1   1   2   2
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   192  832  1   1   --pads_strides_dilations    0   0   1   1   2   2
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   128  832  1   1   --pads_strides_dilations    0   0   1   1   2   2
COMMAND $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  2048    7  7  --weights   192  2048 1   1   --pads_strides_dilations    0   0   1   1   2   2
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	28 28 --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	14 14 --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	7 7   --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	28 28 --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	14 14 --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1
COMMAND	$<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	7 7   --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1
)

add_custom_test(test_conv_group SKIP_UNLESS_ALL ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
#
# Workaround for "Memory access fault by GPU node" during "HIP Release All" - WrW disabled.
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32 --disable-backward-weights
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	4	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	16	24	240	--weights	32	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	16
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	32	12	120	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	64	6	60	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	54	54	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	128	27	27	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	112	112	--weights	128	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	9	224	224	--weights	63	3	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3
#
# Workaround for "Memory access fault by GPU node" during "FP32 gfx908 Hip Release All subset" - WrW disabled.
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	64	112	112	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4 --disable-backward-weights
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	3	224	224	--weights	63	1	7	7	--pads_strides_dilations	3	3	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	192	28	28	--weights	32	12	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	832	7	7	--weights	128	52	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	192	28	28	--weights	32	24	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	832	7	7	--weights	128	104	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	11	23	161	700	--weights	46	1	7	7	--pads_strides_dilations	1	1	2	2	1	1	--group-count	23
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	3
)

if(MIOPEN_TEST_DEEPBENCH)
    add_custom_test(test_deepbench_rnn ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
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


add_custom_test(test_rnn_extra SKIP_UNLESS_ALL ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
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

add_custom_test(test_gru_extra SKIP_UNLESS_ALL ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
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

add_custom_test(test_lstm_extra SKIP_UNLESS_ALL ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
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


add_custom_test(test_conv_extra SKIP_UNLESS_ALL ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
# COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	1	1	1	--weights	1	1	2	2	--pads_strides_dilations	0	0	3	3	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	3	227	227	--weights	4	3	11	11	--pads_strides_dilations	0	0	4	4	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	3	224	224	--weights	4	3	11	11	--pads_strides_dilations	2	2	4	4	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	64	27	27	--weights	192	64	5	5	--pads_strides_dilations	2	2	1	1	1	1
# COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	64	14	14	--weights	24	64	5	5	--pads_strides_dilations	2	2	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	96	14	14	--weights	32	96	5	5	--pads_strides_dilations	2	2	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	16	14	14	--weights	4	16	5	5	--pads_strides_dilations	2	2	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	14	14	--weights	4	32	5	5	--pads_strides_dilations	2	2	1	1	1	1
)


add_custom_test(test_conv_trans SKIP_UNLESS_ALL ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	128	28	28	--weights	128	128	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	default
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	256	28	28	--weights	256	256	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	same
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	32	28	28	--weights	32	32	5	5	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	default
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	512	14	14	--weights	512	512	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	same
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	512	4	4	--weights	512	512	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	valid
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	56	56	--weights	64	64	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	valid
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	100	3	64	64	--weights	3	3	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	100	6	4	4	--weights	6	4	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	128	28	28	--weights	128	16	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	default	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	256	28	28	--weights	256	64	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	same	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	32	28	28	--weights	32	1	5	5	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	default	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	512	14	14	--weights	512	16	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	same	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	512	4	4	--weights	512	16	1	1	--pads_strides_dilations	0	0	1	1	1	1	--cmode	trans	--pmode	valid	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	56	56	--weights	64	2	1	1	--pads_strides_dilations	0	0	2	2	1	1	--cmode	trans	--pmode	valid	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	100	3	64	64	--weights	3	3	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	100	6	4	4	--weights	6	4	1	1	--pads_strides_dilations	2	2	1	1	1	1	--cmode	trans	--pmode	default	--group-count	2
)


add_custom_test(test_conv_3d SKIP_UNLESS_ALL ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input 16    32   4    9     9  --weights    64    32   3  3    3  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  4     3   4  227   227  --weights     4     3   3 11   11  --pads_strides_dilations  0  0  0    1  1   1    1   1   1  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input 16   128   4   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  1  1    1  1   1    1   1   1  --group-count   32  --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input 16   128  56   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  2  3    1  1   1    1   2   3  --group-count   32  --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  4     4   4  161   700  --weights    32     1   3  5   20  --pads_strides_dilations  1  1  1    2  2   2    1   1   1  --group-count   4   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  8   512   4   28    28  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  1   1    1   1   1  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  8   512   4   56    56  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  8   512   3   14    14  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --trans_output_pads 0 0 0 --group-count   1   --cmode trans  --pmode   same
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input 16    64   3    4     4  --weights    64    32   1  3    3  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --trans_output_pads 0 0 0 --group-count   4   --cmode trans  --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input 16    32   4    9     9  --weights    64    32   3  3    3  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  4     3   4  227   227  --weights     4     3   3 11   11  --pads_strides_dilations  0  0  0    1  1   1    1   2   3  --group-count   1   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input 16   128   4   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  2  3    1  1   1    1   2   3  --group-count   32  --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  4     4   4  161   700  --weights    32     1   3  5   20  --pads_strides_dilations  1  2  3    1  2   3    1   2   3  --group-count   4   --cmode conv   --pmode   default
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  8   512   4   28    28  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  1   1    1   2   3  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  8   512   4   56    56  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --group-count   4   --cmode conv   --pmode   same
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input  8   512   3   14    14  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --trans_output_pads 0 0 0 --group-count   1   --cmode trans  --pmode   same
COMMAND $<TARGET_FILE:test_conv3d> --verbose --conv_dim_type conv3d --input 16    64   3    4     4  --weights    64    32   1  3    3  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --trans_output_pads 0 0 0 --group-count   4   --cmode trans  --pmode   default
)

set(DYNAMIC_IMPLICITGEMM_COMMON
    MIOPEN_DEBUG_CONV_FFT=0
    MIOPEN_DEBUG_CONV_GEMM=0
    MIOPEN_DEBUG_CONV_WINOGRAD=0)
set(DYNAMIC_IMPLICITGEMM_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmV4R1DynamicFwd)
set(DYNAMIC_IMPLICITGEMM_1X1_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmV4R1DynamicFwd_1x1)
set(DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicFwdXdlops)
set(DYNAMIC_IMPLICITGEMM_BWD_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmV4R1DynamicBwd)
set(DYNAMIC_IMPLICITGEMM_WRW_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmV4R1DynamicWrw)
set(DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicBwdXdlops)

set(DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicWrwXdlops)

add_custom_test(test_conv_igemm_dynamic_small ALLOW_NONXDLOPS
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  16  16 56 56 --weights 64  16 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  16  64 34 34 --weights 64  64 3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  32  32 17 17 --weights 32  32 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_1X1_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  16 384  8  8 --weights 64 384 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic SKIP_UNLESS_ALL ALLOW_NONXDLOPS
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  64  256 34 34 --weights 256  256 3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input 128  128 35 35 --weights 128  128 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  64 1536  8  8 --weights 256 1536 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input 128   48  7  7 --weights 128   48 5 5 --pads_strides_dilations 2 2 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input 128  128 17 17 --weights 128  128 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_1X1_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  256 28 28 --weights 128  256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_1X1_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64 1536  8  8 --weights 256 1536 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_1X1_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  768 17 17 --weights 128  768 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  32  128 34 34 --weights 64  128  3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  128 35 35 --weights 128  128 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  256 56 56 --weights 64  256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  32  128 34 34 --weights 64  128  3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  128 35 35 --weights 128  128 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  256 56 56 --weights 64  256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic_xdlops_bwd SKIP_UNLESS_ALL
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  4  512 128 128 --weights 12  512  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
)
add_custom_test(test_conv_igemm_dynamic_xdlops_fwd SKIP_UNLESS_ALL ALLOW_HALF
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 64 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 64 2048 7 7 --weights 2048 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 64 3 224 224 --weights 64 3 7 7 --pads_strides_dilations 3 3 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128 128 17 17 --weights 128 128 7 1 --pads_strides_dilations 3 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128 128 17 17 --weights 128 128 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128 192 17 17 --weights 320 192 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128 256 35 35 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128 48 35 35 --weights 64 48 5 5 --pads_strides_dilations 2 2 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 64 512 7 7 --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 64 3 230 230 --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 32 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 2 256 100 104 --weights 12 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 1 256 28 28 --weights 80 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
)
add_custom_test(test_conv_igemm_dynamic_xdlops_wrw SKIP_UNLESS_ALL ALLOW_HALF
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  4  512 128 128 --weights 12  512  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> --verbose --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
)

if(MIOPEN_TEST_DEEPBENCH)
    add_custom_test(test_deepbench_conv ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	16	24	240	--weights	32	16	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	32	12	120	--weights	64	32	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	64	6	60	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	64	3	3	3	--pads_strides_dilations	1	1	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	54	54	--weights	64	64	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	128	27	27	--weights	128	128	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	128	14	14	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	256	7	7	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	224	224	--weights	64	3	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	112	112	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	128	56	56	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	256	28	28	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	512	14	14	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	512	7	7	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	3	224	224	--weights	64	3	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	64	112	112	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	128	56	56	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	256	28	28	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	14	14	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	7	7	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	3	224	224	--weights	64	3	7	7	--pads_strides_dilations	3	3	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	192	28	28	--weights	32	192	5	5	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	14	14	--weights	48	512	5	5	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	832	7	7	--weights	128	832	5	5	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	192	28	28	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	14	14	--weights	48	512	1	1	--pads_strides_dilations	0	0	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	832	7	7	--weights	128	832	1	1	--pads_strides_dilations	0	0	1	1	1	1
)
endif()

if(MIOPEN_TEST_CONV)
    add_custom_test(test_miopen_conv ALLOW_NONXDLOPS ALLOW_GEMM_FEATURE
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	1	1	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	64	56	56	--weights	1	64	1	1	--pads_strides_dilations	0	0	2	2	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	32	32	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	224	224	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	227	227	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	231	231	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	224	224	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	227	227	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	231	231	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	224	224	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	231	231	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	64	56	56	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	64	112	112	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	64	512	1024	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	96	27	27	--weights	1	96	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	96	28	28	--weights	1	96	3	3	--pads_strides_dilations	2	2	1	1	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	32	32	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	224	224	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	227	227	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	231	231	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	32	32	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	224	224	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	227	227	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	231	231	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	224	224	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	3	231	231	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	16	14	14	--weights	1	16	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	16	28	28	--weights	1	16	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	24	14	14	--weights	1	24	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	32	7	7	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	32	8	8	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	32	14	14	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	32	16	16	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	32	28	28	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1
    COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	48	7	7	--weights	1	48	5	5	--pads_strides_dilations	0	0	4	4	1	1
)
endif()
