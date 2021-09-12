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
option( MIOPEN_TEST_GFX908 "Test on MI100 (gfx908)" OFF )
option( MIOPEN_TEST_GFX90a "Test on gfx90a" OFF )
option( MIOPEN_TEST_VEGA "Test on Vega10/20 (gfx900, gfx906)" OFF )
option( MIOPEN_TEST_GFX1030 "Test on Navi21 (gfx1030)" OFF )
option( MIOPEN_TEST_CONV Off)
option( MIOPEN_TEST_DEEPBENCH Off)
option( MIOPEN_TEST_DRIVER_ITER_MODE Off)
option( MIOPEN_TEST_MIOTENSILE "Test MIOpenTensile path" OFF )
option( MIOPEN_TEST_MLIR "Add tests for MLIR -- EXPERIMENTAL" ${MIOPEN_USE_MLIR} )

option( WORKAROUND_ISSUE_898 "" ON)
option( WORKAROUND_ISSUE_936 "" ON)
option( WORKAROUND_ISSUE_1053 "" ON)

# Run the test suite to a depth limit
#limit greater than 2 leads to prolonged testing more than 5hrs per stage.
set(MIOPEN_TEST_LIMIT "2" CACHE STRING "")
set(MIOPEN_TEST_FLAGS "" CACHE STRING "")
set(MIOPEN_TEST_GDB On CACHE BOOL "")

set(MIOPEN_TEST_OPENCL FALSE)
if(MIOPEN_BACKEND_OPENCL)
    set(MIOPEN_TEST_OPENCL TRUE)
endif()

set(MIOPEN_TEST_HIP_NOGPU FALSE)
if(MIOPEN_MODE_NOGPU)
    set(MIOPEN_TEST_HIP_NOGPU TRUE)
endif()

set(MIOPEN_TEST_HIP FALSE)
if(MIOPEN_BACKEND_HIP AND NOT MIOPEN_TEST_HIP_NOGPU)
    set(MIOPEN_TEST_HIP TRUE)
endif()

# Detect GPU type for testing.
# For HIP_NOGPU backend, GPU detection is not required and should be disabled.
# Also we do not detect GPU when target GPU for testing is specified explicitly.
set(MIOPEN_TEST_GPU_DETECTION_FAILED FALSE)
set(MIOPEN_NO_GPU FALSE)
if(NOT (MIOPEN_TEST_VEGA OR MIOPEN_TEST_GFX908 OR MIOPEN_TEST_GFX90a OR MIOPEN_TEST_GFX1030 OR MIOPEN_TEST_HIP_NOGPU))
    find_program(ROCMINFO
        NAMES rocminfo
        PATHS
            /opt/rocm
            ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES
            /bin
    )
    message(STATUS "rocminfo utility: ${ROCMINFO}")
    if(ROCMINFO)
        execute_process (
            COMMAND ${ROCMINFO}
            OUTPUT_VARIABLE ROCMINFO_OUTPUT
            RESULT_VARIABLE ROCMINFO_EXIT_STATUS
        )
        if(ROCMINFO_OUTPUT MATCHES "no GPU devices")
            message(WARNING "ROCk module is NOT loaded, possibly no GPU devices")
            set(MIOPEN_NO_GPU TRUE)         
        elseif (NOT ROCMINFO_EXIT_STATUS EQUAL 0)
            message(WARNING "ROCMINFO FAILED, GPU TYPE UNKNOWN. Manually set respective MIOPEN_TEST_GFX* CMake variable to specify target GPU for testing.")
            set(MIOPEN_TEST_GPU_DETECTION_FAILED TRUE)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx1030")
            set(MIOPEN_TEST_GFX1030 ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx900|gfx906")
            set(MIOPEN_TEST_VEGA ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx908")
            set(MIOPEN_TEST_GFX908 ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx90a")
            set(MIOPEN_TEST_GFX90a ON)
        else()
            message(WARNING "TESTING IS NOT SUPPORTED FOR THE DETECTED GPU")
            set(MIOPEN_TEST_GPU_DETECTION_FAILED TRUE)
        endif()
    else()
        message(WARNING "ROCMINFO NOT FOUND, GPU TYPE UNKNOWN. Manually set respective MIOPEN_TEST_GFX* CMake variable to specify target GPU for testing.")
        set(MIOPEN_TEST_GPU_DETECTION_FAILED TRUE)
    endif()
endif()
message(STATUS "MIOPEN_NO_GPU ${MIOPEN_NO_GPU}")
message(STATUS "MIOPEN_TEST_VEGA ${MIOPEN_TEST_VEGA}")
message(STATUS "MIOPEN_TEST_GFX908 ${MIOPEN_TEST_GFX908}")
message(STATUS "MIOPEN_TEST_GFX90a ${MIOPEN_TEST_GFX90a}")
message(STATUS "MIOPEN_TEST_GFX1030 ${MIOPEN_TEST_GFX1030}")
message(STATUS "MIOPEN_TEST_GPU_DETECTION_FAILED ${MIOPEN_TEST_GPU_DETECTION_FAILED}")

if(MIOPEN_TEST_DRIVER_ITER_MODE)
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=2)
else()
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=1)
endif()

find_package(Threads REQUIRED)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

if(MIOPEN_TEST_GPU_DETECTION_FAILED AND NOT (MIOPEN_NO_GPU))
    add_custom_target(gpu_detection_check COMMAND echo "*** FATAL: GPU DETECTION FAILED DURING CMAKE PHASE, CHECK CMAKE WARNINGS ***" COMMAND exit 1)
    add_dependencies(check gpu_detection_check)
endif()

set(MIOPEN_TEST_FLOAT_ARG)
set(MIOPEN_TEST_FLOAT FALSE)
if(MIOPEN_TEST_HALF)
    set(MIOPEN_TEST_FLOAT_ARG --half)
elseif(MIOPEN_TEST_INT8)
    set(MIOPEN_TEST_FLOAT_ARG --int8)
elseif(MIOPEN_TEST_BFLOAT16)
    set(MIOPEN_TEST_FLOAT_ARG --bfloat16)
else()
    set(MIOPEN_TEST_FLOAT_ARG --float)
    set(MIOPEN_TEST_FLOAT TRUE)
endif()

if(NOT MIOPEN_TEST_MIOTENSILE)
	if(MIOPEN_TEST_HALF)
	    if(MIOPEN_BACKEND_OPENCL)
		set(SKIP_TESTS test_gru test_rnn_vanilla test_lstm)
	    endif()
	elseif(MIOPEN_TEST_INT8)
	    set(SKIP_ALL_EXCEPT_TESTS
                test_tensor_vec test_tensor_cast test_tensor_trans test_tensor_copy test_tensor_set
                test_tensor_transform test_conv2d)
	elseif(MIOPEN_TEST_BFLOAT16)
	    set(SKIP_ALL_EXCEPT_TESTS
                test_conv2d test_tensor_copy test_tensor_set test_tensor_vec test_immed_conv2d
                test_check_numerics_test test_conv_extra test_conv_for_implicit_gemm test_miopen_conv
                test_deepbench_conv)
    endif()
    if(${CODECOV_TEST})
        list(APPEND SKIP_TESTS test_conv3d test_immed_conv3d test_immed_conv2d test_pooling2d)
        # replaced by smaller tests with suffix _codecov
    endif()
else()
	if(MIOPEN_TEST_HALF)
	    set(SKIP_ALL_EXCEPT_TESTS test_conv2d test_conv3d test_conv3d_extra test_immed_conv2d
                test_immed_conv3d test_gru test_rnn_vanilla test_lstm test_gru_extra test_rnn_extra
                test_lstm_extra)
	elseif(MIOPEN_TEST_INT8)
	    set(SKIP_ALL_EXCEPT_TESTS test_conv2d)
	elseif(MIOPEN_TEST_BFLOAT16)
	    set(SKIP_ALL_EXCEPT_TESTS test_conv2d test_immed_conv2d)
    else()
        set(SKIP_ALL_EXCEPT_TESTS test_conv2d test_conv3d test_conv3d_extra test_immed_conv2d
                test_immed_conv3d test_gru test_rnn_vanilla test_lstm test_gru_extra
                test_rnn_extra test_lstm_extra)
	endif()
endif()

if (MIOPEN_NO_GPU)
    set(SKIP_ALL_EXCEPT_TESTS test_include_inliner test_kernel_build_params test_lstm test_lstm_dropout 
            test_test_errors test_type_name test_tensor_test test_sqlite_perfdb test_sequences
            test_pooling3d test_perfdb)
endif()

if(MIOPEN_TEST_GFX1030)
    if(WORKAROUND_ISSUE_1053 AND MIOPEN_TEST_ALL)
        list(APPEND SKIP_TESTS test_lrn_test)
    endif()
endif()

# The usage is non-trivial, see function add_test_command.
if(SKIP_TESTS)
    list(REMOVE_DUPLICATES SKIP_TESTS)
endif()
if(SKIP_ALL_EXCEPT_TESTS)
    list(REMOVE_DUPLICATES SKIP_ALL_EXCEPT_TESTS)
endif()
message(STATUS "SKIP_TESTS: ${SKIP_TESTS}")
message(STATUS "SKIP_ALL_EXCEPT_TESTS: ${SKIP_ALL_EXCEPT_TESTS}")


function(add_test_command NAME EXE)
    # Restrict the use of SKIP_ALL_EXCEPT_TESTS list in the Int8, BF16 and MIOpenTensile tests
    if( (NOT (NAME IN_LIST SKIP_ALL_EXCEPT_TESTS) AND SKIP_ALL_EXCEPT_TESTS)
        OR (NAME IN_LIST SKIP_TESTS)
    )
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

    if(WORKAROUND_ISSUE_898 AND MIOPEN_USE_COMGR)
        set_property(TEST ${TEST_NAME} PROPERTY ENVIRONMENT MIOPEN_DEBUG_COMGR_HIP_PCH_ENFORCE=0)
    endif()
endfunction()

separate_arguments(MIOPEN_TEST_FLAGS_ARGS UNIX_COMMAND ${MIOPEN_TEST_FLAGS})

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
    # Cmake does not add flags correctly for gcc
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
    endif()

    set(TEST_COMMAND ${TEST_NAME} ${MIOPEN_TEST_FLOAT_ARG})
    if(MIOPEN_TEST_ALL)
        set(TEST_COMMAND ${TEST_COMMAND} --all)
        if(MIOPEN_TEST_LIMIT GREATER 0)
            set(TEST_COMMAND ${TEST_COMMAND} --limit ${MIOPEN_TEST_LIMIT})
        endif()
    endif()
    set(TEST_COMMAND ${TEST_COMMAND} ${MIOPEN_TEST_FLAGS_ARGS})


    if(WORKAROUND_ISSUE_936 AND (${TEST_NAME} MATCHES "test_conv2d" OR ${TEST_NAME} MATCHES "test_immed_conv2d") )
        set(TEST_COMMAND ${TEST_COMMAND} --tolerance 130) #increased by 1.625 times
    endif()

    add_test_command(${TEST_NAME} ${TEST_COMMAND})
    rate_added_test(${TEST_NAME})
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
#For files NAME = test_ + (File name with neither the directory nor the longest extension)

set(LONG_TESTS
    test_dropout
    test_conv2d
    test_conv3d
    test_conv_group
    test_soft_max
    test_lrn_test
    test_conv_for_implicit_gemm
    test_immed_conv3d
    test_conv3d_extra
    test_conv_3d
    test_pooling2d
    )

function(rate_added_test NAME)
    if(${NAME} IN_LIST LONG_TESTS)
        set_tests_properties(${NAME} PROPERTIES COST 800)
    else()
        set_tests_properties(${NAME} PROPERTIES COST 600)
    endif()
endfunction()



foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST})
endforeach()



set_tests_properties(test_sqlite_perfdb test_perfdb
    PROPERTIES RUN_SERIAL On)

# add_sanitize_test(perfdb.cpp)
# add_sanitize_test(cache.cpp)
# add_sanitize_test(tensor_test.cpp)
# add_sanitize_test(type_name.cpp)

function(bool_equality_f first_arg sec_arg result)
    if(${first_arg})
        if(${sec_arg})
            set(${result} TRUE PARENT_SCOPE)
        else()
            set(${result} FALSE PARENT_SCOPE)
        endif()
    elseif(${sec_arg})
        set(${result} FALSE PARENT_SCOPE)
    else()
        set(${result} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(bool_and_f first_arg sec_arg result)
    if(${first_arg} AND ${sec_arg})
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(bool_or_f first_arg sec_arg result)
    if(${first_arg} OR ${sec_arg})
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(bool_not_f first_arg result)
    if(${first_arg})
        set(${result} FALSE PARENT_SCOPE)
    else()
        set(${result} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(option_support_check is_enabled is_disabled default_result result)
    if(${is_enabled} AND ${is_disabled})
        message(FATAL_ERROR " Incompatible options used")
    endif()
    if(${is_enabled})
        set(${result} TRUE PARENT_SCOPE)
    elseif(${is_disabled})
        set(${result} FALSE PARENT_SCOPE)
    else()
        set(${result} ${default_result} PARENT_SCOPE)
    endif()
endfunction()

# The add_custom_test function contains options to describe the conditions,
# under which new custom_tests should be run. Options are divided into several types.
# The option can be enabled or disabled, if nothing is specified, the default value is taken.
# You can use any number of options, provided that options do not conflict
#   (e.g. "HALF_ENABLE HALF_DISABLE" is illegal)
# 1)First describes supported data type. ( HALF BF16 INT8 FLOAT ...)
#   The option can be enabled or disabled by using '_ENABLED' and '_DISABLED' suffix.
#   If nothing is specified, the default value is taken.
#       Default: HALF=disabled, BF16=disabled, INT8=disabled, FLOAT=enabled.
# 2)Second options type describes support GPU types (gfx900, gfx906, gfx908 ...)
#   The option can be enabled or disabled by using '_ENABLED' and '_DISABLED' suffix.
#   If nothing is specified, the default value is taken.
#       Default: VEGA=enabled, all others disabled.
# 3)Third type describes support for special internal components (MIOTENSILE MLIR ...)
#   The option can be enabled or disabled by using '_ENABLED' and '_DISABLED' suffix.
#   If nothing is specified, the default value is taken.
#       Default: MIOTENSILE=disabled, MLIR=disabled.
# 4)Fourth type describes testing mode.
#   By default they are disabled and can be enabled by option name.
#   SKIP_UNLESS_ALL - this option means that the test is large and should only run if MIOPEN_TEST_ALL=TRUE.
#   TEST_PERF_DB_RECORD_NOT_FOUND - this option means that test failed if output contains: "Perf Db: record not found"
# 5) Fifth type describes supported backend (OCL, HIP, HIP_NOGPU ...)
#   The option can be enabled or disabled by using '_ENABLED' and '_DISABLED' suffix.
#   If nothing is specified, the default value is taken.
#       Default: OCL=enabled, HIP=enabled, HIP_NOGPU=disabled.

function(add_custom_test NAME)
    set(options
        BF16_ENABLED BF16_DISABLED HALF_ENABLED HALF_DISABLED INT8_ENABLED INT8_DISABLED FLOAT_ENABLED FLOAT_DISABLED
        VEGA_ENABLED VEGA_DISABLED GFX908_ENABLED GFX908_DISABLED GFX1030_ENABLED GFX1030_DISABLED
        GFX90A_ENABLED GFX90A_DISABLED MIOTENSILE_ENABLED MIOTENSILE_DISABLED MLIR_ENABLED MLIR_DISABLED
        SKIP_UNLESS_ALL TEST_PERF_DB_RECORD_NOT_FOUND
        OCL_ENABLED OCL_DISABLED HIP_ENABLED HIP_DISABLED HIP_NOGPU_ENABLED HIP_NOGPU_DISABLED
    )
    set(oneValueArgs)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Many custom tests do test only FP32 data type and therefore
    # added only if none of MIOPEN_TEST_HALF, MIOPEN_TEST_INT8, MIOPEN_TEST_BFLOAT16
    # are set, except the test is allowed explicitly.
    set(is_half_check)
    set(HALF_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_HALF_ENABLED} ${PARSE_HALF_DISABLED} ${HALF_TEST_DEFAULT} is_half_check)
    bool_and_f(${MIOPEN_TEST_HALF} ${is_half_check} is_half_check)

    set(is_bfloat16_check)
    set(BF16_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_BF16_ENABLED} ${PARSE_BF16_DISABLED} ${BF16_TEST_DEFAULT} is_bfloat16_check)
    bool_and_f(${MIOPEN_TEST_BFLOAT16} ${is_bfloat16_check} is_bfloat16_check)

    set(is_int8_check)
    set(INT8_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_INT8_ENABLED} ${PARSE_INT8_DISABLED} ${INT8_TEST_DEFAULT} is_int8_check)
    bool_and_f(${MIOPEN_TEST_INT8} ${is_int8_check} is_int8_check)

    set(is_float_check)
    set(FLOAT_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_FLOAT_ENABLED} ${PARSE_FLOAT_DISABLED} ${FLOAT_TEST_DEFAULT} is_float_check)
    bool_and_f(${MIOPEN_TEST_FLOAT} ${is_float_check} is_float_check)

    set(is_miotensile_check)
    set(MIOTENSILE_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_MIOTENSILE_ENABLED} ${PARSE_MIOTENSILE_DISABLED} ${MIOTENSILE_TEST_DEFAULT} is_miotensile_check)
    bool_not_f(${MIOPEN_TEST_MIOTENSILE} NOT_MIOPEN_TEST_MIOTENSILE)
    bool_or_f(${NOT_MIOPEN_TEST_MIOTENSILE} ${is_miotensile_check} is_miotensile_check)

    set(is_mlir_check)
    set(MLIR_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_MLIR_ENABLED} ${PARSE_MLIR_DISABLED} ${MLIR_TEST_DEFAULT} is_mlir_check)
    bool_not_f(${MIOPEN_TEST_MLIR} NOT_MIOPEN_TEST_MLIR)
    bool_or_f(${NOT_MIOPEN_TEST_MLIR} ${is_mlir_check} is_mlir_check)

    set(is_ocl_check)
    set(OCL_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_OCL_ENABLED} ${PARSE_OCL_DISABLED} ${OCL_TEST_DEFAULT} is_ocl_check)
    bool_not_f(${MIOPEN_TEST_OPENCL} NOT_MIOPEN_TEST_OPENCL)
    bool_or_f(${NOT_MIOPEN_TEST_OPENCL} ${is_ocl_check} is_ocl_check)

    set(is_hip_check)
    set(HIP_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_HIP_ENABLED} ${PARSE_HIP_DISABLED} ${HIP_TEST_DEFAULT} is_hip_check)
    bool_not_f(${MIOPEN_TEST_HIP} NOT_MIOPEN_TEST_HIP)
    bool_or_f(${NOT_MIOPEN_TEST_HIP} ${is_hip_check} is_hip_check)

    set(is_hip_nogpu_check)
    set(HIP_NOGPU_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_HIP_NOGPU_ENABLED} ${PARSE_HIP_NOGPU_DISABLED} ${HIP_NOGPU_TEST_DEFAULT} is_hip_nogpu_check)
    bool_not_f(${MIOPEN_TEST_HIP_NOGPU} NOT_MIOPEN_TEST_HIP_NOGPU)
    bool_or_f(${NOT_MIOPEN_TEST_HIP_NOGPU} ${is_hip_nogpu_check} is_hip_nogpu_check)

    # Some tests are xDLOPs specific and should not run on gfx900/906 targets.
    set(is_vega_check)
    set(VEGA_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_VEGA_ENABLED} ${PARSE_VEGA_DISABLED} ${VEGA_TEST_DEFAULT} is_vega_check)
    bool_and_f(${MIOPEN_TEST_VEGA} ${is_vega_check} is_vega_check)

    set(is_gfx908_check)
    set(GFX908_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_GFX908_ENABLED} ${PARSE_GFX908_DISABLED} ${GFX908_TEST_DEFAULT} is_gfx908_check)
    bool_and_f(${MIOPEN_TEST_GFX908} ${is_gfx908_check} is_gfx908_check)

    set(is_gfx90a_check)
    set(GFX90A_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_GFX90A_ENABLED} ${PARSE_GFX90A_DISABLED} ${GFX90A_TEST_DEFAULT} is_gfx90a_check)
    bool_and_f(${MIOPEN_TEST_GFX90a} ${is_gfx90a_check} is_gfx90a_check)

    set(is_gfx1030_check)
    set(GFX1030_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_GFX1030_ENABLED} ${PARSE_GFX1030_DISABLED} ${GFX1030_TEST_DEFAULT} is_gfx1030_check)
    bool_and_f(${MIOPEN_TEST_GFX1030} ${is_gfx1030_check} is_gfx1030_check)

    set(is_full_check)
    bool_not_f(${PARSE_SKIP_UNLESS_ALL} is_full_check)
    bool_or_f(${is_full_check} ${MIOPEN_TEST_ALL} is_full_check)


    add_custom_target(${NAME} ${PARSE_UNPARSED_ARGUMENTS})
    add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${NAME})
    if(WORKAROUND_ISSUE_898 AND MIOPEN_USE_COMGR)
        set_property(TEST ${NAME} PROPERTY ENVIRONMENT MIOPEN_DEBUG_COMGR_HIP_PCH_ENFORCE=0)
    endif()
    if(  (is_vega_check OR is_gfx908_check OR is_gfx1030_check)
     AND is_full_check
     AND (is_miotensile_check AND is_mlir_check)
     AND ( is_half_check OR is_bfloat16_check OR is_int8_check OR is_float_check)
     AND (is_ocl_check AND is_hip_check AND is_hip_nogpu_check)
    )
        if(PARSE_TEST_PERF_DB_RECORD_NOT_FOUND)
            set_tests_properties(${NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "(FAILED)|(Perf Db: record not found)")
        else()
            set_tests_properties(${NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
        endif()
        rate_added_test(${NAME})
    else()
        set_tests_properties(${NAME} PROPERTIES DISABLED On)
    endif()
endfunction()

if(${CODECOV_TEST})
    add_custom_test(test_conv3d_codecov
        COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 4 4 4 4 --weights 2 4 1 1 1 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    )
    add_custom_test(test_immed_conv2d_codecov
        COMMAND $<TARGET_FILE:test_immed_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input  2 2 14 14 --weights 8 2 3 3 --pads_strides_dilations 0 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    )
    add_custom_test(test_immed_conv3d_codecov
        COMMAND $<TARGET_FILE:test_immed_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 4 4 4 4 --weights 2 4 3 3 3 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    )
    add_custom_test(test_pooling2d_codecov
        COMMAND $<TARGET_FILE:test_pooling2d> ${MIOPEN_TEST_FLOAT_ARG} --input 1, 192, 28, 28 --lens 2 2 --strides 2 2 --pads 0 0 ${MIOPEN_TEST_FLAGS_ARGS}
    )

endif()


set(IMPLICITGEMM_ARGS ${MIOPEN_TEST_FLOAT_ARG})

# ./bin/MIOpenDriver conv -n 128 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
# MIOPEN_DEBUG_CONV_IMMED_FALLBACK=0
if(MIOPEN_EMBED_DB)
    set(MIOPEN_EMBED_TEST_ARG ${MIOPEN_TEST_FLOAT_ARG} --disable-validation --verbose)
    # WORKAROUND for issue #874
    set(MIOPEN_WA_ISSUE_874_F  MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0)
    set(MIOPEN_WA_ISSUE_874_W  MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0)
    set(MIOPEN_WA_ISSUE_874_FW MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0)
    # WORKAROUND for issue #1008
    set(MIOPEN_WA_ISSUE_1008 MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0)
add_custom_test(test_conv_embed_db TEST_PERF_DB_RECORD_NOT_FOUND
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_F}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008}                           $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 128 28 28 --weights 128 128 3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 128 28 28 --weights 512 128 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 2048 7 7 --weights 512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_F}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 14 14 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 14 14 --weights 256 256 3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 128 256 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008}                           $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008}                           $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 3 230 230   --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 1024 512 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 128 512 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 7 7   --weights 2048 512 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 512 7 7   --weights 512 512 3 3  --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 256 64 1 1  --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_FW} $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 64 64 1 1  --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${MIOPEN_WA_ISSUE_1008} ${MIOPEN_WA_ISSUE_874_W}  $<TARGET_FILE:test_conv2d> ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 64 64 3 3   --pads_strides_dilations 1 1 1 1 1 1
)
endif()

if(MIOPEN_TEST_MLIR)
    set(IMPLICITGEMM_MLIR_ENV_BASE MIOPEN_FIND_MODE=normal)
    set(IMPLICITGEMM_MLIR_ENV_F ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwd)
    set(IMPLICITGEMM_MLIR_ENV_B ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwd)
    set(IMPLICITGEMM_MLIR_ENV_W ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrW)

    set(IMPLICITGEMM_MLIR_ARGS_F ${IMPLICITGEMM_ARGS} --verbose --disable-backward-data --disable-backward-weights)
    set(IMPLICITGEMM_MLIR_ARGS_B ${IMPLICITGEMM_ARGS} --verbose --disable-forward --disable-backward-weights)
    set(IMPLICITGEMM_MLIR_ARGS_W ${IMPLICITGEMM_ARGS} --verbose --disable-forward --disable-backward-data)

    add_custom_test(test_conv_igemm_mlir  HALF_ENABLED MLIR_ENABLED GFX1030_ENABLED
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4

        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4

        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
        COMMAND ${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 256 1024 14 14 --weights 1024 32   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 32
    )

    set(IMPLICITGEMM_MLIR_ENV_F_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwdXdlops)
    set(IMPLICITGEMM_MLIR_ENV_B_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwdXdlops)
    set(IMPLICITGEMM_MLIR_ENV_W_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrWXdlops)

    # WORKAROUND: test_conv_igemm_mlir_xdlops target should run on gfx908 target only
    # However, since it is a smoke target with MIOPEN_TEST_FLOAT, !MIOPEN_TEST_TENSILE, !SKIP_UNLESS_ALL flags
    # It will be picked up by a non-gfx908 target. Therefore the need to add this if condition guard.
    if(MIOPEN_TEST_GFX908)
        add_custom_test(test_conv_igemm_mlir_xdlops HALF_ENABLED GFX908_ENABLED VEGA_DISABLED MLIR_ENABLED
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_F} --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4

            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_B} --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4

            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
            COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_MLIR_ARGS_W} --input 256 1024 14 14 --weights 1024 32   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 32
        )
    endif()
endif()

set(IMPLICITGEMM_TESTING_ENV
 MIOPEN_DEBUG_CONV_WINOGRAD=0
 MIOPEN_DEBUG_CONV_FFT=0
 MIOPEN_DEBUG_CONV_DIRECT=0
 MIOPEN_DEBUG_CONV_GEMM=0
 MIOPEN_DEBUG_CONV_SCGEMM=0
 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
)

if(WORKAROUND_ISSUE_936 AND MIOPEN_TEST_HALF)
    LIST(APPEND IMPLICITGEMM_TESTING_ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 MIOPEN_FIND_MODE=normal)
    LIST(APPEND IMPLICITGEMM_ARGS --disable-forward --disable-backward-data)
    #Afther fix need to remove '| grep -v "cannot be executed due to incorrect params"'
endif()
add_custom_test(test_conv_for_implicit_gemm SKIP_UNLESS_ALL BF16_ENABLED HALF_ENABLED GFX1030_ENABLED
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  16  28  28  --weights 192 16  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  16  14  14  --weights 160 16  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  16   7   7  --weights 128 16  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  16  55  55  --weights 96  16  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  16  28  28  --weights 64  16  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  16  14  14  --weights 32  16  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  32  28  28  --weights 192 32  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  32  14  14  --weights 160 32  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  32  7   7   --weights 128 32  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  32  55  55  --weights 96  32  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  32  28  28  --weights 64  32  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  32  14  14  --weights 32  32  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  64  56  56  --weights 256 64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  64  56  56  --weights 64  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  64  73  73  --weights 80  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  64  56  56  --weights 64  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  128 55  55  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  128 28  28  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  128 14  14  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64  128  7   7  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64 56  56  --weights 256  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64 56  56  --weights 64   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64 73  73  --weights 80   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16   64 56  56  --weights 64   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128 55  55  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128 28  28  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "cannot be executed due to incorrect params"
# COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128     14  14  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  128      7   7  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1 | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	55	55	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	28	28	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	14	14	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	 7	 7	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	128	    28	28	--weights	512	128	    1	1	--pads_strides_dilations	0	0	1	1	1	1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	160	    73	73	--weights	64	160	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	    35	35	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	    35	35	--weights	48	192	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	    35	35	--weights	64	192	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	16	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	64	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	192	28	28	--weights	96	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    35	35	--weights	48	256	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    35	35	--weights	64	256	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    56	56	--weights	128	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    56	56	--weights	512	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	    56	56	--weights	64	256	    1	1	--pads_strides_dilations	0	0	1	1	1	1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	28	28	--weights	128	256	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	28	28	--weights	32	256	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	256	28	28	--weights	64	256	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	288	    35	35	--weights	48	288	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	288	    35	35	--weights	64	288	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	384	    35	35	--weights	192	384	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	384	    35	35	--weights	64	384	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	384	    35	35	--weights	96	384	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	16	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	192	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	64	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	480	14	14	--weights	96	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	    28	28	--weights	128	512	    1	1	--pads_strides_dilations	0	0	1	1	1	1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	    28	28	--weights	256	512	    1	1	--pads_strides_dilations	0	0	2	2	1	1 | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	112	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	128	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	144	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	160	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	24	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	32	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose	--input	64	512	14	14	--weights	64	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   32  832  1   1   --pads_strides_dilations    0   0   1   1   1   1      | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   192  832  1   1   --pads_strides_dilations    0   0   1   1   1   1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   128  832  1   1   --pads_strides_dilations    0   0   1   1   1   1     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   32  832  1   1   --pads_strides_dilations    0   0   1   1   2   2      | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   192  832  1   1   --pads_strides_dilations    0   0   1   1   2   2     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 128  832    7  7  --weights   128  832  1   1   --pads_strides_dilations    0   0   1   1   2   2     | grep -v "cannot be executed due to incorrect params"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 16  2048    7  7  --weights   192  2048 1   1   --pads_strides_dilations    0   0   1   1   2   2     | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	28 28 --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1         | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 8    16 14 14 --weights   32   16   1   1   --pads_strides_dilations	1   1	1   1	1   1         | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	14 14 --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1         | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	7 7   --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1         | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	28 28 --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1         | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	14 14 --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1         | grep -v "cannot be executed due to incorrect params"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${IMPLICITGEMM_ARGS} --verbose   --input 64	 32	7 7   --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1         | grep -v "cannot be executed due to incorrect params"
)

add_custom_test(test_conv_group SKIP_UNLESS_ALL  MIOTENSILE_ENABLED GFX1030_ENABLED
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
    add_custom_test(test_deepbench_rnn  MIOTENSILE_ENABLED GFX1030_ENABLED
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


add_custom_test(test_rnn_extra SKIP_UNLESS_ALL  MIOTENSILE_ENABLED GFX1030_ENABLED
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

add_custom_test(test_gru_extra SKIP_UNLESS_ALL  MIOTENSILE_ENABLED GFX1030_ENABLED
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

add_custom_test(test_lstm_extra SKIP_UNLESS_ALL  MIOTENSILE_ENABLED GFX1030_ENABLED
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


add_custom_test(test_conv_extra SKIP_UNLESS_ALL  MIOTENSILE_ENABLED GFX1030_ENABLED
# COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	1	1	1	1	--weights	1	1	2	2	--pads_strides_dilations	0	0	3	3	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	3	227	227	--weights	4	3	11	11	--pads_strides_dilations	0	0	4	4	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	3	224	224	--weights	4	3	11	11	--pads_strides_dilations	2	2	4	4	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1
# Forward disabled since FFT fails verification for the forward direction
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	64	27	27	--weights	192	64	5	5	--pads_strides_dilations	2	2	1	1	1	1 --disable-forward
# COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	64	14	14	--weights	24	64	5	5	--pads_strides_dilations	2	2	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	96	14	14	--weights	32	96	5	5	--pads_strides_dilations	2	2	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	16	14	14	--weights	4	16	5	5	--pads_strides_dilations	2	2	1	1	1	1
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	14	14	--weights	4	32	5	5	--pads_strides_dilations	2	2	1	1	1	1

COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 16 3 64 128 --weights 96 3 11 11 --pads_strides_dilations 0 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 16 3 32 32 --weights 96 3 11 11 --pads_strides_dilations 0 0 2 2 1 1  ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 16 3 64 128 --weights 96 3 11 11 --pads_strides_dilations 5 5 2 2 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 16 3 32 32 --weights 96 3 11 11 --pads_strides_dilations 5 5 2 2 1 1  ${MIOPEN_TEST_FLAGS_ARGS}

COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 1024 2048 --weights 32 16 3 3 --pads_strides_dilations 0 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 1024 2048 --weights 32 16 3 3 --pads_strides_dilations 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 3072 3072 --weights 32 16 3 3 --pads_strides_dilations 0 0 2 2 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 3072 3072 --weights 32 16 3 3 --pads_strides_dilations 2 2 2 2 1 1 ${MIOPEN_TEST_FLAGS_ARGS}

COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 128 320 1 7 --weights 256 320 1 1 --pads_strides_dilations 0 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 128 1024 1 7 --weights 2048 1024 1 1 --pads_strides_dilations 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 352 192 7 1 --weights 320 192 1 1 --pads_strides_dilations 0 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --input 352 16 7 1 --weights 32 16 1 1 --pads_strides_dilations 2 2 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

if (0) #disabled too many errors
if(MIOPEN_TEST_LIMIT GREATER 0)
if(${MIOPEN_USE_MIOPENGEMM} AND (${MIOPEN_hip_VERSION_MAJOR} EQUAL 3) AND (${MIOPEN_hip_VERSION_MINOR} EQUAL 7))
    add_custom_test(test_conv3d_extra  SKIP_UNLESS_ALL GFX1030_ENABLED
    COMMAND MIOPEN_LOG_LEVEL=6 $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND MIOPEN_LOG_LEVEL=6 $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 2 2 2 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND MIOPEN_LOG_LEVEL=6 $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 2 2 2 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND MIOPEN_LOG_LEVEL=6 $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 2 2 2 ${MIOPEN_TEST_FLAGS_ARGS}
    #COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 1 1 1 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}

    #ROCM3.7 compiler problems
    #COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    #COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 2 2 2 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    #COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 140 602 --weights 16 16 3 11 11 --pads_strides_dilations 1 1 1 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    #COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 140 602 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    )
    message(STATUS "test_conv3d_extra reduced set")
else()
    add_custom_test(test_conv3d_extra  SKIP_UNLESS_ALL GFX1030_ENABLED
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50  50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 2 2 2 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 2 2 2 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 2 2 2 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 1 1 1 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 2 2 2 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 140 602 --weights 16 16 3 11 11 --pads_strides_dilations 1 1 1 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    COMMAND $<TARGET_FILE:test_conv3d> ${MIOPEN_TEST_FLOAT_ARG} --input 1 16 4 140 602 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
    )
endif()
endif()
endif()


add_custom_test(test_conv_trans SKIP_UNLESS_ALL  MIOTENSILE_ENABLED GFX1030_ENABLED
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


add_custom_test(test_conv_3d SKIP_UNLESS_ALL  MIOTENSILE_ENABLED GFX1030_ENABLED
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
    MIOPEN_FIND_MODE=normal)
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

set(DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC)

set(DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC)

if(${CODECOV_TEST})
    add_custom_test(test_conv_igemm_dynamic_small GFX908_DISABLED
    COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  32  32 17 17 --weights 32  32 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights --disable-validation
    COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data --disable-validation
    COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights --disable-validation
    )
    set_tests_properties(test_conv_igemm_dynamic_small PROPERTIES COST 800)
else()
    add_custom_test(test_conv_igemm_dynamic_small GFX908_DISABLED
    COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  16  16 56 56 --weights 64  16 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
    COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  16  64 34 34 --weights 64  64 3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
    COMMAND ${DYNAMIC_IMPLICITGEMM_ENVS}     $<TARGET_FILE:test_conv2d> --verbose --input  32  32 17 17 --weights 32  32 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights
    COMMAND ${DYNAMIC_IMPLICITGEMM_1X1_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  16 384  8  8 --weights 64 384 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
    COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
    COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
    COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
    COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
    )
endif() #if CODECOV_TEST

add_custom_test(test_conv_igemm_dynamic SKIP_UNLESS_ALL GFX908_DISABLED
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
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64  512 14 14 --weights 256 512 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input  32  128 34 34 --weights 64  128  3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  128 35 35 --weights 128  128 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS} $<TARGET_FILE:test_conv2d> --verbose --input 128  256 56 56 --weights 64  256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic_xdlops_bwd SKIP_UNLESS_ALL HALF_ENABLED GFX908_ENABLED VEGA_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  8  16 5 5 --weights 8  16  2 2 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic_xdlops_bwd_group SKIP_UNLESS_ALL HALF_ENABLED FLOAT_DISABLED GFX908_ENABLED VEGA_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  32 28 28 --weights 16  16 1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 2 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 17 17 --weights 64  16 1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4 --disable-forward --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  8  128 56 56 --weights 128 16 3 3 --pads_strides_dilations 1 1 1 1 1 1 --group-count 8 --disable-forward --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic_xdlops_bwd_float SKIP_UNLESS_ALL HALF_DISABLED FLOAT_ENABLED GFX908_ENABLED VEGA_DISABLED 
COMMAND ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  4  512 128 128 --weights 12  512  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic_xdlops_fwd SKIP_UNLESS_ALL HALF_ENABLED GFX908_ENABLED VEGA_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 64 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 64 2048 7 7 --weights 2048 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 128 17 17 --weights 128 128 7 1 --pads_strides_dilations 3 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 128 17 17 --weights 128 128 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 192 17 17 --weights 320 192 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 256 35 35 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 48 35 35 --weights 64 48 5 5 --pads_strides_dilations 2 2 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 64 512 7 7 --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 32 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 2 256 100 104 --weights 12 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 1 256 28 28 --weights 80 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic_xdlops_fwd_half SKIP_UNLESS_ALL HALF_ENABLED FLOAT_DISABLED GFX908_ENABLED VEGA_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 64 3 224 224 --weights 64 3 7 7 --pads_strides_dilations 3 3 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${DYNAMIC_IMPLICITGEMM_FWD_GTC_DYNAMIC_XDLOPS_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 64 3 230 230 --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
)

add_custom_test(test_conv_igemm_dynamic_xdlops_wrw SKIP_UNLESS_ALL GFX908_ENABLED VEGA_DISABLED HALF_ENABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  4  512 128 128 --weights 12  512  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
#regression test for issue 540
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward --disable-backward-data

COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-data
# Regression test for SWDEV-295434 (FP16 only).
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  120  256 3 3 --weights 340  256  3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-data
)

add_custom_test(test_conv_igemm_dynamic_xdlops_wrw_half SKIP_UNLESS_ALL VEGA_DISABLED HALF_ENABLED FLOAT_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 3 32 32 --weights 1 3 11 11 --pads_strides_dilations 1 1 2 2 2 1 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 3 224 224 --weights 1 3 3 3 --pads_strides_dilations 0 0 1 1 2 2 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 1 8 8 --weights 1 1 2 2 --pads_strides_dilations 0 0 1 1 2 2 --disable-forward --disable-backward-data
COMMAND ${DYNAMIC_IMPLICITGEMM_WRW_ENVS_XDLOPS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 128 56 56 --weights 1 128 5 5 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward --disable-backward-data
)

add_custom_test(test_conv_igemm_dynamic_xdlops_nhwc_fwd SKIP_UNLESS_ALL HALF_ENABLED GFX908_ENABLED GFX90A_ENABLED VEGA_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   2  64 59 57 --weights  12  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64   3 78 78 --weights  64   3 7 7 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16 192 17 17 --weights 224 192 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_FWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16   3 17 17 --weights  64   3 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
)
add_custom_test(test_conv_igemm_dynamic_xdlops_nhwc_bwd SKIP_UNLESS_ALL HALF_ENABLED GFX908_ENABLED GFX90A_ENABLED VEGA_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  15 256 1  1  --weights 340 256 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_BWD_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  15 128 10 10 --weights 340 128 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward --disable-backward-weights --in_layout NHWC --fil_layout NHWC --out_layout NHWC
)

set(DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC)

set(ARGS_NHWC_WRW
    --disable-forward
    --disable-backward-data
    --in_layout NHWC
    --fil_layout NHWC
    --out_layout NHWC)

add_custom_test(test_conv_igemm_dynamic_xdlops_nhwc_wrw SKIP_UNLESS_ALL HALF_ENABLED GFX908_ENABLED GFX90A_ENABLED VEGA_DISABLED
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}

COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward ${ARGS_NHWC_WRW}

COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-forward ${ARGS_NHWC_WRW}

COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 3 32 32 --weights 1 3 11 11 --pads_strides_dilations 1 1 2 2 2 1 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 3 224 224 --weights 1 3 3 3 --pads_strides_dilations 0 0 1 1 2 2 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 1 8 8 --weights 1 1 2 2 --pads_strides_dilations 0 0 1 1 2 2 --disable-forward ${ARGS_NHWC_WRW}
COMMAND ${DYNAMIC_IMPLICITGEMM_XDLOPS_NHWC_WRW_ENVS} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  1 128 56 56 --weights 1 128 5 5 --pads_strides_dilations 0 0 2 2 1 1 --disable-forward ${ARGS_NHWC_WRW}

)
add_custom_test(test_regression_half_mi100 SKIP_UNLESS_ALL FLOAT_DISABLED HALF_ENABLED GFX908_ENABLED VEGA_DISABLED
# Regression test for SWDEV-291202
COMMAND MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmBwdDataV4R1Xdlops $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  128  24 14 14 --weights 64  24 5 5 --pads_strides_dilations 2 2 1 1 1 1 --disable-forward --disable-backward-weights
)

set(CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV
    MIOPEN_FIND_MODE=normal
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvCkIgemmFwdV6r1DlopsNchw)

add_custom_test(test_conv_ck_igemm_fwd_v6r1_dlops_nchw FLOAT_ENABLED HALF_ENABLED BF16_DISABLED VEGA_ENABLED GFX908_ENABLED GFX1030_ENABLED SKIP_UNLESS_ALL
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 1024 14 14  --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  256 14 14  --weights  256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 1024 14 14  --weights  512 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  128 28 28  --weights  128 1024 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  128 28 28  --weights  512  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  128 58 58  --weights  128  128 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128 2048  7  7  --weights  512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  256 14 14  --weights 1024  256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  256 14 14  --weights  256  256 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  256 30 30  --weights  256  256 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  256 56 56  --weights  128  256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  256 56 56  --weights  512  256 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  256 56 56  --weights   64  256 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  512 16 16  --weights  512  512 3 3 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  512 28 28  --weights 1024  512 1 1 --pads_strides_dilations 0 0 2 2 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  512 28 28  --weights  128  512 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  512 28 28  --weights  256  512 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  512  7  7  --weights 2048  512 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128  512  7  7  --weights  512  512 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128   64 56 56  --weights  256   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128   64 56 56  --weights   64   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 --disable-backward-data --disable-backward-weights
COMMAND ${CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV}     $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 128   64 56 56  --weights   64   64 3 3 --pads_strides_dilations 1 1 1 1 1 1 --disable-backward-data --disable-backward-weights
)

if(MIOPEN_TEST_DEEPBENCH)
    add_custom_test(test_deepbench_conv  MIOTENSILE_ENABLED GFX1030_ENABLED
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
    add_custom_test(test_miopen_conv  MIOTENSILE_ENABLED GFX1030_ENABLED
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

if(MIOPEN_TEST_FLOAT)
    add_custom_test(test_reduce_double SKIP_UNLESS_ALL GFX908_ENABLED GFX1030_ENABLED COMMAND  $<TARGET_FILE:test_reduce_test> --double --all --verbose)
endif()

# Add here regression tests that should be run on Vega10/20 and GFX908 only with FP16.
add_custom_test(test_regression_half_vega_gfx908 FLOAT_DISABLED HALF_ENABLED GFX908_ENABLED
# REGRESSION TEST for issue #894.
# Can't be enabled for GFX10 due to WORKAROUND_SWDEV_271887
COMMAND	MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvOclDirectFwd1x1 $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --disable-backward-data --disable-backward-weights --disable-verification-cache
    --cmode conv --pmode default --group-count 1 --input 1 16 7 7 --weights 16 16 1 1 --pads_strides_dilations 0 0 1 1 1 1
)

set(ENVS_REGRESSION_ISSUE_1012
    MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS=1
    MIOPEN_FIND_MODE=normal)

set(ARGS_REGRESSION_ISSUE_1012
    --verbose
    --disable-forward
    --disable-backward-data
    --disable-validation)

add_custom_test(test_regression_opencl_float_mi100 GFX908_ENABLED VEGA_DISABLED HIP_DISABLED
    # Issue #1012.
    COMMAND	${ENVS_REGRESSION_ISSUE_1012} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --cmode conv --pmode default --group-count 1 --input 128, 832, 7,  7  --weights 32,  832, 1, 1 --pads_strides_dilations 0 0 1 1 1 1 ${ARGS_REGRESSION_ISSUE_1012}
    COMMAND	${ENVS_REGRESSION_ISSUE_1012} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --cmode conv --pmode default --group-count 1 --input 64,  192, 28, 28 --weights 64,  192, 1, 1 --pads_strides_dilations 0 0 1 1 1 1 ${ARGS_REGRESSION_ISSUE_1012}
    COMMAND	${ENVS_REGRESSION_ISSUE_1012} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --cmode conv --pmode default --group-count 1 --input 64,  256, 28, 28 --weights 128, 256, 1, 1 --pads_strides_dilations 0 0 1 1 1 1 ${ARGS_REGRESSION_ISSUE_1012}
    COMMAND	${ENVS_REGRESSION_ISSUE_1012} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --cmode conv --pmode default --group-count 1 --input 64,  480, 14, 14 --weights 64,  480, 1, 1 --pads_strides_dilations 0 0 1 1 1 1 ${ARGS_REGRESSION_ISSUE_1012}
    COMMAND	${ENVS_REGRESSION_ISSUE_1012} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --cmode conv --pmode default --group-count 1 --input 64,  512, 14, 14 --weights 128, 512, 1, 1 --pads_strides_dilations 0 0 1 1 1 1 ${ARGS_REGRESSION_ISSUE_1012}
    COMMAND	${ENVS_REGRESSION_ISSUE_1012} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --cmode conv --pmode default --group-count 1 --input 64,  512, 28, 28 --weights 128, 512, 1, 1 --pads_strides_dilations 0 0 1 1 1 1 ${ARGS_REGRESSION_ISSUE_1012}
    COMMAND	${ENVS_REGRESSION_ISSUE_1012} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --cmode conv --pmode default --group-count 1 --input 64,  64,  56, 56 --weights 256, 64,  1, 1 --pads_strides_dilations 0 0 1 1 1 1 ${ARGS_REGRESSION_ISSUE_1012}
)
