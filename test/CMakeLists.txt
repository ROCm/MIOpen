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

set(test_perf.py ${CMAKE_INSTALL_BINDIR}/test_perf.py)

if( NOT ENABLE_ASAN_PACKAGING )
  install(FILES  test_perf.py
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
      DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

set(MODEL_FILES)
file (GLOB model_src CONFIGURE_DEPENDS perf_models/*.txt)
foreach (src ${model_src})
    list (APPEND MODEL_FILES ${src})
endforeach()

if( NOT ENABLE_ASAN_PACKAGING )
  install(FILES
      ${MODEL_FILES}
      DESTINATION ${DATA_INSTALL_DIR}/perf_models)
endif()
 foreach(MODEL_FILE ${MODEL_FILES})
    get_filename_component(MODEL_FILE_FILENAME "${MODEL_FILE}" NAME)
    configure_file("${MODEL_FILE}" "${PROJECT_BINARY_DIR}/${DATA_INSTALL_DIR}/perf_models/${MODEL_FILE_FILENAME}" COPYONLY)
 endforeach()

include(CheckCXXCompilerFlag)

option( MIOPEN_TEST_ALL "Run the full test suite" OFF )
option( MIOPEN_TEST_HALF "Test in half mode" OFF )
option( MIOPEN_TEST_INT8 "Test in int8 mode" OFF )
option( MIOPEN_TEST_BFLOAT16 "Test in bfloat16 mode" OFF )
option( MIOPEN_TEST_GFX908 "Test on MI100 (gfx908)" OFF )
option( MIOPEN_TEST_GFX90A "Test on MI2X0 (gfx90a)" OFF )
option( MIOPEN_TEST_GFX94X "Test on MI300 (gfx940/1/2)" OFF )
option( MIOPEN_TEST_GFX900 "Test on Vega10 (gfx900)" OFF )
option( MIOPEN_TEST_GFX906 "Test on Vega20 (gfx906)" OFF )
option( MIOPEN_TEST_GFX103X "Test on Navi21/22 (gfx1030/31)" OFF )
option( MIOPEN_TEST_GFX110X "Test on Navi31/32 (gfx1100/02)" OFF )
option( MIOPEN_TEST_GFX120X "Test on gfx1200/01" OFF )
option( MIOPEN_TEST_GPU_XNACK_ENABLED "Test as if XNACK mode is enabled" OFF )
option( MIOPEN_TEST_CONV "" OFF)
option( MIOPEN_TEST_DEEPBENCH "" OFF)
option( MIOPEN_TEST_DRIVER_ITER_MODE "" OFF)
option( MIOPEN_TEST_COMPOSABLEKERNEL "Test with composable_kernel library" ${MIOPEN_USE_COMPOSABLEKERNEL} )
option( MIOPEN_TEST_DISCRETE "Build Discrete Test Binaries" ON)
option( MIOPEN_TEST_DBSYNC "" OFF)
option( CODECOV_TEST "" OFF)

set_var_to_condition(MIOPEN_TEST_WITH_MIOPENDRIVER_DEFAULT MIOPEN_BUILD_DRIVER)
option( MIOPEN_TEST_WITH_MIOPENDRIVER "Use MIOpenDriver in tests" ${MIOPEN_TEST_WITH_MIOPENDRIVER_DEFAULT})

option( WORKAROUND_ISSUE_936 "" ON)

# Run the test suite to a depth limit
#limit greater than 2 leads to prolonged testing more than 5hrs per stage.
set(MIOPEN_TEST_LIMIT "2" CACHE STRING "")
set(MIOPEN_TEST_FLAGS "" CACHE STRING "")
set(MIOPEN_TEST_GDB On CACHE BOOL "")

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
if(NOT (MIOPEN_TEST_GFX900 OR MIOPEN_TEST_GFX906 OR MIOPEN_TEST_GFX908 OR MIOPEN_TEST_GFX90A OR MIOPEN_TEST_GFX94X OR MIOPEN_TEST_GFX103X OR MIOPEN_TEST_GFX110X OR MIOPEN_TEST_GFX120X OR MIOPEN_TEST_HIP_NOGPU))
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
            set(MIOPEN_TEST_GFX103X ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx1031")
            set(MIOPEN_TEST_GFX103X ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx1100")
            set(MIOPEN_TEST_GFX110X ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx1101")
            set(MIOPEN_TEST_GFX110X ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx1102")
            set(MIOPEN_TEST_GFX110X ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx1200")
            set(MIOPEN_TEST_GFX120X ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx1201")
            set(MIOPEN_TEST_GFX120X ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx900")
            set(MIOPEN_TEST_GFX900 ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx906")
            set(MIOPEN_TEST_GFX906 ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx908")
            set(MIOPEN_TEST_GFX908 ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx90a")
            set(MIOPEN_TEST_GFX90A ON)
        elseif(ROCMINFO_OUTPUT MATCHES "gfx94")
            set(MIOPEN_TEST_GFX94X ON)
        else()
            message(WARNING "TESTING IS NOT SUPPORTED FOR THE DETECTED GPU")
            set(MIOPEN_TEST_GPU_DETECTION_FAILED TRUE)
        endif()

        if(NOT (MIOPEN_TEST_GPU_DETECTION_FAILED) AND ROCMINFO_OUTPUT MATCHES "xnack\\+")
            set(MIOPEN_TEST_GPU_XNACK_ENABLED ON)
        endif()

        if(MIOPEN_TEST_GPU_DETECTION_FAILED)
            message(STATUS "${ROCMINFO_OUTPUT}")
        endif()
    else()
        message(WARNING "ROCMINFO NOT FOUND, GPU TYPE UNKNOWN. Manually set respective MIOPEN_TEST_GFX* CMake variable to specify target GPU for testing.")
        set(MIOPEN_TEST_GPU_DETECTION_FAILED TRUE)
    endif()
endif()

set(MIOPEN_LIBMLIR_SUPPORTS_GFX103X_DEFAULT FALSE)
if(MIOPEN_USE_MLIR)
    set_var_to_condition(MIOPEN_LIBMLIR_SUPPORTS_GFX103X_DEFAULT (${rocMLIR_VERSION} VERSION_GREATER_EQUAL 1))
endif()
option( MIOPEN_LIBMLIR_SUPPORTS_GFX103X "libMLIR capability to support GFX10. Enables MLIR tests on GFX103X." ${MIOPEN_LIBMLIR_SUPPORTS_GFX103X_DEFAULT})

set_var_to_condition(MIOPEN_TEST_MLIR_DEFAULT ${MIOPEN_USE_MLIR} AND NOT (${MIOPEN_TEST_GFX103X} AND NOT ${MIOPEN_LIBMLIR_SUPPORTS_GFX103X}))
option( MIOPEN_TEST_MLIR "Test for MLIR compilation backend" ${MIOPEN_TEST_MLIR_DEFAULT} )

message(STATUS "MIOPEN_NO_GPU ${MIOPEN_NO_GPU}")
message(STATUS "MIOPEN_TEST_GFX900 ${MIOPEN_TEST_GFX900}")
message(STATUS "MIOPEN_TEST_GFX906 ${MIOPEN_TEST_GFX906}")
message(STATUS "MIOPEN_TEST_GFX908 ${MIOPEN_TEST_GFX908}")
message(STATUS "MIOPEN_TEST_GFX90A ${MIOPEN_TEST_GFX90A}")
message(STATUS "MIOPEN_TEST_GFX94X ${MIOPEN_TEST_GFX94X}")
message(STATUS "MIOPEN_TEST_GFX103X ${MIOPEN_TEST_GFX103X}")
message(STATUS "MIOPEN_TEST_GFX110X ${MIOPEN_TEST_GFX110X}")
message(STATUS "MIOPEN_TEST_GFX120X ${MIOPEN_TEST_GFX120X}")
message(STATUS "MIOPEN_TEST_GPU_XNACK_ENABLED ${MIOPEN_TEST_GPU_XNACK_ENABLED}")
message(STATUS "MIOPEN_TEST_GPU_DETECTION_FAILED ${MIOPEN_TEST_GPU_DETECTION_FAILED}")
message(STATUS "MIOPEN_TEST_WITH_MIOPENDRIVER ${MIOPEN_TEST_WITH_MIOPENDRIVER}")
message(STATUS "MIOPEN_TEST_MLIR ${MIOPEN_TEST_MLIR}")
message(STATUS "MIOPEN_TEST_CONV ${MIOPEN_TEST_CONV}")
message(STATUS "MIOPEN_TEST_DEEPBENCH ${MIOPEN_TEST_DEEPBENCH}")
message(STATUS "MIOPEN_TEST_DRIVER_ITER_MODE ${MIOPEN_TEST_DRIVER_ITER_MODE}")
message(STATUS "MIOPEN_TEST_COMPOSABLEKERNEL ${MIOPEN_TEST_COMPOSABLEKERNEL}")
message(STATUS "MIOPEN_TEST_DISCRETE ${MIOPEN_TEST_DISCRETE}")
message(STATUS "MIOPEN_TEST_DBSYNC ${MIOPEN_TEST_DBSYNC}")
message(STATUS "CODECOV_TEST ${CODECOV_TEST}")

if(MIOPEN_TEST_DRIVER_ITER_MODE)
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=2)
else()
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=1)
endif()

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

if(MIOPEN_TEST_GPU_DETECTION_FAILED AND NOT (MIOPEN_NO_GPU))
    add_custom_target(gpu_detection_check COMMAND echo "*** FATAL: GPU DETECTION FAILED DURING CMAKE PHASE, CHECK CMAKE WARNINGS ***" COMMAND exit 1)
    add_dependencies(check gpu_detection_check)
endif()

if(MIOPEN_TEST_WITH_MIOPENDRIVER)
    add_dependencies(check MIOpenDriver)
endif()


set(MIOPEN_GTEST_SUFFIX)
set(MIOPEN_TEST_FLOAT_ARG)
set(MIOPEN_TEST_FLOAT FALSE)
set(MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT8)
set(MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT32)
set(MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_FLOAT)
if(MIOPEN_TEST_HALF)
    set(MIOPEN_GTEST_SUFFIX FP16)
    set(MIOPEN_TEST_FLOAT_ARG --half)
    set(MIOPENDRIVER_MODE_CONV convfp16)
    set(MIOPENDRIVER_MODE_BN bnormfp16)
    set(MIOPENDRIVER_MODE_POOL poolfp16)
    set(MIOPENDRIVER_MODE_GEMM gemmfp16)
elseif(MIOPEN_TEST_INT8)
    set(MIOPEN_GTEST_SUFFIX I8)
    set(MIOPEN_TEST_FLOAT_ARG --int8)
    set(MIOPENDRIVER_MODE_CONV convint8)
    set(MIOPENDRIVER_MODE_BN NOT_SUPPORTED)
    set(MIOPENDRIVER_MODE_POOL NOT_SUPPORTED)
    set(MIOPENDRIVER_MODE_GEMM NOT_SUPPORTED)
    set(MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT8 --output_type int8)
    set(MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT32 --output_type int32)
    set(MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_FLOAT --output_type float)
elseif(MIOPEN_TEST_BFLOAT16)
    set(MIOPEN_GTEST_SUFFIX BFP16)
    set(MIOPEN_TEST_FLOAT_ARG --bfloat16)
    set(MIOPENDRIVER_MODE_CONV convbfp16)
    set(MIOPENDRIVER_MODE_BN NOT_SUPPORTED)
    set(MIOPENDRIVER_MODE_POOL NOT_SUPPORTED)
    set(MIOPENDRIVER_MODE_GEMM NOT_SUPPORTED)
else()
    set(MIOPEN_GTEST_SUFFIX FP32)
    set(MIOPEN_TEST_FLOAT_ARG --float)
    set(MIOPEN_TEST_FLOAT TRUE)
    set(MIOPENDRIVER_MODE_CONV conv)
    set(MIOPENDRIVER_MODE_BN bnorm)
    set(MIOPENDRIVER_MODE_POOL pool)
    set(MIOPENDRIVER_MODE_GEMM gemm)
endif()

message(STATUS "MIOPEN_GTEST_SUFFIX ${MIOPEN_GTEST_SUFFIX}")
message(STATUS "MIOPEN_TEST_FLOAT ${MIOPEN_TEST_FLOAT}")
message(STATUS "MIOPEN_TEST_HALF ${MIOPEN_TEST_HALF}")
message(STATUS "MIOPEN_TEST_BFLOAT16 ${MIOPEN_TEST_BFLOAT16}")
message(STATUS "MIOPEN_TEST_INT8 ${MIOPEN_TEST_INT8}")
message(STATUS "MIOPEN_TEST_ALL ${MIOPEN_TEST_ALL}")

set_var_to_condition(WORKAROUND_ISSUE_1187_DEFAULT MIOPEN_TEST_GFX90A AND MIOPEN_TEST_FLOAT)
option( WORKAROUND_ISSUE_1187 "" ${WORKAROUND_ISSUE_1187_DEFAULT})

set_var_to_condition(WORKAROUND_ISSUE_1148_DEFAULT (MIOPEN_TEST_GFX103X OR MIOPEN_TEST_GFX110X) AND MIOPEN_TEST_FLOAT)
option( WORKAROUND_ISSUE_1148 "" ${WORKAROUND_ISSUE_1148_DEFAULT})

if(MIOPEN_TEST_INT8)
    set(SKIP_ALL_EXCEPT_TESTS
              test_tensor_vec test_tensor_cast test_tensor_trans test_tensor_copy test_tensor_set
              test_tensor_transform test_conv2d test_conv2d_find2)
elseif(MIOPEN_TEST_BFLOAT16)
    set(SKIP_ALL_EXCEPT_TESTS
              test_conv2d test_conv2d_find2 test_tensor_copy test_tensor_set test_tensor_vec test_immed_conv2d
              test_check_numerics_test test_conv_extra test_conv_for_implicit_gemm test_miopen_conv
              test_deepbench_conv test_conv_igemm_dynamic_xdlops_nhwc_wrw_bf16
              test_conv_igemm_dynamic_xdlops_nhwc_fwd_bf16
              test_conv_igemm_dynamic_xdlops_nhwc_bwd_bf16)
endif()

if(${CODECOV_TEST})
    list(APPEND SKIP_TESTS
        test_conv3d test_conv3d_find2 test_immed_conv3d test_immed_conv2d test_pooling2d test_pooling2d_asymmetric
        test_pooling2d_wide)
    # replaced by smaller tests with suffix _codecov
endif()

if(MIOPEN_NO_GPU)
    set(SKIP_ALL_EXCEPT_TESTS test_include_inliner test_kernel_build_params
            test_test_errors test_type_name test_tensor_test test_sqlite_perfdb test_sequences
            test_pooling3d test_perfdb)
endif()

#TODO WORKAROUND_ISSUE_1424
if(MIOPEN_TEST_GFX900 OR MIOPEN_TEST_GFX906 OR MIOPEN_TEST_GFX908)
    list(APPEND SKIP_TESTS test_bn_3d_spatial_test)
endif()

#Don't run db verification by default
if(NOT MIOPEN_TEST_DBSYNC)
    list(APPEND SKIP_TESTS db_sync)
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

# List of tests that depend on the XNACK mode.
# Options convention: Tests that depend on the XNACK mode should support the "--xnack" option.
# If "--xnack 0" is specified (this is the default), then such tests should run in XNACK OFF mode.
# If XNACK is enabled on the GPU, then the testing mode must be changed by the --xnack 1 option.
set(XNACK_TESTS test_mdgraph)

function(add_test_command NAME EXE)
    if( (NOT (NAME IN_LIST SKIP_ALL_EXCEPT_TESTS) AND SKIP_ALL_EXCEPT_TESTS)
        OR (NAME IN_LIST SKIP_TESTS)
    )
        add_test(NAME ${NAME} COMMAND echo skipped)
        set_tests_properties(${NAME} PROPERTIES DISABLED On)
    elseif(WIN32)
        add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${EXE}> ${ARGN} WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR}")
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
    set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

separate_arguments(MIOPEN_TEST_FLAGS_ARGS NATIVE_COMMAND ${MIOPEN_TEST_FLAGS})

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    if(WIN32)
        target_compile_definitions(${TEST_NAME} PRIVATE NOMINMAX)
    endif()
    clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} Threads::Threads)
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

    if(MIOPEN_TEST_GPU_XNACK_ENABLED AND (${TEST_NAME} IN_LIST XNACK_TESTS))
        set(TEST_COMMAND ${TEST_COMMAND} --xnack 1)
    endif()

    if(WORKAROUND_ISSUE_936 AND (${TEST_NAME} MATCHES "test_conv2d" OR ${TEST_NAME} MATCHES "test_immed_conv2d") )
        set(TEST_COMMAND ${TEST_COMMAND} --tolerance 130) #increased by 1.625 times
    endif()

    add_test_command(${TEST_NAME} ${TEST_COMMAND})
    rate_added_test(${TEST_NAME})
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
    if(WORKAROUND_ISSUE_1148
        AND (${TEST_NAME} MATCHES "test_soft_max") )
        set_tests_properties(${TEST_NAME} PROPERTIES RUN_SERIAL On)
    endif()
    if(NOT MIOPEN_EMBED_DB STREQUAL "")
        target_link_libraries(${TEST_NAME} MIOpen miopen_data)
    else()
        target_link_libraries(${TEST_NAME} MIOpen)
    endif()
    target_include_directories(${TEST_NAME} PRIVATE ../src/kernels)
    if(WIN32)
        # Refer to https://en.cppreference.com/w/cpp/language/types for details.
        target_compile_options(${TEST_NAME} PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
    endif()
    if(HAS_LIB_STD_FILESYSTEM)
        target_link_libraries(${TEST_NAME} stdc++fs)
    endif()
endfunction()

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

file(GLOB TEST_SOURCES *.cpp)

set(LONG_TESTS
    test_dropout
    test_conv2d
    test_conv2d_find2
    test_conv3d
    test_conv3d_find2
    test_conv_group
    test_soft_max
    test_lrn_test
    test_conv_for_implicit_gemm
    test_immed_conv3d
    test_conv3d_extra
    test_conv_3d
    test_pooling2d
    test_conv_ck_igemm_fwd_v6r1_dlops_nchw
    )

function(rate_added_test NAME)
    if(${NAME} IN_LIST LONG_TESTS)
        set_tests_properties(${NAME} PROPERTIES COST 800)
    else()
        set_tests_properties(${NAME} PROPERTIES COST 600)
    endif()
endfunction()

set(EXCLUDE_TESTS)
if(NOT MIOPEN_ENABLE_SQLITE)
    list(APPEND EXCLUDE_TESTS test_sqlite)
endif()
if(NOT MIOPEN_ENABLE_SQLITE OR (NOT MIOPEN_EMBED_DB STREQUAL ""))
    list(APPEND EXCLUDE_TESTS test_sqlite_perfdb)
endif()

set(EXCLUDE_TEST_SOURCES)
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(BASE_NAME ${TEST_SOURCE} NAME_WE)
    if(test_${BASE_NAME} IN_LIST EXCLUDE_TESTS)
        list(APPEND EXCLUDE_TEST_SOURCES ${TEST_SOURCE})
    endif()
endforeach()
unset(EXCLUDE_TESTS)

foreach(TEST_SOURCE ${EXCLUDE_TEST_SOURCES})
    list(REMOVE_ITEM TEST_SOURCES ${TEST_SOURCE})
endforeach()
unset(EXCLUDE_TEST_SOURCES)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(BASE_NAME ${TEST_SOURCE} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST_SOURCE})
endforeach()

if(MIOPEN_ENABLE_SQLITE AND (MIOPEN_EMBED_DB STREQUAL ""))
    set_tests_properties(test_sqlite_perfdb PROPERTIES RUN_SERIAL On)
endif()
set_tests_properties(test_perfdb PROPERTIES RUN_SERIAL On)

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

# The add_custom_test function contains attributes to describe the conditions,
# under which new custom_tests should be run. Attributes are divided into several types.
# The attribute can be enabled or disabled; if nothing is specified, the default value is taken.
# You can use any number of attributes, in any order, provided that attributes do not conflict
# (e.g. "HALF_ENABLED HALF_DISABLED" is illegal)
#
# Data types: FLOAT HALF BF16 INT8
#   These attributes can be enabled or disabled by using '_ENABLED' and '_DISABLED' suffix.
#   Defaults: FLOAT_ENABLED HALF_DISABLED BF16_DISABLED INT8_DISABLED
#
# GPU types: GFX900, GFX906, GFX908, GFX90A, GFX94X, GFX1030/31, GFX1100/02
#   These attributes can be enabled or disabled by using '_ENABLED' and '_DISABLED' suffix.
#   Defaults: GFX900_ENABLED, GFX906_ENABLED, GFX908_ENABLED, GFX90A_ENABLED, GFX94X_DISABLED, GFX103X_DISABLED, GFX110X_DISABLED
#
# Testing mode:
#   SKIP_UNLESS_ALL - The test should be only run if MIOPEN_TEST_ALL=TRUE. Intended for long tests.
#   TEST_PERF_DB_RECORD_NOT_FOUND - Test should fail if output contains: "Perf Db: record not found".
#   TEST_TUNING - In addition to the standard checks, the test should fail if output contains "Error" or "failed".
#   SKIP_XNACK_ON - Do not run the test if XNACK mode is enabled (xnack+) on the GPU.
#   SKIP_UNLESS_MLIR - The test should be only run if MIOPEN_TEST_MLIR=TRUE.
#   SKIP_UNLESS_COMPOSABLEKERNEL - The test should be only run if MIOPEN_TEST_COMPOSABLEKERNEL=TRUE.
#
# Backend: HIP HIP_NOGPU
#   These attributes can be enabled or disabled by using '_ENABLED' and '_DISABLED' suffix.
#   Default: HIP_ENABLED HIP_NOGPU_DISABLED.

function(add_custom_test NAME)
    set(options
        BF16_ENABLED BF16_DISABLED HALF_ENABLED HALF_DISABLED INT8_ENABLED INT8_DISABLED FLOAT_ENABLED FLOAT_DISABLED
        GFX900_ENABLED GFX900_DISABLED GFX906_ENABLED GFX906_DISABLED GFX908_ENABLED GFX908_DISABLED
        GFX103X_ENABLED GFX103X_DISABLED GFX110X_ENABLED GFX110X_DISABLED GFX90A_ENABLED GFX90A_DISABLED GFX94X_ENABLED GFX94X_DISABLED
        SKIP_UNLESS_MLIR SKIP_UNLESS_COMPOSABLEKERNEL SKIP_UNLESS_ALL TEST_PERF_DB_RECORD_NOT_FOUND TEST_TUNING SKIP_XNACK_ON
        HIP_ENABLED HIP_DISABLED HIP_NOGPU_ENABLED HIP_NOGPU_DISABLED
    )
    set(oneValueArgs)
    set(multiValueArgs ENVIRONMENT)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Many custom tests do test only FP32 data type and therefore
    # added only if none of MIOPEN_TEST_HALF, MIOPEN_TEST_INT8, MIOPEN_TEST_BFLOAT16
    # are set, except the test is allowed explicitly.
    set(HALF_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_HALF_ENABLED} ${PARSE_HALF_DISABLED} ${HALF_TEST_DEFAULT} is_half_check)
    bool_and_f(${MIOPEN_TEST_HALF} ${is_half_check} is_half_check)

    set(BF16_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_BF16_ENABLED} ${PARSE_BF16_DISABLED} ${BF16_TEST_DEFAULT} is_bfloat16_check)
    bool_and_f(${MIOPEN_TEST_BFLOAT16} ${is_bfloat16_check} is_bfloat16_check)

    set(INT8_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_INT8_ENABLED} ${PARSE_INT8_DISABLED} ${INT8_TEST_DEFAULT} is_int8_check)
    bool_and_f(${MIOPEN_TEST_INT8} ${is_int8_check} is_int8_check)

    set(FLOAT_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_FLOAT_ENABLED} ${PARSE_FLOAT_DISABLED} ${FLOAT_TEST_DEFAULT} is_float_check)
    bool_and_f(${MIOPEN_TEST_FLOAT} ${is_float_check} is_float_check)

    bool_not_f(${PARSE_SKIP_UNLESS_MLIR} is_mlir_check)
    bool_or_f(${is_mlir_check} ${MIOPEN_TEST_MLIR} is_mlir_check)

    bool_not_f(${PARSE_SKIP_UNLESS_COMPOSABLEKERNEL} is_composablekernel_check)
    bool_or_f(${is_composablekernel_check} ${MIOPEN_TEST_COMPOSABLEKERNEL} is_composablekernel_check)

    set(HIP_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_HIP_ENABLED} ${PARSE_HIP_DISABLED} ${HIP_TEST_DEFAULT} is_hip_check)
    bool_not_f(${MIOPEN_TEST_HIP} NOT_MIOPEN_TEST_HIP)
    bool_or_f(${NOT_MIOPEN_TEST_HIP} ${is_hip_check} is_hip_check)

    set(HIP_NOGPU_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_HIP_NOGPU_ENABLED} ${PARSE_HIP_NOGPU_DISABLED} ${HIP_NOGPU_TEST_DEFAULT} is_hip_nogpu_check)
    bool_not_f(${MIOPEN_TEST_HIP_NOGPU} NOT_MIOPEN_TEST_HIP_NOGPU)
    bool_or_f(${NOT_MIOPEN_TEST_HIP_NOGPU} ${is_hip_nogpu_check} is_hip_nogpu_check)

    # Some tests are xDLOPs specific and should not run on gfx900/906 targets.
    set(GFX900_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_GFX900_ENABLED} ${PARSE_GFX900_DISABLED} ${GFX900_TEST_DEFAULT} is_gfx900_check)
    bool_and_f(${MIOPEN_TEST_GFX900} ${is_gfx900_check} is_gfx900_check)

    set(GFX906_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_GFX906_ENABLED} ${PARSE_GFX906_DISABLED} ${GFX906_TEST_DEFAULT} is_gfx906_check)
    bool_and_f(${MIOPEN_TEST_GFX906} ${is_gfx906_check} is_gfx906_check)

    set(GFX908_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_GFX908_ENABLED} ${PARSE_GFX908_DISABLED} ${GFX908_TEST_DEFAULT} is_gfx908_check)
    bool_and_f(${MIOPEN_TEST_GFX908} ${is_gfx908_check} is_gfx908_check)

    set(GFX90A_TEST_DEFAULT TRUE)
    option_support_check(${PARSE_GFX90A_ENABLED} ${PARSE_GFX90A_DISABLED} ${GFX90A_TEST_DEFAULT} is_gfx90a_check)
    bool_and_f(${MIOPEN_TEST_GFX90A} ${is_gfx90a_check} is_gfx90a_check)

    set(GFX94X_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_GFX94X_ENABLED} ${PARSE_GFX94X_DISABLED} ${GFX94X_TEST_DEFAULT} is_gfx94x_check)
    bool_and_f(${MIOPEN_TEST_GFX94X} ${is_gfx94x_check} is_gfx94x_check)

    set(GFX103X_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_GFX103X_ENABLED} ${PARSE_GFX103X_DISABLED} ${GFX103X_TEST_DEFAULT} is_gfx103x_check)
    bool_and_f(${MIOPEN_TEST_GFX103X} ${is_gfx103x_check} is_gfx103x_check)

    set(GFX110X_TEST_DEFAULT FALSE)
    option_support_check(${PARSE_GFX110X_ENABLED} ${PARSE_GFX110X_DISABLED} ${GFX110X_TEST_DEFAULT} is_gfx110x_check)
    bool_and_f(${MIOPEN_TEST_GFX110X} ${is_gfx110x_check} is_gfx110x_check)

    # When SKIP_XNACK_ON is set, the test will be skipped if MIOPEN_TEST_GPU_XNACK_ENABLED is set.
    bool_and_f(${PARSE_SKIP_XNACK_ON} ${MIOPEN_TEST_GPU_XNACK_ENABLED} is_xnack_on_check)
    bool_not_f(${is_xnack_on_check} is_xnack_on_check)

    bool_not_f(${PARSE_SKIP_UNLESS_ALL} is_full_check)
    bool_or_f(${is_full_check} ${MIOPEN_TEST_ALL} is_full_check)

    add_custom_target(${NAME} ${PARSE_UNPARSED_ARGUMENTS})
    add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${NAME})
    foreach(__environment ${PARSE_ENVIRONMENT})
        set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "${__environment}")
    endforeach()

    # The tests often change the contents of the user databases, which may affect performance after testing.
    # For example, MIOPEN_DEBUG_FIND_ONLY_SOLVER results in non-optimal records in user-find-db.
    # Another example is tuning tests. These use MIOPEN_DEBUG_TUNING_ITERATIONS_MAX to save testing time,
    # but the side effect of such savings is sub-optimal tuning values in user-perf-db.
    #
    # MIOPEN_USER_DB_PATH setting resolves this potential problem. The user databases are written in
    # the non-default directory. This preserves the state of the actual user databases
    # and prevents performance degradation after the tests complete.
    #
    set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR}")

    if(WORKAROUND_ISSUE_1148
        AND (${NAME} MATCHES "test_conv_3d"
          OR ${NAME} MATCHES "test_conv_group"
          OR ${NAME} MATCHES "test_conv_extra"
          OR ${NAME} MATCHES "test_conv_for_implicit_gemm"
          OR ${NAME} MATCHES "test_conv_ck_igemm_fwd_v6r1_dlops_nchw"))
        set_tests_properties(${NAME} PROPERTIES RUN_SERIAL On)
    endif()

    if(  (is_gfx900_check OR is_gfx906_check OR is_gfx908_check OR is_gfx103x_check OR is_gfx110x_check OR is_gfx90a_check OR is_gfx94x_check)
     AND is_full_check
     AND is_xnack_on_check
     AND is_mlir_check
     AND is_composablekernel_check
     AND (is_half_check OR is_bfloat16_check OR is_int8_check OR is_float_check)
     AND (is_hip_check AND is_hip_nogpu_check)
    )
        if(PARSE_TEST_PERF_DB_RECORD_NOT_FOUND AND PARSE_TEST_TUNING)
            message(FATAL_ERROR " TEST_PERF_DB_RECORD_NOT_FOUND and TEST_TUNING should not be used together")
        endif()

        if(PARSE_TEST_PERF_DB_RECORD_NOT_FOUND)
            set_tests_properties(${NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "(FAILED)|(Perf Db: record not found)")
        elseif(PARSE_TEST_TUNING)
            set_tests_properties(${NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "(FAILED)|(Error)|(failed)")
        else()
            set_tests_properties(${NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
        endif()

        rate_added_test(${NAME})
    else()
        set_tests_properties(${NAME} PROPERTIES DISABLED On)
    endif()

    if(WORKAROUND_ISSUE_1187
        AND (${NAME} MATCHES "test_conv_for_implicit_gemm" ))
        set_tests_properties(${NAME} PROPERTIES DISABLED On)
    endif()
endfunction()

set(IMPLICITGEMM_MLIR_ENV_BASE MIOPEN_FIND_MODE=normal)
set(IMPLICITGEMM_MLIR_ENV_F ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwd)
set(IMPLICITGEMM_MLIR_ENV_B ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwd)
set(IMPLICITGEMM_MLIR_ENV_W ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrW)

set(TEST_CONV_VERBOSE_F ${MIOPEN_TEST_FLOAT_ARG} --verbose --disable-backward-data --disable-backward-weights)
set(TEST_CONV_VERBOSE_B ${MIOPEN_TEST_FLOAT_ARG} --verbose --disable-forward --disable-backward-weights)
set(TEST_CONV_VERBOSE_W ${MIOPEN_TEST_FLOAT_ARG} --verbose --disable-forward --disable-backward-data)

set(IMPLICITGEMM_MLIR_ENV_F_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwdXdlops)
set(IMPLICITGEMM_MLIR_ENV_B_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwdXdlops)
set(IMPLICITGEMM_MLIR_ENV_W_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrWXdlops)

set(IMPLICITGEMM_TESTING_ENV
 MIOPEN_DEBUG_CONV_WINOGRAD=0
 MIOPEN_DEBUG_CONV_FFT=0
 MIOPEN_DEBUG_CONV_DIRECT=0
 MIOPEN_DEBUG_CONV_GEMM=0
 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
)

if(WORKAROUND_ISSUE_936 AND MIOPEN_TEST_HALF)
    SET(SAVE_IMPLICITGEMM_TESTING_ENV ${IMPLICITGEMM_TESTING_ENV})
    LIST(APPEND IMPLICITGEMM_TESTING_ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 MIOPEN_FIND_MODE=normal)
    SET(SAVE_MIOPEN_TEST_FLOAT_ARG ${MIOPEN_TEST_FLOAT_ARG})
    LIST(APPEND MIOPEN_TEST_FLOAT_ARG --disable-forward --disable-backward-data)
    #Afther fix need to remove '| grep -v "No suitable algorithm was found to execute the required convolution"'
endif()



if(WORKAROUND_ISSUE_936 AND MIOPEN_TEST_HALF)
    SET(IMPLICITGEMM_TESTING_ENV ${SAVE_IMPLICITGEMM_TESTING_ENV})
    SET(MIOPEN_TEST_FLOAT_ARG ${SAVE_MIOPEN_TEST_FLOAT_ARG})
endif()

if(MIOPEN_TEST_DEEPBENCH)
    add_custom_test(test_deepbench_rnn GFX94X_ENABLED GFX103X_ENABLED GFX110X_ENABLED
    )
endif()

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

# TODO: disabled for WORKAROUND_ISSUE_1979
#add_custom_test(test_conv_igemm_dynamic_xdlops_bwd_group SKIP_UNLESS_ALL HALF_ENABLED FLOAT_DISABLED GFX90A_DISABLED GFX94X_ENABLED GFX900_DISABLED GFX906_DISABLED SKIP_XNACK_ON
#    ENVIRONMENT ${DYNAMIC_IMPLICITGEMM_BWD_ENVS_XDLOPS}
#    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  64  32 28 28 --weights 16  16 1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 2 --disable-forward --disable-backward-weights
#    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  16  64 17 17 --weights 64  16 1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4 --disable-forward --disable-backward-weights
#    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input  8  128 56 56 --weights 128 16 3 3 --pads_strides_dilations 1 1 1 1 1 1 --group-count 8 --disable-forward --disable-backward-weights
#)

# Be careful to add testings for (x=1, y=1, c % 8 != 0) due to WORKAROUND_SWDEV_306318

set(DYNAMIC_IMPLICITGEMM_DLOPS_NCHWC_FWD_ENVS
    ${DYNAMIC_IMPLICITGEMM_COMMON}
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC)

set(ARGS_NCHWC_NCHWC_FWD_FP16x4
    --cmode convfp16
    --disable-backward-data
    --disable-backward-weights
    --in_layout NCHW
    --fil_layout NCHW
    --out_layout NCHW
    --tensor_vect 1
    --vector_length 4)

set(ARGS_NCHWC_NCHWC_FWD_FP16x8
    --cmode convfp16
    --disable-backward-data
    --disable-backward-weights
    --in_layout NCHW
    --fil_layout NCHW
    --out_layout NCHW
    --tensor_vect 1
    --vector_length 8)

set(ARGS_NCHWC_CHWNC_FWD_FP16x4
    --cmode convfp16
    --disable-backward-data
    --disable-backward-weights
    --in_layout NCHW
    --fil_layout CHWN
    --out_layout NCHW
    --tensor_vect 1
    --vector_length 4)

set(ARGS_NCHWC_CHWNC_FWD_FP16x8
    --cmode convfp16
    --disable-backward-data
    --disable-backward-weights
    --in_layout NCHW
    --fil_layout CHWN
    --out_layout NCHW
    --tensor_vect 1
    --vector_length 8)

add_custom_test(test_regression_issue_2624 SKIP_UNLESS_ALL GFX900_DISABLED GFX906_DISABLED
    # Regression test for SWDEV-433714 (issue 2624)
    ENVIRONMENT ${IMPLICITGEMM_TESTING_ENV} MIOPEN_LOG_LEVEL=5
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose --input 2 1 22 22 --weights 1 1 4 4 --pads_strides_dilations 1 2 4 4 3 2 --disable-backward-data --disable-backward-weights
)

# MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW is explicitly enabled due to the kernel is disabled by default via #2306
set(CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW_ENV
    MIOPEN_FIND_MODE=normal
    MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvCkIgemmFwdV6r1DlopsNchw
    MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW=1)

if(MIOPEN_TEST_FLOAT)
    add_custom_test(test_reduce_double SKIP_UNLESS_ALL GFX94X_ENABLED GFX103X_ENABLED GFX110X_ENABLED COMMAND  $<TARGET_FILE:test_reduce_test> --double --all --verbose)
endif()

# NOTES ON WRITING TESTS FOR TUNABLE SOLVERS
# * Enforce tuning (SEARCH_DB_UPDATE).
# * Use TEST_TUNING. This flag leads to test failure in case of any "Error"
#   message output to the log, which happens if something is broken in the tuning machinery.
# * Use MIOPEN_DEBUG_TUNING_ITERATIONS_MAX to save testing time.

add_custom_test(smoke_solver_ConvHipImplicitGemmFwdXdlops GFX900_DISABLED GFX906_DISABLED GFX90A_DISABLED GFX94X_ENABLED HALF_ENABLED INT8_ENABLED
    ENVIRONMENT MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmFwdXdlops
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 64 56 56 --weights 64 64 1 1 --pads_strides_dilations 0 0 1 1 1 1 ${MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT8} --in_layout NHWC --fil_layout NHWC --out_layout NHWC ${MIOPEN_TEST_FLAGS_ARGS}
)

# FP16 ALT attribute is disabled to enable the backward solver on MI200 for HALF.
add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_3x2 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<3-2>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 3 3 --pads_strides_dilations 1 1 2 2 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_3x3 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<3-3>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 3 3 --pads_strides_dilations 1 1 2 2 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_3x4 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4=1 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<3-4>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 3 3 --pads_strides_dilations 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_3x5 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5=1 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<3-5>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 3 3 --pads_strides_dilations 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_3x6 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6=1 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<3-6>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 3 3 --pads_strides_dilations 1 1 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_5x3 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<5-3>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 5 5 --pads_strides_dilations 2 2 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_5x4 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<5-4>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 5 5 --pads_strides_dilations 2 2 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_7x2 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<7-2>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 7 7 --pads_strides_dilations 3 3 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_7x2x1x1 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<7-2-1-1>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 7 1 --pads_strides_dilations 3 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_1x1x7x2 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<1-1-7-2>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 1 7 --pads_strides_dilations 0 3 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_7x3 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<7-3>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 7 7 --pads_strides_dilations 3 3 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW_7x3x1x1 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<7-3-1-1>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 7 1 --pads_strides_dilations 3 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvWinograd3x3MultipassWrW1x1x7x3 HALF_ENABLED BF16_ENABLED SKIP_XNACK_ON
    ENVIRONMENT MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0 MIOPEN_FIND_MODE=normal MIOPEN_DEBUG_FIND_ONLY_SOLVER='ConvWinograd3x3MultipassWrW<1-1-7-3>'
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 1 16 24 24 --weights 16 16 1 7 --pads_strides_dilations 0 3 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvMlirIgemm_F GFX900_DISABLED GFX908_DISABLED GFX90A_DISABLED GFX103X_ENABLED HALF_ENABLED SKIP_UNLESS_MLIR TEST_TUNING
    ENVIRONMENT MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5 ${IMPLICITGEMM_MLIR_ENV_F}
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 64 128 14 14 --weights 128 128 1 1 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvMlirIgemm_B GFX900_DISABLED GFX908_DISABLED GFX90A_DISABLED GFX103X_ENABLED HALF_ENABLED SKIP_UNLESS_MLIR TEST_TUNING
    ENVIRONMENT MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5 ${IMPLICITGEMM_MLIR_ENV_B}
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 64 256 28 28 --weights 64  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(smoke_solver_ConvMlirIgemm_W GFX900_DISABLED GFX908_DISABLED GFX90A_DISABLED GFX103X_ENABLED HALF_ENABLED SKIP_UNLESS_MLIR TEST_TUNING
    ENVIRONMENT MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5 ${IMPLICITGEMM_MLIR_ENV_W}
    COMMAND $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 64 64  28 28 --weights 64  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 ${MIOPEN_TEST_FLAGS_ARGS}
)
#Dont install the tests if we are building each test separately
#When the tests are built as a single unit, then we want to be
#able to package them
set(INSTALL_GTEST OFF)
if(MIOPEN_TEST_DISCRETE)
add_subdirectory(gtest EXCLUDE_FROM_ALL)
else()
add_subdirectory(gtest)
endif()
