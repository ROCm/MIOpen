include(gtest) #include gtest.cmake
enable_testing()

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
#add_custom_target(tests)

find_package(rocblas)

set(SOURCES
    log_test_helper.cpp
    platform.cpp
    )

function(add_gtest TEST_NAME)
  if( NOT (TEST_NAME IN_LIST SKIP_TESTS))
    message("Adding Test: " ${TEST_NAME})
    add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp  ${SOURCES})
    add_dependencies(tests test_${TEST_NAME})
    add_dependencies(check test_${TEST_NAME})
    target_compile_options(test_${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
    target_include_directories(test_${TEST_NAME} PRIVATE ../)
    if(MIOPEN_ENABLE_AI_KERNEL_TUNING)
      target_include_directories(test_${TEST_NAME} SYSTEM PRIVATE $<BUILD_INTERFACE:${FDEEP_INCLUDE_DIR}>)
      target_include_directories(test_${TEST_NAME} SYSTEM PRIVATE $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIR}/eigen3>)
    endif()
    target_link_libraries(test_${TEST_NAME} gtest gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas>)
    if(NOT MIOPEN_EMBED_DB STREQUAL "")
        target_link_libraries(test_${TEST_NAME} $<BUILD_INTERFACE:miopen_data>)
    endif()
    # Enable CMake to discover the test binary
    gtest_discover_tests(test_${TEST_NAME} PROPERTIES ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL};MIOPEN_TEST_MLIR=${MIOPEN_TEST_MLIR};MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL}")

  endif()
endfunction()

add_gtest(api_convbiasactiv)
add_gtest(cba_infer)
add_gtest(conv_api)
add_gtest(conv_api_strided_tensors)
add_gtest(log_test)
add_gtest(log_test_neg)
add_gtest(na_infer)
add_gtest(solver_convasm3x3u)

if(NOT MIOPEN_BACKEND_OPENCL)
  add_gtest(dumpTensorTest)
endif()
