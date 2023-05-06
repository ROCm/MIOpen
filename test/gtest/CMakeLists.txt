include(GoogleTest)

enable_testing()

function(add_gtest TEST_NAME)
  if( NOT (TEST_NAME IN_LIST SKIP_TESTS))
    message("Adding Test: " ${TEST_NAME})
    add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp log_test_helper.cpp platform.cpp)
    target_compile_options(test_${TEST_NAME} PRIVATE ${__COMPILER_WARNINGS})
    add_dependencies(tests test_${TEST_NAME})
    add_dependencies(check test_${TEST_NAME})
    if(NOT MSVC)
      target_compile_options(test_${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
    endif()
    target_include_directories(test_${TEST_NAME}
      PRIVATE
        ${CMAKE_SOURCE_DIR}/test/
        ${CMAKE_SOURCE_DIR}/src/include/
        ${CMAKE_SOURCE_DIR}/include/
      )
    target_include_directories(test_${TEST_NAME} SYSTEM PRIVATE ${HALF_INCLUDE_DIR})
    if(MIOPEN_ENABLE_AI_KERNEL_TUNING)
      target_include_directories(test_${TEST_NAME} SYSTEM PRIVATE $<BUILD_INTERFACE:${FDEEP_INCLUDE_DIR}>)
      target_include_directories(test_${TEST_NAME} SYSTEM PRIVATE $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIR}/eigen3>)
    endif()
    target_link_libraries(test_${TEST_NAME}
        PRIVATE
            gtest
            gtest_main
            MIOpen
            MIOpen_Static
            hip::host
            nlohmann_json::nlohmann_json
        )
    if(WIN32)
      target_link_libraries(test_${TEST_NAME} PRIVATE bzip2::bzip2)
    else()
      target_include_directories(test_${TEST_NAME} SYSTEM PRIVATE ${BZIP2_INCLUDE_DIR})
      target_link_libraries(test_${TEST_NAME} PRIVATE ${BZIP2_LIBRARIES})
    endif()
    if(rocBLAS_FOUND)
      target_link_libraries(test_${TEST_NAME} PRIVATE roc::rocblas)
    endif()
    if(NOT MIOPEN_EMBED_DB STREQUAL "")
      target_link_libraries(test_${TEST_NAME} PRIVATE miopen_data)
    endif()
#    if(NOT WIN32)
#      # Enable CMake to discover the test binary
#      # TODO: turn on for Windows after enabling all the tests.
#      gtest_discover_tests(test_${TEST_NAME} PROPERTIES ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR}")
#    endif()
    if(MIOPEN_ENABLE_SQLITE)
      target_link_libraries(test_${TEST_NAME} PRIVATE sqlite3::sqlite3)
    endif()
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
