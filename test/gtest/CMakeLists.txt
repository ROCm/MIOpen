find_package(GTest REQUIRED)
find_package(rocblas)

set(SOURCES
    log.cpp
    platform.cpp
    )

if(MIOPEN_BACKEND_OPENCL)
  set(SKIP_TESTS dumpTensorTest)
endif()

function(add_gtest TEST_NAME)
  if( NOT (TEST_NAME IN_LIST SKIP_TESTS))
    message("Adding Test: " ${TEST_NAME})
    add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp  ${SOURCES})
    if(WIN32)
      target_compile_definitions(test_${TEST_NAME} PRIVATE NOMINMAX)
    endif()
    add_dependencies(tests test_${TEST_NAME})
    add_dependencies(check test_${TEST_NAME})
    target_compile_options(test_${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
    target_include_directories(test_${TEST_NAME} PRIVATE ../ ../../src/kernels)
    if(MIOPEN_ENABLE_AI_KERNEL_TUNING)
      target_link_libraries(test_${TEST_NAME} frugally-deep::fdeep Eigen3::Eigen)
    endif()
    target_link_libraries(test_${TEST_NAME} GTest::gtest GTest::gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas>)
    if(NOT MIOPEN_EMBED_DB STREQUAL "")
        target_link_libraries(test_${TEST_NAME} $<BUILD_INTERFACE:miopen_data>)
    endif()
    if(NOT WIN32) # TODO: cannot run on Windows due to missing DLL dependencies
      # Enable CMake to discover the test binary
      gtest_discover_tests(test_${TEST_NAME} PROPERTIES ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};MIOPEN_TEST_FLOAT_ARG=${MIOPEN_TEST_FLOAT_ARG};MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL};MIOPEN_TEST_MLIR=${MIOPEN_TEST_MLIR};MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL}")
    endif()

    if(WIN32)
      # Refer to https://en.cppreference.com/w/cpp/language/types for details.
      target_compile_options(test_${TEST_NAME} PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
    endif()
  endif()
endfunction()

file(GLOB TESTS *.cpp)
# Remove files that do not contain tests
foreach(SOURCE ${SOURCES})
    list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
endforeach()

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_gtest(${BASE_NAME})
endforeach()

message(STATUS "gtest env: MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "gtest env: MIOPEN_TEST_FLOAT_ARG=${MIOPEN_TEST_FLOAT_ARG}")
message(STATUS "gtest env: MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL}")
message(STATUS "gtest env: MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL}")
