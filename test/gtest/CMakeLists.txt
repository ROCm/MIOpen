find_package(GTest REQUIRED)
find_package(rocblas)

if(MIOPEN_USE_HIPBLASLT)
  find_package(hipblas REQUIRED PATHS /opt/rocm $ENV{HIP_PATH})
  find_package(hipblaslt REQUIRED PATHS /opt/rocm $ENV{HIP_PATH})
endif()

set(SOURCES
    log.cpp
    platform.cpp
    conv_common.cpp
    )

if(MIOPEN_BACKEND_OPENCL)
  set(SKIP_TESTS dumpTensorTest.cpp)
endif()

function(add_gtest TEST_NAME TEST_CPP)
  message("Adding Test: " ${TEST_NAME} " : " ${TEST_CPP})
  add_executable(${TEST_NAME} ${TEST_CPP}  ${SOURCES})
  if(WIN32)
    target_compile_definitions(${TEST_NAME} PRIVATE NOMINMAX)
  endif()

  
  add_dependencies(tests ${TEST_NAME})
  add_dependencies(check ${TEST_NAME})
  target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
  target_include_directories(${TEST_NAME} PRIVATE ../ ../../src/kernels)
  if(MIOPEN_ENABLE_AI_KERNEL_TUNING)
    target_link_libraries(${TEST_NAME} frugally-deep::fdeep Eigen3::Eigen)
  endif()
  if(hipblaslt_FOUND)
    target_link_libraries( ${TEST_NAME} roc::hipblaslt )
  endif()
  # Workaround : change in rocm-cmake was causing linking error so had to add ${CMAKE_DL_LIBS} 
  #               We can remove ${CMAKE_DL_LIBS} once root cause is identified.
  target_link_libraries(${TEST_NAME} ${CMAKE_DL_LIBS} GTest::gtest GTest::gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas> )
  if(NOT MIOPEN_EMBED_DB STREQUAL "")
      target_link_libraries(${TEST_NAME} $<BUILD_INTERFACE:miopen_data>)
  endif()
  if(MIOPEN_TEST_DISCRETE)
    string(CONCAT TEST_ENVIRONMENT_VARIABLES
    "ENVIRONMENT;MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};"
    "ENVIRONMENT;MIOPEN_TEST_FLOAT_ARG=${MIOPEN_TEST_FLOAT_ARG};"
    "ENVIRONMENT;MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL};"
    "ENVIRONMENT;MIOPEN_TEST_MLIR=${MIOPEN_TEST_MLIR};"
    "ENVIRONMENT;MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL};"
    "ENVIRONMENT;CODECOV_TEST=${CODECOV_TEST};"
    "ENVIRONMENT;MIOPEN_TEST_DBSYNC=${MIOPEN_TEST_DBSYNC};"
    "ENVIRONMENT;MIOPEN_TEST_CONV=${MIOPEN_TEST_CONV};"
    "ENVIRONMENT;MIOPEN_TEST_DEEPBENCH=${MIOPEN_TEST_DEEPBENCH};"
    "ENVIRONMENT;MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=${MIOPEN_DEBUG_TUNING_ITERATIONS_MAX};"
    "ENVIRONMENT;MIOPEN_TEST_WITH_MIOPENDRIVER=${MIOPEN_TEST_WITH_MIOPENDRIVER};")

    # Enable CMake to discover the test binary
    # Note: Due to the following cmake issue with gtest_discover_tests https://gitlab.kitware.com/cmake/cmake/-/issues/17812 you cannot pass PROPERTIES as a list.
    #       To work around this limitation, we are passing the environment variables in the format ENVIRONMENT;value1=${value1};ENVIRONMENT;value2=${value2}.
    gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR} PROPERTIES ${TEST_ENVIRONMENT_VARIABLES})
  endif()
  target_link_libraries(${TEST_NAME} BZip2::BZip2)
  if(WIN32)
    # Refer to https://en.cppreference.com/w/cpp/language/types for details.
    target_compile_options(${TEST_NAME} PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
  endif()
  if(HAS_LIB_STD_FILESYSTEM)
    target_link_libraries(${TEST_NAME} stdc++fs)
  endif()
endfunction()

file(GLOB TESTS *.cpp)
# Remove files that do not contain tests
foreach(SOURCE ${SOURCES})
    list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
endforeach()

foreach(SOURCE ${SKIP_TESTS})
    list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
endforeach()

if( MIOPEN_TEST_DISCRETE )
  foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_gtest(test_${BASE_NAME} ${BASE_NAME}.cpp)
  endforeach()

else()
  foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME)
    list(APPEND TESTS_CPP ${BASE_NAME})
  endforeach()

  add_gtest(miopen_gtest "${TESTS_CPP}")

  if( NOT ENABLE_ASAN_PACKAGING )
    install(TARGETS miopen_gtest
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT client)
  endif()
endif()

message(STATUS "gtest env: MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "gtest env: MIOPEN_TEST_FLOAT_ARG=${MIOPEN_TEST_FLOAT_ARG}")
message(STATUS "gtest env: MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL}")
message(STATUS "gtest env: MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL}")
