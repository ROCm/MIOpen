find_package(GTest REQUIRED)
find_package(rocblas)

set(SOURCES
    log.cpp
    platform.cpp
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
  target_link_libraries(${TEST_NAME} GTest::gtest GTest::gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas>)
  if(NOT MIOPEN_EMBED_DB STREQUAL "")
      target_link_libraries(${TEST_NAME} $<BUILD_INTERFACE:miopen_data>)
  endif()
  if(NOT WIN32) # TODO: cannot run on Windows due to missing DLL dependencies
    # Enable CMake to discover the test binary
    # Extend GTest DISCOVERY_TIMEOUT to 5 mins
    gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 PROPERTIES ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};MIOPEN_TEST_FLOAT_ARG=${MIOPEN_TEST_FLOAT_ARG};MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL};MIOPEN_TEST_MLIR=${MIOPEN_TEST_MLIR};MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL};CODECOV_TEST=${CODECOV_TEST};MIOPEN_TEST_DBSYNC=${MIOPEN_TEST_DBSYNC};MIOPEN_TEST_DEEPBENCH=${MIOPEN_TEST_DEEPBENCH}")
  endif()
    target_link_libraries(${TEST_NAME} BZip2::BZip2)
  if(WIN32)
    # Refer to https://en.cppreference.com/w/cpp/language/types for details.
    target_compile_options(${TEST_NAME} PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
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
        DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

message(STATUS "gtest env: MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "gtest env: MIOPEN_TEST_FLOAT_ARG=${MIOPEN_TEST_FLOAT_ARG}")
message(STATUS "gtest env: MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL}")
message(STATUS "gtest env: MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL}")
