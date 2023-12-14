find_package(GTest CONFIG REQUIRED)
if(GTest_FOUND)
  MESSAGE(STATUS "GoogleTest: " ${GTest})
  enable_testing()
  include(GoogleTest)
endif()

find_package(rocblas)

set(SOURCES
    log.cpp
    platform.cpp
    )

if(MIOPEN_BACKEND_OPENCL)
  set(SKIP_TESTS dumpTensorTest.cpp)
endif()


file(GLOB TESTS *.cpp)
file(GLOB HEADERS *.hpp)
# Remove files that do not contain tests
foreach(SOURCE ${SOURCES})
    list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
endforeach()

foreach(SOURCE ${SKIP_TESTS})
    list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
endforeach()

message("Test files: " ${TESTS})
foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME)
    list(APPEND TESTS_CPP ${BASE_NAME}) 
endforeach()


message("Adding miopen_gtest: " ${TESTS_CPP})
add_executable(miopen_gtest ${TESTS_CPP} ${SOURCES})
add_dependencies(tests miopen_gtest)
add_dependencies(check miopen_gtest)
target_compile_options(miopen_gtest PRIVATE -Wno-global-constructors -Wno-undef)
target_include_directories(miopen_gtest PRIVATE ../ ../../src/kernels)
if(MIOPEN_ENABLE_AI_KERNEL_TUNING)
  target_include_directories(miopen_gtest SYSTEM PRIVATE $<BUILD_INTERFACE:${FDEEP_INCLUDE_DIR}>)
  target_include_directories(miopen_gtest SYSTEM PRIVATE $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIR}/eigen3>)
endif()
target_link_libraries(miopen_gtest GTest::gtest GTest::gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas>)
if(NOT MIOPEN_EMBED_DB STREQUAL "")
    target_link_libraries(miopen_gtest $<BUILD_INTERFACE:miopen_data>)
endif()
# Enable CMake to discover the test binary
gtest_discover_tests(miopen_gtest PROPERTIES ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};MIOPEN_TEST_FLOAT_ARG=${MIOPEN_TEST_FLOAT_ARG};MIOPEN_TEST_ALL=${MIOPEN_TEST_ALL};MIOPEN_TEST_MLIR=${MIOPEN_TEST_MLIR};MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL}")

if( NOT ENABLE_ASAN_PACKAGING )
  install(TARGETS miopen_gtest
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
      DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
