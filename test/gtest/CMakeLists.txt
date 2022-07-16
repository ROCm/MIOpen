include(googletest) #include googletest.cmake
enable_testing()
include(GoogleTest)

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
#add_custom_target(tests)

find_package(rocblas)

function(add_gtest TEST_NAME)
  message("Adding Test: " ${TEST_NAME})
  add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp )
  add_dependencies(tests test_${TEST_NAME})
  add_dependencies(check test_${TEST_NAME})
  target_include_directories(test_${TEST_NAME} PRIVATE ../ PRIVATE ${PROJECT_SOURCE_DIR}/driver)
  target_compile_options(test_${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
  #message(STATUS, ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_${TEST_NAME} gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas> ${PROJECT_SOURCE_DIR}/build/driver/CMakeFiles/MIOpenDriver.dir/InputFlags.cpp.o)#$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/driver>)#${CMAKE_CURRENT_SOURCE_DIR}/../../build/driver/CMakeFiles/MIOpenDriver.dir)
  message(STATUS, $<BUILD_INTERFACE>)
  #Enable CMake to discover the test binary
  gtest_discover_tests(test_${TEST_NAME})
endfunction()

file(GLOB TESTS *.cpp)
foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_gtest(${BASE_NAME})
endforeach()
