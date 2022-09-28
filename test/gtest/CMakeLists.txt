include(googletest) #include googletest.cmake
enable_testing()
include(GoogleTest)

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
#add_custom_target(tests)

find_package(rocblas)

set(SOURCES ${SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/log_test_helper.cpp"
    )

function(add_gtest TEST_NAME)
  message("Adding Test: " ${TEST_NAME})
  add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp ${SOURCES})
  add_dependencies(tests test_${TEST_NAME})
  add_dependencies(check test_${TEST_NAME})
  target_compile_options(test_${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
  target_include_directories(test_${TEST_NAME} PRIVATE ../)
  if(NOT MIOPEN_EMBED_DB STREQUAL "")
    target_link_libraries(test_${TEST_NAME} gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas> $<BUILD_INTERFACE:miopen_data>)
  else()
    target_link_libraries(test_${TEST_NAME} gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas>)
  endif()
  #Enable CMake to discover the test binary
  gtest_discover_tests(test_${TEST_NAME})
endfunction()

file(GLOB TESTS *.cpp)
# this is the file I want to exclude / remove from the TESTS
list(REMOVE_ITEM TESTS "${SOURCES}")

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_gtest(${BASE_NAME})
endforeach()
