
include(CTest)

option( MLOPEN_TEST_ALL "Run the full test suite" OFF )

if(MLOPEN_TEST_ALL)
    add_definitions(-DMLOPEN_TEST_ALL=1)
else()
    add_definitions(-DMLOPEN_TEST_ALL=0)
endif()

find_package(Threads REQUIRED)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
    # Older versions of cmake do not add flags correctly
    if(${CMAKE_VERSION} VERSION_LESS 3.0 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
    endif()
    if(WIN32)
        add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    else()
        # add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
        # TODO: Quick hack to fix relative paths to kernels
        add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    endif()
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
endfunction(add_test_executable)

file(GLOB TESTS *.cpp)
foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST})
    target_link_libraries(test_${BASE_NAME} MLOpen)
endforeach()

