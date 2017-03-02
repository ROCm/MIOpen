
include(CTest)

option( MLOPEN_TEST_ALL "Run the full test suite" OFF )
option( MLOPEN_TEST_ALL_GENERATORS "Test using all different data generators" OFF )

if(MLOPEN_TEST_ALL_GENERATORS)
    add_definitions(-DMLOPEN_TEST_ALL_GENERATORS=1)
else()
    add_definitions(-DMLOPEN_TEST_ALL_GENERATORS=0)
endif()

find_package(Threads REQUIRED)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
    # Cmake does not add flags correctly for gcc
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
    endif()
    if(MLOPEN_TEST_ALL)
        set(TEST_COMMAND $<TARGET_FILE:${TEST_NAME}> --all)
    else()
        set(TEST_COMMAND $<TARGET_FILE:${TEST_NAME}>)
    endif()
    if(WIN32)
        add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND ${TEST_COMMAND})
    else()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_COMMAND})
        # TODO: Quick hack to fix relative paths to kernels
        # add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${TEST_COMMAND})
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

