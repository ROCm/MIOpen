file(GLOB TESTS *.cpp)

if(MIOPEN_TEST_DRIVER_ITER_MODE)
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=2)
else()
    add_definitions(-DMIOPEN_TEST_DRIVER_MODE=1)
endif()

function(add_speedtest_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
    # Cmake does not add flags correctly for gcc
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
    endif()
    separate_arguments(MIOPEN_TEST_FLAGS_ARGS UNIX_COMMAND ${MIOPEN_TEST_FLAGS})
    target_link_libraries(${TEST_NAME}
        optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
        optimized ${Boost_SYSTEM_LIBRARY_RELEASE}
        debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
        debug ${Boost_SYSTEM_LIBRARY_DEBUG}
    )
    target_link_libraries(${TEST_NAME} MIOpen)
    target_include_directories(${TEST_NAME} SYSTEM PUBLIC ${HALF_INCLUDE_DIR})
    target_include_directories(${TEST_NAME} PRIVATE ../test)
endfunction(add_speedtest_executable)

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_speedtest_executable(speedtest_${BASE_NAME} ${TEST})
endforeach()
