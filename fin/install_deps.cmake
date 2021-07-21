#!/usr/bin/cmake -P

set(ARGS)
foreach(i RANGE 3 ${CMAKE_ARGC})
    list(APPEND ARGS ${CMAKE_ARGV${i}})
endforeach()

include(CMakeParseArguments)

set(options help --minimum)
set(oneValueArgs --prefix)
set(multiValueArgs)

cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGS})

if(PARSE_help)
message("Usage: install_deps.cmake [options] [cmake-args]")
message("")
message("Options:")
message("  --prefix               Set the prefix to install the dependencies.")
message("")
message("Commands:")
message("  help                   Show this message and exit.")
message("")
message("  --minimum                  Install minimum dependencies.")
message("")
return()
endif()

set(_PREFIX /usr/local)
if(PARSE_--prefix)
    set(_PREFIX ${PARSE_--prefix})
endif()

get_filename_component(PREFIX ${_PREFIX} ABSOLUTE)

find_package(CMakeGet QUIET PATHS ${PREFIX})

if(NOT CMakeGet_FOUND)
    set(FILENAME ${PREFIX}/tmp/cmake-get-install.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/pfultz2/cmake-get/master/install.cmake ${FILENAME} STATUS RESULT_LIST)
    list(GET RESULT_LIST 0 RESULT)
    list(GET RESULT_LIST 1 RESULT_MESSAGE)
    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "Download for install.cmake failed: ${RESULT_MESSAGE}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -P ${FILENAME} ${PREFIX})
    file(REMOVE ${FILENAME})
    find_package(CMakeGet REQUIRED PATHS ${PREFIX})
endif()

# Set compiler to hcc if not set
if(NOT DEFINED ENV{CXX} AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(HCC hcc
        PATHS
            /opt/rocm/hcc
            /opt/rocm
        PATH_SUFFIXES
            bin
    )
    if(HCC)
        set(ENV{CXX} ${HCC})
    else()
	message(STATUS "Cannot find hcc")
	find_program(CLANGXX clang++
	    PATHS
	        /opt/rocm/llvm
		/opt/rocm
	    PATH_SUFFIXES
	        bin
	)
	if(CLANGXX)
	    set(ENV{CXX} ${CLANGXX})
	else()
            message(FATAL_ERROR "Cannot find clang++")
	endif()
    endif()
endif()

cmake_get(pfultz2/rocm-recipes PREFIX ${PREFIX} CMAKE_ARGS ${PARSE_UNPARSED_ARGUMENTS})

if(PARSE_--minimum)
    message(STATUS "MIOpen minimum dependency install.")
    set(REQFILE ${CMAKE_CURRENT_LIST_DIR}/min-requirements.txt)
else()
    message(STATUS "MIOpen normal dependency install.")
    set(REQFILE ${CMAKE_CURRENT_LIST_DIR}/requirements.txt)
endif()

cmake_get_from(${REQFILE} PREFIX ${PREFIX} CMAKE_ARGS -DCMAKE_INSTALL_RPATH=${PREFIX}/lib ${PARSE_UNPARSED_ARGUMENTS})
