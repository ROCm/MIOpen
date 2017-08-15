#!/usr/bin/cmake -P

set(ARGS)
foreach(i RANGE 3 ${CMAKE_ARGC})
    list(APPEND ARGS ${CMAKE_ARGV${i}})
endforeach()

include(CMakeParseArguments)

set(options)
set(oneValueArgs --prefix)
set(multiValueArgs)

cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGS})

set(_PREFIX /usr/local)
if(PARSE_--prefix)
    set(_PREFIX ${PARSE_--prefix})
endif()

get_filename_component(PREFIX ${_PREFIX} ABSOLUTE)

find_package(CMakeGet QUIET PATHS ${PREFIX})

if(NOT CMakeGet_FOUND)
set(FILENAME ${PREFIX}/tmp/cmake-get-install.cmake)
file(DOWNLOAD https://raw.githubusercontent.com/pfultz2/cmake-get/master/install.cmake ${FILENAME})
execute_process(COMMAND ${CMAKE_COMMAND} -P ${FILENAME} ${PREFIX})
file(REMOVE ${FILENAME})
find_package(CMakeGet QUIET PATHS ${PREFIX})
endif()

cmake_get(pfultz2/rocm-recipes PREFIX ${PREFIX} CMAKE_ARGS ${PARSE_UNPARSED_ARGUMENTS})
cmake_get_from(${CMAKE_CURRENT_LIST_DIR}/dev-requirements.txt PREFIX ${PREFIX} CMAKE_ARGS ${PARSE_UNPARSED_ARGUMENTS})
