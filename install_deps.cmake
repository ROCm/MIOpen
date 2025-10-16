#!/usr/bin/cmake -P

function(cget_exec)
    execute_process(${ARGN} RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        message("${RESULT}")
        message(FATAL_ERROR "Process failed: ${ARGN}")
    endif()
endfunction()

set(ARGS)
foreach(i RANGE 3 ${CMAKE_ARGC})
    list(APPEND ARGS ${CMAKE_ARGV${i}})
endforeach()

include(CMakeParseArguments)

set(options help --minimum)
set(oneValueArgs --prefix --generator)
set(multiValueArgs)

cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGS})

if(PARSE_help)
message("Usage: install_deps.cmake [options] [cmake-args]")
message("")
message("Options:")
message("  --prefix                   Set the prefix to install the dependencies.")
message("  --generator <generator>    Specify the CMake generator (e.g., Ninja, Unix Makefiles).")
message("")
message("Commands:")
message("  help                       Show this message and exit.")
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

# Optionally map --generator to CGET_EXTRA_OPTIONS
if(DEFINED PARSE_--generator)
    set(VALID_GENERATORS "Ninja" "Unix Makefiles")
    list(FIND VALID_GENERATORS "${PARSE_--generator}" _GENERATOR_INDEX)
    if(_GENERATOR_INDEX EQUAL -1)
        message(WARNING "Invalid generator '${PARSE_--generator}'. Valid options are: ${VALID_GENERATORS}. Ignoring this generator.")
    else()
        list(APPEND CGET_EXTRA_OPTIONS -G "${PARSE_--generator}")
    endif()
endif()

find_program(VIRTUALENV_PYTHON_EXE python3)
if(NOT VIRTUALENV_PYTHON_EXE)
    find_program(VIRTUALENV_PYTHON_EXE python)
endif()

function(virtualenv_create)
    cget_exec(
        COMMAND ${VIRTUALENV_PYTHON_EXE} -m venv ${PREFIX}
    )
endfunction()

function(virtualenv_install)
    virtualenv_create()
    # TODO: Check result
    message("${PREFIX}/pip install ${ARGN}")
    cget_exec(
        COMMAND ${PREFIX}/bin/python ${PREFIX}/bin/pip install ${ARGN}
    )
endfunction()

virtualenv_install(cget)

# Set compiler to hip-clang if not set
if(NOT DEFINED ENV{CXX} AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
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

if(NOT DEFINED ENV{CC} AND NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(CLANGC clang
        PATHS
            /opt/rocm/llvm
            /opt/rocm
        PATH_SUFFIXES
            bin
    )
    if(CLANGC)
        set(ENV{CC} ${CLANGC})
    else()
        message(STATUS "Cannot find clang to set to CC")
    endif()
endif()

function(cget)
    cget_exec(COMMAND ${PREFIX}/bin/cget -p ${PREFIX} ${ARGN})
endfunction()

# Clean directory
# Skip clean since relative symlinks are incorrectly removed
# cget(clean -y)

set(TOOLCHAIN_FLAG)
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    set(TOOLCHAIN_FLAG -t ${CMAKE_TOOLCHAIN_FILE})
endif()
# Initialize directory
cget(init ${TOOLCHAIN_FLAG} -DCMAKE_INSTALL_RPATH=${PREFIX}/lib ${PARSE_UNPARSED_ARGUMENTS})
cget(ignore pcre)

# Install dependencies
cget(install ${CGET_EXTRA_OPTIONS} -U ROCm/rocm-recipes@d7827046100ac0ed8167e16c53999baa126e760d)
cget(install ${CGET_EXTRA_OPTIONS} -U -f requirements.txt)
