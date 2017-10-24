################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
################################################################################

include(CMakeParseArguments)
include(ProcessorCount)

find_program(CPPCHECK_EXE 
    NAMES 
        cppcheck
    PATHS
        /opt/rocm/bin
)

ProcessorCount(CPPCHECK_JOBS)

macro(enable_cppcheck)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs CHECKS SUPPRESS DEFINE UNDEFINE INCLUDE SOURCES)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    string(REPLACE ";" "," CPPCHECK_CHECKS "${PARSE_CHECKS}")
    string(REPLACE ";" "\n" CPPCHECK_SUPPRESS "${PARSE_SUPPRESS};*:/usr/*")
    file(WRITE ${CMAKE_BINARY_DIR}/cppcheck-supressions "${CPPCHECK_SUPPRESS}")
    set(CPPCHECK_DEFINES)
    foreach(DEF ${PARSE_DEFINE})
        list(APPEND CPPCHECK_DEFINES -D ${DEF})
    endforeach()

    set(CPPCHECK_UNDEFINES)
    foreach(DEF ${PARSE_UNDEFINE})
        list(APPEND CPPCHECK_UNDEFINES -U ${DEF})
    endforeach()

    set(CPPCHECK_INCLUDES)
    foreach(INC ${PARSE_INCLUDE})
        list(APPEND CPPCHECK_INCLUDES -I ${INC})
    endforeach()

    set(CPPCHECK_COMMAND 
        ${CPPCHECK_EXE}
        -q
        # --check-config
        # --report-progress
        --platform=native
        --template '{file}:{line}: {severity} ({id}): {message}'
        -i /usr/local/include
        -j ${CPPCHECK_JOBS}
        ${CPPCHECK_DEFINES}
        ${CPPCHECK_UNDEFINES}
        ${CPPCHECK_INCLUDES}
        "--enable=${CPPCHECK_CHECKS}"
        "--suppressions-list=${CMAKE_BINARY_DIR}/cppcheck-supressions"
        "--project=${CMAKE_BINARY_DIR}/compile_commands.json"
        ${PARSE_SOURCES}
    )

    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_COMMAND}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "cppcheck: Running cppcheck..."
    )
endmacro()


