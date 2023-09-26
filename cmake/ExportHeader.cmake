################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

if(COMMAND miopen_generate_export_header)
    return()
endif()

include(GenerateExportHeader)

function(miopen_generate_export_header TARGET)
    cmake_parse_arguments(PARSE "" "DIRECTORY" "" ${ARGN})
    if(PARSE_DIRECTORY)
        set(__directory ${PARSE_DIRECTORY})
    else()
        string(REPLACE "_" "/" __directory ${TARGET})
        string(TOLOWER ${__directory} __directory)
    endif()
    set(__file_name ${CMAKE_BINARY_DIR}/include/${__directory}/export.h)
    generate_export_header(${TARGET} EXPORT_FILE_NAME ${__file_name})
    target_include_directories(${TARGET} PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
    rocm_install(FILES ${__file_name} DESTINATION include/${__directory})
endfunction()
