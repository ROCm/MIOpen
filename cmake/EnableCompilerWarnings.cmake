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
# - Enable warning all for gcc/clang

## Strict warning level
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(__default_cxx_compiler_options /w)
else()
    set(__default_cxx_compiler_options
        -Wall
        -Wextra
        -Wcomment
        -Wendif-labels
        -Wformat
        -Winit-self
        -Wreturn-type
        -Wsequence-point
        -Wswitch
        -Wtrigraphs
        -Wundef
        -Wuninitialized
        -Wunreachable-code
        -Wunused
        -Wno-ignored-qualifiers
        -Wno-sign-compare
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND __default_cxx_compiler_options
            -Weverything
            -Wno-c++98-compat
            -Wno-c++98-compat-pedantic
            -Wno-conversion
            -Wno-double-promotion
            -Wno-exit-time-destructors
            -Wno-extra-semi
            -Wno-extra-semi-stmt
            -Wno-float-conversion
            -Wno-gnu-anonymous-struct
            -Wno-gnu-zero-variadic-macro-arguments
            -Wno-missing-prototypes
            -Wno-nested-anon-types
            -Wno-option-ignored
            -Wno-padded
            -Wno-return-std-move-in-c++11
            -Wno-shorten-64-to-32
            -Wno-sign-conversion
            -Wno-unknown-warning-option
            -Wno-unused-command-line-argument
            -Wno-weak-vtables
            -Wno-covered-switch-default
            -Wno-unused-result
            -Wno-unsafe-buffer-usage
            -Wno-deprecated-declarations
            -Wno-shadow-uncaptured-local)
        if(WIN32)
            list(APPEND __default_cxx_compile_options
                -fdelayed-template-parsing
                -fms-extensions
                -fms-compatibility)
        endif()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.1")
            list(APPEND __default_cxx_compiler_options
                -Wno-ignored-attributes)
        endif()
        list(APPEND __default_cxx_compiler_options
            -Wno-missing-field-initializers
        )
    endif()
endif()

add_compile_options(${__default_cxx_compile_options})
unset(__default_cxx_compile_options)
