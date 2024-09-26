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
# - Enable warning all for gcc/clang or use /W4 for visual studio

## Strict warning level
set(__msvc_cxx_compile_options /W4)

set(__default_cxx_compile_options
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

set(__clang_cxx_compile_options
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
    -Wno-shadow-uncaptured-local
    -Wno-global-constructors
    -Wno-reserved-identifier
    -Wno-zero-as-null-pointer-constant
    -Wno-ignored-attributes
    -Wno-deprecated
    -Wno-incompatible-pointer-types
    -Wno-old-style-cast
    -Wno-unknown-attributes
    -Wno-microsoft-cpp-macro
    -Wno-microsoft-enum-value
    -Wno-language-extension-token
    -Wno-c++11-narrowing
    -Wno-float-equal
    -Wno-redundant-parens
    -Wno-format-nonliteral
    -Wno-unused-template
    -Wno-comma
    -Wno-suggest-destructor-override
    -Wno-switch-enum
    -Wno-shift-sign-overflow
    -Wno-suggest-override
    -Wno-inconsistent-missing-destructor-override
    -Wno-cast-function-type
    -Wno-nonportable-system-include-path
    -Wno-incompatible-pointer-types
    -Wno-documentation
    -Wno-deprecated-builtins
    -Wno-enum-constexpr-conversion
    -Wno-unused-value
    -Wno-unused-parameter
    -Wno-missing-noreturn
    -Wno-tautological-constant-out-of-range-compare
    -Wno-c++20-extensions)
if(WIN32)
    list(APPEND __clang_cxx_compile_options
        -fdelayed-template-parsing
        -fms-extensions
        -fms-compatibility)
endif()

set(__gnu_cxx_compile_options
    -Wno-missing-field-initializers
)

add_compile_options(
    "$<$<CXX_COMPILER_ID:MSVC>:${__msvc_cxx_compile_options}>"
    "$<$<CXX_COMPILER_ID:Clang>:${__default_cxx_compile_options};${__clang_cxx_compile_options}>"
    "$<$<CXX_COMPILER_ID:GNU>:${__default_cxx_compile_options};${__gnu_cxx_compile_options}>"
)

unset(__msvc_cxx_compile_options)
unset(__default_cxx_compile_options)
unset(__gnu_cxx_compile_options)
unset(__clang_cxx_compile_options)
