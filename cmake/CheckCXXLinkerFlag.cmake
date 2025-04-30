################################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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

set(check_cxx_linker_flag_patterns
        FAIL_REGEX "[Uu]nrecogni[sz]ed .*option"               # GNU, NAG
        FAIL_REGEX "switch .* is no longer supported"          # GNU
        FAIL_REGEX "unknown .*option"                          # Clang
        FAIL_REGEX "optimization flag .* not supported"        # Clang
        FAIL_REGEX "unknown argument ignored"                  # Clang (cl)
        FAIL_REGEX "ignoring unknown option"                   # MSVC, Intel
        FAIL_REGEX "warning D9002"                             # MSVC, any lang
        FAIL_REGEX "option.*not supported"                     # Intel
        FAIL_REGEX "invalid argument .*option"                 # Intel
        FAIL_REGEX "ignoring option .*argument required"       # Intel
        FAIL_REGEX "ignoring option .*argument is of wrong type" # Intel
        FAIL_REGEX "[Uu]nknown option"                         # HP
        FAIL_REGEX "[Ww]arning: [Oo]ption"                     # SunPro
        FAIL_REGEX "command option .* is not recognized"       # XL
        FAIL_REGEX "command option .* contains an incorrect subargument" # XL
        FAIL_REGEX "Option .* is not recognized.  Option will be ignored." # XL
        FAIL_REGEX "not supported in this configuration. ignored"       # AIX
        FAIL_REGEX "File with unknown suffix passed to linker" # PGI
        FAIL_REGEX "[Uu]nknown switch"                         # PGI
        FAIL_REGEX "WARNING: unknown flag:"                    # Open64
        FAIL_REGEX "Incorrect command line option:"            # Borland
        FAIL_REGEX "Warning: illegal option"                   # SunStudio 12
        FAIL_REGEX "[Ww]arning: Invalid suboption"             # Fujitsu
        FAIL_REGEX "An invalid option .* appears on the command line" # Cray
)

include (CheckCXXSourceCompiles)

function(check_cxx_linker_flag _flag _var)
    set (_source "int main() { return 0; }")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_flag}")
    check_cxx_source_compiles("${_source}" "_${_var}" ${check_cxx_linker_flag_patterns})
    set(${_var} "${_${_var}}" PARENT_SCOPE)
endfunction()
