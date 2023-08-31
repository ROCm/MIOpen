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

# Try to find the zstd library
#
# If successful, the following variables will be defined:
# zstd_INCLUDE_DIR
# zstd_LIBRARY
# zstd_STATIC_LIBRARY
# zstd_FOUND
#
# Additionally, one of the following import targets will be defined:
# zstd::libzstd_shared
# zstd::libzstd_static

if(MSVC)
  set(zstd_STATIC_LIBRARY_SUFFIX "_static\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
else()
  set(zstd_STATIC_LIBRARY_SUFFIX "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
endif()

find_path(zstd_INCLUDE_DIR NAMES zstd.h)
find_library(zstd_LIBRARY NAMES zstd zstd_static)
find_library(zstd_STATIC_LIBRARY NAMES
  zstd_static
  "${CMAKE_STATIC_LIBRARY_PREFIX}zstd${CMAKE_STATIC_LIBRARY_SUFFIX}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    zstd DEFAULT_MSG
    zstd_LIBRARY zstd_INCLUDE_DIR
)

if(zstd_FOUND)
  if(zstd_LIBRARY MATCHES "${zstd_STATIC_LIBRARY_SUFFIX}$")
    set(zstd_STATIC_LIBRARY "${zstd_LIBRARY}")
  elseif (NOT TARGET zstd::libzstd_shared)
    add_library(zstd::libzstd_shared SHARED IMPORTED)
    if(MSVC)
      # IMPORTED_LOCATION is the path to the DLL and IMPORTED_IMPLIB is the "library".
      get_filename_component(zstd_DIRNAME "${zstd_LIBRARY}" DIRECTORY)
      string(REGEX REPLACE "${CMAKE_INSTALL_LIBDIR}$" "${CMAKE_INSTALL_BINDIR}" zstd_DIRNAME "${zstd_DIRNAME}")
      get_filename_component(zstd_BASENAME "${zstd_LIBRARY}" NAME)
      string(REGEX REPLACE "\\${CMAKE_LINK_LIBRARY_SUFFIX}$" "${CMAKE_SHARED_LIBRARY_SUFFIX}" zstd_BASENAME "${zstd_BASENAME}")
      set_target_properties(zstd::libzstd_shared PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
          IMPORTED_LOCATION "${zstd_DIRNAME}/${zstd_BASENAME}"
          IMPORTED_IMPLIB "${zstd_LIBRARY}")
      unset(zstd_DIRNAME)
      unset(zstd_BASENAME)
    else()
      set_target_properties(zstd::libzstd_shared PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
          IMPORTED_LOCATION "${zstd_LIBRARY}")
    endif()
  endif()
  if(zstd_STATIC_LIBRARY MATCHES "${zstd_STATIC_LIBRARY_SUFFIX}$" AND
     NOT TARGET zstd::libzstd_static)
    add_library(zstd::libzstd_static STATIC IMPORTED)
    set_target_properties(zstd::libzstd_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
        IMPORTED_LOCATION "${zstd_STATIC_LIBRARY}")
  endif()
endif()

unset(zstd_STATIC_LIBRARY_SUFFIX)

mark_as_advanced(zstd_INCLUDE_DIR zstd_LIBRARY zstd_STATIC_LIBRARY)
