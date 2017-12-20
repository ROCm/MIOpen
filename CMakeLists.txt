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
cmake_minimum_required( VERSION 2.8.12 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Default installation path
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/x86_64-w64-mingw32" CACHE PATH "")
else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

project ( MIOpen C CXX )

enable_testing()

find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)


# Strip symbols for release
if(NOT WIN32 AND NOT APPLE)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

rocm_setup_version(VERSION 1.2.0)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.3")
        message(FATAL_ERROR "MIOpen requires at least gcc 5.3")
    endif()
endif()

############################################################
# require C++11
add_compile_options(-std=gnu++14)

############################################################
# OPTION - MIOpen Backend
# - OpenCL
# - HCC
if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    set(MIOPEN_DEFAULT_BACKEND "HIP")
else()
    set(MIOPEN_DEFAULT_BACKEND "OpenCL")
endif()

if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(WINE_CMD "wine")
else()
    set(WINE_CMD)
endif()

option(ENABLE_HIP_WORKAROUNDS Off)
if(ENABLE_HIP_WORKAROUNDS)
    # Add these to CMAKE_PREFIX_PATH to workaround installation problems with hip and hcc
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip)
endif()

set( MIOPEN_BACKEND ${MIOPEN_DEFAULT_BACKEND} CACHE STRING
    "Which of MIOpens's backends to use?" )
set_property( CACHE MIOPEN_BACKEND PROPERTY STRINGS
    OpenCL HIP HIPOC )
# OpenCL 1.2
if( MIOPEN_BACKEND STREQUAL "OpenCL")
    set(MIOPEN_BACKEND_OPENCL 1)
    find_package( OpenCL REQUIRED )
endif()

# HIP
if( MIOPEN_BACKEND STREQUAL "HIP" OR MIOPEN_BACKEND STREQUAL "HIPOC")
    set(MIOPEN_BACKEND_HIP 1)
    find_program(HIP_OC_COMPILER clang-ocl
        PATH_SUFFIXES bin
        PATHS /opt/rocm
    )
    if(HIP_OC_COMPILER)
        message(STATUS "hip compiler: ${HIP_OC_COMPILER}")
        set(HIP_OC_COMPILER "${HIP_OC_COMPILER}")
    else()
        message(FATAL_ERROR "clang-ocl not found")
    endif()
    
    find_package(hip REQUIRED PATHS /opt/rocm)
    link_libraries(stdc++)
    # A hack to make this work without the device enumerator
    link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
endif()
message( STATUS "${MIOPEN_BACKEND} backend selected." )

# Online assembler
find_program(MIOPEN_AMDGCN_ASSEMBLER
    NAMES clang
    PATHS ${MIOPEN_AMDGCN_ASSEMBLER_PATH} /opt/rocm
    PATH_SUFFIXES /opencl/bin/x86_64
    NO_DEFAULT_PATH
)
message(STATUS "AMDGCN assembler: ${MIOPEN_AMDGCN_ASSEMBLER}")

# miopengemm
find_package(miopengemm PATHS /opt/rocm)
if(miopengemm_FOUND)
    message(STATUS "Build with miopengemm")
    set(MIOPEN_USE_MIOPENGEMM 1)
else()
    message(STATUS "Build without miopengemm")
    set(MIOPEN_USE_MIOPENGEMM 0)
endif()

find_package(OpenSSL REQUIRED)
set(BOOST_COMPONENTS filesystem system)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

option( BUILD_DEV "Build for development only" OFF)

set( MIOPEN_INSTALL_DIR miopen)
set( DATA_INSTALL_DIR ${MIOPEN_INSTALL_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/miopen )

if(BUILD_DEV)
    set(MIOPEN_BUILD_DEV 1)
    set(MIOPEN_DB_PATH "${CMAKE_SOURCE_DIR}/src/kernels")
    set(MIOPEN_CACHE_DIR "" CACHE STRING "")
else()
    set(MIOPEN_BUILD_DEV 0)
    set(MIOPEN_DB_PATH "${CMAKE_INSTALL_PREFIX}/${DATA_INSTALL_DIR}/db" CACHE PATH "Default path to search for db")
    set(MIOPEN_CACHE_DIR "~/.cache/miopen/" CACHE STRING "")
endif()

set(CPACK_DEBIAN_PACKAGE_DEPENDS "openssl, rocm-opencl-dev, rocm-utils, hip_hcc, miopengemm")
set(CPACK_RPM_PACKAGE_REQUIRES "openssl, rocm-opencl-dev, rocm-utils, hip_hcc, miopengemm")

rocm_create_package(
    NAME MIOpen-${MIOPEN_BACKEND}
    DESCRIPTION "AMD's DNN Library"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
    # DEPENDS rocm-opencl rocm-utils hip_hcc tinygemm
)

include(EnableCompilerWarnings)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    set(CLANG_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
endif()
include(ClangTidy)
enable_clang_tidy(
    CHECKS 
        *
        -cert-env33-c
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -cert-msc50-cpp
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -google-explicit-constructor
        -google-readability-braces-around-statements
        -google-readability-todo
        -google-runtime-int
        -google-runtime-references
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-no-array-decay
        # We really shouldn't use bitwise operators with signed integers, but
        # opencl leaves us no choice
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-use-equals-default
        -hicpp-use-override
        -llvm-header-guard
        -llvm-include-order
        -misc-macro-parentheses
        -misc-misplaced-const
        -misc-misplaced-widening-cast
        -modernize-pass-by-value
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-else-after-return
        -readability-implicit-bool-cast
        -readability-implicit-bool-conversion
        -readability-misleading-indentation
        -readability-named-parameter
        ${CLANG_TIDY_CHECKS}
    ERRORS
        *
        -readability-inconsistent-declaration-parameter-name
    HEADER_FILTER
        "hpp$"
    EXTRA_ARGS
        -DMIOPEN_USE_CLANG_TIDY
    ANALYZE_TEMPORARY_DTORS ON

)
include(CppCheck)
enable_cppcheck(
    CHECKS 
        all
    FORCE
    SUPPRESS 
        ConfigurationNotChecked
        unmatchedSuppression
        constStatement
        noExplicitConstructor
        passedByValue
        # preprocessorErrorDirective
        unusedStructMember
        # TODO: Try to reenable this without needing inline suppressions
        leakReturnValNotUsed
        *:*.cl
    SOURCES
        include/
        src/
        # test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    DEFINE
        MIOPEN_USE_MIOPENGEMM=1
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(addkernels)
add_subdirectory(doc)
add_subdirectory(src)
add_subdirectory(driver)
add_subdirectory(test)
