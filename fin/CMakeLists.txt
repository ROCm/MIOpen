################################################################################
#
# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc.
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
cmake_minimum_required( VERSION 3.5 )

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

project ( fin C CXX )

find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)

option( BUILD_DEV "Build for development only" OFF)

rocm_setup_version(VERSION 1.0.0)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
include(TargetFlags)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.3")
        message(FATAL_ERROR "MIOpen requires at least gcc 5.3")
    endif()
endif()

############################################################
# require C++14
add_compile_options(-std=c++14)

############################################################
# OPTION - MIOpen Backend
# - OpenCL
# - HCC
check_cxx_compiler_flag("--cuda-host-only -x hip" HAS_HIP)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*hipcc" OR HAS_HIP)
    set(FIN_DEFAULT_BACKEND "HIP")
else()
    set(FIN_DEFAULT_BACKEND "OpenCL")
endif()

if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(WINE_CMD "wine")
else()
    set(WINE_CMD)
endif()

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hcc /opt/rocm/hip)

set( FIN_BACKEND ${FIN_DEFAULT_BACKEND} CACHE STRING
    "Which of MIOpens's backends to use?" )
set_property( CACHE FIN_BACKEND PROPERTY STRINGS
    OpenCL HIP HIPOC )

# HIP is always required
find_package(hip REQUIRED PATHS /opt/rocm)

# OpenCL 1.2
if( FIN_BACKEND STREQUAL "OpenCL")
    message(STATUS "OpenCL Backend")
    set(FIN_BACKEND_OPENCL 1)
    find_package( OpenCL REQUIRED )
    find_program(FIN_HIP_COMPILER hcc
        PATH_SUFFIXES bin
        PATHS /opt/rocm
    )
    find_package(miopengemm PATHS /opt/rocm)
endif()

# HIP
if( FIN_BACKEND STREQUAL "HIP" OR FIN_BACKEND STREQUAL "HIPOC")
    message(STATUS "HIP Backend")
    set(FIN_BACKEND_HIP 1)
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
    
    if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
        link_libraries(stdc++)

        # A hack to make this work without the device enumerator
        link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
        CHECK_CXX_COMPILER_FLAG("-amdgpu-target=gfx906 -Werror" GFX_906_FLAG_VALID)
        if(GFX_906_FLAG_VALID)
            link_libraries(-amdgpu-target=gfx906)
        endif()

        CHECK_CXX_COMPILER_FLAG(-amdgpu-target=gfx908 -Werror, GFX_908_FLAG_VALID)
        if(GFX_908_FLAG_VALID)
            link_libraries(-amdgpu-target=gfx908)
        endif()
        # end hack
    else()
        # Hcc's clang always defines __HCC__ even when not using hcc driver
        add_definitions(-U__HCC__)
    endif()
    # end hack

    set(MIOPEN_HIP_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")

endif()
message( STATUS "${FIN_BACKEND} backend selected." )

# find_package for miopen is broken on the OpenCL backend
# look for and register extractkernel
find_program(EXTRACTKERNEL_BIN extractkernel
    PATH_SUFFIXES bin
    PATHS /opt/rocm
)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    if(EXTRACTKERNEL_BIN)
        message(STATUS "extractkernel found: ${EXTRACTKERNEL_BIN}")
        set(EXTRACTKERNEL_BIN "${EXTRACTKERNEL_BIN}")
    else()
        message(FATAL_ERROR "extractkernel not found")
    endif()
endif()

option(Boost_USE_STATIC_LIBS "Use boost static libraries" ON)
set(BOOST_COMPONENTS filesystem system)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

find_path(HALF_INCLUDE_DIR half.hpp)

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

set(MIOPEN_PACKAGE_REQS "rocm-utils, hip-hcc")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "${MIOPEN_PACKAGE_REQS}, rocm-opencl-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "${MIOPEN_PACKAGE_REQS}, rocm-opencl-devel")

rocm_create_package(
    NAME fin-${FIN_BACKEND}
    DESCRIPTION "Fin MIOpen Driver"
    MAINTAINER "Tuna Devs <dl.MITunaX@amd.com>"
    LDCONFIG
    # DEPENDS rocm-opencl rocm-utils hip-hcc tinygemm
)

include(EnableCompilerWarnings)
set(MIOPEN_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    set(MIOPEN_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
# Enable tidy on hip
elseif(FIN_BACKEND STREQUAL "HIP")
    set(MIOPEN_TIDY_ERRORS ALL)
endif()

if(NOT MIOPEN_ENABLE_FIN)
    message("MIOPEN_ENABLE_FIN is NOT SET")
include(ClangTidy)
enable_clang_tidy(
    CHECKS
        *
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -bugprone-exception-escape
        -cert-env33-c
        -cert-msc32-c
        -cert-msc50-cpp
        -cert-msc51-cpp
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-avoid-c-arrays
        -cppcoreguidelines-avoid-magic-numbers
        -cppcoreguidelines-explicit-virtual-functions
        -cppcoreguidelines-init-variables
        -cppcoreguidelines-macro-usage
        -cppcoreguidelines-non-private-member-variables-in-classes
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -fuchsia-*
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
        -hicpp-avoid-c-arrays
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-uppercase-literal-suffix
        -hicpp-use-auto
        -hicpp-use-equals-default
        -hicpp-use-override
        -llvm-header-guard
        -llvm-include-order
        -misc-misplaced-const
        -misc-non-private-member-variables-in-classes
        -modernize-avoid-c-arrays
        -modernize-pass-by-value
        -modernize-use-auto
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-trailing-return-type
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-else-after-return
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-uppercase-literal-suffix
        -readability-convert-member-functions-to-static

        ${MIOPEN_TIDY_CHECKS}
    ${MIOPEN_TIDY_ERRORS}
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -DMIOPEN_USE_CLANG_TIDY

)
include(CppCheck)
enable_cppcheck(
    CHECKS
        warning
        style
        performance
        portability
    SUPPRESS
        ConfigurationNotChecked
        constStatement
        duplicateCondition
        noExplicitConstructor
        passedByValue
        # preprocessorErrorDirective
        shadowVariable
        unusedFunction
        unusedPrivateFunction
        unusedStructMember
        # Ignore initializer lists in the tests
        useInitializationList:*test/*.cpp
        *:*.cl
        *:*src/kernels/*.h
        knownConditionTrueFalse:*src/kernels/composable_kernel/*/*
        redundantAssignment:*src/kernels/composable_kernel/*/*
        unreadVariable:*src/kernels/composable_kernel/*/*
        unusedScopedObject:*src/kernels/composable_kernel/*/*
        wrongPrintfScanfArgNum:*src/kernels/composable_kernel/*/*
        unmatchedSuppression
    FORCE
    SOURCES
        addkernels/
        # driver/
        include/
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    DEFINE
        CPPCHECK=1
        MIOPEN_USE_MIOPENGEMM=1
        __linux__=1
)
else()
    message("MIOPEN_ENABLE_FIN is SET")
endif()

find_package(miopen)
find_package(rocblas) #MIOpen depends on rocBlas
find_package(Threads REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)
