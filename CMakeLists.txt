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
cmake_minimum_required( VERSION 3.5 )

macro(set_var_to_condition var)
    if(${ARGN})
        set(${var} TRUE)
    else()
        set(${name} FALSE)
    endif()
endmacro()

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

find_package(ROCM 0.7.3 REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)
include(ROCMHeaderWrapper)


set(MIOPEN_ENABLE_SQLITE On CACHE BOOL "")
# Use SQLITE for compiled kernels, when turned off this will use raw files
set(MIOPEN_ENABLE_SQLITE_KERN_CACHE On CACHE BOOL "")
if(MIOPEN_ENABLE_SQLITE)
    # MIOpen now depends on SQLite as well
    find_package(PkgConfig)
    pkg_check_modules(SQLITE3 REQUIRED sqlite3)
endif()
find_package(BZip2)
if(MIOPEN_ENABLE_SQLITE_KERN_CACHE AND NOT MIOPEN_ENABLE_SQLITE)
    message(FATAL_ERROR "MIOPEN_ENABLE_SQLITE_KERN_CACHE requires MIOPEN_ENABLE_SQLITE")
endif()
set(MIOPEN_LOG_FUNC_TIME_ENABLE Off CACHE BOOL "")
set(MIOPEN_ENABLE_SQLITE_BACKOFF On CACHE BOOL "")

option( BUILD_DEV "Build for development only" OFF)
option(MIOPEN_ENABLE_FIN "Enable the fin driver for MIOpen"  OFF)


# Strip symbols for release
if(NOT WIN32 AND NOT APPLE)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

rocm_setup_version(VERSION 2.18.0)

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
# - HIP
check_cxx_compiler_flag("--cuda-host-only -x hip" HAS_HIP)
if(HAS_HIP)
    set(MIOPEN_DEFAULT_BACKEND "HIP")
else()
    set(MIOPEN_DEFAULT_BACKEND "OpenCL")
endif()

if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(WINE_CMD "wine")
else()
    set(WINE_CMD)
endif()

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip)

option(ENABLE_HIP_WORKAROUNDS Off)
set(MIOPEN_INSTALL_CXX_HEADERS Off CACHE BOOL "Install MIOpen's C++ header interface")


# Embedded Build Configuration
set(MIOPEN_EMBED_DB "" CACHE STRING "Semi-colon separated list of architecture to embed on-disk DBs in the binary. Example gfx906_60;gfx900_56")
if(NOT MIOPEN_EMBED_DB STREQUAL "")
    option(MIOPEN_DISABLE_SYSDB  "Disable sys database access" Off)
else()
    option(MIOPEN_DISABLE_SYSDB  "Disable sys database access" ${MIOPEN_EMBED_BUILD})
endif()
set(MIOPEN_BINCACHE_PATH "" CACHE STRING "URL or path containing binary cache files to embed")
option(MIOPEN_EMBED_BUILD "Build with the set of embed flags." Off)
option(MIOPEN_DISABLE_USERDB "Disable user database access" ${MIOPEN_EMBED_BUILD})


# MIOPEN_USE_HIP_KERNELS is a Workaround for COMgr issues
if(MIOPEN_EMBED_BUILD)
    option(BUILD_SHARED_LIBS "Build as a shared library" Off)
    option(MIOPEN_USE_HIP_KERNELS "Use HIP kernels." Off)
    option(MIOPEN_BUILD_DRIVER "Build MIOpenDriver" Off)
else()
    option(BUILD_SHARED_LIBS "Build as a shared library" ON)
    option(MIOPEN_USE_HIP_KERNELS "Use HIP kernels." On)
    option(MIOPEN_BUILD_DRIVER "Build MIOpenDriver" On)
endif()


set( MIOPEN_BACKEND ${MIOPEN_DEFAULT_BACKEND} CACHE STRING
    "Which of MIOpens's backends to use?" )
set_property( CACHE MIOPEN_BACKEND PROPERTY STRINGS
    OpenCL HIP HIPOC HIPNOGPU)

# OpenCL 1.2
if( MIOPEN_BACKEND STREQUAL "OpenCL")
    set(MIOPEN_BACKEND_OPENCL 1)
    find_package( OpenCL REQUIRED )
    set(MIOPEN_USE_MIOPENGEMM ON CACHE BOOL "")
    find_program(MIOPEN_HIP_COMPILER clang++
        PATH_SUFFIXES bin
        PATHS
            /opt/rocm/llvm
            ${CMAKE_INSTALL_PREFIX}/llvm
    )
    if(MIOPEN_HIP_COMPILER)
        message(STATUS "hip compiler: ${MIOPEN_HIP_COMPILER}")
    else()
        message(FATAL_ERROR "hip compiler not found")
    endif()

    # MIOpenTensile does not support opencl backend yet.
    set(MIOPEN_USE_MIOPENTENSILE OFF CACHE BOOL "")

    # TODO (priority_low) Use to build HIP and ASM kernels.
    if(MIOPEN_USE_COMGR)
        message(FATAL_ERROR "comgr cannot be used with OpenCL backend")
    endif()

    # This is to pass all necessary build flags to HIP compiler
    # for device code compilation. Used within "find_package(hip...".
    # See https://github.com/ROCm-Developer-Tools/HIP/pull/2035#issuecomment-616861118.
    set (HIP_CXX_COMPILER ${MIOPEN_HIP_COMPILER})
endif()


# HIP is always required
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")
target_flags(HIP_COMPILER_FLAGS hip::device)
# Remove cuda arch flags
string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REGEX REPLACE --offload-arch=[a-z0-9:+-]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
# Skip library paths since hip will incorrectly treat it as a source file
string(APPEND HIP_COMPILER_FLAGS " ")
foreach(_unused RANGE 2)
    string(REGEX REPLACE " /[^ ]+\\.(a|so) " " " HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
endforeach()

# Override HIP version in config.h, if necessary.
# The variables set by find_package() can't be overwritten,
# therefore let's use intermediate variables.
set(MIOPEN_hip_VERSION_MAJOR "${hip_VERSION_MAJOR}")
set(MIOPEN_hip_VERSION_MINOR "${hip_VERSION_MINOR}")
set(MIOPEN_hip_VERSION_PATCH "${hip_VERSION_PATCH}")
if( DEFINED MIOPEN_OVERRIDE_HIP_VERSION_MAJOR )
    set(MIOPEN_hip_VERSION_MAJOR "${MIOPEN_OVERRIDE_HIP_VERSION_MAJOR}")
    message(STATUS "MIOPEN_hip_VERSION_MAJOR overriden with ${MIOPEN_OVERRIDE_HIP_VERSION_MAJOR}")
endif()
if( DEFINED MIOPEN_OVERRIDE_HIP_VERSION_MINOR )
    set(MIOPEN_hip_VERSION_MINOR "${MIOPEN_OVERRIDE_HIP_VERSION_MINOR}")
    message(STATUS "MIOPEN_hip_VERSION_MINOR overriden with ${MIOPEN_OVERRIDE_HIP_VERSION_MINOR}")
endif()
if( DEFINED MIOPEN_OVERRIDE_HIP_VERSION_PATCH )
    set(MIOPEN_hip_VERSION_PATCH "${MIOPEN_OVERRIDE_HIP_VERSION_PATCH}")
    message(STATUS "MIOPEN_hip_VERSION_PATCH overriden with ${MIOPEN_OVERRIDE_HIP_VERSION_PATCH}")
endif()

set_var_to_condition(MIOPEN_USE_COMGR_DEFAULT (NOT DEFINED MIOPEN_BACKEND_OPENCL) AND (NOT (MIOPEN_BACKEND STREQUAL "HIPNOGPU")))
option(MIOPEN_USE_COMGR "Use comgr to build kernels instead of offline tools" ${MIOPEN_USE_COMGR_DEFAULT})

# TODO: Unify MIOPEN_HIP_VERSION_FLAT in config.h with this.
# in config.h we reserve 6 digits for patch, but here only 5 digits reserved to avoid overflow.
# Actually 5 digits should be Ok everywhere.
math(EXPR MIOPEN_hip_VERSION_FLAT "(${MIOPEN_hip_VERSION_MAJOR} * 1000 + ${MIOPEN_hip_VERSION_MINOR}) * 100000 + ${MIOPEN_hip_VERSION_PATCH}")

# Do not enable HIPRTC by default for older ROCm versions in order to avoid
# build time errors, because HIPRTC is a relatively new component.
set_var_to_condition(MIOPEN_USE_HIPRTC_DEFAULT ${MIOPEN_USE_COMGR} AND (${MIOPEN_hip_VERSION_FLAT} GREATER 500000000))
option(MIOPEN_USE_HIPRTC "Use HIPRTC to build HIP kernels instead of COMGR" ${MIOPEN_USE_HIPRTC_DEFAULT})

message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")

add_definitions("-DHIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}")


# HIP
if( MIOPEN_BACKEND STREQUAL "HIP" OR MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIPNOGPU")
    if( MIOPEN_BACKEND STREQUAL "HIPNOGPU")
        set(MIOPEN_MODE_NOGPU 1)
    endif()
    set(MIOPEN_BACKEND_HIP 1)
    set(MIOPEN_USE_MIOPENGEMM OFF CACHE BOOL "")
    # miopentensile default off
    set(MIOPEN_USE_MIOPENTENSILE OFF CACHE BOOL "")

    find_program(HIP_OC_COMPILER clang-ocl
        PATH_SUFFIXES bin
        PATHS
            /opt/rocm
            ${CMAKE_INSTALL_PREFIX}
    )
    if(HIP_OC_COMPILER)
        message(STATUS "OpenCL compiler: ${HIP_OC_COMPILER}")
        set(HIP_OC_COMPILER "${HIP_OC_COMPILER}")
    else()
        message(STATUS "OpenCL compiler not found")
    endif()

        # Hcc's clang always defines __HCC__ even when not using hcc driver
        add_definitions(-U__HCC__)

    set(MIOPEN_HIP_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")

    # rocblas
    set(MIOPEN_USE_ROCBLAS ON CACHE BOOL "")
    if(MIOPEN_USE_ROCBLAS)
        find_package(rocblas REQUIRED PATHS /opt/rocm)
        message(STATUS "Build with rocblas")
    else()
        message(STATUS "Build without rocblas")
    endif()
else()
    if(MIOPEN_USE_HIPRTC)
        message(FATAL_ERROR "HIPRTC cannot be used without HIP backend")
    endif()
endif()
message( STATUS "${MIOPEN_BACKEND} backend selected." )

# look for and register clang-offload-bundler
if(MIOPEN_HIP_COMPILER MATCHES ".*clang\\+\\+$")
    find_program(MIOPEN_OFFLOADBUNDLER_BIN clang-offload-bundler
        PATH_SUFFIXES bin
        PATHS
            /opt/rocm/llvm
            ${CMAKE_INSTALL_PREFIX}/llvm
    )
endif()
if(MIOPEN_OFFLOADBUNDLER_BIN)
    message(STATUS "clang-offload-bundler found: ${MIOPEN_OFFLOADBUNDLER_BIN}")
    set(MIOPEN_OFFLOADBUNDLER_BIN "${MIOPEN_OFFLOADBUNDLER_BIN}")
else()
    # look for and register extractkernel
    message(STATUS "clang-offload-bundler not found")

    find_program(EXTRACTKERNEL_BIN extractkernel
        PATH_SUFFIXES bin
        PATHS
            /opt/rocm/hip
            /opt/rocm
            ${CMAKE_INSTALL_PREFIX}/hip
            ${CMAKE_INSTALL_PREFIX}
    )
    if(EXTRACTKERNEL_BIN)
        message(STATUS "extractkernel found: ${EXTRACTKERNEL_BIN}")
        set(EXTRACTKERNEL_BIN "${EXTRACTKERNEL_BIN}")
    else()
        message(FATAL_ERROR "extractkernel not found")
    endif()
endif()

set_var_to_condition(MIOPEN_USE_MLIR_DEFAULT NOT (NOT ${BUILD_SHARED_LIBS} AND ${MIOPEN_USE_COMGR}))
option(MIOPEN_USE_MLIR "Use MLIR compilation backend" ${MIOPEN_USE_MLIR_DEFAULT})

if(MIOPEN_USE_MLIR)
    # REQUIRED is not supported before cmake 3.18
    find_library(LIBMLIRMIOPEN MLIRMIOpen REQUIRED)
    if(NOT ${BUILD_SHARED_LIBS} AND ${MIOPEN_USE_COMGR})
        message(FATAL_ERROR "Potential symbol conflict between mlir and comgr in static build")
    endif()

    if(NOT LIBMLIRMIOPEN)
        message(FATAL_ERROR "libMLIRMIOpen not found, please reinstall dependencies. \
        Refer to https://github.com/ROCmSoftwarePlatform/MIOpen#installing-the-dependencies")
    else()
        message(STATUS "Build with libMLIRMIOpen: " ${LIBMLIRMIOPEN})
    endif()
endif()


# Online assembler
find_program(MIOPEN_AMDGCN_ASSEMBLER
    NAMES clang
    PATHS
        ${MIOPEN_AMDGCN_ASSEMBLER_PATH}
        /opt/rocm
        /opt/rocm/llvm
        ${CMAKE_INSTALL_PREFIX}
        ${CMAKE_INSTALL_PREFIX}/llvm
    PATH_SUFFIXES
        /opencl/bin/x86_64
        /opencl/bin
        /bin
)
message(STATUS "AMDGCN assembler: ${MIOPEN_AMDGCN_ASSEMBLER}")

# miopentensile
if(MIOPEN_USE_MIOPENTENSILE)
    find_package(miopentensile)
    message(STATUS "Build with miopentensile")
else()
    message(STATUS "Build without miopentensile")
endif()

# miopengemm
if(MIOPEN_USE_MIOPENGEMM)
    find_package(miopengemm PATHS /opt/rocm)
    if(NOT miopengemm_FOUND)
        set(MIOPEN_USE_MIOPENGEMM 0)
    endif()
endif()

if(MIOPEN_USE_MIOPENGEMM)
    message(STATUS "Build with miopengemm")
else()
    if(MIOPEN_BACKEND_OPENCL)
        message(FATAL_ERROR "MIOpen OpenCL backend requires MIOpenGEMM")
    endif()
    message(STATUS "Build without miopengemm")
endif()

if(MIOPEN_USE_COMGR)
    find_package(amd_comgr REQUIRED CONFIG)
    message(STATUS "Build with comgr ${amd_comgr_VERSION}")
endif()

if(MIOPEN_USE_HIPRTC)
    if(NOT MIOPEN_USE_COMGR)
        message(FATAL_ERROR "HIPRTC can be used only together with COMGR")
    else()
        message(STATUS "Build with HIPRTC")
    endif()
endif()

option(Boost_USE_STATIC_LIBS "Use boost static libraries" ON)
set(BOOST_COMPONENTS filesystem)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

find_path(HALF_INCLUDE_DIR half.hpp)
message(STATUS "HALF_INCLUDE_DIR: ${HALF_INCLUDE_DIR}")

option( MIOPEN_DEBUG_FIND_DB_CACHING "Use system find-db caching" ON)

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" ON)

set( MIOPEN_INSTALL_DIR miopen)
set( DATA_INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}/miopen )

set(MIOPEN_GPU_SYNC Off CACHE BOOL "")
if(BUILD_DEV)
    set(MIOPEN_BUILD_DEV 1)
    set(MIOPEN_SYSTEM_DB_PATH "${CMAKE_SOURCE_DIR}/src/kernels" CACHE PATH "Default path of system db files")
    set(MIOPEN_USER_DB_PATH "${CMAKE_SOURCE_DIR}/src/kernels" CACHE PATH "Default path of user db files")
    set(MIOPEN_USER_DB_SUFFIX "${MIOPEN_BACKEND}.${MIOpen_VERSION_MAJOR}_${MIOpen_VERSION_MINOR}_${MIOpen_VERSION_PATCH}" CACHE PATH "Filename suffix for the user find-db files")
    set(MIOPEN_CACHE_DIR "" CACHE STRING "")
else()
    set(MIOPEN_BUILD_DEV 0)
    set(MIOPEN_USER_DB_PATH "~/.config/miopen/" CACHE STRING "Default path of user db files")
    set(MIOPEN_CACHE_DIR "~/.cache/miopen/" CACHE STRING "")
    set(MIOPEN_USER_DB_SUFFIX "${MIOPEN_BACKEND}.${MIOpen_VERSION_MAJOR}_${MIOpen_VERSION_MINOR}_${MIOpen_VERSION_PATCH}_${MIOpen_VERSION_TWEAK}" CACHE PATH "Filename suffix for the user find-db files")
endif()
set(MIOPEN_SYSTEM_FIND_DB_SUFFIX "${MIOPEN_BACKEND}" CACHE PATH "Filename suffix for the system find-db files")

set(MIOPEN_PACKAGE_REQS "hip-rocclr")

if(MIOPEN_USE_MIOPENGEMM)
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, miopengemm")
endif()

if(MIOPEN_USE_ROCBLAS)
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, rocblas")
endif()

if(MIOPEN_BACKEND STREQUAL "HIP")
    # In HIP backend, there is a posibility of COMGR disabled.
    # In this case add the clang-ocl as dependency for runtime kernel compilation
    if(NOT MIOPEN_USE_COMGR)
        set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, rocm-clang-ocl")
    endif()
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${MIOPEN_PACKAGE_REQS}")
    set(CPACK_RPM_PACKAGE_REQUIRES "${MIOPEN_PACKAGE_REQS}")

    # Make backends explicitly conflict
    set(CPACK_DEBIAN_PACKAGE_CONFLICTS miopen-opencl)
    set(CPACK_RPM_PACKAGE_CONFLICTS miopen-opencl)

elseif(MIOPEN_BACKEND STREQUAL "OpenCL")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${MIOPEN_PACKAGE_REQS}, rocm-opencl-dev")
    set(CPACK_RPM_PACKAGE_REQUIRES "${MIOPEN_PACKAGE_REQS}, rocm-opencl-devel")

    # Make backends explicitly conflict
    set(CPACK_DEBIAN_PACKAGE_CONFLICTS miopen-hip)
    set(CPACK_RPM_PACKAGE_CONFLICTS miopen-hip)
endif()

rocm_create_package(
    NAME MIOpen-${MIOPEN_BACKEND}
    DESCRIPTION "AMD's DNN Library"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
    # DEPENDS rocm-opencl hip-rocclr tinygemm
)

include(EnableCompilerWarnings)
set(MIOPEN_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    set(MIOPEN_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
# Enable tidy on hip
elseif(MIOPEN_BACKEND STREQUAL "HIP" OR MIOPEN_BACKEND STREQUAL "HIPNOGPU")
    set(MIOPEN_TIDY_ERRORS ALL)

endif()

include(ClangTidy)
enable_clang_tidy(
    CHECKS
        *
        -abseil-*
        -altera-struct-pack-align
        -altera-unroll-loops
        -android-cloexec-fopen
        -bugprone-exception-escape
        -bugprone-macro-parentheses
        # too many narrowing conversions in our code
        -bugprone-narrowing-conversions
        -cert-dcl37-c
        -cert-dcl51-cpp
        -cert-env33-c
        # Yea we shouldn't be using rand()
        -cert-msc30-c
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
        -cppcoreguidelines-narrowing-conversions
        -cppcoreguidelines-non-private-member-variables-in-classes
        -cppcoreguidelines-prefer-member-initializer
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
        # We really shouldn't use bitwise operators with signed integers, but
        # opencl leaves us no choice
        -hicpp-avoid-c-arrays
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-named-parameter
        -hicpp-no-array-decay
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-uppercase-literal-suffix
        -hicpp-use-auto
        -hicpp-use-equals-default
        -hicpp-use-override
        -hicpp-vararg
        -llvm-else-after-return
        -llvm-header-guard
        -llvm-include-order
        -llvmlibc-callee-namespace
        -llvmlibc-implementation-in-namespace
        -llvmlibc-restrict-system-libc-headers
        -llvm-qualified-auto
        -misc-misplaced-const
        -misc-non-private-member-variables-in-classes
        -misc-no-recursion
        -modernize-avoid-bind
        -modernize-avoid-c-arrays
        -modernize-pass-by-value
        -modernize-use-auto
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-trailing-return-type
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-convert-member-functions-to-static
        -readability-else-after-return
        # we are not ready to use it, but very useful
        -readability-function-cognitive-complexity
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-qualified-auto
        -readability-redundant-string-init
        -readability-uppercase-literal-suffix
        #TODO Code Quality WORKAROUND ROCm 5.1 update
        -altera-id-dependent-backward-branch
        -bugprone-easily-swappable-parameters
        -bugprone-implicit-widening-of-multiplication-result
        -cert-err33-c
        -google-readability-casting
        -hicpp-use-emplace
        -modernize-use-emplace
        -performance-unnecessary-copy-initialization
        -readability-container-data-pointer
        -readability-identifier-length
        -readability-suspicious-call-argument

        ${MIOPEN_TIDY_CHECKS}
    ${MIOPEN_TIDY_ERRORS}
    HEADER_FILTER
        "\.hpp$"
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
        *:*src/sqlite/*.cpp
        *:*.cl
        *:*src/kernels/*.h
        knownConditionTrueFalse:*src/kernels/static_composable_kernel/*/*
        redundantAssignment:*src/kernels/static_composable_kernel/*/*
        unreadVariable:*src/kernels/static_composable_kernel/*/*
        unusedScopedObject:*src/kernels/static_composable_kernel/*/*
        wrongPrintfScanfArgNum:*src/kernels/static_composable_kernel/*/*
        knownConditionTrueFalse:*src/composable_kernel/composable_kernel/*/*
        identicalConditionAfterEarlyExit:*src/composable_kernel/composable_kernel/*/*
        duplicateExpression:*src/composable_kernel/composable_kernel/*/*
        multiCondition:*src/composable_kernel/composable_kernel/*/*
        unreadVariable:*src/composable_kernel/composable_kernel/*/*
        unreadVariable:*src/composable_kernel/host/*/*
        unreadVariable:*src/composable_kernel/external/*/*
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


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(addkernels)
add_subdirectory(doc)
add_subdirectory(src)
if(MIOPEN_BUILD_DRIVER)
    add_subdirectory(driver)
endif()
add_subdirectory(test)
add_subdirectory(speedtests)
add_subdirectory(utils)
if(MIOPEN_ENABLE_FIN)
add_subdirectory(fin)
endif()
