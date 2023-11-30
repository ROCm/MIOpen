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
cmake_minimum_required( VERSION 3.15 )

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

macro(set_var_to_condition var)
    if(${ARGN})
        set(${var} TRUE)
    else()
        set(${var} FALSE)
    endif()
endmacro()

get_property(MIOPEN_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.
if(MIOPEN_GENERATOR_IS_MULTI_CONFIG)
    if (NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING
            "Available build types (configurations) on multi-config generators")
    endif()
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
    endif()
endif()

# Default installation path
if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

project ( MIOpen C CXX )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

find_package(Threads REQUIRED)
find_package(ROCM 0.7.3 REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)
include(ROCMHeaderWrapper)

# Build library with Beta APIs
add_definitions("-DMIOPEN_BETA_API=1")

set(MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK On CACHE BOOL "Enable AI-based fallback for Immediate Mode")
set(MIOPEN_ENABLE_AI_KERNEL_TUNING On CACHE BOOL "Enable AI heuristic for kernel tuning")
set(MIOPEN_ENABLE_SQLITE On CACHE BOOL "")
# Use SQLITE for compiled kernels, when turned off this will use raw files
set(MIOPEN_ENABLE_SQLITE_KERN_CACHE On CACHE BOOL "")

# By default build shared libraries
option(BUILD_SHARED_LIBS "Create shared libraries" ON)
option(Boost_USE_STATIC_LIBS "Use boost static libraries" ON)
add_definitions(-DBOOST_ALL_NO_LIB=1)

if(WIN32)
    add_subdirectory(extern)
else()
    if(MIOPEN_ENABLE_SQLITE)
        # MIOpen now depends on SQLite as well
        find_package(SQLite3 REQUIRED)
    endif()
    find_package(BZip2 REQUIRED)
    find_package(nlohmann_json 3.9.1 REQUIRED)
    find_package(Boost 1.74 REQUIRED COMPONENTS filesystem system)
endif()
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

rocm_setup_version(VERSION 3.00.0)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
include(TargetFlags)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.3")
        message(FATAL_ERROR "MIOpen requires at least gcc 5.3")
    endif()
endif()

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
option(MIOPEN_EMBED_BINCACHE "Embed Binary Cache or KDB" Off)
option(MIOPEN_EMBED_BUILD "Build with the set of embed flags." Off)
option(MIOPEN_DISABLE_USERDB "Disable user database access" ${MIOPEN_EMBED_BUILD})

# MIOPEN_USE_HIP_KERNELS is a Workaround for COMgr issues
if(MIOPEN_EMBED_BUILD)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build as a shared library" FORCE)
    option(MIOPEN_USE_HIP_KERNELS "Use HIP kernels." Off)
    option(MIOPEN_BUILD_DRIVER "Build MIOpenDriver" Off)
else()
    option(MIOPEN_USE_HIP_KERNELS "Use HIP kernels." On)
    option(MIOPEN_BUILD_DRIVER "Build MIOpenDriver" On)
endif()

if(MIOPEN_EMBED_BUILD)
    if(MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK)
        message(FATAL_ERROR "AI-based fallback for Immediate Mode cannot be used \
        with database embedding")
    endif()
    if(MIOPEN_ENABLE_AI_KERNEL_TUNING)
        message(FATAL_ERROR "AI Kernel tuning cannot be used with database embedding")
    endif()
endif()

set( MIOPEN_BACKEND ${MIOPEN_DEFAULT_BACKEND} CACHE STRING
    "Which of MIOpens's backends to use?" )
set_property( CACHE MIOPEN_BACKEND PROPERTY STRINGS
    OpenCL HIP HIPOC HIPNOGPU)

# OpenCL 1.2
if( MIOPEN_BACKEND STREQUAL "OpenCL")
    set(MIOPEN_BACKEND_OPENCL 1)
    find_package( OpenCL REQUIRED )
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

# Depend on Composable Kernels
option(MIOPEN_USE_COMPOSABLEKERNEL "Enable MIOpen to use composable kernels for various operations" On)
if(MIOPEN_BACKEND_OPENCL)
    set(MIOPEN_USE_COMPOSABLEKERNEL OFF)
endif()
message(STATUS "Enable Composable Kernels: ${MIOPEN_USE_COMPOSABLEKERNEL}")

set_var_to_condition(MIOPEN_USE_COMGR_DEFAULT (NOT DEFINED MIOPEN_BACKEND_OPENCL) AND (NOT (MIOPEN_BACKEND STREQUAL "HIPNOGPU")))
option(MIOPEN_USE_COMGR "Use comgr to build kernels instead of offline tools" ${MIOPEN_USE_COMGR_DEFAULT})

set(MIOPEN_hip_VERSION ${MIOPEN_hip_VERSION_MAJOR}.${MIOPEN_hip_VERSION_MINOR}.${MIOPEN_hip_VERSION_PATCH})

# Do not enable HIPRTC by default for older ROCm versions in order to avoid
# build time errors, because HIPRTC is a relatively new component.
set_var_to_condition(MIOPEN_USE_HIPRTC_DEFAULT (MIOPEN_USE_COMGR AND (MIOPEN_hip_VERSION VERSION_GREATER_EQUAL 5)))
option(MIOPEN_USE_HIPRTC "Use HIPRTC to build HIP kernels instead of COMGR" ${MIOPEN_USE_HIPRTC_DEFAULT})

# WORKAROUND_SWDEV_413293 - do not use on Windows. Compiler error message: unknown command line option
if(NOT WIN32 AND MIOPEN_hip_VERSION VERSION_GREATER_EQUAL 5.7.23302)
    string(APPEND HIP_COMPILER_FLAGS " -fno-offload-uniform-block ")
endif()

if(WIN32)
    string(REPLACE "\\" "/" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
endif()

message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")

add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:HIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}>)

# HIP
if( MIOPEN_BACKEND STREQUAL "HIP" OR MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIPNOGPU")
    if(MIOPEN_USE_COMPOSABLEKERNEL)
        find_package(composable_kernel 1.0.0 COMPONENTS device_other_operations device_gemm_operations device_conv_operations device_contraction_operations device_reduction_operations)
    endif()
    if( MIOPEN_BACKEND STREQUAL "HIPNOGPU")
        set(MIOPEN_MODE_NOGPU 1)
    endif()
    set(MIOPEN_BACKEND_HIP 1)

    find_program(HIP_OC_COMPILER clang
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
        message(STATUS "Build with rocblas ${rocblas_VERSION}")
    else()
        message(STATUS "Build without rocblas")
    endif()
else()
    #CK is only enabled when HIP backend is selected   
    set(MIOPEN_USE_COMPOSABLEKERNEL Off)
    if(MIOPEN_USE_HIPRTC)
        message(FATAL_ERROR "HIPRTC cannot be used without HIP backend")
    endif()
endif()
message( STATUS "${MIOPEN_BACKEND} backend selected." )

# look for and register clang-offload-bundler
if(MIOPEN_HIP_COMPILER MATCHES ".*clang\\+\\+.*")
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
    message(STATUS "clang-offload-bundler not found")
endif()

set_var_to_condition(MIOPEN_USE_MLIR_DEFAULT NOT (NOT ${BUILD_SHARED_LIBS} AND ${MIOPEN_USE_COMGR}))
option(MIOPEN_USE_MLIR "Use MLIR compilation backend" ${MIOPEN_USE_MLIR_DEFAULT})

if(MIOPEN_USE_MLIR)
    if(NOT ${BUILD_SHARED_LIBS} AND ${MIOPEN_USE_COMGR})
        message(FATAL_ERROR "Potential symbol conflict between mlir and comgr in static build")
    endif()
    if(WIN32)
        # Windows does not support earlier ROCm versions hence no fallback to MLIRMIOpen.
        find_package(rocMLIR 1.0.0 CONFIG REQUIRED)
    else()
        # Try to find package rocMLIR
        # REQUIRED is omitted since we do not want cmake to abort if the package is not found
        find_package(rocMLIR 1.0.0 CONFIG)
    endif()
    if(NOT rocMLIR_FOUND)
        message(STATUS "Falling back to find library libMLIRMIOpen")
        # Backward compatibility with ROCm 5.3
        # If the rocMLIR package is not found, try to find the library libMLIRMIOpen directly
        find_library(LIBMLIRMIOPEN MLIRMIOpen REQUIRED)
        if(NOT LIBMLIRMIOPEN)
            message(FATAL_ERROR "library libMLIRMIOpen not found, please reinstall dependencies. \
            Refer to https://github.com/ROCmSoftwarePlatform/MIOpen#installing-the-dependencies")
        else()
            message(STATUS "Build with library libMLIRMIOpen: " ${LIBMLIRMIOPEN})
            set(rocMLIR_VERSION 0.0.1)
        endif()
    else()
        message(STATUS "Build with rocMLIR::rockCompiler ${rocMLIR_VERSION}")
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

if(MIOPEN_USE_COMGR)
    find_package(amd_comgr REQUIRED CONFIG)
    message(STATUS "Build with comgr ${amd_comgr_VERSION}")
endif()

if(MIOPEN_USE_HIPRTC)
    if(NOT MIOPEN_USE_COMGR)
        message(FATAL_ERROR "HIPRTC can be used only together with COMGR")
    endif()
    if(WIN32)
        find_package(hiprtc REQUIRED)
    endif()
    message(STATUS "Build with HIPRTC")
endif()

if(NOT WIN32)
    find_path(HALF_INCLUDE_DIR half/half.hpp)
endif()
message(STATUS "HALF_INCLUDE_DIR: ${HALF_INCLUDE_DIR}")

option( MIOPEN_DEBUG_FIND_DB_CACHING "Use system find-db caching" ON)

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)

set( MIOPEN_INSTALL_DIR miopen)
set( DATA_INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}/miopen )

if(WIN32)
    set(KERNELS_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)
else()
    set(KERNELS_BINARY_DIR ${PROJECT_BINARY_DIR}/share/miopen/db)
endif()

file(MAKE_DIRECTORY ${KERNELS_BINARY_DIR})

set(MIOPEN_GPU_SYNC Off CACHE BOOL "")
if(BUILD_DEV)
    set(MIOPEN_BUILD_DEV 1)
    set(MIOPEN_SYSTEM_DB_PATH "${KERNELS_BINARY_DIR}" CACHE PATH "Default path of system db files")
    set(MIOPEN_USER_DB_PATH "${KERNELS_BINARY_DIR}" CACHE PATH "Default path of user db files")
    set(MIOPEN_USER_DB_SUFFIX "${MIOPEN_BACKEND}.${MIOpen_VERSION_MAJOR}_${MIOpen_VERSION_MINOR}_${MIOpen_VERSION_PATCH}" CACHE PATH "Filename suffix for the user find-db files")
    set(MIOPEN_CACHE_DIR "" CACHE STRING "")
else()
    set(MIOPEN_BUILD_DEV 0)
    if(WIN32)
        set(MIOPEN_USER_DB_PATH "$USERPROFILE\\\\.miopen\\\\db\\\\" CACHE STRING "Default path to user db files")
        set(MIOPEN_CACHE_DIR "$USERPROFILE\\\\.miopen\\\\cache\\\\" CACHE STRING "")
    else()
        set(MIOPEN_USER_DB_PATH "~/.config/miopen/" CACHE STRING "Default path of user db files")
        set(MIOPEN_CACHE_DIR "~/.cache/miopen/" CACHE STRING "")
    endif()
    set(MIOPEN_USER_DB_SUFFIX "${MIOPEN_BACKEND}.${MIOpen_VERSION_MAJOR}_${MIOpen_VERSION_MINOR}_${MIOpen_VERSION_PATCH}_${MIOpen_VERSION_TWEAK}" CACHE PATH "Filename suffix for the user find-db files")
endif()
set(MIOPEN_SYSTEM_FIND_DB_SUFFIX "${MIOPEN_BACKEND}" CACHE PATH "Filename suffix for the system find-db files")

set(MIOPEN_PACKAGE_REQS "hip-rocclr")

if(MIOPEN_USE_ROCBLAS)
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, rocblas")
endif()

if(MIOPEN_BACKEND STREQUAL "HIP")
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

set(KERNELS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/kernels)

find_program(UNZIPPER lbzcat bzcat REQUIRED)

function(unpack_db db_bzip2_file)
    get_filename_component(__fname ${db_bzip2_file} NAME_WLE)
    add_custom_command(OUTPUT ${KERNELS_BINARY_DIR}/${__fname}
                       COMMAND ${UNZIPPER} -k ${db_bzip2_file} > ${KERNELS_BINARY_DIR}/${__fname})
    string(REPLACE "." "_" __tname ${__fname})
    add_custom_target(generate_${__tname} ALL DEPENDS ${KERNELS_BINARY_DIR}/${__fname})
endfunction()

file(GLOB PERF_DB_BZIP_FILES "${KERNELS_SOURCE_DIR}/*.db.bz2")
foreach(DB_BZIP_FILE ${PERF_DB_BZIP_FILES})
    unpack_db(${DB_BZIP_FILE})
endforeach()

file(GLOB FIND_DB_BZIP_FILES "${KERNELS_SOURCE_DIR}/*.fdb.txt.bz2")
foreach(DB_BZIP_FILE ${FIND_DB_BZIP_FILES})
    unpack_db(${DB_BZIP_FILE})
endforeach()

# Begin KDB package creation

function(install_kdb FILE_NAME COMPONENT_NAME)
    file(READ ${FILE_NAME} __contents LIMIT 7)
    get_filename_component(__fname ${FILE_NAME} NAME_WLE)
    if(__contents MATCHES "version")
        message(WARNING "GIT LFS file not pulled down, skipping: ${__fname}")
    else()
        unpack_db(${FILE_NAME})
        if( NOT ENABLE_ASAN_PACKAGING )
            rocm_install(FILES ${KERNELS_BINARY_DIR}/${__fname}
                         DESTINATION ${DATA_INSTALL_DIR}/db
                         COMPONENT ${COMPONENT_NAME})
        endif()
        set(CPACK_COMPONENTS_ALL
            "${CPACK_COMPONENTS_ALL};${__component}" PARENT_SCOPE)
    endif()
endfunction()

# Both the lists below should be in sync always
set(KDB_BZ2_FILES gfx90a.kdb.bz2 gfx1030.kdb.bz2 gfx908.kdb.bz2 gfx906.kdb.bz2 gfx900.kdb.bz2)
set(COMPONENT_LST gfx90akdb gfx1030kdb gfx908kdb gfx906kdb gfx900kdb)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    foreach(__file __component IN ZIP_LISTS KDB_BZ2_FILES COMPONENT_LST)
        install_kdb(${KERNELS_SOURCE_DIR}/${__file} ${__component})
    endforeach()
else()
    # TODO: Upgrade minimum CMake to version 3.17+ and use IN ZIP_LISTS instead
    list(LENGTH KDB_BZ2_FILES __length)
    math(EXPR __high "${__length} - 1")
    foreach(__index RANGE ${__high})
        list(GET COMPONENT_LST ${__index} __component)
        list(GET KDB_BZ2_FILES ${__index} __file)
        install_kdb(${KERNELS_SOURCE_DIR}/${__file} ${__component})
    endforeach()
endif()

# End KDB package creation

rocm_create_package(
    NAME MIOpen-${MIOPEN_BACKEND}
    DESCRIPTION "AMD's DNN Library"
    MAINTAINER "MIOpen Maintainer <miopen-lib.support@amd.com>"
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
        # There is no ODR violation because of using separate executables,
        # but cppcheck doesn't understand that as it assumes everything
        # will be compiled together in one binary.
        ctuOneDefinitionRuleViolation:*test/*
        ctuOneDefinitionRuleViolation:*src/composable_kernel/composable_kernel/*/*
        ctuOneDefinitionRuleViolation:*src/composable_kernel/host/*/*
        # There are many FPs with this, let's disable this (ditto in MIGraphX)
        ctuPointerArith:*test/*
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
        ###################################################################
        # TODO Code Quality WORKAROUND ROCm 5.3 &&
        # Ubuntu 22.04 && C++17 && cppcheck 2.9 update
        ###################################################################
        constParameter
        constVariable
        variableScope
        missingReturn
        cstyleCast
        uselessCallsSubstr
        uninitMemberVar
        overlappingWriteUnion
        operatorEqVarError
        returnTempReference
        objectIndex
        integerOverflowCond
        rethrowNoCurrentException
        mismatchingContainers
        unreadVariable
        CastIntegerToAddressAtReturn
        knownConditionTrueFalse
        shadowFunction
        moduloofone
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
        __linux__=1
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(addkernels)
add_subdirectory(src)
if(MIOPEN_BUILD_DRIVER)
    add_subdirectory(driver)
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
    add_subdirectory(speedtests)
endif()

add_subdirectory(utils)
if(MIOPEN_ENABLE_FIN)
    add_subdirectory(fin)
endif()
