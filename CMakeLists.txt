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
cmake_minimum_required( VERSION 3.11 )

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

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
if(MIOPEN_ENABLE_SQLITE)
    # MIOpen now depends on SQLite as well
    find_package(PkgConfig)
    pkg_check_modules(SQLITE3 REQUIRED sqlite3)
endif()
find_package(BZip2)
find_package(nlohmann_json 3.9.1 REQUIRED)
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
option(MIOPEN_EMBED_BINCACHE "Embed Binary Cache or KDB" Off)
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

# TODO: Unify MIOPEN_HIP_VERSION_FLAT in config.h with this.
# in config.h we reserve 6 digits for patch, but here only 5 digits reserved to avoid overflow.
# Actually 5 digits should be Ok everywhere.
math(EXPR MIOPEN_hip_VERSION_FLAT "(${MIOPEN_hip_VERSION_MAJOR} * 1000 + ${MIOPEN_hip_VERSION_MINOR}) * 100000 + ${MIOPEN_hip_VERSION_PATCH}")

# Do not enable HIPRTC by default for older ROCm versions in order to avoid
# build time errors, because HIPRTC is a relatively new component.
set_var_to_condition(MIOPEN_USE_HIPRTC_DEFAULT ${MIOPEN_USE_COMGR} AND (${MIOPEN_hip_VERSION_FLAT} GREATER 500000000))
option(MIOPEN_USE_HIPRTC "Use HIPRTC to build HIP kernels instead of COMGR" ${MIOPEN_USE_HIPRTC_DEFAULT})

# WORKAROUND_SWDEV_413293
if(HAS_HIP)
    check_cxx_compiler_flag("-x hip -fno-offload-uniform-block" MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK)
else()
    # CXX compiler is not HIP compiler, let's analyze HIP version.
    set(MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK Off)
    if(${MIOPEN_hip_VERSION_FLAT} GREATER_EQUAL 500723302)
        set(MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK On)
    endif()
    message(STATUS "MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK: ${MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK}")
endif()
if(MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK)
    string(APPEND HIP_COMPILER_FLAGS " -fno-offload-uniform-block ")
endif()

message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")

add_definitions("-DHIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}")


# HIP
if( MIOPEN_BACKEND STREQUAL "HIP" OR MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIPNOGPU")
    if(MIOPEN_USE_COMPOSABLEKERNEL)
        find_package(composable_kernel 1.0.0 COMPONENTS device_operations)
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
    message(STATUS "clang-offload-bundler not found")
endif()

set_var_to_condition(MIOPEN_USE_MLIR_DEFAULT NOT (NOT ${BUILD_SHARED_LIBS} AND ${MIOPEN_USE_COMGR}))
option(MIOPEN_USE_MLIR "Use MLIR compilation backend" ${MIOPEN_USE_MLIR_DEFAULT})

set(rocMLIR_VERSION_FLAT 0)
if(MIOPEN_USE_MLIR)
    if(NOT ${BUILD_SHARED_LIBS} AND ${MIOPEN_USE_COMGR})
        message(FATAL_ERROR "Potential symbol conflict between mlir and comgr in static build")
    endif()
    # Try to find package rocMLIR
    # REQUIRED is omitted since we do not want cmake to abort if the package is not found
    find_package(rocMLIR 1.0.0 CONFIG)
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
            set(rocMLIR_VERSION_MAJOR 0)
            set(rocMLIR_VERSION_MINOR 0)
            set(rocMLIR_VERSION_PATCH 1)
        endif()
    else()
        message(STATUS "Build with rocMLIR::rockCompiler ${rocMLIR_VERSION}")
    endif()
    math(EXPR rocMLIR_VERSION_FLAT "(${rocMLIR_VERSION_MAJOR} * 1000 + ${rocMLIR_VERSION_MINOR}) * 100000 + ${rocMLIR_VERSION_PATCH}")
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
    else()
        message(STATUS "Build with HIPRTC")
    endif()
endif()

option(Boost_USE_STATIC_LIBS "Use boost static libraries" ON)
set(BOOST_COMPONENTS filesystem)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

find_path(HALF_INCLUDE_DIR half/half.hpp)
message(STATUS "HALF_INCLUDE_DIR: ${HALF_INCLUDE_DIR}")

option( MIOPEN_DEBUG_FIND_DB_CACHING "Use system find-db caching" ON)

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)

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

# Unpack DB files
function(unpack_db db_bzip2_file)
    set(KERNELS_DIR "${CMAKE_SOURCE_DIR}/src/kernels")
    STRING(REPLACE ".bz2" "" db_file "${db_bzip2_file}")
    find_program(UNZIPPER NAMES lbunzip2 bunzip2)
    if(EXISTS "${db_file}")
        message(STATUS "WARNING: ${db_file} already exists and will be overwritten")
    endif()
    if(EXISTS "${db_bzip2_file}")
        message(STATUS "${UNZIPPER} -k -f ${db_bzip2_file}")
        execute_process(
            COMMAND ${UNZIPPER} -k -f ${db_bzip2_file}
            WORKING_DIRECTORY ${KERNELS_DIR}
            RESULT_VARIABLE ret
            )
        if(NOT ret EQUAL "0")
            message(FATAL_ERROR "${db_file} could not be extracted, file is empty!")
            execute_process(
                COMMAND touch ${db_file}
                WORKING_DIRECTORY ${KERNELS_DIR}
                )
        endif()
    endif()
endfunction()

file(GLOB PERF_DB_BZIP_FILES "${CMAKE_SOURCE_DIR}/src/kernels/*.db.bz2")
foreach(DB_BZIP_FILE ${PERF_DB_BZIP_FILES})
    unpack_db(${DB_BZIP_FILE})
endforeach()

file(GLOB FIND_DB_BZIP_FILES "${CMAKE_SOURCE_DIR}/src/kernels/*.fdb.txt.bz2")
foreach(DB_BZIP_FILE ${FIND_DB_BZIP_FILES})
    unpack_db(${DB_BZIP_FILE})
endforeach()

# Begin KDB component install 
function(install_kdb)
    set(oneValueArgs NAME COMPONENT_NAME)
    cmake_parse_arguments(PARSE "" "${oneValueArgs}" "" ${ARGN})

    if(PARSE_NAME)
        set(FILE_NAME ${PARSE_NAME})
    endif()
    STRING(REPLACE - _ FILE_NAME "${ARCH_NAME}")
    file(READ "${CMAKE_SOURCE_DIR}/src/kernels/${FILE_NAME}.kdb.bz2" FILE_CONTENTS LIMIT 7)
    string(STRIP "${FILE_CONTENTS}" FILE_CONTENTS)
    if(FILE_CONTENTS STREQUAL "version")
        set(LFS_MISSING_FILES ${LFS_MISSING_FILES} ${FILE_NAME}.kdb.bz2 PARENT_SCOPE)
    else()
        unpack_db("${CMAKE_SOURCE_DIR}/src/kernels/${FILE_NAME}.kdb.bz2")
        message("Installing ${FILE_NAME} in component ${PARSE_COMPONENT_NAME}")
        if( NOT ENABLE_ASAN_PACKAGING )
          rocm_install(FILES
              src/kernels/${FILE_NAME}.kdb
          DESTINATION ${DATA_INSTALL_DIR}/db
          COMPONENT ${PARSE_COMPONENT_NAME})
        endif()
    endif()
endfunction()

# Both the lists below should be in sync always
list(APPEND ARCH_LST gfx90a gfx1030 gfx908 gfx906 gfx900)
list(APPEND ARCH_FILE_LST gfx90akdb gfx1030kdb gfx908kdb gfx906kdb gfx900kdb)

list(LENGTH ARCH_LST FULL_LEN)
math(EXPR ARCH_LST_LEN "${FULL_LEN} - 1")


foreach(IDX RANGE ${ARCH_LST_LEN})
    list(GET ARCH_LST ${IDX} ARCH_NAME)
    list(GET ARCH_FILE_LST ${IDX} ARCH_FILE_NAME)
    install_kdb(
        NAME ${ARCH_NAME}
        COMPONENT_NAME ${ARCH_FILE_NAME}
    )
endforeach()

if(LFS_MISSING_FILES)
    string(REPLACE ";" "; " LFS_MISSING_FILES "${LFS_MISSING_FILES}")
    message(WARNING "GIT LFS Files not pulled down, skipped: ${LFS_MISSING_FILES}")
    set(MIOPEN_NO_LFS_PULLED TRUE CACHE INTERNAL "")
else()
    set(CPACK_COMPONENTS_ALL ${ARCH_FILE_LST})
endif()

#end kdb package creation

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
        *
        -abseil-*
        -altera-*
        -android-cloexec-fopen
        # This check is useless for us. Many objects (like tensors or problem descriptions)
        # naturally have mutiple parameters of the same type.
        -bugprone-easily-swappable-parameters
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
        -hicpp-avoid-c-arrays
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-named-parameter
        -hicpp-no-array-decay
        # We really shouldn't use bitwise operators with signed integers, but opencl leaves us no choice
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
        # This one is extremely slow, and probably has lots of FPs.
        -misc-confusable-identifiers
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
        -modernize-use-nodiscard
        -modernize-concat-nested-namespaces
        -modernize-unary-static-assert
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-convert-member-functions-to-static
        -readability-else-after-return
        # TODO We are not ready to use it, but very useful.
        -readability-function-cognitive-complexity
        # We dont think this is a useful check. Disabled on migraphx.
        -readability-identifier-length
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-qualified-auto
        -readability-redundant-string-init
        # There are many FPs with this, let's disable it (ditto in MIGraphX)
        -readability-suspicious-call-argument
        -readability-uppercase-literal-suffix
        ###################################################################
        # TODO Code Quality WORKAROUND ROCm 5.1 update
        ###################################################################
        -cert-err33-c
        -google-readability-casting
        -hicpp-use-emplace
        -modernize-use-emplace
        -performance-unnecessary-copy-initialization
        -readability-container-data-pointer
        ###################################################################
        # TODO Code Quality WORKAROUND ROCm 5.3 &&
        # Ubuntu 22.04 && C++17 && cppcheck 2.9 update
        ###################################################################
        -bugprone-use-after-move
        -hicpp-invalid-access-moved
        -modernize-use-nodiscard
        -modernize-unary-static-assert
        -modernize-macro-to-enum
        -modernize-concat-nested-namespaces
        -readability-redundant-declaration
        -readability-simplify-boolean-expr
        -hicpp-deprecated-headers
        -hicpp-member-init
        -performance-no-automatic-move
        -clang-analyzer-cplusplus.NewDeleteLeaks
        -modernize-deprecated-headers
        ###################################################################
        # TODO Code Quality WORKAROUND ROCm 5.4.2
        ###################################################################
        -misc-const-correctness
        ###################################################################
        # TODO Code Quality WORKAROUND ROCm 5.6
        ###################################################################
        -cppcoreguidelines-avoid-const-or-ref-data-members
        -cppcoreguidelines-avoid-do-while
        -misc-use-anonymous-namespace
        ###################################################################
        # TODO Code Quality WORKAROUND ROCm 5.7
        ###################################################################
        -llvmlibc-inline-function-decl
        -cppcoreguidelines-avoid-capture-default-when-capturing-this
        -cppcoreguidelines-rvalue-reference-param-not-moved
        -readability-avoid-unconditional-preprocessor-if
        -bugprone-lambda-function-name
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
add_subdirectory(test)
add_subdirectory(speedtests)
add_subdirectory(utils)
if(MIOPEN_ENABLE_FIN)
add_subdirectory(fin)
endif()
