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

macro(set_if_bools_are_different var in1 in2)
    set(${var} FALSE)
    if(${in1})
        if(NOT ${in2})
           set(${var} TRUE)
        endif()
    else()
        if(${in2})
           set(${var} TRUE)
        endif()
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

if(MIOPEN_ENABLE_SQLITE)
    # MIOpen now depends on SQLite as well
    find_package(SQLite3 REQUIRED)
endif()
find_package(BZip2 REQUIRED)
find_package(nlohmann_json 3.9.1 REQUIRED)
if(MIOPEN_ENABLE_SQLITE_KERN_CACHE AND NOT MIOPEN_ENABLE_SQLITE)
    message(FATAL_ERROR "MIOPEN_ENABLE_SQLITE_KERN_CACHE requires MIOPEN_ENABLE_SQLITE")
endif()
set(MIOPEN_LOG_FUNC_TIME_ENABLE Off CACHE BOOL "")
set(MIOPEN_ENABLE_SQLITE_BACKOFF On CACHE BOOL "")

option( BUILD_DEV "Build for development only" OFF)
option(MIOPEN_ENABLE_FIN "Enable the fin driver for MIOpen"  OFF)
option(MIOPEN_STRIP_SYMBOLS "Strip symbols in release mode" ON)

option(MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM "Workaround: Use boost::filesystem instead of std::filesystem" OFF)
message(STATUS "MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM ${MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM}")

# Strip symbols for release
if(MIOPEN_STRIP_SYMBOLS AND NOT WIN32 AND NOT APPLE)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

rocm_setup_version(VERSION 3.3.0)

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

if(NOT WIN32 AND NOT MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM)
    include(CheckCXXLinkerFlag)
    check_cxx_linker_flag(-lstdc++fs HAS_LIB_STD_FILESYSTEM)
endif()

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip)

option(ENABLE_HIP_WORKAROUNDS Off)
set(MIOPEN_INSTALL_CXX_HEADERS Off CACHE BOOL "Install MIOpen's C++ header interface")

set_var_to_condition(MIOPEN_OFFLINE_COMPILER_PATHS_V2_DEFAULT FALSE)
option(MIOPEN_OFFLINE_COMPILER_PATHS_V2 "Use rocm-core to find offline GPU compiler" ${MIOPEN_OFFLINE_COMPILER_PATHS_V2_DEFAULT})
message( STATUS "MIOPEN_OFFLINE_COMPILER_PATHS_V2: ${MIOPEN_OFFLINE_COMPILER_PATHS_V2}" )

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
else()
    option(MIOPEN_USE_HIP_KERNELS "Use HIP kernels." On)
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

set_var_to_condition(MIOPEN_BUILD_DRIVER_DEFAULT (NOT MIOPEN_EMBED_BUILD) AND (NOT (MIOPEN_BACKEND STREQUAL "HIPNOGPU")))
option(MIOPEN_BUILD_DRIVER "Build MIOpenDriver (and use it in tests)" ${MIOPEN_BUILD_DRIVER_DEFAULT})
message(STATUS "MIOPEN_BUILD_DRIVER: ${MIOPEN_BUILD_DRIVER}" )

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

# HIP SDK on Windows does not detect platform correctly, defaulting to "amd"
if(NOT HIP_PLATFORM AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(HIP_PLATFORM "amd")
endif()

# HIP is always required
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION} ${hip_DIR}")

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
set_var_to_condition(MIOPEN_USE_HIPRTC_DEFAULT MIOPEN_USE_COMGR)
option(MIOPEN_USE_HIPRTC "Use HIPRTC to build HIP kernels instead of COMGR" ${MIOPEN_USE_HIPRTC_DEFAULT})

set_if_bools_are_different(MIOPEN_CONFIGURATION_ERROR_COMGR_HIPRTC MIOPEN_USE_COMGR MIOPEN_USE_HIPRTC)
if(MIOPEN_CONFIGURATION_ERROR_COMGR_HIPRTC)
   message(FATAL_ERROR "MIOPEN_USE_COMGR (${MIOPEN_USE_COMGR}) and MIOPEN_USE_HIPRTC (${MIOPEN_USE_HIPRTC}) should be set to the same value")
endif()

# Do not append system include directories to HIP compiler flags when HIPRTC is used
set_var_to_condition(MIOPEN_HIP_COMPILER_USE_SYSTEM_INCLUDE_DIRECTORIES_DEFAULT
              (NOT (MIOPEN_USE_HIPRTC AND (MIOPEN_hip_VERSION VERSION_GREATER_EQUAL 6.1.40091))))
option(MIOPEN_HIP_COMPILER_USE_SYSTEM_INCLUDE_DIRECTORIES "Append include directories to compiler flags"
       ${MIOPEN_HIP_COMPILER_USE_SYSTEM_INCLUDE_DIRECTORIES_DEFAULT})

target_flags(HIP_COMPILER_FLAGS hip::device)
# Remove cuda arch flags
string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REGEX REPLACE --offload-arch=[a-z0-9:+-]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
# Skip library paths since hip will incorrectly treat it as a source file
string(APPEND HIP_COMPILER_FLAGS " ")
foreach(_unused RANGE 2)
    string(REGEX REPLACE " /[^ ]+\\.(a|so) " " " HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
endforeach()

# WORKAROUND_SWDEV_413293
# Assume that any HIP kernel can be launched with non-uniform block size; otherwise
# the "Failed to launch kernel: invalid argument" error may happen at run time.
# References: SWDEV-413293 and https://reviews.llvm.org/D155213 effective HIP_FLAT_VERSION 500723302 on Linux.
# This may lead to perf drops in the future therefore https://github.com/ROCm/MIOpen/issues/2708 is opened.
if(HAS_HIP)
    # HIP version is unreliable on Windows and on Fedora, so we use compiler flag detection,
    # if this is possible. See issue 2734 and PR 2719.
    check_cxx_compiler_flag("-x hip -fno-offload-uniform-block" MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK)
else()
    # CXX compiler is not HIP compiler, let's analyze HIP version.
    set(MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK Off)
    if(MIOPEN_hip_VERSION_FLAT GREATER_EQUAL 500723302)
        set(MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK On)
    endif()
    message(STATUS "MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK: ${MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK}")
endif()
if(MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK)
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
        find_package(composable_kernel 1.0.0 COMPONENTS device_other_operations device_gemm_operations device_conv_operations device_reduction_operations)
    endif()
    if( MIOPEN_BACKEND STREQUAL "HIPNOGPU")
        set(MIOPEN_MODE_NOGPU 1)
    endif()
    set(MIOPEN_BACKEND_HIP 1)

    find_program(HIP_OC_COMPILER NAMES amdclang clang
        PATH_SUFFIXES bin
        PATHS
            /opt/rocm
            ${CMAKE_INSTALL_PREFIX}
        ENV HIP_PATH
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
        message(STATUS "Build with rocblas ${rocblas_VERSION} ${rocblas_DIR}")
    else()
        message(STATUS "Build without rocblas")
    endif()

    # hipblaslt
    set_var_to_condition(MIOPEN_USE_HIPBLASLT_DEFAULT NOT WIN32)
    option(MIOPEN_USE_HIPBLASLT "Use hipBlasLt" ${MIOPEN_USE_HIPBLASLT_DEFAULT})
    if(MIOPEN_USE_HIPBLASLT)
        find_package(hipblas REQUIRED PATHS /opt/rocm $ENV{HIP_PATH})
        message(STATUS "Build with hipbBLAS ${hipblas_VERSION} ${hipblas_DIR}")
        find_package(hipblaslt REQUIRED PATHS /opt/rocm $ENV{HIP_PATH})
        message(STATUS "Build with hipbBLASLt ${hipblaslt_VERSION} ${hipblaslt_DIR}")
    else()
        message(STATUS "Build without hipbBLASLt")
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
        if(NOT rocMLIR_FOUND)
            message(STATUS "Falling back to find library libMLIRMIOpen")
            # Backward compatibility with ROCm 5.3
            # If the rocMLIR package is not found, try to find the library libMLIRMIOpen directly
            find_library(LIBMLIRMIOPEN MLIRMIOpen REQUIRED)
            if(NOT LIBMLIRMIOPEN)
                message(FATAL_ERROR "library libMLIRMIOpen not found, please reinstall dependencies. \
                Refer to https://github.com/ROCm/MIOpen#installing-the-dependencies")
            else()
                message(STATUS "Build with library libMLIRMIOpen: " ${LIBMLIRMIOPEN})
                set(rocMLIR_VERSION 0.0.1)
            endif()
        endif()
    endif()
    message(STATUS "Build with rocMLIR::rockCompiler ${rocMLIR_VERSION} ${rocMLIR_DIR}")
endif()

# Update HIP Runtime Package Dependency
if(ENABLE_ASAN_PACKAGING)
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
else()
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
endif()
set(MIOPEN_PACKAGE_REQS "${DEPENDS_HIP_RUNTIME}")

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
    message(STATUS "Build with amd_comgr ${amd_comgr_VERSION} ${amd_comgr_DIR}")
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, comgr")
endif()

if(MIOPEN_USE_HIPRTC)
    if(NOT MIOPEN_USE_COMGR)
        message(FATAL_ERROR "HIPRTC can be used only together with COMGR")
    endif()
    find_package(hiprtc REQUIRED)
    message(STATUS "Build with hiprtc ${hiprtc_VERSION} ${hiprtc_DIR}")
endif()

option(Boost_USE_STATIC_LIBS "Use boost static libraries" ON)
set(BOOST_COMPONENTS filesystem)
if(MIOPEN_BUILD_DRIVER)
    # boost core is a header only component that can't be found by find_package
    # list(APPEND BOOST_COMPONENTS core)
endif()

# The FindBoost module has been removed since CMake 3.30. Use the Boost 1.83 configuration file instead.
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS} CONFIG)

find_path(HALF_INCLUDE_DIR half/half.hpp)
message(STATUS "HALF_INCLUDE_DIR: ${HALF_INCLUDE_DIR}")

option( MIOPEN_DEBUG_FIND_DB_CACHING "Use system find-db caching" ON)

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)

if(WIN32)
    set( DATA_INSTALL_DIR bin )
    set( DATABASE_INSTALL_DIR ${DATA_INSTALL_DIR})
else()
    set( MIOPEN_INSTALL_DIR miopen)
    set( DATA_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/${MIOPEN_INSTALL_DIR} )
    set( DATABASE_INSTALL_DIR ${DATA_INSTALL_DIR}/db )
endif()

if(MIOPEN_ENABLE_AI_KERNEL_TUNING OR MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK)
    find_package(frugally-deep CONFIG REQUIRED)
    message(STATUS "Build with frugally-deep ${frugally-deep_VERSION} ${frugally-deep_DIR}")
    find_package(Eigen3 REQUIRED)
    message(STATUS "Build with Eigen3 ${Eigen3_VERSION} ${Eigen3_DIR}")
endif()

if(WIN32)
    set(KERNELS_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)
else()
    set(KERNELS_BINARY_DIR ${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR})
endif()

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

# PR-2391 Add the ability to log function calls to roctx.
# This allows attached profilers to see which MIOpen calls are being called by application and which kernels are being invoked by MIOpen.
# Enabled via the MIOPEN_ENABLE_LOGGING_ROCTX env var.
set(MIOPEN_USE_ROCTRACER ON CACHE BOOL "")
if(NOT WIN32 AND MIOPEN_USE_ROCTRACER)
    find_library(rocTracer roctx64)
    if(rocTracer)
        MESSAGE(STATUS "Build with rocTracer: " ${rocTracer})
        set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, roctracer")
    else()
        message(WARNING "rocTracer cannot be found! Build without rocTracer")
        set(MIOPEN_USE_ROCTRACER OFF)
    endif()
else()
    message(STATUS "Build without rocTracer")
    set(MIOPEN_USE_ROCTRACER OFF)
endif()

if(MIOPEN_USE_ROCBLAS)
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, rocblas")
endif()

if(MIOPEN_USE_HIPBLASLT)
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, hipblaslt")
endif()

if(MIOPEN_OFFLINE_COMPILER_PATHS_V2)
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, rocm-core")
endif()

if(MIOPEN_BUILD_DRIVER)
    # PR #2785 MIOpenDriver to use rocrand to init buffers
    find_package(rocrand REQUIRED)
    message(STATUS "Build with rocrand ${rocrand_VERSION} ${rocrand_DIR}")
    set(MIOPEN_PACKAGE_REQS "${MIOPEN_PACKAGE_REQS}, rocrand")
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

find_program(UNZIPPER bzip2 REQUIRED)
file(MAKE_DIRECTORY ${KERNELS_BINARY_DIR})

add_custom_target(generate_kernels ALL)

set(MIOPEN_USE_SQLITE_PERFDB Off CACHE BOOL "Use sqlite perfdb instead of text-based.")
if(MIOPEN_USE_SQLITE_PERFDB)
    set(PERFDB_SUFFIX "")
else()
    set(PERFDB_SUFFIX ".txt")
endif()

function(unpack_db db_bzip2_file)
    get_filename_component(__fname ${db_bzip2_file} NAME_WLE)
    add_custom_command(OUTPUT ${KERNELS_BINARY_DIR}/${__fname}
                       COMMAND ${UNZIPPER} -dc -k ${db_bzip2_file} > ${KERNELS_BINARY_DIR}/${__fname})
    string(REPLACE "." "_" __tname ${__fname})
    add_custom_target(generate_${__tname} ALL DEPENDS ${KERNELS_BINARY_DIR}/${__fname})

    get_filename_component(__extension ${__fname} LAST_EXT)

    if(NOT MIOPEN_USE_SQLITE_PERFDB AND __extension STREQUAL ".db")
        add_custom_command(OUTPUT ${KERNELS_BINARY_DIR}/${__fname}.txt
                           DEPENDS sqlite2txt generate_${__tname}
                           COMMAND $<TARGET_FILE:sqlite2txt> ${KERNELS_BINARY_DIR}/${__fname} ${KERNELS_BINARY_DIR}/${__fname}.txt
        )
        add_custom_target(generate_${__tname}_txt ALL DEPENDS ${KERNELS_BINARY_DIR}/${__fname}.txt)
        add_dependencies(generate_kernels generate_${__tname}_txt)
        set(__fname ${__fname}.txt)
    else()
        add_dependencies(generate_kernels generate_${__tname})
    endif()
    set(__fname ${__fname} PARENT_SCOPE)
endfunction()

file(GLOB PERF_DB_BZIP_FILES CONFIGURE_DEPENDS "${KERNELS_SOURCE_DIR}/*.db.bz2")
file(GLOB FIND_DB_BZIP_FILES CONFIGURE_DEPENDS "${KERNELS_SOURCE_DIR}/*.fdb.txt.bz2")

foreach(DB_BZIP_FILE ${PERF_DB_BZIP_FILES} ${FIND_DB_BZIP_FILES})
    unpack_db(${DB_BZIP_FILE})
    if(MIOPEN_EMBED_DB STREQUAL "" AND NOT MIOPEN_DISABLE_SYSDB AND NOT ENABLE_ASAN_PACKAGING)
        install(FILES ${KERNELS_BINARY_DIR}/${__fname}
                DESTINATION ${DATABASE_INSTALL_DIR})
    endif()
endforeach()

# Begin KDB package creation

function(install_kdb FILE_NAME COMPONENT_NAME)
    file(READ ${FILE_NAME} __contents LIMIT 7)
    get_filename_component(__fname ${FILE_NAME} NAME_WLE)
    if(__contents MATCHES "version")
        list(APPEND LFS_MISSING_FILES ${__fname})
    else()
        unpack_db(${FILE_NAME})
        if( NOT ENABLE_ASAN_PACKAGING )
            if( NOT MIOPEN_TEST_DBSYNC )
                set(__component_name COMPONENT ${COMPONENT_NAME})
            endif()
            rocm_install(FILES ${KERNELS_BINARY_DIR}/${__fname}
                         DESTINATION ${DATABASE_INSTALL_DIR}
                         ${__component_name})
        endif()
    endif()
    set(LFS_MISSING_FILES ${LFS_MISSING_FILES} PARENT_SCOPE)
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

if(LFS_MISSING_FILES)
    string(REPLACE ";" ", " __lfs_missing_files "${LFS_MISSING_FILES}")
    message(WARNING "GIT LFS files not pulled down, skipped: ${__lfs_missing_files}")
    set(MIOPEN_NO_LFS_PULLED TRUE CACHE INTERNAL "")
else()
    set(CPACK_COMPONENTS_ALL ${COMPONENT_LST})
endif()

# End KDB package creation

if(NOT MIOPEN_TEST_DISCRETE)
list(APPEND CPACK_COMPONENTS_ALL client)
endif()

rocm_create_package(
    NAME MIOpen-${MIOPEN_BACKEND}
    DESCRIPTION "AMD DNN Library"
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
        ###################################################################
        # TODO Code Quality WORKAROUND ROCm 6.0 &&
        # Ubuntu 22.04 && cppcheck 2.12.1 update
        ###################################################################
        duplInheritedMember
        constParameterCallback
        constParameterReference
        constParameterPointer
        constVariableReference
        constVariablePointer
        useStlAlgorithm
        uselessOverride
        unusedScopedObject
    FORCE
    SOURCES
        addkernels/
        tools/sqlite2txt/
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

if(NOT MIOPEN_USE_SQLITE_PERFDB)
    add_subdirectory(tools/sqlite2txt)
endif()
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
