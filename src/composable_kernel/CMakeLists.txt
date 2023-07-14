cmake_minimum_required(VERSION 3.5)
project(composable_kernel)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)

## C++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

## OpenMP
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# workaround issue hipcc in rocm3.5 cannot find openmp
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
else()
	find_package(OpenMP REQUIRED)
endif()

message("OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message("OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
message("OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
link_libraries(${OpenMP_gomp_LIBRARY})
link_libraries(${OpenMP_pthread_LIBRARY})

## HIP
find_package(HIP REQUIRED)
message(STATUS "Build with HIP ${hip_VERSION}")

## half
#find_path(HALF_INCLUDE_DIR half.hpp)
message("HALF_INCLUDE_DIR: ${HALF_INCLUDE_DIR}")

# CMAKE_CXX_FLAGS
SET(BUILD_DEV ON CACHE BOOL "BUILD_DEV")
if(BUILD_DEV)
    string(APPEND CMAKE_CXX_FLAGS " -Werror -Weverything")
endif()
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

## tidy
include(EnableCompilerWarnings)
set(MIOPEN_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
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
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -bugprone-exception-escape
        -bugprone-macro-parentheses
        -cert-env33-c
        -cert-msc32-c
        -cert-msc50-cpp
        -cert-msc51-cpp
        -cert-dcl37-c
        -cert-dcl51-cpp
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
        -hicpp-vararg
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-named-parameter
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
        #-llvmlibc-*
        -llvmlibc-restrict-system-libc-headers
        -llvmlibc-callee-namespace
        -llvmlibc-implementation-in-namespace
        -llvm-else-after-return
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
        -readability-else-after-return
        # we are not ready to use it, but very useful
        -readability-function-cognitive-complexity
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-uppercase-literal-suffix
        -readability-convert-member-functions-to-static
        -readability-qualified-auto
        -readability-redundant-string-init
        # too many narrowing conversions in our code
        -bugprone-narrowing-conversions
        -cppcoreguidelines-narrowing-conversions
        -altera-struct-pack-align
        -cppcoreguidelines-prefer-member-initializer

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
        preprocessorErrorDirective
        shadowVariable
        unusedFunction
        unusedPrivateFunction
        unusedStructMember
        unmatchedSuppression
    FORCE
    SOURCES
        host/host_tensor/src
        host/driver_offline/src
        composable_kernel/src/kernel_wrapper
    INCLUDE
        host/host_tensor/include
        host/solver/include
        host/driver_offline/include
        composable_kernel/include/*
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
    DEFINE
        CPPCHECK=1
        __linux__=1
)

add_subdirectory(host)
