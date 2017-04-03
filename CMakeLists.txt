cmake_minimum_required( VERSION 2.8.12 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
	set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Default installation path
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

project ( SuperBuild.MLOpen C CXX )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

############################################################
# require C++11
add_compile_options(-std=c++11)

option( USE_Cobalt "Use Cobalt for GEMM operations" OFF )

# Determine whether to build 64-bit (default) or 32-bit
if( MSVC_IDE )
	set( BUILD_64 ${CMAKE_CL_64} )
else()
	option( BUILD_64 "Build a 64-bit product" ON )
endif()

############################################################
# OPTION - MLOpen Backend
# - OpenCL
# - HCC
if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
	set(MLOPEN_DEFAULT_BACKEND "HIPOC")
else()
	set(MLOPEN_DEFAULT_BACKEND "OpenCL")
endif()


set( MLOPEN_BACKEND ${MLOPEN_DEFAULT_BACKEND} CACHE STRING
	"Which of MLOpens's backends to use?" )
set_property( CACHE MLOPEN_BACKEND PROPERTY STRINGS
	OpenCL HCC HIP HIPOC )
find_package(tinygemm)
# OpenCL 1.2
if( MLOPEN_BACKEND STREQUAL "OpenCL")
	add_definitions( -DMLOPEN_BACKEND_OPENCL=1 )
	add_definitions( -DMLOPEN_BACKEND_HCC=0 )
	add_definitions( -DMLOPEN_BACKEND_HIP=0 )
	add_definitions( -DMLOPEN_BACKEND_HIPOC=0 )
	find_package( OpenCL REQUIRED )
endif()
# HCC
if( MLOPEN_BACKEND STREQUAL "HCC")
	add_definitions( -DMLOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMLOPEN_BACKEND_HCC=1 )
	add_definitions( -DMLOPEN_BACKEND_HIP=0 )
	add_definitions( -DMLOPEN_BACKEND_HIPOC=0 )
	message( SEND_ERROR "${MLOPEN_BACKEND} backend not yet supported." )
endif()
# HIP
if( MLOPEN_BACKEND STREQUAL "HIP")
	add_definitions( -DMLOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMLOPEN_BACKEND_HCC=0 )
	add_definitions( -DMLOPEN_BACKEND_HIP=1 )
	add_definitions( -DMLOPEN_BACKEND_HIPOC=0 )
	message( SEND_ERROR "${MLOPEN_BACKEND} backend not yet supported." )
endif()
# HIPOC
if( MLOPEN_BACKEND STREQUAL "HIPOC")
	add_definitions( -DMLOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMLOPEN_BACKEND_HCC=0 )
	add_definitions( -DMLOPEN_BACKEND_HIP=0 )
	add_definitions( -DMLOPEN_BACKEND_HIPOC=1 )
	find_program(HIP_OC_COMPILER clang-ocl
		PATHS /opt/rocm
	)
	if(HIP_OC_COMPILER)
		message(STATUS "hip compiler: ${HIP_OC_COMPILER}")
		add_definitions(-DHIP_OC_COMPILER="${HIP_OC_COMPILER}")
	else()
		find_program(HIP_OC_COMPILER aoc2
			PATH_SUFFIXES
				bin
				bin/x86_64
				x86_64
		)
		find_program(HIP_OC_FINALIZER amdhsafin
			PATH_SUFFIXES
				bin
				bin/x86_64
				x86_64
		)
		message(STATUS "hip compiler: ${HIP_OC_COMPILER}")
		add_definitions(-DHIP_OC_COMPILER="${HIP_OC_COMPILER}")
		message(STATUS "hip finalizer: ${HIP_OC_FINALIZER}")
		add_definitions(-DHIP_OC_FINALIZER="${HIP_OC_FINALIZER}")
	endif()
	
	find_package(hip PATHS /opt/rocm)
	link_libraries(stdc++)
endif()
message( STATUS "${MLOPEN_BACKEND} backend selected." )

# tinygemm
if(tinygemm_FOUND)
	add_definitions(-DMLOPEN_USE_TINYGEMM=1)
else()
	add_definitions(-DMLOPEN_USE_TINYGEMM=0)
endif()

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

option( BUILD_DEV "Build for development only" OFF)

if(BUILD_DEV)
	set(MLOPEN_DB_PATH "${CMAKE_SOURCE_DIR}/src/Kernels")
else()
	set(MLOPEN_DB_PATH "${CMAKE_INSTALL_PREFIX}/share/mlopen/db" CACHE PATH "Default path to search for db")
endif()

include(EnableCompilerWarnings)

include(ClangTidy)
enable_clang_tidy(
	CHECKS 
		*
		-cert-err60-cpp
		# Yea we shouldn't be using rand()
		-cert-msc30-c
		-cert-msc50-cpp
		-clang-analyzer-alpha.core.CastToStruct
		-clang-analyzer-optin.performance.Padding
		-clang-diagnostic-deprecated-declarations
		-clang-diagnostic-extern-c-compat
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
		-llvm-header-guard
		-llvm-include-order
		-misc-macro-parentheses
		-misc-misplaced-const
		-misc-misplaced-widening-cast
		-modernize-loop-convert
		-modernize-pass-by-value
		-modernize-use-default-member-init
		-modernize-use-emplace
		-modernize-use-equals-default
		-performance-unnecessary-value-param
		-readability-braces-around-statements
		-readability-else-after-return
		-readability-implicit-bool-cast
		-readability-misleading-indentation
		-readability-named-parameter
	ERRORS
		*
		-readability-inconsistent-declaration-parameter-name
	HEADER_FILTER
		"hpp$"
	EXTRA_ARGS
		-DMLOPEN_USE_CLANG_TIDY
	ANALYZE_TEMPORARY_DTORS ON

)

add_subdirectory(AddKernels)
add_subdirectory(src)
add_subdirectory(driver)
add_subdirectory(test)
