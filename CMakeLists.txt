cmake_minimum_required( VERSION 2.8.12 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
	set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Default installation path
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

project ( MIOpen C CXX )

# Check if cmake supports the new VERSION tag for project() commands
# MIOpen becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( MIOpen VERSION 0.10.0.0 LANGUAGES C CXX )
else( )
	project( MIOpen C CXX )
	# Define a version for the code
	if( NOT DEFINED MIOpen_VERSION_MAJOR )
		set( MIOpen_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED MIOpen_VERSION_MINOR )
		set( MIOpen_VERSION_MINOR 10 )
	endif( )

	if( NOT DEFINED MIOpen_VERSION_PATCH )
		set( MIOpen_VERSION_PATCH 0 )
	endif( )

	if( NOT DEFINED MIOpen_VERSION_TWEAK )
		set( MIOpen_VERSION_TWEAK 0 )
	endif( )

	set( MIOpen_VERSION "${MIOpen_VERSION_MAJOR}.${MIOpen_VERSION_MINOR}.${MIOpen_VERSION_PATCH}.${MIOpen_VERSION_TWEAK}")
endif( )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

############################################################
# require C++11
add_compile_options(-std=c++11)

############################################################
# OPTION - MIOpen Backend
# - OpenCL
# - HCC
if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
	set(MIOPEN_DEFAULT_BACKEND "HIPOC")
else()
	set(MIOPEN_DEFAULT_BACKEND "OpenCL")
endif()


set( MIOPEN_BACKEND ${MIOPEN_DEFAULT_BACKEND} CACHE STRING
	"Which of MIOpens's backends to use?" )
set_property( CACHE MIOPEN_BACKEND PROPERTY STRINGS
	OpenCL HCC HIP HIPOC )
find_package(tinygemm)
# OpenCL 1.2
if( MIOPEN_BACKEND STREQUAL "OpenCL")
	add_definitions( -DMIOPEN_BACKEND_OPENCL=1 )
	add_definitions( -DMIOPEN_BACKEND_HCC=0 )
	add_definitions( -DMIOPEN_BACKEND_HIP=0 )
	add_definitions( -DMIOPEN_BACKEND_HIPOC=0 )
	find_package( OpenCL REQUIRED )
	find_program(AMDGCN_ASSEMBLER
		NAMES clang
		PATHS ${MIOPEN_AMDGCN_ASSEMBLER_PATH} /opt/rocm
		PATH_SUFFIXES /opencl/bin/x86_64
		NO_DEFAULT_PATH
	)
	if(AMDGCN_ASSEMBLER)
		add_definitions(-DMIOPEN_AMDGCN_ASSEMBLER="${AMDGCN_ASSEMBLER}")
	endif()
	message(STATUS "AMDGCN assembler: ${AMDGCN_ASSEMBLER}")
endif()
# HCC
if( MIOPEN_BACKEND STREQUAL "HCC")
	add_definitions( -DMIOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMIOPEN_BACKEND_HCC=1 )
	add_definitions( -DMIOPEN_BACKEND_HIP=0 )
	add_definitions( -DMIOPEN_BACKEND_HIPOC=0 )
	message( SEND_ERROR "${MIOPEN_BACKEND} backend not yet supported." )
endif()
# HIP
if( MIOPEN_BACKEND STREQUAL "HIP")
	add_definitions( -DMIOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMIOPEN_BACKEND_HCC=0 )
	add_definitions( -DMIOPEN_BACKEND_HIP=1 )
	add_definitions( -DMIOPEN_BACKEND_HIPOC=0 )
	message( SEND_ERROR "${MIOPEN_BACKEND} backend not yet supported." )
endif()
# HIPOC
if( MIOPEN_BACKEND STREQUAL "HIPOC")
	add_definitions( -DMIOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMIOPEN_BACKEND_HCC=0 )
	add_definitions( -DMIOPEN_BACKEND_HIP=0 )
	add_definitions( -DMIOPEN_BACKEND_HIPOC=1 )
	find_program(HIP_OC_COMPILER clang-ocl
		PATH_SUFFIXES bin
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
message( STATUS "${MIOPEN_BACKEND} backend selected." )

# tinygemm
if(tinygemm_FOUND)
	add_definitions(-DMIOPEN_USE_TINYGEMM=1)
else()
	add_definitions(-DMIOPEN_USE_TINYGEMM=0)
endif()

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

option( BUILD_DEV "Build for development only" OFF)

include(GNUInstallDirs)
set( MIOPEN_INSTALL_DIR miopen)
set( BIN_INSTALL_DIR ${MIOPEN_INSTALL_DIR}/${CMAKE_INSTALL_BINDIR} )
set( LIB_INSTALL_DIR ${MIOPEN_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${MIOPEN_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR} )
set( DATA_INSTALL_DIR ${MIOPEN_INSTALL_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/miopen )
set( CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/miopen )

if(BUILD_DEV)
	add_definitions(-DMIOPEN_BUILD_DEV=1)
	set(MIOPEN_DB_PATH "${CMAKE_SOURCE_DIR}/src/Kernels")
else()
	add_definitions(-DMIOPEN_BUILD_DEV=0)
	set(MIOPEN_DB_PATH "${DATA_INSTALL_DIR}/db" CACHE PATH "Default path to search for db")
endif()

include(CreatePackage)
create_package(
	NAME MIOpen-${MIOPEN_BACKEND}
	DESCRIPTION "AMD's DNN Library"
	MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
	LDCONFIG ${LIB_INSTALL_DIR}
)

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
		-DMIOPEN_USE_CLANG_TIDY
	ANALYZE_TEMPORARY_DTORS ON

)

add_subdirectory(AddKernels)
add_subdirectory(src)
add_subdirectory(driver)
add_subdirectory(test)
