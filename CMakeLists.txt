cmake_minimum_required( VERSION 2.8.12 )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/rocblas-package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
	set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project ( SuperBuild.MLOpen C CXX )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
set( TARGET_PLATFORM "64" )
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )


############################################################
# require C++11
add_compile_options(-std=c++11)

# suppress all warnings for now as many occur in cl.hpp
add_compile_options(-w)

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
set( MLOPEN_BACKEND "OpenCL" CACHE STRING
	"Which of Cobalt's backends to use?" )
set_property( CACHE MLOPEN_BACKEND PROPERTY STRINGS
	OpenCL HCC HIP )
# OpenCL 1.2
if( MLOPEN_BACKEND MATCHES "OpenCL")
	add_definitions( -DMLOPEN_BACKEND_OPENCL=1 )
	add_definitions( -DMLOPEN_BACKEND_HCC=0 )
	add_definitions( -DMLOPEN_BACKEND_HIP=0 )
	find_package( OpenCL REQUIRED )
	message( STATUS "${MLOPEN_BACKEND} backend selected." )
endif()
# HCC
if( MLOPEN_BACKEND MATCHES "HCC")
	add_definitions( -DMLOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMLOPEN_BACKEND_HCC=1 )
	add_definitions( -DMLOPEN_BACKEND_HIP=0 )
	message( SEND_ERROR "${Cobalt_BACKEND} backend not yet supported." )
endif()
# HIP
if( MLOPEN_BACKEND MATCHES "HIP")
	add_definitions( -DMLOPEN_BACKEND_OPENCL=0 )
	add_definitions( -DMLOPEN_BACKEND_HCC=0 )
	add_definitions( -DMLOPEN_BACKEND_HIP=1 )
	message( SEND_ERROR "${Cobalt_BACKEND} backend not yet supported." )
endif()

# If Cobalt is used in future
if( USE_Cobalt)
endif()

set ( MLOpen_Headers_Public
	${PROJECT_SOURCE_DIR}/include/mlopen.h
	${PROJECT_BINARY_DIR}/src/include/MLOpen-version.h
	${PROJECT_BINARY_DIR}/src/mlopen_export.h
	)

install(FILES ${MLOpen_Headers_Public} DESTINATION include)

set( MLOpen_DIRS_INCLUDE
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_SOURCE_DIR}/src/include
	${CMAKE_SOURCE_DIR}/src
	${PROJECT_BINARY_DIR}/src)

set( MLOpen_API ${CMAKE_SOURCE_DIR}/include/mlopen.h )
source_group( "Header Files\\Public" FILES ${MLOpen_Headers_Public} )

include(ClangTidy)

enable_clang_tidy(
	CHECKS *
)

add_subdirectory(src)
add_subdirectory(driver)
add_subdirectory(test)
