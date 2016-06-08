cmake_minimum_required( VERSION 2.8.12 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
	set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Check if cmake supports the new VERSION tag for project() commands
# MLOpen becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( MLOpen VERSION 0.10.0.0 LANGUAGES C CXX )
else( )
	project( MLOpen C CXX )
	# Define a version for the code
	if( NOT DEFINED MLOpen_VERSION_MAJOR )
		set( MLOpen_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED MLOpen_VERSION_MINOR )
		set( MLOpen_VERSION_MINOR 10 )
	endif( )

	if( NOT DEFINED MLOpen_VERSION_PATCH )
		set( MLOpen_VERSION_PATCH 0 )
	endif( )

	if( NOT DEFINED MLOpen_VERSION_TWEAK )
		set( MLOpen_VERSION_TWEAK 0 )
	endif( )

	set( MLOpen_VERSION "${MLOpen_VERSION_MAJOR}.${MLOpen_VERSION_MINOR}.${MLOpen_VERSION_PATCH}.${MLOpen_VERSION_TWEAK}")
endif( )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${PROJECT_SOURCE_DIR}/../include/MLOpen-version.h.in" "${PROJECT_BINARY_DIR}/include/MLOpen-version.h" )

message( STATUS "MLOpen_VERSION= ${MLOpen_VERSION}" )
message( STATUS "CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}" )

# This is incremented when the ABI to the library changes
set( MLOpen_SOVERSION 1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

set( MLOpen_Source
	Handle.cpp
	Handle.hpp
	include/mlopenTypes.h
	)

include_directories (
	${MLOpen_DIRS_INCLUDE}
	)

# build library
add_library( MLOpen
	${MLOpen_Source}
	${MLOpen_API}
	)

############################################################
# MLOpen depends on OpenCL
if( MLOpen_BACKEND MATCHES "OpenCL")
	MESSAGE( "MLOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
	include_directories( ${OPENCL_INCLUDE_DIRS} )
	target_link_libraries( MLOpen ${OPENCL_LIBRARIES} )
endif()
