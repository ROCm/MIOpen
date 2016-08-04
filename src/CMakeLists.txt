cmake_minimum_required( VERSION 2.8.12 )

include (GenerateExportHeader)
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
	errors.cpp
	Handle.hpp
	Handle_API.cpp
	Tensor.cpp
	Tensor.hpp
	Tensor_API.cpp
	Convolution.hpp
	Convolution.cpp
	pooling.hpp
	Convolution_API.cpp
	pooling.cpp
	pooling_api.cpp
	KernelCache.cpp
	KernelCache.hpp
	OCLKernel.hpp
	OCLKernel.cpp
	include/mlopenTypes.h
	mloCLUtil.cpp
	mloDirConv.cpp
	mlo_internal.hpp
	mloUtils.hpp
	Common.hpp
	)

if( MLOpen_BACKEND MATCHES "OpenCL")
	list(APPEND MLOpen_Source OCL/HandleOCL.cpp
		OCL/HandleOCL.cpp
		OCL/TensorOCL.cpp
		OCL/ConvolutionOCL.cpp
		OCL/pooling_ocl.cpp
		OCL/CLHelper.hpp
		OCL/CLHelper.cpp)
endif()

if( MLOpen_BACKEND MATCHES "HIP")
	list(APPEND MLOpen_Source HIP/HandleHIP.cpp
		HIP/TensorHIP.cpp
		HIP/ConvolutionHIP.cpp)
endif()

# build library
add_library( MLOpen SHARED
	${MLOpen_Source}
	${MLOpen_API}
	)

target_include_directories(MLOpen PUBLIC ${MLOpen_DIRS_INCLUDE})

install(TARGETS MLOpen 
	RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

GENERATE_EXPORT_HEADER(MLOpen)
############################################################
# MLOpen depends on OpenCL
if( MLOpen_BACKEND MATCHES "OpenCL")
	MESSAGE( STATUS "MLOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
	target_include_directories(MLOpen PUBLIC ${OPENCL_INCLUDE_DIRS} )
	target_link_libraries( MLOpen ${OPENCL_LIBRARIES} )
endif()

