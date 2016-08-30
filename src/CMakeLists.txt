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

include(Bin2H)
function(add_kernels)
	set(INIT_KERNELS_LIST)
	set(KERNELS_HEADER "${PROJECT_BINARY_DIR}/include/mlopen_kernels.h")
	file(WRITE ${KERNELS_HEADER} "")
	foreach(KERNEL_FILE ${ARGN})
		if("${CMAKE_VERSION}" VERSION_LESS 3.0)
			configure_file(${KERNEL_FILE} ${KERNEL_FILE}.delete)
		else()
			set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
		endif()
		get_filename_component(BASE_NAME ${KERNEL_FILE} NAME_WE)
		message(STATUS "Storing ${BASE_NAME}")
		string(TOUPPER "${BASE_NAME}" KEY_NAME)
		string(MAKE_C_IDENTIFIER "${KEY_NAME}" VAR_NAME)
		bin2h(SOURCE_FILE ${KERNEL_FILE} HEADER_FILE ${KERNELS_HEADER} VARIABLE_NAME ${VAR_NAME} NULL_TERMINATE APPEND)
		list(APPEND INIT_KERNELS_LIST "    { \"${KEY_NAME}\", ${VAR_NAME} }")
	endforeach()
	string(REPLACE ";" ",\n" INIT_KERNELS "${INIT_KERNELS_LIST}")
	configure_file(Kernels/kernel.cpp.in kernel.cpp)
endfunction()

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
set( MLOpen_Source
	convolution.cpp
	convolution_api.cpp
	errors.cpp
	pooling_api.cpp
	lrn_api.cpp
	handle_api.cpp
	include/mlopen/common.hpp
	include/mlopen/convolution.hpp
	include/mlopen/errors.hpp
	include/mlopen/handle.hpp
	include/mlopen/kernel_cache.hpp
	include/mlopen/mlo_internal.hpp
	include/mlopen/mlo_utils.hpp
	include/mlopen/oclkernel.hpp
	include/mlopen/tensor.hpp
	include/mlopenTypes.h
	include/mlopen/pooling.hpp
	include/mlopen/lrn.hpp
	tensor.cpp
	tensor_api.cpp
	)

if( MLOPEN_BACKEND MATCHES "OpenCL")
	add_kernels(
		Kernels/MLOpenConvDirUni.cl
		Kernels/MLOpenConvDirGenFwd.cl
		Kernels/MLOpenLRN.cl
		Kernels/MLOpenNeuron.cl
		Kernels/MLOpenPooling.cl
		Kernels/MLOpenPoolingBwd.cl
	)
	configure_file(db.cpp.in db.cpp)
	list(APPEND MLOpen_Source 
		kernel_cache.cpp
		lrn.cpp
		mlo_dir_conv.cpp
		ocl/clhelper.cpp
		ocl/convolutionocl.cpp
		ocl/handleocl.cpp
		ocl/handleocl.cpp
		ocl/lrn_ocl.cpp
		ocl/mloNeuron.cpp
		ocl/mloNorm.cpp
		ocl/mloPooling.cpp
		ocl/pooling_ocl.cpp
		ocl/tensorocl.cpp
		pooling.cpp
        ocl/oclerrors.cpp
        ocl_kernel.cpp
        ${PROJECT_BINARY_DIR}/db.cpp
        ${PROJECT_BINARY_DIR}/kernel.cpp
		)
	# Install db files
	install(FILES 
		Kernels/Fiji_64_1000.cd.pdb.txt
		Kernels/Fiji_64_1050.cd.pdb.txt
	 DESTINATION share/mlopen/db)
endif()

if( MLOPEN_BACKEND MATCHES "HIP")
	list(APPEND MLOpen_Source 
		hip/handlehip.cpp
		hip/convolutionhip.cpp
		hip/tensorhip.cpp
		)
endif()

set ( MLOpen_Headers_Public
	${CMAKE_SOURCE_DIR}/include/mlopen.h
	${PROJECT_BINARY_DIR}/include/MLOpen-version.h
	${PROJECT_BINARY_DIR}/include/mlopen_export.h
	)


# build library
add_library( MLOpen
	${MLOpen_Source}
	)

set( MLOpen_DIRS_INCLUDE
	# TODO: Remove this
	${PROJECT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include)

target_include_directories(MLOpen PUBLIC ${MLOpen_DIRS_INCLUDE})

install(TARGETS MLOpen 
	RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(FILES ${MLOpen_Headers_Public} DESTINATION include)

GENERATE_EXPORT_HEADER(MLOpen
	EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/mlopen_export.h
)
############################################################
# MLOpen depends on OpenCL
if( MLOPEN_BACKEND MATCHES "OpenCL")
	MESSAGE( STATUS "MLOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
	target_include_directories(MLOpen PUBLIC ${OPENCL_INCLUDE_DIRS} )
	target_link_libraries( MLOpen ${OPENCL_LIBRARIES} )
endif()

