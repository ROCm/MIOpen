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

function(add_kernels KERNEL_FILES)
	set(INIT_KERNELS_LIST)
	foreach(KERNEL_FILE ${KERNEL_FILES})
		if("${CMAKE_VERSION}" VERSION_LESS 3.0)
			configure_file(${KERNEL_FILE} ${KERNEL_FILE}.delete)
		else()
			set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
		endif()
		get_filename_component(BASE_NAME ${KERNEL_FILE} NAME_WE)
		string(TOUPPER "${BASE_NAME}" KEY_NAME)
		string(MAKE_C_IDENTIFIER "${KEY_NAME}" VAR_NAME)
		list(APPEND INIT_KERNELS_LIST "    { \"${KEY_NAME}\", std::string(reinterpret_cast<const char*>(${VAR_NAME}), ${VAR_NAME}_SIZE) }")
	endforeach()
	string(REPLACE ";" ",\n" INIT_KERNELS "${INIT_KERNELS_LIST}")
	configure_file(Kernels/kernel.cpp.in kernel.cpp)
endfunction()

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
set( MLOpen_Source
	convolution.cpp
	convolution_api.cpp
	convolution_fft.cpp
	errors.cpp
	pooling_api.cpp
	logger.cpp
	lrn_api.cpp
	activ_api.cpp
	handle_api.cpp
	softmax_api.cpp
	include/mlopen/common.hpp
	include/mlopen/convolution.hpp
	include/mlopen/convolution_fft.hpp
	include/mlopen/errors.hpp
	include/mlopen/handle.hpp
	include/mlopen/kernel_cache.hpp
	include/mlopen/mlo_internal.hpp
	include/mlopen/mlo_utils.hpp
	include/mlopen/oclkernel.hpp
	include/mlopen/tensor.hpp
	include/mlopen/tensor_ops.hpp
	include/mlopenTypes.h
	include/mlopen/pooling.hpp
	include/mlopen/lrn.hpp
	include/mlopen/activ.hpp
	include/mlopen/softmax.hpp
	tensor.cpp
	tensor_api.cpp
	)

if( MLOPEN_BACKEND MATCHES "OpenCL" OR MLOPEN_BACKEND STREQUAL "HIPOC")
	set(MLOPEN_KERNELS
		Kernels/MLOpenConvDirUni.cl
		Kernels/MLOpenConvDirGenFwd.cl
		Kernels/MLOpenLRNBwd.cl
		Kernels/MLOpenLRNFwd.cl
		Kernels/MLOpenNeuron.cl
		Kernels/MLOpenPooling.cl
		Kernels/MLOpenPoolingBwd.cl
		Kernels/MLOpenConvDirUniC.cl		
		Kernels/MLOpenConv1x1.cl
		Kernels/MLOpenSoftmax.cl
		Kernels/MLOpenConvD3x3.cl
		Kernels/MLOpenUtilKernels.cl
		Kernels/MLOpenConvBwdWrW_LxL_P.cl
		Kernels/MLOpenConvBwdWrW_LxG_P53.cl
		Kernels/MLOpenConvBwdWrW_LxG_5x5.cl
		Kernels/MLOpenConvBwdWrW_GxL_1x1.cl
		Kernels/MLOpenConvFwd_LxL_11.cl

 		Kernels/MLOpenConvFFT.cl
		Kernels/conv3x3.s
		Kernels/conv5x10u2v2f1.s

    Kernels/MLOpenTensorKernels.cl
		Kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m21.so
		Kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m10.so
		Kernels/MLOpenConvBwdBias.cl)
	
	add_kernels("${MLOPEN_KERNELS}")
	configure_file(db.cpp.in db.cpp)
	list(APPEND MLOpen_Source 
		activ.cpp
		kernel_cache.cpp
		lrn.cpp
		mlo_dir_conv.cpp
		ocl/activ_ocl.cpp
		ocl/convolutionocl.cpp
		ocl/convolutionocl_fft.cpp
		ocl/lrn_ocl.cpp
		ocl/mloNeuron.cpp
		ocl/mloNorm.cpp
		ocl/mloPooling.cpp
		ocl/pooling_ocl.cpp
		ocl/tensorocl.cpp
		ocl/softmaxocl.cpp
		ocl/utilocl.cpp
		ocl/gcn_asm_utils.cpp
		pooling.cpp
        ${PROJECT_BINARY_DIR}/db.cpp
        ${PROJECT_BINARY_DIR}/kernel.cpp
		)
	# Install db files
	install(FILES 
		Kernels/gfx803_64.cd.pdb.txt
	 DESTINATION share/mlopen/db)
endif()

if(tinygemm_FOUND)
    list(APPEND MLOpen_Source 
        gemm.cpp
        gemm_api.cpp
        gemm_geometry.cpp
    )
endif()

if( MLOPEN_BACKEND STREQUAL "OpenCL" )
	list(APPEND MLOpen_Source 
		ocl/handleocl.cpp
        ocl_kernel.cpp
        ocl/oclerrors.cpp
		ocl/clhelper.cpp
	)
endif()

if( MLOPEN_BACKEND STREQUAL "HIPOC")
	list(APPEND MLOpen_Source 
		hip/hiperrors.cpp
		hip/handlehip.cpp
		hipoc/hipoc_kernel.cpp
		hipoc/hipoc_program.cpp
		)
endif()

if( MLOPEN_BACKEND STREQUAL "HIP")
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

if( MLOPEN_BACKEND MATCHES "OpenCL" OR MLOPEN_BACKEND STREQUAL "HIPOC")
	add_custom_target(addkernels_run
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		DEPENDS addkernels
		COMMAND $<TARGET_FILE:addkernels> -guard GUARD_MLOPEN_KERNELS_HPP_ -target ${PROJECT_BINARY_DIR}/include/mlopen_kernels.h -source ${MLOPEN_KERNELS}
		)
	
	add_dependencies(MLOpen addkernels_run)
	add_dependencies(tidy addkernels_run)
endif()

clang_tidy_check(MLOpen)
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
if( MLOPEN_BACKEND STREQUAL "OpenCL")
	MESSAGE( STATUS "MLOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
	target_include_directories(MLOpen SYSTEM PUBLIC ${OPENCL_INCLUDE_DIRS} )
	target_link_libraries( MLOpen ${OPENCL_LIBRARIES} )
elseif(MLOPEN_BACKEND STREQUAL "HIPOC")
	target_link_libraries( MLOpen ${hip_LIBRARIES} )
endif()

############################################################
# MLOpen depends on tinygemm
if(tinygemm_FOUND)
	target_link_libraries( MLOpen tinygemm )
endif()
