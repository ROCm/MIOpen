cmake_minimum_required( VERSION 2.8.12 )

include (GenerateExportHeader)

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${PROJECT_SOURCE_DIR}/../include/miopen-version.h.in" "${PROJECT_BINARY_DIR}/include/miopen-version.h" )

message( STATUS "MIOpen_VERSION= ${MIOpen_VERSION}" )
message( STATUS "CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}" )

# This is incremented when the ABI to the library changes
set( MIOpen_SOVERSION 1 )

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
set( MIOpen_Source
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
	include/miopen/common.hpp
	include/miopen/convolution.hpp
	include/miopen/convolution_fft.hpp
	include/miopen/errors.hpp
	include/miopen/handle.hpp
	include/miopen/kernel_cache.hpp
	include/miopen/mlo_internal.hpp
	include/miopen/mlo_utils.hpp
	include/miopen/oclkernel.hpp
	include/miopen/tensor.hpp
	include/miopen/tensor_ops.hpp
	include/miopenTypes.h
	include/miopen/pooling.hpp
	include/miopen/lrn.hpp
	include/miopen/activ.hpp
	include/miopen/softmax.hpp
	tensor.cpp
	tensor_api.cpp
	)

if( MIOPEN_BACKEND MATCHES "OpenCL" OR MIOPEN_BACKEND STREQUAL "HIPOC")
	set(MIOPEN_KERNELS
		Kernels/MIOpenConvDirUni.cl
		Kernels/MIOpenConvDirGenFwd.cl
		Kernels/MIOpenLRNBwd.cl
		Kernels/MIOpenLRNFwd.cl
		Kernels/MIOpenNeuron.cl
		Kernels/MIOpenPooling.cl
		Kernels/MIOpenPoolingBwd.cl
		Kernels/MIOpenConvDirUniC.cl		
		Kernels/MIOpenConv1x1Fwd.cl
		Kernels/MIOpenConv1x1Bwd.cl
		Kernels/MIOpenSoftmax.cl
		Kernels/MIOpenConvD3x3.cl
		Kernels/MIOpenUtilKernels.cl
		Kernels/MIOpenConvBwdWrW_LxL_P.cl
		Kernels/MIOpenConvBwdWrW_LxG_P53.cl
		Kernels/MIOpenConvBwdWrW_LxG_5x5.cl
		Kernels/MIOpenConvBwdWrW_GxL_1x1.cl
		Kernels/MIOpenConvFwd_LxL_11.cl
		Kernels/MIOpenConvFFT.cl
		Kernels/conv3x3.s
		Kernels/conv5x10u2v2f1.s
		Kernels/MIOpenTensorKernels.cl
		Kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m30.so
		Kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m21.so
		Kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m10.so
		Kernels/MIOpenConvBwdBias.cl)
	
	add_kernels("${MIOPEN_KERNELS}")
	configure_file(db.cpp.in db.cpp)
	list(APPEND MIOpen_Source 
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
	 DESTINATION share/miopen/db)
endif()

if(tinygemm_FOUND)
    list(APPEND MIOpen_Source 
        gemm.cpp
        gemm_api.cpp
        gemm_geometry.cpp
    )
endif()

if( MIOPEN_BACKEND STREQUAL "OpenCL" )
	list(APPEND MIOpen_Source 
		ocl/handleocl.cpp
        ocl_kernel.cpp
        ocl/oclerrors.cpp
		ocl/clhelper.cpp
	)
endif()

if( MIOPEN_BACKEND STREQUAL "HIPOC")
	list(APPEND MIOpen_Source 
		hip/hiperrors.cpp
		hip/handlehip.cpp
		hipoc/hipoc_kernel.cpp
		hipoc/hipoc_program.cpp
		)
endif()

if( MIOPEN_BACKEND STREQUAL "HIP")
	list(APPEND MIOpen_Source 
		hip/handlehip.cpp
		hip/convolutionhip.cpp
		hip/tensorhip.cpp
		)
endif()

set ( MIOpen_Headers_Public
	${CMAKE_SOURCE_DIR}/include/miopen.h
	${PROJECT_BINARY_DIR}/include/MIOpen-version.h
	${PROJECT_BINARY_DIR}/include/miopen_export.h
	)


# build library
add_library( MIOpen
	${MIOpen_Source}
	)

if( MIOPEN_BACKEND MATCHES "OpenCL" OR MIOPEN_BACKEND STREQUAL "HIPOC")
	add_custom_target(addkernels_run
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		DEPENDS addkernels
		COMMAND $<TARGET_FILE:addkernels> -guard GUARD_MIOPEN_KERNELS_HPP_ -target ${PROJECT_BINARY_DIR}/include/miopen_kernels.h -source ${MIOPEN_KERNELS}
		)
	
	add_dependencies(MIOpen addkernels_run)
	add_dependencies(tidy addkernels_run)
endif()

clang_tidy_check(MIOpen)
set( MIOpen_DIRS_INCLUDE
	# TODO: Remove this
	${PROJECT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include)

target_include_directories(MIOpen PUBLIC ${MIOpen_DIRS_INCLUDE})

install(TARGETS MIOpen 
	RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(FILES ${MIOpen_Headers_Public} DESTINATION include)

generate_export_header(MIOpen
	EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/miopen_export.h
)
############################################################
# MIOpen depends on OpenCL
if( MIOPEN_BACKEND STREQUAL "OpenCL")
	MESSAGE( STATUS "MIOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
	target_include_directories(MIOpen SYSTEM PUBLIC ${OPENCL_INCLUDE_DIRS} )
	target_link_libraries( MIOpen ${OPENCL_LIBRARIES} )
elseif(MIOPEN_BACKEND STREQUAL "HIPOC")
	target_link_libraries( MIOpen ${hip_LIBRARIES} )
endif()

############################################################
# MIOpen depends on tinygemm
if(tinygemm_FOUND)
	target_link_libraries( MIOpen tinygemm )
endif()
