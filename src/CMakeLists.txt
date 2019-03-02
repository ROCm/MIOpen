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
cmake_minimum_required( VERSION 2.8.12 )

cmake_policy(SET CMP0057 NEW)

include (GenerateExportHeader)

configure_file("${PROJECT_SOURCE_DIR}/include/miopen/config.h.in" "${PROJECT_BINARY_DIR}/include/miopen/config.h")

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${PROJECT_SOURCE_DIR}/include/miopen/version.h.in" "${PROJECT_BINARY_DIR}/include/miopen/version.h" )

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
    configure_file(kernels/kernel.cpp.in ${PROJECT_BINARY_DIR}/kernel.cpp)
endfunction()

set( MIOpen_Source
    check_numerics.cpp
    convolution.cpp
    convolution_api.cpp
    convolution_fft.cpp
    db.cpp
    db_record.cpp
    expanduser.cpp
    find_controls.cpp
    fusion.cpp
    op_args.cpp
    operator.cpp
    fused_api.cpp
	load_file.cpp
    pooling_api.cpp
    kernel_warnings.cpp
    logger.cpp
    lock_file.cpp
    lrn_api.cpp
    activ_api.cpp
    handle_api.cpp
    softmax_api.cpp
    batch_norm.cpp
    batch_norm_api.cpp
    rnn.cpp
    rnn_api.cpp
    temp_file.cpp
    problem_description.cpp
    kernel_build_params.cpp
    include/miopen/temp_file.hpp
    include/miopen/db.hpp
    include/miopen/db_record.hpp
    include/miopen/lock_file.hpp
    include/miopen/find_controls.hpp
    include/miopen/batch_norm.hpp
    include/miopen/check_numerics.hpp
    include/miopen/common.hpp
    include/miopen/convolution.hpp
    include/miopen/convolution_fft.hpp
    include/miopen/errors.hpp
    include/miopen/handle.hpp
    include/miopen/kernel_cache.hpp
    include/miopen/solver.hpp
    include/miopen/generic_search.hpp
    include/miopen/problem_description.hpp
    include/miopen/mlo_internal.hpp
    include/miopen/mlo_utils.hpp
    include/miopen/oclkernel.hpp
    include/miopen/tensor.hpp
    include/miopen/tensor_ops.hpp
    include/miopen/pooling.hpp
    include/miopen/lrn.hpp
    include/miopen/activ.hpp
    include/miopen/softmax.hpp
    include/miopen/rnn.hpp
    include/miopen/md_graph.hpp
    include/miopen/fusion_ops.hpp
    include/miopen/fusion.hpp
    include/miopen/mdg_expr.hpp
    include/miopen/kernel_build_params.hpp
    md_graph.cpp
    mdg_expr.cpp
    tensor.cpp
    tensor_api.cpp
    solver.cpp
    solver/conv_asm_3x3u.cpp
    solver/conv_asm_1x1u.cpp
    solver/conv_asm_1x1u_bias_activ.cpp
    solver/conv_asm_5x10u2v2f1.cpp
    solver/conv_asm_5x10u2v2b1.cpp
    solver/conv_asm_7x7c3h224w224k64u2v2p3q3f1.cpp
    solver/conv_asm_dir_BwdWrW3x3.cpp
    solver/conv_asm_dir_BwdWrW1x1.cpp
    solver/conv_bin_wino3x3U.cpp
    solver/conv_bin_winoRxS.cpp
    solver/conv_ocl_dir2D_bwdWrW_2.cpp
    solver/conv_ocl_dir2D_bwdWrW_53.cpp
    solver/conv_ocl_dir2D_bwdWrW_1x1.cpp
    solver/conv_ocl_dir2Dfwdgen.cpp
    solver/conv_ocl_dir2D11x11.cpp
    solver/conv_ocl_dir2D3x3.cpp
    solver/conv_ocl_dir2Dfwd_exhaustive_search.cpp
    solver/conv_ocl_dir2Dfwd.cpp
    solver/conv_ocl_dir2Dfwd1x1.cpp
    )

list(APPEND MIOpen_Source tmp_dir.cpp binary_cache.cpp md5.cpp)

if( MIOPEN_BACKEND MATCHES "OpenCL" OR MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP")
    set(MIOPEN_KERNEL_INCLUDES
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride1.inc
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dec.inc
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dil.inc
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride1.inc
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dec.inc
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dil.inc
        kernels/Conv_Winograd_v13_3_12_epilogue.inc
        kernels/Conv_Winograd_v13_3_12_prologue.inc
        kernels/Conv_Winograd_v16_3_0_epilogue.inc
        kernels/Conv_Winograd_v16_3_0_prologue.inc
        kernels/Conv_Winograd_v16_3_0_stride1.inc
        kernels/inst_wrappers.inc
        kernels/common.inc
        kernels/neuron.inc
        kernels/conv_sizes.inc
        kernels/gpr_alloc.inc)

    set(MIOPEN_KERNELS
        kernels/MIOpenCheckNumerics.cl
        kernels/MIOpenBatchNormActivBwdPerAct.cl
        kernels/MIOpenBatchNormActivBwdSpatial.cl
        kernels/MIOpenBatchNormActivFwdTrainPerAct.cl
        kernels/MIOpenBatchNormActivFwdTrainSpatial.cl
        kernels/MIOpenBatchNormFwdTrainSpatial.cl
        kernels/MIOpenBatchNormFwdTrainPerAct.cl
        kernels/MIOpenBatchNormFwdInferSpatial.cl
        kernels/MIOpenBatchNormFwdInferPerAct.cl
        kernels/MIOpenBatchNormBwdSpatial.cl
        kernels/MIOpenBatchNormBwdPerAct.cl
        kernels/MIOpenConvDirUni.cl
        kernels/MIOpenConvDirBatchNormActiv.cl
        kernels/MIOpenConvDirGenFwd.cl
        kernels/MIOpenLRNBwd.cl
        kernels/MIOpenLRNFwd.cl
        kernels/MIOpenNeuron.cl
        kernels/MIOpenPooling.cl
        kernels/MIOpenPoolingBwd.cl
        kernels/MIOpenConv1x1S.cl
        kernels/MIOpenConv1x1J1.cl
        kernels/MIOpenConv1x1J1_stride.cl
        kernels/MIOpenSoftmax.cl
        kernels/MIOpenConvD3x3.cl
        kernels/MIOpenUtilKernels.cl
        kernels/MIOpenUtilKernels2.cl
        kernels/MIOpenUtilKernels3.cl
        kernels/MIOpenUtilKernels4.cl
        kernels/MIOpenUtilKernels5.cl
        kernels/MIOpenConvBwdWrWS2.cl
        kernels/MIOpenGroupConvBwdWrWS2.cl
        kernels/MIOpenConvBwdWrW_LxG_P53.cl
        kernels/MIOpenGroupConvBwdWrW_LxG_P53.cl
        kernels/MIOpenConvBwdWrW_LxG_5x5.cl
        kernels/MIOpenConvBwdWrW1x1_PAD_read4.cl
        kernels/MIOpenConvFwd_LxL_11.cl
        kernels/MIOpenConvFFT.cl
        kernels/bugzilla_34765_detect.s
        kernels/dummy_kernel.s
        kernels/conv3x3.s
        kernels/conv1x1u.s
        kernels/conv1x1u_bias_activ.s
        kernels/conv3x3wrw.s
        kernels/conv1x1wrw.s
        kernels/conv5x10u2v2f1.s
        kernels/conv5x10u2v2b1.s
        kernels/conv7x7c3h224w224k64u2v2p3q3f1.s
        kernels/gcnAsmBNFwdTrainSpatial.s
        kernels/MIOpenTensorKernels.cl
        kernels/MIOpenSubTensorOpWithScalarKernel.cl
        kernels/MIOpenSubTensorOpWithSubTensorKernel.cl
        kernels/MIOpenSubTensorOpWithCastTensorKernel.cl
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride1.s
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dec.s
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dil.s
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride1.s
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dec.s
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dil.s
        kernels/Conv_Winograd_v16_3_0_stride1.s
        kernels/conv_3x3_wheel_alpha_v9_2_7_gfx803_md10.so
        kernels/conv_3x3_wheel_alpha_v9_2_7_gfx900_md10.so
        kernels/conv_3x3_wheel_alpha_v9_2_7_gfx906_md10.so
        kernels/conv_3x3_wheel_alpha_v9_2_7_stride_2_dec_gfx803_md10.so
        kernels/conv_3x3_wheel_alpha_v9_2_7_stride_2_dec_gfx900_md10.so
        kernels/conv_3x3_wheel_alpha_v9_2_7_stride_2_dec_gfx906_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx803_m30.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dil_gfx803_m30.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dec_gfx803_m30.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx900_m30.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dil_gfx900_m30.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dec_gfx900_m30.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx803_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dil_gfx803_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dec_gfx803_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx900_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dil_gfx900_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dec_gfx900_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx906_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dil_gfx906_md10.so
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dec_gfx906_md10.so
        kernels/conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10.so
        kernels/conv_3x3_wheel_alpha_v7_0_3b_gfx906_md10.so
        kernels/conv_3x3_wheel_alpha_v7_0_3b_gfx900.so
        kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_md10.so
        kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m30.so
        kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m21.so
        kernels/conv_3x3_wheel_alpha_v3_0b_gfx803_m10.so
        kernels/MIOpenConvBwdBias.cl
        kernels/MIOpenBatchNormActivInfer.cl)

    add_kernels("${MIOPEN_KERNELS}")
    configure_file(db_path.cpp.in ${PROJECT_BINARY_DIR}/db_path.cpp)
    list(APPEND MIOpen_Source
        activ.cpp
        kernel_cache.cpp
        lrn.cpp
        mlo_dir_conv.cpp
        ocl/activ_ocl.cpp
        ocl/batchnormocl.cpp
        ocl/convolutionocl.cpp
        ocl/convolutionocl_fft.cpp
        ocl/lrn_ocl.cpp
        ocl/mloNeuron.cpp
        ocl/mloNorm.cpp
        ocl/mloPooling.cpp
        ocl/pooling_ocl.cpp
        ocl/tensorocl.cpp
        ocl/softmaxocl.cpp
        ocl/rnnocl.cpp
        ocl/utilocl.cpp
        ocl/gcn_asm_utils.cpp
        pooling.cpp
        ocl/fusionopconvocl.cpp
        ocl/fusionopbiasbnactivocl.cpp
        ${PROJECT_BINARY_DIR}/db_path.cpp
        ${PROJECT_BINARY_DIR}/kernel.cpp
        )
endif()

if(miopengemm_FOUND)
    list(APPEND MIOpen_Source
        gemm.cpp
        gemm_v2.cpp
        miopengemm.cpp
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

if( MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP")
    list(APPEND MIOpen_Source
        hip/hiperrors.cpp
        hip/handlehip.cpp
        hipoc/hipoc_kernel.cpp
        hipoc/hipoc_program.cpp
        )
endif()

if( MIOPEN_BACKEND MATCHES "OpenCL" OR MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP")
    list(APPEND MIOpen_Source ${PROJECT_BINARY_DIR}/include/miopen_kernels.h)

    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/include/miopen_kernels.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS addkernels ${MIOPEN_KERNELS} ${MIOPEN_KERNEL_INCLUDES}
        COMMAND ${WINE_CMD} $<TARGET_FILE:addkernels> -guard GUARD_MIOPEN_KERNELS_HPP_ -target ${PROJECT_BINARY_DIR}/include/miopen_kernels.h -source ${MIOPEN_KERNELS}
        COMMENT "Inlining MIOpen kernels"
        )

    add_custom_target(miopen_tidy_inlining
        DEPENDS ${PROJECT_BINARY_DIR}/include/miopen_kernels.h
        )

    add_dependencies(tidy miopen_tidy_inlining)
endif()

# build library
add_library( MIOpen
    ${MIOpen_Source}
    )

set_target_properties(MIOpen PROPERTIES SOVERSION 1)

clang_tidy_check(MIOpen)

function(target_internal_library TARGET)
    target_link_libraries(${TARGET} PRIVATE ${ARGN})
    set(PASS_ARGS debug optimized)
    set(DEPS)
    foreach(DEP ${ARGN})
        if(DEP IN_LIST PASS_ARGS)
            list(APPEND DEPS ${DEP})
        else()
            list(APPEND DEPS $<BUILD_INTERFACE:${DEP}>)
        endif()
    endforeach()
    target_link_libraries(${TARGET} INTERFACE ${DEPS})
endfunction()

target_include_directories(MIOpen PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_include_directories(MIOpen SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

generate_export_header(MIOpen
    EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/miopen/export.h
)
set(PACKAGE_DEPENDS)
############################################################
# MIOpen depends on OpenCL
if( MIOPEN_BACKEND STREQUAL "OpenCL")
    MESSAGE( STATUS "MIOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
    target_include_directories(MIOpen SYSTEM PUBLIC ${OPENCL_INCLUDE_DIRS} )
    target_link_libraries( MIOpen PUBLIC ${OPENCL_LIBRARIES} )
    list(APPEND PACKAGE_DEPENDS PACKAGE OpenCL)
    set(BACKEND_PACKAGE "OpenCL")
elseif(MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP")
    target_link_libraries( MIOpen PRIVATE hip::hip_hcc hip::device )
    target_link_libraries( MIOpen INTERFACE hip::hip_hcc )
    if(ENABLE_HIP_WORKAROUNDS)
        # Workaround hip not setting its usage requirements correctly
        target_compile_definitions( MIOpen PRIVATE -D__HIP_PLATFORM_HCC__=1 )
    endif()
    # This is helpful for the tests
    target_link_libraries( MIOpen INTERFACE $<BUILD_INTERFACE:${hip_LIBRARIES}> )
    list(APPEND PACKAGE_DEPENDS PACKAGE hip)
    set(BACKEND_PACKAGE "hip")
endif()

############################################################
# MIOpen depends on miopengemm
if(miopengemm_FOUND)
    list(APPEND PACKAGE_DEPENDS PACKAGE miopengemm)
    target_internal_library(MIOpen miopengemm)
endif()

if(rocblas_FOUND)
    target_link_libraries( MIOpen INTERFACE $<BUILD_INTERFACE:roc::rocblas> )
    target_link_libraries( MIOpen PRIVATE roc::rocblas )
endif()

if(WIN32 AND NOT MSVC)
    if(BUILD_DEV)
        target_link_libraries(MIOpen PUBLIC -Wl,-export-all-symbols -Wl,-exclude-symbols=_Unwind_Resume)
    endif()
    target_link_libraries(MIOpen PUBLIC -Wl,--whole-archive -lgcc -lstdc++-6 -Wl,--no-whole-archive -Wl,--allow-multiple-definition)
endif()

target_include_directories(MIOpen PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
target_internal_library(MIOpen
    optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
    optimized ${Boost_SYSTEM_LIBRARY_RELEASE}
    debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
    debug ${Boost_SYSTEM_LIBRARY_DEBUG}
)
if(NOT WIN32 AND NOT APPLE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lib.def "
MIOPEN_${MIOPEN_BACKEND}_1
{
global:
    *;
local:
    *;
};
")
    target_link_libraries(MIOpen PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/lib.def")
    if(NOT BUILD_DEV)
        # set_target_properties(MIOpen PROPERTIES CXX_VISIBILITY_PRESET hidden)
        if(NOT APPLE)
            target_link_libraries(MIOpen PRIVATE "-Wl,--exclude-libs,ALL")
        endif()
    endif()
endif()

############################################################
# MIOpen depends on librt for Boost.Interprocess
if(NOT WIN32 AND NOT APPLE)
    find_library(LIBRT rt)
    if(LIBRT)
        MESSAGE(STATUS "Librt: " ${LIBRT})
        target_link_libraries(MIOpen PUBLIC ${LIBRT})
    endif()
endif()

############################################################
# Installation
rocm_install_targets(
  TARGETS MIOpen
  INCLUDE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  PREFIX ${MIOPEN_INSTALL_DIR}
 )

rocm_export_targets(
  TARGETS MIOpen
  PREFIX ${MIOPEN_INSTALL_DIR}
  DEPENDS
    ${PACKAGE_DEPENDS}
)


# Install db files
install(FILES
    kernels/gfx803_36.cd.pdb.txt
    kernels/gfx803_64.cd.pdb.txt
    kernels/gfx900_64.cd.pdb.txt
    kernels/gfx900_56.cd.pdb.txt
    kernels/gfx906_64.cd.pdb.txt
    kernels/gfx906_60.cd.pdb.txt
 DESTINATION ${DATA_INSTALL_DIR}/db)

rocm_install_symlink_subdir(${MIOPEN_INSTALL_DIR})
