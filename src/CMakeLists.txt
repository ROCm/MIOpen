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

cmake_policy(SET CMP0057 NEW)

include (GenerateExportHeader)
add_subdirectory(sqlite)

# Truncation rounding or (default) rounding to nearest even (RNE) is enabled.
# This switch controls two related but different aspects of MIOpen behavior
# 1.  How host code performs conversions of float to bfloat16, important only
#     for testing.
# 2.  How BF16 kernels (which are kind of mixed-precision now and expected to
#     remain in the future)  perform final conversion (and rounding) of FP32
#     to BF16 results. This affects the main functionality of the library.
option( MIOPEN_USE_RNE_BFLOAT16 "Sets rounding scheme for bfloat16 type" ON )
set ( MIOPEN_DEFAULT_FIND_MODE "DynamicHybrid" CACHE STRING "Sets the default find mode")
set_property(CACHE MIOPEN_DEFAULT_FIND_MODE PROPERTY STRINGS 
    Normal Fast Hybrid FastHybrid DynamicHybrid)

configure_file("${PROJECT_SOURCE_DIR}/include/miopen/config.h.in" "${PROJECT_BINARY_DIR}/include/miopen/config.h")

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${PROJECT_SOURCE_DIR}/include/miopen/version.h.in" "${PROJECT_BINARY_DIR}/include/miopen/version.h" )


if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
#Copy header to Build Dir for generating backward compatibility
configure_file( "${PROJECT_SOURCE_DIR}/include/miopen/miopen.h" "${PROJECT_BINARY_DIR}/include/miopen/miopen.h" )
endif()

message( STATUS "MIOpen_VERSION= ${MIOpen_VERSION}" )
message( STATUS "CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}" )

# This is incremented when the ABI to the library changes
set( MIOpen_SOVERSION 1.0 )


function(add_kernels FILE_NAME VAR_PREFIX VAR_SUFFIX KERNEL_FILES)
    set(INIT_KERNELS_LIST)
    set(KERNELS_DECLS)
    foreach(KERNEL_FILE ${KERNEL_FILES})
        if("${CMAKE_VERSION}" VERSION_LESS 3.0)
            configure_file(${KERNEL_FILE} ${KERNEL_FILE}.delete)
        else()
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
        endif()
        get_filename_component(KERNEL_FILENAME ${KERNEL_FILE} NAME)
        get_filename_component(BASE_NAME ${KERNEL_FILE} NAME_WE)
        string(TOUPPER "${BASE_NAME}" KEY_NAME)
        string(MAKE_C_IDENTIFIER "${KEY_NAME}" VAR_NAME)
        string(APPEND KERNELS_DECLS "extern const size_t ${VAR_PREFIX}${VAR_NAME}${VAR_SUFFIX}_SIZE;\n")
        string(APPEND KERNELS_DECLS "extern const unsigned char ${VAR_PREFIX}${VAR_NAME}${VAR_SUFFIX}[];\n")
        list(APPEND INIT_KERNELS_LIST "    { \"${KERNEL_FILENAME}\", std::string(reinterpret_cast<const char*>(${VAR_PREFIX}${VAR_NAME}${VAR_SUFFIX}), ${VAR_PREFIX}${VAR_NAME}${VAR_SUFFIX}_SIZE) }")
    endforeach()
    string(REPLACE ";" ",\n" INIT_KERNELS "${INIT_KERNELS_LIST}")
    configure_file(kernels/${FILE_NAME}.in ${PROJECT_BINARY_DIR}/${FILE_NAME})
endfunction()

set( MIOpen_Source
    activ/problem_description.cpp
    activ_api.cpp
    batch_norm.cpp
    batch_norm_api.cpp
    batchnorm/problem_description.cpp
    buffer_info.cpp
    check_numerics.cpp
    conv/invokers/gcn_asm_1x1u.cpp
    conv/invokers/gcn_asm_1x1u_ss.cpp
    conv/invokers/gcn_asm_1x1u_us.cpp
    conv/invokers/gen_x_w_y_pad.cpp
    conv/invokers/impl_gemm.cpp
    conv/invokers/impl_gemm_dynamic.cpp
    conv/invokers/ocl_wrw_rdc.cpp
    conv/problem_description.cpp
    conv_algo_name.cpp
    convolution.cpp
    convolution_api.cpp
    ctc.cpp
    ctc_api.cpp
    db.cpp
    db_record.cpp
    dropout.cpp
    dropout_api.cpp
    execution_context.cpp
    expanduser.cpp
    find_controls.cpp
    find_db.cpp
    fused_api.cpp
    fusion.cpp
    handle_api.cpp
    invoker_cache.cpp
    kernel_build_params.cpp
    kernel_warnings.cpp
    load_file.cpp
    lock_file.cpp
    logger.cpp
    lrn_api.cpp
    md_graph.cpp
    mdg_expr.cpp
    op_args.cpp
    operator.cpp
    pooling/problem_description.cpp
    pooling_api.cpp
    problem_description.cpp
    ramdb.cpp
    readonlyramdb.cpp
    reducetensor.cpp
    reducetensor_api.cpp
    rnn.cpp
    rnn_api.cpp
    softmax_api.cpp
    solver.cpp
    solver/activ/bwd_0.cpp
    solver/activ/bwd_1.cpp
    solver/activ/fwd_0.cpp
    solver/activ/fwd_1.cpp
    solver/batchnorm/backward_per_activation.cpp
    solver/batchnorm/backward_spatial_multiple.cpp
    solver/batchnorm/backward_spatial_single.cpp
    solver/batchnorm/forward_inference.cpp
    solver/batchnorm/forward_per_activation.cpp
    solver/batchnorm/forward_spatial_multiple.cpp
    solver/batchnorm/forward_spatial_single.cpp
    solver/conv_asm_1x1u.cpp
    solver/conv_asm_1x1u_bias_activ.cpp
    solver/conv_asm_1x1u_stride2.cpp
    solver/conv_asm_3x3u.cpp
    solver/conv_asm_5x10u2v2b1.cpp
    solver/conv_asm_5x10u2v2f1.cpp
    solver/conv_asm_7x7c3h224w224k64u2v2p3q3f1.cpp
    solver/conv_asm_dir_BwdWrW1x1.cpp
    solver/conv_asm_dir_BwdWrW3x3.cpp
    solver/conv_asm_implicit_gemm_bwd_v4r1_dynamic.cpp
    solver/conv_asm_implicit_gemm_gtc_bwd.cpp
    solver/conv_asm_implicit_gemm_gtc_bwd_nhwc.cpp
    solver/conv_asm_implicit_gemm_gtc_fwd.cpp
    solver/conv_asm_implicit_gemm_gtc_fwd_nhwc.cpp
    solver/conv_asm_implicit_gemm_gtc_perf_config.cpp
    solver/conv_asm_implicit_gemm_gtc_wrw_nhwc.cpp
    solver/conv_asm_implicit_gemm_v4r1_dynamic.cpp
    solver/conv_asm_implicit_gemm_wrw_gtc_dynamic_xdlops.cpp
    solver/conv_asm_implicit_gemm_wrw_v4r1_dynamic.cpp
    solver/conv_bin_wino3x3U.cpp
    solver/conv_bin_winoRxS.cpp
    solver/conv_ck_igemm_fwd_v6r1_dlops_nchw.cpp
    solver/conv_direct_naive_conv.cpp
    solver/conv_direct_naive_conv_bwd.cpp
    solver/conv_direct_naive_conv_fwd.cpp
    solver/conv_direct_naive_conv_wrw.cpp
    solver/conv_hip_implicit_gemm_bwd_v1r1.cpp
    solver/conv_hip_implicit_gemm_bwd_v1r1_xdlops.cpp
    solver/conv_hip_implicit_gemm_bwd_v4r1.cpp
    solver/conv_hip_implicit_gemm_bwd_v4r1_xdlops.cpp
    solver/conv_hip_implicit_gemm_fwd_v4r1.cpp
    solver/conv_hip_implicit_gemm_fwd_v4r4.cpp
    solver/conv_hip_implicit_gemm_fwd_v4r4_xdlops.cpp
    solver/conv_hip_implicit_gemm_fwd_v4r4_xdlops_padded_gemm.cpp
    solver/conv_hip_implicit_gemm_fwd_v4r5_xdlops.cpp
    solver/conv_hip_implicit_gemm_nonxdlops_common.cpp
    solver/conv_hip_implicit_gemm_wrw_v4r4.cpp
    solver/conv_hip_implicit_gemm_wrw_v4r4_xdlops.cpp
    solver/conv_hip_implicit_gemm_wrw_v4r4_xdlops_padded_gemm.cpp
    solver/conv_hip_implicit_gemm_xdlops_common.cpp
    solver/conv_mlir_igemm_bwd.cpp
    solver/conv_mlir_igemm_bwd_xdlops.cpp
    solver/conv_mlir_igemm_fwd.cpp
    solver/conv_mlir_igemm_fwd_xdlops.cpp
    solver/conv_mlir_igemm_wrw.cpp
    solver/conv_mlir_igemm_wrw_xdlops.cpp
    solver/conv_MP_bidirectional_winograd.cpp
    solver/conv_multipass_wino3x3WrW.cpp
    solver/conv_ocl_dir2D_bwdWrW_1x1.cpp
    solver/conv_ocl_dir2D_bwdWrW_2.cpp
    solver/conv_ocl_dir2D_bwdWrW_53.cpp
    solver/conv_ocl_dir2D11x11.cpp
    solver/conv_ocl_dir2Dfwd.cpp
    solver/conv_ocl_dir2Dfwd_exhaustive_search.cpp
    solver/conv_ocl_dir2Dfwd1x1.cpp
    solver/conv_ocl_dir2Dfwdgen.cpp
    solver/conv_winoRxS.cpp
    solver/fft.cpp
    solver/gemm.cpp
    solver/gemm_bwd.cpp
    solver/gemm_common.cpp
    solver/gemm_wrw.cpp
    solver/pooling/forward2d.cpp
    solver/pooling/forwardNd.cpp
    solver/pooling/backward2d.cpp
    solver/pooling/backwardNd.cpp
    subbuffers.cpp
    target_properties.cpp
    temp_file.cpp
    tensor.cpp
    tensor_api.cpp
    )

list(APPEND MIOpen_Source tmp_dir.cpp binary_cache.cpp md5.cpp)
if(MIOPEN_ENABLE_SQLITE)
    list(APPEND MIOpen_Source sqlite_db.cpp)
endif()

if(MIOPEN_ENABLE_SQLITE AND MIOPEN_ENABLE_SQLITE_KERN_CACHE)
    list(APPEND MIOpen_Source kern_db.cpp bz2.cpp)
endif()

if( MIOPEN_BACKEND MATCHES "OpenCL" OR MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP" OR MIOPEN_BACKEND STREQUAL "HIPNOGPU")
    file(GLOB_RECURSE STATIC_COMPOSABLE_KERNEL_INCLUDE "kernels/static_composable_kernel/include/*/*.hpp")
    file(GLOB_RECURSE STATIC_COMPOSABLE_KERNEL_SOURCE "kernels/static_composable_kernel/src/*/*.cpp")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_INCLUDE "composable_kernel/composable_kernel/include/*.hpp")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_SOURCE "composable_kernel/composable_kernel/src/*.cpp")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_ASM_SOURCE "kernels/dynamic_igemm/*.s")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_ASM_INCLUDE "kernels/dynamic_igemm/*.inc")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_CPP_SOURCE "kernels/dynamic_igemm/*.cpp")
    file(GLOB_RECURSE GPU_REFERENCE_KERNEL_HIP "kernels/gpu_reference_kernel/*.cpp")
    file(GLOB_RECURSE GPU_REFERENCE_KERNEL_ASM "kernels/gpu_reference_kernel/*.s")
    file(GLOB_RECURSE GPU_BATCHED_TRANSPOSE_KERNEL_HIP "kernels/gpu_batched_transpose_kernel/*.cpp")
    file(GLOB_RECURSE GPU_GENERAL_TENSOR_REORDER_KERNEL_HIP_INCLUDE "kernels/gpu_general_tensor_reorder_kernel/*.hpp")
    file(GLOB_RECURSE GPU_GENERAL_TENSOR_REORDER_KERNEL_HIP_SOURCE "kernels/gpu_general_tensor_reorder_kernel/*.cpp")


    set(MIOPEN_KERNEL_INCLUDES
        ${STATIC_COMPOSABLE_KERNEL_INCLUDE}
        ${COMPOSABLE_KERNEL_INCLUDE}
        ${COMPOSABLE_KERNEL_DYNAMIC_ASM_INCLUDE}
        ${GPU_GENERAL_TENSOR_REORDER_KERNEL_HIP_INCLUDE}
        include/miopen/implicitgemm_params.hpp
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride1.inc
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dec.inc
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dil.inc
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride1.inc
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dec.inc
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dil.inc
        kernels/Conv_Winograd_v13_3_12_epilogue.inc
        kernels/Conv_Winograd_v13_3_12_prologue.inc
        kernels/Conv_Winograd_v16_5_0_epilogue.inc
        kernels/Conv_Winograd_v16_5_0_prologue.inc
        kernels/Conv_Winograd_v16_5_0_stride1.inc
        kernels/conv_3x3_wheel_alpha_v9_2_7_epilogue.inc
        kernels/conv_3x3_wheel_alpha_v9_2_7_prologue.inc
        kernels/conv_3x3_wheel_alpha_v9_2_7_gfx8_stride_2_dec.inc
        kernels/conv_3x3_wheel_alpha_v9_2_7_gfx8.inc
        kernels/conv_3x3_wheel_alpha_v3_0b_epilogue.inc
        kernels/conv_3x3_wheel_alpha_v3_0b_prologue.inc
        kernels/conv_3x3_wheel_alpha_v3_0b.inc
        kernels/conv_3x3_wheel_alpha_v7_0_3b_epilogue.inc
        kernels/conv_3x3_wheel_alpha_v7_0_3b_prologue.inc
        kernels/conv_3x3_wheel_alpha_v7_0_3b.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_epilogue.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_prologue.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx8_stride_2_dil.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx8_stride_2_dec.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx8.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx9_stride_2_dil.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx9_stride_2_dec.inc
        kernels/conv_3x3_wheel_alpha_v9_0_15_gfx9.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_f3x2_fp16_dot2_edc_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_f3x2_fp16_dot2_edc_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_f3x2_fp16_dot2_edc_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_f3x2_fp32_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_f3x2_fp32_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_f3x2_fp32_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp16_dot2_edc_dilation2.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp16_dot2_edc_stride1.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp16_dot2_edc_stride2.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp32_dilation2.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp32_stride1.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp32_stride2.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp16_dot2_edc_dilation2.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp16_dot2_edc_stride1.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp16_dot2_edc_stride2.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp32_dilation2.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp32_stride1.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp32_stride2.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp16_dot2_edc_dilation2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp16_dot2_edc_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp16_dot2_edc_stride2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp32_dilation2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp32_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx9_fp32_stride2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp16_dot2_edc_dilation2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp16_dot2_edc_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp16_dot2_edc_stride2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp32_dilation2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp32_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx90a_fp32_stride2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp16_dot2_edc_dilation2.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp16_dot2_edc_stride1.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp16_dot2_edc_stride2.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp32_dilation2.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp32_stride1.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp32_stride2.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp16_dot2_edc_dilation2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp16_dot2_edc_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp16_dot2_edc_stride2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp32_dilation2_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp32_stride1_group.inc
        kernels/Conv_Winograd_v21_1_3_gfx10_fp32_stride2_group.inc
        kernels/Conv_Winograd_v21_1_3_metadata.inc
        kernels/xform_bidirect_winograd_code.inc
        kernels/rocm_version.inc
        kernels/inst_wrappers.inc
        kernels/conv_common.inc
        kernels/utilities.inc
        kernels/xform_data_filter.inc
        kernels/xform_kd_cov2.inc
        kernels/xform_metadata.inc
        kernels/neuron.inc
        kernels/conv_sizes.inc
        kernels/gpr_alloc.inc
        kernels/bfloat16_dev.hpp
        kernels/float_types.h
        kernels/workaround_issue_1431.hpp
        )

    set(MIOPEN_KERNELS
        ${STATIC_COMPOSABLE_KERNEL_SOURCE}
        ${COMPOSABLE_KERNEL_SOURCE}
        ${COMPOSABLE_KERNEL_DYNAMIC_ASM_SOURCE}
        ${COMPOSABLE_KERNEL_DYNAMIC_CPP_SOURCE}
        ${GPU_REFERENCE_KERNEL_HIP}
        ${GPU_REFERENCE_KERNEL_ASM}
        ${GPU_BATCHED_TRANSPOSE_KERNEL_HIP}
        ${GPU_GENERAL_TENSOR_REORDER_KERNEL_HIP_SOURCE}
        kernels/detect_llvm_amdgcn_buffer_atomic_fadd_f32_float.cpp
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
        kernels/MIOpenPoolingND.cl
        kernels/MIOpenPoolingBwdND.cl
        kernels/MIOpenConv1x1S.cl
        kernels/MIOpenConv1x1J1.cl
        kernels/MIOpenConv1x1J1_stride.cl
        kernels/MIOpenSoftmax.cl
        kernels/MIOpenUtilKernels3.cl
        kernels/MIOpenUtilKernels4.cl
        kernels/MIOpenUtilKernels5.cl
        kernels/MIOpenIm2d2Col.cl
        kernels/MIOpenIm3d2Col.cl
        kernels/MIOpenCol2Im2d.cl
        kernels/MIOpenCol2Im3d.cl
        kernels/MIOpenConvBwdWrWS2.cl
        kernels/MIOpenGroupConvBwdWrWS2.cl
        kernels/MIOpenConvBwdWrW_LxG_P53.cl
        kernels/MIOpenGroupConvBwdWrW_LxG_P53.cl
        kernels/MIOpenConvBwdWrW_LxG_5x5.cl
        kernels/MIOpenConvBwdWrW1x1_PAD_read4.cl
        kernels/MIOpenConvFwd_LxL_11.cl
        kernels/MIOpenConvFFT.cl
        kernels/MIOpenRNNHiddenStateUpdate.cl
        kernels/bugzilla_34765_detect.s
        kernels/dummy_kernel.s
        kernels/conv3x3.s
        kernels/conv1x1u.s
        kernels/conv1x1u_stride2.s
        kernels/conv1x1u_bias_activ.s
        kernels/conv3x3wrw.s
        kernels/conv1x1wrw.s
        kernels/conv5x10u2v2f1.s
        kernels/conv5x10u2v2b1.s
        kernels/conv7x7c3h224w224k64u2v2p3q3f1.s
        kernels/xform_out.s
        kernels/gcnAsmBNBwdTrainSpatial.s
        kernels/MIOpenTensorKernels.cl
        kernels/MIOpenSubTensorOpWithScalarKernel.cl
        kernels/MIOpenSubTensorOpWithSubTensorKernel.cl
        kernels/MIOpenSubTensorOpWithCastTensorKernel.cl
        kernels/MIOpenSubTensorOpWithTransformKernel.cl
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride1.s
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dec.s
        kernels/Conv_Winograd_v13_3_12_fp16dot_stride2_dil.s
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride1.s
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dec.s
        kernels/Conv_Winograd_v14_3_3_fp16dot_stride2_dil.s
        kernels/Conv_Winograd_v16_5_0_stride1.s
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dil.s
        kernels/conv_3x3_wheel_alpha_v9_0_15_stride_2_dec.s
        kernels/conv_3x3_wheel_alpha_v9_0_15.s
        kernels/conv_3x3_wheel_alpha_v7_0_3b.s
        kernels/conv_3x3_wheel_alpha_v3_0b.s
        kernels/conv_3x3_wheel_alpha_v9_2_7.s
        kernels/conv_3x3_wheel_alpha_v9_2_7_stride_2_dec.s
        kernels/Conv_Winograd_v21_1_3_f3x2_fp16_dot2_edc_stride1_group.s
        kernels/Conv_Winograd_v21_1_3_f3x2_fp32_stride1_group.s
        kernels/Conv_Winograd_v21_1_3_fp16_dot2_edc_dilation2.s
        kernels/Conv_Winograd_v21_1_3_fp16_dot2_edc_stride1.s
        kernels/Conv_Winograd_v21_1_3_fp16_dot2_edc_stride2.s
        kernels/Conv_Winograd_v21_1_3_fp32_dilation2.s
        kernels/Conv_Winograd_v21_1_3_fp32_stride1.s
        kernels/Conv_Winograd_v21_1_3_fp32_stride2.s
        kernels/Conv_Winograd_v21_1_3_fp16_dot2_edc_dilation2_group.s
        kernels/Conv_Winograd_v21_1_3_fp16_dot2_edc_stride1_group.s
        kernels/Conv_Winograd_v21_1_3_fp16_dot2_edc_stride2_group.s
        kernels/Conv_Winograd_v21_1_3_fp32_dilation2_group.s
        kernels/Conv_Winograd_v21_1_3_fp32_stride1_group.s
        kernels/Conv_Winograd_v21_1_3_fp32_stride2_group.s
        kernels/MIOpenConvBwdBias.cl
        kernels/MIOpenBatchNormActivInfer.cl
        kernels/MIOpenCTCLoss.cl
        kernels/MIOpenDropout.cl
        kernels/xform_data.s
        kernels/xform_filter.s
        kernels/xform_bidirect_winograd_data.s
        kernels/xform_bidirect_winograd_filter.s
        kernels/xform_bidirect_winograd_out.s
        kernels/UniversalTranspose.cl)

    # Kernels in development lists.
    # Should be ALWAYS empty in develop branch (at the time of PR merge)
    # Intention: to speed up kernel development rebuild time
    set(MIOPEN_DEVELOPMENT_KERNELS)

    # Only referenced by MIOPEN_DEVELOPMENT_KERNELS
    set(MIOPEN_DEVELOPMENT_KERNEL_INCLUDES)

    LIST(LENGTH MIOPEN_DEVELOPMENT_KERNELS MIOPEN_DEVELOPMENT_KERNELS_COUNT)
    LIST(LENGTH MIOPEN_DEVELOPMENT_KERNEL_INCLUDES MIOPEN_DEVELOPMENT_KERNEL_INCLUDES_COUNT)

    add_kernels("kernel.cpp" "MIOPEN_KERNEL_" "" "${MIOPEN_KERNELS}")
    add_kernels("kernel_includes.cpp" "MIOPEN_KERNEL_" "_INCLUDE" "${MIOPEN_KERNEL_INCLUDES}")

    if(${MIOPEN_DEVELOPMENT_KERNELS_COUNT})
        add_kernels("kernel.cpp" "MIOPEN_KERNEL_" "" "${MIOPEN_DEVELOPMENT_KERNELS}")
    endif()
    
    if(${MIOPEN_DEVELOPMENT_KERNEL_INCLUDES_COUNT})
        add_kernels("kernel_includes.cpp" "MIOPEN_KERNEL_" "_INCLUDE" "${MIOPEN_DEVELOPMENT_KERNEL_INCLUDES}")
    endif()

    configure_file(db_path.cpp.in ${PROJECT_BINARY_DIR}/db_path.cpp)
    list(APPEND MIOpen_Source
        activ.cpp
        kernel_cache.cpp
        lrn.cpp
        mlo_dir_conv.cpp
        exec_utils.cpp
        ocl/activ_ocl.cpp
        ocl/batchnormocl.cpp
        ocl/convolutionocl.cpp
        ocl/lrn_ocl.cpp
        ocl/mloNeuron.cpp
        ocl/mloNorm.cpp
        ocl/pooling_ocl.cpp
        ocl/tensorocl.cpp
        ocl/softmaxocl.cpp
        ocl/rnnocl.cpp
        ocl/utilocl.cpp
        ocl/ctcocl.cpp
        ocl/dropoutocl.cpp
        ocl/gcn_asm_utils.cpp
        ocl/rnn_util_ocl.cpp
        hip/hip_build_utils.cpp
        hip/batched_transpose_sol.cpp
        hip/general_tensor_reorder_sol.cpp
        pooling.cpp
        ocl/fusionopconvocl.cpp
        ocl/fusionopbiasbnactivocl.cpp
        ${PROJECT_BINARY_DIR}/db_path.cpp
        )

    list(INSERT MIOpen_Source 0
        ${PROJECT_BINARY_DIR}/kernel.cpp
        ${PROJECT_BINARY_DIR}/kernel_includes.cpp
        )
endif()

if(miopengemm_FOUND OR MIOPEN_USE_ROCBLAS OR MIOPEN_USE_MIOPENTENSILE)
    list(APPEND MIOpen_Source
        gemm_v2.cpp
        miopengemm.cpp
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

if( MIOPEN_BACKEND STREQUAL "HIPNOGPU")
    list(APPEND MIOpen_Source
        hip/hiperrors.cpp
        nogpu/handle.cpp
        hipoc/hipoc_kernel.cpp
        hipoc/hipoc_program.cpp
        )
endif()

if( MIOPEN_BACKEND MATCHES "OpenCL" OR MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP" OR MIOPEN_BACKEND STREQUAL "HIPNOGPU")
    set(KERNELS_SRC_BATCH_FACTOR 50 CACHE STRING "Amount of kernel source files to inline to a single object file.")
    set(KERNELS_BATCH_ID 0)

    function(inline_kernels_src BATCH_FACTOR KERNELS KERNEL_INCLUDES EXTRA_OPTIONS MESSAGE_SUFFIX)
        set(KERNELS_BATCH)
        set(KERNELS_BATCH_SIZE 0)
        set(PROCESSED 0)
        list(LENGTH KERNELS KERNELS_NUMBER)

        foreach(KERNEL ${KERNELS})
            list(APPEND KERNELS_BATCH ${KERNEL})
            list(LENGTH KERNELS_BATCH KERNELS_BATCH_SIZE)
            math(EXPR PROCESSED "1+${PROCESSED}")
            if((KERNELS_BATCH_SIZE EQUAL ${BATCH_FACTOR}) OR (PROCESSED EQUAL KERNELS_NUMBER))
                set(KERNEL_SRC_HPP_FILENAME batch_${KERNELS_BATCH_ID}.cpp.hpp)
                set(KERNEL_SRC_HPP_PATH ${PROJECT_BINARY_DIR}/inlined_kernels/${KERNEL_SRC_HPP_FILENAME})
                set(KERNEL_SRC_CPP_PATH ${PROJECT_BINARY_DIR}/inlined_kernels/batch_${KERNELS_BATCH_ID}.cpp)

                add_custom_command(
                    OUTPUT ${KERNEL_SRC_HPP_PATH}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    DEPENDS addkernels ${KERNELS_BATCH} ${KERNEL_INCLUDES}
                    COMMAND ${WINE_CMD} $<TARGET_FILE:addkernels> -target ${KERNEL_SRC_HPP_PATH} -extern ${EXTRA_OPTIONS} -source ${KERNELS_BATCH}
                    COMMENT "Inlining kernels batch #${KERNELS_BATCH_ID}${MESSAGE_SUFFIX}"
                    )
                configure_file(kernels/kernels_batch.cpp.in ${KERNEL_SRC_CPP_PATH})
                list(APPEND MIOpen_Source ${KERNEL_SRC_CPP_PATH} ${KERNEL_SRC_HPP_PATH})

                set(KERNELS_BATCH)
                math(EXPR KERNELS_BATCH_ID "1+${KERNELS_BATCH_ID}")
            endif()
        endforeach()

        set(KERNELS_BATCH_ID ${KERNELS_BATCH_ID} PARENT_SCOPE)
        set(MIOpen_Source ${MIOpen_Source} PARENT_SCOPE)
    endfunction()

    inline_kernels_src(${KERNELS_SRC_BATCH_FACTOR} "${MIOPEN_KERNELS}" "${MIOPEN_KERNEL_INCLUDES}" "" "")
    inline_kernels_src(${KERNELS_SRC_BATCH_FACTOR} "${MIOPEN_KERNEL_INCLUDES}" "" "-no-recurse;-mark-includes" " (includes)")

    set(MIOPEN_DEVELOPMENT_KERNELS_DEPS ${MIOPEN_KERNEL_INCLUDES})
    list(APPEND MIOPEN_DEVELOPMENT_KERNELS_DEPS ${MIOPEN_DEVELOPMENT_KERNEL_INCLUDES})

    if(${MIOPEN_DEVELOPMENT_KERNELS_COUNT})
        inline_kernels_src(${KERNELS_SRC_BATCH_FACTOR} "${MIOPEN_DEVELOPMENT_KERNELS}" "${MIOPEN_DEVELOPMENT_KERNELS_DEPS}" "" " (dev kernels)")
    endif()
    
    if(${MIOPEN_DEVELOPMENT_KERNEL_INCLUDES_COUNT})
        inline_kernels_src(${KERNELS_SRC_BATCH_FACTOR} "${MIOPEN_DEVELOPMENT_KERNEL_INCLUDES}" "" "-no-recurse;-mark-includes" " (dev includes)")
    endif()

endif()

if(MIOPEN_USE_COMGR)
    list(APPEND MIOpen_Source comgr.cpp)
endif()

if(MIOPEN_USE_MLIR)
    list(APPEND MIOpen_Source
        mlir_build.cpp
        solver/mlir_common.cpp
        conv/invokers/mlir_impl_gemm.cpp
        )
endif()

# build library
add_library( MIOpen
    ${MIOpen_Source}
    $<TARGET_OBJECTS:sqlite_memvfs>
    )

rocm_set_soversion(MIOpen ${MIOpen_SOVERSION})

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
target_include_directories(MIOpen SYSTEM PRIVATE ${BZIP2_INCLUDE_DIR})
target_link_libraries(MIOpen PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${BZIP2_LIBRARIES})
generate_export_header(MIOpen
    EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/miopen/export.h
)
set(PACKAGE_DEPENDS)
set(PACKAGE_STATIC_DEPENDS)
############################################################
# MIOpen depends on OpenCL
if( MIOPEN_BACKEND STREQUAL "OpenCL")
    MESSAGE( STATUS "MIOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
    target_include_directories(MIOpen SYSTEM PUBLIC ${OPENCL_INCLUDE_DIRS} )
    target_link_libraries( MIOpen PUBLIC ${OPENCL_LIBRARIES} )
    list(APPEND PACKAGE_DEPENDS PACKAGE OpenCL)
elseif(MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP")
    target_link_libraries( MIOpen PRIVATE hip::device )
    target_link_libraries( MIOpen INTERFACE hip::host )
    if(ENABLE_HIP_WORKAROUNDS)
        # Workaround hip not setting its usage requirements correctly
        target_compile_definitions( MIOpen PRIVATE -D__HIP_PLATFORM_HCC__=1 )
    endif()
    # This is helpful for the tests
    target_link_libraries( MIOpen INTERFACE $<BUILD_INTERFACE:hip::device> )
    list(APPEND PACKAGE_DEPENDS PACKAGE hip)
endif()

############################################################
# MIOpen depends on miopengemm
if(miopengemm_FOUND)
    list(APPEND PACKAGE_DEPENDS PACKAGE miopengemm)
    target_internal_library(MIOpen miopengemm)
    list(APPEND PACKAGE_STATIC_DEPENDS PACKAGE miopengemm)
endif()

if(MIOPEN_USE_COMGR)
    list(APPEND PACKAGE_DEPENDS PACKAGE amd_comgr)
    target_internal_library(MIOpen amd_comgr)
endif()

if(rocblas_FOUND)
    target_link_libraries( MIOpen INTERFACE $<BUILD_INTERFACE:roc::rocblas> )
    target_link_libraries( MIOpen PRIVATE roc::rocblas )
    list(APPEND PACKAGE_STATIC_DEPENDS PACKAGE rocblas)
endif()

# MIOpen depends on miopentensile
if(miopentensile_FOUND)
    target_link_libraries(MIOpen PRIVATE MIOpenTensile)
endif()

if(LIBMLIRMIOPEN)
  target_link_libraries(MIOpen PRIVATE ${LIBMLIRMIOPEN})
endif()

if(WIN32 AND NOT MSVC)
    if(BUILD_DEV)
        target_link_libraries(MIOpen PUBLIC -Wl,-export-all-symbols -Wl,-exclude-symbols=_Unwind_Resume)
    endif()
    target_link_libraries(MIOpen PUBLIC -Wl,--whole-archive -lgcc -lstdc++-6 -Wl,--no-whole-archive -Wl,--allow-multiple-definition)
endif()

target_internal_library(MIOpen
    Boost::filesystem
)
list(APPEND PACKAGE_STATIC_DEPENDS PACKAGE Boost COMPONENTS filesystem)
if(NOT WIN32 AND NOT APPLE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lib.def "
MIOPEN_${MIOPEN_BACKEND}_1
{
global:
    miopen*;
    extern \"C++\" {
        miopen::*;
    };
local:
    *boost*;
    extern \"C++\" {
        std::*;
    };
};
")
    target_link_libraries(MIOpen PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/lib.def")
    target_link_libraries(MIOpen PRIVATE "-Wl,--exclude-libs,ALL")
    # set_target_properties(MIOpen PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(MIOpen PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()
#######################################
if(MIOPEN_ENABLE_SQLITE)
    # MIOpen depends on SQLite
    target_include_directories(MIOpen SYSTEM PRIVATE ${SQLITE3_STATIC_INCLUDE_DIRS})
    target_include_directories(MIOpen SYSTEM INTERFACE $<BUILD_INTERFACE:${SQLITE3_STATIC_INCLUDE_DIRS}>)
    target_compile_options(MIOpen PRIVATE ${SQLITE3_STATIC_CFLAGS})
    target_compile_options(MIOpen INTERFACE $<BUILD_INTERFACE:${SQLITE3_STATIC_CFLAGS}>)
    target_link_libraries(MIOpen PRIVATE ${SQLITE3_STATIC_LDFLAGS})
    target_link_libraries(MIOpen INTERFACE $<BUILD_INTERFACE:${SQLITE3_STATIC_LDFLAGS}>)
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
set(MIOPEN_CXX_HEADER_PATH)
if(MIOPEN_INSTALL_CXX_HEADERS)
set(MIOPEN_CXX_HEADER_PATH ${PROJECT_SOURCE_DIR}/src/include)
endif()

rocm_install_targets(
  TARGETS MIOpen
  INCLUDE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MIOPEN_CXX_HEADER_PATH}
)

rocm_export_targets(
  TARGETS MIOpen
  DEPENDS
    ${PACKAGE_DEPENDS}
  STATIC_DEPENDS
    ${PACKAGE_STATIC_DEPENDS}
)

# Install db files
if(NOT MIOPEN_EMBED_DB STREQUAL "")
    include(embed)
    set(CODE_OBJECTS)
# embed find db
    foreach(EMBED_ARCH ${MIOPEN_EMBED_DB})
        message(STATUS "Adding find db for arch: ${EMBED_ARCH}")
        list(APPEND CODE_OBJECTS "kernels/${EMBED_ARCH}.${MIOPEN_BACKEND}.fdb.txt")
        message(STATUS "Adding perf db for arch: ${EMBED_ARCH}")
        list(APPEND CODE_OBJECTS "kernels/${EMBED_ARCH}.db")
    endforeach()
# Embed Bin Cache
    if(NOT MIOPEN_BINCACHE_PATH STREQUAL "")
        foreach(EMBED_ARCH ${MIOPEN_EMBED_DB})
            message(STATUS "Adding binary cache for arch: ${EMBED_ARCH}")
            download_binary(OUTPUT_PATH "${MIOPEN_BINCACHE_PATH}" "${EMBED_ARCH}")
            list(APPEND CODE_OBJECTS "${OUTPUT_PATH}")
        endforeach()
    endif()
    add_embed_library(miopen_data ${CODE_OBJECTS})
    target_link_libraries(MIOpen PRIVATE $<BUILD_INTERFACE:miopen_data> )
else()
    file(GLOB FIND_DB_FILES kernels/*.fdb.txt)
    file(GLOB PERF_DB_FILES kernels/*.db)
    list(APPEND FIND_DB_FILES ${PERF_DB_FILES})
    if(NOT MIOPEN_DISABLE_SYSDB)
        install(FILES
            ${FIND_DB_FILES}
         DESTINATION ${DATA_INSTALL_DIR}/db)
         foreach(DB_FILE ${FIND_DB_FILES})
            get_filename_component(DB_FILE_FILENAME "${DB_FILE}" NAME)
            configure_file("${DB_FILE}" "${PROJECT_BINARY_DIR}/share/miopen/db/${DB_FILE_FILENAME}" COPYONLY)
         endforeach()

    endif()
endif()

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  #Generating Wrapper of each headers for backward compatibility
  rocm_wrap_header_file(
    export.h config.h version.h miopen.h
    HEADER_LOCATION include/miopen
    GUARDS WRAPPER
    WRAPPER_LOCATIONS miopen/${CMAKE_INSTALL_INCLUDEDIR}/miopen
    ORIGINAL_FILES ${PROJECT_BINARY_DIR}/include/miopen/version.h ${PROJECT_BINARY_DIR}/include/miopen/config.h
  )

  #Installing Wrapper Headers
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/miopen/include"
        DESTINATION "miopen" )
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()
