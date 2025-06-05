// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_splitk_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace miopen {
namespace kernels {
namespace ck_header_only {
namespace layernorm {

using F16 = ck::half_t;
using F32 = float;
using index_t = ck::index_t;

template <typename BaseOp, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<BaseOp>>& op_instances,
                                    const NewOpInstances& new_op_instances)
{
    ck::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        const auto new_op_instance = std::get<i>(new_op_instances);

        using NewOpInstance = ck::remove_cvref_t<decltype(new_op_instance)>;

        static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                      "wrong! NewOpInstance should be derived from BaseOp");

        op_instances.push_back(std::make_unique<NewOpInstance>(new_op_instance));
    });
}

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f16_instances =
    // clang-format off
    std::tuple <
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize, SaveMeanInvStdScalarPerVector>
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 8, 1, 8, 1, 8, 8, 2>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>
        // clang-format on
        >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_splitk_f16_instances =
    // clang-format off
    std::tuple <
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize, SaveMeanInvStdScalarPerVector>
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 8, 1, 8, 1, 8, 8, 2>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>
        // clang-format on
        >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f16_generic_instance = std::tuple<
    // clang-format off
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F16, F16, F16, F32, F16, F16, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // clang-format on
    >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f32_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize, SaveMeanInvStdScalarPerVector>
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 4, 1, 4, 1, 4, 4, 2>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 2, 8, 1, 4, 1, 4, 1, 4, 4, 2>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>
    // clang-format on
    >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_splitk_f32_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize, SaveMeanInvStdScalarPerVector>
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1>,   // irregular size
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 4, 1, 4, 1, 4, 4, 2>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 2, 8, 1, 4, 1, 4, 1, 4, 4, 2>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>
    // clang-format on
    >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f32_generic_instance = std::tuple<
    // clang-format off
        ck::tensor_operation::device::DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // clang-format on
    >;

} // namespace layernorm
} // ck_header_only
} // kernels
} // namespace miopen
