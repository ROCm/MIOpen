// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

using F16         = ck::half_t;
using F32         = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using index_t     = ck::index_t;

namespace miopen::kernels::ck_header_only::layernorm {

// FP16
void add_device_normalization_fwd_rank_2_1_f16_instances(
    std::vector<
        std::unique_ptr<ck::tensor_operation::device::
                            DeviceNormalizationFwd<F16, F16, F16, F16, F16, PassThrough, 2, 1>>>&);

void add_device_normalization_fwd_rank_4_3_f16_instances(
    std::vector<
        std::unique_ptr<ck::tensor_operation::device::
                            DeviceNormalizationFwd<F16, F16, F16, F16, F16, PassThrough, 4, 3>>>&);

// FP32
void add_device_normalization_fwd_rank_2_1_f32_instances(
    std::vector<
        std::unique_ptr<ck::tensor_operation::device::
                            DeviceNormalizationFwd<F32, F32, F32, F32, F32, PassThrough, 2, 1>>>&);

void add_device_normalization_fwd_rank_4_3_f32_instances(
    std::vector<
        std::unique_ptr<ck::tensor_operation::device::
                            DeviceNormalizationFwd<F32, F32, F32, F32, F32, PassThrough, 4, 3>>>&);

template <typename DeviceOp, typename Tag = void>
struct DeviceOperationInstanceFactory;

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceNormalizationFwd<
    XDataType,
    GammaDataType,
    BetaDataType,
    YDataType,
    SaveMeanInvStdDataType,
    ck::tensor_operation::element_wise::PassThrough,
    Rank,
    NumReduceDim>>
{
    using DeviceOp = ck::tensor_operation::device::DeviceNormalizationFwd<
        XDataType,
        GammaDataType,
        BetaDataType,
        YDataType,
        SaveMeanInvStdDataType,
        ck::tensor_operation::element_wise::PassThrough,
        Rank,
        NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(ck::is_same_v<XDataType, F16> && ck::is_same_v<GammaDataType, F16> &&
                     ck::is_same_v<BetaDataType, F16> && ck::is_same_v<YDataType, F16> &&
                     ck::is_same_v<SaveMeanInvStdDataType, F16>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
            {
                miopen::kernels::ck_header_only::layernorm::
                    add_device_normalization_fwd_rank_2_1_f16_instances(op_ptrs);
            }
            else if constexpr(Rank == 4 && NumReduceDim == 3)
            {
                miopen::kernels::ck_header_only::layernorm::
                    add_device_normalization_fwd_rank_4_3_f16_instances(op_ptrs);
            }
        }

        if constexpr(ck::is_same_v<XDataType, F32> && ck::is_same_v<GammaDataType, F32> &&
                     ck::is_same_v<BetaDataType, F32> && ck::is_same_v<YDataType, F32> &&
                     ck::is_same_v<SaveMeanInvStdDataType, F32>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
            {
                miopen::kernels::ck_header_only::layernorm::
                    add_device_normalization_fwd_rank_2_1_f32_instances(op_ptrs);
            }
            else if constexpr(Rank == 4 && NumReduceDim == 3)
            {
                miopen::kernels::ck_header_only::layernorm::
                    add_device_normalization_fwd_rank_4_3_f32_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace miopen::kernels::ck_header_only::layernorm
