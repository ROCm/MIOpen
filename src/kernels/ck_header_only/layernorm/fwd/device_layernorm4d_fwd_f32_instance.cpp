// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "normalization_fwd_instance_common.hpp"

namespace miopen::kernels::ck_header_only::layernorm {

using F16  = ck::half_t;
using F32  = float;
using Pass = ck::tensor_operation::element_wise::PassThrough;

void add_device_normalization_fwd_rank_4_3_f32_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceNormalizationFwd<F32, F32, F32, F32, F32, Pass, 4, 3>>>&
        instances)
{
    using ck::tensor_operation::device::instance::add_device_operation_instances;

    add_device_operation_instances(instances,
                                   device_normalization_f32_generic_instance<Pass, 4, 3>{});
    add_device_operation_instances(instances, device_normalization_f32_instances<Pass, 4, 3>{});
    add_device_operation_instances(instances,
                                   device_normalization_splitk_f32_instances<Pass, 4, 3>{});
}

} // namespace miopen::kernels::ck_header_only::layernorm
