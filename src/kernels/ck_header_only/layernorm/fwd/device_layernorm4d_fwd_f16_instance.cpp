// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "normalization_fwd_instance_common.hpp"

namespace miopen {
namespace kernels {
namespace ck_header_only {
namespace layernorm {

using F16  = ck::half_t;
using F32  = float;
using Pass = ck::tensor_operation::element_wise::PassThrough;

void add_device_normalization_fwd_rank_4_3_f16_instances(
    std::vector<std::unique_ptr<
        ck::tensor_operation::device::DeviceNormalizationFwd<F16, F16, F16, F16, F16, Pass, 4, 3>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_normalization_f16_generic_instance<Pass, 4, 3>{});
    add_device_operation_instances(instances, device_normalization_f16_instances<Pass, 4, 3>{});
    add_device_operation_instances(instances,
                                   device_normalization_splitk_f16_instances<Pass, 4, 3>{});
}

} // namespace layernorm
} // namespace ck_header_only
} // namespace kernels
} // namespace miopen
