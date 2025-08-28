/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <cstddef>
#include <unordered_map>
#include <mutex>
#include <sstream>
#include <filesystem>
#include "miopen/direct_ck_mgr.hpp"

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#endif

#include "../composable_kernel/composable_kernel/src/kernel_wrapper/device_grouped_conv_bwd_weight_dl_v4.hpp"
#include <array>  
#include <functional>
#include <unordered_map>
#include <utility>

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using F8   = ck::f8_t;
using BF8  = ck::bf8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

using ALayout = ck::tensor_layout::convolution::GNHWC;
using BLayout = ck::tensor_layout::convolution::GKYXC;
using ELayout = ck::tensor_layout::convolution::GNHWK;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_QUN_CONV_BWD)

// Hash combining utility
template <typename T>
inline void hash_combine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for arrays
template <typename T, std::size_t N>
std::size_t hash_array(const std::array<T, N>& arr) {
    std::size_t seed = 0;
    for (const auto& elem : arr) {
        hash_combine(seed, elem);
    }
    return seed;
}


static std::mutex s_fileMutex;
// Initializing the unordered map with CacheData objects

static std::filesystem::path exp_path;

static void ReadCacheFile()
{
    std::lock_guard<std::mutex> lock(s_fileMutex);
    auto ckMgr = DirectCkMgr::GetInst();
    ckMgr->ReadCacheFile(ckMgr->s_qun_wrw, ckMgr->path_qun_wrw);
}


static void AppendToCache(CacheData cd)
{
    if (DirectCkMgr::GetInst()->enableConvCache == false)   return;

    std::lock_guard<std::mutex> lock(s_fileMutex);
    auto ckMgr = DirectCkMgr::GetInst();
    ckMgr->newKernelCount[ST_QUN_WRW] ++;
    ckMgr->AppendToCache(ckMgr->s_qun_wrw, cd);
}


namespace miopen {
namespace solver {
namespace conv {
using DeviceConvBwdWeightFactory = std::tuple<   
    //                                                   NDimSpatial BlockSize InLayout WeiLayout OutLayout  InDataType WeiDataType  OutDataType     AccDatType BlockTileSize FilterSize  FilterParam(dilation, stride, pad)                                        NBatch NumWavePerTile InScalarPerVector OutScalarPerVector DstScalarPerVector  RequirePadding
      ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                2,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 16,    1,             1,                1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1,     4,             8,                8,                 1,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1,     2,             4,                4,                 1,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 16,    1,             1,                1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1,     4,             8,                4,                 1,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                2,                 2,                 false>

     // 28 x 5 x 1
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
    
     // 14 x 5 x 1
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>

     // 7 x 5 x 1
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             1,                1,                 8,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             1,                1,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             1,                1,                 2,                 false>

     // 56 x 5 x 2
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             4,                2,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             4,                2,                 2,                 false, 2>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false, 2>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false, 2>

     // 14 x 5 x 2
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                1,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                1,                 8,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                1,                 4,                 false>

     // 112 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             4,                4,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 4>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 2>
   
    // 56 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             4,                4,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false, 4>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 2>

    // 28 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>

    // 14 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                2,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>

    // 7 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             1,               1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             1,               1,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             1,               1,                 2,                 false>

    // 112 x 3 x 2
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             4,                2,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 4>

    // 28 x 3 x 3
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             4,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             4,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             4,                2,                 4,                 false>
>;

using ProblemDescription = miopen::conv::ProblemDescription;
namespace
{
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);
        Di = ProblemInterpreter::GetInputDepthDi(problem);
        Do = ProblemInterpreter::GetOutputDepthDo(problem);
        Z  = ProblemInterpreter::GetFilterDepthZ(problem);

        input_lengths   = {G, N, C, Hi, Wi}; // input
        out_lens        = {G, N, K, Ho, Wo}; // output
        wei_lens        = {G, K, C, Y, X};   // filter = wei
        bias_lens       = {G, 1, K, 1, 1};
        bias_strides    = {K, 0, 1, 0, 0};

        const std::string layout = problem.GetInLayout();
        if (layout == "NCHW")
        {
            in_strides  = { Hi*Wi*C,  G*Hi*Wi*C,  1,  Wi*C,  C};
            out_strides = { Ho*Wo*K,  G*Ho*Wo*K,  1,  Wo*K,  K};
            wei_strides = { Y*X*C,    G*Y*X*C,    1,  X*C,   C};
        }
        else
        {
               in_strides  = { Hi*Wi*C,  G*Hi*Wi*C,  1,  Wi*C,  C};
               out_strides = { Ho*Wo*K,  G*Ho*Wo*K,  1,  Wo*K,  K};
               wei_strides = { Y*X*C,    G*Y*X*C,    1,  X*C,   C};
        }

        filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding        = {ProblemInterpreter::GetInputLeftPadH(problem),
                           ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                           ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    size_t GetParamHash() const
    {
        size_t seed = ST_QUN_WRW;
        // Combine hashes of each parameter  
        hash_combine(seed, hash_array(input_lengths));
        hash_combine(seed, hash_array(in_strides));
        hash_combine(seed, hash_array(out_lens));
        hash_combine(seed, hash_array(out_strides));
        hash_combine(seed, hash_array(wei_lens));
        hash_combine(seed, hash_array(wei_strides));
        hash_combine(seed, hash_array(bias_lens));
        hash_combine(seed, hash_array(bias_strides));
        hash_combine(seed, hash_array(filter_stride));
        hash_combine(seed, hash_array(filter_dilation));
        hash_combine(seed, hash_array(lPadding));
        hash_combine(seed, hash_array(rPadding));

        std::array<ck::index_t, 5> others = {C1, K1, Di, Do, Z };
        hash_combine(seed, hash_array(others));
    
        return seed;
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;
    ~CKArgs()                        = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    Data_t      w,
                    ConstData_t out,
                    ck::index_t split_k) const
    {
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             out,
                                             input_lengths,
                                             in_strides,
                                             wei_lens,
                                             wei_strides,
                                             out_lens,
                                             out_strides,
                                             filter_stride,
                                             filter_dilation,
                                             lPadding,
                                             rPadding,
                                             InElementOp{},
                                             WeiElementOp{},
                                             OutElementOp{},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr&         conv_ptr,
                    const ConvWrwTensors&  tensors,
                    ck::index_t            split_k) const
    {
        return MakeArgPtr(conv_ptr, tensors.x, tensors.dw, tensors.dy, split_k);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr,
                       ck::index_t    split_k = 1) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, split_k);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    std::size_t GetFlops() const
    {
        // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
        return static_cast<std::size_t>(2) * G * N * K * C *
            std::accumulate(std::next(std::begin(out_lens), 3),
                            std::end(out_lens),
                            static_cast<std::size_t>(1), std::multiplies<>()) *
            std::accumulate(std::next(std::begin(wei_lens), 3),
                            std::end(wei_lens),
                            static_cast<std::size_t>(1), std::multiplies<>());
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Di;
    int Ho;
    int Wo;
    int Do;
    int Y;
    int X;
    int Z;
    std::array<ck::index_t, 5> input_lengths;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> out_lens;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> wei_lens;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 5> bias_lens;
    std::array<ck::index_t, 5> bias_strides;
    std::array<ck::index_t, 2> filter_stride;
    std::array<ck::index_t, 2> filter_dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};
}

ConvDepthwiseWrw::ConvDepthwiseWrw()
{
}

bool ConvDepthwiseWrw::IsApplicable(const ExecutionContext&   ctx,
                                  const ProblemDescription& problem) const
{
    if (DirectCkMgr::GetInst()->enableOptConv == false)   return false;
    if(!miopen::debug::AlwaysEnableConvDirectNaive)
    {
        if(env::disabled(MIOPEN_DEBUG_CONV_QUN_CONV_BWD))
            return false;
        if(!ctx.use_hip_kernels)
            return false;
    }

    if(!ConvDirectNaiveConvIsApplicableByKernelType(ctx, problem))
        return false;

    if(!problem.IsLayoutDefault() && !problem.IsLayoutNHWC())
        return false;

#if 0
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16() || problem.IsFp8() ||
         problem.IsBfp8()))
        return false;
#else
    // todo support more data type
    if(!problem.IsFp16())
        return false;
#endif
    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
    {
        auto test_cast = [&](const TensorDescriptor& desc) {
            if(desc.GetCastType())
            {
                const auto cast_type = *desc.GetCastType();
                if(cast_type == miopenFloat8_fnuz || cast_type == miopenBFloat8_fnuz)
                    return false;
            }
            // all tested tensors must have cast type set
            return true;
        };
        if(test_cast(problem.GetIn()))
            return false;
        if(test_cast(problem.GetOut()))
            return false;
    }

    if (GetSupportedSolutionCount(ctx, problem) == 0)
    {
        return false;
    }

    return true;
}

uint32_t ConvDepthwiseWrw::GetSupportedSolutionCount(const ExecutionContext& ctx,
                                                   const miopen::conv::ProblemDescription& problem) const
{
    uint32_t solutionCount = 0;
    const auto& ck_args    = CKArgs{problem};

    auto factory_list = DeviceConvBwdWeightFactory{};
    ck::static_for<0, std::tuple_size_v<DeviceConvBwdWeightFactory>, 1>{}([&](auto i) -> void {
        const auto conv_ptr = std::get<i>(factory_list);

        auto argument = conv_ptr.MakeArgument(nullptr, nullptr, nullptr,
                                                ck_args.input_lengths,
                                                ck_args.in_strides,
                                                ck_args.wei_lens,
                                                ck_args.wei_strides,
                                                ck_args.out_lens,
                                                ck_args.out_strides,
                                                ck_args.filter_stride,
                                                ck_args.filter_dilation,
                                                ck_args.lPadding,
                                                ck_args.rPadding,
                                                InElementOp{},
                                                WeiElementOp{},
                                                OutElementOp{},
                                                1);
        if(conv_ptr.IsSupportedArgument(argument))
        {
            solutionCount ++;
        }
    });

    return solutionCount;
}

bool ConvDepthwiseWrw::FindCachedSolution(const ExecutionContext& ctx, size_t hashcode, const miopen::conv::ProblemDescription& problem, ConvSolution& sol) const
{
    if (DirectCkMgr::GetInst()->enableConvCache == false) return false;

    bool found = false;
    size_t best_kernel;
    int best_split_k;
    {
        auto& s_cacheTable = DirectCkMgr::GetInst()->s_qun_wrw;
        std::lock_guard<std::mutex> lock(s_fileMutex);
        auto it = s_cacheTable.find(hashcode);
        found = it != s_cacheTable.end();
        if (found)
        {
            const CacheData cd = it->second;
            best_kernel  = cd.kernelhash;
            best_split_k = cd.split_k;

            MIOPEN_LOG_I("Find cached solution " << std::hex << std::setw(16) << std::setfill('0') << cd.hashcode 
            << ", kernal hash:" << std::hex << std::setw(16) << std::setfill('0') << best_kernel 
            <<", split_k:"<< best_split_k);
        }
    }
    bool foundBest = false;
    if (found)
    {
        auto factory_list = DeviceConvBwdWeightFactory{};
        ck::static_for<0, std::tuple_size_v<DeviceConvBwdWeightFactory>, 1>{}([&](auto i) -> void {

            const auto device_conv_bwd_weight_instance = std::get<i>(factory_list);
            using DeviceConvBwdWeightInstance = ck::remove_cvref_t<decltype(device_conv_bwd_weight_instance)>;
            auto conv_ptr = std::make_shared<DeviceConvBwdWeightInstance>();

            size_t curKernelCache = DirectCkMgr::GetInst()->GetStringHash(conv_ptr->GetTypeString());
            if (curKernelCache == best_kernel)
            {
                MIOPEN_LOG_I("Find best cached kernel " << conv_ptr->GetTypeString() << " , best split_k" <<best_split_k);
                foundBest = true;

                const auto is_nhwc = (problem.IsLayoutDefault() == false);
                const int ho     = problem.GetInHeight();
                const int wo     = problem.GetInWidth();
                const int n      = problem.GetInBatchSize();
                const int k      = problem.GetInChannels();
                const int c      = problem.GetOutChannels();
                const int hi     = problem.GetOutHeight();
                const int wi     = problem.GetOutWidth();

                size_t trans_input_size   = 0;
                size_t trans_output_size   = 0;

                bool trans_input_skippable  = true;
                bool trans_output_skippable = true;

                int trans_input_idx  = -1;
                int trans_output_idx = -1;

                std::vector<std::vector<OpKernelArg>> opArgsTrans;

                if (is_nhwc)
                {   
                    TransposeSolutionNhwc2Default trans_input(ctx, problem.GetInDataType(), n, c, hi, wi);
                    TransposeSolutionNhwc2Default trans_output(ctx, problem.GetOutDataType(), n, k, ho, wo);

                    trans_input_skippable  = trans_input.IsSkippable();
                    trans_output_skippable = trans_output.IsSkippable();

                    opArgsTrans.emplace_back(trans_input.GetKernelArg());
                    opArgsTrans.emplace_back(trans_output.GetKernelArg());

                
                    trans_input_size  = trans_input_skippable ? 0 : trans_input.GetOutputTensorSize();
                    trans_output_size = trans_output_skippable ? 0 : trans_output.GetOutputTensorSize();
                        
                    std::ostringstream msg;
                    sol.construction_params.push_back(trans_input.GetKernelInfo());
                    if(miopen::IsLogging(LoggingLevel::Info2))
                        msg << ", inp trans:" << trans_input.GetKernelName();

                    sol.construction_params.push_back(trans_output.GetKernelInfo());
                    if(miopen::IsLogging(LoggingLevel::Info2))
                        msg << ", out trans:" << trans_output.GetKernelName();

                    trans_input_idx=0;
                    trans_output_idx=1;
                }
                sol.workspace_sz = GetWorkspaceSize(ctx, problem);
                sol.invoker_factory = [=](const std::vector<Kernel>& kernels) mutable{
                    return [=](const Handle& handle, const AnyInvokeParams& primitive_params) mutable {
                        const auto& data_ctx = primitive_params.CastTo<miopen::conv::WrWInvokeParams>();
                        const auto& ck_args  = CKArgs{problem};
                        const auto& workSpace   = data_ctx.workSpace;

                        float elapsed = 0;
                        auto trans_input_buf =
                            trans_input_size == 0
                            ? shared<Data_t>{}
                            : handle.CreateSubBuffer(workSpace, 0, trans_input_size);
                        auto trans_output_buf =
                            trans_output_size == 0
                            ? shared<Data_t>{}
                            : handle.CreateSubBuffer(workSpace, trans_input_size, trans_output_size);

                        if(!trans_input_skippable)
                        {
                            auto& karg_input = opArgsTrans[trans_input_idx];
                            karg_input[0]    = OpKernelArg(trans_input_buf.get()); //dst
                            karg_input[1]    = OpKernelArg(data_ctx.tensors.x);
                            handle.Run(kernels[trans_input_idx])(karg_input);
                            if(handle.IsProfilingEnabled())
                                elapsed += handle.GetKernelTime();
                        }

                        if(!trans_output_skippable)
                        {
                            auto& karg_output = opArgsTrans[trans_output_idx];
                            karg_output[0]    = OpKernelArg(trans_output_buf.get());  //dst
                            karg_output[1]    = OpKernelArg(data_ctx.tensors.dy);   // src
                            handle.Run(kernels[trans_output_idx])(karg_output);
                            if(handle.IsProfilingEnabled())
                                elapsed += handle.GetKernelTime();
                        }
                  

                        auto invoker  = conv_ptr->MakeInvoker();
                        auto argument = conv_ptr->MakeArgument(static_cast<const InDataType*>(
                                                                trans_input_skippable==true ? data_ctx.tensors.x:
                                                                trans_input_buf.get()),
                                                            static_cast<WeiDataType*>(data_ctx.tensors.dw),
                                                            static_cast<const OutDataType*>(
                                                                trans_output_skippable==true ? data_ctx.tensors.dy:
                                                                trans_output_buf.get()),
                                                            ck_args.input_lengths,
                                                            ck_args.in_strides,
                                                            ck_args.wei_lens,
                                                            ck_args.wei_strides,
                                                            ck_args.out_lens,
                                                            ck_args.out_strides,
                                                            ck_args.filter_stride,
                                                            ck_args.filter_dilation,
                                                            ck_args.lPadding,
                                                            ck_args.rPadding,
                                                            InElementOp{},
                                                            WeiElementOp{},
                                                            OutElementOp{},
                                                            best_split_k);

                        auto gemm_buf = handle.CreateSubBuffer(workSpace, trans_input_size+trans_output_size, conv_ptr->GetWorkSpaceSize(&argument));                                         
                        conv_ptr->SetWorkSpacePointer(&argument, gemm_buf.get());
                        {
                            invoker.ShowInfo(argument);
                            {
                                WorkAroundHipEventProfiler prf(handle);
                                invoker.Run(argument, StreamConfig{handle.GetStream(), false});
                            }

                            if (DirectCkMgr::GetInst()->enableLog)
                                std::cout << "Cached qun wrw is called" << std::endl;
                            if(handle.IsProfilingEnabled())
                            {
                                elapsed += handle.GetKernelTime();
                                handle.ResetKernelTime();
                                handle.AccumKernelTime(elapsed);

                                DirectCkMgr::GetInst()->launchCount[ST_QUN_WRW] ++;
                                DirectCkMgr::GetInst()->hitCacheCount[ST_QUN_WRW] ++;
                            }
                        }
                    };
                };
            }

            if (foundBest) true;

        });
    }

    return foundBest;
}

ConvSolution ConvDepthwiseWrw::GetBestSolution(const ExecutionContext& ctx,
                                             const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution sol;
    const auto& ck_args   = CKArgs{problem};
    const size_t argsHash = ck_args.GetParamHash();
    CacheData cd;
    cd.hashcode = argsHash;
    if (FindCachedSolution(ctx, argsHash, problem, sol))
    {
        return sol;
    }

    Tensor<InDataType> in_g_n_c_wis(std::initializer_list<ck::index_t>{ck_args.G, ck_args.N, ck_args.C, ck_args.Hi, ck_args.Wi});
    Tensor<WeiDataType> wei_g_k_c_xs(std::initializer_list<ck::index_t>{ck_args.G, ck_args.K, ck_args.C, ck_args.Y, ck_args.X});
    Tensor<OutDataType> out_g_n_k_wos(std::initializer_list<ck::index_t>{ck_args.G, ck_args.N, ck_args.K, ck_args.Ho, ck_args.Wo});

    in_g_n_c_wis.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 0.2});
    out_g_n_k_wos.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.1, 0.1});

    DeviceMem in_device_buf(sizeof(InDataType)   * in_g_n_c_wis.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_g_k_c_xs.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_g_n_k_wos.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in_g_n_c_wis.mData.data());
    out_device_buf.ToDevice(out_g_n_k_wos.mData.data());

    // init to 0
    wei_device_buf.SetZero();

    // Find the best
    ck::index_t split_k     = -1;
    float best_tflops       = 0;
    //float best_gb_per_sec   = 0;
    float best_avg_time     = 3.4e+30;
    std::string best_kernel = "";
    ck::index_t split_k_array[] = {1, 2, 4, 8, 16, 32};
    ck::index_t split_k_count = 6;
    ck::index_t instance_idx = 0;
    ck::index_t best_split_k = 0;

    if (split_k != -1)
    {
        split_k_count = 1;
        split_k_array[0] = split_k;
    }

    bool found_kernel= false;
    auto factory_list = DeviceConvBwdWeightFactory{};
    ck::static_for<0, std::tuple_size_v<DeviceConvBwdWeightFactory>, 1>{}([&](auto i) -> void {
        const auto device_conv_bwd_weight_instance = std::get<i>(factory_list);
        using DeviceConvBwdWeightInstance = ck::remove_cvref_t<decltype(device_conv_bwd_weight_instance)>;
        auto conv_ptr = std::make_shared<DeviceConvBwdWeightInstance>();

        const std::string kernelName = conv_ptr->GetTypeString();
        for (ck::index_t j = 0; j < split_k_count; j++)
        {
            ck::index_t cur_split_k = split_k_array[j];
            auto invoker  = conv_ptr->MakeInvoker();
            auto argument = conv_ptr->MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                            static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                            ck_args.input_lengths,
                                            ck_args.in_strides,
                                            ck_args.wei_lens,
                                            ck_args.wei_strides,
                                            ck_args.out_lens,
                                            ck_args.out_strides,
                                            ck_args.filter_stride,
                                            ck_args.filter_dilation,
                                            ck_args.lPadding,
                                            ck_args.rPadding,
                                            InElementOp{},
                                            WeiElementOp{},
                                            OutElementOp{},
                                            cur_split_k);

            if(conv_ptr->IsSupportedArgument(argument))
            {
                DeviceMem gemm_workspace_dev(conv_ptr->GetWorkSpaceSize(&argument));
                conv_ptr->SetWorkSpacePointer(&argument, gemm_workspace_dev.GetDeviceBuffer());
               
                found_kernel = true;
                MIOPEN_LOG_I("Run conv : (split_K:" << cur_split_k << ") " << kernelName);
                invoker.ShowInfo(argument);
                float avg_time = invoker.Run(argument, StreamConfig{nullptr, true});
                {
                    std::size_t flop = ck_args.GetFlops();
                    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                    MIOPEN_LOG_I("avg_time:" << avg_time <<" , tflops:" << tflops);
                    if (avg_time < best_avg_time)
                    {
                        best_tflops     = tflops;
                        best_avg_time   = avg_time;
                        best_split_k    = cur_split_k;
                        best_kernel     = kernelName;
                        MIOPEN_LOG_I("* ^best kernel so far^* ");
                        instance_idx = i;
                        cd.split_k   = static_cast<int>(cur_split_k);

                        sol.invoker_factory = [
                        conv_ptr, problem, best_split_k
                        ](const std::vector<Kernel>& kernels) {
                            return [conv_ptr, problem, best_split_k](const Handle& handle, const AnyInvokeParams& primitive_params) {
                                const auto& data_ctx = primitive_params.CastTo<miopen::conv::WrWInvokeParams>();
                                const auto& workSpace   = data_ctx.workSpace;
                                
                                const auto& ck_args  = CKArgs{problem};
                                auto invoker  = conv_ptr->MakeInvoker();
                                auto argument = conv_ptr->MakeArgument(static_cast<const InDataType*>(data_ctx.tensors.x),
                                                                    static_cast<WeiDataType*>(data_ctx.tensors.dw),
                                                                    static_cast<const OutDataType*>(data_ctx.tensors.dy),
                                                                    ck_args.input_lengths,
                                                                    ck_args.in_strides,
                                                                    ck_args.wei_lens,
                                                                    ck_args.wei_strides,
                                                                    ck_args.out_lens,
                                                                    ck_args.out_strides,
                                                                    ck_args.filter_stride,
                                                                    ck_args.filter_dilation,
                                                                    ck_args.lPadding,
                                                                    ck_args.rPadding,
                                                                    InElementOp{},
                                                                    WeiElementOp{},
                                                                    OutElementOp{},
                                                                    best_split_k);

                                if(conv_ptr->IsSupportedArgument(argument))
                                {
                                    auto gemm_buf = handle.CreateSubBuffer(workSpace, 0, conv_ptr->GetWorkSpaceSize(&argument));                                         
                                    conv_ptr->SetWorkSpacePointer(&argument, gemm_buf.get());

                                    invoker.ShowInfo(argument);
                                    WorkAroundHipEventProfiler prf(handle);
                                    float avg_time = invoker.Run(argument, StreamConfig{handle.GetStream(), false});

                                    if(handle.IsProfilingEnabled())
                                    {
                                        avg_time = handle.GetKernelTime();
                                        handle.ResetKernelTime();
                                        handle.AccumKernelTime(avg_time);
                                        DirectCkMgr::GetInst()->launchCount[ST_QUN_WRW] ++;
                                        if (DirectCkMgr::GetInst()->enableLog)
                                        {
                                            std::cout << "Un-cached qun wrw is called" << std::endl;
                                        }

                                    }
                                }
                            };
                        };
                    }
                }
            }
            else
            {
                break;
            }
        }
    });

    if (found_kernel)
    {
        cd.kernelhash = DirectCkMgr::GetInst()->GetStringHash(best_kernel);
        MIOPEN_LOG_I("*** ^ best kernel ^*** " << std::hex << cd.kernelhash);
        AppendToCache(cd);
    }

    return sol;
}

ConvSolution ConvDepthwiseWrw::GetSolution(const ExecutionContext& ctx,
                                         const ProblemDescription& problem) const
{
    ReadCacheFile();
    return GetBestSolution(ctx, problem);
}

size_t ConvDepthwiseWrw::GetWorkspaceSize(const ExecutionContext& ctx,
                                            const ProblemDescription& problem) const
{
    const auto is_nhwc = (problem.IsLayoutDefault() == false);
    const int hi       = problem.GetOutHeight();
    const int wi       = problem.GetOutWidth();
    const int n        = problem.GetBatchSize();
    const int k        = problem.GetInChannels();
    const int c        = problem.GetOutChannels();
    const int ho       = problem.GetInHeight();
    const int wo       = problem.GetInWidth();
    const int y        = problem.GetWeightsHeight();
    const int x        = problem.GetWeightsWidth();
    const auto group   = problem.GetGroupCount();

    size_t workspace_size = 0;
    if (is_nhwc)
    {
        size_t size_trans_input  = 0;
        size_t size_trans_output = 0;

        TransposeSolutionNhwc2Default trans_input(ctx, problem.GetInDataType(), n, c, hi, wi);
        TransposeSolutionNhwc2Default trans_output(ctx, problem.GetOutDataType(), n, k, ho, wo);

        bool trans_input_skippable  = trans_input.IsSkippable();
        bool trans_output_skippable = trans_output.IsSkippable();

        size_trans_input  = trans_input_skippable ? 0 : trans_input.GetOutputTensorSize();
        size_trans_output = trans_output_skippable ? 0 : trans_output.GetOutputTensorSize();
        workspace_size += size_trans_input;
        workspace_size += size_trans_output;
    }
    
    size_t size_gemm = (4* group  * x * y + 127) / 128 * 128;

    workspace_size += size_gemm;
    return workspace_size;
}

}
} // namespace solver
} // namespace miopen
