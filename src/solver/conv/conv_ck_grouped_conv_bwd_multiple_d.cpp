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

#define DISABLE_INPUT_LDS 1
#include "../composable_kernel/composable_kernel/src/kernel_wrapper/device_grouped_conv_bwd_data_multiple_d.hpp"
#include <array>  
#include <functional>
#include <unordered_map>
#include <utility>

using BF16 = ck::bhalf_t;
using FP16  = ck::half_t;
using FP32  = float;
using FP8   = ck::f8_t;
using BF8  = ck::bf8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// kernel data types
using InKernelDataType  = FP16;
using WeiKernelDataType = FP16;
using AccDataType       = FP32;
using CShuffleDataType  = FP16;
using OutKernelDataType = FP16;

// tensor data types
using InDataType  = InKernelDataType;
using WeiDataType = WeiKernelDataType;
using OutDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

// using ALayout = ck::tensor_layout::convolution::GNHWC;
// using BLayout = ck::tensor_layout::convolution::GKYXC;
// using ELayout = ck::tensor_layout::convolution::GNHWK;

using InType  = InKernelDataType;
using WeiType = WeiKernelDataType;
using AccType = AccDataType;
using OutType = OutKernelDataType;

constexpr ck::index_t NDimSpatial = 2;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_JIN_MD_BWD)

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
static std::filesystem::path exp_path;

static void ReadCacheFile()
{
    std::lock_guard<std::mutex> lock(s_fileMutex);
    auto ckMgr = DirectCkMgr::GetInst();
    ckMgr->ReadCacheFile(ckMgr->s_jin_bwd, ckMgr->path_jin_bwd);
}


static void AppendToCache(CacheData cd)
{
    if (DirectCkMgr::GetInst()->enableConvCache == false)   return;

    std::lock_guard<std::mutex> lock(s_fileMutex);
    auto ckMgr = DirectCkMgr::GetInst();
    ckMgr->newKernelCount[ST_JIN_BWD] ++;
    ckMgr->AppendToCache(ckMgr->s_jin_bwd, cd);
}

namespace miopen {
namespace solver {
namespace conv {
using DeviceConvBwdFactory = std::tuple<

//                                                NDimSpatial BlockSize In      Wei        Acc    Out      BlockTileSize FilterSize   FilterParam (dilation, stride, padding)                                     NBatch  SubTileH W  ScalarPerVector(in out)    RequirePadding>
      ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<7, 7>,     5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<14, 14>,   5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       2, 2,                     false>
 //   , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<14, 14>,   5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       4, 4,                     false>
  //  , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,     InType, WeiType, AccType, OutType,  S<14, 14>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       2, 1,                     false>
   //  , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,     InType, WeiType, AccType, OutType,  S<14, 14>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       4, 2,                     false>
  //  , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<56, 56>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  1,       8, 8,       8, 4,                     false>
 //   , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<56, 56>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  1,       8, 8,       1, 1,                     false>

    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<7, 7>,     3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<14, 14>,   3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       2, 2,                     false>
 //   , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<14, 14>,   3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<56, 56>,   3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  8,       7, 8,       8, 8,                     false>
 //   , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<56, 56>,   3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  8,       7, 8,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<112, 112>, 3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  2,       14, 16,     8, 8,                     false>
  //  , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<112, 112>, 3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  2,       14, 16,     1, 1,                     false>

    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   3,           ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       4, 2,                     false>
 //   , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   3,           ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<112, 112>, 3,           ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  8,       16, 16,       8, 8,                     false>
  //  , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<112, 112>, 3,           ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  8,       16, 16,       1, 1,                     false>
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
            #if 0
            // --in_layout NHWC --out_layout NHWC --fil_layout NHWC
          //  in_strides  = { G*Hi*Wi*C,  G*Wi*C,  G*C,  C,  1};  // NHWGC
           // out_strides = { Ho*Wo*G*K,  G*Wo*K,  G*K,  K,  1}; // NHWGK
          //  wei_strides = { K*Y*X*C,    Y*X*C,    X*C,  C,   1};  // GKYXC

            in_strides  = { C,  G*Hi*Wi*C,  1,  G*Wi*C,  G*C};
            out_strides = { K, Ho*Wo*G*K, 1, G*Wo*K,  G*K}; // NHWGK
            wei_strides = { K*Y*X*C,    Y*X*C,   1,  X*C,  C}; 
            #else
               in_strides  = { Hi*Wi*C,  G*Hi*Wi*C,  1,  Wi*C,  C};
               out_strides = { Ho*Wo*K,  G*Ho*Wo*K,  1,  Wo*K,  K};
               wei_strides = { Y*X*C,    G*Y*X*C,    1,  X*C,   C};
            #endif
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
        size_t seed = ST_JIN_BWD;
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
                                             std::array<const void*, 0>{},
                                             out,
                                             input_lengths,
                                             in_strides,
                                             wei_lens,
                                             wei_strides,
                                             std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                             std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                             out_lens,
                                             out_strides,
                                             filter_stride,
                                             filter_dilation,
                                             lPadding,
                                             rPadding,
                                             InElementOp{},
                                             WeiElementOp{},
                                             OutElementOp{});
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

ConvDepthwiseBwd::ConvDepthwiseBwd()
{
}

bool ConvDepthwiseBwd::IsApplicable(const ExecutionContext&   ctx,
                                  const ProblemDescription& problem) const
{
    if (DirectCkMgr::GetInst()->enableOptConv == false)   return false;
    if(!miopen::debug::AlwaysEnableConvDirectNaive)
    {
        if(env::disabled(MIOPEN_DEBUG_CONV_JIN_MD_BWD))
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

    if(!problem.IsDirectionBackwardData())
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

uint32_t ConvDepthwiseBwd::GetSupportedSolutionCount(const ExecutionContext& ctx,
                                                   const miopen::conv::ProblemDescription& problem) const
{
    uint32_t solutionCount = 0;
    const auto& ck_args    = CKArgs{problem};

    ck::static_for<0, std::tuple_size_v<DeviceConvBwdFactory>, 1>{}([&](auto i) -> void {
        const auto conv_ptr = std::get<i>(DeviceConvBwdFactory{});

        auto argument = conv_ptr.MakeArgument(nullptr, nullptr,
                                                std::array<const void*, 0>{},
                                                nullptr,
                                                ck_args.input_lengths,
                                                ck_args.in_strides,
                                                ck_args.wei_lens,
                                                ck_args.wei_strides,
                                                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                                ck_args.out_lens,
                                                ck_args.out_strides,
                                                ck_args.filter_stride,
                                                ck_args.filter_dilation,
                                                ck_args.lPadding,
                                                ck_args.rPadding,
                                                InElementOp{},
                                                WeiElementOp{},
                                                OutElementOp{});
        if(conv_ptr.IsSupportedArgument(argument))
        {
            solutionCount ++;
        }
    });

    return solutionCount;
}

bool ConvDepthwiseBwd::FindCachedSolution(const ExecutionContext& ctx, size_t hashcode, const miopen::conv::ProblemDescription& problem, ConvSolution& sol) const
{
    if (DirectCkMgr::GetInst()->enableConvCache == false) return false;

    bool found = false;
    size_t best_kernel;
    {
        auto& s_cacheTable = DirectCkMgr::GetInst()->s_jin_bwd;
        std::lock_guard<std::mutex> lock(s_fileMutex);
        auto it = s_cacheTable.find(hashcode);
        found = it != s_cacheTable.end();
        if (found)
        {
            const CacheData cd = it->second;
            best_kernel  = cd.kernelhash;

            MIOPEN_LOG_I("Find cached solution " << std::hex << std::setw(16) << std::setfill('0') << cd.hashcode 
            << ", kernal hash:" << std::hex << std::setw(16) << std::setfill('0') << best_kernel);
        }
    }
    bool foundBest = false;
    if (found)
    {
        ck::static_for<0, std::tuple_size_v<DeviceConvBwdFactory>, 1>{}([&](auto i) -> void {

            const auto device_conv_bwd_weight_instance = std::get<i>(DeviceConvBwdFactory{});
            using DeviceConvBwdWeightInstance = ck::remove_cvref_t<decltype(device_conv_bwd_weight_instance)>;
            auto conv_ptr = std::make_shared<DeviceConvBwdWeightInstance>();

            size_t curKernelCache = DirectCkMgr::GetInst()->GetStringHash(conv_ptr->GetTypeString());
            if (curKernelCache == best_kernel)
            {
                MIOPEN_LOG_I("Find best cached kernel " << conv_ptr->GetTypeString());
                foundBest = true;
#if 1
#if 1

                const auto is_nhwc = (problem.IsLayoutDefault() == false);
                const int ho     = problem.GetInHeight();
                const int wo     = problem.GetInWidth();
                const int n       = problem.GetInBatchSize();
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
                    TransposeSolutionDefault2Nhwc trans_input(ctx, problem.GetInDataType(), n, c, hi, wi);
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

            #endif
                sol.workspace_sz = GetWorkspaceSize(ctx, problem);
                sol.invoker_factory = [=](const std::vector<Kernel>& kernels) mutable {
                    return [=](const Handle& handle, const AnyInvokeParams& primitive_params) mutable {
                        const auto& bwd_ctx     = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                        const auto& ck_args     = CKArgs{problem};
                        const auto& workSpace   = bwd_ctx.workSpace;

                        float elapsed = 0;
                        auto trans_input_buf =
                            trans_input_size == 0
                            ? shared<Data_t>{}
                            : handle.CreateSubBuffer(workSpace, 0, trans_input_size);
                        auto trans_output_buf =
                            trans_output_size == 0
                            ? shared<Data_t>{}
                            : handle.CreateSubBuffer(workSpace, trans_input_size, trans_output_size);

                        if(!trans_output_skippable)
                        {
                            auto& karg_output = opArgsTrans[trans_output_idx];
                            karg_output[0]    = OpKernelArg(trans_output_buf.get());  //dst
                            karg_output[1]    = OpKernelArg(bwd_ctx.tensors.in);   // src
                            handle.Run(kernels[trans_output_idx])(karg_output);
                            if(handle.IsProfilingEnabled())
                                elapsed += handle.GetKernelTime();
                        }

                        auto invoker  = conv_ptr->MakeInvoker();
                        auto argument = conv_ptr->MakeArgument(
                                                            (trans_input_skippable==true)? bwd_ctx.tensors.out:trans_input_buf.get(),
                                                            bwd_ctx.tensors.w,
                                                            std::array<const void*, 0>{},
                                                            (trans_output_skippable==true)? bwd_ctx.tensors.in:trans_output_buf.get(),
                                                            ck_args.input_lengths,
                                                            ck_args.in_strides,
                                                            ck_args.wei_lens,
                                                            ck_args.wei_strides,
                                                            std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                                            std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                                            ck_args.out_lens,
                                                            ck_args.out_strides,
                                                            ck_args.filter_stride,
                                                            ck_args.filter_dilation,
                                                            ck_args.lPadding,
                                                            ck_args.rPadding,
                                                            InElementOp{},
                                                            WeiElementOp{},
                                                            OutElementOp{});
                            {
                                WorkAroundHipEventProfiler prf(handle);
                                invoker.Run(argument, StreamConfig{handle.GetStream(), false});
                            }
                            if (DirectCkMgr::GetInst()->enableLog)
                                    std::cout << "Cached jin bwd is called" << std::endl;

                            if(handle.IsProfilingEnabled())
                            {
                                elapsed += handle.GetKernelTime();
                                DirectCkMgr::GetInst()->launchCount[ST_JIN_BWD] ++;
                                DirectCkMgr::GetInst()->hitCacheCount[ST_JIN_BWD] ++;
                            }

                        if(!trans_input_skippable)
                        {
                            auto& karg_input = opArgsTrans[trans_input_idx];
                            karg_input[0]    = OpKernelArg(bwd_ctx.tensors.out); //dst
                            karg_input[1]    = OpKernelArg(trans_input_buf.get());
                            handle.Run(kernels[trans_input_idx])(karg_input);
                            if(handle.IsProfilingEnabled())
                                elapsed += handle.GetKernelTime();
                        }
                        if(handle.IsProfilingEnabled())
                        {
                            handle.ResetKernelTime();
                            handle.AccumKernelTime(elapsed);
                        }
                    };
                };
                #endif
            }

            if (foundBest) true;

        });
    }

    return foundBest;
}

ConvSolution ConvDepthwiseBwd::GetBestSolution(const ExecutionContext& ctx,
                                             const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution sol;
    const auto& ck_args   = CKArgs{problem};
    const size_t argsHash = ck_args.GetParamHash();
    CacheData cd;
    cd.hashcode = argsHash;
    cd.split_k  = 1;
    if (FindCachedSolution(ctx, argsHash, problem, sol))
    {
        return sol;
    }

    Tensor<InDataType> in_g_n_c_wis(std::initializer_list<ck::index_t>{ck_args.G, ck_args.N, ck_args.C, ck_args.Hi, ck_args.Wi});
    Tensor<WeiDataType> wei_g_k_c_xs(std::initializer_list<ck::index_t>{ck_args.G, ck_args.K, ck_args.C, ck_args.Y, ck_args.X});
    Tensor<OutDataType> out_g_n_k_wos(std::initializer_list<ck::index_t>{ck_args.G, ck_args.N, ck_args.K, ck_args.Ho, ck_args.Wo});

    out_g_n_k_wos.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
    wei_g_k_c_xs.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});

    DeviceMem in_device_buf(sizeof(InDataType)   * in_g_n_c_wis.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_g_k_c_xs.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_g_n_k_wos.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out_g_n_k_wos.mData.data());
    wei_device_buf.ToDevice(wei_g_k_c_xs.mData.data());

    // Find the best
    float best_tflops       = 0;
    //float best_gb_per_sec   = 0;
    float best_avg_time     = 3.4e+30;
    std::string best_kernel = "";
    ck::index_t instance_idx = 0;

    bool found_kernel= false;

    ck::static_for<0, std::tuple_size_v<DeviceConvBwdFactory>, 1>{}([&](auto i) -> void {
        const auto device_conv_bwd_instance = std::get<i>(DeviceConvBwdFactory{});
        using DeviceConvBwdInstance = ck::remove_cvref_t<decltype(device_conv_bwd_instance)>;
        auto conv_ptr = std::make_shared<DeviceConvBwdInstance>();

        {
            auto invoker  = conv_ptr->MakeInvoker();
            auto argument = conv_ptr->MakeArgument(in_device_buf.GetDeviceBuffer(),
                                            wei_device_buf.GetDeviceBuffer(),
                                            std::array<const void*, 0>{},
                                            out_device_buf.GetDeviceBuffer(),
                                            ck_args.input_lengths,
                                            ck_args.in_strides,
                                            ck_args.wei_lens,
                                            ck_args.wei_strides,
                                            std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                            std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                            ck_args.out_lens,
                                            ck_args.out_strides,
                                            ck_args.filter_stride,
                                            ck_args.filter_dilation,
                                            ck_args.lPadding,
                                            ck_args.rPadding,
                                            InElementOp{},
                                            WeiElementOp{},
                                            OutElementOp{});

            if(conv_ptr->IsSupportedArgument(argument))
            {
                found_kernel = true;
                MIOPEN_LOG_I("Run conv : " << conv_ptr->GetTypeString());
                float avg_time = invoker.Run(argument, StreamConfig{nullptr, true});
                {
                    std::size_t flop = ck_args.GetFlops();
                    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                    MIOPEN_LOG_I("avg_time:" << avg_time <<" , tflops:" << tflops);
                    if (avg_time < best_avg_time)
                    {
                        best_tflops     = tflops;
                        best_avg_time   = avg_time;
                        best_kernel     = conv_ptr->GetTypeString();
                        MIOPEN_LOG_I("* ^best kernel so far^* ");
                        instance_idx = i;

                        sol.invoker_factory = [
                        conv_ptr = std::move(conv_ptr),
                        problem
                        ](const std::vector<Kernel>& kernels) {
                            return [conv_ptr = std::move(conv_ptr),
                                    problem](const Handle& handle, const AnyInvokeParams& primitive_params) {
                                const auto& fwd_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                                const auto& ck_args  = CKArgs{problem};
                                auto invoker  = conv_ptr->MakeInvoker();
                                auto argument = conv_ptr->MakeArgument(static_cast<InDataType*>(fwd_ctx.tensors.out),
                                                                    static_cast<const WeiDataType*>(fwd_ctx.tensors.w),
                                                                    std::array<const void*, 0>{},
                                                                    static_cast<const OutDataType*>(fwd_ctx.tensors.in),
                                                                    ck_args.input_lengths,
                                                                    ck_args.in_strides,
                                                                    ck_args.wei_lens,
                                                                    ck_args.wei_strides,
                                                                    std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                                                    std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                                                    ck_args.out_lens,
                                                                    ck_args.out_strides,
                                                                    ck_args.filter_stride,
                                                                    ck_args.filter_dilation,
                                                                    ck_args.lPadding,
                                                                    ck_args.rPadding,
                                                                    InElementOp{},
                                                                    WeiElementOp{},
                                                                    OutElementOp{});

                                {
                                    WorkAroundHipEventProfiler prf(handle);
                                    float avg_time = invoker.Run(argument, StreamConfig{handle.GetStream(), false});

                                    if(handle.IsProfilingEnabled())
                                    {
                                        avg_time = handle.GetKernelTime();
                                        handle.ResetKernelTime();
                                        handle.AccumKernelTime(avg_time);
                                        DirectCkMgr::GetInst()->launchCount[ST_JIN_BWD] ++;
                                        if (DirectCkMgr::GetInst()->enableLog)
                                            std::cout << "Un-cached jin bwd is called" << std::endl;
                                    }
                                }
                            };
                        };
                    }
                }
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

ConvSolution ConvDepthwiseBwd::GetSolution(const ExecutionContext& ctx,
                                         const ProblemDescription& problem) const
{
    ReadCacheFile();
    return GetBestSolution(ctx, problem);
}
size_t ConvDepthwiseBwd::GetWorkspaceSize(const ExecutionContext& ctx,
                                          const ProblemDescription& problem) const
{
    const auto is_nhwc = (problem.IsLayoutDefault() == false);
    const int ho     = problem.GetInHeight();
    const int wo     = problem.GetInWidth();
    const int n       = problem.GetInBatchSize();
    const int k      = problem.GetInChannels();
    const int c      = problem.GetOutChannels();
    const int hi     = problem.GetOutHeight();
    const int wi     = problem.GetOutWidth();

    size_t trans_input_size   = 0;
    size_t trans_output_size   = 0;

    bool trans_input_skippable  = true;
    bool trans_output_skippable = true;

    if (is_nhwc)
    {   
        TransposeSolutionDefault2Nhwc trans_input(ctx, problem.GetInDataType(), n, c, hi, wi);
        TransposeSolutionNhwc2Default trans_output(ctx, problem.GetOutDataType(), n, k, ho, wo);

        trans_input_skippable  = trans_input.IsSkippable();
        trans_output_skippable = trans_output.IsSkippable();

        trans_input_size  = trans_input_skippable ? 0 : trans_input.GetOutputTensorSize();
        trans_output_size = trans_output_skippable ? 0 : trans_output.GetOutputTensorSize();
    }

    return trans_input_size + trans_output_size;
}

}
} // namespace solver
} // namespace miopen
