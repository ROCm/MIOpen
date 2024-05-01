/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#pragma once

#include <miopen/conv_solution.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/problem_description.hpp>
#include "miopen/../../kernels/stride_array.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace miopen {
namespace solver {
namespace conv {

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&);
std::string ConvDirectNaiveConvKernelName(const miopen::conv::ProblemDescription&);
std::string ConvDirectNaiveConvKernelFile(const ExecutionContext& ctx,
                                          const miopen::conv::ProblemDescription& problem);
std::string ConvDirectNaiveConvCompileOption(const ExecutionContext& ctx,
                                             const miopen::conv::ProblemDescription& problem);
bool ConvDirectNaiveConvIsApplicableByKernelType(const ExecutionContext&,
                                                 const miopen::conv::ProblemDescription&);

bool IsInputFp32(const miopen::conv::ProblemDescription&);
bool IsInputFp16(const miopen::conv::ProblemDescription&);
bool IsInputBfp16(const miopen::conv::ProblemDescription&);
bool IsInputInt8(const miopen::conv::ProblemDescription&);
bool IsAccFp64(const miopen::conv::ProblemDescription&);
bool IsAccInt32(const miopen::conv::ProblemDescription&);
bool IsOutputFp32(const miopen::conv::ProblemDescription&);
bool IsOutputFp16(const miopen::conv::ProblemDescription&);
bool IsOutputBfp16(const miopen::conv::ProblemDescription&);
bool IsOutputInt8(const miopen::conv::ProblemDescription&);
bool IsOutputInt32(const miopen::conv::ProblemDescription&);

namespace conv_internal {

void DebugPrintTensorStrides(const TensorDescriptor& inDesc,
                             const TensorDescriptor& wDesc,
                             const TensorDescriptor& outDesc);

/**
 * Get the index where group (G) stride should go. For NCHW, we want to convert
 * its strides to NGCHW, and for NHWC, we want to convert its strides to NHWGC.
 * Same applies for the 3D case.
 */
int GetGroupStrideIndex(const miopen::conv::ProblemDescription& problem);

/**
 * split the strides for C dimension in a tensor descriptor into (G, C_per_group).
 * Normally, (in packed case) num channels is a multiplying factor in the stride of
 * whatever lies to the left of C, e.g., in NCHW, N's stride contains C as a
 * factor. We output NGCHW for NCHW (and NHWGC for NHWC)
 * where the stride[G] = stride[N] / num_groups
 */
template <typename V>
V SplitStrideCtoGC(int num_groups, const V& orig_strides, int G_stride_idx)
{
    assert(G_stride_idx > 0 && G_stride_idx <= orig_strides.size());
    // (G_stride_idx - 1) is the stride index of whatever lies to the left and
    // contains C or K as a multiplying factor. We divide this value by num_groups
    // to get G_stride_val
    assert(orig_strides[G_stride_idx - 1] % num_groups == 0);

    V ret{orig_strides};
    auto G_stride_val = orig_strides[G_stride_idx - 1] / num_groups;

    ret.insert(ret.begin() + G_stride_idx, G_stride_val);

    return ret;
}

/**
 * Weight tensor has original dims: [K, C_per_group, Y, X] (2D case)
 * We return a new stride vector with strides for [G, K_per_group, C_per_group, Y, X]
 * Stride for G is computed as stride[C_per_group] * K_per_group and inserted at
 * left most position
 */
template <typename V>
V SplitWeiStrideKtoGK(int k_per_group, const V& wei_strides)
{
    V ret{wei_strides};
    ret.insert(ret.begin(), wei_strides[0] * k_per_group);
    return ret;
}

template <unsigned N>
struct ChooseStride
{
};

template <>
struct ChooseStride<5u>
{
    using type = Strides5D;
};

template <>
struct ChooseStride<6u>
{
    using type = Strides6D;
};

template <unsigned N, typename V>
auto MakeStrideArray(V vec)
{
    typename ChooseStride<N>::type ret;
    assert(vec.size() == N);

    // MIOpen stores strides for NHWC in NCHW order, i.e. C stride in 2nd from left.
    // We sort the input stride vector so that smallest stride is at index 0. This
    // (little-endian) order is what naive convolution kernel expects for strides
    std::sort(vec.begin(), vec.end());

    for(unsigned i = 0; i < N; ++i)
    {
        ret[i] = static_cast<StrideIndexType>(vec[i]);
    }
    return ret;
}

::miopen::solver::ConvSolution
GetConv2DFWDSolution(const ExecutionContext& ctx,
                     const ::miopen::conv::ProblemDescription& problem);

::miopen::solver::ConvSolution
GetConv3DFWDSolution(const ExecutionContext& ctx,
                     const ::miopen::conv::ProblemDescription& problem);

::miopen::solver::ConvSolution
GetConv2DWRWSolution(const ExecutionContext& ctx,
                     const ::miopen::conv::ProblemDescription& problem);

::miopen::solver::ConvSolution
GetConv3DWRWSolution(const ExecutionContext& ctx,
                     const ::miopen::conv::ProblemDescription& problem);

::miopen::solver::ConvSolution
GetConv2DBWDSolution(const ExecutionContext& ctx,
                     const ::miopen::conv::ProblemDescription& problem);

::miopen::solver::ConvSolution
GetConv3DBWDSolution(const ExecutionContext& ctx,
                     const ::miopen::conv::ProblemDescription& problem);
} // end namespace conv_internal

} // namespace conv
} // namespace solver
} // namespace miopen
