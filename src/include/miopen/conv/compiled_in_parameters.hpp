/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/conv/context.hpp>
#include <miopen/handle.hpp>

#include <cassert>

namespace miopen {
/*
 * returns parameter values that are compiled in legacy kernels for kernels using them as
 * arguments.
 */
inline void GetCompiledInParameters(const ExecutionContext& ctx,
                                    const ProblemDescription& problem,
                                    int* const N,
                                    int* const C,
                                    int* const H,
                                    int* const W,
                                    int* const K,
                                    int* const n_groups)
{
    assert(N && C && H && W && K && n_groups);
    *N        = problem.GetBatchSize_();
    *C        = problem.GetInChannels_();
    *H        = problem.GetInHeight_();
    *W        = problem.GetInWidth_();
    *K        = problem.GetOutChannels_();
    *n_groups = ctx.GetStream().GetMaxComputeUnits();
}

inline void GetCompiledInParameters(const ExecutionContext& ctx,
                                    const ProblemDescription& problem,
                                    int* const N,
                                    int* const C,
                                    int* const H,
                                    int* const W,
                                    int* const K,
                                    int* const n_groups,
                                    int* const out_H,
                                    int* const out_W)
{
    GetCompiledInParameters(ctx, problem, N, C, H, W, K, n_groups);
    assert(out_H && out_W);
    *out_H = problem.GetOutHeight_();
    *out_W = problem.GetOutWidth_();
}

inline void GetCompiledInParameters(const ExecutionContext& ctx,
                                    const ProblemDescription& problem,
                                    int* const N,
                                    int* const C,
                                    int* const H,
                                    int* const W,
                                    int* const K,
                                    int* const n_groups,
                                    int* const out_H,
                                    int* const out_W,
                                    int* const filter_size_H,
                                    int* const filter_size_W,
                                    int* const pad_H,
                                    int* const pad_W)
{
    GetCompiledInParameters(ctx, problem, N, C, H, W, K, n_groups, out_H, out_W);
    assert(filter_size_H && filter_size_W && pad_H && pad_W);
    *filter_size_H = problem.GetWeightsHeight_();
    *filter_size_W = problem.GetWeightsWidth_();
    *pad_H         = problem.direction.IsForward() ? problem.GetPadH() : problem.GetBackwardPadH();
    *pad_W         = problem.direction.IsForward() ? problem.GetPadW() : problem.GetBackwardPadW();
}

} // namespace miopen
