/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_SOLVER_GEMM_COMMON_HPP
#define GUARD_MIOPEN_SOLVER_GEMM_COMMON_HPP

#include <miopen/config.h>
#include <miopen/conv/problem_description.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>

#include <cstddef>

namespace miopen {
namespace solver {
namespace conv {
namespace gemm {

std::size_t MaxMemAllocSz(const Handle& h,
                          const miopen::conv::ProblemDescription& problem,
                          bool double_limit_for_fp32 = false);

constexpr bool IsBf16Supported = MIOPEN_USE_ROCBLAS;
constexpr bool IsFp16Supported = MIOPEN_USE_ROCBLAS;

bool IsAnyBufferBf16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc);
bool IsAnyBufferFp16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc);

double SlowdownFactor(int n_oper, double oper_factor, double multiple_oper_factor);

} // namespace gemm
} // namespace conv
} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_SOLVER_GEMM_COMMON_HPP
