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

#ifndef GUARD_MLIR_COMMON_HPP_
#define GUARD_MLIR_COMMON_HPP_

#include <miopen/conv/context.hpp>

#include <string>

namespace miopen {
namespace solver {
namespace mlir {

std::string GetKernelName(const ConvolutionContext& ctx, bool is_xdlops, int kernel_id = 0);

std::string ConstructBuildOptions(const ConvolutionContext& ctx,
                                  bool is_xdlops,
                                  int kernel_id = 0);

template <typename T>
std::string ConstructBuildOptions(const ConvolutionContext& ctx,
                                  const T& perf_config,
                                  bool is_xdlops,
                                  int kernel_id = 0)
{
    std::ostringstream options{ConstructBuildOptions(ctx, is_xdlops, kernel_id), std::ios::ate};

    // Library does heuristic initialization when no perf_config
    // is specified
    if(!(perf_config == T::MlirHeuristicInitRequest()))
        options << " --perf_config " + perf_config.ToString();

    return options.str();
}

} // namespace mlir
} // namespace solver
} // namespace miopen

#endif
