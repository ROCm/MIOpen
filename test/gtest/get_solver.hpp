/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>
#include "conv_common.hpp"
#include "get_handle.hpp"
#include "tensor_util.hpp"
#include <miopen/conv/data_invoke_params.hpp>

#include <miopen/type_name.hpp>
#include <miopen/rank.hpp>

template <typename Solver, typename Context, typename Problem>
auto GetSolutionImpl(miopen::rank<1>, Solver s, const Context& ctx, const Problem& problem)
    -> decltype(s.GetSolution(ctx, problem, s.GetDefaultPerformanceConfig(ctx, problem)))
{
    return s.GetSolution(ctx, problem, s.GetDefaultPerformanceConfig(ctx, problem));
}

template <typename Solver, typename Context, typename Problem>
auto GetSolutionImpl(miopen::rank<0>, Solver s, const Context& ctx, const Problem& problem)
    -> decltype(s.GetSolution(ctx, problem))
{
    return s.GetSolution(ctx, problem);
}

template <typename Solver, typename Context, typename Problem>
miopen::solver::ConvSolution GetSolution(Solver s, const Context& ctx, const Problem& problem)
{
    auto solution = GetSolutionImpl(miopen::rank<1>{}, s, ctx, problem);
    return solution;
}
