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
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "group_solver.hpp"

using namespace group_conv_2d;

#define DEFINE_GROUP_CONV_TESTS(type, dir) \
struct GroupConv2D_##dir##_##type : GroupConvTestFix<type, Direction::dir> {}; \
TEST_P(GroupConv2D_##dir##_##type , GroupConv2D_##dir##_##type##_Test) { RunSolver(); } \
INSTANTIATE_TEST_SUITE_P(GroupConv2D_##dir##_##type##_Suite, \
                         GroupConv2D_##dir##_##type, \
                         testing::Combine(testing::ValuesIn(ConvTestConfigs()), \
                                          testing::Values(miopenTensorNHWC)));

DEFINE_GROUP_CONV_TESTS(float, Forward);
DEFINE_GROUP_CONV_TESTS(half, Forward);
DEFINE_GROUP_CONV_TESTS(int8_t, Forward);
