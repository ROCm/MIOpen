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

#include <miopen/graphapi/reduction.hpp>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::OperationReduction;
using miopen::graphapi::OperationReductionBuilder;
using miopen::graphapi::Reduction;
using miopen::graphapi::Tensor;

} // namespace

TEST(GraphApiOperationReductionBuilder, Test)
{
    Reduction reduction{MIOPEN_REDUCE_TENSOR_ADD, miopenFloat};
    Tensor x{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 1, false};
    Tensor y[] = {{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 2, false},
                  {miopenFloat, {8, 1, 64}, {64, 64, 1}, 2, false},
                  {miopenFloat, {8, 64, 1}, {64, 1, 1}, 2, false},
                  {miopenFloat, {8, 1, 1}, {1, 1, 1}, 2, false},
                  {miopenFloat, {8, 128, 1}, {128, 1, 1}, 2, false}};
    Tensor badY{miopenFloat, {8, 32, 32}, {32 * 32, 32, 1}, 2, false};

    for(Tensor& pY : y)
    {
        EXPECT_NO_THROW({
            OperationReductionBuilder().setReduction(&reduction).setX(&x).setY(&pY).build();
        }) << "Builder failed on valid attributes";
    }
    EXPECT_ANY_THROW({
        OperationReductionBuilder().setReduction(&reduction).setX(&x).setY(&badY).build();
    }) << "Builder failed on invalid attributes";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setReduction(nullptr); })
        << "OperationReductionBuilder::setReduction failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setX(nullptr); })
        << "OperationReductionBuilder::setX failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setY(nullptr); })
        << "OperationReductionBuilder::setY failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setX(&x).setY(y).build(); })
        << "Builder failed to detect missing setReduction call";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setReduction(&reduction).setY(y).build(); })
        << "Builder failed to detect missing setX call";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setReduction(&reduction).setX(&x).build(); })
        << "Builder failed to detect missing setY call";
}
