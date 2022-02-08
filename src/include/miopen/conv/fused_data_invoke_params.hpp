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

#include <miopen/invoke_params.hpp>
#include <miopen/conv/tensors.hpp>

namespace miopen {
namespace conv {

struct FusedDataInvokeParams : InvokeParams
{
    FusedConvDataTensors tensors;
    Data_t workSpace;
    std::size_t workSpaceSize;
    bool gfx90aFp16alt;

    FusedDataInvokeParams(FusedConvDataTensors tensors_,
                          Data_t workSpace_,
                          std::size_t workSpaceSize_,
                          bool gfx90aFp16alt_)
        : tensors(tensors_),
          workSpace(workSpace_),
          workSpaceSize(workSpaceSize_),
          gfx90aFp16alt(gfx90aFp16alt_)
    {
    }

    FusedDataInvokeParams(InvokeType type_,
                          FusedConvDataTensors tensors_,
                          Data_t workSpace_,
                          std::size_t workSpaceSize_,
                          bool gfx90aFp16alt_)
        : InvokeParams{type_},
          tensors(tensors_),
          workSpace(workSpace_),
          workSpaceSize(workSpaceSize_),
          gfx90aFp16alt(gfx90aFp16alt_)
    {
    }
};

} // namespace conv
} // namespace miopen
