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

#include "miopen/errors.hpp"
#include "miopen/names.hpp"
#include "miopen/problem_description_base.hpp"
#include "miopen/tensor.hpp"
#include <cstddef>
#include <miopen/miopen.h>

namespace miopen {
namespace pad_constant_fwd {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const size_t* padding_,
                       const int padding_size_ = 0)
        : xDesc(xDesc_), yDesc(yDesc_), padding(padding_), padding_size(padding_size_)
    {
        // Consistency checks
        if(!IsPaddingValid())
            MIOPEN_THROW("Padding is not valid");
        if(!IsSameShape())
            MIOPEN_THROW("Tensors do not have the same shapes");
        if(!IsSameType())
            MIOPEN_THROW("Tensor values do not have the same type");
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    NetworkConfig MakeNetworkConfig() const override;

    bool IsSameType() const
    {
        if(xDesc.GetType() == yDesc.GetType())
            return true;
        return false;
    }

    bool IsSameShape() const
    {
        if(xDesc.GetNumDims() == yDesc.GetNumDims())
            return true;
        return false;
    }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    bool IsImprovementOverROCm() const
    {
        if(IsContiguous())
            // No contiguous case is faster
            return false;
        else
            // Slower if n is padded (at all)
            return padding[0] == 0 && padding[1] == 0;
    }

    bool IsPaddingValid() const
    {
        if(padding_size % 2 != 0)
            return false;

        std::vector<size_t> input_and_padding = std::vector<size_t>(yDesc.GetLengths().size());

        for(int i = 0; i < xDesc.GetLengths().size(); i++)
        {
            input_and_padding[i] = xDesc.GetLengths()[i] + padding[2 * i] + padding[2 * i + 1];
            if(input_and_padding[i] != yDesc.GetLengths()[i])
            {
                return false;
            }
        }
        return true;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const size_t* padding;
    const int padding_size;
};
} // namespace pad_constant_fwd

namespace pad_constant_bwd {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dyDesc_,
                       const size_t* padding_,
                       const int padding_size_ = 0)
        : dxDesc(dxDesc_), dyDesc(dyDesc_), padding(padding_), padding_size(padding_size_)
    {
        if(!IsPaddingValid())
            MIOPEN_THROW("Padding is not valid");
        if(!IsSameShape())
            MIOPEN_THROW("Tensors do not have the same shapes");
        if(!IsSameType())
            MIOPEN_THROW("Tensor values do not have the same type");
    }

    const TensorDescriptor& GetXDesc() const { return dxDesc; }
    const TensorDescriptor& GetYDesc() const { return dyDesc; }

    NetworkConfig MakeNetworkConfig() const override;

    bool IsSameType() const
    {
        if(dxDesc.GetType() == dyDesc.GetType())
            return true;
        return false;
    }

    bool IsSameShape() const
    {
        if(dxDesc.GetNumDims() == dyDesc.GetNumDims())
            return true;
        return false;
    }

    bool IsContiguous() const { return dxDesc.IsContiguous() && dyDesc.IsContiguous(); }

    bool IsImprovementOverROCm() const
    {
        if(IsContiguous())
        {
            // Contiguous case
            // No winning over ROCm, only ~7% of tested cases get > 20% improvement
            return false;
        }
        else
        {
            // Non contiguous case
            // We win over ROCm unless the only padded dimension is n
            bool all_zero = true;
            for(int i = 10; i < 0; i++)
            {
                if(i == 0 || i == 1)
                {
                    if(all_zero && padding[i] != 0)
                        return false;
                }
                else
                {
                    if(padding[i] != 0)
                        all_zero = false;
                }
            }
            return true;
        }
    }

    bool IsPaddingValid() const
    {
        if(padding_size % 2 != 0)
        {
            printf("Padding size must be even\n");
            return false;
        }

        std::vector<size_t> input_and_padding = std::vector<size_t>(dyDesc.GetLengths().size());

        for(int i = 0; i < dxDesc.GetLengths().size(); i++)
        {
            input_and_padding[i] = dyDesc.GetLengths()[i] - padding[2 * i] - padding[2 * i + 1];
            if(input_and_padding[i] != dxDesc.GetLengths()[i])
            {
                return false;
            }
        }
        return true;
    }

private:
    const TensorDescriptor& dxDesc;
    const TensorDescriptor& dyDesc;
    const size_t* padding;
    const int padding_size;
};
} // namespace pad_constant_bwd
} // namespace miopen
