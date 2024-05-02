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

#include <cstdint>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace glu {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& outputDesc_,
                       int32_t dim_)
        : direction(Direction::Forward), inputDesc(inputDesc_), outputDesc(outputDesc_), dim(dim_)
    {
        if(inputDesc.GetLengths().size() != outputDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GLU::ProblemDescription: Number of tensor dimension do not match.");
        }
        if(inputDesc.GetLengths()[dim] % 2 != 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GLU::ProblemDescription: The split dimension size of input tensor should "
                         "be divisible by 2.");
        }
    }

    // Backward constructor
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& inputGradDesc_,
                       const TensorDescriptor& outputGradDesc_,
                       int32_t dim_)
        : direction(Direction::Backward), inputDesc(inputDesc_), inputGradDesc(inputGradDesc_), outputGradDesc(outputGradDesc_), dim(dim_)
    {
        if(inputDesc.GetLengths().size() != inputGradDesc.GetLengths().size() ||
           inputDesc.GetLengths().size() != outputGradDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GLU::ProblemDescription: Number of tensor dimension do not match.");
        }
        if(inputDesc.GetLengths()[dim] % 2 != 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GLU::ProblemDescription: The split dimension size of input tensor should "
                         "be divisible by 2.");
        }
    }

    Direction GetDirection() const { return direction; }
    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    int32_t GetDim() const { return dim; }

    bool IsSameType() const
    {
        if (direction == Direction::Forward) {
            if(inputDesc.GetType() != outputDesc.GetType())
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "GLU: Tensor types do not match.");
#else
                return false;
#endif
            }
        } else {
            if(inputDesc.GetType() != inputGradDesc.GetType() 
                || inputGradDesc.GetType() != outputGradDesc.GetType())
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "GLU: Tensor types do not match.");
#else
                return false;
#endif                
            }
        }
        
        return true;
    }

    bool IsRightLength() const
    {
        if (direction == Direction::Forward) {
            for(int32_t i = 0; i < inputDesc.GetLengths().size(); i++)
            {
                if(i == dim)
                {
                    if(inputDesc.GetLengths()[i] / 2 != outputDesc.GetLengths()[i])
                    {
                        return false;
                    }
                }
                else
                {
                    if(inputDesc.GetLengths()[i] != outputDesc.GetLengths()[i])
                    {
                        return false;
                    }
                }
            }
        } else
        {
            for(int32_t i = 0; i < inputDesc.GetLengths().size(); i++)
            {
                if(i == dim)
                {
                    if(inputDesc.GetLengths()[i] / 2 != outputGradDesc.GetLengths()[i] || inputDesc.GetLengths()[i] != inputGradDesc.GetLengths()[i])
                    {
                        return false;
                    }
                }
                else
                {
                    if(inputDesc.GetLengths()[i] != inputGradDesc.GetLengths()[i] || inputDesc.GetLengths()[i] != outputGradDesc.GetLengths()[i])
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool IsRightDim() const
    {
        if((dim < 0) || (dim > outputDesc.GetLengths().size()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(
                miopenStatusBadParm,
                "GLU: Dimension is greater than 0 and less than or equal tensor dimension length.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if (direction == Direction::Forward) {
            if(!(inputDesc.IsPacked() && outputDesc.IsPacked()))
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "GLU: Unpacked tensors not supported.");
#else
                return false;
#endif
            }            
        } else {
            if(!(inputDesc.IsPacked() && inputGradDesc.IsPacked() && outputGradDesc.IsPacked()))
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "GLU: Unpacked tensors not supported.");
#else
                return false;
#endif
            }
        }

        return true;
    }

    bool IsFirstDim() const
    {
        if(dim != 0)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "GLU: Dimension is not 0.");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    Direction direction;
    TensorDescriptor inputDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor outputGradDesc;

    int32_t dim;

    NetworkConfig MakeForwardNetworkConfig() const;
    NetworkConfig MakeBackwardNetworkConfig() const;
};

} // namespace glu

} // namespace miopen
