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

#include <miopen/datatype.hpp>
#include <miopen/glu/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace glu {

ProblemDescription::ProblemDescription(const TensorDescriptor& inputDesc_,
                                       const TensorDescriptor& outputDesc_,
                                       uint32_t dim_)
    : direction(Direction::Forward), inputDesc(inputDesc_), outputDesc(outputDesc_), dim(dim_)
{
    if(inputDesc.GetNumDims() != outputDesc.GetNumDims())
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "GLU::ProblemDescription: Number of dimensions between input and output "
                     "tensor do not match.");
    }

    if(dim >= inputDesc.GetNumDims())
    {
        MIOPEN_THROW(miopenStatusBadParm, "GLU::ProblemDescription: Dimension is out of range.");
    }

    if(inputDesc.GetLengths()[dim] % 2 != 0)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "GLU::ProblemDescription: The split dimension size of input tensor should "
                     "be divisible by 2.");
    }

    for(auto i = 0; i < inputDesc.GetNumDims(); i++)
    {
        if(i == dim)
        {
            if(inputDesc.GetLengths()[i] / 2 != outputDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "GLU::ProblemDescription: Dimension sizes don't match between "
                             "input tensor and output tensor.");
            }
        }
        else
        {
            if(inputDesc.GetLengths()[i] != outputDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "GLU::ProblemDescription: Dimension sizes don't match between "
                             "input tensor and output tensor.");
            }
        }
    }

    if(!IsSameType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "GLU::ProblemDescription: Tensor types do not match.");
    }
}

ProblemDescription::ProblemDescription(const TensorDescriptor& inputDesc_,
                                       const TensorDescriptor& outputGradDesc_,
                                       const TensorDescriptor& inputGradDesc_,
                                       uint32_t dim_)
    : direction(Direction::Backward),
      inputDesc(inputDesc_),
      outputGradDesc(outputGradDesc_),
      inputGradDesc(inputGradDesc_),
      dim(dim_)
{
    if(inputDesc.GetNumDims() != inputGradDesc.GetNumDims() ||
       inputDesc.GetNumDims() != outputGradDesc.GetNumDims())
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "GLU::ProblemDescription: Number of tensor dimensions between input and "
                     "output tensor do not match.");
    }

    if(dim >= inputDesc.GetNumDims())
    {
        MIOPEN_THROW(miopenStatusBadParm, "GLU::ProblemDescription: Dimension is out of range.");
    }

    if(inputDesc.GetLengths()[dim] % 2 != 0)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "GLU::ProblemDescription: The split dimension size of input tensor should "
                     "be divisible by 2.");
    }

    for(auto i = 0; i < inputDesc.GetNumDims(); i++)
    {
        if(i == dim)
        {
            if(inputDesc.GetLengths()[i] / 2 != outputGradDesc.GetLengths()[i] ||
               inputDesc.GetLengths()[i] != inputGradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "GLU::ProblemDescription: Dimension sizes don't match between "
                             "input tensor and output tensor.");
            }
        }
        else
        {
            if(inputDesc.GetLengths()[i] != inputGradDesc.GetLengths()[i] ||
               inputDesc.GetLengths()[i] != outputGradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "GLU::ProblemDescription: Dimension sizes don't match between "
                             "input tensor and output tensor.");
            }
        }
    }

    if(!IsSameType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "GLU::ProblemDescription: Tensor types do not match.");
    }
}

bool ProblemDescription::IsSameType() const
{
    if(direction == Direction::Forward)
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            return false;
        }
    }
    else
    {
        if(inputDesc.GetType() != inputGradDesc.GetType() ||
           inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }
    }

    return true;
}

bool ProblemDescription::IsAllContiguous() const
{
    if(direction == Direction::Forward)
    {
        if(!(inputDesc.IsContiguous() && outputDesc.IsContiguous()))
        {
            return false;
        }
    }
    else
    {
        if(!(inputDesc.IsContiguous() && inputGradDesc.IsContiguous() &&
             outputGradDesc.IsContiguous()))
        {
            return false;
        }
    }

    return true;
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto input_numel = inputDesc.GetElementSize();
    auto io_dtype    = miopen::GetDataType(inputDesc.GetType());

    std::ostringstream ss;

    ss << "io_dtype" << io_dtype;
    ss << "dim" << dim;
    ss << "input_numel" << input_numel;
    ss << IsAllContiguous();

    return NetworkConfig{ss.str()};
}

} // namespace glu

} // namespace miopen
