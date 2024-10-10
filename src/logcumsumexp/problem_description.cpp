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

#include <miopen/logcumsumexp/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace logcumsumexp {

ForwardProblemDescription::ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                                                     const TensorDescriptor& outputDesc_,
                                                     const int dim_)
    : inputDesc(inputDesc_), outputDesc(outputDesc_), dim(dim_)
{
    if(IsValidDim())
        dim = (dim < 0 ? dim + inputDesc.GetNumDims() : dim);
    IsSameLength();
    IsSameType();
}

bool ForwardProblemDescription::IsValidDim() const
{
    const int ndims = inputDesc.GetNumDims();
    if(dim < -ndims || ndims - 1 < dim)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream() << "LogCumSumExp: Operating dim value must be in range ["
                                          << -ndims << "," << ndims - 1 << "].")
                         .str());
    }
    return true;
}

bool ForwardProblemDescription::IsSameLength() const
{
    if(inputDesc.GetLengths() != outputDesc.GetLengths())
        MIOPEN_THROW(miopenStatusBadParm,
                     "LogCumSumExp: Input and Output tensor sizes do not match.");
    return true;
}

bool ForwardProblemDescription::IsSameType() const
{
    if(inputDesc.GetType() != outputDesc.GetType())
        MIOPEN_THROW(miopenStatusBadParm,
                     "LogCumSumExp: Input and Output tensor type do not match.");
    return true;
}

bool ForwardProblemDescription::IsSameStride() const
{
    if(inputDesc.GetStrides() != outputDesc.GetStrides())
        return false;
    return true;
}

bool ForwardProblemDescription::IsAllPacked() const
{
    if(!inputDesc.IsPacked() || !outputDesc.IsPacked())
        return false;
    return true;
}

bool ForwardProblemDescription::IsAllDimStride1() const
{
    if(inputDesc.GetStrides()[dim] != 1)
        return false;
    if(outputDesc.GetStrides()[dim] != 1)
        return false;
    return true;
}

BackwardProblemDescription::BackwardProblemDescription(const TensorDescriptor& inputDesc_,
                                                       const TensorDescriptor& outputDesc_,
                                                       const TensorDescriptor& doutputDesc_,
                                                       const TensorDescriptor& dinputDesc_,
                                                       const int& dim_)
    : ForwardProblemDescription(inputDesc_, outputDesc_, dim_),
      doutputDesc(doutputDesc_),
      dinputDesc(dinputDesc_)
{
    IsSameLength();
    IsSameType();
}

bool BackwardProblemDescription::IsSameLength() const
{
    if(!ForwardProblemDescription::IsSameLength())
        return false;
    if(inputDesc.GetLengths() != dinputDesc.GetLengths())
        MIOPEN_THROW(miopenStatusBadParm,
                     "LogCumSumExp: Input and its Gradient tensor sizes do not match.");
    if(outputDesc.GetLengths() != doutputDesc.GetLengths())
        MIOPEN_THROW(miopenStatusBadParm,
                     "LogCumSumExp: Output and its Gradient tensor sizes do not match.");
    return true;
}

bool BackwardProblemDescription::IsSameType() const
{
    if(!ForwardProblemDescription::IsSameType())
        return false;
    if(inputDesc.GetType() != dinputDesc.GetType())
        MIOPEN_THROW(miopenStatusBadParm,
                     "LogCumSumExp: Input and its Gradient tensor type do not match.");
    if(outputDesc.GetType() != doutputDesc.GetType())
        MIOPEN_THROW(miopenStatusBadParm,
                     "LogCumSumExp: Output and its Gradient tensor type do not match.");
    return true;
}

bool BackwardProblemDescription::IsSameStride() const
{
    if(!ForwardProblemDescription::IsSameStride())
        return false;
    if(inputDesc.GetStrides() != dinputDesc.GetStrides())
        return false;
    if(outputDesc.GetStrides() != doutputDesc.GetStrides())
        return false;
    return true;
}

bool BackwardProblemDescription::IsAllPacked() const
{
    if(!ForwardProblemDescription::IsAllPacked())
        return false;
    if(!dinputDesc.IsPacked() || !doutputDesc.IsPacked())
        return false;
    return true;
}

bool BackwardProblemDescription::IsAllDimStride1() const
{
    if(!ForwardProblemDescription::IsAllDimStride1())
        return false;
    if(dinputDesc.GetStrides()[dim] != 1)
        return false;
    if(doutputDesc.GetStrides()[dim] != 1)
        return false;
    return true;
}

NetworkConfig ForwardProblemDescription::MakeNetworkConfig() const
{
    auto dtype      = inputDesc.GetType();
    auto size       = inputDesc.GetElementSize();
    auto inner_size = inputDesc.GetLengths()[dim];
    auto outer_size = size / inner_size;

    std::ostringstream ss;

    ss << "logcumsumexp_fwd";
    ss << "dtype" << dtype;
    ss << "outer" << outer_size;
    ss << "inner" << inner_size;
    ss << "packed" << IsAllPacked();
    ss << "dimstride1" << IsAllDimStride1();

    return NetworkConfig{ss.str()};
}

NetworkConfig BackwardProblemDescription::MakeNetworkConfig() const
{
    auto dtype      = inputDesc.GetType();
    auto size       = inputDesc.GetElementSize();
    auto inner_size = inputDesc.GetLengths()[dim];
    auto outer_size = size / inner_size;

    std::ostringstream ss;

    ss << "logcumsumexp_bwd";
    ss << "dtype" << dtype;
    ss << "outer" << outer_size;
    ss << "inner" << inner_size;
    ss << "packed" << IsAllPacked();
    ss << "dimstride1" << IsAllDimStride1();

    return NetworkConfig{ss.str()};
}

} // namespace logcumsumexp

} // namespace miopen
