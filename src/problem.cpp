/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/problem.hpp>

#include <miopen/convolution.hpp>
#include <miopen/handle.hpp>
#include <miopen/solution.hpp>
#include <miopen/search_options.hpp>

namespace miopen {

std::vector<Solution> Problem::FindSolutions(Handle& handle, const SearchOptions& options) const
{
    if(!operator_descriptor)
        MIOPEN_THROW(miopenStatusInvalidValue, "Problem operator descriptor has not been set.");

    switch(operator_descriptor->GetPrimitive())
    {
    case solver::Primitive::Convolution: {

        const auto& conv_desc = *static_cast<ConvolutionDescriptor*>(operator_descriptor.get());

        if(tensor_descriptors.size() != 3)
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Convolution problem should have exactly three tensor descriptors.");

        const auto checked_get_tensor_descriptor = [&](auto name, const std::string& name_str) {
            const auto found = tensor_descriptors.find(name);
            if(found == tensor_descriptors.end())
                MIOPEN_THROW(miopenStatusInvalidValue,
                             "Convolution problem is missing " + name_str + " tensor descriptor.");
            return found->second;
        };

        const auto& x_desc =
            checked_get_tensor_descriptor(miopenTensorConvolutionX, "miopenTensorConvolutionX");
        const auto& w_desc =
            checked_get_tensor_descriptor(miopenTensorConvolutionW, "miopenTensorConvolutionW");
        const auto& y_desc =
            checked_get_tensor_descriptor(miopenTensorConvolutionY, "miopenTensorConvolutionY");

        auto x = handle.Create(x_desc.GetElementSpace());
        auto w = handle.Create(w_desc.GetElementSpace());
        auto y = handle.Create(y_desc.GetElementSpace());

        std::ignore = conv_desc;
        std::ignore = x_desc;
        std::ignore = w_desc;
        std::ignore = y_desc;
        std::ignore = x;
        std::ignore = w;
        std::ignore = y;
        std::ignore = options;

        MIOPEN_THROW(miopenStatusNotImplemented);
    }
    break;
    case solver::Primitive::Activation:
    case solver::Primitive::Batchnorm:
    case solver::Primitive::Pooling:
    case solver::Primitive::Invalid:
    default: MIOPEN_THROW(miopenStatusNotImplemented); break;
    }
}

} // namespace miopen
