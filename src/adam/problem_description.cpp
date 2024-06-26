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

#include <miopen/adam/problem_description.hpp>
#include <miopen/datatype.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace adam {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype    = paramInDesc.GetType();
    auto kernel   = IsAmp() ? "ampadam" : "adam";
    auto step_ind = ExistStepTensor() ? "device" : "host";

    std::ostringstream ss;

    ss << kernel;
    if(IsAdamW())
        ss << "w";
    if(IsAllContiguous())
        ss << "cont";
    ss << "step" << step_ind;
    ss << "dtype" << dtype;
    if(IsAmp())
    {
        auto grad_dtype = gradInDesc.GetType();
        ss << "grad_dtype" << grad_dtype;
    }

    return NetworkConfig{ss.str()};
}

} // namespace adam

} // namespace miopen
