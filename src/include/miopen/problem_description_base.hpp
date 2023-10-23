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

#pragma once

#include <miopen/miopen.h>
#include <miopen/names.hpp>

#include <string>

namespace miopen {

inline std::string GetDataTypeName(miopenDataType_t data_type)
{
    switch(data_type)
    {
    case miopenFloat: return "FP32";
    case miopenHalf: return "FP16";
    case miopenInt8: return "INT8";
    case miopenInt32: return "INT32";
    case miopenBFloat16: return "BF16";
    case miopenDouble: return "FP64";
    case miopenFloat8: return "FP8";
    case miopenBFloat8: return "BF8";
    }

    return "Unknown(" + std::to_string(data_type) + ")";
}

struct ProblemDescriptionBase
{
    ProblemDescriptionBase()                              = default;
    ProblemDescriptionBase(const ProblemDescriptionBase&) = default;
    virtual ~ProblemDescriptionBase()                     = default;

    ProblemDescriptionBase& operator=(const ProblemDescriptionBase&) = default;

    [[nodiscard]] virtual NetworkConfig MakeNetworkConfig() const = 0;
};

} // namespace miopen
