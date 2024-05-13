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
#ifndef MIOPEN_MHADESCRIPTOR_HPP_
#define MIOPEN_MHADESCRIPTOR_HPP_

#include <miopen/common.hpp>
#include <miopen/mha/mha.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <nlohmann/json_fwd.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

struct MhaDescriptor : miopenMhaDescriptor
{
    MhaDescriptor() {}

    void SetParams(float scale_) { scale = scale_; }

    float GetScale() const { return scale; }

    friend std::ostream& operator<<(std::ostream& stream, const MhaDescriptor& x);

    friend void to_json(nlohmann::json& json, const MhaDescriptor& descriptor);
    friend void from_json(const nlohmann::json& json, MhaDescriptor& descriptor);

private:
    float scale;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenMhaDescriptor, miopen::MhaDescriptor);

#endif // MIOPEN_MHADESCRIPTOR_HPP_
