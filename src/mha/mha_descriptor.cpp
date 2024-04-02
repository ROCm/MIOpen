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
#include <miopen/mha/mha_descriptor.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

namespace miopen {

extern "C" miopenStatus_t miopenCreateMhaDescriptor(miopenMhaDescriptor_t* mhaDesc)
{
    MIOPEN_LOG_FUNCTION(mhaDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(mhaDesc);
        desc       = new miopen::MhaDescriptor();
    });
}

extern "C" miopenStatus_t miopenSetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float scale)
{
    MIOPEN_LOG_FUNCTION(mhaDesc, scale);
    return miopen::try_([&] { miopen::deref(mhaDesc).SetParams(scale); });
}

extern "C" miopenStatus_t miopenGetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float* scale)
{
    MIOPEN_LOG_FUNCTION(mhaDesc);
    return miopen::try_([&] { *scale = miopen::deref(mhaDesc).GetScale(); });
}

std::ostream& operator<<(std::ostream& stream, const MhaDescriptor& x)
{
    stream << "mha,"
           << "scale" << x.GetScale() << ",";

    return stream;
}

void to_json(nlohmann::json& json, const MhaDescriptor& descriptor)
{
    json = nlohmann::json{
        {"scale", descriptor.GetScale()},
    };
}

void from_json(const nlohmann::json& json, MhaDescriptor& descriptor)
{
    json.at("scale").get_to(descriptor.scale);
}

} // namespace miopen
