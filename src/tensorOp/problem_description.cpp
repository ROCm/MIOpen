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

#include <miopen/tensorOp/problem_description.hpp>
#include <miopen/names.hpp>

namespace miopen {

namespace tensorOp {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    auto alens = aTensorDesc.GetLengths();
    auto blens = bTensorDesc.GetLengths();

    auto astrides = aTensorDesc.GetStrides();
    auto bstrides = bTensorDesc.GetStrides();
    auto cstrides = cTensorDesc.GetStrides();

    std::string alens_str{};
    std::string blens_str{};
    std::string astrides_str{};
    std::string bstrides_str{};
    std::string cstrides_str{};

    for(uint32_t i = 0; i < alens.size(); i++)
    {
        alens_str += std::to_string(alens[i]);
        blens_str += std::to_string(blens[i]);
        astrides_str += std::to_string(astrides[i]);
        bstrides_str += std::to_string(bstrides[i]);
        cstrides_str += std::to_string(cstrides[i]);

        if(i != (alens.size() - 1))
        {
            alens_str += "x";
            blens_str += "x";
            astrides_str += "x";
            bstrides_str += "x";
            cstrides_str += "x";
        }
    }

    ss << std::to_string(aTensorDesc.GetType()) << "-" << std::to_string(tensorOp) << "-"
       << alens_str << "-" << blens_str << "-" << astrides_str << "-" << bstrides_str << "-"
       << cstrides_str << "-" << std::to_string((beta == 0));

    return NetworkConfig{ss.str()};
}

} // namespace tensorOp

} // namespace miopen
