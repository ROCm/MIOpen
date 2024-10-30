/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/tensor/problem_description.hpp>
#include <miopen/names.hpp>

namespace miopen {

namespace tensor {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    const auto tensor_dim = aTensorDesc.GetLengths().size();

    ss << std::to_string(bTensorDesc.GetType()) << "-" << std::to_string(aTensorDesc.GetType())
       << "-" << std::to_string(tensorOp);

    if(tensor_dim == 1)
    {
        size_t local_threads = 256;
        int max_num_wg       = 4096;
        int num_wg =
            std::clamp(cTensorDesc.GetLengths()[0] / local_threads, size_t(1), size_t(max_num_wg));
        num_wg                = num_wg > max_num_wg ? max_num_wg : num_wg;
        size_t global_threads = num_wg * local_threads;
        ss << "-" << std::to_string(global_threads) << "-" << std::to_string(local_threads);

        if(aTensorDesc.AllDimsFitIntoInt())
        {
            ss << "-32bit";
        }
        else
        {
            ss << "-64bit";
        }
    }

    return NetworkConfig{ss.str()};
}

} // namespace tensor

} // namespace miopen
