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

#include <miopen/softmax/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace softmax {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{  
    std::string network_config;

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(yDesc.GetLengths());

    // using workgroup size of 256 by default
    int grid_size   = mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? n : n * h * w;
    int spatial_dim = mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? 1 : h * w;
    int vector_size = mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? c * h * w : c;
    // num_spatial_dims or pixels each workgroup can compute
    int num_batch = vector_size < 256 ? nextPow2(256 / vector_size) : 1;

    const std::vector<size_t> vld{256, 1, 1};

    bool usefp16 = false;
    bool usefp32 = true;
    if(yDesc.GetType() == miopenHalf)
    {
        usefp16 = true;
        usefp32 = false;
    }

    auto alpha_fp = *(static_cast<const float*>(alpha));
    auto beta_fp  = *(static_cast<const float*>(beta));

    if (num_batch == 1)
    {
        size_t workgroups = std::min(grid_size, 64 * 40 * 8);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        network_config = "sfmfwd-n" + std::to_string(num_batch) + "half" +
        std::to_string(static_cast<int>(usefp16)) + "float" +
        std::to_string(static_cast<int>(usefp32)) + "g" + std::to_string(vgd[0]) + "l" +
        std::to_string(vld[0]) + "dim" + std::to_string(spatial_dim) + "grid" +
        std::to_string(grid_size) + "wg" + std::to_string(workgroups) + "v" +
        std::to_string(vector_size) + "xpk" +
        std::to_string(static_cast<int>(xDesc.IsPacked())) + "ypk" +
        std::to_string(static_cast<int>(yDesc.IsPacked())) + "a" + std::to_string(alpha_fp) +
        "b" + std::to_string(beta_fp) + "algo" + std::to_string(static_cast<int>(algorithm)) +
        "mode" + std::to_string(static_cast<int>(mode));
    }
    else
    {
        int batch_size = 256 / num_batch;
        // num_channels each threads iterates over to cover all the channels
        int u_batch_size = (vector_size > batch_size) ? nextPow2(vector_size / batch_size) : 1;

        size_t workgroups =
            (grid_size % num_batch == 0) ? (grid_size / num_batch) : (grid_size / num_batch + 1);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        if((u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
            MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");

        //std::string algo_name = "SoftmaxForwardMultiBatch";
        network_config =
            "sfmfwd-n" + std::to_string(num_batch) + "half" +
            std::to_string(static_cast<int>(usefp16)) + "float" +
            std::to_string(static_cast<int>(usefp32)) + "g" + std::to_string(vgd[0]) + "l" +
            std::to_string(vld[0]) + "dim" + std::to_string(spatial_dim) + "grid" +
            std::to_string(grid_size) + "wg" + std::to_string(workgroups) + "v" +
            std::to_string(vector_size) + "ubatch" + std::to_string(u_batch_size) + "batch" +
            std::to_string(batch_size) + "xpk" +
            std::to_string(static_cast<int>(xDesc.IsPacked())) + "ypk" +
            std::to_string(static_cast<int>(yDesc.IsPacked())) + "a" + std::to_string(alpha_fp) +
            "b" + std::to_string(beta_fp) + "algo" + std::to_string(static_cast<int>(algorithm)) +
            "mode" + std::to_string(static_cast<int>(mode));
    }

    return NetworkConfig{network_config};
}

} // namespace softmax

} // namespace miopen
 