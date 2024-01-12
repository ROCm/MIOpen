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

int nextPow2(int v)
{

    if(v == 1)
    {
        return (v << 1);
    }
    else
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
}

void getParams(const TensorDescriptor& in_yDesc, 
                miopenSoftmaxMode_t in_mode, 
                int& out_n, 
                int& out_c, 
                int& out_h, 
                int& out_w, 
                int& out_grid_size, 
                int& out_spatial_dim,
                int& out_vector_size,
                int& out_num_batch,
                bool& out_usefp16,
                bool& out_usefp32,
                std::vector<size_t>& out_vld,
                std::vector<size_t>& out_vgd,
                size_t& out_workgroups,
                int& out_batch_size,
                int& out_u_batch_size)
{
    std::tie(out_n, out_c, out_h, out_w) = tien<4>(in_yDesc.GetLengths());

    // using workgroup size of 256 by default
    out_grid_size   = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? out_n : out_n * out_h * out_w;
    out_spatial_dim = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? 1 : out_h * out_w;
    out_vector_size = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? out_c * out_h * out_w : out_c;
    // num_spatial_dims or pixels each workgroup can compute
    out_num_batch = out_vector_size < 256 ? nextPow2(256 / out_vector_size) : 1;

    out_vld = {256, 1, 1};

    out_usefp16 = false;
    out_usefp32 = true;
    if (in_yDesc.GetType() == miopenHalf)
    {
        out_usefp16 = true;
        out_usefp32 = false;
    }

    if (out_num_batch == 1)
    {
        out_workgroups = std::min(out_grid_size, 64 * 40 * 8);
        out_vgd = {out_workgroups * out_vld[0], 1, 1};

        out_batch_size = 0;
        out_u_batch_size = 0;
    }
    else
    {
        out_batch_size = 256 / out_num_batch;
        // num_channels each threads iterates over to cover all the channels
        out_u_batch_size = (out_vector_size > out_batch_size) ? nextPow2(out_vector_size / out_batch_size) : 1;

        out_workgroups = (out_grid_size % out_num_batch == 0) ? (out_grid_size / out_num_batch) : (out_grid_size / out_num_batch + 1);
        out_vgd = {out_workgroups * out_vld[0], 1, 1};
    }
}


NetworkConfig ProblemDescription::MakeNetworkConfig() const
{  
    std::string network_config;

    int n, c, h, w;
    int grid_size, spatial_dim, vector_size, num_batch;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool usefp16, usefp32;

    size_t workgroups;
    int batch_size;
    int u_batch_size;

    getParams(yDesc, mode, n, c, h, w, grid_size, spatial_dim, vector_size, num_batch, usefp16, usefp32, vld, vgd, workgroups, batch_size, u_batch_size);

    auto alpha_fp = *(static_cast<const float*>(alpha));
    auto beta_fp  = *(static_cast<const float*>(beta));

    if (num_batch != 1)
    {
        if((u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
            MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");
    }

    network_config = "sfmfwd-n" + std::to_string(num_batch) + "half" +
    std::to_string(static_cast<int>(usefp16)) + "float" +
    std::to_string(static_cast<int>(usefp32)) + "g" + std::to_string(vgd[0]) + "l" +
    std::to_string(vld[0]) + "dim" + std::to_string(spatial_dim) + "grid" +
    std::to_string(grid_size) + "wg" + std::to_string(workgroups) + "v";

    if (num_batch != 1)
    {
        network_config += std::to_string(vector_size) + "ubatch" + std::to_string(u_batch_size) + "batch";
    }

    network_config += std::to_string(vector_size) + "xpk" +
    std::to_string(static_cast<int>(xDesc.IsPacked())) + "ypk" +
    std::to_string(static_cast<int>(yDesc.IsPacked())) + "a" + std::to_string(alpha_fp) +
    "b" + std::to_string(beta_fp) + "algo" + std::to_string(static_cast<int>(algorithm)) +
    "mode" + std::to_string(static_cast<int>(mode));

    return NetworkConfig{network_config};
}

} // namespace softmax

} // namespace miopen
