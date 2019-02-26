/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/kernel_cache.hpp>
#include <miopen/softmax.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

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

miopenStatus_t SoftmaxForward(
    Handle& handle, const void* alpha, const void* beta, const TensorDescriptor& yDesc, Data_t y)
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(yDesc.GetLengths());
    // using workgroup size of 256 by default
    int grid_size   = n * h * w;
    int spatial_dim = h * w;
    // num_spatial_dims or pixels each workgroup can compute
    int num_batch = c < 256 ? nextPow2(256 / c) : 1;

    const std::vector<size_t> vld{256, 1, 1};

    bool usefp16 = false;
    bool usefp32 = true;
    if(yDesc.GetType() == miopenHalf)
    {
        usefp16 = true;
        usefp32 = false;
    }

    // See Kernels/MIOpenSoftmax.cl for description
    if(num_batch == 1)
    { // CSR-Vector like approach

        // Control the max. number of workgroups launched so that we do not
        // start getting workgroup scheduling overheads
        size_t workgroups = std::min(grid_size, 64 * 40 * 8);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        std::string algo_name = "SoftmaxForwardOneBatch";
        std::string network_config =
            std::to_string(num_batch) + std::to_string(static_cast<int>(usefp16)) +
            std::to_string(static_cast<int>(usefp32)) + std::to_string(vgd[0]) +
            std::to_string(vld[0]) + std::to_string(spatial_dim) + std::to_string(grid_size) +
            std::to_string(workgroups) + std::to_string(c);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            // auto kernel = kernels.front();
            kernels.front()(y, c, grid_size, spatial_dim);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxForward";

            // compile parameters
            std::string parms = "-DNUM_BATCH=" + std::to_string(num_batch) + " -DMIOPEN_USE_FP16=" +
                                std::to_string(static_cast<int>(usefp16)) + " -DMIOPEN_USE_FP32=" +
                                std::to_string(static_cast<int>(usefp32));
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                y, c, grid_size, spatial_dim);
        }
    }
    else
    { // CSR-Stream like approach

        // num_threads iterating over channels for one spatial_dim
        int batch_size = 256 / num_batch;
        // num_channels each threads iterates over to cover all the channels
        int u_batch_size = (c > batch_size) ? nextPow2(c / batch_size) : 1;

        // size_t workgroups =
        //    grid_size % num_batch == 0 ? grid_size / num_batch : grid_size / num_batch + 1;
        size_t workgroups =
            (grid_size % num_batch == 0) ? (grid_size / num_batch) : (grid_size / num_batch + 1);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        std::string algo_name = "SoftmaxForwardMultiBatch";
        std::string network_config =
            std::to_string(num_batch) + std::to_string(static_cast<int>(usefp16)) +
            std::to_string(static_cast<int>(usefp32)) + std::to_string(vgd[0]) +
            std::to_string(vld[0]) + std::to_string(spatial_dim) + std::to_string(grid_size) +
            std::to_string(workgroups) + std::to_string(c) + std::to_string(u_batch_size) +
            std::to_string(batch_size);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            kernels.front()(y, c, grid_size, spatial_dim);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxForward";
            std::string parms = "-DNUM_BATCH=" + std::to_string(num_batch) + " -DBATCH_SIZE=" +
                                std::to_string(batch_size) + " -DU_BATCH_SIZE=" +
                                std::to_string(u_batch_size) + " -DMIOPEN_USE_FP16=" +
                                std::to_string(static_cast<int>(usefp16)) + " -DMIOPEN_USE_FP32=" +
                                std::to_string(static_cast<int>(usefp32));

            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                y, c, grid_size, spatial_dim);
        }
    }
    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxBackward(Handle& handle,
                               const void* alpha,
                               const TensorDescriptor& yDesc,
                               ConstData_t y,
                               const void* beta,
                               const TensorDescriptor& dxDesc,
                               Data_t dx)
{
    if(yDesc != dxDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, yDesc, y);
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(dxDesc.GetLengths());

    // using workgroup size of 256 by default
    int grid_size   = n * h * w;
    int spatial_dim = h * w;
    int num_batch   = (c < 256) ? nextPow2(256 / c) : 1;

    const std::vector<size_t> vld{256, 1, 1};

    bool usefp16 = false;
    bool usefp32 = true;
    if(yDesc.GetType() == miopenHalf)
    {
        usefp16 = true;
        usefp32 = false;
    }

    // See Kernels/MIOpenSoftmax.cl for description
    if(num_batch == 1)
    { // CSR-Vector like approach

        // Control the max. number of workgroups launched so that we do not
        // start getting workgroup scheduling overheads
        size_t workgroups = std::min(grid_size, 64 * 40 * 8);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        std::string algo_name = "SoftmaxBackwardOneBatch";
        std::string network_config =
            std::to_string(num_batch) + std::to_string(static_cast<int>(usefp16)) +
            std::to_string(static_cast<int>(usefp32)) + std::to_string(vgd[0]) +
            std::to_string(vld[0]) + std::to_string(spatial_dim) + std::to_string(grid_size) +
            std::to_string(workgroups) + std::to_string(c);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            kernels.front()(y, dx, c, grid_size, spatial_dim);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxBackward";
            std::string parms = "-DNUM_BATCH=" + std::to_string(num_batch) + " -DMIOPEN_USE_FP16=" +
                                std::to_string(static_cast<int>(usefp16)) + " -DMIOPEN_USE_FP32=" +
                                std::to_string(static_cast<int>(usefp32));
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                y, dx, c, grid_size, spatial_dim);
        }
    }
    else
    { // CSR-Stream like approach
        int batch_size   = 256 / num_batch;
        int u_batch_size = (c > batch_size) ? nextPow2(c / batch_size) : 1;
        size_t workgroups =
            (grid_size % num_batch == 0) ? (grid_size / num_batch) : (grid_size / num_batch + 1);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        std::string algo_name = "SoftmaxBackwardMultiBatch";
        std::string network_config =
            std::to_string(num_batch) + std::to_string(static_cast<int>(usefp16)) +
            std::to_string(static_cast<int>(usefp32)) + std::to_string(vgd[0]) +
            std::to_string(vld[0]) + std::to_string(spatial_dim) + std::to_string(grid_size) +
            std::to_string(workgroups) + std::to_string(c) + std::to_string(u_batch_size) +
            std::to_string(batch_size);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            kernels.front()(y, dx, c, grid_size, spatial_dim);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxBackward";
            std::string parms = "-DNUM_BATCH=" + std::to_string(num_batch) + " -DBATCH_SIZE=" +
                                std::to_string(batch_size) + " -DU_BATCH_SIZE=" +
                                std::to_string(u_batch_size) + " -DMIOPEN_USE_FP16=" +
                                std::to_string(static_cast<int>(usefp16)) + " -DMIOPEN_USE_FP32=" +
                                std::to_string(static_cast<int>(usefp32));
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                y, dx, c, grid_size, spatial_dim);
        }
    }
    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }

    return miopenStatusSuccess;
}

} // namespace miopen
