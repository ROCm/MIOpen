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

miopenStatus_t SoftmaxForward(const Handle& handle,
                              const void* alpha,
                              const void* beta,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              miopenSoftmaxAlgorithm_t algorithm,
                              miopenSoftmaxMode_t mode,
                              int x_offset,
                              int y_offset)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    if(xDesc.GetType() != yDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
    }

    if(xDesc.GetLengths() != yDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(yDesc.GetLengths());

    int in_nstr, in_cstr, in_hstr;
    std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = tien<4>(xDesc.GetStrides());

    int out_nstr, out_cstr, out_hstr;
    std::tie(out_nstr, out_cstr, out_hstr, std::ignore) = tien<4>(yDesc.GetStrides());

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

    // See Kernels/MIOpenSoftmax.cl for description
    if(num_batch == 1)
    { // CSR-Vector like approach

        // Control the max. number of workgroups launched so that we do not
        // start getting workgroup scheduling overheads
        size_t workgroups = std::min(grid_size, 64 * 40 * 8);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        std::string algo_name = "SoftmaxForwardOneBatch";
        std::string network_config =
            "sfmfwd-n" + std::to_string(num_batch) + "half" +
            std::to_string(static_cast<int>(usefp16)) + "float" +
            std::to_string(static_cast<int>(usefp32)) + "g" + std::to_string(vgd[0]) + "l" +
            std::to_string(vld[0]) + "dim" + std::to_string(spatial_dim) + "grid" +
            std::to_string(grid_size) + "wg" + std::to_string(workgroups) + "v" +
            std::to_string(vector_size) + "xpk" +
            std::to_string(static_cast<int>(xDesc.IsPacked())) + "ypk" +
            std::to_string(static_cast<int>(yDesc.IsPacked())) + "a" + std::to_string(alpha_fp) +
            "b" + std::to_string(beta_fp) + "algo" + std::to_string(static_cast<int>(algorithm)) +
            "mode" + std::to_string(static_cast<int>(mode));

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            kernels.front()(x,
                            y,
                            vector_size,
                            grid_size,
                            spatial_dim,
                            h,
                            w,
                            in_nstr,
                            in_cstr,
                            in_hstr,
                            out_nstr,
                            out_cstr,
                            out_hstr,
                            x_offset,
                            y_offset,
                            alpha_fp,
                            beta_fp);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxForward";

            // compile parameters
            std::string parms = "-DNUM_BATCH=" + std::to_string(num_batch) +
                                " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(usefp16)) +
                                " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(usefp32));

            if(algorithm == MIOPEN_SOFTMAX_LOG)
                parms += " -DUSE_SOFTMAX_LOG=1";
            else if(algorithm == MIOPEN_SOFTMAX_FAST)
                parms += " -DUSE_SOFTMAX_FAST=1";
            else
                parms += " -DUSE_SOFTMAX_ACCURATE=1";

            if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
                parms += " -DUSE_SOFTMAX_MODE_INSTANCE=1";
            else
                parms += " -DUSE_SOFTMAX_MODE_CHANNEL=1";

            parms += " -DRUN_FORWARD=1";
            parms += " -DIS_INPUT_PACKED=" + std::to_string(static_cast<int>(xDesc.IsPacked())) +
                     " -DIS_OUTPUT_PACKED=" + std::to_string(static_cast<int>(yDesc.IsPacked()));

            if(!float_equal(alpha_fp, 1.0))
                parms += " -DUSE_ALPHA=1";

            if(!float_equal(beta_fp, 0))
                parms += " -DUSE_BETA=1";

            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                y,
                vector_size,
                grid_size,
                spatial_dim,
                h,
                w,
                in_nstr,
                in_cstr,
                in_hstr,
                out_nstr,
                out_cstr,
                out_hstr,
                x_offset,
                y_offset,
                alpha_fp,
                beta_fp);
        }
    }
    else
    { // CSR-Stream like approach

        // num_threads iterating over channels for one spatial_dim
        int batch_size = 256 / num_batch;
        // num_channels each threads iterates over to cover all the channels
        int u_batch_size = (vector_size > batch_size) ? nextPow2(vector_size / batch_size) : 1;

        size_t workgroups =
            (grid_size % num_batch == 0) ? (grid_size / num_batch) : (grid_size / num_batch + 1);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        if((u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
            MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");

        std::string algo_name = "SoftmaxForwardMultiBatch";
        std::string network_config =
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

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            kernels.front()(x,
                            y,
                            vector_size,
                            grid_size,
                            spatial_dim,
                            h,
                            w,
                            in_nstr,
                            in_cstr,
                            in_hstr,
                            out_nstr,
                            out_cstr,
                            out_hstr,
                            x_offset,
                            y_offset,
                            alpha_fp,
                            beta_fp);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxForward";
            std::string parms        = "-DNUM_BATCH=" + std::to_string(num_batch) +
                                " -DBATCH_SIZE=" + std::to_string(batch_size) +
                                " -DU_BATCH_SIZE=" + std::to_string(u_batch_size) +
                                " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(usefp16)) +
                                " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(usefp32));

            if(algorithm == MIOPEN_SOFTMAX_LOG)
                parms += " -DUSE_SOFTMAX_LOG=1";
            else if(algorithm == MIOPEN_SOFTMAX_FAST)
                parms += " -DUSE_SOFTMAX_FAST=1";
            else
                parms += " -DUSE_SOFTMAX_ACCURATE=1";

            if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
                parms += " -DUSE_SOFTMAX_MODE_INSTANCE=1";
            else
                parms += " -DUSE_SOFTMAX_MODE_CHANNEL=1";

            parms += " -DRUN_FORWARD=1";
            parms += " -DIS_INPUT_PACKED=" + std::to_string(static_cast<int>(xDesc.IsPacked())) +
                     " -DIS_OUTPUT_PACKED=" + std::to_string(static_cast<int>(yDesc.IsPacked()));

            if(!float_equal(alpha_fp, 1.0))
                parms += " -DUSE_ALPHA=1";

            if(!float_equal(beta_fp, 0))
                parms += " -DUSE_BETA=1";

            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                y,
                vector_size,
                grid_size,
                spatial_dim,
                h,
                w,
                in_nstr,
                in_cstr,
                in_hstr,
                out_nstr,
                out_cstr,
                out_hstr,
                x_offset,
                y_offset,
                alpha_fp,
                beta_fp);
        }
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxBackward(const Handle& handle,
                               const void* alpha,
                               const TensorDescriptor& yDesc,
                               ConstData_t y,
                               const TensorDescriptor& dyDesc,
                               ConstData_t dy,
                               const void* beta,
                               const TensorDescriptor& dxDesc,
                               Data_t dx,
                               miopenSoftmaxAlgorithm_t algorithm,
                               miopenSoftmaxMode_t mode,
                               int y_offset,
                               int dy_offset,
                               int dx_offset)
{
    if(dx == nullptr || y == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    if(yDesc != dyDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(dxDesc.GetType() != dyDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
    }

    if(dxDesc.GetLengths() != dyDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, yDesc, y);
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(dxDesc.GetLengths());

    int din_nstr, din_cstr, din_hstr;
    std::tie(din_nstr, din_cstr, din_hstr, std::ignore) = tien<4>(dxDesc.GetStrides());

    int dout_nstr, dout_cstr, dout_hstr;
    std::tie(dout_nstr, dout_cstr, dout_hstr, std::ignore) = tien<4>(dyDesc.GetStrides());

    int out_nstr, out_cstr, out_hstr;
    std::tie(out_nstr, out_cstr, out_hstr, std::ignore) = tien<4>(yDesc.GetStrides());

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

    // See Kernels/MIOpenSoftmax.cl for description
    if(num_batch == 1)
    { // CSR-Vector like approach

        // Control the max. number of workgroups launched so that we do not
        // start getting workgroup scheduling overheads
        size_t workgroups = std::min(grid_size, 64 * 40 * 8);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        std::string algo_name = "SoftmaxBackwardOneBatch";
        std::string network_config =
            "sfmbwd-n" + std::to_string(num_batch) + "half" +
            std::to_string(static_cast<int>(usefp16)) + "float" +
            std::to_string(static_cast<int>(usefp32)) + "g" + std::to_string(vgd[0]) + "l" +
            std::to_string(vld[0]) + "dim" + std::to_string(spatial_dim) + "grid" +
            std::to_string(grid_size) + "wg" + std::to_string(workgroups) + "v" +
            std::to_string(vector_size) + "ypk" +
            std::to_string(static_cast<int>(yDesc.IsPacked())) + "dypk" +
            std::to_string(static_cast<int>(dyDesc.IsPacked())) + "dxpk" +
            std::to_string(static_cast<int>(dxDesc.IsPacked())) + "a" + std::to_string(alpha_fp) +
            "b" + std::to_string(beta_fp) + "algo" + std::to_string(static_cast<int>(algorithm)) +
            "mode" + std::to_string(static_cast<int>(mode));

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            kernels.front()(y,
                            dy,
                            dx,
                            vector_size,
                            grid_size,
                            spatial_dim,
                            h,
                            w,
                            out_nstr,
                            out_cstr,
                            out_hstr,
                            dout_nstr,
                            dout_cstr,
                            dout_hstr,
                            din_nstr,
                            din_cstr,
                            din_hstr,
                            y_offset,
                            dy_offset,
                            dx_offset,
                            alpha_fp,
                            beta_fp);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxBackward";
            std::string parms        = "-DNUM_BATCH=" + std::to_string(num_batch) +
                                " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(usefp16)) +
                                " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(usefp32));

            if(algorithm == MIOPEN_SOFTMAX_LOG)
                parms += " -DUSE_SOFTMAX_LOG=1";
            else if(algorithm == MIOPEN_SOFTMAX_FAST)
                parms += " -DUSE_SOFTMAX_FAST=1";
            else
                parms += " -DUSE_SOFTMAX_ACCURATE=1";

            if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
                parms += " -DUSE_SOFTMAX_MODE_INSTANCE=1";
            else
                parms += " -DUSE_SOFTMAX_MODE_CHANNEL=1";

            parms += " -DRUN_FORWARD=0";
            parms += " -DIS_OUTPUT_PACKED=" + std::to_string(static_cast<int>(yDesc.IsPacked())) +
                     " -DIS_DOUTPUT_PACKED=" + std::to_string(static_cast<int>(dyDesc.IsPacked())) +
                     " -DIS_DINPUT_PACKED=" + std::to_string(static_cast<int>(dxDesc.IsPacked()));

            if(!float_equal(alpha_fp, 1.0))
                parms += " -DUSE_ALPHA=1";

            if(!float_equal(beta_fp, 0))
                parms += " -DUSE_BETA=1";

            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                y,
                dy,
                dx,
                vector_size,
                grid_size,
                spatial_dim,
                h,
                w,
                out_nstr,
                out_cstr,
                out_hstr,
                dout_nstr,
                dout_cstr,
                dout_hstr,
                din_nstr,
                din_cstr,
                din_hstr,
                y_offset,
                dy_offset,
                dx_offset,
                alpha_fp,
                beta_fp);
        }
    }
    else
    { // CSR-Stream like approach
        int batch_size   = 256 / num_batch;
        int u_batch_size = (vector_size > batch_size) ? nextPow2(vector_size / batch_size) : 1;
        size_t workgroups =
            (grid_size % num_batch == 0) ? (grid_size / num_batch) : (grid_size / num_batch + 1);
        const std::vector<size_t> vgd{workgroups * vld[0], 1, 1};

        if((2 * u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
            MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");

        std::string algo_name = "SoftmaxBackwardMultiBatch";
        std::string network_config =
            "sfmbwd-n" + std::to_string(num_batch) + "half" +
            std::to_string(static_cast<int>(usefp16)) + "float" +
            std::to_string(static_cast<int>(usefp32)) + "g" + std::to_string(vgd[0]) + "l" +
            std::to_string(vld[0]) + "dim" + std::to_string(spatial_dim) + "grid" +
            std::to_string(grid_size) + "wg" + std::to_string(workgroups) + "v" +
            std::to_string(vector_size) + "ubatch" + std::to_string(u_batch_size) + "batch" +
            std::to_string(batch_size) + "ypk" +
            std::to_string(static_cast<int>(yDesc.IsPacked())) + "dypk" +
            std::to_string(static_cast<int>(dyDesc.IsPacked())) + "dxpk" +
            std::to_string(static_cast<int>(dxDesc.IsPacked())) + "a" + std::to_string(alpha_fp) +
            "b" + std::to_string(beta_fp) + "algo" + std::to_string(static_cast<int>(algorithm)) +
            "mode" + std::to_string(static_cast<int>(mode));

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            kernels.front()(y,
                            dy,
                            dx,
                            vector_size,
                            grid_size,
                            spatial_dim,
                            h,
                            w,
                            out_nstr,
                            out_cstr,
                            out_hstr,
                            dout_nstr,
                            dout_cstr,
                            dout_hstr,
                            din_nstr,
                            din_cstr,
                            din_hstr,
                            y_offset,
                            dy_offset,
                            dx_offset,
                            alpha_fp,
                            beta_fp);
        }
        else
        {
            std::string program_name = "MIOpenSoftmax.cl";
            std::string kernel_name  = "SoftmaxBackward";
            std::string parms        = "-DNUM_BATCH=" + std::to_string(num_batch) +
                                " -DBATCH_SIZE=" + std::to_string(batch_size) +
                                " -DU_BATCH_SIZE=" + std::to_string(u_batch_size) +
                                " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(usefp16)) +
                                " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(usefp32));

            if(algorithm == MIOPEN_SOFTMAX_LOG)
                parms += " -DUSE_SOFTMAX_LOG=1";
            else if(algorithm == MIOPEN_SOFTMAX_FAST)
                parms += " -DUSE_SOFTMAX_FAST=1";
            else
                parms += " -DUSE_SOFTMAX_ACCURATE=1";

            if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
                parms += " -DUSE_SOFTMAX_MODE_INSTANCE=1";
            else
                parms += " -DUSE_SOFTMAX_MODE_CHANNEL=1";

            parms += " -DRUN_FORWARD=0";
            parms += " -DIS_OUTPUT_PACKED=" + std::to_string(static_cast<int>(yDesc.IsPacked())) +
                     " -DIS_DOUTPUT_PACKED=" + std::to_string(static_cast<int>(dyDesc.IsPacked())) +
                     " -DIS_DINPUT_PACKED=" + std::to_string(static_cast<int>(dxDesc.IsPacked()));

            if(!float_equal(alpha_fp, 1.0))
                parms += " -DUSE_ALPHA=1";

            if(!float_equal(beta_fp, 0))
                parms += " -DUSE_BETA=1";

            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                y,
                dy,
                dx,
                vector_size,
                grid_size,
                spatial_dim,
                h,
                w,
                out_nstr,
                out_cstr,
                out_hstr,
                dout_nstr,
                dout_cstr,
                dout_hstr,
                din_nstr,
                din_cstr,
                din_hstr,
                y_offset,
                dy_offset,
                dx_offset,
                alpha_fp,
                beta_fp);
        }
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }

    return miopenStatusSuccess;
}

} // namespace miopen
