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
#include "test.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <miopen/softmax.hpp>
#include <miopen/miopen.h>
#include <miopen/solution.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <miopen/kernel_build_params.hpp>
#include <miopen/float_equal.hpp>

using namespace miopen;

constexpr int nextPow2(int v)
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

void getParams(const TensorDescriptor& in_desc,
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
    std::tie(out_n, out_c, out_h, out_w) = tien<4>(in_desc.GetLengths());

    // using workgroup size of 256 by default
    out_grid_size   = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? out_n : out_n * out_h * out_w;
    out_spatial_dim = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? 1 : out_h * out_w;
    out_vector_size = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? out_c * out_h * out_w : out_c;
    // num_spatial_dims or pixels each workgroup can compute

    /// \todo Magic numbers
    out_num_batch = out_vector_size < 256 ? nextPow2(256 / out_vector_size) : 1;

    out_vld = {256, 1, 1};

    out_usefp16 = false;
    out_usefp32 = true;
    if(in_desc.GetType() == miopenHalf)
    {
        out_usefp16 = true;
        out_usefp32 = false;
    }

    if(out_num_batch == 1)
    {
        out_workgroups = std::min(out_grid_size, 64 * 40 * 8);
        out_vgd        = {out_workgroups * out_vld[0], 1, 1};

        out_batch_size   = 0;
        out_u_batch_size = 0;
    }
    else
    {
        out_batch_size = 256 / out_num_batch;
        // num_channels each threads iterates over to cover all the channels
        out_u_batch_size =
            (out_vector_size > out_batch_size) ? nextPow2(out_vector_size / out_batch_size) : 1;

        out_workgroups = (out_grid_size % out_num_batch == 0) ? (out_grid_size / out_num_batch)
                                                              : (out_grid_size / out_num_batch + 1);
        out_vgd        = {out_workgroups * out_vld[0], 1, 1};
    }
}

void SoftmaxForwardGPU(miopen::Handle& handle,
                       const void* alpha,
                       const miopen::TensorDescriptor& xDesc,
                       const void* x,
                       const void* beta,
                       const miopen::TensorDescriptor& yDesc,
                       void* y,
                       miopenSoftmaxAlgorithm_t algorithm,
                       miopenSoftmaxMode_t mode,
                       int x_offset,
                       int y_offset,
                       bool useHIP = false)

{
    int n, c, h, w;
    // using workgroup size of 256 by default
    int grid_size, spatial_dim, vector_size, num_batch;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool usefp16, usefp32;

    size_t workgroups;
    int batch_size;
    int u_batch_size;

    float alpha_val = *static_cast<const float*>(alpha);
    float beta_val  = *static_cast<const float*>(beta);

    getParams(yDesc,
              mode,
              n,
              c,
              h,
              w,
              grid_size,
              spatial_dim,
              vector_size,
              num_batch,
              usefp16,
              usefp32,
              vld,
              vgd,
              workgroups,
              batch_size,
              u_batch_size);

    if(num_batch > 1)
    {

        /// \todo Magic numbers
        if((u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
            MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");
    }

    miopen::KernelBuildParameters build_params = KernelBuildParameters{{"NUM_BATCH", num_batch}};

    if(num_batch > 1)
    {
        build_params.Define("BATCH_SIZE", batch_size);
        build_params.Define("U_BATCH_SIZE", u_batch_size);
    }

    build_params.Define("MIOPEN_USE_FP16", static_cast<int>(usefp16));
    build_params.Define("MIOPEN_USE_FP32", static_cast<int>(usefp32));

    if(algorithm == MIOPEN_SOFTMAX_LOG)
        build_params.Define("USE_SOFTMAX_LOG", 1);
    else if(algorithm == MIOPEN_SOFTMAX_FAST)
        build_params.Define("USE_SOFTMAX_FAST", 1);
    else
        build_params.Define("USE_SOFTMAX_ACCURATE", 1);

    if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        build_params.Define("USE_SOFTMAX_MODE_INSTANCE", 1);
    else
        build_params.Define("USE_SOFTMAX_MODE_CHANNEL", 1);

    build_params.Define("RUN_FORWARD", 1);

    build_params.Define("IS_INPUT_PACKED", static_cast<int>(xDesc.IsPacked()));
    build_params.Define("IS_OUTPUT_PACKED", static_cast<int>(yDesc.IsPacked()));

    build_params.Define("IS_DINPUT_PACKED", 1);
    build_params.Define("IS_DOUTPUT_PACKED", 1);

    if(!float_equal(alpha_val, 1.0))
        build_params.Define("USE_ALPHA", 1);

    if(!float_equal(beta_val, 0))
        build_params.Define("USE_BETA", 1);

    std::string comp_options;
    std::string kernel_file;
    std::string kernel_name;

    if(useHIP)
    {
        comp_options = build_params.GenerateFor(kbp::HIP{});
        kernel_file  = "MIOpenSoftmaxHIP.cpp";
        kernel_name  = "SoftmaxForwardHIP";
    }
    else
    {
        comp_options = build_params.GenerateFor(kbp::OpenCL{});
        kernel_file  = "MIOpenSoftmax.cl";
        kernel_name  = "SoftmaxForward";
    }

    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;

    for(unsigned int i = 0; i < 2; ++i)
    {
        l_wk.push_back(vld[i]);
        g_wk.push_back(vgd[i]);
    }

    int in_nstr, in_cstr, in_hstr;
    std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = tien<4>(xDesc.GetStrides());

    int out_nstr, out_cstr, out_hstr;
    std::tie(out_nstr, out_cstr, out_hstr, std::ignore) = tien<4>(yDesc.GetStrides());

    std::string network_config = "softmax-ocl";

    handle.AddKernel(kernel_name, network_config, kernel_file, kernel_name, vld, vgd, comp_options)(
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
        alpha,
        beta);
}

void SoftmaxBackwardGPU(miopen::Handle& handle,
                        const void* alpha,
                        const miopen::TensorDescriptor& yDesc,
                        const void* y,
                        const miopen::TensorDescriptor& dyDesc,
                        const void* dy,
                        const void* beta,
                        const miopen::TensorDescriptor& dxDesc,
                        void* dx,
                        miopenSoftmaxAlgorithm_t algorithm,
                        miopenSoftmaxMode_t mode,
                        int y_offset,
                        int dy_offset,
                        int dx_offset,
                        bool useHIP = false)

{
    int n, c, h, w;
    // using workgroup size of 256 by default
    int grid_size, spatial_dim, vector_size, num_batch;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool usefp16, usefp32;

    size_t workgroups;
    int batch_size;
    int u_batch_size;

    float alpha_val = *static_cast<const float*>(alpha);
    float beta_val  = *static_cast<const float*>(beta);

    getParams(yDesc,
              mode,
              n,
              c,
              h,
              w,
              grid_size,
              spatial_dim,
              vector_size,
              num_batch,
              usefp16,
              usefp32,
              vld,
              vgd,
              workgroups,
              batch_size,
              u_batch_size);

    if(num_batch > 1)
    {

        /// \todo Magic numbers
        if((2 * u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
            MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");
    }

    miopen::KernelBuildParameters build_params = KernelBuildParameters{{"NUM_BATCH", num_batch}};

    if(num_batch > 1)
    {
        build_params.Define("BATCH_SIZE", batch_size);
        build_params.Define("U_BATCH_SIZE", u_batch_size);
    }

    build_params.Define("MIOPEN_USE_FP16", static_cast<int>(usefp16));
    build_params.Define("MIOPEN_USE_FP32", static_cast<int>(usefp32));

    if(algorithm == MIOPEN_SOFTMAX_LOG)
        build_params.Define("USE_SOFTMAX_LOG", 1);
    else if(algorithm == MIOPEN_SOFTMAX_FAST)
        build_params.Define("USE_SOFTMAX_FAST", 1);
    else
        build_params.Define("USE_SOFTMAX_ACCURATE", 1);

    if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        build_params.Define("USE_SOFTMAX_MODE_INSTANCE", 1);
    else
        build_params.Define("USE_SOFTMAX_MODE_CHANNEL", 1);

    build_params.Define("RUN_FORWARD", 0);

    build_params.Define("IS_OUTPUT_PACKED", static_cast<int>(yDesc.IsPacked()));
    build_params.Define("IS_DOUTPUT_PACKED", static_cast<int>(dyDesc.IsPacked()));
    build_params.Define("IS_DINPUT_PACKED", static_cast<int>(dxDesc.IsPacked()));

    build_params.Define("IS_INPUT_PACKED", 0);

    if(!float_equal(alpha_val, 1.0))
        build_params.Define("USE_ALPHA", 1);

    if(!float_equal(beta_val, 0))
        build_params.Define("USE_BETA", 1);

    std::string comp_options;
    std::string kernel_file;
    std::string kernel_name;

    if(useHIP)
    {
        comp_options = build_params.GenerateFor(kbp::HIP{});
        kernel_file  = "MIOpenSoftmaxHIP.cpp";
        kernel_name  = "SoftmaxBackwardHIP";
    }
    else
    {
        comp_options = build_params.GenerateFor(kbp::OpenCL{});
        kernel_file  = "MIOpenSoftmax.cl";
        kernel_name  = "SoftmaxBackward";
    }

    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;

    for(unsigned int i = 0; i < 2; ++i)
    {
        l_wk.push_back(vld[i]);
        g_wk.push_back(vgd[i]);
    }

    int din_nstr, din_cstr, din_hstr;
    std::tie(din_nstr, din_cstr, din_hstr, std::ignore) = tien<4>(dxDesc.GetStrides());

    int dout_nstr, dout_cstr, dout_hstr;
    std::tie(dout_nstr, dout_cstr, dout_hstr, std::ignore) = tien<4>(dyDesc.GetStrides());

    int out_nstr, out_cstr, out_hstr;
    std::tie(out_nstr, out_cstr, out_hstr, std::ignore) = tien<4>(yDesc.GetStrides());

    std::string network_config = "softmax-backward-ocl";

    handle.AddKernel(kernel_name, network_config, kernel_file, kernel_name, vld, vgd, comp_options)(
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
        alpha,
        beta);
}

class SoftmaxTest
{
public:
    SoftmaxTest(bool forward) : isForward(forward) { Initialize(); }

    void TestRunSolutionsForward(Handle& handle)
    {
        auto in_gpu  = handle.Write(xTensor.data);
        auto out_gpu = handle.Write(yTensor.data);

        auto alpha     = softmax_descriptor.GetAlpha();
        auto beta      = softmax_descriptor.GetBeta();
        auto mode      = softmax_descriptor.GetMode();
        auto algorithm = softmax_descriptor.GetAlgorithm();

        tensor<float> yTensorRef = tensor<float>{test_n, test_c, test_h, test_w};

        auto out_gpu_ref = handle.Write(yTensorRef.data);

        // Run softmax using OpenCL
        SoftmaxForwardGPU(handle,
                          &alpha,
                          xTensor.desc,
                          in_gpu.get(),
                          &beta,
                          yTensor.desc,
                          out_gpu_ref.get(),
                          algorithm,
                          mode,
                          0,
                          0,
                          false);

        // Run softmax using HIP
        SoftmaxForwardGPU(handle,
                          &alpha,
                          xTensor.desc,
                          in_gpu.get(),
                          &beta,
                          yTensor.desc,
                          out_gpu.get(),
                          algorithm,
                          mode,
                          0,
                          0,
                          true);

        yTensor.data    = handle.Read<float>(out_gpu, yTensor.data.size());
        yTensorRef.data = handle.Read<float>(out_gpu_ref, yTensorRef.data.size());

        double error           = miopen::rms_range(yTensorRef.data, yTensorRef.data);
        const double tolerance = 1e-3;

        EXPECT_TRUE(std::isfinite(error) && error <= tolerance)
            << "Outputs do not match each other. Error:" << error;
    }

    void TestRunSolutionsBackward(Handle& handle)
    {

        auto in1_gpu = handle.Write(yTensor.data);
        auto in2_gpu = handle.Write(dyTensor.data);
        auto out_gpu = handle.Write(dxTensor.data);

        auto alpha     = softmax_descriptor.GetAlpha();
        auto beta      = softmax_descriptor.GetBeta();
        auto mode      = softmax_descriptor.GetMode();
        auto algorithm = softmax_descriptor.GetAlgorithm();

        tensor<float> dxTensorRef = tensor<float>{test_n, test_c, test_h, test_w};

        auto out_gpu_ref = handle.Write(dxTensorRef.data);

        // Run softmax using OpenCL
        SoftmaxBackwardGPU(handle,
                           &alpha,
                           yTensor.desc,
                           in1_gpu.get(),
                           dyTensor.desc,
                           in2_gpu.get(),
                           &beta,
                           dxTensor.desc,
                           out_gpu_ref.get(),
                           algorithm,
                           mode,
                           0,
                           0,
                           0,
                           false);

        // Run softmax using HIP
        SoftmaxBackwardGPU(handle,
                           &alpha,
                           yTensor.desc,
                           in1_gpu.get(),
                           dyTensor.desc,
                           in2_gpu.get(),
                           &beta,
                           dxTensor.desc,
                           out_gpu.get(),
                           algorithm,
                           mode,
                           0,
                           0,
                           0,
                           true);

        yTensor.data     = handle.Read<float>(out_gpu, yTensor.data.size());
        dxTensorRef.data = handle.Read<float>(out_gpu_ref, dxTensorRef.data.size());

        double error           = miopen::rms_range(dxTensorRef.data, yTensor.data);
        const double tolerance = 1e-3;

        EXPECT_TRUE(std::isfinite(error) && error <= tolerance)
            << "Outputs do not match each other. Error:" << error;
    }

private:
    void Initialize()
    {
        softmax_descriptor.SetParams(
            1.0f, 0.0f, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);

        if(isForward)
        {
            xTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            yTensor = tensor<float>{test_n, test_c, test_h, test_w};
        }
        else
        {
            yTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            dyTensor =
                tensor<float>{test_n, test_c, test_h, test_w}.generate(tensor_elem_gen_integer{17});
            dxTensor = tensor<float>{test_n, test_c, test_h, test_w};
        }
    }

private:
    tensor<float> xTensor;
    tensor<float> yTensor;

    tensor<float> dxTensor;
    tensor<float> dyTensor;

    SoftmaxDescriptor softmax_descriptor;

    bool isForward;

    const unsigned int test_n = 100;
    const unsigned int test_c = 3;
    const unsigned int test_h = 32;
    const unsigned int test_w = 32;
};

TEST(TestSoftmax, softmaxForward)
{
    Handle& handle = get_handle();

    SoftmaxTest test(true);

    test.TestRunSolutionsForward(handle);
}

TEST(TestSoftmax, softmaxBackward)
{
    Handle& handle = get_handle();

    SoftmaxTest test(false);

    test.TestRunSolutionsBackward(handle);
}
