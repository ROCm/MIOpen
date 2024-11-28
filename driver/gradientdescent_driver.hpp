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
#pragma once

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <vector>

template <typename Tgpu, typename Tcheck, typename Tseg>
int32_t mloGradientDescentRunHost(const miopenTensorDescriptor_t varInDesc,
                                  const miopenTensorDescriptor_t varOutDesc,
                                  const miopenTensorDescriptor_t alphaInDesc,
                                  const miopenTensorDescriptor_t deltaInDesc,
                                  const Tgpu* var_in,
                                  Tcheck* var_out,
                                  const Tgpu* alpha_in,
                                  const Tgpu* delta_in)
{
    auto var_in_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(varInDesc));
    auto var_out_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(varOutDesc));
    auto alpha_in_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(alphaInDesc));
    auto delta_in_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(deltaInDesc));
    uint64_t N       = miopen::deref(varInDesc).GetElementSize();

    par_ford(N)([&](uint64_t gid) {
        auto tensor_layout = tensor_layout_t<5>(var_in_tv, gid);
        double var   = static_cast<double>(var_in[var_in_tv.get_tensor_view_idx(tensor_layout)]);
        double alpha = static_cast<double>(alpha_in[alpha_in_tv.get_tensor_view_idx({0})]);
        double delta =
            static_cast<double>(delta_in[delta_in_tv.get_tensor_view_idx(tensor_layout)]);

        var -= alpha * delta;

        var_out[var_out_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tcheck>(var);
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref = Tgpu>
class GradientDescentDriver : public Driver
{
public:
    GradientDescentDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&varInDesc);
        miopenCreateTensorDescriptor(&varOutDesc);
        miopenCreateTensorDescriptor(&alphaInDesc);
        miopenCreateTensorDescriptor(&deltaInDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~GradientDescentDriver() override
    {
        miopenDestroyTensorDescriptor(varInDesc);
        miopenDestroyTensorDescriptor(varOutDesc);
        miopenDestroyTensorDescriptor(alphaInDesc);
        miopenDestroyTensorDescriptor(deltaInDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t varInDesc;
    miopenTensorDescriptor_t varOutDesc;
    miopenTensorDescriptor_t alphaInDesc;
    miopenTensorDescriptor_t deltaInDesc;

    std::unique_ptr<GPUMem> var_in_dev;
    std::unique_ptr<GPUMem> var_out_dev;
    std::unique_ptr<GPUMem> alpha_in_dev;
    std::unique_ptr<GPUMem> delta_in_dev;

    std::vector<Tgpu> var_in;
    std::vector<Tgpu> var_out;
    std::vector<Tgpu> var_out_init;
    std::vector<Tgpu> alpha_in;
    std::vector<Tgpu> delta_in;

    std::vector<Tref> var_out_host;

    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_dims = inflags.GetValueTensor("input_dims").lengths;
    std::vector<int> alpha_dim  = {1};
    std::vector<int> in_stride  = ComputeStrides(input_dims);

    if(SetTensorNd(varInDesc, input_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing var_in tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(varOutDesc, input_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing var_out tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(alphaInDesc, alpha_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing alpha_in tensor: {1}.");
    if(SetTensorNd(deltaInDesc, input_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing delta_in tensor: " + inflags.GetValueStr("input_dims") + ".");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> GradientDescentDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward GradientDescent (Default=1)", "int");
    inflags.AddTensorFlag(
        "input_dims",
        'd',
        "2x3x7x50x10",
        "The dimensional lengths of the input tensor: N,C,D,H Example: 2x3x7x50x10.");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time Each Layer (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t element_size = miopen::deref(varInDesc).GetElementSize();

    uint32_t ctx = 0;

    var_in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    var_out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    alpha_in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(Tgpu)));
    delta_in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));

    var_in       = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    var_out      = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    var_out_init = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    alpha_in     = std::vector<Tgpu>(1, static_cast<Tgpu>(0));
    delta_in     = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));

    var_out_host = std::vector<Tref>(element_size, static_cast<Tref>(0));

    for(int i = 0; i < element_size; i++)
    {
        var_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    for(int i = 0; i < element_size; i++)
    {
        delta_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    for(int i = 0; i < 1; i++)
    {
        alpha_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(var_in_dev->ToGPU(GetStream(), var_in.data()) != 0)
    {
        std::cerr << "Error copying var_in to GPU, size: " << var_in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(var_out_dev->ToGPU(GetStream(), var_out.data()) != 0)
    {
        std::cerr << "Error copying var_out to GPU, size: " << var_out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(alpha_in_dev->ToGPU(GetStream(), alpha_in.data()) != 0)
    {
        std::cerr << "Error copying alpha_in to GPU, size: " << alpha_in_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(delta_in_dev->ToGPU(GetStream(), delta_in.data()) != 0)
    {
        std::cerr << "Error copying delta_in to GPU, size: " << delta_in_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(var_out_dev->ToGPU(GetStream(), var_out_init.data()) != 0)
        {
            std::cerr << "Error copying var_out to GPU, size: " << var_out_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        auto status = miopenGradientDescent(GetHandle(),
                                            varInDesc,
                                            var_in_dev->GetMem(),
                                            varOutDesc,
                                            var_out_dev->GetMem(),
                                            alphaInDesc,
                                            alpha_in_dev->GetMem(),
                                            deltaInDesc,
                                            delta_in_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenGradientDescent");

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward GradientDescent Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward GradientDescent Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(var_out_dev->FromGPU(GetStream(), var_out.data()) != 0)
    {
        std::cerr << "Error copying var_out_dev from GPU, size: " << var_out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloGradientDescentRunHost<Tgpu, Tref, int>(varInDesc,
                                                        varOutDesc,
                                                        alphaInDesc,
                                                        deltaInDesc,
                                                        var_in.data(),
                                                        var_out_host.data(),
                                                        alpha_in.data(),
                                                        delta_in.data());
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloGradientDescentRunHost");

    return status;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
Tref GradientDescentDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(var_out_host, var_out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward GradientDescent Verifies FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward GradientDescent Verifies OK on CPU reference (err=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GradientDescentDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
