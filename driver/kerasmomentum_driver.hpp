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

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <vector>

template <typename Tgpu, typename Tcheck>
int32_t mloKerasMomentumRunHost(const miopenTensorDescriptor_t varInDesc,
                                const miopenTensorDescriptor_t varOutDesc,
                                const miopenTensorDescriptor_t accumInDesc,
                                const miopenTensorDescriptor_t accumOutDesc,
                                const miopenTensorDescriptor_t gradInDesc,
                                const Tgpu* var_in,
                                Tcheck* var_out,
                                const Tgpu* accum_in,
                                Tcheck* accum_out,
                                const Tgpu* lr_in,
                                const Tgpu* grad_in,
                                const Tgpu* momentum_in,
                                const bool nesterov)
{
    auto var_in_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(varInDesc));
    auto var_out_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(varOutDesc));
    auto accum_in_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(accumInDesc));
    auto accum_out_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(accumOutDesc));
    auto grad_in_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(gradInDesc));

    uint64_t N = miopen::deref(varInDesc).GetElementSize();

    par_ford(N)([&](uint64_t gid) {
        auto tensor_layout = tensor_layout_t<5>(var_in_tv, gid);
        double var = static_cast<double>(var_in[var_in_tv.get_tensor_view_idx(tensor_layout)]);
        double accum =
            static_cast<double>(accum_in[accum_in_tv.get_tensor_view_idx(tensor_layout)]);
        double grad = static_cast<double>(grad_in[grad_in_tv.get_tensor_view_idx(tensor_layout)]);
        double lr   = static_cast<double>(lr_in[0]);
        double momentum = static_cast<double>(momentum_in[0]);

        accum = accum * momentum - grad * lr;

        if(nesterov)
        {
            var += accum * momentum - grad * lr;
        }
        else
        {
            var += accum;
        }

        var_out[var_out_tv.get_tensor_view_idx(tensor_layout)]     = static_cast<Tcheck>(var);
        accum_out[accum_out_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tcheck>(accum);
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref = Tgpu>
class KerasMomentumDriver : public Driver
{
public:
    KerasMomentumDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&varInDesc);
        miopenCreateTensorDescriptor(&varOutDesc);
        miopenCreateTensorDescriptor(&accumInDesc);
        miopenCreateTensorDescriptor(&accumOutDesc);
        miopenCreateTensorDescriptor(&lrInDesc);
        miopenCreateTensorDescriptor(&gradInDesc);
        miopenCreateTensorDescriptor(&momentumInDesc);

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
    ~KerasMomentumDriver() override
    {
        miopenDestroyTensorDescriptor(varInDesc);
        miopenDestroyTensorDescriptor(varOutDesc);
        miopenDestroyTensorDescriptor(accumInDesc);
        miopenDestroyTensorDescriptor(accumOutDesc);
        miopenDestroyTensorDescriptor(lrInDesc);
        miopenDestroyTensorDescriptor(gradInDesc);
        miopenDestroyTensorDescriptor(momentumInDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t varInDesc;
    miopenTensorDescriptor_t varOutDesc;
    miopenTensorDescriptor_t accumInDesc;
    miopenTensorDescriptor_t accumOutDesc;
    miopenTensorDescriptor_t lrInDesc;
    miopenTensorDescriptor_t gradInDesc;
    miopenTensorDescriptor_t momentumInDesc;

    std::unique_ptr<GPUMem> var_in_dev;
    std::unique_ptr<GPUMem> var_out_dev;
    std::unique_ptr<GPUMem> accum_in_dev;
    std::unique_ptr<GPUMem> accum_out_dev;
    std::unique_ptr<GPUMem> lr_in_dev;
    std::unique_ptr<GPUMem> grad_in_dev;
    std::unique_ptr<GPUMem> momentum_in_dev;

    std::vector<Tgpu> var_in;
    std::vector<Tgpu> var_out;
    std::vector<Tgpu> accum_in;
    std::vector<Tgpu> accum_out;
    std::vector<Tgpu> lr_in;
    std::vector<Tgpu> grad_in;
    std::vector<Tgpu> momentum_in;

    std::vector<Tref> var_out_host;
    std::vector<Tref> accum_out_host;

    bool nesterov;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    nesterov     = inflags.GetValueInt("nesterov") == 1 ? true : false;
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_dims = inflags.GetValueTensor("input_dims").lengths;
    std::vector<int> one_dim    = {1};
    std::vector<int> stride     = ComputeStrides(input_dims);

    if(SetTensorNd(varInDesc, input_dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing var_in tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(varOutDesc, input_dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing var_out tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(accumInDesc, input_dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing accum_in tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(accumOutDesc, input_dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing accum_out tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(lrInDesc, one_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing lr_in tensor.");
    if(SetTensorNd(gradInDesc, input_dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing grad_in tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(momentumInDesc, one_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing momentum_in tensor.");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> KerasMomentumDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int KerasMomentumDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward KerasMomentum (Default=1)", "int");
    inflags.AddTensorFlag(
        "input_dims",
        'd',
        "2x3x7x50x10",
        "The dimensional lengths of the input tensor: N,C,D,H Example: 2x3x7x50x10.");
    inflags.AddInputFlag("nesterov", 'n', "1", "nesterov (Default=1)", "int");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time Each Layer (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t element_size = miopen::deref(varInDesc).GetElementSize();

    uint32_t ctx = 0;

    var_in_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    var_out_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    accum_in_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    accum_out_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    lr_in_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(Tgpu)));
    grad_in_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    momentum_in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(Tgpu)));

    var_in      = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    var_out     = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    accum_in    = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    accum_out   = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    lr_in       = std::vector<Tgpu>(1, static_cast<Tgpu>(0));
    grad_in     = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    momentum_in = std::vector<Tgpu>(1, static_cast<Tgpu>(0));

    var_out_host   = std::vector<Tref>(element_size, static_cast<Tref>(0));
    accum_out_host = std::vector<Tref>(element_size, static_cast<Tref>(0));

    for(int i = 0; i < element_size; i++)
    {
        var_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.1), static_cast<Tgpu>(1.0));
    }
    for(int i = 0; i < element_size; i++)
    {
        accum_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.1), static_cast<Tgpu>(1.0));
    }
    for(int i = 0; i < element_size; i++)
    {
        grad_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.1), static_cast<Tgpu>(1.0));
    }
    lr_in[0]       = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.1), static_cast<Tgpu>(1.0));
    momentum_in[0] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.1), static_cast<Tgpu>(1.0));

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
    if(accum_in_dev->ToGPU(GetStream(), accum_in.data()) != 0)
    {
        std::cerr << "Error copying accum_in to GPU, size: " << accum_in_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(accum_out_dev->ToGPU(GetStream(), accum_out.data()) != 0)
    {
        std::cerr << "Error copying accum_out to GPU, size: " << accum_out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(lr_in_dev->ToGPU(GetStream(), lr_in.data()) != 0)
    {
        std::cerr << "Error copying lr_in to GPU, size: " << lr_in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(grad_in_dev->ToGPU(GetStream(), grad_in.data()) != 0)
    {
        std::cerr << "Error copying grad_in to GPU, size: " << grad_in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(momentum_in_dev->ToGPU(GetStream(), momentum_in.data()) != 0)
    {
        std::cerr << "Error copying momentum_in to GPU, size: " << momentum_in_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenKerasMomentum(GetHandle(),
                                          varInDesc,
                                          var_in_dev->GetMem(),
                                          varOutDesc,
                                          var_out_dev->GetMem(),
                                          accumInDesc,
                                          accum_in_dev->GetMem(),
                                          accumOutDesc,
                                          accum_out_dev->GetMem(),
                                          lrInDesc,
                                          lr_in_dev->GetMem(),
                                          gradInDesc,
                                          grad_in_dev->GetMem(),
                                          momentumInDesc,
                                          momentum_in_dev->GetMem(),
                                          nesterov);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenKerasMomentum");

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
            std::cout << "Wall-clock Time Forward KerasMomentum Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward KerasMomentum Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(var_out_dev->FromGPU(GetStream(), var_out.data()) != 0)
    {
        std::cerr << "Error copying var_out_dev from GPU, size: " << var_out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(accum_out_dev->FromGPU(GetStream(), accum_out.data()) != 0)
    {
        std::cerr << "Error copying accum_out_dev from GPU, size: " << accum_out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloKerasMomentumRunHost<Tgpu, Tref>(varInDesc,
                                                 varOutDesc,
                                                 accumInDesc,
                                                 accumOutDesc,
                                                 gradInDesc,
                                                 var_in.data(),
                                                 var_out_host.data(),
                                                 accum_in.data(),
                                                 accum_out_host.data(),
                                                 lr_in.data(),
                                                 grad_in.data(),
                                                 momentum_in.data(),
                                                 nesterov);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloKerasMomentumRunHost");

    return status;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
Tref KerasMomentumDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_var       = miopen::rms_range(var_out_host, var_out);

    if(!std::isfinite(error_var) || error_var > tolerance)
    {
        std::cout << "Forward KerasMomentum Verifies var_out FAILED: " << error_var << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward KerasMomentum Verifies var_out OK on CPU reference (err=" << error_var
                  << " < " << tolerance << ')' << std::endl;
    }

    auto error_accum = miopen::rms_range(accum_out_host, accum_out);

    if(!std::isfinite(error_accum) || error_accum > tolerance)
    {
        std::cout << "Forward KerasMomentum Verifies accum_out FAILED: " << error_accum << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward KerasMomentum Verifies accum_out OK on CPU reference (err="
                  << error_accum << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KerasMomentumDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
