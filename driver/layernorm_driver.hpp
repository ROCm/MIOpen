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
#ifndef GUARD_MIOPEN_LAYERNORM_DRIVER_HPP
#define GUARD_MIOPEN_LAYERNORM_DRIVER_HPP

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>

template <typename Tgpu, typename Tcheck>
int32_t mloLayerNormForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                   Tgpu* input,
                                   Tgpu* weight,
                                   Tgpu* bias,
                                   Tcheck* outputhost,
                                   Tcheck* meanhost,
                                   Tcheck* rstdhost,
                                   float eps,
                                   int32_t normalized_dim,
                                   miopenNormMode_t mode)
{
    auto dims         = miopen::deref(inputDesc).GetLengths();
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t norm_dim   = static_cast<size_t>(normalized_dim);

    for(size_t i = 0ULL; i < dims.size(); ++i)
    {
        if(i < norm_dim)
            outer_size *= dims[i];
        else
            inner_size *= dims[i];
    }

    int32_t ret = 0;

    for(int32_t o = 0; o < outer_size; o++)
    {
        Tcheck pmean = 0.0f;
        Tcheck pvar  = 0.0f;
        for(int32_t i = 0; i < inner_size; i++)
        {
            Tcheck tmp = static_cast<Tcheck>(input[o * inner_size + i]);
            pmean += tmp;
            pvar += tmp * tmp;
        }

        pmean        = pmean / inner_size;
        pvar         = pvar / inner_size - pmean * pmean;
        Tcheck prstd = 1.0f / sqrt(pvar + eps);

        meanhost[o] = pmean;
        rstdhost[o] = prstd;

        for(int32_t i = 0; i < inner_size; i++)
        {
            Tcheck pweight =
                (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 1 : static_cast<Tcheck>(weight[i]);
            Tcheck pbias = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 0 : static_cast<Tcheck>(bias[i]);
            outputhost[o * inner_size + i] =
                (static_cast<Tcheck>(input[o * inner_size + i]) - pmean) * prstd * pweight + pbias;
        }
    }
    return ret;
}

template <typename Tgpu, typename Tref>
class LayerNormDriver : public Driver
{
public:
    LayerNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&weightDesc);
        miopenCreateTensorDescriptor(&biasDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&meanDesc);
        miopenCreateTensorDescriptor(&rstdDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~LayerNormDriver() override
    {

        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(weightDesc);
        miopenDestroyTensorDescriptor(biasDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(meanDesc);
        miopenDestroyTensorDescriptor(rstdDesc);
    }

private:
    InputFlags inflags;

    int dim_size;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t weightDesc;
    miopenTensorDescriptor_t biasDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t meanDesc;
    miopenTensorDescriptor_t rstdDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> bias_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> mean_dev;
    std::unique_ptr<GPUMem> rstd_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> bias;
    std::vector<Tgpu> out;
    std::vector<Tgpu> mean;
    std::vector<Tgpu> rstd;
    std::vector<Tref> outhost;
    std::vector<Tref> meanhost;
    std::vector<Tref> rstdhost;

    float eps;
    int dim;
    miopenNormMode_t mode;
};

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::GetandSetData()
{
    auto inTensorParam = inflags.GetValueTensor("input");

    auto in_len = inTensorParam.lengths;

    dim = inflags.GetValueInt("normalized_dim");

    MIOPEN_THROW_IF(dim < 0 || static_cast<size_t>(dim) >= in_len.size(),
                    "normalized_dim out of range");

    std::vector<int> inner_len;
    if(dim == in_len.size())
        inner_len = {1};
    else
        inner_len = {in_len.begin() + dim, in_len.end()};

    std::vector<int> outer_len;
    if(dim == 0)
        outer_len = {1};
    else
        outer_len = {in_len.begin(), in_len.end() - (in_len.size() - dim)};

    if(SetTensorNd(inputDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(weightDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting weight tensor.");

    if(SetTensorNd(biasDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting bias tensor.");

    if(SetTensorNd(outputDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting doutput tensor.");

    if(SetTensorNd(meanDesc, outer_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting mean tensor.");

    if(SetTensorNd(rstdDesc, outer_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting rstd tensor.");

    eps  = static_cast<double>(inflags.GetValueDouble("eps"));
    mode = miopenNormMode_t(inflags.GetValueInt("mode"));

    return 0;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward LayerNorm (Default=1)", "int");
    inflags.AddTensorFlag("input", 'X', "100x3x32x32", "input tensor descriptor");

    inflags.AddInputFlag("eps", 'e', "0.00001", "Alpha (Default=0.00001)", "double");
    inflags.AddInputFlag("normalized_dim", 'o', "3", "Nomalized Dim (Default=3)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "0", "elemwise affine mode (0), weight and bias mode (1) (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    const Tgpu Tgpu0val = static_cast<Tgpu>(0.0);
    const Tgpu Tgpu1val = static_cast<Tgpu>(1.0);
    const Tref Tref0ref = static_cast<Tref>(0.0);
    size_t in_sz        = GetTensorSize(inputDesc);
    size_t weight_sz    = GetTensorSize(weightDesc);
    size_t bias_sz      = GetTensorSize(biasDesc);
    size_t out_sz       = GetTensorSize(outputDesc);
    size_t mean_sz      = GetTensorSize(meanDesc);
    size_t rstd_sz      = GetTensorSize(rstdDesc);

    uint32_t ctx = 0;

    in_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    weight_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
    bias_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, bias_sz, sizeof(Tgpu)));
    out_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    mean_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, mean_sz, sizeof(Tgpu)));
    rstd_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, rstd_sz, sizeof(Tgpu)));

    in       = std::vector<Tgpu>(in_sz, Tgpu0val);
    weight   = std::vector<Tgpu>(weight_sz, Tgpu0val);
    bias     = std::vector<Tgpu>(bias_sz, Tgpu0val);
    out      = std::vector<Tgpu>(out_sz, Tgpu0val);
    mean     = std::vector<Tgpu>(mean_sz, Tgpu0val);
    rstd     = std::vector<Tgpu>(rstd_sz, Tgpu0val);
    outhost  = std::vector<Tref>(out_sz, Tref0ref);
    meanhost = std::vector<Tref>(mean_sz, Tref0ref);
    rstdhost = std::vector<Tref>(rstd_sz, Tref0ref);

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(Tgpu0val, Tgpu1val);
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    for(int i = 0; i < weight_sz; i++)
    {
        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
            weight[i] = static_cast<Tgpu>(1);
        else
            weight[i] = prng::gen_A_to_B<Tgpu>(Tgpu0val, Tgpu1val);
    }

    if(weight_dev->ToGPU(GetStream(), weight.data()) != 0)
        std::cerr << "Error copying (weight) to GPU, size: " << weight_dev->GetSize() << std::endl;

    for(int i = 0; i < bias_sz; i++)
    {
        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
            bias[i] = Tgpu0val;
        else
            bias[i] = prng::gen_A_to_B<Tgpu>(Tgpu0val, Tgpu1val);
    }
    if(bias_dev->ToGPU(GetStream(), bias.data()) != 0)
        std::cerr << "Error copying (bias) to GPU, size: " << bias_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    if(mean_dev->ToGPU(GetStream(), mean.data()) != 0)
        std::cerr << "Error copying (mean) to GPU, size: " << mean_dev->GetSize() << std::endl;

    if(rstd_dev->ToGPU(GetStream(), rstd.data()) != 0)
        std::cerr << "Error copying (rstd) to GPU, size: " << rstd_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenLayerNormForward(GetHandle(),
                               mode,
                               inputDesc,
                               in_dev->GetMem(),
                               weightDesc,
                               weight_dev->GetMem(),
                               biasDesc,
                               bias_dev->GetMem(),
                               eps,
                               dim,
                               outputDesc,
                               out_dev->GetMem(),
                               meanDesc,
                               mean_dev->GetMem(),
                               rstdDesc,
                               rstd_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward LayerNorm Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward LayerNorm Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    if(mean_dev->FromGPU(GetStream(), mean.data()) != 0)
        std::cerr << "Error copying (mean_dev) from GPU, size: " << mean_dev->GetSize()
                  << std::endl;

    if(rstd_dev->FromGPU(GetStream(), rstd.data()) != 0)
        std::cerr << "Error copying (rstd_dev) from GPU, size: " << rstd_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloLayerNormForwardRunHost<Tgpu, Tref>(inputDesc,
                                           in.data(),
                                           weight.data(),
                                           bias.data(),
                                           outhost.data(),
                                           meanhost.data(),
                                           rstdhost.data(),
                                           eps,
                                           dim,
                                           mode);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref LayerNormDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward LayerNorm FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LayerNorm Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    auto meanerror = miopen::rms_range(meanhost, mean);
    if(!std::isfinite(meanerror) || meanerror > tolerance)
    {
        std::cout << "Forward Layernorm mean FAILED: " << meanerror << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LayerNorm mean Verifies OK on CPU reference (" << meanerror << " < "
                  << tolerance << ')' << std::endl;
    }

    auto rstderror = miopen::rms_range(rstdhost, rstd);
    if(!std::isfinite(rstderror) || rstderror > tolerance)
    {
        std::cout << "Forward LayerNorm rstd FAILED: " << rstderror << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LayerNorm rstd Verifies OK on CPU reference (" << rstderror << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LayerNormDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_LAYERNORM_DRIVER_HPP
