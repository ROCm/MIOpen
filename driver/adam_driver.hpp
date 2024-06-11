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
#ifndef GUARD_MIOPEN_ADAM_DRIVER_HPP
#define GUARD_MIOPEN_ADAM_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include "../test/verify.hpp"

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

#ifndef MLO_ADAMHOST_H_
#define MLO_ADAMHOST_H_

template <typename Tref>
void mloAdamRunHost(miopenTensorDescriptor_t paramDesc,
                    Tref* params,
                    Tref* grads,
                    Tref* exp_avgs,
                    Tref* exp_avg_sqs,
                    Tref* max_exp_avg_sqs,
                    int32_t step,
                    float lr,
                    float beta1,
                    float beta2,
                    float weight_decay,
                    float eps,
                    bool amsgrad,
                    bool maximize,
                    bool adamw,
                    bool is_amp,
                    int32_t grad_scale,
                    bool found_inf)
{
    if(is_amp && found_inf)
        return;

    size_t numel = miopen::deref(paramDesc).GetElementSize();
    for(int i = 0; i < numel; i++)
    {
        Tref exp_avg    = exp_avgs[i];
        Tref exp_avg_sq = exp_avg_sqs[i];

        Tref param = params[i];
        Tref grad  = grads[i];
        if(maximize)
            grad *= -1;
        if(is_amp)
            grad /= grad_scale;

        float bias_correction1 = 1 - pow(beta1, step);
        float bias_correction2 = 1 - pow(beta2, step);

        if(weight_decay != 0)
        {
            if(adamw)
                param -= lr * weight_decay * param;
            else
                grad += param * weight_decay;
        }

        exp_avg    = exp_avg * beta1 + grad * (1 - beta1);
        exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);

        float denom;
        if(amsgrad)
        {
            Tref max_exp_avg_sq = max_exp_avg_sqs[i];
            if(exp_avg_sq > max_exp_avg_sq)
            {
                max_exp_avg_sq     = exp_avg_sq;
                max_exp_avg_sqs[i] = max_exp_avg_sq;
            }

            denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        }
        else
        {
            denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        }

        params[i] = param - (lr / bias_correction1) * exp_avg / denom;
    }
}

#endif

template <typename Tgpu, typename Tref = Tgpu, typename Tgrad = Tgpu>
class AdamDriver : public Driver
{
public:
    AdamDriver(bool adamw_ = false, bool is_amp_ = false) : Driver(), adamw(adamw_), is_amp(is_amp_)
    {
        miopenCreateTensorDescriptor(&paramDesc);
        miopenCreateTensorDescriptor(&gradDesc);
        miopenCreateTensorDescriptor(&expAvgDesc);
        miopenCreateTensorDescriptor(&expAvgSqDesc);
        miopenCreateTensorDescriptor(&paramOutDesc);
        miopenCreateTensorDescriptor(&dummyOutDesc);
        if(is_amp)
        {
            miopenCreateTensorDescriptor(&stepDesc);
            miopenCreateTensorDescriptor(&gradScaleDesc);
            miopenCreateTensorDescriptor(&foundInfDesc);
        }

        data_type = miopen_type<Tgpu>{};
        grad_type = miopen_type<Tgrad>{};
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
    ~AdamDriver() override
    {
        miopenDestroyTensorDescriptor(paramDesc);
        miopenDestroyTensorDescriptor(gradDesc);
        miopenDestroyTensorDescriptor(expAvgDesc);
        miopenDestroyTensorDescriptor(expAvgSqDesc);
        miopenDestroyTensorDescriptor(paramOutDesc);
        miopenDestroyTensorDescriptor(dummyOutDesc);
        if(maxExpAvgSqDesc)
            miopenDestroyTensorDescriptor(maxExpAvgSqDesc);
        if(stepDesc)
            miopenDestroyTensorDescriptor(stepDesc);
        if(gradScaleDesc)
            miopenDestroyTensorDescriptor(gradScaleDesc);
        if(stepDesc)
            miopenDestroyTensorDescriptor(foundInfDesc);
    }

private:
    InputFlags inflags;

    int forw = 1;

    miopenTensorDescriptor_t paramDesc       = nullptr;
    miopenTensorDescriptor_t gradDesc        = nullptr;
    miopenTensorDescriptor_t expAvgDesc      = nullptr;
    miopenTensorDescriptor_t expAvgSqDesc    = nullptr;
    miopenTensorDescriptor_t maxExpAvgSqDesc = nullptr;
    miopenTensorDescriptor_t stepDesc        = nullptr;
    miopenTensorDescriptor_t gradScaleDesc   = nullptr;
    miopenTensorDescriptor_t foundInfDesc    = nullptr;
    miopenTensorDescriptor_t paramOutDesc    = nullptr;
    miopenTensorDescriptor_t dummyOutDesc    = nullptr;

    std::unique_ptr<GPUMem> param_dev;
    std::unique_ptr<GPUMem> param_out_dev;
    std::unique_ptr<GPUMem> dummy_out_dev;
    std::unique_ptr<GPUMem> grad_dev;
    std::unique_ptr<GPUMem> exp_avg_dev;
    std::unique_ptr<GPUMem> exp_avg_sq_dev;
    std::unique_ptr<GPUMem> max_exp_avg_sq_dev;
    std::unique_ptr<GPUMem> step_dev;
    std::unique_ptr<GPUMem> scale_dev;
    std::unique_ptr<GPUMem> found_inf_dev;

    std::vector<Tgpu> param;
    std::vector<Tgrad> grad;
    std::vector<Tgpu> exp_avg;
    std::vector<Tgpu> exp_avg_sq;
    std::vector<Tgpu> max_exp_avg_sq;

    std::vector<Tref> param_host;
    std::vector<Tref> grad_host;
    std::vector<Tref> exp_avg_host;
    std::vector<Tref> exp_avg_sq_host;
    std::vector<Tref> max_exp_avg_sq_host;

    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float eps;
    bool amsgrad   = false;
    bool maximize  = false;
    bool found_inf = false;
    bool adamw     = false;
    bool is_amp    = false;
    int grad_scale = 1;
    int iter       = 0;

    miopenDataType_t grad_type;
};

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::GetandSetData()
{
    auto param_len = GetInputTensorLengthsFromCmdLine();
    lr             = inflags.GetValueDouble("lr");
    beta1          = inflags.GetValueDouble("beta1");
    beta2          = inflags.GetValueDouble("beta2");
    eps            = inflags.GetValueDouble("eps");
    weight_decay   = inflags.GetValueDouble("weight_decay");
    amsgrad        = inflags.GetValueInt("amsgrad");
    maximize       = inflags.GetValueInt("maximize");
    iter           = inflags.GetValueInt("iter");

    if(is_amp)
    {
        grad_scale = inflags.GetValueInt("scale");
        found_inf  = inflags.GetValueInt("found_inf");
    }

    std::vector<int> one_size = {1};
    SetTensorNd(paramDesc, param_len, data_type);
    SetTensorNd(paramOutDesc, param_len, data_type);
    SetTensorNd(gradDesc, param_len, grad_type);
    SetTensorNd(expAvgDesc, param_len, data_type);
    SetTensorNd(expAvgSqDesc, param_len, data_type);
    SetTensorNd(dummyOutDesc, param_len, data_type);

    if(amsgrad)
    {
        miopenCreateTensorDescriptor(&maxExpAvgSqDesc);
        SetTensorNd(maxExpAvgSqDesc, param_len, data_type);
    }

    if(is_amp)
    {
        SetTensorNd(stepDesc, one_size, miopenInt32);
        SetTensorNd(gradScaleDesc, one_size, miopenInt32);
        SetTensorNd(foundInfDesc, one_size, miopenInt32);
    }

    return 0;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward GroupNorm (Default=1)", "int");
    inflags.AddTensorFlag("dims", 'd', "64x32x128", "params tensor dims (Default=64x32x128)");

    inflags.AddInputFlag("lr", 'l', "0.001", "learning rate (Default=0.001)", "float");
    inflags.AddInputFlag("beta1", '1', "0.9", "beta1 (Default=0.9)", "float");
    inflags.AddInputFlag("beta2", '2', "0.999", "beta2 (Default=0.999)", "float");
    inflags.AddInputFlag("eps", 'e', "0.00000001", "eps (Default=0.00000001)", "float");
    inflags.AddInputFlag("weight_decay", 'W', "0", "weight decay (Default=0)", "float");
    inflags.AddInputFlag("amsgrad", 'a', "0", "whether to use the AMSGrad (Default=0)", "int");
    inflags.AddInputFlag("maximize", 'm', "0", "whether to use the maximize (Default=0)", "int");

    if(is_amp)
    {
        inflags.AddInputFlag("scale", 's', "65536", "grad scale factor (Default=65536)", "int");
        inflags.AddInputFlag("found_inf", 'f', "0", "found inf in grad (Default=0)", "int");
    }

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tgrad>
std::vector<int> AdamDriver<Tgpu, Tref, Tgrad>::GetInputTensorLengthsFromCmdLine()
{
    std::vector<int> ret;
    auto tensor = inflags.GetValueTensor("dims");
    if(!tensor.lengths.empty())
        return tensor.lengths;
    return ret;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::AllocateBuffersAndCopy()
{
    size_t param_sz = GetTensorSize(paramDesc);

    uint32_t ctx   = 0;
    param_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    grad_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgrad)));
    exp_avg_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    exp_avg_sq_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    param_out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    dummy_out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));

    if(amsgrad)
        max_exp_avg_sq_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));

    if(is_amp)
    {
        step_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(int)));
        scale_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(int)));
        found_inf_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(bool)));
    }

    param      = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));
    grad       = std::vector<Tgrad>(param_sz, static_cast<Tgrad>(0));
    exp_avg    = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));
    exp_avg_sq = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));

    param_host      = std::vector<Tref>(param_sz, static_cast<Tref>(0));
    grad_host       = std::vector<Tref>(param_sz, static_cast<Tref>(0));
    exp_avg_host    = std::vector<Tref>(param_sz, static_cast<Tref>(0));
    exp_avg_sq_host = std::vector<Tref>(param_sz, static_cast<Tref>(0));

    if(amsgrad)
    {
        max_exp_avg_sq      = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));
        max_exp_avg_sq_host = std::vector<Tref>(param_sz, static_cast<Tref>(0));
    }

    for(int i = 0; i < param_sz; i++)
    {
        param[i]        = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        grad[i]         = prng::gen_A_to_B<Tgrad>(static_cast<Tgrad>(0.0), static_cast<Tgrad>(0.1));
        exp_avg[i]      = prng::gen_A_to_B<Tgrad>(static_cast<Tgrad>(0), static_cast<Tgrad>(0.1));
        exp_avg_sq[i]   = prng::gen_A_to_B<Tgrad>(static_cast<Tgrad>(0), static_cast<Tgrad>(0.1));
        param_host[i]   = param[i];
        exp_avg_host[i] = exp_avg[i];
        exp_avg_sq_host[i] = exp_avg_sq[i];

        if(amsgrad)
        {
            max_exp_avg_sq[i] =
                prng::gen_A_to_B<Tgrad>(static_cast<Tgrad>(0.5), static_cast<Tgrad>(1.0));
            max_exp_avg_sq_host[i] = max_exp_avg_sq[i];
        }

        if(is_amp)
        {
            grad[i] *= grad_scale;
            if(!found_inf && (std::isnan(grad[i]) || std::isinf(grad[i])))
            {
                std::cerr << "Error init (grad), idx: " << i << ", value: " << grad[i] << std::endl;
                found_inf = true;
            }
        }
        grad_host[i] = grad[i];
    }

    if(param_dev->ToGPU(GetStream(), param.data()) != 0)
        std::cerr << "Error copying (param) to GPU, size: " << param_dev->GetSize() << std::endl;

    if(grad_dev->ToGPU(GetStream(), grad.data()) != 0)
        std::cerr << "Error copying (grad) to GPU, size: " << grad_dev->GetSize() << std::endl;

    if(exp_avg_dev->ToGPU(GetStream(), exp_avg.data()) != 0)
        std::cerr << "Error copying (exp_avg) to GPU, size: " << exp_avg_dev->GetSize()
                  << std::endl;

    if(exp_avg_sq_dev->ToGPU(GetStream(), exp_avg_sq.data()) != 0)
        std::cerr << "Error copying (exp_avg_sq) to GPU, size: " << exp_avg_sq_dev->GetSize()
                  << std::endl;

    if(amsgrad)
    {
        if(max_exp_avg_sq_dev->ToGPU(GetStream(), max_exp_avg_sq.data()) != 0)
            std::cerr << "Error copying (max_exp_avg_sq) to GPU, size: "
                      << max_exp_avg_sq_dev->GetSize() << std::endl;
    }

    if(is_amp)
    {
        int step = 0;
        if(step_dev->ToGPU(GetStream(), &step) != 0)
            std::cerr << "Error copying (step) to GPU, size: " << step_dev->GetSize() << std::endl;

        if(scale_dev->ToGPU(GetStream(), &grad_scale) != 0)
            std::cerr << "Error copying (scale) to GPU, size: " << scale_dev->GetSize()
                      << std::endl;
        if(found_inf_dev->ToGPU(GetStream(), &found_inf) != 0)
            std::cerr << "Error copying (found_inf) to GPU, size: " << found_inf_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    void* max_exp_avg_sq_ptr = amsgrad ? max_exp_avg_sq_dev->GetMem() : nullptr;
    void* grad_scale_ptr     = is_amp ? scale_dev->GetMem() : nullptr;
    void* found_inf_ptr      = is_amp ? found_inf_dev->GetMem() : nullptr;
    void* state_step_ptr     = is_amp ? step_dev->GetMem() : nullptr;

    Timer t;
    START_TIME

    for(int i = 0; i < iter; i++)
    {
        miopenFusedAdamWithOutput(GetHandle(),
                                  paramDesc,
                                  param_dev->GetMem(),
                                  paramOutDesc,
                                  param_out_dev->GetMem(),
                                  nullptr,
                                  nullptr,
                                  gradDesc,
                                  grad_dev->GetMem(),
                                  expAvgDesc,
                                  exp_avg_dev->GetMem(),
                                  dummyOutDesc,
                                  dummy_out_dev->GetMem(),
                                  expAvgSqDesc,
                                  exp_avg_sq_dev->GetMem(),
                                  dummyOutDesc,
                                  dummy_out_dev->GetMem(),
                                  maxExpAvgSqDesc,
                                  max_exp_avg_sq_ptr,
                                  dummyOutDesc,
                                  dummy_out_dev->GetMem(),
                                  stepDesc,
                                  state_step_ptr,
                                  stepDesc,
                                  state_step_ptr,
                                  i + 1,
                                  lr,
                                  beta1,
                                  beta2,
                                  weight_decay,
                                  eps,
                                  amsgrad,
                                  maximize,
                                  adamw,
                                  gradScaleDesc,
                                  grad_scale_ptr,
                                  foundInfDesc,
                                  found_inf_ptr);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward Adam Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Adam Elapsed: %f ms\n", kernel_average_time);
    }

    if(param_out_dev->FromGPU(GetStream(), param.data()) != 0)
        std::cerr << "Error copying (param_dev) from GPU, size: " << param_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::RunForwardCPU()
{
    mloAdamRunHost<Tref>(paramDesc,
                         param_host.data(),
                         grad_host.data(),
                         exp_avg_host.data(),
                         exp_avg_sq_host.data(),
                         max_exp_avg_sq_host.data(),
                         iter,
                         lr,
                         beta1,
                         beta2,
                         weight_decay,
                         eps,
                         amsgrad,
                         maximize,
                         adamw,
                         is_amp,
                         grad_scale,
                         found_inf);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tgrad>
Tref AdamDriver<Tgpu, Tref, Tgrad>::GetTolerance()
{
    if(data_type == miopenHalf)
    {
        return 1e-3;
    }
    else if(data_type == miopenFloat)
    {
        return 5e-5;
    }
    else if(data_type == miopenDouble)
    {
        return 1e-10;
    }
    else if(data_type == miopenBFloat16)
    {
        return 5e-3;
    }
    return 0;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(param_host, param);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Adam FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "Forward Adam Verifies OK on CPU reference" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tgrad>
int AdamDriver<Tgpu, Tref, Tgrad>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_ADAM_DRIVER_HPP
