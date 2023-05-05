/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TENSOROP_DRIVER_HPP
#define GUARD_MIOPEN_TENSOROP_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "miopen/miopen.h"
#include "miopen/tensor.hpp"
#include "random.hpp"
#include "timer.hpp"

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

template <typename Tgpu, typename Tref>
class TensorOpDriver : public Driver
{
public:
    TensorOpDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&aTensor);
        miopenCreateTensorDescriptor(&bTensor);
        miopenCreateTensorDescriptor(&cTensor);
        // TODO: check the dataype
        data_type = miopenFloat;
        op        = miopenTensorOpAdd;
        is_set    = false;
        is_scale  = false;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetTensorOpFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override { return 0; }
    int RunBackwardCPU() { return 0; }

    int VerifyForward() override;
    int VerifyBackward() override { return 0; }

    ~TensorOpDriver() override
    {
        miopenDestroyTensorDescriptor(aTensor);
        miopenDestroyTensorDescriptor(bTensor);
        miopenDestroyTensorDescriptor(cTensor);
    }

private:
    std::function<Tgpu(Tgpu, Tgpu)> TensorOpFn(miopenTensorOp_t op);
    int CheckTensor(std::vector<Tgpu>& cpu_res, std::vector<Tgpu>& gpu_res, double allowedEps);
    InputFlags inflags;

    miopenTensorDescriptor_t aTensor;
    miopenTensorDescriptor_t bTensor;
    miopenTensorDescriptor_t cTensor;

    std::unique_ptr<GPUMem> a_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::unique_ptr<GPUMem> c_dev;

    std::vector<Tgpu> a;
    std::vector<Tgpu> b;
    std::vector<Tgpu> c;

    std::vector<Tgpu> a_verif;
    std::vector<Tgpu> b_verif;
    std::vector<Tgpu> c_verif;

    miopenTensorOp_t op;
    bool is_set;
    bool is_scale;
    double alpha1;
    double alpha2;
    double beta;
    double tensor_val;
};
template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    if(inflags.GetValueInt("time") == 1)
        miopenEnableProfiling(GetHandle(), true);
    return miopenStatusSuccess;
}
template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    SetTensor4d(aTensor, in_len, data_type);
    SetTensor4d(bTensor, in_len, data_type);
    SetTensor4d(cTensor, in_len, data_type);
    SetTensorOpFromCmdLineArgs();
    return (0);
}

template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Direction of operation (not used)", "int");
    inflags.AddInputFlag("batchsize", 'n', "64", "Mini-batch size (Default=64)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("alpha1", 'A', "1", "Activation alpha1 (Default=1)", "double");
    inflags.AddInputFlag("alpha2", 'B', "1", "Activation alpha2 (Default=1)", "double");
    inflags.AddInputFlag("beta", 'G', "0", "Activation beta (Default=1)", "double");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("tensor_op",
                         'o',
                         "0",
                         "Tensor Op to execute (Default = 0), 0 - SetTensor, 1 - ScaleTensor, 2 - "
                         "Add, 3 - Mul, 4 - Min, 5 - Max",
                         "int");
    inflags.AddInputFlag(
        "tensor_val", 'v', "1", "Scalar value for SetTensor and ScaleTensor", "double");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> TensorOpDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");
    return {in_n, in_c, in_h, in_w};
}

template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::SetTensorOpFromCmdLineArgs()
{
    alpha1     = inflags.GetValueDouble("alpha1");
    alpha2     = inflags.GetValueDouble("alpha2");
    beta       = inflags.GetValueDouble("beta");
    tensor_val = inflags.GetValueDouble("tensor_val");
    int raw_op = inflags.GetValueInt("tensor_op");
    if(raw_op == 0)
        is_set = true;
    else if(raw_op == 1)
        is_scale = true;
    else
    {
        if((raw_op - 2) <= static_cast<int>(miopenTensorOpMax))
            op = static_cast<miopenTensorOp_t>(raw_op - 2);
        else
        {
            Usage();
            exit(-1); // NOLINT (concurrency-mt-unsafe)
        }
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    size_t sz = GetTensorSize(aTensor);

    a_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sz, sizeof(Tgpu)));
    if(!is_set && !is_scale)
    {
        b_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sz, sizeof(Tgpu)));
        c_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sz, sizeof(Tgpu)));
    }

    a       = std::vector<Tgpu>(sz, static_cast<Tgpu>(0));
    a_verif = std::vector<Tgpu>(sz, static_cast<Tgpu>(0));
    if(!is_set && !is_scale)
    {
        b       = std::vector<Tgpu>(sz, static_cast<Tgpu>(0));
        b_verif = std::vector<Tgpu>(sz, static_cast<Tgpu>(0));
        c       = std::vector<Tgpu>(sz, static_cast<Tgpu>(0));
        c_verif = std::vector<Tgpu>(sz, static_cast<Tgpu>(0));
    }

    for(int i = 0; i < sz; ++i)
    {
        a[i]       = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
        a_verif[i] = a[i];
        if(!is_set && !is_scale)
        {
            b[i]       = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            b_verif[i] = b[i];
            c[i]       = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            c_verif[i] = c[i];
        }
    }

#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
    int status;
#endif

    status = a_dev->ToGPU(q, a.data());
    if(!is_set && !is_scale)
    {
        status |= b_dev->ToGPU(q, b.data());
        status |= c_dev->ToGPU(q, c.data());
    }

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::RunForwardGPU()
{
    float falpha1     = static_cast<float>(alpha1);
    float falpha2     = static_cast<float>(alpha2);
    float fbeta       = static_cast<float>(beta);
    float ftensor_val = static_cast<float>(tensor_val);

    int iters       = inflags.GetValueInt("iter");
    double fulltime = 0;
    float avgtime   = 0.0f;
    float min_time  = 100000000.0f;

    Timer t;

    for(int i = 0; i < iters; ++i)
    {
        START_TIME

        if(!is_set && !is_scale)
            miopenOpTensor(GetHandle(),
                           op,
                           &falpha1,
                           aTensor,
                           a_dev->GetMem(),
                           &falpha2,
                           bTensor,
                           b_dev->GetMem(),
                           &fbeta,
                           cTensor,
                           c_dev->GetMem());
        else if(is_set)
            miopenSetTensor(GetHandle(), aTensor, a_dev->GetMem(), &ftensor_val);
        else if(is_scale)
            miopenScaleTensor(GetHandle(), aTensor, a_dev->GetMem(), &ftensor_val);

        miopen::deref(GetHandle()).Finish();

        STOP_TIME
        if(WALL_CLOCK)
        {
            if(iters > 1)
                fulltime += t.gettime_ms();
            else if(iters == 1)
                fulltime = t.gettime_ms();
        }
        if(inflags.GetValueInt("time") == 1)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            min_time = (time < min_time) ? time : min_time;
            if(iters > 1)
                avgtime += time;
        }
    }

    if(WALL_CLOCK)
        printf("Wall-clock Time Tensor Ops Elapsed: %f ms, for %d iterations.\n",
               (iters == 1) ? t.gettime_ms() : (fulltime / float(iters - 1)),
               (iters > 1) ? iters - 1 : 1);
    if(inflags.GetValueInt("time") == 1)
    {
        printf("GPU Kernel Min Time Tensor Op Elapsed: %f ms\n", min_time);
        if(iters > 1)
            printf("GPU Kernel Avg Time Tensor Op Elapsed: %f ms, for %d iterations.\n",
                   avgtime / (iters - 1),
                   iters - 1);
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(aTensor).GetLengths());
        size_t dataSz =
            in_n * in_c * in_h * in_w * miopen::GetTypeSize(miopen::deref(aTensor).GetType());

        printf("stats: name, bytesRead, bytesWritten, GB/s, timeMs\n");
        printf("stats: tensor op, %zu, %zu, %f, %f\n",
               3 * dataSz,
               dataSz,
               4 * dataSz / min_time / 1e6,
               avgtime / (iters - 1));
    }
    if(!is_set && !is_scale)
        c_dev->FromGPU(GetStream(), c.data());
    else
        a_dev->FromGPU(GetStream(), a.data());
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::function<Tgpu(Tgpu, Tgpu)> TensorOpDriver<Tgpu, Tref>::TensorOpFn(miopenTensorOp_t op_)
{
    switch(op_)
    {
    case miopenTensorOpAdd: return [&](Tgpu a_, Tgpu b_) { return a_ + b_; };
    case miopenTensorOpMul: return [&](Tgpu a_, Tgpu b_) { return a_ * b_; };
    case miopenTensorOpMin: return [&](Tgpu a_, Tgpu b_) { return (a_ < b_) ? a_ : b_; };
    case miopenTensorOpMax: return [&](Tgpu a_, Tgpu b_) { return (a_ > b_) ? a_ : b_; };
    }
    return {};
}

template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::RunForwardCPU()
{
    int iters = inflags.GetValueInt("iter");
    for(auto idx = 0; idx < iters; ++idx)
    {
        if(is_set)
            std::transform(a_verif.begin(), a_verif.end(), a_verif.begin(), [&](auto) {
                return static_cast<Tgpu>(tensor_val);
            });
        else if(is_scale)
            std::transform(a_verif.begin(), a_verif.end(), a_verif.begin(), [&](auto element) {
                return (element * static_cast<Tgpu>(tensor_val));
            });
        else
        {
            auto op_fn = TensorOpFn(op);
            if(miopen::float_equal(beta, 0.0))
                std::transform(
                    a_verif.begin(), a_verif.end(), b_verif.begin(), c_verif.begin(), op_fn);
            else
            {
                std::vector<Tgpu> tmp(a_verif.size(), static_cast<Tgpu>(0.0));
                std::transform(a_verif.begin(), a_verif.end(), b_verif.begin(), tmp.begin(), op_fn);
                std::transform(tmp.begin(),
                               tmp.end(),
                               c_verif.begin(),
                               c_verif.begin(),
                               [&](auto el_tmp, auto el_c) { return el_tmp + (beta * el_c); });
            }
        }
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::CheckTensor(std::vector<Tgpu>& cpu_res,
                                            std::vector<Tgpu>& gpu_res,
                                            double allowedEps)
{
    int match = 1;

    for(auto idx = 0; idx < cpu_res.size() && match; ++idx)
    {
        Tref cpu_val   = cpu_res[idx];
        Tref gpu_val   = static_cast<Tref>(gpu_res[idx]);
        double err     = std::abs(cpu_val - gpu_val);
        double err_rel = calculate_relative_error(cpu_val, gpu_val);

        if((err > allowedEps && err_rel > allowedEps) || std::isnan(cpu_val) ||
           std::isnan(gpu_val) || !std::isfinite(cpu_val) || !std::isfinite(gpu_val))
        {
            std::cout << "Difference in Tensor Op result: " << err << " too large at " << idx
                      << " cpu value = " << cpu_val << " , gpu_val = " << gpu_val
                      << " tolreance = " << allowedEps << std::endl;
            match = 0;
        }
    }
    return match;
}
template <typename Tgpu, typename Tref>
int TensorOpDriver<Tgpu, Tref>::VerifyForward()
{
    double allowedEps = std::numeric_limits<Tgpu>::epsilon() * 80;
    int match         = 1;

    RunForwardCPU();

    match = CheckTensor(
        (!is_set && !is_scale) ? c_verif : a_verif, (!is_set && !is_scale) ? c : a, allowedEps);

    if(match)
        printf("Tensor Op verifies on CPU and GPU\n");
    return miopenStatusSuccess;
}

#endif // #ifndef GUARD_MIOPEN_TENSOROP_DRIVER_HPP
