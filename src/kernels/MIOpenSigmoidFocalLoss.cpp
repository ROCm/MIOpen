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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

#ifndef IN_OUT_TYPE
#define IN_OUT_TYPE float
#endif

#ifndef CVT_ACCUM2FLOAT
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#endif

#ifndef CVT_FLOAT2ACCUM
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#endif

template <typename TIO>
__device__ void sigmoidFocalLossFwd(const TIO* input,
                                    TIO* target,
                                    TIO* workspace,
                                    float alpha,
                                    float gamma,
                                    float divisor,
                                    tensor_view_t<5> input_tv,
                                    tensor_view_t<5> target_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<5> idx(input_tv, gid);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM i = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t = CVT_FLOAT2ACCUM(target[target_tv.get_tensor_view_idx(idx)]);

    FLOAT_ACCUM p      = 1 / (1 + exp(-i));
    FLOAT_ACCUM ceLoss = -(t * log(p) + (1 - t) * log(1 - p));
    FLOAT_ACCUM pT     = p * t + (1 - p) * (1 - t);
    FLOAT_ACCUM loss   = ceLoss * pow(1 - pT, gamma);

    if(alpha >= 0)
    {
        FLOAT_ACCUM alpha_t = alpha * t + (1 - alpha) * (1 - t);
        loss                = alpha_t * loss;
    }

    workspace[gid] = CVT_ACCUM2FLOAT(loss / divisor);
}

extern "C" __global__ void SigmoidFocalLossFwd(const IN_OUT_TYPE* input,
                                               IN_OUT_TYPE* target,
                                               IN_OUT_TYPE* workspace,
                                               float alpha,
                                               float gamma,
                                               float divisor,
                                               tensor_view_t<5> input_tv,
                                               tensor_view_t<5> target_tv)
{
    sigmoidFocalLossFwd<IN_OUT_TYPE>(
        input, target, workspace, alpha, gamma, divisor, input_tv, target_tv);
}

template <typename TIO>
__device__ void sigmoidFocalLossBwd(const TIO* input,
                                    const TIO* target,
                                    const TIO* doutput,
                                    TIO* dinput,
                                    TIO* dtarget,
                                    float alpha,
                                    float gamma,
                                    float divisor,
                                    tensor_view_t<5> input_tv,
                                    tensor_view_t<5> target_tv,
                                    tensor_view_t<5> doutput_tv,
                                    tensor_view_t<5> dinput_tv,
                                    tensor_view_t<5> dtarget_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<5> idx(input_tv, gid);
    tensor_layout_t<5> doIdx(doutput_tv, 0);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM i  = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t  = CVT_FLOAT2ACCUM(target[target_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM dO = CVT_FLOAT2ACCUM(doutput[doutput_tv.get_tensor_view_idx(doIdx)]);

    FLOAT_ACCUM p       = 1 / (1 + exp(-i));
    FLOAT_ACCUM ceLoss  = -(t * log(p) + (1 - t) * log(1 - p));
    FLOAT_ACCUM pT      = p * t + (1 - p) * (1 - t);
    FLOAT_ACCUM powPt   = pow(1 - pT, gamma);
    FLOAT_ACCUM alpha_t = alpha * t + (1 - alpha) * (1 - t);

    if(dinput)
    {
        FLOAT_ACCUM dpdi = exp(-i) / pow(1 + exp(-i), 2);
        // dceloss/di = dceloss/dp * dp/di
        FLOAT_ACCUM dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
        // dpowt/di = dpowt/dpT * dpT/dp * dp/di
        FLOAT_ACCUM dpowptdi = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

        // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
        FLOAT_ACCUM dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
        FLOAT_ACCUM grad = dO * dLdi;

        if(alpha >= 0)
        {
            grad *= alpha_t;
        }
        grad /= divisor;
        dinput[dinput_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(grad);
    }

    if(dtarget)
    {
        FLOAT_ACCUM dcelossdt = -log(p) + log(1 - p);
        FLOAT_ACCUM dpowptdt  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * p);
        // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
        FLOAT_ACCUM dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
        FLOAT_ACCUM gradTarget = dO * dLdt;

        if(alpha >= 0)
        {
            // alpha_t * dL/dt + dalpha_t/dt * dL
            gradTarget = alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt;
        }
        gradTarget /= divisor;
        dtarget[dtarget_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(gradTarget);
    }
}

extern "C" __global__ void SigmoidFocalLossBwd(const IN_OUT_TYPE* input,
                                               IN_OUT_TYPE* target,
                                               IN_OUT_TYPE* doutput,
                                               IN_OUT_TYPE* dinput,
                                               IN_OUT_TYPE* dtarget,
                                               float alpha,
                                               float gamma,
                                               float divisor,
                                               tensor_view_t<5> input_tv,
                                               tensor_view_t<5> target_tv,
                                               tensor_view_t<5> doutput_tv,
                                               tensor_view_t<5> dinput_tv,
                                               tensor_view_t<5> dtarget_tv)
{
    sigmoidFocalLossBwd<IN_OUT_TYPE>(input,
                                     target,
                                     doutput,
                                     dinput,
                                     dtarget,
                                     alpha,
                                     gamma,
                                     divisor,
                                     input_tv,
                                     target_tv,
                                     doutput_tv,
                                     dinput_tv,
                                     dtarget_tv);
}

template <typename TIO>
__device__ void sigmoidFocalLossUnreducedFwd(const TIO* input,
                                             TIO* target,
                                             TIO* output,
                                             float alpha,
                                             float gamma,
                                             tensor_view_t<5> input_tv,
                                             tensor_view_t<5> target_tv,
                                             tensor_view_t<5> output_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<5> idx(input_tv, gid);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM i = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t = CVT_FLOAT2ACCUM(target[target_tv.get_tensor_view_idx(idx)]);

    FLOAT_ACCUM p      = 1 / (1 + exp(-i));
    FLOAT_ACCUM ceLoss = -(t * log(p) + (1 - t) * log(1 - p));
    FLOAT_ACCUM pT     = p * t + (1 - p) * (1 - t);
    FLOAT_ACCUM loss   = ceLoss * pow(1 - pT, gamma);

    if(alpha >= 0)
    {
        FLOAT_ACCUM alpha_t = alpha * t + (1 - alpha) * (1 - t);
        loss                = alpha_t * loss;
    }

    output[output_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void SigmoidFocalLossUnreducedFwd(const IN_OUT_TYPE* input,
                                                        IN_OUT_TYPE* target,
                                                        IN_OUT_TYPE* output,
                                                        float alpha,
                                                        float gamma,
                                                        tensor_view_t<5> input_tv,
                                                        tensor_view_t<5> target_tv,
                                                        tensor_view_t<5> output_tv)
{
    sigmoidFocalLossUnreducedFwd<IN_OUT_TYPE>(
        input, target, output, alpha, gamma, input_tv, target_tv, output_tv);
}

template <typename TIO>
__device__ void sigmoidFocalLossUnreducedBwd(const TIO* input,
                                             const TIO* target,
                                             const TIO* doutput,
                                             TIO* dinput,
                                             TIO* dtarget,
                                             float alpha,
                                             float gamma,
                                             tensor_view_t<5> input_tv,
                                             tensor_view_t<5> target_tv,
                                             tensor_view_t<5> doutput_tv,
                                             tensor_view_t<5> dinput_tv,
                                             tensor_view_t<5> dtarget_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<5> idx(input_tv, gid);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM i  = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t  = CVT_FLOAT2ACCUM(target[target_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM dO = CVT_FLOAT2ACCUM(doutput[doutput_tv.get_tensor_view_idx(idx)]);

    FLOAT_ACCUM p       = 1 / (1 + exp(-i));
    FLOAT_ACCUM ceLoss  = -(t * log(p) + (1 - t) * log(1 - p));
    FLOAT_ACCUM pT      = p * t + (1 - p) * (1 - t);
    FLOAT_ACCUM powPt   = pow(1 - pT, gamma);
    FLOAT_ACCUM alpha_t = alpha * t + (1 - alpha) * (1 - t);

    if(dinput)
    {
        FLOAT_ACCUM dpdi = exp(-i) / pow(1 + exp(-i), 2);
        // dceloss/di = dceloss/dp * dp/di
        FLOAT_ACCUM dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
        // dpowt/di = dpowt/dpT * dpT/dp * dp/di
        FLOAT_ACCUM dpowptdi = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

        // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
        FLOAT_ACCUM dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
        FLOAT_ACCUM grad = dO * dLdi;

        if(alpha >= 0)
        {
            grad *= alpha_t;
        }
        dinput[dinput_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(grad);
    }

    if(dtarget)
    {
        FLOAT_ACCUM dcelossdt = -log(p) + log(1 - p);
        FLOAT_ACCUM dpowptdt  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * p);
        // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
        FLOAT_ACCUM dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
        FLOAT_ACCUM gradTarget = dO * dLdt;

        if(alpha >= 0)
        {
            // alpha_t * dL/dt + dalpha_t/dt * dL
            gradTarget = alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt;
        }
        dtarget[dtarget_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(gradTarget);
    }
}

extern "C" __global__ void SigmoidFocalLossUnreducedBwd(const IN_OUT_TYPE* input,
                                                        IN_OUT_TYPE* target,
                                                        IN_OUT_TYPE* doutput,
                                                        IN_OUT_TYPE* dinput,
                                                        IN_OUT_TYPE* dtarget,
                                                        float alpha,
                                                        float gamma,
                                                        tensor_view_t<5> input_tv,
                                                        tensor_view_t<5> target_tv,
                                                        tensor_view_t<5> doutput_tv,
                                                        tensor_view_t<5> dinput_tv,
                                                        tensor_view_t<5> dtarget_tv)
{
    sigmoidFocalLossUnreducedBwd<IN_OUT_TYPE>(input,
                                              target,
                                              doutput,
                                              dinput,
                                              dtarget,
                                              alpha,
                                              gamma,
                                              input_tv,
                                              target_tv,
                                              doutput_tv,
                                              dinput_tv,
                                              dtarget_tv);
}
