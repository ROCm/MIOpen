#pragma once

#include "miopen/miopen.h"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>
#include <cmath>

template <class TIO>
void cpu_sigmoid_focal_loss_forward(tensor<TIO> input,
                                    tensor<TIO> target,
                                    tensor<TIO>& outputHost,
                                    float alpha,
                                    float gamma,
                                    miopenLossReductionMode_t reduction,
                                    float divisor)
{
    auto input_tv     = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv    = miopen::get_inner_expanded_tv<5>(target.desc);
    auto output_tv    = miopen::get_inner_expanded_tv<5>(outputHost.desc);
    size_t inputSize  = input.desc.GetElementSize();
    float outputFloat = 0;

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);

        float sig    = 1 / (1 + std::exp(-i));
        float ceLoss = -(t * std::log(sig) + (1 - t) * std::log(1 - sig));
        float sigT   = sig * t + (1 - sig) * (1 - t);
        float loss   = ceLoss * std::pow(1 - sigT, gamma);

        if(alpha >= 0)
        {
            float alphaT = alpha * t + (1 - alpha) * (1 - t);
            loss         = alphaT * loss;
        }

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            outputHost[output_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(loss);
        }
        else
        {
            outputFloat += loss / divisor;
        }
    }

    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
    {
        outputHost[0] = static_cast<TIO>(outputFloat);
    }
}

template <class TIO>
void cpu_sigmoid_focal_loss_backward(tensor<TIO> input,
                                     tensor<TIO> target,
                                     tensor<TIO> doutput,
                                     tensor<TIO>& dinput,
                                     tensor<TIO>& dtarget,
                                     float alpha,
                                     float gamma,
                                     miopenLossReductionMode_t reduction,
                                     float divisor)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv  = miopen::get_inner_expanded_tv<5>(target.desc);
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(doutput.desc);
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(dinput.desc);
    auto dtarget_tv = miopen::get_inner_expanded_tv<5>(dtarget.desc);

    size_t inputSize = input.desc.GetElementSize();

    tensor_layout_t<5> doIdx(input_tv, 0);
    float dO = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(doIdx)]);

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            dO = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(idx)]);
        }

        float p       = 1 / (1 + std::exp(-i));
        float ceLoss  = -(t * std::log(p) + (1 - t) * std::log(1 - p));
        float pT      = p * t + (1 - p) * (1 - t);
        float powPt   = std::pow(1 - pT, gamma);
        float alpha_t = alpha * t + (1 - alpha) * (1 - t);

        if(dinput.data.size() > 0)
        {
            float dpdi      = std::exp(-i) / std::pow(1 + std::exp(-i), 2);
            float dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
            float dpowptdi  = gamma * std::pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

            // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
            float dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
            float grad = dO * dLdi;

            if(alpha >= 0)
            {
                grad *= alpha_t;
            }
            if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
            {
                grad /= divisor;
            }
            dinput[dinput_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(grad);
        }

        if(dtarget.data.size() > 0)
        {
            float dcelossdt = -std::log(p) + std::log(1 - p);
            float dpowptdt  = gamma * std::pow(1 - pT, gamma - 1) * (1 - 2 * p);
            // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
            float dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
            float gradTarget = dO * dLdt;

            if(alpha >= 0)
            {
                // alpha_t * dL/dt + dalpha_t/dt * dL
                gradTarget = alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt;
            }
            if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
            {
                gradTarget /= divisor;
            }
            dtarget[dtarget_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(gradTarget);
        }
    }
}

template <typename TIO>
float get_tolerance(miopenLossReductionMode_t reduction)
{
    float tolerance;
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<TIO, bfloat16>::value)
            tolerance *= 8.0;
    }
    else
    {
        tolerance = std::is_same<TIO, float>::value ? 1.0e-2 : 8.2e-1;
    }

    return tolerance;
}
