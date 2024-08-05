#pragma once

#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>
#include <cmath>

template <class TIO>
void cpu_sigmoid_focal_loss_unreduced_forward(tensor<TIO> input,
                                              tensor<TIO> target,
                                              tensor<TIO>& outputHost,
                                              float alpha = 0.25,
                                              float gamma = 2)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv   = miopen::get_inner_expanded_tv<5>(target.desc);
    auto output_tv   = miopen::get_inner_expanded_tv<5>(outputHost.desc);
    size_t inputSize = input.desc.GetElementSize();

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

        outputHost[output_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(loss);
    }
}

template <class TIO>
void cpu_sigmoid_focal_loss_unreduced_backward(tensor<TIO> input,
                                               tensor<TIO> target,
                                               tensor<TIO> doutput,
                                               tensor<TIO>& dinput,
                                               tensor<TIO>& dtarget,
                                               float alpha = 0.25,
                                               float gamma = 2)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv   = miopen::get_inner_expanded_tv<5>(target.desc);
    auto doutput_tv  = miopen::get_inner_expanded_tv<5>(doutput.desc);
    auto dinput_tv   = miopen::get_inner_expanded_tv<5>(dinput.desc);
    auto dtarget_tv  = miopen::get_inner_expanded_tv<5>(dtarget.desc);
    size_t inputSize = input.desc.GetElementSize();

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i  = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t  = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);
        float dO = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(idx)]);

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
            dtarget[dtarget_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(gradTarget);
        }
    }
}

template <class TIO>
void cpu_sigmoid_focal_loss_forward(tensor<TIO> input,
                                    tensor<TIO> target,
                                    tensor<TIO>& workspace,
                                    tensor<TIO>& outputHost,
                                    float alpha   = 0.25,
                                    float gamma   = 2,
                                    float divisor = 1)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv   = miopen::get_inner_expanded_tv<5>(target.desc);
    size_t inputSize = input.desc.GetElementSize();
    // float reduction_float;

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
        // reduction_float += (loss / divisor);

        workspace[id] = static_cast<TIO>(loss / divisor);
    }
    // std::cout << "Reduction result in float" << reduction_float << " " << divisor << std::endl;

    // Reduce loss
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = inputSize;
    size_t _size         = inputSize;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            TIO shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                outputHost[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class TIO>
void cpu_sigmoid_focal_loss_backward(tensor<TIO> input,
                                     tensor<TIO> target,
                                     tensor<TIO> doutput,
                                     tensor<TIO>& dinput,
                                     tensor<TIO>& dtarget,
                                     float alpha   = 0.25,
                                     float gamma   = 2,
                                     float divisor = 1)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv  = miopen::get_inner_expanded_tv<5>(target.desc);
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(doutput.desc);
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(dinput.desc);
    auto dtarget_tv = miopen::get_inner_expanded_tv<5>(dtarget.desc);

    size_t inputSize = input.desc.GetElementSize();

    tensor_layout_t<5> doIdx(input_tv, 0);

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i  = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t  = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);
        float dO = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(doIdx)]);

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
            grad /= divisor;
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
            gradTarget /= divisor;
            dtarget[dtarget_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(gradTarget);
        }
    }
}
