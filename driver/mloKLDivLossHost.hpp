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
#ifndef MLO_KLDIVLOSSHOST_H_
#define MLO_KLDIVLOSSHOST_H_

#include <miopen/tensor.hpp>
#include <miopen/tensor_view.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloKLDivLossUnreducedBackwardRunHost5d(const miopenTensorDescriptor_t inputDesc,
                                               const miopenTensorDescriptor_t targetDesc,
                                               const miopenTensorDescriptor_t outputGradDesc,
                                               const miopenTensorDescriptor_t inputGradDesc,
                                               const miopenTensorDescriptor_t targetGradDesc,
                                               const Tgpu* input,
                                               const Tgpu* target,
                                               const Tgpu* output_grad,
                                               Tcheck* input_grad,
                                               Tcheck* target_grad,
                                               bool log_target,
                                               bool input_grad_out,
                                               bool target_grad_out)
{
    auto I_tv  = get_inner_expanded_tv_5d(miopen::deref(inputDesc));
    auto T_tv  = get_inner_expanded_tv_5d(miopen::deref(targetDesc));
    auto dO_tv = get_inner_expanded_tv_5d(miopen::deref(outputGradDesc));
    auto dI_tv = get_inner_expanded_tv_5d(miopen::deref(inputGradDesc));
    auto dT_tv = get_inner_expanded_tv_5d(miopen::deref(targetGradDesc));

    auto numel = miopen::deref(inputDesc).GetElementSize();

    for(size_t i = 0; i < numel; ++i)
    {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, dI_tv);
        size_t Iidx  = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx  = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx = TV5D_IDX(dO_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dIidx = TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dTidx = TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4]);

        Tgpu input_value       = input[Iidx];
        Tgpu target_value      = target[Tidx];
        Tgpu output_grad_value = output_grad[dOidx];
        Tgpu forward_output;

        if(log_target)
        {
            Tgpu exp_target = static_cast<Tgpu>(exp(target_value));
            forward_output  = exp_target * (target_value - input_value);
            if(input_grad_out)
            {
                Tgpu input_grad_value =
                    std::isnan(forward_output)
                        ? static_cast<Tgpu>(0.0f)
                        : static_cast<Tgpu>(-1.0f) * exp_target * output_grad_value;
                input_grad[dIidx] = static_cast<Tcheck>(input_grad_value);
            }
            if(target_grad_out)
            {
                Tgpu target_grad_value =
                    static_cast<Tgpu>(forward_output + exp_target) * output_grad_value;
                target_grad[dTidx] = static_cast<Tcheck>(target_grad_value);
            }
        }
        else
        {
            forward_output = target_value * (static_cast<Tgpu>(log(target_value)) - input_value);
            if(input_grad_out)
            {
                Tgpu input_grad_value = std::isnan(forward_output)
                                            ? static_cast<Tgpu>(0.0f)
                                            : -target_value * output_grad_value;
                input_grad[dIidx]     = static_cast<Tcheck>(input_grad_value);
            }
            if(target_grad_out)
            {
                Tgpu target_grad_value =
                    (target_value == 0) ? static_cast<Tgpu>(0.0f)
                                        : (static_cast<Tgpu>(1.0f) +
                                           (static_cast<Tgpu>(log(target_value)) - input_value)) *
                                              output_grad_value;
                target_grad[dTidx] = static_cast<Tcheck>(target_grad_value);
            }
        }
    }
    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloKLDivLossReducedBackwardRunHost5d(const miopenTensorDescriptor_t inputDesc,
                                             const miopenTensorDescriptor_t targetDesc,
                                             const miopenTensorDescriptor_t outputGradDesc,
                                             const miopenTensorDescriptor_t inputGradDesc,
                                             const miopenTensorDescriptor_t targetGradDesc,
                                             const Tgpu* input,
                                             const Tgpu* target,
                                             const Tgpu* output_grad,
                                             Tcheck* input_grad,
                                             Tcheck* target_grad,
                                             float divisor,
                                             bool log_target,
                                             bool input_grad_out,
                                             bool target_grad_out)
{
    auto I_tv  = get_inner_expanded_tv_5d(miopen::deref(inputDesc));
    auto T_tv  = get_inner_expanded_tv_5d(miopen::deref(targetDesc));
    auto dO_tv = get_inner_expanded_tv_1d(miopen::deref(outputGradDesc));
    auto dI_tv = get_inner_expanded_tv_5d(miopen::deref(inputGradDesc));
    auto dT_tv = get_inner_expanded_tv_5d(miopen::deref(targetGradDesc));

    auto numel = miopen::deref(inputDesc).GetElementSize();

    for(size_t i = 0; i < numel; ++i)
    {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, dI_tv);
        size_t Iidx  = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx  = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx = TV1D_IDX(dO_tv, 0);
        size_t dIidx = TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dTidx = TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4]);

        Tgpu input_value       = input[Iidx];
        Tgpu target_value      = target[Tidx];
        Tgpu output_grad_value = output_grad[dOidx];
        Tgpu forward_output;
        Tgpu d = static_cast<Tgpu>(divisor);

        if(log_target)
        {
            Tgpu exp_target = static_cast<Tgpu>(exp(target_value));
            forward_output  = exp_target * (target_value - input_value);
            if(input_grad_out)
            {
                Tgpu input_grad_value =
                    std::isnan(forward_output)
                        ? static_cast<Tgpu>(0.0f)
                        : static_cast<Tgpu>(-1.0f) * exp_target / d * output_grad_value;
                input_grad[dIidx] = static_cast<Tcheck>(input_grad_value);
            }
            if(target_grad_out)
            {
                Tgpu target_grad_value = ((forward_output + exp_target) / d) * output_grad_value;
                target_grad[dTidx]     = static_cast<Tcheck>(target_grad_value);
            }
        }
        else
        {
            forward_output =
                target_value *
                (static_cast<Tgpu>(log(static_cast<double>(target_value))) - input_value);
            if(input_grad_out)
            {
                Tgpu input_grad_value =
                    std::isnan(forward_output)
                        ? static_cast<Tgpu>(0.0f)
                        : static_cast<Tgpu>(-1.0f) * target_value / d * output_grad_value;
                input_grad[dIidx] = static_cast<Tcheck>(input_grad_value);
            }
            if(target_grad_out)
            {
                Tgpu target_grad_value =
                    (target_value == static_cast<Tgpu>(0.0f))
                        ? static_cast<Tgpu>(0.0f)
                        : (static_cast<Tgpu>(1.0f) +
                           static_cast<Tgpu>(log(static_cast<double>(target_value))) -
                           input_value) /
                              d * output_grad_value;
                target_grad[dTidx] = static_cast<Tcheck>(target_grad_value);
            }
        }
    }
    return 0;
}

#endif // MLO_KLDIVLOSSHOST_H_
