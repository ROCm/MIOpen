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

template <typename Tgpu, typename Tcheck>
int32_t mloMarginRankingLossForwardRunHost(const miopenTensorDescriptor_t input1Desc,
                                           const Tgpu* input1,
                                           const miopenTensorDescriptor_t input2Desc,
                                           const Tgpu* input2,
                                           const miopenTensorDescriptor_t targetDesc,
                                           const Tgpu* target,
                                           const miopenTensorDescriptor_t outputDesc,
                                           Tcheck* output,
                                           float margin,
                                           float divisor,
                                           miopenMarginRakningLossReductionMode_t reduction_mode)
{
    tensor_view_t<5> I1_tv = get_inner_expanded_tv<5>(miopen::deref(input1Desc));
    tensor_view_t<5> I2_tv = get_inner_expanded_tv<5>(miopen::deref(input2Desc));
    tensor_view_t<5> T_tv  = get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    tensor_view_t<5> O_tv  = get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    uint64_t tensor_size   = miopen::deref(targetDesc).GetElementSize();
    float sum_loss         = 0;

    for(uint64_t gid = 0; gid < tensor_size; ++gid)
    {
        uint64_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
        uint64_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
        uint64_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
        uint64_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];

        if(!(n0 < I1_tv.size[0]))
            return 0;

        uint64_t I1idx = I1_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        uint64_t I2idx = I2_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        uint64_t Tidx  = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

        float output_accum =
            -static_cast<float>(target[Tidx]) *
                (static_cast<float>(input1[I1idx]) - static_cast<float>(input2[I2idx])) +
            margin;
        if(output_accum < 0.0f)
            output_accum = 0.0f;

        if(reduction_mode == MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE)
        {
            uint64_t Oidx = O_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
            output[Oidx]  = static_cast<Tcheck>(output_accum);
        }
        else
        {
            sum_loss += (output_accum / divisor);
        }
    }
    if(reduction_mode != MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE)
    {
        output[0] = static_cast<Tcheck>(sum_loss);
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMarginRankingLossBackwardRunHost(const miopenTensorDescriptor_t input1Desc,
                                            const Tgpu* input1,
                                            const miopenTensorDescriptor_t input2Desc,
                                            const Tgpu* input2,
                                            const miopenTensorDescriptor_t targetDesc,
                                            const Tgpu* target,
                                            const miopenTensorDescriptor_t outGradDesc,
                                            const Tgpu* outGrad,
                                            const miopenTensorDescriptor_t in1GradDesc,
                                            Tcheck* in1Grad,
                                            const miopenTensorDescriptor_t in2GradDesc,
                                            Tcheck* in2Grad,
                                            float margin,
                                            float divisor,
                                            miopenMarginRakningLossReductionMode_t reduction_mode)
{
    tensor_view_t<5> I1_tv  = get_inner_expanded_tv<5>(miopen::deref(input1Desc));
    tensor_view_t<5> I2_tv  = get_inner_expanded_tv<5>(miopen::deref(input2Desc));
    tensor_view_t<5> T_tv   = get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    tensor_view_t<5> dO_tv  = get_inner_expanded_tv<5>(miopen::deref(outGradDesc));
    tensor_view_t<5> dI1_tv = get_inner_expanded_tv<5>(miopen::deref(in1GradDesc));
    tensor_view_t<5> dI2_tv = get_inner_expanded_tv<5>(miopen::deref(in2GradDesc));
    uint64_t tensor_size    = miopen::deref(targetDesc).GetElementSize();

    par_ford(tensor_size)([&](uint64_t gid) {
        uint64_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
        uint64_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
        uint64_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
        uint64_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];
        uint64_t dOidx = 0;

        if(!(n0 < I1_tv.size[0]))
            return;

        uint64_t I1idx  = I1_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        uint64_t I2idx  = I2_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        uint64_t dI1idx = dI1_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        uint64_t dI2idx = dI2_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        uint64_t Tidx   = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

        float t = -static_cast<float>(target[Tidx]) *
                      (static_cast<float>(input1[I1idx]) - static_cast<float>(input2[I2idx])) +
                  margin;

        if(t < 0)
        {
            in1Grad[dI1idx] = 0.0f;
            in2Grad[dI2idx] = 0.0f;
        }
        else
        {
            if(reduction_mode == MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE)
            {
                dOidx   = dO_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
                divisor = 1;
            }
            float d_accum =
                static_cast<float>(target[Tidx]) * static_cast<float>(outGrad[dOidx]) / divisor;
            in1Grad[dI1idx] = static_cast<Tcheck>(-d_accum);
            in2Grad[dI2idx] = static_cast<Tcheck>(d_accum);
        }
    });

    return 0;
}
