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
#ifndef MLO_NLLLOSSHOST_H_
#define MLO_NLLLOSSHOST_H_

#include <miopen/tensor.hpp>
#include <miopen/nllloss/utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                         const miopenTensorDescriptor_t targetDesc,
                                         const miopenTensorDescriptor_t weightDesc,
                                         const miopenTensorDescriptor_t outputDesc,
                                         const Tgpu* input,
                                         const int32_t* target,
                                         const Tgpu* weight,
                                         Tcheck* output,
                                         const int32_t ignore_index)
{
    auto num_dims = miopen::deref(inputDesc).GetNumDims();
    if(num_dims == 2)
    {
        mloNLLLossUnreduceForwardRunHost2d(inputDesc,
                                           targetDesc,
                                           weightDesc,
                                           outputDesc,
                                           input,
                                           target,
                                           weight,
                                           output,
                                           ignore_index);
    }
    else if(num_dims < 5)
    {
        mloNLLLossUnreduceForwardRunHost4d(inputDesc,
                                           targetDesc,
                                           weightDesc,
                                           outputDesc,
                                           input,
                                           target,
                                           weight,
                                           output,
                                           ignore_index);
    }
    else if(num_dims < 6)
    {
        mloNLLLossUnreduceForwardRunHost5d(inputDesc,
                                           targetDesc,
                                           weightDesc,
                                           outputDesc,
                                           input,
                                           target,
                                           weight,
                                           output,
                                           ignore_index);
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossReduceForward5dRunHost(const miopenTensorDescriptor_t inputDesc,
                                         const miopenTensorDescriptor_t targetDesc,
                                         const miopenTensorDescriptor_t weightDesc,
                                         const Tgpu* input,
                                         const int32_t* target,
                                         const Tgpu* weight,
                                         Tcheck* output,
                                         Tcheck* workspace,
                                         const int32_t ignore_index,
                                         const float divisor)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_tv  = miopen::solver::nllloss::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));

    for(size_t i = 0; i < numel; i++)
    {
        auto tensor_layout = tensor_layout_t<4>(target_tv, i);
        uint64_t n[4];
        n[0] = tensor_layout.layout[0];
        n[1] = tensor_layout.layout[1];
        n[2] = tensor_layout.layout[2];
        n[3] = tensor_layout.layout[3];

        size_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<5> input_layout(n[0], t, n[1], n[2], n[3]);
        size_t input_index = input_tv.get_tensor_view_idx(input_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            workspace[i] = static_cast<Tcheck>(0);
        }
        else
        {
            float w = static_cast<float>(weight[weight_index]);

            float input_value = static_cast<float>(input[input_index]);
            float d           = !std::isnan(divisor) ? divisor : 1.0f;
            workspace[i]      = static_cast<Tcheck>((-w * input_value) / d);
        }
    }

    auto size            = numel;
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            float shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? static_cast<float>(workspace[offset_a + i + j]) : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < local_size; ++j)
                    if(j < offset)
                        shared[j] += shared[j + offset];
            if(_size <= local_size)
                output[0] = static_cast<Tcheck>(shared[0]);
            else
                workspace[offset_b + i / local_size] = static_cast<Tcheck>(shared[0]);
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceBackwardRunHost(const miopenTensorDescriptor_t inputGradDesc,
                                          const miopenTensorDescriptor_t targetDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          const miopenTensorDescriptor_t outputGradDesc,
                                          Tcheck* input_grad,
                                          const int32_t* target,
                                          const Tgpu* weight,
                                          const Tgpu* output_grad,
                                          const int32_t ignore_index)
{
    auto num_dims = miopen::deref(inputGradDesc).GetNumDims();
    if(num_dims == 2)
    {
        mloNLLLossUnreduceBackwardRunHost2d(inputGradDesc,
                                            targetDesc,
                                            weightDesc,
                                            outputGradDesc,
                                            input_grad,
                                            target,
                                            weight,
                                            output_grad,
                                            ignore_index);
    }
    else if(num_dims < 5)
    {
        mloNLLLossUnreduceBackwardRunHost4d(inputGradDesc,
                                            targetDesc,
                                            weightDesc,
                                            outputGradDesc,
                                            input_grad,
                                            target,
                                            weight,
                                            output_grad,
                                            ignore_index);
    }
    else if(num_dims < 6)
    {
        mloNLLLossUnreduceBackwardRunHost5d(inputGradDesc,
                                            targetDesc,
                                            weightDesc,
                                            outputGradDesc,
                                            input_grad,
                                            target,
                                            weight,
                                            output_grad,
                                            ignore_index);
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossReduceBackwardRunHost(const miopenTensorDescriptor_t inputGradDesc,
                                        const miopenTensorDescriptor_t targetDesc,
                                        const miopenTensorDescriptor_t weightDesc,
                                        Tcheck* input_grad,
                                        const int32_t* target,
                                        const Tgpu* weight,
                                        const Tgpu* output_grad,
                                        const int32_t ignore_index,
                                        const float divisor)
{
    auto num_dims = miopen::deref(inputGradDesc).GetNumDims();
    if(num_dims == 2)
    {
        mloNLLLossReduceBackwardRunHost2d(inputGradDesc,
                                          targetDesc,
                                          weightDesc,
                                          input_grad,
                                          target,
                                          weight,
                                          output_grad,
                                          ignore_index,
                                          divisor);
    }
    else if(num_dims < 6)
    {
        mloNLLLossReduceBackwardRunHost5d(inputGradDesc,
                                          targetDesc,
                                          weightDesc,
                                          input_grad,
                                          target,
                                          weight,
                                          output_grad,
                                          ignore_index,
                                          divisor);
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceForwardRunHost2d(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t targetDesc,
                                           const miopenTensorDescriptor_t weightDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const Tgpu* input,
                                           const int32_t* target,
                                           const Tgpu* weight,
                                           Tcheck* output,
                                           const int32_t ignore_index)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_tv  = miopen::solver::nllloss::get_inner_expanded_tv<2>(miopen::deref(inputDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));
    auto output_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(outputDesc));

    for(size_t i = 0; i < numel; i++)
    {
        tensor_layout_t<1> target_layout(i);
        size_t target_index = target_tv.get_tensor_view_idx(target_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<2> input_layout(i, t);
        size_t input_index = input_tv.get_tensor_view_idx(input_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        size_t output_index = output_tv.get_tensor_view_idx(target_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            output[output_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w              = static_cast<float>(weight[weight_index]);
            float input_value    = static_cast<float>(input[input_index]);
            float output_value   = -w * input_value;
            output[output_index] = static_cast<Tcheck>(output_value);
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceForwardRunHost4d(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t targetDesc,
                                           const miopenTensorDescriptor_t weightDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const Tgpu* input,
                                           const int32_t* target,
                                           const Tgpu* weight,
                                           Tcheck* output,
                                           const int32_t ignore_index)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_tv  = miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<3>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));
    auto output_tv = miopen::solver::nllloss::get_inner_expanded_tv<3>(miopen::deref(outputDesc));

    for(size_t i = 0; i < numel; i++)
    {
        auto tensor_layout = tensor_layout_t<3>(output_tv, i);
        uint64_t n[3];
        n[0] = tensor_layout.layout[0];
        n[1] = tensor_layout.layout[1];
        n[2] = tensor_layout.layout[2];
        tensor_layout_t<3> target_layout(n[0], n[1], n[2]);
        size_t target_index = target_tv.get_tensor_view_idx(target_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<4> input_layout(n[0], t, n[1], n[2]);
        size_t input_index = input_tv.get_tensor_view_idx(input_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        tensor_layout_t<3> output_layout(n[0], n[1], n[2]);
        size_t output_index = output_tv.get_tensor_view_idx(output_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            output[output_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w              = static_cast<float>(weight[weight_index]);
            float input_value    = static_cast<float>(input[input_index]);
            float output_value   = -w * input_value;
            output[output_index] = static_cast<Tcheck>(output_value);
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceForwardRunHost5d(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t targetDesc,
                                           const miopenTensorDescriptor_t weightDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const Tgpu* input,
                                           const int32_t* target,
                                           const Tgpu* weight,
                                           Tcheck* output,
                                           const int32_t ignore_index)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_tv  = miopen::solver::nllloss::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));
    auto output_tv = miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(outputDesc));

    for(size_t i = 0; i < numel; i++)
    {
        auto tensor_layout = tensor_layout_t<4>(output_tv, i);
        uint64_t n[4];
        n[0] = tensor_layout.layout[0];
        n[1] = tensor_layout.layout[1];
        n[2] = tensor_layout.layout[2];
        n[3] = tensor_layout.layout[3];

        tensor_layout_t<4> target_layout(n[0], n[1], n[2], n[3]);
        size_t target_index = target_tv.get_tensor_view_idx(target_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<5> input_layout(n[0], t, n[1], n[2], n[3]);
        size_t input_index = input_tv.get_tensor_view_idx(input_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        size_t output_index = output_tv.get_tensor_view_idx(tensor_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            output[output_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w              = static_cast<float>(weight[weight_index]);
            float input_value    = static_cast<float>(input[input_index]);
            float output_value   = -w * input_value;
            output[output_index] = static_cast<Tcheck>(output_value);
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceBackwardRunHost2d(const miopenTensorDescriptor_t inputGradDesc,
                                            const miopenTensorDescriptor_t targetDesc,
                                            const miopenTensorDescriptor_t weightDesc,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            Tcheck* input_grad,
                                            const int32_t* target,
                                            const Tgpu* weight,
                                            const Tgpu* output_grad,
                                            const int32_t ignore_index)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_grad_tv =
        miopen::solver::nllloss::get_inner_expanded_tv<2>(miopen::deref(inputGradDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));
    auto output_grad_tv =
        miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(outputGradDesc));

    for(size_t i = 0; i < numel; i++)
    {
        auto tensor_layout  = tensor_layout_t<1>(target_tv, i);
        size_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<2> input_grad_layout(i, t);
        size_t input_grad_index = input_grad_tv.get_tensor_view_idx(input_grad_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        size_t output_grad_index = output_grad_tv.get_tensor_view_idx(tensor_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w                 = static_cast<float>(weight[weight_index]);
            float output_grad_value = static_cast<float>(output_grad[output_grad_index]);

            float input_grad_value       = -w * output_grad_value;
            input_grad[input_grad_index] = static_cast<Tcheck>(input_grad_value);
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceBackwardRunHost4d(const miopenTensorDescriptor_t inputGradDesc,
                                            const miopenTensorDescriptor_t targetDesc,
                                            const miopenTensorDescriptor_t weightDesc,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            Tcheck* input_grad,
                                            const int32_t* target,
                                            const Tgpu* weight,
                                            const Tgpu* output_grad,
                                            const int32_t ignore_index)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_tv = miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(inputGradDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<3>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));
    auto output_grad_tv =
        miopen::solver::nllloss::get_inner_expanded_tv<3>(miopen::deref(outputGradDesc));

    for(size_t i = 0; i < numel; i++)
    {
        auto tensor_layout = tensor_layout_t<3>(output_grad_tv, i);
        uint64_t n[3];
        n[0] = tensor_layout.layout[0];
        n[1] = tensor_layout.layout[1];
        n[2] = tensor_layout.layout[2];

        size_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<4> input_grad_layout(n[0], t, n[1], n[2]);
        size_t input_grad_index = input_tv.get_tensor_view_idx(input_grad_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        size_t output_grad_index = output_grad_tv.get_tensor_view_idx(tensor_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w                 = static_cast<float>(weight[weight_index]);
            float output_grad_value = static_cast<float>(output_grad[output_grad_index]);

            float input_grad_value       = -w * output_grad_value;
            input_grad[input_grad_index] = static_cast<Tcheck>(input_grad_value);
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceBackwardRunHost5d(const miopenTensorDescriptor_t inputGradDesc,
                                            const miopenTensorDescriptor_t targetDesc,
                                            const miopenTensorDescriptor_t weightDesc,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            Tcheck* input_grad,
                                            const int32_t* target,
                                            const Tgpu* weight,
                                            const Tgpu* output_grad,
                                            const int32_t ignore_index)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_grad_tv =
        miopen::solver::nllloss::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));
    auto output_grad_tv =
        miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(outputGradDesc));

    for(size_t i = 0; i < numel; i++)
    {
        auto tensor_layout = tensor_layout_t<4>(output_grad_tv, i);
        uint64_t n[4];
        n[0] = tensor_layout.layout[0];
        n[1] = tensor_layout.layout[1];
        n[2] = tensor_layout.layout[2];
        n[3] = tensor_layout.layout[3];

        size_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<5> input_grad_layout(n[0], t, n[1], n[2], n[3]);
        size_t input_grad_index = input_grad_tv.get_tensor_view_idx(input_grad_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        size_t output_grad_index = output_grad_tv.get_tensor_view_idx(tensor_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w                 = static_cast<float>(weight[weight_index]);
            float output_grad_value = static_cast<float>(output_grad[output_grad_index]);

            float input_grad_value       = -w * output_grad_value;
            input_grad[input_grad_index] = static_cast<Tcheck>(input_grad_value);
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossReduceBackwardRunHost2d(const miopenTensorDescriptor_t inputGradDesc,
                                          const miopenTensorDescriptor_t targetDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          Tcheck* input_grad,
                                          const int32_t* target,
                                          const Tgpu* weight,
                                          const Tgpu* output_grad,
                                          const int32_t ignore_index,
                                          const float divisor)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_grad_tv =
        miopen::solver::nllloss::get_inner_expanded_tv<2>(miopen::deref(inputGradDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));

    for(size_t i = 0; i < numel; i++)
    {
        tensor_layout_t<1> target_layout(i);
        size_t target_index = target_tv.get_tensor_view_idx(target_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<2> input_grad_layout(i, t);
        size_t input_grad_index = input_grad_tv.get_tensor_view_idx(input_grad_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w                 = static_cast<float>(weight[weight_index]);
            float output_grad_value = static_cast<float>(output_grad[0]);

            float input_grad_value       = -w * output_grad_value / divisor;
            input_grad[input_grad_index] = static_cast<Tcheck>(input_grad_value);
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossReduceBackwardRunHost5d(const miopenTensorDescriptor_t inputGradDesc,
                                          const miopenTensorDescriptor_t targetDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          Tcheck* input_grad,
                                          const int32_t* target,
                                          const Tgpu* weight,
                                          const Tgpu* output_grad,
                                          const int32_t ignore_index,
                                          const float divisor)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(targetDesc).GetElementSize();
    size_t C   = dims[1];

    auto input_grad_tv =
        miopen::solver::nllloss::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    auto target_tv = miopen::solver::nllloss::get_inner_expanded_tv<4>(miopen::deref(targetDesc));
    auto weight_tv = miopen::solver::nllloss::get_inner_expanded_tv<1>(miopen::deref(weightDesc));

    for(size_t i = 0; i < numel; i++)
    {
        auto tensor_layout = tensor_layout_t<4>(target_tv, i);
        uint64_t n[4];
        n[0] = tensor_layout.layout[0];
        n[1] = tensor_layout.layout[1];
        n[2] = tensor_layout.layout[2];
        n[3] = tensor_layout.layout[3];

        size_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
        int32_t t           = target[target_index];

        tensor_layout_t<5> input_grad_layout(n[0], t, n[1], n[2], n[3]);
        size_t input_grad_index = input_grad_tv.get_tensor_view_idx(input_grad_layout);

        tensor_layout_t<1> weight_layout(t);
        size_t weight_index = weight_tv.get_tensor_view_idx(weight_layout);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<Tcheck>(0);
        }
        else
        {
            float w                 = static_cast<float>(weight[weight_index]);
            float output_grad_value = static_cast<float>(output_grad[0]);

            float input_grad_value       = -w * output_grad_value / divisor;
            input_grad[input_grad_index] = static_cast<Tcheck>(input_grad_value);
        }
    }

    return 0;
}

#endif // MLO_NLLLOSSHOST_H_
