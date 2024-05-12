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
#include <miopen/tensor_view.hpp>

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
    auto dims = miopen::deref(inputDesc).GetLengths();
    size_t N  = dims[0];
    size_t C  = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    auto I_tv = get_inner_expanded_tv_4d(miopen::deref(inputDesc));
    auto T_tv = get_inner_expanded_tv_3d(miopen::deref(targetDesc));
    auto W_tv = get_inner_expanded_tv_1d(miopen::deref(weightDesc));
    auto O_tv = get_inner_expanded_tv_3d(miopen::deref(outputDesc));

    for(size_t i = 0; i < N * D1 * D2; i++)
    {
        uint64_t n[3];
        GET_NCD(n[0], n[1], n[2], i, O_tv);
        size_t target_index = TV3D_IDX(T_tv, n[0], n[1], n[2]);
        int32_t t           = target[target_index];
        size_t input_index  = TV4D_IDX(I_tv, n[0], t, n[1], n[2]);
        size_t weight_index = TV1D_IDX(W_tv, t);
        size_t output_index = TV3D_IDX(O_tv, n[0], n[1], n[2]);

        if(t < 0 || t == ignore_index || t >= C)
        {
            output[output_index] = static_cast<Tcheck>(0);
        }
        else
        {
            output[output_index] =
                static_cast<Tcheck>(-1.0f) * weight[weight_index] * input[input_index];
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossReduceForwardRunHost(const miopenTensorDescriptor_t inputDesc,
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
    auto dims = miopen::deref(inputDesc).GetLengths();
    size_t N  = dims[0];
    size_t C  = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    auto I_tv = get_inner_expanded_tv_4d(miopen::deref(inputDesc));
    auto T_tv = get_inner_expanded_tv_3d(miopen::deref(targetDesc));
    auto W_tv = get_inner_expanded_tv_1d(miopen::deref(weightDesc));

    for(size_t i = 0; i < N * D1 * D2; i++)
    {
        uint64_t n[3];
        GET_NCD(n[0], n[1], n[2], i, T_tv);
        size_t target_index = TV3D_IDX(T_tv, n[0], n[1], n[2]);
        int32_t t           = target[target_index];
        size_t input_index  = TV4D_IDX(I_tv, n[0], t, n[1], n[2]);
        size_t weight_index = TV1D_IDX(W_tv, t);

        if(t < 0 || t == ignore_index || t >= C)
        {
            workspace[target_index] = static_cast<Tcheck>(0);
        }
        else
        {
            workspace[target_index] =
                (static_cast<Tcheck>(-1.0f) * weight[weight_index] * input[input_index]) / static_cast<Tcheck>(divisor);
        }
    }

    auto size = N * D1 * D2;
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            Tcheck shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : static_cast<Tcheck>(0.0f);
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < local_size; ++j)
                    if(j < offset)
                        shared[j] += shared[j + offset];
            if(_size <= local_size)
                output[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossUnreduceBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                          const miopenTensorDescriptor_t targetDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          const miopenTensorDescriptor_t outputDesc,
                                          Tcheck* input_grad,
                                          const int32_t* target,
                                          const Tgpu* weight,
                                          const Tgpu* output_grad,
                                          const int32_t ignore_index)
{
    auto dims = miopen::deref(inputDesc).GetLengths();
    size_t N  = dims[0];
    size_t C  = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    auto I_tv = get_inner_expanded_tv_4d(miopen::deref(inputDesc));
    auto T_tv = get_inner_expanded_tv_3d(miopen::deref(targetDesc));
    auto W_tv = get_inner_expanded_tv_1d(miopen::deref(weightDesc));
    auto O_tv = get_inner_expanded_tv_3d(miopen::deref(outputDesc));

    for(size_t i = 0; i < N * D1 * D2; i++)
    {
        uint64_t n[3];
        GET_NCD(n[0], n[1], n[2], i, O_tv);
        size_t target_index      = TV3D_IDX(T_tv, n[0], n[1], n[2]);
        int32_t t                = target[target_index];
        size_t input_grad_index  = TV4D_IDX(I_tv, n[0], t, n[1], n[2]);
        size_t weight_index      = TV1D_IDX(W_tv, t);
        size_t output_grad_index = TV3D_IDX(O_tv, n[0], n[1], n[2]);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<Tcheck>(0);
        }
        else
        {
            input_grad[input_grad_index] =
                static_cast<Tcheck>(-1.0f) * weight[weight_index] * output_grad[output_grad_index];
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossReduceBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                          const miopenTensorDescriptor_t targetDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          Tcheck* input_grad,
                                          const int32_t* target,
                                          const Tgpu* weight,
                                          const Tgpu* output_grad,
                                          const int32_t ignore_index,
                                          const float divisor)
{
    auto dims = miopen::deref(inputDesc).GetLengths();
    size_t N  = dims[0];
    size_t C  = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    auto I_tv = get_inner_expanded_tv_4d(miopen::deref(inputDesc));
    auto T_tv = get_inner_expanded_tv_3d(miopen::deref(targetDesc));
    auto W_tv = get_inner_expanded_tv_1d(miopen::deref(weightDesc));

    for(size_t i = 0; i < N * D1 * D2; i++)
    {
        uint64_t n[3];
        GET_NCD(n[0], n[1], n[2], i, T_tv);
        size_t target_index      = TV3D_IDX(T_tv, n[0], n[1], n[2]);
        int32_t t                = target[target_index];
        size_t input_grad_index  = TV4D_IDX(I_tv, n[0], t, n[1], n[2]);
        size_t weight_index      = TV1D_IDX(W_tv, t);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<Tcheck>(0);
        }
        else
        {
            input_grad[input_grad_index] =
                (static_cast<Tcheck>(-1.0f) * weight[weight_index] * output_grad[0]) / static_cast<Tcheck>(divisor);
        }
    }

    return 0;
}

#endif // MLO_NLLLOSSHOST_H_
