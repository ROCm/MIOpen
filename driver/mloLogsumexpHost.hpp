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
#ifndef MLO_LOGSUMEXPHOST_H_
#define MLO_LOGSUMEXPHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloLogSumExpForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                   miopenTensorDescriptor_t outputDesc,
                                   Tgpu* input,
                                   Tcheck* output,
                                   std::vector<int32_t> dims_vector)
{
    auto input_dims     = miopen::deref(inputDesc).GetLengths();
    auto input_strides  = miopen::deref(inputDesc).GetStrides();
    auto output_dims    = miopen::deref(outputDesc).GetLengths();
    auto output_strides = miopen::deref(outputDesc).GetStrides();

    for(int64_t d = input_dims.size() - 1; d >= 0; --d)
    {
        if(!(std::find(dims_vector.begin(), dims_vector.end(), d) != dims_vector.end()))
            continue;
        for(int64_t dd = input_dims.size() - 1; dd > d; --dd)
        {
            if(std::find(dims_vector.begin(), dims_vector.end(), dd) != dims_vector.end())
                continue;
            std::swap(input_dims[d], input_dims[dd]);
            std::swap(input_strides[d], input_strides[dd]);
            std::swap(output_dims[d], output_dims[dd]);
            std::swap(output_strides[d], output_strides[dd]);
        }
    }

    auto input_numel =
        std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());

    auto K = input_numel / output_numel;

    for(size_t gid = 0; gid < output_numel; ++gid)
    {
        std::vector<float> vals(K);
        float max = std::numeric_limits<float>::lowest();

        for(int64_t k = 0; k < K; ++k)
        {
            std::vector<int64_t> input_idx(input_dims.size(), 0);
            int64_t tmp_gid = gid * K + k;
            for(int i = input_dims.size() - 1; i >= 0; --i)
            {
                input_idx[i] = tmp_gid % input_dims[i];
                tmp_gid /= input_dims[i];
            }
            float val = input[std::inner_product(
                input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0))];
            max       = max > val ? max : val;
            vals[k]   = val;
        }

        float logsum = static_cast<float>(0.0);
        for(int64_t k = 0; k < K; ++k)
        {
            logsum += std::exp(vals[k] - max);
        }

        std::vector<int64_t> output_idx(input_dims.size(), 0);

        int64_t tmp_gid = gid;
        for(int i = output_dims.size() - 1; i >= 0; --i)
        {
            output_idx[i] = tmp_gid % output_dims[i];
            tmp_gid /= output_dims[i];
        }

        output[std::inner_product(
            output_idx.begin(), output_idx.end(), output_strides.begin(), static_cast<size_t>(0))] =
            static_cast<Tcheck>(max + std::log(logsum));
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloLogSumExpBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                                    miopenTensorDescriptor_t inputGradDesc,
                                    miopenTensorDescriptor_t outputDesc,
                                    miopenTensorDescriptor_t outputGradDesc,
                                    Tgpu* input,
                                    Tcheck* input_grad,
                                    Tgpu* output,
                                    Tgpu* output_grad,
                                    int32_t* dims,
                                    int32_t num_dims)
{
    auto input_dims          = miopen::deref(inputDesc).GetLengths();
    auto input_strides       = miopen::deref(inputDesc).GetStrides();
    auto input_grad_dims     = miopen::deref(inputGradDesc).GetLengths();
    auto input_grad_strides  = miopen::deref(inputGradDesc).GetStrides();
    auto output_dims         = miopen::deref(outputDesc).GetLengths();
    auto output_strides      = miopen::deref(outputDesc).GetStrides();
    auto output_grad_dims    = miopen::deref(outputGradDesc).GetLengths();
    auto output_grad_strides = miopen::deref(outputGradDesc).GetStrides();

    auto input_grad_numel = std::accumulate(
        input_grad_dims.begin(), input_grad_dims.end(), 1LL, std::multiplies<int64_t>());

    for(size_t gid = 0; gid < input_grad_numel; ++gid)
    {
        std::vector<int64_t> input_idx(input_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_idx[i] = tmp_gid % input_dims[i];
            tmp_gid /= input_dims[i];
        }

        std::vector<int64_t> reduced_idx(input_dims.size(), 0);
        for(int i = 0; i < input_dims.size(); ++i)
        {
            if(std::find(dims, dims + num_dims, i) == dims + num_dims)
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        int64_t input_index = std::inner_product(
            input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0));
        int64_t input_grad_index = std::inner_product(
            input_idx.begin(), input_idx.end(), input_grad_strides.begin(), static_cast<size_t>(0));
        int64_t output_index = std::inner_product(
            reduced_idx.begin(), reduced_idx.end(), output_strides.begin(), static_cast<size_t>(0));
        int64_t output_grad_index = std::inner_product(reduced_idx.begin(),
                                                       reduced_idx.end(),
                                                       output_grad_strides.begin(),
                                                       static_cast<size_t>(0));

        float x  = input[input_index];
        float y  = output[output_index];
        float dy = output_grad[output_grad_index];

        input_grad[input_grad_index] = static_cast<Tcheck>(dy * std::exp(x - y));
    }

    return 0;
}

#endif // MLO_LOGSUMEXPHOST_H_
