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

#include <math.h>

template <typename Tgpu, typename Tcheck>
int32_t mloPdistBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                const miopenTensorDescriptor_t outputDesc,
                                const miopenTensorDescriptor_t doutputDesc,
                                const miopenTensorDescriptor_t dinputDesc,
                                const Tgpu* input,
                                const Tgpu* output,
                                const Tgpu* doutput,
                                Tcheck* dinputHost,
                                const double p)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto N           = miopen::deref(inputDesc).GetLengths()[0];
    auto M           = miopen::deref(inputDesc).GetLengths()[1];

    // Fill dinputHost with zeros
    for(size_t i = 0; i < input_numel; i++)
    {
        dinputHost[i] = 0;
    }

    auto sign_ = [](double val) { return (0 < val) - (val < 0); };

    auto backward = [&](double diff, double grad, double dist, double p) -> double {
        if(p == 1.f)
        { // one
            return grad * sign_(diff);
        }
        else if(p < 2.f)
        { // lt_two
            return (dist == 0.0 || (diff == 0.0 && p < 1))
                       ? 0
                       : (sign_(diff) * pow(fabs(diff), p - 1) * grad / pow(dist, p - 1));
        }
        else if(p == 2.f)
        { // two
            return dist == 0.0 ? 0 : grad * diff / dist;
        }
        else if(isinf(p))
        { // inf
            // std::cout << "[Outside kernel] Hit here infinity\n";
            // printf("[Outside Kernel] diff: %f, dist: %f, fabs(diff) == dist: %f\n",
            // diff, dist, static_cast<double>(fabs(diff) == dist));
            return grad * sign_(diff) * (fabs(diff) == dist);
        }
        else
        { // general p
            return dist == 0.0 ? 0 : diff * pow(fabs(diff), p - 2) * grad / pow(dist, p - 1);
        }
    };

    // double p_ = static_cast<double>(p);

    // auto N = input.desc.GetLengths()[0];
    // // auto NO = output.desc.GetLengths()[0];
    // auto M = input.desc.GetLengths()[1];

    // int idx = 0;
    for(int i = 0; i < N; ++i)
    {
        for(int j = i + 1; j < N; ++j)
        {
            long k = j + N * i - i * (i + 1) / 2 - i - 1;
            // long k = (2 * N - i - 1) * i / 2 + (j - i - 1);
            // std::cout << "k: " << k << std::endl;
            // T output_k = output[k];
            // T grad_k   = output_grad[k];
            // std::cout << "k: " << k << std::endl;
            double grad_k   = static_cast<double>(doutput[k]);
            double output_k = static_cast<double>(output[k]);

            for(int m = 0; m < M; ++m)
            {
                double input_first  = static_cast<double>(input[i * M + m]);
                double input_second = static_cast<double>(input[j * M + m]);
                // T diff     = input[i * M + m] - input[j * M + m];
                double diff = input_first - input_second;

                // std::cout << "diff: " << diff << std::endl;

                // T res = backward(diff, grad_k, output_k, p);
                Tcheck res = static_cast<Tcheck>(backward(diff, grad_k, output_k, p));

                // std::cout << "res: " << res << std::endl;

                // printf("[Outside kernel] k: %d, i: %d, j: %d, m: %d, res: %f\n", k, i, j, m,
                // res);

                dinputHost[i * M + m] += res;
                dinputHost[j * M + m] -= res;
            }
        }
    }

    return miopenStatusSuccess;
}
