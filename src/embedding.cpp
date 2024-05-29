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
#include <miopen/embedding.hpp>
#include <miopen/embedding/invoke_params.hpp>
#include <miopen/embedding/solvers.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t EmbeddingForward(Handle& handle,
                                const TensorDescriptor& inputDesc,
                                ConstData_t input,
                                const TensorDescriptor& weightDesc,
                                Data_t weight,
                                const TensorDescriptor& outputDesc,
                                Data_t output,
                                const TensorDescriptor& errorDesc,
                                Data_t error,
                                bool has_max_norm,
                                float max_norm,
                                float norm_type)
{
    const auto problem = embedding::ProblemDescription{
        inputDesc, weightDesc, outputDesc, errorDesc, has_max_norm, max_norm, norm_type};

    const auto invoke_params = [&]() {
        auto tmp = embedding::InvokeParams{};
        tmp.type = InvokeType::Run;

        tmp.inputDesc  = &inputDesc;
        tmp.weightDesc = &weightDesc;
        tmp.outputDesc = &outputDesc;
        tmp.input      = input;
        tmp.weight     = weight;
        tmp.output     = output;
        tmp.error      = error;

        tmp.has_max_norm = has_max_norm;
        tmp.max_norm     = max_norm;
        tmp.norm_type    = norm_type;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"EmbeddingForward"};
    const auto solvers = solver::SolverContainer<solver::embedding::EmbeddingForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t EmbeddingBackward(Handle& handle,
                                 void* workspace,
                                 size_t workspaceSizeInBytes,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& weightGradDesc,
                                 Data_t weight_grad,
                                 const TensorDescriptor& outputGradDesc,
                                 Data_t output_grad,
                                 const TensorDescriptor& errorDesc,
                                 Data_t error,
                                 int64_t padding_idx,
                                 bool scale_grad_by_freq,
                                 bool deterministic_mode)
{
    const auto problem = embedding::ProblemDescription{
        inputDesc, weightGradDesc, outputGradDesc, errorDesc, padding_idx, scale_grad_by_freq};

    const auto invoke_params = [&]() {
        auto tmp = embedding::InvokeParams{};
        tmp.type = InvokeType::Run;

        tmp.inputDesc      = &inputDesc;
        tmp.weightGradDesc = &weightGradDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.input          = input;
        tmp.weight_grad    = weight_grad;
        tmp.output_grad    = output_grad;
        tmp.error          = error;
        tmp.indices_freq   = workspace;

        tmp.padding_idx        = padding_idx;
        tmp.scale_grad_by_freq = scale_grad_by_freq;
        tmp.deterministic_mode = deterministic_mode;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"EmbeddingBackward"};
    const auto solvers = solver::SolverContainer<solver::embedding::EmbeddingBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}
} // namespace miopen
