/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/layernorm.hpp>
#include <miopen/layernorm/invoke_params.hpp>
#include <miopen/layernorm/solvers.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t LayerNormForward(const Handle& handle,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& weightDesc,
                                ConstData_t weight,
                                const TensorDescriptor& biasDesc,
                                ConstData_t bias,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                const TensorDescriptor& meanDesc,
                                Data_t mean,
                                const TensorDescriptor& rstdDesc,
                                Data_t rstd,
                                miopenNormMode_t mode,
                                float epsilon,
                                int32_t normalized_dim)
{
    const auto problem = layernorm::ProblemDescription{
        mode, xDesc, weightDesc, biasDesc, yDesc, meanDesc, rstdDesc, epsilon, normalized_dim};

    const auto invoke_params = [&]() {
        auto tmp           = layernorm::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.xDesc          = &xDesc;
        tmp.x              = x;
        tmp.weight         = weight;
        tmp.bias           = bias;
        tmp.y              = y;
        tmp.mean           = mean;
        tmp.rstd           = rstd;
        tmp.epsilon        = epsilon;
        tmp.normalized_dim = normalized_dim;
        tmp.mode           = mode;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"LayerNormForward"};
    const auto solvers = solver::SolverContainer<solver::layernorm::Layernorm2DCKForward,
                                                 solver::layernorm::Layernorm4DCKForward,
                                                 solver::layernorm::LayernormForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

std::size_t GetLayerNormBackwardWorkspaceSize(const Handle& handle,
                                              const TensorDescriptor& dyDesc,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& weightDesc,
                                              const TensorDescriptor& meanDesc,
                                              const TensorDescriptor& rstdDesc,
                                              const TensorDescriptor& dxDesc,
                                              const TensorDescriptor& dwDesc,
                                              const TensorDescriptor& dbDesc,
                                              miopenNormMode_t mode,
                                              int32_t normalized_dim)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = layernorm::ProblemDescription{mode,
                                                       dyDesc,
                                                       xDesc,
                                                       weightDesc,
                                                       meanDesc,
                                                       rstdDesc,
                                                       dxDesc,
                                                       dwDesc,
                                                       dbDesc,
                                                       normalized_dim};

    const auto algo    = AlgorithmName{"LayerNormBackward"};
    const auto solvers = solver::SolverContainer<solver::layernorm::LayernormBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem, true);

    return pair_size_vector.empty() ? static_cast<size_t>(0) : pair_size_vector.front().second;
}

miopenStatus_t LayerNormBackward(const Handle& handle,
                                 Data_t workspace,
                                 size_t workspaceSizeInBytes,
                                 const TensorDescriptor& dyDesc,
                                 ConstData_t dy,
                                 const TensorDescriptor& xDesc,
                                 ConstData_t x,
                                 const TensorDescriptor& weightDesc,
                                 ConstData_t weight,
                                 const TensorDescriptor& meanDesc,
                                 ConstData_t mean,
                                 const TensorDescriptor& rstdDesc,
                                 ConstData_t rstd,
                                 const TensorDescriptor& dxDesc,
                                 Data_t dx,
                                 const TensorDescriptor& dwDesc,
                                 Data_t dw,
                                 const TensorDescriptor& dbDesc,
                                 Data_t db,
                                 miopenNormMode_t mode,
                                 int32_t normalized_dim)
{
    const auto problem = layernorm::ProblemDescription{mode,
                                                       dyDesc,
                                                       xDesc,
                                                       weightDesc,
                                                       meanDesc,
                                                       rstdDesc,
                                                       dxDesc,
                                                       dwDesc,
                                                       dbDesc,
                                                       normalized_dim};

    const auto invoke_params = [&]() {
        auto tmp      = layernorm::BwdInvokeParams{};
        tmp.type      = InvokeType::Run;
        tmp.dyDesc    = &dyDesc;
        tmp.workspace = workspace, tmp.workspace_size = workspaceSizeInBytes;
        tmp.dy             = dy;
        tmp.x              = x;
        tmp.weight         = weight;
        tmp.mean           = mean;
        tmp.rstd           = rstd;
        tmp.dx             = dx;
        tmp.dw             = dw;
        tmp.db             = db;
        tmp.mode           = mode;
        tmp.normalized_dim = normalized_dim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"LayerNormBackward"};
    const auto solvers = solver::SolverContainer<solver::layernorm::LayernormBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
