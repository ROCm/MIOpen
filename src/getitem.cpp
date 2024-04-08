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
#include <miopen/getitem.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>
#include <miopen/item/invoke_params.hpp>
#include <miopen/item/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

std::size_t GetGetitemWorkspaceSize(Handle& handle,
                                    int32_t indexCount,
                                    const TensorDescriptor* const* indexDescs)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = item::ProblemDescription{indexCount, indexDescs};

    const auto algo    = AlgorithmName{"GetitemBackward"};
    const auto solvers = solver::SolverContainer<solver::item::GetitemBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t GetitemBackward(Handle& handle,
                               Data_t workspace,
                               size_t workspaceSizeInBytes,
                               const TensorDescriptor& dyDesc,
                               ConstData_t dy,
                               int32_t indexCount,
                               const TensorDescriptor* const* indexDescs,
                               ConstData_t* indexs,
                               const TensorDescriptor& dxDesc,
                               Data_t dx,
                               const TensorDescriptor& errorDesc,
                               Data_t error,
                               int32_t dimCount,
                               const int32_t* dims,
                               int32_t sliceCount,
                               const int32_t* slices,
                               int32_t offset)
{
    const auto problem = item::ProblemDescription{dyDesc,
                                                  indexCount,
                                                  indexDescs,
                                                  dxDesc,
                                                  errorDesc,
                                                  dimCount,
                                                  dims,
                                                  sliceCount,
                                                  slices,
                                                  offset};

    const auto invoke_params = item::GetitemInvokeParams{workspace,
                                                         workspaceSizeInBytes,
                                                         dyDesc,
                                                         dy,
                                                         indexCount,
                                                         indexDescs,
                                                         indexs,
                                                         dxDesc,
                                                         dx,
                                                         errorDesc,
                                                         error,
                                                         dimCount,
                                                         dims,
                                                         sliceCount,
                                                         slices,
                                                         offset};

    const auto algo    = AlgorithmName{"GetitemBackward"};
    const auto solvers = solver::SolverContainer<solver::item::GetitemBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
