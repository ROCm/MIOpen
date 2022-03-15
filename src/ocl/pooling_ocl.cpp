/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <miopen/pooling.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/pooling/problem_description.hpp>
#include <miopen/pooling/solvers.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/datatype.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/mlo_internal.hpp>

namespace miopen {

static auto PoolingForwardSolvers()
{
    return solver::SolverContainer<solver::pooling::PoolingForward2d,
                                   solver::pooling::PoolingForwardNd,
                                   solver::pooling::TransposedPoolingFwd2d,
                                   solver::pooling::TransposedPoolingFwdNd>{};
}

static auto PoolingBackwardSolvers()
{
    return solver::SolverContainer<solver::pooling::PoolingBackward2d,
                                   solver::pooling::PoolingBackwardNd,
                                   solver::pooling::TransposedPoolingBwd2d,
                                   solver::pooling::TransposedPoolingBwdNd>{};
}

miopenStatus_t PoolingDescriptor::Forward(Handle& handle,
                                          const void* alpha,
                                          const TensorDescriptor& xDesc,
                                          ConstData_t x,
                                          const void* beta,
                                          const TensorDescriptor& yDesc,
                                          Data_t y,
                                          bool save_index,
                                          Data_t workSpace,
                                          size_t /*workSpaceSize*/) const
{

    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, yDesc, y);
        }
    }

    int pool_dim = xDesc.GetSize();
    if(pool_dim != 4 && pool_dim != 5)
    {
        MIOPEN_THROW("Unsupported pooling dimension");
    }

    auto index_max = get_index_max(GetIndexType());

    // for kernel implementation max pooling backward pass,
    //   "index_max" means ghost, and thus should not be reached
    if(mode == miopenPoolingMax && save_index)
    {
        if((workspaceIndexMode == miopenPoolingWorkspaceIndexMask &&
            !(index_max >= std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<int>()))) ||
           (workspaceIndexMode == miopenPoolingWorkspaceIndexImage &&
            !(index_max >= std::accumulate(xDesc.GetLengths().begin() + 2,
                                           xDesc.GetLengths().end(),
                                           1,
                                           std::multiplies<int>()))))
        {
            MIOPEN_THROW("Index range not enough for max pooling bwd");
        }

        if(workspaceIndexMode == miopenPoolingWorkspaceIndexMask && pool_dim == 5)
        {
            MIOPEN_THROW("3D pooling doesn't support workspace index mask mode");
        }

        if(workSpace == nullptr)
        {
            throw std::invalid_argument("workSpace cannot be NULL in Forward Pooling MAX mode when "
                                        "backward pass is requested");
        }
    }

    const auto algo_name =
        AlgorithmName{pool_dim == 5 ? "miopenPoolingNdForward" : "miopenPooling2dForward"};
    const auto problem = pooling::ProblemDescription{*this, xDesc, yDesc, save_index};

    const auto invoke_params = [&]() {
        auto tmp      = pooling::FwdInvokeParams{};
        tmp.type      = InvokeType::Run;
        tmp.xDesc     = xDesc;
        tmp.yDesc     = yDesc;
        tmp.pooling   = *this;
        tmp.x         = x;
        tmp.y         = y;
        tmp.workspace = workSpace;
        return tmp;
    }();

    PoolingForwardSolvers().ExecutePrimitive(handle, problem, algo_name, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }

    return miopenStatusSuccess;
}

miopenStatus_t PoolingDescriptor::Backward(Handle& handle,
                                           const void* alpha,
                                           const TensorDescriptor& yDesc,
                                           ConstData_t /*y*/,
                                           const TensorDescriptor& dyDesc,
                                           ConstData_t dy,
                                           const TensorDescriptor& xDesc,
                                           ConstData_t /*x*/,
                                           const void* beta,
                                           const TensorDescriptor& dxDesc,
                                           Data_t dx,
                                           Data_t workSpace) const
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled())
    {
        // miopen::checkNumericsInput(handle, yDesc, y); // not actually used?
        miopen::checkNumericsInput(handle, dyDesc, dy);
        // miopen::checkNumericsInput(handle, xDesc, x); // not actually used?
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, dxDesc, dx);
        }
    }

    assert(yDesc.GetElementSize() == dyDesc.GetElementSize() &&
           xDesc.GetElementSize() == dxDesc.GetElementSize());

    int pool_dim = dyDesc.GetSize();
    if(pool_dim != 4 && pool_dim != 5)
    {
        MIOPEN_THROW("Unsupported pooling dimension");
    }

    const auto problem = pooling::ProblemDescription{*this, xDesc, yDesc, dxDesc, dyDesc};
    const auto algo_name =
        AlgorithmName{pool_dim == 5 ? "miopenPoolingNdBackward" : "miopenPooling2dBackward"};

    const auto invoke_params = [&]() {
        auto tmp      = pooling::BwdInvokeParams{};
        tmp.type      = InvokeType::Run;
        tmp.dxDesc    = dxDesc;
        tmp.dyDesc    = dyDesc;
        tmp.pooling   = *this;
        tmp.dx        = dx;
        tmp.dy        = dy;
        tmp.workspace = workSpace;
        return tmp;
    }();

    PoolingBackwardSolvers().ExecutePrimitive(handle, problem, algo_name, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }

    return miopenStatusSuccess;
}

} // namespace miopen
