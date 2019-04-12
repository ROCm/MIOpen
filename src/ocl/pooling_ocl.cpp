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
#include <miopen/kernel_cache.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/pooling.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/datatype.hpp>

namespace miopen {

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
    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, yDesc, y);
        }
    }

    mlo_construct_pooling2D construct_params(1); // forward

    construct_params.setStream(&handle);

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut)                         = tien<4>(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tien<4>(yDesc.GetStrides());

    construct_params.setTopDescFromMLDesc(yDesc);

    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tien<4>(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tien<4>(xDesc.GetStrides());

    auto index_max = get_index_max(GetIndexType());

    // for kernel implementation max pooling backward pass,
    //   "index_max" means ghost, and thus should not be reached
    if(save_index && GetMode() == miopenPoolingMax && !(index_max >= lens[0] * lens[1]))
    {
        MIOPEN_THROW("Index range not enough for max pooling bwd");
    }

    construct_params.setBotDescFromMLDesc(xDesc);

    if(mode == miopenPoolingMax && save_index && workSpace == nullptr)
    {
        throw std::invalid_argument(
            "workSpace cannot be NULL in Forward Pooling MAX mode when backward pass is requested");
    }
    int pooling_method =
        (mode == miopenPoolingMax)
            ? MLO_POOLING_OP_MAX
            : ((mode == miopenPoolingAverage) ? MLO_POOLING_OP_AVE : MLO_POOLING_OP_AVE_INCLUSIVE);
    construct_params.setPoolingDescr(
        pooling_method, GetIndexType(), lens[0], lens[1], pads[0], pads[1], strides[0], strides[1]);

    std::string network_config =
        std::to_string(pooling_method) + std::to_string(static_cast<int>(save_index)) +
        std::to_string(xDesc.GetType()) + std::to_string(nInStride) + std::to_string(nOutStride) +
        std::to_string(nIn) + std::to_string(nOut) + std::to_string(nInStride) +
        std::to_string(nOutStride) + std::to_string(cIn) + std::to_string(cOut) +
        std::to_string(cInStride) + std::to_string(cOutStride) + std::to_string(hIn) +
        std::to_string(hOut) + std::to_string(hInStride) + std::to_string(hOutStride) +
        std::to_string(lens[0]) + std::to_string(lens[1]) + std::to_string(strides[0]) +
        std::to_string(strides[1]) + std::to_string(pads[0]) + std::to_string(pads[1]) +
        std::to_string(GetIndexType());

    std::string algo_name = "miopenPooling2dForward";
    // printf("Pooling forward network_config: %s\n", network_config.c_str());
    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(x, y, workSpace);
    }
    else
    {
        construct_params.doBackward(save_index);

        mloConstruct(construct_params);
        std::string parms              = construct_params.getCompilerOptions(); // kernel parameters
        std::string program_name       = construct_params.getKernelFile(); // CL kernel filename
        std::string kernel_name        = construct_params.getKernelName(); // kernel name
        const std::vector<size_t>& vld = construct_params.getLocalWkSize();
        const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, y, workSpace);
    }
    if(miopen::CheckNumericsEnabled() != 0)
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
                                           ConstData_t workSpace) const
{

    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled() != 0)
    {
        // miopen::checkNumericsInput(handle, yDesc, y); // not actually used?
        miopen::checkNumericsInput(handle, dyDesc, dy);
        // miopen::checkNumericsInput(handle, xDesc, x); // not actually used?
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, dxDesc, dx);
        }
    }

    miopenStatus_t status = miopenStatusSuccess;
    mlo_construct_pooling2D construct_params(0); // backward

    construct_params.setStream(&handle);

    int ndOut;
    int cdOut;
    int hdOut;
    int wdOut;
    int ndOutStride;
    int cdOutStride;
    int hdOutStride;
    int wdOutStride;

    std::tie(ndOut, cdOut, hdOut, wdOut)                         = tien<4>(dyDesc.GetLengths());
    std::tie(ndOutStride, cdOutStride, hdOutStride, wdOutStride) = tien<4>(dyDesc.GetStrides());

    construct_params.setTopDfDescFromMLDesc(dyDesc);

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut)                         = tien<4>(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tien<4>(yDesc.GetStrides());

    construct_params.setTopDescFromMLDesc(yDesc);

    int ndIn;
    int cdIn;
    int hdIn;
    int wdIn;
    int ndInStride;
    int cdInStride;
    int hdInStride;
    int wdInStride;

    std::tie(ndIn, cdIn, hdIn, wdIn)                         = tien<4>(dxDesc.GetLengths());
    std::tie(ndInStride, cdInStride, hdInStride, wdInStride) = tien<4>(dxDesc.GetStrides());

    construct_params.setBotDfDescFromMLDesc(dxDesc);

    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tien<4>(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tien<4>(xDesc.GetStrides());

    auto index_max = get_index_max(GetIndexType());

    // for kernel implementation max pooling backward pass,
    //   "index_max" means ghost, and thus should not be reached
    if(GetMode() == miopenPoolingMax && !(index_max >= lens[0] * lens[1]))
    {
        MIOPEN_THROW("Index range not enough for max pooling bwd");
    }

    construct_params.setBotDescFromMLDesc(xDesc);

    if(mode == miopenPoolingMax && workSpace == nullptr)
    {
        throw std::invalid_argument("workSpace cannot be NULL in Backward Pooling MAX mode");
    }
    int pooling_method =
        (mode == miopenPoolingMax)
            ? MLO_POOLING_OP_MAX
            : ((mode == miopenPoolingAverage) ? MLO_POOLING_OP_AVE : MLO_POOLING_OP_AVE_INCLUSIVE);
    construct_params.setPoolingDescr(
        pooling_method, GetIndexType(), lens[0], lens[1], pads[0], pads[1], strides[0], strides[1]);

    std::string network_config =
        std::to_string(pooling_method) + std::to_string(xDesc.GetType()) +
        std::to_string(nInStride) + std::to_string(nOutStride) + std::to_string(nIn) +
        std::to_string(nOut) + std::to_string(nInStride) + std::to_string(nOutStride) +
        std::to_string(cIn) + std::to_string(cOut) + std::to_string(cInStride) +
        std::to_string(cOutStride) + std::to_string(hIn) + std::to_string(hOut) +
        std::to_string(hInStride) + std::to_string(hOutStride) + std::to_string(lens[0]) +
        std::to_string(lens[1]) + std::to_string(strides[0]) + std::to_string(strides[1]) +
        std::to_string(pads[0]) + std::to_string(pads[1]) + std::to_string(GetIndexType());
    // printf("Pooling backward network_config: %s\n", network_config.c_str());
    std::string algo_name = "miopenPooling2dBackward";

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        if(mode == miopenPoolingMax)
        {
            kernels.front()(dy, dx, workSpace);
        }
        else
        {
            kernels.front()(dy, dx);
        }
    }
    else
    {

        mloConstruct(construct_params);
        const std::vector<size_t>& vld = construct_params.getLocalWkSize();
        const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();
        std::string program_name       = construct_params.getKernelFile(); // CL kernel filename
        std::string kernel_name        = construct_params.getKernelName(); // kernel name
        std::string parms              = construct_params.getCompilerOptions(); // kernel parameters
        auto k =
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms);

        if(mode == miopenPoolingMax)
        {
            k(dy, dx, workSpace);
        }
        else
        {
            k(dy, dx);
        }
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }

    return (status);
}
} // namespace miopen
