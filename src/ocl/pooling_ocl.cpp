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

namespace miopen {

std::size_t PoolingDescriptor::GetWorkSpaceSize(const TensorDescriptor& tensorDesc) const
{
    return tensorDesc.GetElementSize() * sizeof(uint8_t);
}

miopenStatus_t PoolingDescriptor::Forward(Handle& handle,
                                          const void* /*alpha*/,
                                          const TensorDescriptor& xDesc,
                                          ConstData_t x,
                                          const void* /*beta*/,
                                          const TensorDescriptor& yDesc,
                                          Data_t y,
                                          bool do_backward,
                                          Data_t workSpace,
                                          size_t /*workSpaceSize*/) const
{

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

    std::tie(nOut, cOut, hOut, wOut)                         = tie4(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tie4(yDesc.GetStrides());

    construct_params.setTopDescr(
        "NCHW", "FP32", nOut, cOut, hOut, wOut, nOutStride, cOutStride, hOutStride, wOutStride);
    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tie4(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tie4(xDesc.GetStrides());

    if(((lens[0] * lens[1]) >= std::numeric_limits<uint8_t>::max()) && do_backward)
    {
        MIOPEN_THROW("Pooling window too large to do backwards");
    }

    construct_params.setBotDescr(
        "NCHW", "FP32", nIn, cIn, hIn, wIn, nInStride, cInStride, hInStride, wInStride);

    if(mode == miopenPoolingMax && do_backward && workSpace == nullptr)
    {
        throw std::invalid_argument(
            "workSpace cannot be NULL in Forward Pooling MAX mode when backward pass is requested");
    }
    int pooling_method = (mode == miopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;
    construct_params.setPoolingDescr(
        pooling_method, lens[0], lens[1], pads[0], pads[1], strides[0], strides[1]);

    construct_params.doBackward(do_backward);

    construct_params.mloConstruct();

    std::string program_name = construct_params.getKernelFile();      // CL kernel filename
    std::string kernel_name  = construct_params.getKernelName();      // kernel name
    std::string parms        = construct_params.getCompilerOptions(); // kernel parameters

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    const std::vector<size_t>& vld = construct_params.getLocalWkSize();
    const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

    handle.GetKernel("miopenPooling2dDForward", "", program_name, kernel_name, vld, vgd, parms)(
        x, y, workSpace);

    return miopenStatusSuccess;
}

miopenStatus_t PoolingDescriptor::Backward(Handle& handle,
                                           const void* /*alpha*/,
                                           const TensorDescriptor& yDesc,
                                           ConstData_t /*y*/,
                                           const TensorDescriptor& dyDesc,
                                           ConstData_t dy,
                                           const TensorDescriptor& xDesc,
                                           ConstData_t /*x*/,
                                           const void* /*beta*/,
                                           const TensorDescriptor& dxDesc,
                                           Data_t dx,
                                           ConstData_t workSpace) const
{

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

    std::tie(ndOut, cdOut, hdOut, wdOut)                         = tie4(dyDesc.GetLengths());
    std::tie(ndOutStride, cdOutStride, hdOutStride, wdOutStride) = tie4(dyDesc.GetStrides());

    construct_params.setTopDfDescr("NCHW",
                                   "FP32",
                                   ndOut,
                                   cdOut,
                                   hdOut,
                                   wdOut,
                                   ndOutStride,
                                   cdOutStride,
                                   hdOutStride,
                                   wdOutStride);

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut)                         = tie4(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tie4(yDesc.GetStrides());

    construct_params.setTopDescr(
        "NCHW", "FP32", nOut, cOut, hOut, wOut, nOutStride, cOutStride, hOutStride, wOutStride);

    int ndIn;
    int cdIn;
    int hdIn;
    int wdIn;
    int ndInStride;
    int cdInStride;
    int hdInStride;
    int wdInStride;

    std::tie(ndIn, cdIn, hdIn, wdIn)                         = tie4(dxDesc.GetLengths());
    std::tie(ndInStride, cdInStride, hdInStride, wdInStride) = tie4(dxDesc.GetStrides());

    construct_params.setBotDfDescr(
        "NCHW", "FP32", ndIn, cdIn, hdIn, wdIn, ndInStride, cdInStride, hdInStride, wdInStride);

    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tie4(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tie4(xDesc.GetStrides());

    if(((lens[0] * lens[1]) >= std::numeric_limits<uint8_t>::max()))
    {
        MIOPEN_THROW("Pooling window too large to do backwards");
    }

    construct_params.setBotDescr(
        "NCHW", "FP32", nIn, cIn, hIn, wIn, nInStride, cInStride, hInStride, wInStride);

    if(mode == miopenPoolingMax && workSpace == nullptr)
    {
        throw std::invalid_argument("workSpace cannot be NULL in Backward Pooling MAX mode");
    }
    int pooling_method = (mode == miopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;
    construct_params.setPoolingDescr(
        pooling_method, lens[0], lens[1], pads[0], pads[1], strides[0], strides[1]);

    status = static_cast<miopenStatus_t>(construct_params.mloConstruct());

    std::string program_name = construct_params.getKernelFile();      // CL kernel filename
    std::string kernel_name  = construct_params.getKernelName();      // kernel name
    std::string parms        = construct_params.getCompilerOptions(); // kernel parameters

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    const std::vector<size_t>& vld = construct_params.getLocalWkSize();
    const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

    // Compile the kernel if not aleady compiled
    auto k =
        handle.GetKernel("miopenPooling2dBackward", "", program_name, kernel_name, vld, vgd, parms);

    // Set kernel arguments
    // Use proper arguments
    if(mode == miopenPoolingMax)
    {
        k(dy, dx, workSpace);
    }
    else
    {
        k(dy, dx);
    }

    return (status);
}
} // namespace miopen
