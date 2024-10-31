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

#include <miopen/cartesianprod.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

static void
LogCmdCartesianProd(const miopenTensorDescriptor_t* iDescs, const int32_t iCount, const bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(iDescs[0]).GetType();
        if(dtype == miopenHalf)
        {
            ss << "cartesianprodfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "cartesianprodfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "cartesianprodbfp16";
        }
        ss << " -n " << iCount;
        ss << " -Is ";
        for(int i = 0; i < iCount; i++)
        {
            auto iDesc = iDescs[i];
            ss << miopen::deref(iDesc).GetLengths();
        }
        ss << " -F " << ((is_fwd) ? "1" : "2");
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetCartesianProdForwardWorkspaceSize(miopenHandle_t handle,
                                           const size_t inputCount,
                                           const miopenTensorDescriptor_t* inputDescs,
                                           const miopenTensorDescriptor_t outputDesc,
                                           size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, inputCount, inputDescs, outputDesc, sizeInBytes);
    return miopen::try_([&] {
        std::vector<miopen::TensorDescriptor*> inputDescsCast;
        std::transform(inputDescs,
                       inputDescs + inputCount,
                       std::back_inserter(inputDescsCast),
                       [](const auto& inputDesc) { return &miopen::deref(inputDesc); });
        miopen::deref(sizeInBytes) = miopen::cartesianprod::GetCartesianProdForwardWorkspaceSize(
            miopen::deref(handle), inputCount, inputDescsCast.data(), miopen::deref(outputDesc));
    });
}

extern "C" miopenStatus_t miopenCartesianProdForward(miopenHandle_t handle,
                                                     void* workspace,
                                                     const size_t workspaceSizeInBytes,
                                                     const size_t inputCount,
                                                     const miopenTensorDescriptor_t* inputDescs,
                                                     const void* const* inputs,
                                                     const miopenTensorDescriptor_t outputDesc,
                                                     void* output)
{
    MIOPEN_LOG_FUNCTION(
        handle, workspace, workspaceSizeInBytes, inputDescs, inputs, outputDesc, output);
    LogCmdCartesianProd(inputDescs, inputCount, true);
    std::vector<ConstData_t> inputCast;
    std::vector<miopen::TensorDescriptor*> inputDescsCast;
    std::transform(inputDescs,
                   inputDescs + inputCount,
                   std::back_inserter(inputDescsCast),
                   [](const auto& inputDesc) { return &miopen::deref(inputDesc); });
    std::transform(inputs,
                   inputs + inputCount,
                   std::back_inserter(inputCast),
                   [](const void* input) { return DataCast(input); });
    return miopen::try_([&] {
        miopen::cartesianprod::CartesianProdForward(miopen::deref(handle),
                                                    DataCast(workspace),
                                                    workspaceSizeInBytes,
                                                    inputCount,
                                                    inputDescsCast.data(),
                                                    inputCast.data(),
                                                    miopen::deref(outputDesc),
                                                    DataCast(output));
    });
}

extern "C" miopenStatus_t
miopenCartesianProdBackward(miopenHandle_t handle,
                            const size_t inputCount,
                            const miopenTensorDescriptor_t outputGradDesc,
                            const void* output_grad,
                            const miopenTensorDescriptor_t* inputGradDescs,
                            void** input_grads)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputCount, outputGradDesc, output_grad, inputGradDescs, input_grads);
    LogCmdCartesianProd(inputGradDescs, inputCount, false);
    return miopen::try_([&] {
        std::vector<Data_t> inputGradCast;
        std::vector<miopen::TensorDescriptor*> inputGradDescsCast;
        std::transform(inputGradDescs,
                       inputGradDescs + inputCount,
                       std::back_inserter(inputGradDescsCast),
                       [](const auto& inputGradDesc) { return &miopen::deref(inputGradDesc); });
        std::transform(input_grads,
                       input_grads + inputCount,
                       std::back_inserter(inputGradCast),
                       [](void* input_grad) { return DataCast(input_grad); });
        miopen::cartesianprod::CartesianProdBackward(miopen::deref(handle),
                                                     inputCount,
                                                     miopen::deref(outputGradDesc),
                                                     DataCast(output_grad),
                                                     inputGradDescsCast.data(),
                                                     inputGradCast.data());
    });
}
