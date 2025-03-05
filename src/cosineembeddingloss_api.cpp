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
#include <miopen/cosineembeddingloss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

inline void LogCmdCosineEmbeddingLoss(const miopenTensorDescriptor_t x1Desc,
                                      const miopenTensorDescriptor_t x2Desc,
                                      const miopenTensorDescriptor_t tDesc,
                                      bool is_fwd,
                                      const miopenLossReductionMode_t reduction)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(x1Desc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "cosineembeddinglossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "cosineembeddingloss";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "cosineembeddinglossbfp16";
        }

        MIOPEN_LOG_FUNCTION(x1Desc, x2Desc, tDesc);
        ss << " -D " << miopen::deref(x1Desc).GetLengths();
        ss << " -Si1 " << miopen::deref(x1Desc).GetStrides();
        ss << " -Si2 " << miopen::deref(x2Desc).GetStrides();
        ss << " -St " << miopen::deref(tDesc).GetStrides();

        ss << " -F " << ((is_fwd) ? "1" : "2");
        ss << " -R " << reduction;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetCosineEmbeddingLossForwardWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t input1Desc,
                                                 const miopenTensorDescriptor_t input2Desc,
                                                 const miopenTensorDescriptor_t targetDesc,
                                                 const miopenTensorDescriptor_t outputDesc,
                                                 const float margin,
                                                 size_t* sizeInBytes,
                                                 const miopenLossReductionMode_t reduction)
{

    MIOPEN_LOG_FUNCTION(
        handle, input1Desc, input2Desc, targetDesc, outputDesc, margin, sizeInBytes, reduction);

    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        return miopen::try_([&] {
            miopen::deref(sizeInBytes) =
                miopen::cosineembeddingloss::GetCosineEmbeddingLossUnreducedForwardWorkspaceSize(
                    miopen::deref(handle),
                    miopen::deref(input1Desc),
                    miopen::deref(input2Desc),
                    miopen::deref(targetDesc),
                    miopen::deref(outputDesc),
                    margin);
        });
    }
    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::cosineembeddingloss::GetCosineEmbeddingLossReducedForwardWorkspaceSize(
                miopen::deref(handle),
                miopen::deref(input1Desc),
                miopen::deref(input2Desc),
                miopen::deref(targetDesc),
                miopen::deref(outputDesc),
                margin);
    });
}

extern "C" miopenStatus_t
miopenCosineEmbeddingLossForward(miopenHandle_t handle,
                                 void* workspace,
                                 size_t workspaceSizeInBytes,
                                 const miopenTensorDescriptor_t input1Desc,
                                 const void* input1,
                                 const miopenTensorDescriptor_t input2Desc,
                                 const void* input2,
                                 const miopenTensorDescriptor_t targetDesc,
                                 const void* target,
                                 const miopenTensorDescriptor_t outputDesc,
                                 void* output,
                                 const float margin,
                                 const miopenLossReductionMode_t reduction)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        input1Desc,
                        input1,
                        input2Desc,
                        input2,
                        targetDesc,
                        target,
                        outputDesc,
                        output,
                        margin,
                        reduction);

    LogCmdCosineEmbeddingLoss(input1Desc, input2Desc, targetDesc, true, reduction);
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        return miopen::try_([&] {
            miopen::cosineembeddingloss::CosineEmbeddingLossUnreducedForward(
                miopen::deref(handle),
                DataCast(workspace),
                workspaceSizeInBytes,
                miopen::deref(input1Desc),
                DataCast(input1),
                miopen::deref(input2Desc),
                DataCast(input2),
                miopen::deref(targetDesc),
                DataCast(target),
                miopen::deref(outputDesc),
                DataCast(output),
                margin);
        });
    }
    return miopen::try_([&] {
        miopen::cosineembeddingloss::CosineEmbeddingLossReducedForward(miopen::deref(handle),
                                                                       DataCast(workspace),
                                                                       workspaceSizeInBytes,
                                                                       miopen::deref(input1Desc),
                                                                       DataCast(input1),
                                                                       miopen::deref(input2Desc),
                                                                       DataCast(input2),
                                                                       miopen::deref(targetDesc),
                                                                       DataCast(target),
                                                                       miopen::deref(outputDesc),
                                                                       DataCast(output),
                                                                       margin,
                                                                       reduction);
    });
}

extern "C" miopenStatus_t
miopenGetCosineEmbeddingLossBackwardWorkspaceSize(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t input1Desc,
                                                  const miopenTensorDescriptor_t input2Desc,
                                                  const miopenTensorDescriptor_t targetDesc,
                                                  const miopenTensorDescriptor_t outputGradDesc,
                                                  const miopenTensorDescriptor_t input1GradDesc,
                                                  const miopenTensorDescriptor_t input2GradDesc,
                                                  const float margin,
                                                  size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle,
                        input1Desc,
                        input2Desc,
                        targetDesc,
                        outputGradDesc,
                        input1GradDesc,
                        input2GradDesc,
                        margin,
                        sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::cosineembeddingloss::GetCosineEmbeddingLossBackwardWorkspaceSize(
                miopen::deref(handle),
                miopen::deref(input1Desc),
                miopen::deref(input2Desc),
                miopen::deref(targetDesc),
                miopen::deref(outputGradDesc),
                miopen::deref(input1GradDesc),
                miopen::deref(input2GradDesc),
                margin);
    });
}

extern "C" miopenStatus_t
miopenCosineEmbeddingLossBackward(miopenHandle_t handle,
                                  void* workspace,
                                  size_t workspaceSizeInBytes,
                                  const miopenTensorDescriptor_t input1Desc,
                                  const void* input1,
                                  const miopenTensorDescriptor_t input2Desc,
                                  const void* input2,
                                  const miopenTensorDescriptor_t targetDesc,
                                  const void* target,
                                  const miopenTensorDescriptor_t outputGradDesc,
                                  const void* output_grad,
                                  const miopenTensorDescriptor_t input1GradDesc,
                                  void* input1_grad,
                                  const miopenTensorDescriptor_t input2GradDesc,
                                  void* input2_grad,
                                  const float margin,
                                  const miopenLossReductionMode_t reduction)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        input1Desc,
                        input1,
                        input2Desc,
                        input2,
                        targetDesc,
                        target,
                        outputGradDesc,
                        output_grad,
                        input1GradDesc,
                        input1_grad,
                        input2GradDesc,
                        input2_grad,
                        margin,
                        reduction);

    LogCmdCosineEmbeddingLoss(input1Desc, input2Desc, targetDesc, false, reduction);

    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        return miopen::try_([&] {
            miopen::cosineembeddingloss::CosineEmbeddingLossUnreducedBackward(
                miopen::deref(handle),
                DataCast(workspace),
                workspaceSizeInBytes,
                miopen::deref(input1Desc),
                DataCast(input1),
                miopen::deref(input2Desc),
                DataCast(input2),
                miopen::deref(targetDesc),
                DataCast(target),
                miopen::deref(outputGradDesc),
                DataCast(output_grad),
                miopen::deref(input1GradDesc),
                DataCast(input1_grad),
                miopen::deref(input2GradDesc),
                DataCast(input2_grad),
                margin);
        });
    }
    return miopen::try_([&] {
        miopen::cosineembeddingloss::CosineEmbeddingLossReducedBackward(
            miopen::deref(handle),
            DataCast(workspace),
            workspaceSizeInBytes,
            miopen::deref(input1Desc),
            DataCast(input1),
            miopen::deref(input2Desc),
            DataCast(input2),
            miopen::deref(targetDesc),
            DataCast(target),
            miopen::deref(outputGradDesc),
            DataCast(output_grad),
            miopen::deref(input1GradDesc),
            DataCast(input1_grad),
            miopen::deref(input2GradDesc),
            DataCast(input2_grad),
            margin,
            reduction);
    });
}
