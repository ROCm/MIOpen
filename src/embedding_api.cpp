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
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdEmbedding(const miopenTensorDescriptor_t inputDesc,
                            const miopenTensorDescriptor_t weightDesc,
                            const bool is_fwd,
                            const bool has_max_norm,
                            const float max_norm,
                            const float norm_type,
                            const int64_t padding_idx,
                            const bool scale_grad_by_freq,
                            const bool deterministic_mode)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        ss << "embedding";
        ss << (is_fwd ? "fwd" : "bwd");

        if(dtype == miopenHalf)
        {
            ss << "fp16";
        }

        std::string input_sz, weight_sz;
        auto input_dims  = miopen::deref(inputDesc).GetLengths();
        auto weight_dims = miopen::deref(weightDesc).GetLengths();
        for(auto dim : input_dims)
        {
            input_sz += std::to_string(dim);
            input_sz += "x";
        }
        for(auto dim : weight_dims)
        {
            weight_sz += std::to_string(dim);
            weight_sz += "x";
        }
        input_sz.pop_back();
        weight_sz.pop_back();
        ss << " -I " << input_sz << " -W " << weight_sz;
        if(is_fwd)
            ss << " -f 1 "
               << " -m " << max_norm << " -n " << norm_type;
        else
            ss << " -f 0 "
               << " -p " << padding_idx << " -s " << scale_grad_by_freq << " -d "
               << deterministic_mode;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenEmbeddingForward(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t inputDesc,
                                                 const void* input,
                                                 const miopenTensorDescriptor_t outputDesc,
                                                 void* output,
                                                 const miopenTensorDescriptor_t weightDesc,
                                                 void* weight,
                                                 const miopenTensorDescriptor_t errorDesc,
                                                 void* error,
                                                 bool has_max_norm,
                                                 float max_norm,
                                                 float norm_type)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        outputDesc,
                        output,
                        weightDesc,
                        weight,
                        errorDesc,
                        error,
                        has_max_norm,
                        max_norm,
                        norm_type);

    LogCmdEmbedding(
        inputDesc, weightDesc, has_max_norm, true, max_norm, norm_type, 0, false, false);

    return miopen::try_([&] {
        miopen::EmbeddingForward(miopen::deref(handle),
                                 miopen::deref(inputDesc),
                                 DataCast(input),
                                 miopen::deref(outputDesc),
                                 DataCast(output),
                                 miopen::deref(weightDesc),
                                 DataCast(weight),
                                 miopen::deref(errorDesc),
                                 DataCast(error),
                                 has_max_norm,
                                 max_norm,
                                 norm_type);
    });
}

extern "C" miopenStatus_t miopenEmbeddingBackward(miopenHandle_t handle,
                                                  void* workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputGradDesc,
                                                  const void* outputGrad,
                                                  const miopenTensorDescriptor_t weightGradDesc,
                                                  void* weightGrad,
                                                  const miopenTensorDescriptor_t errorDesc,
                                                  void* error,
                                                  long long padding_idx,
                                                  bool scale_grad_by_freq,
                                                  bool deterministic_mode)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        outputGradDesc,
                        outputGrad,
                        weightGradDesc,
                        weightGrad,
                        errorDesc,
                        error,
                        padding_idx,
                        scale_grad_by_freq,
                        deterministic_mode);

    LogCmdEmbedding(inputDesc,
                    weightGradDesc,
                    padding_idx,
                    false,
                    0,
                    0,
                    padding_idx,
                    scale_grad_by_freq,
                    deterministic_mode);

    return miopen::try_([&] {
        miopen::EmbeddingBackward(miopen::deref(handle),
                                  DataCast(workspace),
                                  workspaceSizeInBytes,
                                  miopen::deref(inputDesc),
                                  DataCast(input),
                                  miopen::deref(outputGradDesc),
                                  DataCast(outputGrad),
                                  miopen::deref(weightGradDesc),
                                  DataCast(weightGrad),
                                  miopen::deref(errorDesc),
                                  DataCast(error),
                                  padding_idx,
                                  scale_grad_by_freq,
                                  deterministic_mode);
    });
}
