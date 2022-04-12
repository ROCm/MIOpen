/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/ctc.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/stringutils.hpp>
#include <vector>

extern "C" miopenStatus_t miopenCreateCTCLossDescriptor(miopenCTCLossDescriptor_t* ctcLossDesc)
{
    MIOPEN_LOG_FUNCTION(ctcLossDesc);
    return miopen::try_([&] { miopen::deref(ctcLossDesc) = new miopen::CTCLossDescriptor(); });
}

extern "C" miopenStatus_t miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc)
{
    MIOPEN_LOG_FUNCTION(ctcLossDesc);
    return miopen::try_([&] { miopen_destroy_object(ctcLossDesc); });
}

extern "C" miopenStatus_t miopenGetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t* dataType,
                                                     int* blank_label_id       = nullptr,
                                                     bool* apply_softmax_layer = nullptr)
{
    MIOPEN_LOG_FUNCTION(ctcLossDesc, dataType, blank_label_id, apply_softmax_layer);
    return miopen::try_([&] {
        miopen::deref(dataType) = miopen::deref(ctcLossDesc).dataType;
        if(blank_label_id != nullptr)
            *blank_label_id = miopen::deref(ctcLossDesc).blank_label_id;
        if(apply_softmax_layer != nullptr)
            *apply_softmax_layer = miopen::deref(ctcLossDesc).apply_softmax_layer;
    });
}

extern "C" miopenStatus_t miopenSetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t dataType,
                                                     const int blank_label_id = 0,
                                                     bool apply_softmax_layer = true)
{
    MIOPEN_LOG_FUNCTION(ctcLossDesc, dataType, blank_label_id, apply_softmax_layer);
    return miopen::try_([&] {
        miopen::deref(ctcLossDesc).dataType            = dataType;
        miopen::deref(ctcLossDesc).blank_label_id      = blank_label_id;
        miopen::deref(ctcLossDesc).apply_softmax_layer = apply_softmax_layer;
    });
}

extern "C" miopenStatus_t
miopenGetCTCLossWorkspaceSize(miopenHandle_t handle,
                              const miopenTensorDescriptor_t probsDesc,
                              const miopenTensorDescriptor_t gradientsDesc,
                              const int* labels,
                              const int* labelLengths,
                              const int* inputLengths,
                              miopenCTCLossAlgo_t algo,
                              const miopenCTCLossDescriptor_t ctcLossDesc,
                              size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(probsDesc,
                        gradientsDesc,
                        labels,
                        labelLengths,
                        inputLengths,
                        algo,
                        ctcLossDesc,
                        workSpaceSize);

    return miopen::try_([&] {
        miopen::deref(workSpaceSize) = miopen::deref(ctcLossDesc)
                                           .GetCTCLossWorkspaceSize(miopen::deref(handle),
                                                                    miopen::deref(probsDesc),
                                                                    miopen::deref(gradientsDesc),
                                                                    labels,
                                                                    labelLengths,
                                                                    inputLengths,
                                                                    algo);
    });
}

extern "C" miopenStatus_t miopenCTCLoss(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t probsDesc,
                                        const void* probs,
                                        const int* labels,
                                        const int* labelLengths,
                                        const int* inputLengths,
                                        void* losses,
                                        const miopenTensorDescriptor_t gradientsDesc,
                                        void* gradients,
                                        miopenCTCLossAlgo_t algo,
                                        const miopenCTCLossDescriptor_t ctcLossDesc,
                                        void* workSpace,
                                        size_t workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(probsDesc,
                        probs,
                        labels,
                        labelLengths,
                        inputLengths,
                        losses,
                        gradientsDesc,
                        gradients,
                        algo,
                        ctcLossDesc,
                        workSpace,
                        workSpaceSize);

    // bfloat16 not supported for ctc operation
    if(miopen::deref(probsDesc).GetType() == miopenBFloat16 ||
       miopen::deref(gradientsDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        // clang-format off
        ss << " ctc "
           << " -a " << algo
           << " -b " << miopen::deref(ctcLossDesc).blank_label_id
           << " -c " << miopen::deref(probsDesc).GetLengths()[2] - 1
           << " -m " << miopen::deref(ctcLossDesc).apply_softmax_layer
           << " -n " << miopen::deref(probsDesc).GetLengths()[1];
        // clang-format on
        auto merge_vec = [&](const int* lens) {
            auto batch_sz = miopen::deref(probsDesc).GetLengths()[1];
            std::vector<std::string> inputs(batch_sz);
            for(std::size_t idx = 0; idx < batch_sz; ++idx)
                inputs[idx] = std::to_string(lens[idx]);
            return inputs;
        };
        if(labelLengths != nullptr)
            ss << " -l " << miopen::JoinStrings(merge_vec(labelLengths), ",");

        if(inputLengths != nullptr)
            ss << " -k " << miopen::JoinStrings(merge_vec(inputLengths), ",");
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }

    return miopen::try_([&] {
        miopen::deref(ctcLossDesc)
            .CTCLoss(miopen::deref(handle),
                     miopen::deref(probsDesc),
                     DataCast(probs),
                     labels,
                     labelLengths,
                     inputLengths,
                     DataCast(losses),
                     miopen::deref(gradientsDesc),
                     DataCast(gradients),
                     algo,
                     DataCast(workSpace),
                     workSpaceSize);
    });
}
