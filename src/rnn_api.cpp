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

#include <miopen/rnn.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <vector>

enum RNNDir_t
{
    ForwardInference,
    ForwardTraining,
    BackwardData,
    BackwardWeights
};

extern "C" miopenStatus_t
miopenSetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t dataType,
                                    miopenRNNBaseLayout_t layout,
                                    int maxSequenceLen,
                                    int batchSize,
                                    int vectorSize,
                                    const int* sequenceLenArray,
                                    void* paddingMarker)
{
    MIOPEN_LOG_FUNCTION(
        seqTensorDesc, dataType, layout, maxSequenceLen, batchSize, vectorSize, sequenceLenArray);

    return miopen::try_([&] {
        miopen::deref(seqTensorDesc) =
            miopen::RNNDescriptor::makeSeqTensorDescriptor(dataType,
                                                           layout,
                                                           maxSequenceLen,
                                                           batchSize,
                                                           vectorSize,
                                                           sequenceLenArray,
                                                           paddingMarker);
    });
}

extern "C" miopenStatus_t
miopenGetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t* dataType,
                                    miopenRNNBaseLayout_t* layout,
                                    int* maxSequenceLen,
                                    int* batchSize,
                                    int* vectorSize,
                                    int sequenceLenArrayLimit,
                                    int* sequenceLenArray,
                                    void* paddingMarker)
{
    MIOPEN_LOG_FUNCTION(seqTensorDesc, sequenceLenArrayLimit);

    const auto status = miopen::try_([&] {
        *dataType = miopen::deref(seqTensorDesc).GetType();
        *layout = miopen::RNNDescriptor::getBaseLayoutFromDataTensor(miopen::deref(seqTensorDesc));

        miopen::tie_deref(batchSize, maxSequenceLen, vectorSize) =
            miopen::tien<3>(miopen::deref(seqTensorDesc).GetLengths());

        if(sequenceLenArray != nullptr)
        {
            size_t copy_sz =
                std::min(static_cast<size_t>(sequenceLenArrayLimit),
                         miopen::deref(seqTensorDesc).GetSequenceLengthsVector().size());
            const auto seq_lens_begin =
                miopen::deref(seqTensorDesc).GetSequenceLengthsVector().begin();
            std::copy_n(seq_lens_begin, copy_sz, sequenceLenArray);
        }
        auto& padding_marker_ptr = miopen::deref(seqTensorDesc).GetPaddingMarkerHolder();
        if(paddingMarker != nullptr && !padding_marker_ptr.empty())
        {
            std::copy(padding_marker_ptr.begin(),
                      padding_marker_ptr.end(),
                      reinterpret_cast<char*>(paddingMarker));
        }
    });

    return status;
}

extern "C" miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc)
{
    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(rnnDesc);
        desc       = new miopen::RNNDescriptor();
    });
}

extern "C" miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc)
{
    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] { miopen_destroy_object(rnnDesc); });
}

extern "C" miopenStatus_t miopenGetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 miopenRNNMode_t* rnnMode,
                                                 miopenRNNAlgo_t* algoMode,
                                                 miopenRNNInputMode_t* inputMode,
                                                 miopenRNNDirectionMode_t* dirMode,
                                                 miopenRNNBiasMode_t* biasMode,
                                                 int* hiddenSize,
                                                 int* layer)
{

    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] {
        if(rnnMode != nullptr)
        {
            miopen::deref(rnnMode) = miopen::deref(rnnDesc).rnnMode;
        }
        if(algoMode != nullptr)
        {
            miopen::deref(algoMode) = miopen::deref(rnnDesc).algoMode;
        }
        if(inputMode != nullptr)
        {
            miopen::deref(inputMode) = miopen::deref(rnnDesc).inputMode;
        }
        if(layer != nullptr)
        {
            miopen::deref(layer) = miopen::deref(rnnDesc).nLayers;
        }
        if(biasMode != nullptr)
        {
            miopen::deref(biasMode) = miopen::deref(rnnDesc).biasMode;
        }
        if(dirMode != nullptr)
        {
            miopen::deref(dirMode) = miopen::deref(rnnDesc).dirMode;
        }
        if(hiddenSize != nullptr)
        {
            miopen::deref(hiddenSize) = miopen::deref(rnnDesc).hsize;
        }
    });
}

extern "C" miopenStatus_t miopenGetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                    int* hiddenSize,
                                                    int* layer,
                                                    miopenDropoutDescriptor_t* dropoutDesc,
                                                    miopenRNNInputMode_t* inputMode,
                                                    miopenRNNDirectionMode_t* dirMode,
                                                    miopenRNNMode_t* rnnMode,
                                                    miopenRNNBiasMode_t* biasMode,
                                                    miopenRNNAlgo_t* algoMode,
                                                    miopenDataType_t* dataType)
{
    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] {
        if(rnnMode != nullptr)
        {
            miopen::deref(rnnMode) = miopen::deref(rnnDesc).rnnMode;
        }
        if(algoMode != nullptr)
        {
            miopen::deref(algoMode) = miopen::deref(rnnDesc).algoMode;
        }
        if(inputMode != nullptr)
        {
            miopen::deref(inputMode) = miopen::deref(rnnDesc).inputMode;
        }
        if(layer != nullptr)
        {
            miopen::deref(layer) = miopen::deref(rnnDesc).nLayers;
        }
        if(biasMode != nullptr)
        {
            miopen::deref(biasMode) = miopen::deref(rnnDesc).biasMode;
        }
        if(dirMode != nullptr)
        {
            miopen::deref(dirMode) = miopen::deref(rnnDesc).dirMode;
        }
        if(hiddenSize != nullptr)
        {
            miopen::deref(hiddenSize) = miopen::deref(rnnDesc).hsize;
        }
        if(dropoutDesc != nullptr)
        {
            miopen::deref(dropoutDesc) = miopen::deref(rnnDesc).dropoutDesc;
        }
        if(dataType != nullptr)
        {
            miopen::deref(dataType) = miopen::deref(rnnDesc).dataType;
        }
    });
}

extern "C" miopenStatus_t miopenSetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 const int hsize,
                                                 const int nlayers,
                                                 miopenRNNInputMode_t inMode,
                                                 miopenRNNDirectionMode_t direction,
                                                 miopenRNNMode_t rnnMode,
                                                 miopenRNNBiasMode_t biasMode,
                                                 miopenRNNAlgo_t algo,
                                                 miopenDataType_t dataType)
{

    MIOPEN_LOG_FUNCTION(
        rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo, dataType);
    return miopen::try_([&] {
        miopen::deref(rnnDesc) = miopen::RNNDescriptor(
            hsize, nlayers, rnnMode, inMode, direction, biasMode, algo, dataType);
    });
}

extern "C" miopenStatus_t miopenSetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                    const int hsize,
                                                    const int nlayers,
                                                    miopenDropoutDescriptor_t dropoutDesc,
                                                    miopenRNNInputMode_t inMode,
                                                    miopenRNNDirectionMode_t direction,
                                                    miopenRNNMode_t rnnMode,
                                                    miopenRNNBiasMode_t biasMode,
                                                    miopenRNNAlgo_t algo,
                                                    miopenDataType_t dataType)
{

    MIOPEN_LOG_FUNCTION(
        rnnDesc, hsize, nlayers, dropoutDesc, inMode, direction, rnnMode, biasMode, algo, dataType);
    return miopen::try_([&] {
        miopen::deref(rnnDesc) = miopen::RNNDescriptor(
            hsize, nlayers, rnnMode, inMode, direction, biasMode, algo, dataType, dropoutDesc);
    });
}

extern "C" miopenStatus_t miopenGetRNNWorkspaceSize(miopenHandle_t handle,
                                                    const miopenRNNDescriptor_t rnnDesc,
                                                    const int sequenceLen,
                                                    const miopenTensorDescriptor_t* xDesc,
                                                    size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, sequenceLen, xDesc);
    miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) =
            miopen::deref(rnnDesc).GetWorkspaceSize(miopen::deref(handle), sequenceLen, xDescArray);
    });
}

extern "C" miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          int sequenceLen,
                                                          const miopenTensorDescriptor_t* xDesc,
                                                          size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, sequenceLen, xDesc);
    miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) =
            miopen::deref(rnnDesc).GetReserveSize(miopen::deref(handle), sequenceLen, xDescArray);
    });
}

extern "C" miopenStatus_t miopenGetRNNTempSpaceSizes(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     miopenSeqTensorDescriptor_t xDesc,
                                                     miopenRNNFWDMode_t fwdMode,
                                                     size_t* workSpaceSize,
                                                     size_t* reserveSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, xDesc, fwdMode);

    return miopen::try_([&] {
        if(workSpaceSize != nullptr)
        {
            miopen::deref(workSpaceSize) = miopen::deref(rnnDesc).GetMaxWorkspaceSize(
                miopen::deref(handle), miopen::deref(xDesc), fwdMode);
        }

        if((fwdMode == miopenRNNTraining) && reserveSpaceSize != nullptr)
        {
            miopen::deref(reserveSpaceSize) = miopen::deref(rnnDesc).GetMaxReserveSize(
                miopen::deref(handle), miopen::deref(xDesc));
        }
    });
}

extern "C" miopenStatus_t miopenGetRNNParamsDescriptor(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       miopenTensorDescriptor_t xDesc,
                                                       miopenTensorDescriptor_t wDesc,
                                                       miopenDataType_t dtype)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, xDesc, dtype);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetParamsDescriptor(
            miopen::deref(handle), miopen::deref(xDesc), miopen::deref(wDesc), dtype);
    });
}

extern "C" miopenStatus_t miopenGetRNNParamsSize(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 miopenTensorDescriptor_t xDesc,
                                                 size_t* numBytes,
                                                 miopenDataType_t dtype)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, xDesc, dtype);
    return miopen::try_([&] {
        miopen::deref(numBytes) = miopen::deref(rnnDesc).GetParamsSize(
            miopen::deref(handle), miopen::deref(xDesc), dtype);
    });
}

extern "C" miopenStatus_t miopenGetRNNInputTensorSize(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      const int seqLen,
                                                      miopenTensorDescriptor_t* xDesc,
                                                      size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, seqLen, xDesc);
    miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(seqLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) = miopen::deref(rnnDesc).GetRNNInputSuperTensorSize(
            miopen::deref(handle), seqLen, xDescArray);
    });
}

extern "C" miopenStatus_t miopenGetRNNHiddenTensorSize(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int seqLen,
                                                       miopenTensorDescriptor_t* xDesc,
                                                       size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, seqLen, xDesc);
    miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(seqLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) =
            miopen::deref(rnnDesc).GetRNNHiddenSuperTensorSize(miopen::deref(handle), xDescArray);
    });
}

extern "C" miopenStatus_t miopenGetRNNLayerParamSize(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     const int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const int paramID,
                                                     size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, layer, xDesc, paramID);
    return miopen::try_([&] {
        miopen::deref(numBytes) = miopen::deref(rnnDesc).GetLayerParamSize(
            miopen::deref(handle), layer, miopen::deref(xDesc), paramID);
    });
}

extern "C" miopenStatus_t miopenGetRNNLayerBiasSize(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int layer,
                                                    const int biasID,
                                                    size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, layer, biasID);
    return miopen::try_([&] {
        miopen::deref(numBytes) =
            miopen::deref(rnnDesc).GetLayerBiasSize(miopen::deref(handle), layer, biasID);
    });
}

extern "C" miopenStatus_t miopenGetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 const void* w,
                                                 const int paramID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 void* layerParam)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, layer, xDesc, wDesc, w, paramID);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetLayerParam(miopen::deref(handle),
                                             layer,
                                             miopen::deref(xDesc),
                                             miopen::deref(wDesc),
                                             DataCast(w),
                                             paramID,
                                             miopen::deref(paramDesc),
                                             DataCast(layerParam));
    });
}

extern "C" miopenStatus_t miopenGetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const int biasID,
                                                miopenTensorDescriptor_t biasDesc,
                                                void* layerBias)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, layer, xDesc, wDesc, w, biasID);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetLayerBias(miopen::deref(handle),
                                            layer,
                                            miopen::deref(xDesc),
                                            miopen::deref(wDesc),
                                            DataCast(w),
                                            biasID,
                                            miopen::deref(biasDesc),
                                            DataCast(layerBias));
    });
}

extern "C" miopenStatus_t miopenGetRNNLayerParamOffset(miopenRNNDescriptor_t rnnDesc,
                                                       const int layer,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const int paramID,
                                                       miopenTensorDescriptor_t paramDesc,
                                                       size_t* layerParamOffset)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, layer, xDesc, paramID);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetLayerParamOffset(
            layer, miopen::deref(xDesc), paramID, miopen::deref(paramDesc), layerParamOffset);
    });
}

extern "C" miopenStatus_t miopenGetRNNLayerBiasOffset(miopenRNNDescriptor_t rnnDesc,
                                                      const int layer,
                                                      miopenTensorDescriptor_t xDesc,
                                                      const int biasID,
                                                      miopenTensorDescriptor_t biasDesc,
                                                      size_t* layerBiasOffset)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, layer, xDesc, biasID);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetLayerBiasOffset(
            layer, miopen::deref(xDesc), biasID, miopen::deref(biasDesc), layerBiasOffset);
    });
}

extern "C" miopenStatus_t miopenSetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 void* w,
                                                 const int paramID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 const void* layerParam)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).SetLayerParam(miopen::deref(handle),
                                             layer,
                                             miopen::deref(xDesc),
                                             miopen::deref(wDesc),
                                             DataCast(w),
                                             paramID,
                                             miopen::deref(paramDesc),
                                             DataCast(layerParam));
    });
}

extern "C" miopenStatus_t miopenSetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                void* w,
                                                const int biasID,
                                                miopenTensorDescriptor_t biasDesc,
                                                const void* layerBias)
{
    MIOPEN_LOG_FUNCTION(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).SetLayerBias(miopen::deref(handle),
                                            layer,
                                            miopen::deref(xDesc),
                                            miopen::deref(wDesc),
                                            DataCast(w),
                                            biasID,
                                            miopen::deref(biasDesc),
                                            DataCast(layerBias));
    });
}

extern "C" miopenStatus_t miopenSetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t paddingMode)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, paddingMode);
    return miopen::try_([&] { miopen::deref(rnnDesc).SetPaddingmode(paddingMode); });
}

extern "C" miopenStatus_t miopenGetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t* paddingMode)
{
    auto ret =
        miopen::try_([&] { miopen::deref(paddingMode) = miopen::deref(rnnDesc).paddingMode; });
    MIOPEN_LOG_FUNCTION(rnnDesc, paddingMode);
    return ret;
}

static void LogCmdRNN(const miopenTensorDescriptor_t* xDesc,
                      const miopenRNNDescriptor_t rnnDesc,
                      const int seqLength,
                      const RNNDir_t dir)
{
    if(miopen::IsLoggingCmd() && seqLength > 0)
    {
        std::string mode;
        miopenRNNMode_t rnnMode = miopen::deref(rnnDesc).rnnMode;

        switch(rnnMode)
        {
        case miopenRNNRELU: mode = "relu"; break;
        case miopenRNNTANH: mode = "tanh"; break;
        case miopenLSTM: mode = "lstm"; break;
        case miopenGRU: mode = "gru"; break;
        default: mode = "<UNKNOWN>";
        }

        std::string batch_sz;
        if(miopen::deref(xDesc[0]).GetLengths()[0] ==
           miopen::deref(xDesc[seqLength - 1]).GetLengths()[0])
        {
            batch_sz = std::to_string(miopen::deref(xDesc[0]).GetLengths()[0]);
        }
        else
        {
            for(int i = 0; i < seqLength; i++)
            {
                batch_sz += std::to_string(miopen::deref(xDesc[i]).GetLengths()[0]);
                batch_sz += ",";
            }
            batch_sz.pop_back();
        }

        std::stringstream ss;
        if(miopen::deref(xDesc[0]).GetType() == miopenFloat)
            ss << "rnn";
        else if(miopen::deref(xDesc[0]).GetType() == miopenHalf)
            ss << "rnnfp16";
        ss << " -n " << batch_sz // clang-format off
           << " -W " << miopen::deref(xDesc[0]).GetLengths()[1]
           << " -H " << miopen::deref(rnnDesc).hsize
           << " -l " << miopen::deref(rnnDesc).nLayers
           << " -b " << (miopen::deref(rnnDesc).biasMode == miopenRNNNoBias ? "0" : "1")
           << " -m " << mode
           << " -p " << (miopen::deref(rnnDesc).inputMode == miopenRNNlinear ? "0" : "1")
           << " -r " << (miopen::deref(rnnDesc).dirMode == miopenRNNunidirection ? "0" : "1")
           << " -q " << (miopen::deref(rnnDesc).paddingMode == miopenRNNIONotPadded ? "0" : "1")
           << " -k " << seqLength;
        if(dir == ForwardInference || dir == ForwardTraining)
           ss << " -c " << ((dir == ForwardTraining)?"0":"1");
        ss << " -F " << ((dir == ForwardInference || dir == ForwardTraining)?"1":(dir == BackwardData)?"2":"4")
           << " -t 1 -w 1";
        if(miopen::deref(miopen::deref(rnnDesc).dropoutDesc).dropout > 0)
        {
            ss << " -U 1 -P " << std::to_string(miopen::deref(miopen::deref(rnnDesc).dropoutDesc).dropout)
               << " -L " << (miopen::deref(miopen::deref(rnnDesc).dropoutDesc).seed & 0xFFFFFFFF)
               << " -M " << ((miopen::deref(miopen::deref(rnnDesc).dropoutDesc).seed >> 32) & 0xFFFFFFFF);
        } // clang-format on
        std::cout << ss.str() << std::endl;
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

static void LogCmdRNN(const miopenSeqTensorDescriptor_t xDesc,
                      const miopenRNNDescriptor_t rnnDesc,
                      const RNNDir_t dir)
{
    if(miopen::IsLoggingCmd())
    {
        std::string mode;
        miopenRNNMode_t rnnMode = miopen::deref(rnnDesc).rnnMode;

        switch(rnnMode)
        {
        case miopenRNNRELU: mode = "relu"; break;
        case miopenRNNTANH: mode = "tanh"; break;
        case miopenLSTM: mode = "lstm"; break;
        case miopenGRU: mode = "gru"; break;
        default: mode = "<UNKNOWN>";
        }

        std::string seq_len_array;
        {
            auto& seq_len = miopen::deref(xDesc).GetSequenceLengthsVector();

            seq_len_array = std::accumulate(
                std::next(seq_len.begin()),
                seq_len.end(),
                std::to_string(seq_len[0]),
                [](std::string a, int b) { return std::move(a) + ',' + std::to_string(b); });
        }

        auto& xdesc_lens = miopen::deref(xDesc).GetLengths();

        std::string rnn_io_layout;
        {
            auto in_layout =
                miopen::RNNDescriptor::getBaseLayoutFromDataTensor(miopen::deref(xDesc));
            rnn_io_layout =
                " -I " + std::to_string(in_layout == miopenRNNDataSeqMajorNotPadded  ? 1
                                        : in_layout == miopenRNNDataSeqMajorPadded   ? 2
                                        : in_layout == miopenRNNDataBatchMajorPadded ? 3
                                                                                     : 0);
        }
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenFloat)
            ss << "rnn_seq";
        else if(miopen::deref(xDesc).GetType() == miopenHalf)
            ss << "rnn_seqfp16";

        ss << " -n " << xdesc_lens[0] // clang-format off
           << " -W " << xdesc_lens[2]
           << " -H " << miopen::deref(rnnDesc).hsize
           << " -l " << miopen::deref(rnnDesc).nLayers
           << " -b " << (miopen::deref(rnnDesc).biasMode == miopenRNNNoBias ? "0" : "1")
           << " -m " << mode
           << " -p " << (miopen::deref(rnnDesc).inputMode == miopenRNNlinear ? "0" : "1")
           << " -r " << (miopen::deref(rnnDesc).dirMode == miopenRNNunidirection ? "0" : "1")
           << " -k " << xdesc_lens[1]
           << " -K " << seq_len_array
           << " -a " << (miopen::deref(rnnDesc).algoMode == miopenRNNdefault ? "0" : "1")
           << rnn_io_layout;


        if(dir == ForwardInference || dir == ForwardTraining)
           ss << " -c " << ((dir == ForwardTraining)?"0":"1");

        ss << " -F " << ((dir == ForwardInference || dir == ForwardTraining)?"1":(dir == BackwardData)?"2":"4")
           << " -t 1 -w 1";
        if(miopen::deref(miopen::deref(rnnDesc).dropoutDesc).dropout > 0)
        {
            ss << " -U 1 -P " << std::to_string(miopen::deref(miopen::deref(rnnDesc).dropoutDesc).dropout)
               << " -L " << (miopen::deref(miopen::deref(rnnDesc).dropoutDesc).seed & 0xFFFFFFFF)
               << " -M " << ((miopen::deref(miopen::deref(rnnDesc).dropoutDesc).seed >> 32) & 0xFFFFFFFF);
        } // clang-format on
        std::cout << ss.str() << std::endl;
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenRNNForward(miopenHandle_t handle,
                                           const miopenRNNDescriptor_t rnnDesc,
                                           miopenRNNFWDMode_t fwdMode,
                                           const miopenSeqTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t hDesc,
                                           const void* hx,
                                           void* hy,
                                           const miopenTensorDescriptor_t cDesc,
                                           const void* cx,
                                           void* cy,
                                           const miopenSeqTensorDescriptor_t yDesc,
                                           void* y,
                                           const void* w,
                                           size_t weightSpaceSize,
                                           void* workSpace,
                                           size_t workSpaceNumBytes,
                                           void* reserveSpace,
                                           size_t reserveSpaceNumBytes)
{
    MIOPEN_LOG_FUNCTION(handle,
                        rnnDesc,
                        fwdMode,
                        xDesc,
                        x,
                        hDesc,
                        hx,
                        hy,
                        cDesc,
                        cx,
                        cy,
                        yDesc,
                        y,
                        w,
                        weightSpaceSize,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);

    // bfloat16 not supported for rnn operation
    if(miopen::deref(rnnDesc).dataType == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    LogCmdRNN(xDesc,
              rnnDesc,
              fwdMode == miopenRNNFWDMode_t::miopenRNNInference ? ForwardInference
                                                                : ForwardTraining);

    return miopen::try_([&] {
        miopen::deref(rnnDesc).RNNForward(miopen::deref(handle),
                                          fwdMode,
                                          miopen::deref(xDesc),
                                          DataCast(x),
                                          miopen::deref(hDesc),
                                          DataCast(hx),
                                          DataCast(hy),
                                          miopen::deref(cDesc),
                                          DataCast(cx),
                                          DataCast(cy),
                                          miopen::deref(yDesc),
                                          DataCast(y),
                                          DataCast(w),
                                          weightSpaceSize,
                                          DataCast(workSpace),
                                          workSpaceNumBytes,
                                          DataCast(reserveSpace),
                                          reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNBackwardSeqData(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const miopenSeqTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t hDesc,
                                                   const void* hx,
                                                   const void* dhy,
                                                   void* dhx,
                                                   const miopenTensorDescriptor_t cDesc,
                                                   const void* cx,
                                                   const void* dcy,
                                                   void* dcx,
                                                   const miopenSeqTensorDescriptor_t xDesc,
                                                   void* dx,
                                                   const void* w,
                                                   size_t weightSpaceSize,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{
    MIOPEN_LOG_FUNCTION(handle,
                        rnnDesc,
                        yDesc,
                        y,
                        dy,
                        hDesc,
                        hx,
                        dhy,
                        dhx,
                        cDesc,
                        cx,
                        dcy,
                        dcx,
                        xDesc,
                        dx,
                        w,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);

    // bfloat16 not supported for rnn operation
    if(miopen::deref(rnnDesc).dataType == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdRNN(xDesc, rnnDesc, BackwardData);

    return miopen::try_([&] {
        miopen::deref(rnnDesc).RNNBackwardData(miopen::deref(handle),
                                               miopen::deref(yDesc),
                                               DataCast(y),
                                               DataCast(dy),
                                               miopen::deref(hDesc),
                                               DataCast(hx),
                                               DataCast(dhy),
                                               DataCast(dhx),
                                               miopen::deref(cDesc),
                                               DataCast(cx),
                                               DataCast(dcy),
                                               DataCast(dcx),
                                               miopen::deref(xDesc),
                                               DataCast(dx),
                                               DataCast(w),
                                               weightSpaceSize,
                                               DataCast(workSpace),
                                               workSpaceNumBytes,
                                               DataCast(reserveSpace),
                                               reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNBackwardWeightsSeqTensor(miopenHandle_t handle,
                                                            const miopenRNNDescriptor_t rnnDesc,
                                                            const miopenSeqTensorDescriptor_t xDesc,
                                                            const void* x,
                                                            const miopenTensorDescriptor_t hDesc,
                                                            const void* hx,
                                                            const miopenSeqTensorDescriptor_t yDesc,
                                                            const void* y,
                                                            void* dw,
                                                            size_t weightSpaceSize,
                                                            void* workSpace,
                                                            size_t workSpaceNumBytes,
                                                            const void* reserveSpace,
                                                            size_t reserveSpaceNumBytes)
{
    MIOPEN_LOG_FUNCTION(handle,
                        rnnDesc,
                        xDesc,
                        x,
                        hDesc,
                        hx,
                        yDesc,
                        y,
                        dw,
                        weightSpaceSize,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);

    // bfloat16 not supported for rnn operation
    if(miopen::deref(rnnDesc).dataType == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdRNN(xDesc, rnnDesc, BackwardWeights);

    return miopen::try_([&] {
        miopen::deref(rnnDesc).RNNBackwardWeights(miopen::deref(handle),
                                                  miopen::deref(xDesc),
                                                  DataCast(x),
                                                  miopen::deref(hDesc),
                                                  DataCast(hx),
                                                  miopen::deref(yDesc),
                                                  DataCast(y),
                                                  DataCast(dw),
                                                  weightSpaceSize,
                                                  DataCast(workSpace),
                                                  workSpaceNumBytes,
                                                  DataCast(reserveSpace),
                                                  reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   const miopenTensorDescriptor_t* xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t hxDesc,
                                                   const void* hx,
                                                   const miopenTensorDescriptor_t cxDesc,
                                                   const void* cx,
                                                   const miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   const miopenTensorDescriptor_t* yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t hyDesc,
                                                   void* hy,
                                                   const miopenTensorDescriptor_t cyDesc,
                                                   void* cy,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(handle,
                        rnnDesc,
                        sequenceLen,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        cxDesc,
                        cx,
                        wDesc,
                        w,
                        yDesc,
                        y,
                        hyDesc,
                        hy,
                        cyDesc,
                        cy,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);

    // bfloat16 not supported for rnn operation
    if(miopen::deref(wDesc).GetType() == miopenBFloat16 ||
       miopen::deref(hyDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdRNN(xDesc, rnnDesc, sequenceLen, ForwardTraining);

    return miopen::try_([&] {
        miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
        miopen::c_array_view<const miopenTensorDescriptor_t> yDescArray{yDesc, size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNForwardTraining(miopen::deref(handle),
                                                  sequenceLen,
                                                  xDescArray,
                                                  DataCast(x),
                                                  miopen::deref(hxDesc),
                                                  DataCast(hx),
                                                  miopen::deref(cxDesc),
                                                  DataCast(cx),
                                                  miopen::deref(wDesc),
                                                  DataCast(w),
                                                  yDescArray,
                                                  DataCast(y),
                                                  miopen::deref(hyDesc),
                                                  DataCast(hy),
                                                  miopen::deref(cyDesc),
                                                  DataCast(cy),
                                                  DataCast(workSpace),
                                                  workSpaceNumBytes,
                                                  DataCast(reserveSpace),
                                                  reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNBackwardData(miopenHandle_t handle,
                                                const miopenRNNDescriptor_t rnnDesc,
                                                const int sequenceLen,
                                                const miopenTensorDescriptor_t* yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t* dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t dhyDesc,
                                                const void* dhy,
                                                const miopenTensorDescriptor_t dcyDesc,
                                                const void* dcy,
                                                const miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const miopenTensorDescriptor_t hxDesc,
                                                const void* hx,
                                                const miopenTensorDescriptor_t cxDesc,
                                                const void* cx,
                                                const miopenTensorDescriptor_t* dxDesc,
                                                void* dx,
                                                const miopenTensorDescriptor_t dhxDesc,
                                                void* dhx,
                                                const miopenTensorDescriptor_t dcxDesc,
                                                void* dcx,
                                                void* workSpace,
                                                size_t workSpaceNumBytes,
                                                void* reserveSpace,
                                                size_t reserveSpaceNumBytes)
{
    MIOPEN_LOG_FUNCTION(handle,
                        rnnDesc,
                        sequenceLen,
                        yDesc,
                        y,
                        dyDesc,
                        dy,
                        dhyDesc,
                        dhy,
                        dcyDesc,
                        dcy,
                        wDesc,
                        w,
                        hxDesc,
                        hx,
                        cxDesc,
                        cx,
                        dxDesc,
                        dx,
                        dhxDesc,
                        dhx,
                        dcxDesc,
                        dcx,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);

    // bfloat16 not supported for rnn operation
    if(miopen::deref(wDesc).GetType() == miopenBFloat16 ||
       miopen::deref(cxDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdRNN(dxDesc, rnnDesc, sequenceLen, BackwardData);

    return miopen::try_([&] {
        miopen::c_array_view<const miopenTensorDescriptor_t> yDescArray{yDesc, size_t(sequenceLen)};
        miopen::c_array_view<const miopenTensorDescriptor_t> dyDescArray{dyDesc,
                                                                         size_t(sequenceLen)};
        miopen::c_array_view<const miopenTensorDescriptor_t> dxDescArray{dxDesc,
                                                                         size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNBackwardData(miopen::deref(handle),
                                               sequenceLen,
                                               yDescArray,
                                               DataCast(y),
                                               dyDescArray,
                                               DataCast(dy),
                                               miopen::deref(dhyDesc),
                                               DataCast(dhy),
                                               miopen::deref(dcyDesc),
                                               DataCast(dcy),
                                               miopen::deref(wDesc),
                                               DataCast(w),
                                               miopen::deref(hxDesc),
                                               DataCast(hx),
                                               miopen::deref(cxDesc),
                                               DataCast(cx),
                                               dxDescArray,
                                               DataCast(dx),
                                               miopen::deref(dhxDesc),
                                               DataCast(dhx),
                                               miopen::deref(dcxDesc),
                                               DataCast(dcx),
                                               DataCast(workSpace),
                                               workSpaceNumBytes,
                                               DataCast(reserveSpace),
                                               reserveSpaceNumBytes);
    });
}

miopenStatus_t miopenRNNBackwardWeights(miopenHandle_t handle,
                                        const miopenRNNDescriptor_t rnnDesc,
                                        const int sequenceLen,
                                        const miopenTensorDescriptor_t* xDesc,
                                        const void* x,
                                        const miopenTensorDescriptor_t hxDesc,
                                        const void* hx,
                                        const miopenTensorDescriptor_t* yDesc,
                                        const void* y,
                                        const miopenTensorDescriptor_t dwDesc,
                                        void* dw,
                                        void* workSpace,
                                        size_t workSpaceNumBytes,
                                        const void* reserveSpace,
                                        size_t reserveSpaceNumBytes)
{
    MIOPEN_LOG_FUNCTION(handle,
                        rnnDesc,
                        sequenceLen,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        yDesc,
                        y,
                        dwDesc,
                        dw,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);

    // bfloat16 not supported for rnn operation
    if(miopen::deref(hxDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dwDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdRNN(xDesc, rnnDesc, sequenceLen, BackwardWeights);

    return miopen::try_([&] {
        miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
        miopen::c_array_view<const miopenTensorDescriptor_t> yDescArray{yDesc, size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNBackwardWeights(miopen::deref(handle),
                                                  sequenceLen,
                                                  xDescArray,
                                                  DataCast(x),
                                                  miopen::deref(hxDesc),
                                                  DataCast(hx),
                                                  yDescArray,
                                                  DataCast(y),
                                                  miopen::deref(dwDesc),
                                                  DataCast(dw),
                                                  DataCast(workSpace),
                                                  workSpaceNumBytes,
                                                  DataCast(reserveSpace),
                                                  reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNForwardInference(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int sequenceLen,
                                                    const miopenTensorDescriptor_t* xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t hxDesc,
                                                    const void* hx,
                                                    const miopenTensorDescriptor_t cxDesc,
                                                    const void* cx,
                                                    const miopenTensorDescriptor_t wDesc,
                                                    const void* w,
                                                    const miopenTensorDescriptor_t* yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t hyDesc,
                                                    void* hy,
                                                    const miopenTensorDescriptor_t cyDesc,
                                                    void* cy,
                                                    void* workSpace,
                                                    size_t workSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(handle,
                        rnnDesc,
                        sequenceLen,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        cxDesc,
                        cx,
                        wDesc,
                        w,
                        yDesc,
                        y,
                        hyDesc,
                        hy,
                        cyDesc,
                        cy,
                        workSpace,
                        workSpaceNumBytes);
    LogCmdRNN(xDesc, rnnDesc, sequenceLen, ForwardInference);
    return miopen::try_([&] {
        miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
        miopen::c_array_view<const miopenTensorDescriptor_t> yDescArray{yDesc, size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNForwardInference(miopen::deref(handle),
                                                   sequenceLen,
                                                   xDescArray,
                                                   DataCast(x),
                                                   miopen::deref(hxDesc),
                                                   DataCast(hx),
                                                   miopen::deref(cxDesc),
                                                   DataCast(cx),
                                                   miopen::deref(wDesc),
                                                   DataCast(w),
                                                   yDescArray,
                                                   DataCast(y),
                                                   miopen::deref(hyDesc),
                                                   DataCast(hy),
                                                   miopen::deref(cyDesc),
                                                   DataCast(cy),
                                                   DataCast(workSpace),
                                                   workSpaceNumBytes);
    });
}
