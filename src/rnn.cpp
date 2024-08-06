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

#include <miopen/handle.hpp>
#include <miopen/rnn.hpp>
#include <miopen/rnn_util.hpp>

#include <cassert>
#include <cstddef>
#include <numeric>
#include <ostream>

// Disable specific warnings
#define MIO_RNN_DEBUG 0

#define MIOPEN_RNN_SYNCH 0
#define MIO_RNN_CPP_PROF 0

namespace miopen {

void profileRNNkernels(const Handle& handle, unsigned char select, float& ctime)
{

    float ktime = 0.;
    assert((select < 3) && "profileSequence case incorrect");
    switch(select)
    {

    case 0:
        if(handle.IsProfilingEnabled())
        {
            handle.ResetKernelTime();
            ctime = 0.;

#if(MIO_RNN_CPP_PROF == 1)
            printf("init ktime: %f\n", ktime);
            printf("init ctime: %f\n", ctime);
#endif
        }
#if(MIOPEN_RNN_SYNCH == 1)
        else
        {
            handle.Finish();
        }
#endif
        break;
    case 1:
        if(handle.IsProfilingEnabled())
        {
            ktime = handle.GetKernelTime();
            ctime += ktime;

#if(MIO_RNN_CPP_PROF == 1)
            printf("intermediate ktime: %f\n", ktime);
            printf("intermediate ctime: %f\n", ctime);
#endif
        }
#if(MIOPEN_RNN_SYNCH == 1)
        else
        {
            handle.Finish();
        }
#endif
        break;

    case 2:
        if(handle.IsProfilingEnabled())
        {
#if(MIO_RNN_CPP_PROF == 1)
            ktime = handle.GetKernelTime();
            printf("Final time: %f\n", ktime + ctime);
            handle.AccumKernelTime(ctime);
#else
            handle.AccumKernelTime(ctime);
#endif
        }
        break;
    default: assert(false);
    }
}

size_t RNNDescriptor::biasOffsetCalculation(const TensorDescriptor& /*xDesc*/,
                                            const int layer,
                                            const int biasID) const
{
    if(biasMode == miopenRNNNoBias)
    {
        return 0;
    }

    size_t layerJump = 0;

    if(dirMode != 0u)
    {
        layerJump += (hsize * 2) * nHiddenTensorsPerLayer * (layer / 2) * 2;

        if(biasID >= nHiddenTensorsPerLayer)
        {
            layerJump += hsize * nHiddenTensorsPerLayer;
        }

        layerJump += (layer % 2 == 1) ? nHiddenTensorsPerLayer * hsize : 0;

        layerJump += hsize * biasID;
    }
    else
    {
        layerJump += (hsize * 2) * nHiddenTensorsPerLayer * layer;

        layerJump += hsize * biasID;
    }

    return layerJump;
}

size_t RNNDescriptor::paramsOffsetCalculation(const TensorDescriptor& xDesc,
                                              const int layer,
                                              const int paramID) const
{
    auto inputVectorLen = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
    {
        inputVectorLen = 0;
    }

    size_t layerJump = 0;
    if(dirMode != 0u)
    {
        if(layer > 1)
        {
            layerJump += (inputVectorLen * hsize + hsize * hsize) * nHiddenTensorsPerLayer * 2;
            layerJump +=
                (hsize * hsize * 2 + hsize * hsize) * nHiddenTensorsPerLayer * (layer / 2 - 1) * 2;

            if(paramID >= nHiddenTensorsPerLayer)
            {
                layerJump += hsize * hsize * 2 * nHiddenTensorsPerLayer * 2;
                layerJump += (layer % 2 == 1) ? nHiddenTensorsPerLayer * (hsize * hsize) : 0;
                layerJump += (hsize * hsize) * (paramID - nHiddenTensorsPerLayer);
            }
            else
            {
                layerJump += (layer % 2 == 1) ? nHiddenTensorsPerLayer * (2 * hsize * hsize) : 0;
                layerJump += (2 * hsize * hsize) * paramID;
            }
        }
        else
        {
            if(paramID >= nHiddenTensorsPerLayer)
            {
                if(isNotRNNskip())
                {
                    layerJump += (inputVectorLen * hsize) * nHiddenTensorsPerLayer * 2;
                }
                layerJump += (layer == 1) ? nHiddenTensorsPerLayer * (hsize * hsize) : 0;
                layerJump += (hsize * hsize) * (paramID - nHiddenTensorsPerLayer);
            }
            else
            {
                layerJump += (layer == 1) ? nHiddenTensorsPerLayer * (inputVectorLen * hsize) : 0;
                layerJump += (inputVectorLen * hsize) * paramID;
            }
        }
    }
    else
    {

        if(layer > 0)
        {
            layerJump += (inputVectorLen * hsize + hsize * hsize) * nHiddenTensorsPerLayer;
            layerJump += (hsize * hsize * 2) * nHiddenTensorsPerLayer * (layer - 1);
            layerJump += (hsize * hsize) * paramID;
        }
        else
        {
            if(paramID >= nHiddenTensorsPerLayer)
            {
                if(isNotRNNskip())
                {
                    layerJump += (inputVectorLen * hsize) * nHiddenTensorsPerLayer;
                }
                layerJump += (hsize * hsize) * (paramID - nHiddenTensorsPerLayer);
            }
            else
            {
                layerJump += (inputVectorLen * hsize) * paramID;
            }
        }
    }
    return layerJump;
}

std::vector<int> RNNDescriptor::pTensorLengthsCalculation(const TensorDescriptor& xDesc,
                                                          const int layer,
                                                          const int paramID) const
{
    auto inputVectorLen = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
    {
        inputVectorLen = 0;
    }

    std::vector<int> tdim(2, 0);

    if(dirMode != 0u)
    {
        if(layer > 1) // NOT the input layer
        {
            if(paramID >= nHiddenTensorsPerLayer)
            {
                tdim[0] = tdim[1] = hsize;
            }
            else
            {
                tdim[0] = hsize;
                tdim[1] = hsize * 2;
            }
        }
        else // IS the input layer
        {
            if(paramID >= nHiddenTensorsPerLayer)
            {
                tdim[0] = tdim[1] = hsize;
            }
            else
            {
                tdim[0] = hsize;
                tdim[1] = inputVectorLen;
            }
        }
    }
    else
    {
        if(layer > 0) // NOT the input layer
        {
            tdim[0] = tdim[1] = hsize;
        }
        else
        {
            if(paramID >= nHiddenTensorsPerLayer)
            {
                tdim[0] = tdim[1] = hsize;
            }
            else
            {
                tdim[0] = hsize;
                tdim[1] = inputVectorLen;
            }
        }
    }
    return tdim;
}

RNNDescriptor::RNNDescriptor()
{
    nLayers                     = 1;
    hsize                       = 0;
    nHiddenTensorsPerLayer      = 0;
    rnnMode                     = miopenRNNTANH;
    dirMode                     = miopenRNNunidirection;
    biasMode                    = miopenRNNNoBias;
    algoMode                    = miopenRNNdefault;
    inputMode                   = miopenRNNlinear;
    dataType                    = miopenFloat;
    typeSize                    = 4;
    workspaceScale              = 1;
    miopen::deref(&dropoutDesc) = new miopen::DropoutDescriptor();
}

RNNDescriptor::RNNDescriptor(int hsz,
                             int layers,
                             miopenRNNMode_t rmode,
                             miopenRNNInputMode_t inMode,
                             miopenRNNDirectionMode_t bidir,
                             miopenRNNBiasMode_t bmode,
                             miopenRNNAlgo_t amode,
                             miopenDataType_t dType)
{

    if(hsz < 0 || layers < 0)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). RNN hidden size and "
                     "layer number must be positive integers.");
    }
    if(!(rmode == miopenRNNRELU || rmode == miopenRNNTANH || rmode == miopenLSTM ||
         rmode == miopenGRU))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). RNN mode must be "
                     "vanilla activated with ReLU or Tanh, LSTM or GRU.");
    }
    if(bidir != 0 && bidir != 1)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). Parameters to RNN "
                     "directional type must be 0 for uni-direction or 1 for "
                     "bi-direction.");
    }
    if(bmode != 0 && bmode != 1)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). Parameters to RNN bias "
                     "type must be 0 for disabled bias or 1 for enabled "
                     "bias.");
    }
    if(dType != miopenFloat && dType != miopenHalf)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). RNN datatype must be float or half.");
    }
    else
    {
        typeSize = dType == miopenHalf ? 2 : 4;
    }

    hsize                       = hsz;
    nLayers                     = layers;
    inputMode                   = inMode;
    dirMode                     = bidir;
    rnnMode                     = rmode;
    algoMode                    = amode;
    biasMode                    = bmode;
    dataType                    = dType;
    miopen::deref(&dropoutDesc) = new miopen::DropoutDescriptor();

    switch(rmode)
    {
    case 0: // RNN vanilla
    case 1: // RNN vanilla
        nHiddenTensorsPerLayer = 1;
        workspaceScale         = 1;
        break;
    case 2: // LSTM
        nHiddenTensorsPerLayer = 4;
        workspaceScale         = 6;
        break;
    case 3: // GRU
        nHiddenTensorsPerLayer = 3;
        workspaceScale         = 4;
        break;
    }
}

RNNDescriptor::RNNDescriptor(int hsz,
                             int layers,
                             miopenRNNMode_t rmode,
                             miopenRNNInputMode_t inMode,
                             miopenRNNDirectionMode_t bidir,
                             miopenRNNBiasMode_t bmode,
                             miopenRNNAlgo_t amode,
                             miopenDataType_t dType,
                             miopenDropoutDescriptor_t dropDesc)
    : hsize(size_t(hsz)),
      nLayers(size_t(layers)),
      rnnMode(rmode),
      dirMode(bidir),
      algoMode(amode),
      inputMode(inMode),
      biasMode(bmode),
      dataType(dType),
      dropoutDesc(dropDesc)
{

    if(hsz < 0 || layers < 0)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). RNN hidden size and "
                     "layer number must be positive integers.");
    }
    if(!(rmode == miopenRNNRELU || rmode == miopenRNNTANH || rmode == miopenLSTM ||
         rmode == miopenGRU))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). RNN mode must be "
                     "vanilla activated with ReLU or Tanh, LSTM or GRU.");
    }
    if(bidir != 0 && bidir != 1)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). Parameters to RNN "
                     "directional type must be 0 for uni-direction or 1 for "
                     "bi-direction.");
    }
    if(bmode != 0 && bmode != 1)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). Parameters to RNN bias "
                     "type must be 0 for disabled bias or 1 for enabled "
                     "bias.");
    }
    if(dType != miopenFloat && dType != miopenHalf)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "RNNDescriptor: Bad parameter(s). RNN datatype must be float or half.");
    }
    else
    {
        typeSize = dType == miopenHalf ? 2 : 4;
    }

    switch(rmode)
    {
    case 0: // RNN vanilla
    case 1: // RNN vanilla
        nHiddenTensorsPerLayer = 1;
        workspaceScale         = 1;
        break;
    case 2: // LSTM
        nHiddenTensorsPerLayer = 4;
        workspaceScale         = 6;
        break;
    case 3: // GRU
        nHiddenTensorsPerLayer = 3;
        workspaceScale         = 4;
        break;
    }
}

// Main Solution
// RNN pure algo miopenRNNDataSeqMajorNotPadded
size_t RNNDescriptor::GetMainSolWorkspaceSize(size_t batchLenSum,
                                              miopenRNNFWDMode_t, // fwdMode,
                                              miopenRNNBaseLayout_t ioLayout) const
{
    if(ioLayout != miopenRNNDataSeqMajorNotPadded)
        MIOPEN_THROW(miopenStatusInternalError, "wrong ioLayout");

    const bool is_bidirect = dirMode == miopenRNNbidirection;
    // const bool isTraining = fwdMode == miopenRNNFWDMode_t::miopenRNNTraining;

    return (workspaceScale * nLayers * batchLenSum * hsize * typeSize) * (is_bidirect ? 2 : 1);
}

size_t RNNDescriptor::GetWorkspaceSize(Handle& handle,
                                       const SeqTensorDescriptor& xDesc,
                                       miopenRNNFWDMode_t fwdMode) const
{
    if(xDesc.GetType() != dataType)
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    if(!xDesc.IsZeroBytePadding())
        MIOPEN_THROW(miopenStatusInternalError, "wrong BytePadding ");

    const auto io_layout        = getBaseLayoutFromDataTensor(xDesc);
    const bool is_transform_req = io_layout != miopenRNNDataSeqMajorNotPadded;

    size_t transformer_tmp_space = 0;
    if(is_transform_req)
    {
        transformer_tmp_space = RNNTransformerWorkspaceSize(xDesc, fwdMode);
    }

    const std::size_t total_sequence_len = xDesc.GetTotalSequenceLen();

    size_t reduction_ws = ReductionWorkspaceSize(handle,
                                                 total_sequence_len,
                                                 nHiddenTensorsPerLayer,
                                                 workspaceScale,
                                                 hsize,
                                                 dirMode == miopenRNNbidirection,
                                                 dataType);

    return transformer_tmp_space + reduction_ws +
           GetMainSolWorkspaceSize(total_sequence_len, fwdMode, miopenRNNDataSeqMajorNotPadded);
}

size_t RNNDescriptor::GetMaxWorkspaceSize(Handle& handle,
                                          const SeqTensorDescriptor& xDesc,
                                          miopenRNNFWDMode_t fwdMode) const
{
    if(xDesc.GetType() != dataType)
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");

    const SeqTensorDescriptor x_max = SeqTensorDescriptor(dataType,
                                                          xDesc.GetLayoutVector(),
                                                          xDesc.GetLengths(),
                                                          xDesc.GetPadding(),
                                                          xDesc.IsPaddedSeqLayout());

    return GetWorkspaceSize(handle, x_max, fwdMode);
}

// legacy
size_t RNNDescriptor::GetWorkspaceSize(Handle& handle,
                                       const int seqLength,
                                       c_array_view<const miopenTensorDescriptor_t> xDesc) const
{
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }

    size_t padding_converter_tmp_space = 0;
    if(paddingMode == miopenRNNPaddingMode_t::miopenRNNIOWithPadding)
    {
        size_t packedXInSpace, packedYOutSpace;
        std::tie(packedXInSpace, packedYOutSpace) =
            RNNTensorPaddingConverter::GetTempPackedBuffersSpace(*this, xDesc);
        padding_converter_tmp_space = packedXInSpace + packedYOutSpace;
    }

    std::size_t total_sequence_len = 0;
    total_sequence_len             = std::accumulate(
        xDesc.data, xDesc.data + seqLength, 0ULL, [](size_t x, miopenTensorDescriptor_t y) {
            return x + deref(y).GetLengths()[0];
        });

    size_t reduction_ws = ReductionWorkspaceSize(handle,
                                                 total_sequence_len,
                                                 nHiddenTensorsPerLayer,
                                                 workspaceScale,
                                                 hsize,
                                                 dirMode == miopenRNNbidirection,
                                                 dataType);

    return padding_converter_tmp_space + reduction_ws +
           GetMainSolWorkspaceSize(
               total_sequence_len, miopenRNNInference, miopenRNNDataSeqMajorNotPadded);
}

/////////////////////////////////

size_t RNNDescriptor::GetReserveSize(size_t batchLenSum) const
{
    auto x = 2 * workspaceScale * nLayers * batchLenSum * hsize * typeSize;
    if(algoMode == miopenRNNdefault && rnnMode == miopenLSTM)
    {
        x /= 2;
        x += nLayers * batchLenSum * hsize * typeSize;
    }
    if(!float_equal(miopen::deref(dropoutDesc).dropout, 0))
    {
        x += (nLayers - 1) * batchLenSum * hsize * typeSize;
        x += (nLayers - 1) * batchLenSum * hsize * sizeof(bool);
    }
    return size_t(dirMode == miopenRNNbidirection ? 2 * x : x);
}

// This function should return the size of the Reserve buffer which will be sufficient for the
// tensor
//  with tensor with maximum sequence length and maximum count of non empty sequences.
// The previous version of this function returned a size sufficient only for the current tensor
// size.
size_t RNNDescriptor::GetMaxReserveSize(Handle& /* handle */,
                                        const SeqTensorDescriptor& xDesc) const
{
    if(xDesc.GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    return GetReserveSize(xDesc.GetMaxSequenceLength() * xDesc.GetMaxCountOfSequences());
}

// Legacy.
size_t RNNDescriptor::GetReserveSize(Handle& /* handle */,
                                     const int seqLength,
                                     c_array_view<const miopenTensorDescriptor_t> xDesc) const
{

    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    std::size_t inputBatchLenSum = 0;
    inputBatchLenSum             = std::accumulate(
        xDesc.data, xDesc.data + seqLength, 0ULL, [](size_t x, miopenTensorDescriptor_t y) {
            return x + deref(y).GetLengths()[0];
        });
    return GetReserveSize(inputBatchLenSum);
}

size_t RNNDescriptor::GetParamsSize(size_t inputVector) const
{
    if(inputMode == miopenRNNskip)
    {
        if(inputVector != hsize)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "In miopenRNNskip mode input_vector size and hidden_size shoud be same.");
        }
        inputVector = 0;
    }

    int bi  = dirMode == miopenRNNbidirection ? 2 : 1;
    auto sz = nHiddenTensorsPerLayer * hsize * bi *
              (inputVector + hsize + (nLayers - 1) * (bi + 1) * hsize);
#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr, "weight size: %lu\n", sz);
#endif
    if(biasMode == miopenRNNwithBias)
    {
        sz += nLayers * 2 * nHiddenTensorsPerLayer * hsize * bi;
    }
    return size_t(typeSize * sz);
}

size_t RNNDescriptor::GetParamsSize(Handle& /* handle */,
                                    const TensorDescriptor& xDesc,
                                    miopenDataType_t dtype) const
{
    if(xDesc.GetType() != dataType || dtype != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch.");
    }
    assert(xDesc.GetLengths().size() > 1);
    auto input_vector_len = xDesc.GetLengths()[1];

    return GetParamsSize(input_vector_len);
}

size_t RNNDescriptor::GetRNNInputSuperTensorSize(Handle& /* handle */,
                                                 const int seqLength,
                                                 c_array_view<miopenTensorDescriptor_t> xDesc) const
{
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    std::size_t inputBatchLenSum = 0;
    if(paddingMode == miopenRNNIONotPadded)
    {
        inputBatchLenSum = std::accumulate(
            xDesc.data, xDesc.data + seqLength, 0ULL, [](size_t x, miopenTensorDescriptor_t y) {
                return x + deref(y).GetLengths()[0];
            });
    }
    else
    {
        auto maxBatchSize = xDesc[0].GetLengths()[0];
        inputBatchLenSum  = seqLength * maxBatchSize;
    }
    auto x = inputBatchLenSum * xDesc[0].GetLengths()[1] * typeSize;

    return size_t(x);
}

size_t
RNNDescriptor::GetRNNHiddenSuperTensorSize(Handle& /* handle */,
                                           c_array_view<miopenTensorDescriptor_t> xDesc) const
{
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    auto x = xDesc[0].GetLengths()[0] * hsize * nLayers * typeSize;
    return size_t(dirMode == miopenRNNbidirection ? 2 * x : x);
}

void RNNDescriptor::GetParamsDescriptor(Handle& /* handle */,
                                        const TensorDescriptor& xDesc,
                                        TensorDescriptor& wDesc,
                                        miopenDataType_t dtype) const
{

    if(dtype != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch.");
    }

    auto inputVectorLen = xDesc.GetLengths()[1]; // input vector size
    if(inputMode == miopenRNNskip)
        inputVectorLen = 0;

    // Create weight super tensor descriptor
    int bi = (dirMode == miopenRNNbidirection) ? 2 : 1;
    std::vector<int> weight_lens(2, 0);
    weight_lens[0] = inputVectorLen + ((nLayers - 1) * (bi + 1) + 1) * hsize;
    weight_lens[1] = bi * hsize * nHiddenTensorsPerLayer;
    if(biasMode == miopenRNNwithBias)
    {
        weight_lens[0] += (nLayers * 2);
    }

    wDesc = miopen::TensorDescriptor(dtype, weight_lens);
}

std::size_t RNNDescriptor::GetLayerParamSize(Handle& /*handle*/,
                                             int layer,
                                             const TensorDescriptor& xDesc,
                                             int paramID) const
{
    if(xDesc.GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch.");
    }
    auto inputVectorLen = xDesc.GetLengths()[1]; // input vector size
    inputVectorLen      = (inputMode == miopenRNNskip) ? 0 : inputVectorLen;

    // Assuming Djikstra counting
    if((((dirMode != 0u) && layer <= 1) || ((dirMode == 0u) && layer < 1)))
    {
        if(paramID >= nHiddenTensorsPerLayer)
            return size_t(typeSize * hsize * hsize);
        else if(isNotRNNskip())
            return size_t(typeSize * inputVectorLen * hsize);
        else
            return 0;
    }
    else if((dirMode != 0u) && paramID < nHiddenTensorsPerLayer)
    {
        return size_t(typeSize * hsize * hsize * 2);
    }
    else
    {
        return size_t(typeSize * hsize * hsize);
    }
}

std::size_t
RNNDescriptor::GetLayerBiasSize(Handle& /* handle */, int /*layer*/, int /*biasID*/) const
{
    return size_t(typeSize * hsize); // is ther more needed here?
}

void RNNDescriptor::GetLayerParam(const Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& /* wDesc */,
                                  ConstData_t w,
                                  const int paramID,
                                  TensorDescriptor& paramDesc,
                                  Data_t param) const
{

    if(!isNotRNNskip() && (((dirMode != 0u) && layer <= 1 && paramID < nHiddenTensorsPerLayer) ||
                           ((dirMode == 0u) && layer < 1 && paramID < nHiddenTensorsPerLayer)))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Parameter of input layer is null in input skip mode");
    }

    // Get the dimensions of the parameter matrix
    auto pDims = pTensorLengthsCalculation(xDesc, layer, paramID);
    paramDesc  = miopen::TensorDescriptor(dataType, pDims);
    if(param == nullptr)
    {
        return;
    }

    // Calculate the location of the matrix via paramID, bidirection setting, and params
    auto poffset = paramsOffsetCalculation(xDesc, layer, paramID);

#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr,
            "GetLayerParam layer: %d layerID: %d offset: %lu size: %lu\n",
            layer,
            paramID,
            poffset,
            paramDesc.GetElementSize());
#endif

    // Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, paramDesc, w, paramDesc, param, poffset, 0);
}

void RNNDescriptor::GetLayerBias(const Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& /* wDesc */,
                                 ConstData_t w,
                                 const int biasID,
                                 TensorDescriptor& biasDesc,
                                 Data_t bias) const
{
    if(biasMode == miopenRNNNoBias)
    {
        return;
    }

    // Get the dimensions of the parameter matrix
    auto bdim = hsize;
    biasDesc  = miopen::TensorDescriptor(dataType, {bdim});
    if(bias == nullptr)
    {
        return;
    }

    // Calculate the location of the matrix via layerID, bidirection setting, and params
    int x        = static_cast<int>((dirMode == miopenRNNbidirection) ? nLayers * 2 : nLayers);
    auto poffset = paramsOffsetCalculation(xDesc, x, 0);
    auto boffset = biasOffsetCalculation(xDesc, layer, biasID) + poffset;

#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr,
            "GetLayerBias layer: %d layerID: %d offset: %lu = %lu + %lu size: %lu\n",
            layer,
            biasID,
            boffset,
            poffset,
            boffset - poffset,
            biasDesc.GetElementSize());
#endif

    // Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, biasDesc, w, biasDesc, bias, boffset, 0);
}

void RNNDescriptor::SetLayerParam(const Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& /* wDesc */,
                                  Data_t w,
                                  const int paramID,
                                  const TensorDescriptor& paramDesc,
                                  ConstData_t param) const
{
    if(!isNotRNNskip() && (((dirMode != 0u) && layer <= 1 && paramID < nHiddenTensorsPerLayer) ||
                           ((dirMode == 0u) && layer < 1 && paramID < nHiddenTensorsPerLayer)))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Parameter of input layer is null in input skip mode");
    }

    // TODO dlowell: Need guard checks here, or have them caught at the copy call?
    if(param == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "param data cannot be null");
    }

    // 1. Calculate the location of the matrix via paramID, bidirection setting, and params
    auto poffset = paramsOffsetCalculation(xDesc, layer, paramID);

    // 2. Calculate the strides for the matrix
    std::vector<int> pstride(2, 1);

    pstride[1] = paramDesc.GetLengths()[0];

    std::vector<int> intLens(paramDesc.GetLengths().begin(), paramDesc.GetLengths().end());

    // 3. Construct descriptor to access into w
    auto paramSrc = miopen::TensorDescriptor(dataType, intLens, pstride);

    if(paramSrc.GetLengths() != paramDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "mismatch between descriptors");
    }

#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr,
            "SetLayerParam layer: %d layerID: %d offset: %lu size: %lu\n",
            layer,
            paramID,
            poffset,
            paramDesc.GetElementSize());
#endif

    // 4. Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, paramDesc, param, paramSrc, w, 0, poffset);
}

void RNNDescriptor::SetLayerBias(const Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& /* wDesc */,
                                 Data_t w,
                                 const int biasID,
                                 const TensorDescriptor& biasDesc,
                                 ConstData_t bias) const
{
    if(biasMode == miopenRNNNoBias)
    {
        return;
    }

    // TODO dlowell: Need guard checks here, or have them caught at the copy call?
    if(bias == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "bias data cannot be null");
    }

    // 1. Calculate the location of the matrix via layerID, bidirection setting, and params
    int x        = static_cast<int>((dirMode == miopenRNNbidirection) ? nLayers * 2 : nLayers);
    auto poffset = paramsOffsetCalculation(xDesc, x, 0);
    auto boffset = biasOffsetCalculation(xDesc, layer, biasID) + poffset;

    // 2. Calculate the strides for the matrix
    std::vector<int> bstride(1, 1);

    std::vector<int> intLens(biasDesc.GetLengths().begin(), biasDesc.GetLengths().end());

    // 3. Construct descriptor to access into w
    auto biasSrc = miopen::TensorDescriptor(dataType, intLens, bstride);

    if(biasSrc.GetLengths() != biasDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "mismatch between descriptors");
    }

#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr,
            "SetLayerBias layer: %d layerID: %d offset: %lu = %lu + %lu size: %lu\n",
            layer,
            biasID,
            boffset,
            poffset,
            boffset - poffset,
            biasSrc.GetElementSize());
#endif

    // 4. Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, biasSrc, bias, biasDesc, w, 0, boffset);
}

void RNNDescriptor::SetPaddingmode(miopenRNNPaddingMode_t padding)
{
    if(padding != miopenRNNIOWithPadding && padding != miopenRNNIONotPadded)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "SetPaddingmode: Bad parameter. RNN padding mode must be "
                     "miopenRNNIOWithPadding or miopenRNNIONotPadded.");
    }

    paddingMode = padding;
}

void RNNDescriptor::GetLayerParamOffset(const int layer,
                                        const TensorDescriptor& xDesc,
                                        const int paramID,
                                        TensorDescriptor& paramDesc,
                                        size_t* paramOffset) const
{
    if(!isNotRNNskip() && (((dirMode != 0u) && layer <= 1 && paramID < nHiddenTensorsPerLayer) ||
                           ((dirMode == 0u) && layer < 1 && paramID < nHiddenTensorsPerLayer)))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Parameter of input layer is null in input skip mode");
    }

    // Get the dimensions of the parameter matrix
    auto pDims = pTensorLengthsCalculation(xDesc, layer, paramID);
    paramDesc  = miopen::TensorDescriptor(dataType, pDims);
    if(paramOffset == nullptr)
    {
        return;
    }

    // Calculate the location of the matrix via paramID, bidirection setting, and params
    *paramOffset = paramsOffsetCalculation(xDesc, layer, paramID);

#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr,
            "GetLayerParamOffset layer: %d layerID: %d offset: %lu size: %lu\n",
            layer,
            paramID,
            *paramOffset,
            paramDesc.GetElementSize());
#endif
}

void RNNDescriptor::GetLayerBiasOffset(const int layer,
                                       const TensorDescriptor& xDesc,
                                       const int biasID,
                                       TensorDescriptor& biasDesc,
                                       size_t* biasOffset) const
{
    // Get the dimensions of the parameter matrix
    if(biasMode == miopenRNNNoBias)
    {
        return;
    }

    auto bdim = hsize;
    biasDesc  = miopen::TensorDescriptor(dataType, {bdim});
    if(biasOffset == nullptr)
    {
        return;
    }

    int x        = static_cast<int>((dirMode == miopenRNNbidirection) ? nLayers * 2 : nLayers);
    auto poffset = paramsOffsetCalculation(xDesc, x, 0);
    *biasOffset  = biasOffsetCalculation(xDesc, layer, biasID) + poffset;

#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr,
            "GetLayerBiasOffset layer: %d layerID: %d offset: %lu = %lu + %lu size: %lu\n",
            layer,
            biasID,
            *biasOffset,
            poffset,
            *biasOffset - poffset,
            biasDesc.GetElementSize());
#endif
}

std::ostream& operator<<(std::ostream& stream, const RNNDescriptor& r)
{
    stream << r.hsize << ", ";
    stream << r.nLayers << ", ";
    stream << r.nHiddenTensorsPerLayer << ", ";
    stream << r.workspaceScale << ", ";
    stream << r.rnnMode << ", ";
    stream << r.dirMode << ", ";
    stream << r.algoMode << ", ";
    stream << r.inputMode << ", ";
    stream << r.biasMode << ", ";
    stream << r.dropoutDesc << ", ";
    return stream;
}

std::tuple<std::vector<unsigned int>, bool>
RNNDescriptor::convertRNNBaseLayout(miopenRNNBaseLayout_t layout)
{
    switch(layout)
    {
    case miopenRNNDataBatchMajorPadded: return {std::vector<unsigned int>{0, 1, 2}, true};
    case miopenRNNDataSeqMajorNotPadded: return {std::vector<unsigned int>{1, 0, 2}, false};
    case miopenRNNDataSeqMajorPadded: return {std::vector<unsigned int>{1, 0, 2}, true};

    case miopenRNNDataUnknownLayout:
    default: MIOPEN_THROW(miopenStatusBadParm, "error: Unknown miopenRNNBaseLayout_t "); break;
    }
}

miopenRNNBaseLayout_t RNNDescriptor::getBaseLayoutFromDataTensor(const SeqTensorDescriptor& desc)
{
    std::initializer_list<miopenRNNBaseLayout_t> base_layouts = {
        miopenRNNDataBatchMajorPadded, miopenRNNDataSeqMajorNotPadded, miopenRNNDataSeqMajorPadded};

    const std::vector<unsigned> desc_dim_order = desc.GetLayoutVector();
    bool desc_seq_is_padded                    = desc.IsPaddedSeqLayout();

    if(desc_dim_order.size() == 3)
    {
        for(auto base_layout : base_layouts)
        {
            const auto [base_dim_order, base_seq_is_padded] = convertRNNBaseLayout(base_layout);

            bool layout_equal =
                (desc_seq_is_padded == base_seq_is_padded) &&
                std::equal(desc_dim_order.begin(), desc_dim_order.end(), base_dim_order.begin());

            if(layout_equal)
                return base_layout;
        }
    }
    return miopenRNNDataUnknownLayout;
}

SeqTensorDescriptor RNNDescriptor::makeSeqTensorDescriptor(miopenDataType_t t,
                                                           miopenRNNBaseLayout_t layout,
                                                           int maxSeqLength,
                                                           int batchSize,
                                                           int vectorSize,
                                                           const int* lensPerSeq,
                                                           const void* padding_marker_ptr)
{
    const std::vector<int> lens = {batchSize, maxSeqLength, vectorSize};

    const auto [dim_order, padded_sequences] = convertRNNBaseLayout(layout);

    std::vector<char> padding_marker_in;
    if(padding_marker_ptr != nullptr)
    {
        auto t_sz = GetTypeSize(t);
        padding_marker_in.resize(t_sz);
        std::copy_n(
            reinterpret_cast<const char*>(padding_marker_ptr), t_sz, padding_marker_in.data());
    }

    return {t,
            dim_order,
            lens,
            std::vector<int>(lensPerSeq, lensPerSeq + batchSize),
            padding_marker_in,
            true,
            padded_sequences};
}

SeqTensorDescriptor
RNNDescriptor::makeSeqTensorDescriptor(c_array_view<const miopenTensorDescriptor_t> descs,
                                       size_t seq_len,
                                       miopenRNNBaseLayout_t layout)
{
    assert(layout == miopenRNNDataSeqMajorNotPadded || layout == miopenRNNDataSeqMajorPadded);

    auto max_batch = descs[0].GetLengths()[0];
    auto vec_size  = descs[0].GetLengths()[1];

    std::vector<size_t> lens_per_seq;
    lens_per_seq.reserve(max_batch);
    auto push_back_n_lens = [&lens_per_seq](auto N, auto seq) {
        for(auto i = N; i > 0; --i)
            lens_per_seq.push_back(seq);
    };

    std::vector<size_t> batch_cache =
        [](c_array_view<const miopenTensorDescriptor_t> const& descs_array) {
            auto size = descs_array.size();
            std::vector<size_t> vec;
            vec.reserve(size);
            for(size_t i = 0; i < size; ++i)
                vec.push_back(descs_array[i].GetLengths()[0]);
            return vec;
        }(descs);

    size_t last_cnt = 0;
    auto it         = batch_cache.rbegin();

    while(it != batch_cache.rend())
    {
        const auto new_cnt = *it;

        if(new_cnt != 0)
            push_back_n_lens(new_cnt - last_cnt, std::distance(it, batch_cache.rend()));

        last_cnt = new_cnt;
        it       = std::lower_bound(it, batch_cache.rend(), *it, std::less_equal<size_t>{});
    }

    const std::vector<size_t> lens = {max_batch, seq_len, vec_size};

    const auto [dim_order, padded_sequences] = convertRNNBaseLayout(layout);

    return {descs[0].GetType(),
            dim_order,
            lens,
            lens_per_seq,
            std::vector<char>{},
            true,
            padded_sequences};
}

void RNNDescriptor::SeqTensorToTensorDescArray(const SeqTensorDescriptor& desc,
                                               std::vector<miopen::TensorDescriptor>& td,
                                               std::vector<miopenTensorDescriptor_t>& ptd)
{
    if(!desc.IsPacked())
        MIOPEN_THROW(miopenStatusInternalError, "Only packed SeqTensorDescriptor supported.");

    const std::vector<size_t> bs = desc.GetBatchesPerSequence();
    const size_t vector_size     = desc.GetLengths()[2];
    const auto data_type         = desc.GetType();

    td.reserve(bs.size());
    ptd.reserve(bs.size());

    std::transform(
        bs.begin(), bs.end(), std::back_inserter(td), [data_type, vector_size](size_t batch_size) {
            return miopen::TensorDescriptor(data_type, {batch_size, vector_size});
        });
    std::transform(td.begin(), td.end(), std::back_inserter(ptd), [](miopen::TensorDescriptor& x) {
        return &x;
    });
}

void RNNDescriptor::RNNVanillaForward(Handle& handle,
                                      miopenRNNFWDMode_t fwdMode,
                                      ConstData_t w,
                                      const SeqTensorDescriptor& xDesc,
                                      ConstData_t x,
                                      const TensorDescriptor& hDesc,
                                      ConstData_t hx,
                                      Data_t hy,
                                      const TensorDescriptor& cDesc,
                                      ConstData_t cx,
                                      Data_t cy,
                                      const SeqTensorDescriptor& yDesc,
                                      Data_t y,
                                      Data_t workSpace,
                                      size_t workSpaceSize,
                                      Data_t reserveSpace,
                                      size_t reserveSpaceSize) const
{
    std::vector<miopen::TensorDescriptor> input_cpp_descs, output_cpp_descs;
    std::vector<miopenTensorDescriptor_t> input_descs, output_descs;

    SeqTensorToTensorDescArray(xDesc, input_cpp_descs, input_descs);
    SeqTensorToTensorDescArray(yDesc, output_cpp_descs, output_descs);

    auto seq_len = input_descs.size();

    miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{input_descs.data(), seq_len};
    miopen::c_array_view<const miopenTensorDescriptor_t> yDescArray{output_descs.data(), seq_len};

    if(fwdMode == miopenRNNFWDMode_t::miopenRNNTraining)
    {
        return RNNForwardTrainingPackedTensors(handle,
                                               seq_len,
                                               xDescArray,
                                               x,
                                               hDesc,
                                               hx,
                                               cDesc,
                                               cx,
                                               TensorDescriptor(this->dataType, {1, 1})
                                               /* wDesc used only for GetType()*/,
                                               w,
                                               yDescArray,
                                               y,
                                               hDesc,
                                               hy,
                                               cDesc,
                                               cy,
                                               reserveSpace,
                                               reserveSpaceSize);
    }
    else
    {
        return RNNForwardInferencePacked(handle,
                                         seq_len,
                                         xDescArray,
                                         x,
                                         hDesc,
                                         hx,
                                         cDesc,
                                         cx,
                                         TensorDescriptor(this->dataType, {1, 1})
                                         /* wDesc used only for GetType()*/,
                                         w,
                                         yDescArray,
                                         y,
                                         hDesc,
                                         hy,
                                         cDesc,
                                         cy,
                                         workSpace,
                                         workSpaceSize);
    }
}

void RNNDescriptor::RNNVanillaBackwardData(Handle& handle,
                                           const SeqTensorDescriptor& yDesc,
                                           ConstData_t dy,
                                           const TensorDescriptor& hDesc,
                                           ConstData_t hx,
                                           ConstData_t dhy,
                                           Data_t dhx,
                                           const TensorDescriptor& cDesc,
                                           ConstData_t cx,
                                           ConstData_t dcy,
                                           Data_t dcx,
                                           const SeqTensorDescriptor& xDesc,
                                           Data_t dx,
                                           ConstData_t w,
                                           Data_t workSpace,
                                           size_t workSpaceSize,
                                           Data_t reserveSpace,
                                           size_t reserveSpaceSize) const
{
    std::vector<miopen::TensorDescriptor> input_cpp_descs, output_cpp_descs;
    std::vector<miopenTensorDescriptor_t> input_descs, output_descs;

    SeqTensorToTensorDescArray(xDesc, input_cpp_descs, input_descs);
    SeqTensorToTensorDescArray(yDesc, output_cpp_descs, output_descs);

    auto seq_len = input_descs.size();

    miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{input_descs.data(), seq_len};
    miopen::c_array_view<const miopenTensorDescriptor_t> yDescArray{output_descs.data(), seq_len};

    return RNNBackwardDataPackedTensors(handle,
                                        seq_len,
                                        yDescArray,
                                        dy,
                                        dhy,
                                        dcy,
                                        w,
                                        hx,
                                        cx,
                                        xDescArray,
                                        dx,
                                        hDesc,
                                        dhx,
                                        cDesc,
                                        dcx,
                                        workSpace,
                                        workSpaceSize,
                                        reserveSpace,
                                        reserveSpaceSize);
}

void RNNDescriptor::RNNVanillaBackwardWeights(Handle& handle,
                                              const SeqTensorDescriptor& xDesc,
                                              ConstData_t x,
                                              const TensorDescriptor& hDesc,
                                              ConstData_t hx,
                                              const SeqTensorDescriptor& yDesc,
                                              Data_t dw,
                                              Data_t workSpace,
                                              size_t workSpaceSize,
                                              ConstData_t reserveSpace,
                                              size_t reserveSpaceSize) const
{
    std::vector<miopen::TensorDescriptor> input_cpp_descs, output_cpp_descs;
    std::vector<miopenTensorDescriptor_t> input_descs, output_descs;

    SeqTensorToTensorDescArray(xDesc, input_cpp_descs, input_descs);
    SeqTensorToTensorDescArray(yDesc, output_cpp_descs, output_descs);

    auto seq_len = input_descs.size();

    miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{input_descs.data(), seq_len};
    miopen::c_array_view<const miopenTensorDescriptor_t> yDescArray{output_descs.data(), seq_len};

    return RNNBackwardWeightsPackedTensors(handle,
                                           seq_len,
                                           xDescArray,
                                           x,
                                           hDesc,
                                           hx,
                                           yDescArray,
                                           TensorDescriptor(this->dataType, {1, 1})
                                           /* wDesc used only for GetType()*/,
                                           dw,
                                           workSpace,
                                           workSpaceSize,
                                           reserveSpace,
                                           reserveSpaceSize);
}

void RNNDescriptor::RNNForward(Handle& handle,
                               miopenRNNFWDMode_t fwdMode,
                               const SeqTensorDescriptor& xDesc,
                               ConstData_t x,
                               const TensorDescriptor& hDesc,
                               ConstData_t hx,
                               Data_t hy,
                               const TensorDescriptor& cDesc,
                               ConstData_t cx,
                               Data_t cy,
                               const SeqTensorDescriptor& yDesc,
                               Data_t y,
                               ConstData_t w,
                               size_t weightSpaceSize,
                               Data_t workSpace,
                               size_t workSpaceSize,
                               Data_t reserveSpace,
                               size_t reserveSpaceSize) const
{
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(hDesc.GetNumDims() != cDesc.GetNumDims())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(reserveSpaceSize < GetMaxReserveSize(handle, xDesc))
    {
        MIOPEN_THROW("Reservespace is required");
    }

    if(workSpaceSize < GetMaxWorkspaceSize(handle, xDesc, fwdMode))
    {
        MIOPEN_THROW("Workspace is required");
    }
    if(weightSpaceSize < GetParamsSize(xDesc.GetLengths()[2]))
    {
        MIOPEN_THROW("WeightSpace is too small");
    }

#if MIOPEN_BACKEND_HIP
    RnnHipAutoProfiler kernel_profiler{handle};

    try
    {
#endif
        const auto xDesc_base_layout = RNNDescriptor::getBaseLayoutFromDataTensor(xDesc);

        if(xDesc_base_layout == miopenRNNDataSeqMajorNotPadded)
        {
            RNNVanillaForward(handle,
                              fwdMode,
                              w,
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
                              workSpace,
                              workSpaceSize,
                              reserveSpace,
                              reserveSpaceSize);
        }
        else
        {
            RNNTransformerForward(handle,
                                  fwdMode,
                                  w,
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
                                  workSpace,
                                  workSpaceSize,
                                  reserveSpace,
                                  reserveSpaceSize);
        }

#if MIOPEN_BACKEND_HIP
    }
    catch(...)
    {
        kernel_profiler.abortProfiling();
        throw;
    }

#endif
}

void RNNDescriptor::RNNBackwardData(Handle& handle,
                                    const SeqTensorDescriptor& yDesc,
                                    ConstData_t,
                                    ConstData_t dy,
                                    const TensorDescriptor& hDesc,
                                    ConstData_t hx,
                                    ConstData_t dhy,
                                    Data_t dhx,
                                    const TensorDescriptor& cDesc,
                                    ConstData_t cx,
                                    ConstData_t dcy,
                                    Data_t dcx,
                                    const SeqTensorDescriptor& xDesc,
                                    Data_t dx,
                                    ConstData_t w,
                                    size_t weightSpaceSize,
                                    Data_t workSpace,
                                    size_t workSpaceSize,
                                    Data_t reserveSpace,
                                    size_t reserveSpaceSize) const
{

    if(dx == nullptr || w == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(hDesc.GetNumDims() != cDesc.GetNumDims())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(reserveSpaceSize < GetMaxReserveSize(handle, xDesc))
    {
        MIOPEN_THROW("Reservespace is required");
    }

    if(workSpaceSize < GetMaxWorkspaceSize(handle, xDesc, miopenRNNTraining))
    {
        MIOPEN_THROW("Workspace is required");
    }

    if(weightSpaceSize < GetParamsSize(xDesc.GetLengths()[2]))
    {
        MIOPEN_THROW("WeightSpace is too small");
    }

#if MIOPEN_BACKEND_HIP
    RnnHipAutoProfiler kernel_profiler{handle};

    try
    {
#endif
        const auto xDesc_base_layout = RNNDescriptor::getBaseLayoutFromDataTensor(xDesc);

        if(xDesc_base_layout == miopenRNNDataSeqMajorNotPadded)
        {
            RNNVanillaBackwardData(handle,
                                   yDesc,
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
                                   workSpaceSize,
                                   reserveSpace,
                                   reserveSpaceSize);
        }
        else
        {
            RNNTransformerBackwardData(handle,
                                       yDesc,
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
                                       workSpaceSize,
                                       reserveSpace,
                                       reserveSpaceSize);
        }

#if MIOPEN_BACKEND_HIP
    }
    catch(...)
    {
        kernel_profiler.abortProfiling();

        throw;
    }
#endif
}

void RNNDescriptor::RNNBackwardWeights(Handle& handle,
                                       const SeqTensorDescriptor& xDesc,
                                       ConstData_t x,
                                       const TensorDescriptor& hDesc,
                                       ConstData_t hx,
                                       const SeqTensorDescriptor& yDesc,
                                       ConstData_t,
                                       Data_t dw,
                                       size_t weightSpaceSize,
                                       Data_t workSpace,
                                       size_t workSpaceSize,
                                       ConstData_t reserveSpace,
                                       size_t reserveSpaceSize) const
{

    if(x == nullptr || dw == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(reserveSpaceSize < GetMaxReserveSize(handle, xDesc))
    {
        MIOPEN_THROW("Reservespace is required");
    }

    if(workSpaceSize < GetMaxWorkspaceSize(handle, xDesc, miopenRNNTraining))
    {
        MIOPEN_THROW("Workspace is required");
    }

    if(weightSpaceSize < GetParamsSize(xDesc.GetLengths()[2]))
    {
        MIOPEN_THROW("WeightSpace is too small");
    }

#if MIOPEN_BACKEND_HIP
    RnnHipAutoProfiler kernel_profiler{handle};

    try
    {
#endif
        const auto xDesc_base_layout = RNNDescriptor::getBaseLayoutFromDataTensor(xDesc);

        if(xDesc_base_layout == miopenRNNDataSeqMajorNotPadded)
        {
            RNNVanillaBackwardWeights(handle,
                                      xDesc,
                                      x,
                                      hDesc,
                                      hx,
                                      yDesc,
                                      dw,
                                      workSpace,
                                      workSpaceSize,
                                      reserveSpace,
                                      reserveSpaceSize);
        }
        else
        {
            RNNTransformerBackwardWeights(handle,
                                          xDesc,
                                          x,
                                          hDesc,
                                          hx,
                                          yDesc,
                                          dw,
                                          workSpace,
                                          workSpaceSize,
                                          reserveSpace,
                                          reserveSpaceSize);
        }

#if MIOPEN_BACKEND_HIP
    }
    catch(...)
    {
        kernel_profiler.abortProfiling();
        throw;
    }
#endif
}

} // namespace miopen
