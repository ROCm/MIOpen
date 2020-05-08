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

#include <miopen/handle.hpp>
#include <miopen/rnn.hpp>

#include <cassert>
#include <cstddef>
#include <numeric>
#include <ostream>

// MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
// MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)

// Disable specific warnings
#define MIO_RNN_DEBUG 0

#define MIOPEN_RNN_SYNCH 0
#define MIO_RNN_CPP_PROF 0

namespace miopen {

void profileRNNkernels(Handle& handle, unsigned char select, float& ctime)
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
            ktime = handle.GetKernelTime();
            ctime = ktime;

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
            handle.GetKernelTime();
            handle.AccumKernelTime(ctime);
#endif
        }
        break;
    default: assert(false);
    }
}

size_t RNNDescriptor::biasOffsetCalculation(const TensorDescriptor& /*xDesc*/,
                                            const int layer,
                                            const int biasID)
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
                                              const int paramID)
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
                                                          const int paramID)
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

size_t RNNDescriptor::GetWorkspaceSize(Handle& /* handle */,
                                       const int seqLength,
                                       c_array_view<const miopenTensorDescriptor_t> xDesc) const
{

    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }

    std::size_t inputBatchLenSum = 0;
    inputBatchLenSum             = std::accumulate(
        xDesc.data, xDesc.data + seqLength, 0, [](size_t x, miopenTensorDescriptor_t y) {
            return x + deref(y).GetLengths()[0];
        });
    auto x = workspaceScale * nLayers * inputBatchLenSum * hsize * typeSize;
    return size_t(dirMode == miopenRNNbidirection ? 2 * x : x);
}

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
        xDesc.data, xDesc.data + seqLength, 0, [](size_t x, miopenTensorDescriptor_t y) {
            return x + deref(y).GetLengths()[0];
        });
    auto x = 2 * workspaceScale * nLayers * inputBatchLenSum * hsize * typeSize;
    if(algoMode == miopenRNNdefault && rnnMode == miopenLSTM)
    {
        x /= 2;
        x += nLayers * inputBatchLenSum * hsize * typeSize;
    }
    if(!float_equal(miopen::deref(dropoutDesc).dropout, 0))
    {
        x += (nLayers - 1) * inputBatchLenSum * hsize * typeSize;
        x += (nLayers - 1) * inputBatchLenSum * hsize * sizeof(bool);
    }
    return size_t(dirMode == miopenRNNbidirection ? 2 * x : x);
}

size_t RNNDescriptor::GetParamsSize(Handle& /* handle */,
                                    const TensorDescriptor& xDesc,
                                    miopenDataType_t dtype)
{
    if(xDesc.GetType() != dataType || dtype != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch.");
    }
    assert(xDesc.GetLengths().size() > 1);
    auto inputVectorLen = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        inputVectorLen = 0;

    int bi  = dirMode == miopenRNNbidirection ? 2 : 1;
    auto sz = nHiddenTensorsPerLayer * hsize * bi *
              (inputVectorLen + hsize + (nLayers - 1) * (bi + 1) * hsize);
#if(MIO_RNN_DEBUG == 1)
    fprintf(stderr, "weight size: %lu\n", sz);
#endif
    if(biasMode == miopenRNNwithBias)
    {
        sz += nLayers * 2 * nHiddenTensorsPerLayer * hsize * bi;
    }
    return size_t(typeSize * sz);
}

size_t RNNDescriptor::GetRNNInputSuperTensorSize(Handle& /* handle */,
                                                 const int seqLength,
                                                 c_array_view<miopenTensorDescriptor_t> xDesc)
{
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    std::size_t inputBatchLenSum = 0;
    inputBatchLenSum             = std::accumulate(
        xDesc.data, xDesc.data + seqLength, 0, [](size_t x, miopenTensorDescriptor_t y) {
            return x + deref(y).GetLengths()[0];
        });
    auto x = inputBatchLenSum * xDesc[0].GetLengths()[1] * typeSize;
    return size_t(x);
}

size_t RNNDescriptor::GetRNNHiddenSuperTensorSize(Handle& /* handle */,
                                                  c_array_view<miopenTensorDescriptor_t> xDesc)
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
                                        miopenDataType_t dtype)
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

    wDesc = miopen::TensorDescriptor(dtype, weight_lens.data(), 2);
}

std::size_t RNNDescriptor::GetLayerParamSize(Handle& /*handle*/,
                                             int layer,
                                             const TensorDescriptor& xDesc,
                                             int paramID)
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

std::size_t RNNDescriptor::GetLayerBiasSize(Handle& /* handle */, int /*layer*/, int /*biasID*/)
{
    return size_t(typeSize * hsize); // is ther more needed here?
}

void RNNDescriptor::GetLayerParam(Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& /* wDesc */,
                                  ConstData_t w,
                                  const int paramID,
                                  TensorDescriptor& paramDesc,
                                  Data_t param)
{

    if(!isNotRNNskip() && (((dirMode != 0u) && layer <= 1 && paramID < nHiddenTensorsPerLayer) ||
                           ((dirMode == 0u) && layer < 1 && paramID < nHiddenTensorsPerLayer)))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Parameter of input layer is null in input skip mode");
    }

    // Get the dimensions of the parameter matrix
    auto pDims = pTensorLengthsCalculation(xDesc, layer, paramID);
    paramDesc  = miopen::TensorDescriptor(dataType, pDims.data(), 2);
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

void RNNDescriptor::GetLayerBias(Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& /* wDesc */,
                                 ConstData_t w,
                                 const int biasID,
                                 TensorDescriptor& biasDesc,
                                 Data_t bias)
{
    if(biasMode == miopenRNNNoBias)
    {
        return;
    }

    // Get the dimensions of the parameter matrix
    auto bdim = int(hsize);
    biasDesc  = miopen::TensorDescriptor(dataType, &bdim, 1);
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

void RNNDescriptor::SetLayerParam(Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& /* wDesc */,
                                  Data_t w,
                                  const int paramID,
                                  const TensorDescriptor& paramDesc,
                                  ConstData_t param)
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
    auto paramSrc = miopen::TensorDescriptor(dataType, intLens.data(), pstride.data(), 2);

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

void RNNDescriptor::SetLayerBias(Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& /* wDesc */,
                                 Data_t w,
                                 const int biasID,
                                 const TensorDescriptor& biasDesc,
                                 ConstData_t bias)
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
    auto biasSrc = miopen::TensorDescriptor(dataType, intLens.data(), bstride.data(), 1);

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

void RNNDescriptor::GetLayerParamOffset(const int layer,
                                        const TensorDescriptor& xDesc,
                                        const int paramID,
                                        TensorDescriptor& paramDesc,
                                        size_t* paramOffset)
{
    if(!isNotRNNskip() && (((dirMode != 0u) && layer <= 1 && paramID < nHiddenTensorsPerLayer) ||
                           ((dirMode == 0u) && layer < 1 && paramID < nHiddenTensorsPerLayer)))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Parameter of input layer is null in input skip mode");
    }

    // Get the dimensions of the parameter matrix
    auto pDims = pTensorLengthsCalculation(xDesc, layer, paramID);
    paramDesc  = miopen::TensorDescriptor(dataType, pDims.data(), 2);
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
                                       size_t* biasOffset)
{
    // Get the dimensions of the parameter matrix
    if(biasMode == miopenRNNNoBias)
    {
        return;
    }

    auto bdim = int(hsize);
    biasDesc  = miopen::TensorDescriptor(dataType, &bdim, 1);
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

} // namespace miopen
