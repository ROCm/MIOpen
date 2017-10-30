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

#include <miopen/rnn.hpp>
#include <miopen/errors.hpp>
#include <miopen/env.hpp>

// MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
// MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

namespace miopen {

size_t RNNDescriptor::biasOffsetCalculation(const TensorDescriptor& xDesc,
                                            const TensorDescriptor& wDesc,
                                            const int layer,
                                            const int layerID)
{
    size_t layerJump       = 0;
    size_t inVectorSize    = xDesc.GetLengths()[1];
    unsigned int direction = (layer % 2);
    if(dirMode)
    {
        if(layer > 1) // NOT the input layer
        {

            // jump over input layer's input matrices
            layerJump = 2 * inVectorSize * hsize * nHiddenTensorsPerLayer;

            // jump over input layer's hidden matrices
            layerJump += 2 * hsize * hsize * nHiddenTensorsPerLayer;

            // jump over all other layers' matrices
            layerJump += 4 * hsize * hsize * nHiddenTensorsPerLayer * (nLayers - 2);

            // jump over all other biases
            layerJump += 2 * hsize * nHiddenTensorsPerLayer * layer;

            // forward or backward direction bias
            layerJump += direction * hsize * nHiddenTensorsPerLayer;

            // jump to either the input bias, or weight bias
            layerJump += 2 * hsize * layerID * nHiddenTensorsPerLayer;

            return layerJump;
        }
        else // IS the input layer
        {
            // jump over input layer's input bias
            layerJump = layerID * 2 * inVectorSize * hsize * nHiddenTensorsPerLayer;

            // forward or backward direction bias
            layerJump += direction * hsize * nHiddenTensorsPerLayer;

            return layerJump;
        }
    }
    else
    {
        if(layer > 1) // NOT the input layer
        {
            // jump over input layer's input matrices
            layerJump = inVectorSize * hsize * nHiddenTensorsPerLayer;

            // jump over input layer's hidden matrices
            layerJump += hsize * hsize * nHiddenTensorsPerLayer;

            // jump over all other layers' matrices
            layerJump += 2 * hsize * hsize * nHiddenTensorsPerLayer * nLayers;

            // jump to either the input bias, or weight bias
            layerJump += hsize * layerID * nHiddenTensorsPerLayer;

            return layerJump;
        }
        else
        {
            // jump over input layer's input bias
            layerJump = layerID * inVectorSize * hsize * nHiddenTensorsPerLayer;

            return layerJump;
        }
    }
}

size_t RNNDescriptor::paramsOffsetCalculation(const TensorDescriptor& xDesc,
                                              const TensorDescriptor& wDesc,
                                              const int layer,
                                              const int layerID)
{
    size_t layerJump       = 0;
    size_t inVectorSize    = xDesc.GetLengths()[1];
    unsigned int direction = (layer % 2);
    if(dirMode)
    {
        if(layer > 1) // NOT the input layer
        {
            // jump over input layer's input matrices
            layerJump = 2 * inVectorSize * hsize * nHiddenTensorsPerLayer;

            // jump over input layer's hidden matrices
            layerJump += 2 * hsize * hsize * nHiddenTensorsPerLayer;

            // jump over all other layers' matrices
            layerJump += 4 * hsize * hsize * nHiddenTensorsPerLayer * (layer - 2);

            // forward or backward direction matrix
            layerJump += direction * hsize * nHiddenTensorsPerLayer;

            // jump to either the input matrix, or weight matrix
            layerJump += 2 * hsize * hsize * layerID * nHiddenTensorsPerLayer;

            return layerJump;
        }
        else // IS the input layer
        {
            // jump over input layer's input matrices
            layerJump = layerID * 2 * inVectorSize * hsize * nHiddenTensorsPerLayer;

            // forward or backward direction matrix
            layerJump += direction * hsize * nHiddenTensorsPerLayer;

            return layerJump;
        }
    }
    else
    {
        if(layer > 1) // NOT the input layer
        {
            // jump over input layer's input matrices
            layerJump = inVectorSize * hsize * nHiddenTensorsPerLayer;

            // jump over input layer's hidden matrices
            layerJump += hsize * hsize * nHiddenTensorsPerLayer;

            // jump over all other layers' matrices
            layerJump += 2 * hsize * hsize * nHiddenTensorsPerLayer * (layer - 1);

            // jump to either the input matrix, or weight matrix
            layerJump += hsize * hsize * layerID * nHiddenTensorsPerLayer;

            return layerJump;
        }
        else
        {
            // jump over input layer's input matrices
            layerJump = layerID * inVectorSize * hsize * nHiddenTensorsPerLayer;

            return layerJump;
        }
    }
}

RNNDescriptor::RNNDescriptor()
{
    nLayers                = 1;
    hsize                  = 0;
    inputBatchLenSum       = 0;
    nHiddenTensorsPerLayer = 0;
    rnnMode                = miopenRNNTANH;
    dirMode                = miopenRNNunidirection;
    biasMode               = miopenRNNNoBias;
    algoMode               = miopenRNNdefault;
    inputMode              = miopenRNNlinear;
    dataType               = miopenFloat;
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
        MIOPEN_THROW(miopenStatusBadParm, "Parameter to RNN must be a positive integer.");
    }
    if(!(rmode == miopenRNNRELU || rmode == miopenRNNTANH || rmode == miopenLSTM ||
         rmode == miopenGRU))
    {
        MIOPEN_THROW(miopenStatusBadParm, "RNN mode not supported");
    }
    if(bidir != 0 && bidir != 1)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Parameters to RNN directional type not supported");
    }
    if(bmode != 0 && bmode != 1)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Parameters to RNN bias type not supported");
    }
    if(dType != miopenFloat)
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Only float datatype is supported");
    }

    hsize     = hsz;
    nLayers   = layers;
    inputMode = inMode;
    dirMode   = bidir;
    rnnMode   = rmode;
    algoMode  = amode;
    biasMode  = bmode;
    dataType  = dType;

    assert(rmode < 4);
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
    inputBatchLenSum = 0; // init
}

size_t RNNDescriptor::GetWorkspaceSize(Handle& handle,
                                       const int sLen,
                                       c_array_view<miopenTensorDescriptor_t> xDesc)
{
    // NOTE dlowell: this calculation WILL change during development.
    // currently this is calculated the same as Workspace size
    // x = maxSequenceLen * batchSize * vector_size * numLayers * bytesForDataType *
    // numberOfHiddenMatricesPerCell + Extra
    // GetElemSize will get vector len * batch_size
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }

    if(!inputBatchLenSum)
    {
        for(int i = 0; i < sLen; i++)
        {
            inputBatchLenSum += xDesc[i].GetLengths()[0];
        }
    }

    auto x = workspaceScale * nLayers * inputBatchLenSum * hsize * sizeof(xDesc[0].GetType());
    return dirMode == miopenRNNbidirection ? size_t(2 * x) : size_t(x);
}

size_t RNNDescriptor::GetReserveSize(Handle& handle,
                                     const int sLen,
                                     c_array_view<miopenTensorDescriptor_t> xDesc)
{
    // NOTE dlowell: this calculation WILL change during development.
    // x = maxSequenceLen * batchSize * vector_size * numLayers * bytesForDataType *
    // numberOfHiddenMatricesPerCell + Extra
    // GetElemSize will get vector len * batch_size
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    if(!inputBatchLenSum)
    {
        for(int i = 0; i < sLen; i++)
        {
            inputBatchLenSum += xDesc[i].GetLengths()[0];
        }
    }

    // auto x = workspaceScale * nLayers * inputBatchLenSum * hsize * sizeof(xDesc[0].GetType());
    auto x = 2 * workspaceScale * nLayers * inputBatchLenSum * hsize *
             sizeof(xDesc[0].GetType()); // switch to this after offset activ and ops applied
    return dirMode == miopenRNNbidirection ? size_t(2 * x) : size_t(x);
}

size_t RNNDescriptor::GetParamsSize(Handle& handle,
                                    const TensorDescriptor& xDesc,
                                    miopenDataType_t dtype) const
{
    // DLOWELL : The factor of 4 counts the input matrix, hidden matrix, input bias, hidden bias
    // to each of the activated section of the RNN cell.
    // h_t = sigma(Wx_t + Rh_t-1 + bw + br)
    // for one layer: wDesc <-- (v_hidden x v_input) + (v_hidden x v_hidden) + 2*(1 x v_hidden)
    assert(xDesc.GetLengths().size() > 1);
    auto ih = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        ih  = 0;
    int bi  = dirMode == miopenRNNbidirection ? 2 : 1;
    auto sz = nHiddenTensorsPerLayer * hsize * bi * (ih + hsize + (nLayers - 1) * (bi + 1) * hsize);
    if(biasMode == miopenRNNwithBias)
    {
        auto in_bias = inputMode == miopenRNNskip ? 1 : 2;
        sz += (in_bias + (nLayers - 1) * (bi + 1)) * nHiddenTensorsPerLayer * hsize * bi;
    }
    return size_t(sz);
}

size_t RNNDescriptor::GetRNNInputSuperTensorSize(Handle& handle,
                                                 const int seqLength,
                                                 c_array_view<miopenTensorDescriptor_t> xDesc)
{
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    if(!inputBatchLenSum)
    {
        for(int i = 0; i < seqLength; i++)
        {
            inputBatchLenSum += xDesc[i].GetLengths()[0];
        }
    }
    auto x = inputBatchLenSum * xDesc[0].GetLengths()[1];
    return size_t(x);
}

size_t RNNDescriptor::GetRNNHiddenSuperTensorSize(Handle& handle,
                                                  c_array_view<miopenTensorDescriptor_t> xDesc)
{
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    auto x = xDesc[0].GetLengths()[0] * hsize * nLayers;
    return dirMode == miopenRNNbidirection ? size_t(2 * x) : size_t(x);
}

/* Get weight super tensor size
temporary function assuming output matrix exists */
size_t RNNDescriptor::GetRNNWeightSuperTensorSize(Handle& handle,
                                                  const TensorDescriptor& xDesc,
                                                  const TensorDescriptor& yDesc) const
{
    auto ih = xDesc.GetLengths()[1], oh = yDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        ih = 0;
    int bi = (dirMode == miopenRNNbidirection) ? 2 : 1;
    auto sz =
        nHiddenTensorsPerLayer * hsize * bi * (ih + hsize + (nLayers - 1) * (bi + 1) * hsize) +
        oh * hsize * bi;
    if(biasMode == miopenRNNwithBias)
    {
        auto in_bias = inputMode == miopenRNNskip ? 1 : 2;
        sz += (in_bias + (nLayers - 1) * (bi + 1)) * nHiddenTensorsPerLayer * hsize * bi + bi * oh;
    }

    return size_t(sz);
}

void RNNDescriptor::GetParamsDescriptor(Handle& handle,
                                        const TensorDescriptor& xDesc,
                                        TensorDescriptor& wDesc,
                                        miopenDataType_t dtype) const
{

    int inputVecLen = xDesc.GetLengths()[1]; // input vector size
    int bi          = (dirMode == miopenRNNbidirection) ? 2 : 1;
    std::array<int, 2> weight_lens;

    // NOTE dlowell: This is my calc, instead using Jing's
    //    if(biasMode == miopenRNNNoBias)
    //    {
    //        weight_lens[0] = inputVecLen + (2 * nLayers - 1) * hsize;
    //    }
    //    else
    //    {
    //        weight_lens[0] = inputVecLen + (2 * nLayers - 1) * hsize + 2 * nLayers;
    //    }

    // Jing's calculations
    weight_lens[0] = inputVecLen + ((nLayers - 1) * (bi + 1) + 1) * hsize;
    weight_lens[1] = bi * hsize * nHiddenTensorsPerLayer;

    miopenSetTensorDescriptor(&wDesc, miopenFloat, 2, weight_lens.data(), nullptr);
}

void RNNDescriptor::GetLayerParam(Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& wDesc,
                                  ConstData_t w,
                                  const int layerID,
                                  TensorDescriptor& paramDesc,
                                  Data_t param) const
{
    // input is
    // 1. Calculate the location of the matrix via layerID, bidirection setting, and params

    // 2. Construct descriptor for param matrix
    // 3. Copy over data to previously allocated param tensor
}

void RNNDescriptor::GetLayerBias(Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& wDesc,
                                 ConstData_t w,
                                 const int layerID,
                                 TensorDescriptor& biasDesc,
                                 Data_t bias) const
{
    // TODO: FILL
}

void RNNDescriptor::SetLayerParam(Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& wDesc,
                                  ConstData_t w,
                                  const int layerID,
                                  const TensorDescriptor& paramDesc,
                                  ConstData_t param)
{
    // 1. Find the location of the matrix via layer and
}

void RNNDescriptor::SetLayerBias(Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& wDesc,
                                 ConstData_t w,
                                 const int layerID,
                                 const TensorDescriptor& biasDesc,
                                 ConstData_t bias)
{
    // TODO: FILL
}

std::ostream& operator<<(std::ostream& stream, const RNNDescriptor& r)
{
    stream << r.hsize << ", ";
    stream << r.nLayers << ", ";
    stream << r.nHiddenTensorsPerLayer << ", ";
    stream << r.workspaceScale << ", ";
    stream << r.inputBatchLenSum << ", ";
    return stream;
}

} // namespace miopen

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#endif
