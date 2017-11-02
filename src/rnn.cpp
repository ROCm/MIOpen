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

#define MIOPEN_RNN_SYNCH 0
#define MIO_RNN_CPP_PROF 1

namespace miopen {
    
    
void profileSequence(Handle& handle, unsigned char select)
{

    float ktime        = 0.;
    static float ctime = 0.;
    assert((select < 3 && select >= 0) && "profileSequence case incorrect");
    switch(select)
    {

    case 0:
        if(handle.IsProfilingEnabled())
        {
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
            ktime = handle.GetKernelTime();
            handle.AccumKernelTime(ctime);
            printf("Final time: %f\n", ktime+ctime);
        } 
        break;
    }
}
    
    
void profileSequence(Handle& handle, float& ctime, const int start, const int finish, const bool init)
{
    // Select the profile mode
    // Finish in this function needs to be 1 larger than the max value start.
    unsigned char select = (!start && !finish)? 1 : ((start==0) ? 0 : (start==(finish-1)) ? 2 : 1);
    printf("Init select: %d\n",select);
    // Start of loop and not the first kernel call, 
    // or if at the last loop, but we ARE the first call
    // then do intermediate time update
    select = ((!select && !init) || (select==2 && init)) ? 1 : select;
    printf("Post select: %d\n",select);
    float ktime        = 0.;
    switch(select)
    {

    case 0: // Initial call
        if(handle.IsProfilingEnabled())
        {
            ktime = handle.GetKernelTime();
            ctime = ktime;

#if(MIO_RNN_CPP_PROF == 1)
            printf("ktime: %f\n", ktime);
       //     printf("ctime: %f\n", ctime);
#endif
        }
#if(MIOPEN_RNN_SYNCH == 1)
        else // explicit barrier, but OCL spec compliance not enforced...
        {
            handle.Finish();
        }
#endif
        break;
        
    case 1: // Intermediate call
        if(handle.IsProfilingEnabled())
        {
            ktime = handle.GetKernelTime();
            ctime += ktime;
#if(MIO_RNN_CPP_PROF == 1)
            printf("ktime: %f\n", ktime);
         //   printf("intermediate ctime: %f\n", ctime);
#endif
        }
        break;

    case 2: // Final call
        if(handle.IsProfilingEnabled())
        {
            ktime = handle.GetKernelTime();
            #if(MIO_RNN_CPP_PROF == 1)
         //   printf("ktime: %f\n", ktime);
            printf("final time: %f\n", ctime+ktime);
#endif
            handle.AccumKernelTime(ctime);
        }
        break;
    }
}

size_t RNNDescriptor::biasOffsetCalculation(const TensorDescriptor& xDesc,
                                            const TensorDescriptor& wDesc,
                                            const int layer,
                                            const int layerID)
{
    if(biasMode == miopenRNNNoBias)
    {
        return 0;
    }
    
    auto ih = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        ih = 0;
    
    if(ih != inputVectorLen)
    {
        //Reset these in case of xDesc changing.
        pTensorDims.clear();
        biasOffset.clear();
        paramOffset.clear();    
        inputVectorLen = ih;
    }
    
    auto boIt = biasOffset.find(std::make_pair(layer, layerID));
    if(boIt != biasOffset.end())
        return boIt->second;
    else
    {
        size_t layerJump       = 0;    
        unsigned int direction = (layer % 2);
        if(dirMode)
        {

            if(layer > 1) // NOT the input layer
            {

                // jump over input layer's input matrices
                layerJump = 2 * ih * hsize * nHiddenTensorsPerLayer;

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
            }
            else // IS the input layer
            {
                // jump over input layer's input bias
                layerJump = layerID * 2 * ih * hsize * nHiddenTensorsPerLayer;

                // forward or backward direction bias
                layerJump += direction * hsize * nHiddenTensorsPerLayer;
            }
        }
        else
        {
            if(layer > 1) // NOT the input layer
            {
                // jump over input layer's input matrices
                layerJump = ih * hsize * nHiddenTensorsPerLayer;

                // jump over input layer's hidden matrices
                layerJump += hsize * hsize * nHiddenTensorsPerLayer;

                // jump over all other layers' matrices
                layerJump += 2 * hsize * hsize * nHiddenTensorsPerLayer * nLayers;

                // jump to either the input bias, or weight bias
                layerJump += hsize * layerID * nHiddenTensorsPerLayer;
            }
            else
            {
                // jump over input layer's input bias
                layerJump = layerID * ih * hsize * nHiddenTensorsPerLayer;
            }
        }
        biasOffset.emplace(std::make_pair(layer, layerID), layerJump);
        return layerJump;
    }
}

size_t RNNDescriptor::paramsOffsetCalculation(const TensorDescriptor& xDesc,
                                              const TensorDescriptor& wDesc,
                                              const int layer,
                                              const int layerID)
{
    auto ih = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        ih = 0;
    
    if(ih != inputVectorLen)
    {
        //Reset these in case of xDesc changing.
        pTensorDims.clear();
        biasOffset.clear();
        paramOffset.clear();  
        inputVectorLen = ih;
    }

    auto poIt = paramOffset.find(std::make_pair(layer, layerID));
    if(poIt != paramOffset.end())
        return poIt->second;
    else
    {
        size_t layerJump       = 0;
        unsigned int direction = (layer % 2);
        if(dirMode)
        {
            if(layer > 1) // NOT the input layer
            {
                // jump over input layer's input matrices
                layerJump = 2 * ih * hsize * nHiddenTensorsPerLayer;

                // jump over input layer's hidden matrices
                layerJump += 2 * hsize * hsize * nHiddenTensorsPerLayer;

                // jump over all other layers' matrices
                layerJump += 4 * hsize * hsize * nHiddenTensorsPerLayer * (layer - 2);

                // forward or backward direction matrix
                layerJump += direction * hsize * nHiddenTensorsPerLayer;

                // jump to either the input matrix, or weight matrix
                layerJump += 2 * hsize * hsize * layerID * nHiddenTensorsPerLayer;
            }
            else // IS the input layer
            {
                // jump over input layer's input matrices
                layerJump = layerID * 2 * ih * hsize * nHiddenTensorsPerLayer;

                // forward or backward direction matrix
                layerJump += direction * hsize * nHiddenTensorsPerLayer;
            }
        }
        else
        {
            if(layer > 1) // NOT the input layer
            {
                // jump over input layer's input matrices
                layerJump = ih * hsize * nHiddenTensorsPerLayer;

                // jump over input layer's hidden matrices
                layerJump += hsize * hsize * nHiddenTensorsPerLayer;

                // jump over all other layers' matrices
                layerJump += 2 * hsize * hsize * nHiddenTensorsPerLayer * (layer - 1);

                // jump to either the input matrix, or weight matrix
                layerJump += hsize * hsize * layerID * nHiddenTensorsPerLayer;
            }
            else
            {
                // jump over input layer's input matrices
                layerJump = layerID * ih * hsize * nHiddenTensorsPerLayer;
            }
        }
        paramOffset.emplace(std::make_pair(layer, layerID), layerJump);
        return layerJump;
    }
}





std::vector<int> RNNDescriptor::pTensorLengthsCalculation(const TensorDescriptor& xDesc,
                                              const int layer,
                                              const int layerID)
{
    auto ih = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        ih = 0;
    
    if(ih != inputVectorLen)
    {
        //Reset these in case of xDesc changing.
        pTensorDims.clear();
        biasOffset.clear();
        paramOffset.clear();  
        inputVectorLen = ih;
    }
    
    auto ptIt = pTensorDims.find(std::make_pair(layer, layerID));
    if(ptIt != pTensorDims.end())
        return ptIt->second;
    else
    {
        std::vector<int> tdim(2,0);
        if(dirMode)
        {
            if(layer > 2) // NOT the input layer
            {
                tdim[0] = tdim[1] = hsize;
            }
            else // IS the input layer
            {
                tdim[0] = hsize;
                tdim[1] = inputVectorLen;
            }
        }
        else
        {
            if(layer > 1) // NOT the input layer
            {
                tdim[0] = tdim[1] = hsize;
            }
            else
            {
                tdim[0] = hsize;
                tdim[1] = inputVectorLen;
            }
        }
        pTensorDims.emplace(std::make_pair(layer, layerID), tdim);
        return tdim;
    }
}



RNNDescriptor::RNNDescriptor()
{
    nLayers                = 1;
    hsize                  = 0;
    inputBatchLenSum       = 0;
    nHiddenTensorsPerLayer = 0;
    inputVectorLen         = 0;
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
    
    inputVectorLen = 0;

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
    
    pTensorDims.clear();
    biasOffset.clear();
    paramOffset.clear();
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
                                    miopenDataType_t dtype) 
{
    // DLOWELL : The factor of 4 counts the input matrix, hidden matrix, input bias, hidden bias
    // to each of the activated section of the RNN cell.
    // h_t = sigma(Wx_t + Rh_t-1 + bw + br)
    // for one layer: wDesc <-- (v_hidden x v_input) + (v_hidden x v_hidden) + 2*(1 x v_hidden)
    assert(xDesc.GetLengths().size() > 1);
    auto ih = xDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        ih = 0;
    
    if(ih != inputVectorLen)
    {
        //Reset these in case of xDesc changing.
        pTensorDims.clear();
        biasOffset.clear();
        paramOffset.clear();   
        inputVectorLen = ih;
    }
    
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
                                                  const TensorDescriptor& yDesc) 
{
    auto ih = xDesc.GetLengths()[1], oh = yDesc.GetLengths()[1];
    if(inputMode == miopenRNNskip)
        ih = 0;
    
    if(ih != inputVectorLen)
    {
        //Reset these in case of xDesc changing.
        pTensorDims.clear();
        biasOffset.clear();
        paramOffset.clear();    
        inputVectorLen = ih;
    }
    
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
                                        miopenDataType_t dtype) 
{

    auto ih = xDesc.GetLengths()[1]; // input vector size
    if(inputMode == miopenRNNskip)
        ih = 0;
    if(ih != inputVectorLen)
    {
        //Reset these in case of xDesc changing.
        pTensorDims.clear();
        biasOffset.clear();
        paramOffset.clear();  
        inputVectorLen = ih;
    }
    int bi          = (dirMode == miopenRNNbidirection) ? 2 : 1;
    std::array<int, 2> weight_lens;

    weight_lens[0] = inputVectorLen + ((nLayers - 1) * (bi + 1) + 1) * hsize;
    weight_lens[1] = bi * hsize * nHiddenTensorsPerLayer;

    wDesc = miopen::TensorDescriptor(dataType, weight_lens.data(), 2);
}

void RNNDescriptor::GetLayerParam(Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& wDesc,
                                  ConstData_t w,
                                  const int layerID,
                                  TensorDescriptor& paramDesc,
                                  Data_t paramTensor)
{
    // 1. Calculate the location of the matrix via layerID, bidirection setting, and params
    auto poffset = paramsOffsetCalculation(xDesc, wDesc, layer, layerID);
    
    // 2. Get the dimensions of the parameter matrix
    auto pDims = pTensorLengthsCalculation(xDesc, layer, layerID);
    
    // 3. Construct descriptor for param matrix
    paramDesc = miopen::TensorDescriptor(dataType, pDims.data(), 2);
    
    // 4. Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, paramDesc, w, paramDesc, paramTensor, poffset, 0);
}

void RNNDescriptor::GetLayerBias(Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& wDesc,
                                 ConstData_t w,
                                 const int layerID,
                                 TensorDescriptor& biasDesc,
                                 Data_t biasTensor)
{
    if(biasMode == miopenRNNNoBias)
    {
        biasTensor = nullptr;
        return;
    }
    // 1. Calculate the location of the matrix via layerID, bidirection setting, and params
    auto boffset = biasOffsetCalculation(xDesc, wDesc, layer, layerID);
    
    // 2. Get the dimensions of the parameter matrix
    int bdim = int(hsize);
    // 3. Construct descriptor for param matrix
    biasDesc = miopen::TensorDescriptor(dataType, &bdim, 1);
    
    // 4. Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, biasDesc, w, biasDesc, biasTensor, boffset, 0);
}





void RNNDescriptor::SetLayerParam(Handle& handle,
                                  const int layer,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& wDesc,
                                  Data_t w,
                                  const int layerID,
                                  const TensorDescriptor& paramDesc,
                                  ConstData_t param)
{
    // TODO dlowell: Need guard checks here, or have them caught at the copy call?
    
    // 1. Calculate the location of the matrix via layerID, bidirection setting, and params
    auto poffset = paramsOffsetCalculation(xDesc, wDesc, layer, layerID);
    
    // 2. Calculate the strides for the matrix
    std::vector<int> pstride(2,1);
    pstride[0] = ((dirMode == miopenRNNbidirection) ? 2 : 1) * nHiddenTensorsPerLayer;
    
    std::vector<int> intLens(paramDesc.GetLengths().begin(), paramDesc.GetLengths().end());
    
    // 3. Construct descriptor to access into w
    auto pDesc = miopen::TensorDescriptor(dataType, intLens.data(), pstride.data(), 2);
    
    // 4. Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, paramDesc, param, pDesc, w, 0, poffset);    
}





void RNNDescriptor::SetLayerBias(Handle& handle,
                                 const int layer,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& wDesc,
                                 Data_t w,
                                 const int layerID,
                                 const TensorDescriptor& biasDesc,
                                 ConstData_t bias)
{
    if(biasMode == miopenRNNNoBias)
    {
        return;
    }
    // TODO dlowell: Need guard checks here, or have them caught at the copy call?
    
    // 1. Calculate the location of the matrix via layerID, bidirection setting, and params
    auto boffset = biasOffsetCalculation(xDesc, wDesc, layer, layerID);
    
    // 2. Calculate the strides for the matrix
    std::vector<int> bstride(2,1);
    bstride[0] = nHiddenTensorsPerLayer;
    
    std::vector<int> intLens(biasDesc.GetLengths().begin(), biasDesc.GetLengths().end());
    
    // 3. Construct descriptor to access into w
    auto bDesc = miopen::TensorDescriptor(dataType, intLens.data(), bstride.data(), 2);
    
    // 4. Copy over data to previously allocated param tensor
    miopen::CopyTensor(handle, biasDesc, bias, bDesc, w, 0, boffset);    
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
