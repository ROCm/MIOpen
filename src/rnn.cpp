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

RNNDescriptor::RNNDescriptor()
{
    nLayers                = 1;
    hsize                  = 0;
    inputBatchLenSum       = 0;
    nHiddenTensorsPerLayer = 0;
	rnnMode = miopenRNNTANH;
	dirMode = miopenRNNunidirection;
	biasMode = miopenRNNNoBias;
	algoMode = miopenRNNdefault;
	inputMode = miopenRNNlinear;
	dataType = miopenFloat;
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
    case 0:
    case 1:
        nHiddenTensorsPerLayer = 1;
        workspaceScale         = 1;
        break;
    case 2:
        nHiddenTensorsPerLayer = 4;
        workspaceScale         = 6;
        break;
    case 3:
        nHiddenTensorsPerLayer = 3;
        workspaceScale         = 4;
        break;
    }

    inputBatchLenSum = 0; // init
}

size_t RNNDescriptor::GetWorkspaceSize(Handle& handle, const int sLen, TensorDescriptor* xDesc)
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
/*    auto x = workspaceScale * sLen * inputBatchLenSum * nLayers * sizeof(xDesc[0].GetType()) *
             nHiddenTensorsPerLayer;
    return size_t(x);*/

	auto x = workspaceScale * nLayers * inputBatchLenSum * hsize * sizeof(xDesc[0].GetType());
	return dirMode == miopenRNNbidirection ? size_t(2 * x) : size_t(x);
}

size_t RNNDescriptor::GetReserveSize(Handle& handle, const int sLen, TensorDescriptor* xDesc)
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
/*    auto x = workspaceScale * sLen * inputBatchLenSum * nLayers * sizeof(xDesc[0].GetType()) *
             nHiddenTensorsPerLayer;
    return size_t(x);*/

	auto x = workspaceScale * nLayers * inputBatchLenSum * hsize * sizeof(xDesc[0].GetType());
	//	auto x = 2 * workspaceScale * nLayers * inputBatchLenSum * hsize * sizeof(xDesc[0].GetType());  // switch to this after offset activ and ops applied
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
    auto inputVecSize = xDesc.GetLengths()[1];
    size_t x = 0;
    auto biHiddenSize = hsize;
    if(dirMode) 
    {
        biHiddenSize *= 2;
    }
    if(biasMode)
    {
        x = (biHiddenSize * inputVecSize) + nLayers * nHiddenTensorsPerLayer * ((biHiddenSize * biHiddenSize) + 2 * biHiddenSize);
    }
    else
    {
        x = (biHiddenSize * inputVecSize) + nLayers * nHiddenTensorsPerLayer * (biHiddenSize * biHiddenSize);
    }
    return x;

	/* auto ih = xDesc.GetLengths()[1];
	int bi = dirMode == miopenRNNbidirection ? 2 : 1;
    auto x = nHiddenTensorsPerLayer * hsize * bi * (ih + hsize + (nLayers - 1) * (bi + 1) * hsize);
	if (biasMode == miopenRNNwithBias)
	{
		x += (2 + (nLayers - 1) * (bi + 1)) * nHiddenTensorsPerLayer * hsize * bi; // bias size need to discuss
	}

    return size_t(x); */
}

/* Get weight super tensor size
temporary function assuming output matrix exists */
size_t RNNDescriptor::GetRNNWeightSuperTensorSize(Handle& handle,
	const TensorDescriptor& xDesc,
	const TensorDescriptor& yDesc) const
{
	auto ih = xDesc.GetLengths()[1], oh = yDesc.GetLengths()[1];
	int bi = dirMode == miopenRNNbidirection ? 2 : 1;
	auto sz = nHiddenTensorsPerLayer * hsize * bi * (ih + hsize + (nLayers - 1) * (bi + 1) * hsize) + oh * hsize * bi;
	if (biasMode == miopenRNNwithBias)
	{
		sz += (2 + (nLayers - 1) * (bi + 1)) * nHiddenTensorsPerLayer * hsize * bi + bi * oh;
	}

	return size_t(sz);
}

void RNNDescriptor::GetLayerParam(Handle& handle,
                                    const TensorDescriptor& xDesc,
                                    const TensorDescriptor& wDesc,
                                    ConstData_t w,
                                    const int layerID,
                                    const TensorDescriptor& paramDesc,
                                    Data_t param) const
{

    /*If mode in rnnDesc was set to CUDNN_RNN_RELU or
CUDNN_RNN_TANH a value of 0 references the matrix
multiplication applied to the input from the previous layer, a
value of 1 references the matrix multiplication applied to the
recurrent input.*/
    // 0 --> Wx_t
    // 1 --> Rh_t-1

    // TODO: FILL
}

void RNNDescriptor::GetLayerBias(Handle& handle,
                                   const TensorDescriptor& xDesc,
                                   const TensorDescriptor& wDesc,
                                   ConstData_t w,
                                   const int layerID,
                                   const TensorDescriptor& biasDesc,
                                   Data_t bias) const
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
