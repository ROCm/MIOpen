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

#include <miopen/rnn.hpp>
#include <miopen/rnn_util.hpp>

#include <miopen/activ.hpp>
#include <miopen/env.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/logger.hpp>

#include <vector>
#include <numeric>
#include <algorithm>

#include <miopen/rnn/solvers.hpp>

#include <miopen/rnn/tmp_buffer_utils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_RNNBWDMS_EXP)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_RNNBWMS_EXP)

namespace miopen {

bool RNNBwdMSIsFast(const int seqLen)
{
    if(env::enabled(MIOPEN_RNNBWDMS_EXP))
        return true;

    if(seqLen >= 32 && !env::disabled(MIOPEN_RNNBWDMS_EXP))
        return true;
    return false;
}

bool RNNBwWeightMSIsFast(const int seqLen)
{
    if(env::enabled(MIOPEN_RNNBWDMS_EXP))
        return true;

    if(seqLen >= 32 && !env::disabled(MIOPEN_RNNBWDMS_EXP))
        return true;
    return false;
}

void RNNDescriptor::ModularForward(Handle& handle,
                                   miopenRNNFWDMode_t fwdMode,
                                   ConstData_t w,
                                   const SeqTensorDescriptor& xDesc,
                                   ConstData_t x,
                                   const TensorDescriptor& hDesc,
                                   ConstData_t hx,
                                   Data_t hy,
                                   const TensorDescriptor& /*cDesc*/,
                                   ConstData_t cx,
                                   Data_t cy,
                                   const SeqTensorDescriptor& yDesc,
                                   Data_t y,
                                   Data_t workSpace,
                                   size_t /*workSpaceSize*/,
                                   Data_t reserveSpace,
                                   size_t /*reserveSpaceSize*/) const
{
    rnn_base::RNNModularSingleStreamFWD single_stream{*this, xDesc, yDesc, hDesc, fwdMode};
    single_stream.ComputeFWD(
        handle, rnn_base::runtimeArgsFwd{x, hx, cx, y, hy, cy, w, workSpace, reserveSpace});
}

void RNNDescriptor::ModularBackward(Handle& handle,
                                    const SeqTensorDescriptor& yDesc,
                                    ConstData_t dy,
                                    const TensorDescriptor& hDesc,
                                    ConstData_t /*hx*/,
                                    ConstData_t dhy,
                                    Data_t dhx,
                                    const TensorDescriptor& /*cDesc*/,
                                    ConstData_t cx,
                                    ConstData_t dcy,
                                    Data_t dcx,
                                    const SeqTensorDescriptor& xDesc,
                                    Data_t dx,
                                    ConstData_t w,
                                    Data_t workSpace,
                                    size_t /*workSpaceSize*/,
                                    Data_t reserveSpace,
                                    size_t /*reserveSpaceSize*/) const
{
    if(RNNBwdMSIsFast(xDesc.GetMaxSequenceLength()))
    {
        rnn_base::RNNModularMultiStreamBWD multi_stream{
            *this, xDesc, yDesc, hDesc, miopenRNNFWDMode_t::miopenRNNTraining};
        multi_stream.ComputeBWD(handle, dy, dhy, dhx, cx, dcy, dcx, dx, w, workSpace, reserveSpace);
    }
    else
    {
        rnn_base::RNNModularSingleStreamBWD single_stream{
            *this, xDesc, yDesc, hDesc, miopenRNNFWDMode_t::miopenRNNTraining};
        single_stream.ComputeBWD(
            handle, dy, dhy, dhx, cx, dcy, dcx, dx, w, workSpace, reserveSpace);
    }
}

void RNNDescriptor::ModularBackwardWeights(Handle& handle,
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
    if(RNNBwWeightMSIsFast(xDesc.GetMaxSequenceLength()))
    {
        rnn_base::RNNModularMultiStreamBWWeights multi_stream{*this, xDesc, yDesc, hDesc};
        multi_stream.Compute(
            handle, x, hx, dw, workSpace, workSpaceSize, reserveSpace, reserveSpaceSize);
    }
    else
    {
        rnn_base::RNNModularSingleStreamBWWeights single_stream{*this, xDesc, yDesc, hDesc};
        single_stream.Compute(
            handle, x, hx, dw, workSpace, workSpaceSize, reserveSpace, reserveSpaceSize);
    }
}

} // namespace miopen
