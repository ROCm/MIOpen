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

#pragma once

#include <miopen/rnn.hpp>
#include <miopen/rnn_util.hpp>
#include "miopen/rnn/tmp_buffer_utils.hpp"

#include "miopen/rnn/algorithms/default_algo_utils.hpp"
#include "miopen/rnn/algorithms/dynamic_algo_utils.hpp"

namespace miopen {

namespace rnn_base {

//
// Forward data
//

class RNNModularSingleStreamFWD
{
public:
    RNNModularSingleStreamFWD(const RNNDescriptor& rnn,
                              const SeqTensorDescriptor& xDesc,
                              const SeqTensorDescriptor& yDesc,
                              const TensorDescriptor& hDesc,
                              miopenRNNFWDMode_t mode)
        : rnnAlgoModules(RNNModuleAlgoBase::create(rnn, xDesc, yDesc, hDesc, mode)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    void ComputeFWD(const Handle& handle, const runtimeArgsFwd& runtimeArgs) const;

    const rnn_base::RNNForwardDataModularAlgo rnnAlgoModules;

    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

class RNNDynamicModularSingleStreamFWD
{
private:
public:
    RNNDynamicModularSingleStreamFWD(const RNNDescriptor& rnn,
                                     const SeqTensorDescriptor& xDesc,
                                     const SeqTensorDescriptor& yDesc,
                                     const TensorDescriptor& hDesc,
                                     miopenRNNFWDMode_t mode)
        : rnnAlgoModules(rnn, xDesc, yDesc, hDesc, mode), rnnDesc(rnn)
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    auto getTempBuffersSize(const Handle& handle) const
    {
        return rnnAlgoModules.getTempBuffersSize(handle);
    }

    static auto getTempBuffersSize(const Handle& handle,
                                   const RNNDescriptor& rnn,
                                   const SeqTensorDescriptor& xDesc)
    {
        return rnn_base::RNNModuleAlgoDynamic::getTempBuffersSize(handle, rnn, xDesc);
    }

    void ComputeFWD(const Handle& handle, const runtimeArgsFwd& runtimeArgs) const;

    const rnn_base::RNNModuleAlgoDynamic rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
};

//
// Backward Data
//

class RNNModularSingleStreamBWD
{
public:
    RNNModularSingleStreamBWD(const RNNDescriptor& rnn,
                              const SeqTensorDescriptor& xDesc,
                              const SeqTensorDescriptor& yDesc,
                              const TensorDescriptor& hDesc,
                              miopenRNNFWDMode_t mode)
        : rnnAlgoModules(RNNModuleAlgoBase::create(rnn, xDesc, yDesc, hDesc, mode)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    void ComputeBWD(const Handle& handle,
                    ConstData_t dy,
                    ConstData_t dhy,
                    Data_t dhx,
                    ConstData_t cx,
                    ConstData_t dcy,
                    Data_t dcx,
                    Data_t dx,
                    ConstData_t w,
                    Data_t workSpace,
                    Data_t reserveSpace) const;

    const rnn_base::RNNBackwardDataModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

class RNNDynamicModularSingleStreamBWD
{
private:
public:
    RNNDynamicModularSingleStreamBWD(const RNNDescriptor& rnn,
                                     const SeqTensorDescriptor& xDesc,
                                     const SeqTensorDescriptor& yDesc,
                                     const TensorDescriptor& hDesc,
                                     miopenRNNFWDMode_t mode)
        : rnnAlgoModules(rnn, xDesc, yDesc, hDesc, mode), rnnDesc(rnn)
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    auto getTempBuffersSize(const Handle& handle) const
    {
        return rnnAlgoModules.getTempBuffersSize(handle);
    }

    static auto getTempBuffersSize(const Handle& handle,
                                   const RNNDescriptor& rnn,
                                   const SeqTensorDescriptor& xDesc)
    {
        return decltype(rnnAlgoModules)::getTempBuffersSize(handle, rnn, xDesc);
    }

    void ComputeBWD(const Handle& handle, const runtimeArgsBwd& runtimeArgs) const;

    const rnn_base::RNNBackwardModuleAlgoDynamic rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
};

class RNNModularMultiStreamBWD
{
public:
    RNNModularMultiStreamBWD(const RNNDescriptor& rnn,
                             const SeqTensorDescriptor& xDesc,
                             const SeqTensorDescriptor& yDesc,
                             const TensorDescriptor& hDesc,
                             miopenRNNFWDMode_t mode)
        : rnnAlgoModules(RNNModuleAlgoBase::create(rnn, xDesc, yDesc, hDesc, mode)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    void ComputeBWD(const Handle& handle,
                    ConstData_t dy,
                    ConstData_t dhy,
                    Data_t dhx,
                    ConstData_t cx,
                    ConstData_t dcy,
                    Data_t dcx,
                    Data_t dx,
                    ConstData_t w,
                    Data_t workSpace,
                    Data_t reserveSpace) const;

    bool ChunkDispatch(const runtimeArgsBwd& args,
                       size_t chunk_size,
                       size_t chunk_time_offset,
                       size_t chunk_layer_offset) const;

private:
    void PrologueDispatch(const runtimeArgsBwd& args) const;

    const rnn_base::RNNBackwardDataModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

//
// Backward Weights
//

class RNNModularSingleStreamBWWeights
{
public:
    RNNModularSingleStreamBWWeights(const RNNDescriptor& rnn,
                                    const SeqTensorDescriptor& xDesc,
                                    const SeqTensorDescriptor& yDesc,
                                    const TensorDescriptor& hDesc)
        : rnnAlgoModules(RNNModuleAlgoBase::create(rnn, xDesc, yDesc, hDesc, miopenRNNTraining)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    void Compute(const Handle& handle,
                 ConstData_t x,
                 ConstData_t hx,
                 Data_t dw,
                 Data_t workSpace,
                 size_t /*workSpaceSize*/,
                 ConstData_t reserveSpace,
                 size_t /*reserveSpaceSize*/) const;

    const rnn_base::RNNBackwardWeightsModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

class RNNDynamicModularSingleStreamBWWeights
{
private:
public:
    RNNDynamicModularSingleStreamBWWeights(const RNNDescriptor& rnn,
                                           const SeqTensorDescriptor& xDesc,
                                           const SeqTensorDescriptor& yDesc,
                                           const TensorDescriptor& hDesc,
                                           miopenRNNFWDMode_t mode)
        : rnnAlgoModules(rnn, xDesc, yDesc, hDesc, mode),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    auto getTempBuffersSize(const Handle& handle) const
    {
        return rnnAlgoModules.getTempBuffersSize(handle);
    }

    static auto getTempBuffersSize(const Handle& handle,
                                   const RNNDescriptor& rnn,
                                   const SeqTensorDescriptor& xDesc)
    {
        return decltype(rnnAlgoModules)::getTempBuffersSize(handle, rnn, xDesc);
    }

    runtimeArgsBWWeights createRuntimeArgsBase(const Handle& handle,
                                               ConstData_t x,
                                               ConstData_t hx,
                                               Data_t dw,
                                               Data_t workSpace,
                                               size_t workSpaceSize,
                                               ConstData_t reserveSpace,
                                               size_t /*reserveSpaceSize*/) const
    {
        const ConstData_t back_data_space = workSpace;
        const auto back_data_byte_size =
            rnnAlgoModules.workspaceInfo.getBufferSizeImpl() * GetTypeSize(rnnDesc.dataType);

        const Data_t free_ws    = moveDataPtrByte(workSpace, back_data_byte_size);
        const auto free_ws_size = workSpaceSize - back_data_byte_size;

        return runtimeArgsBWWeights{
            &handle, x, hx, dw, back_data_space, reserveSpace, free_ws, free_ws_size};
    }

    void Compute(const Handle& handle,
                 ConstData_t x,
                 ConstData_t hx,
                 Data_t dw,
                 Data_t workSpace,
                 size_t /*workSpaceSize*/,
                 ConstData_t reserveSpace,
                 size_t /*reserveSpaceSize*/) const;

    const RNNBackwardWeiModuleAlgoDynamic rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

class RNNModularMultiStreamBWWeights
{
public:
    RNNModularMultiStreamBWWeights(const RNNDescriptor& rnn,
                                   const SeqTensorDescriptor& xDesc,
                                   const SeqTensorDescriptor& yDesc,
                                   const TensorDescriptor& hDesc)
        : rnnAlgoModules(RNNModuleAlgoBase::create(rnn, xDesc, yDesc, hDesc, miopenRNNTraining)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    void Compute(const Handle& handle,
                 ConstData_t x,
                 ConstData_t hx,
                 Data_t dw,
                 Data_t workSpace,
                 size_t /*workSpaceSize*/,
                 ConstData_t reserveSpace,
                 size_t /*reserveSpaceSize*/) const;

private:
    void PrologueDispatch(const runtimeArgsBWWeights& args) const;

    const rnn_base::RNNBackwardWeightsModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

} // namespace rnn_base
} // namespace miopen
