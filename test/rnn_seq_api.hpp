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

#pragma once

#include "driver.hpp"
#include "dropout_util.hpp"
#include "get_handle.hpp"

#include "random.hpp"
#include <random>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <miopen/rnn.hpp>
#include <miopen/miopen.h>
#include <miopen/float_equal.hpp>

//
// Native rnn tensor format
//
#include "seq_tensor_holder.hpp"
#include "rnn_util.hpp"
#include "cpu_rnn.hpp"
///

template <class TensorT>
miopen::Allocator::ManageDataPtr
createTensorAtGPUOrNullptr(miopen::Handle& handle, TensorT& tensor, bool is_nullptr)
{
    if(!is_nullptr)
        return handle.Create(tensor.GetDataByteSize());
    else
        return nullptr;
}

template <class TensorT>
miopen::Allocator::ManageDataPtr
transferTensorToGPUOrNullptr(miopen::Handle& handle, TensorT& tensor, bool is_nullptr)
{
    if(!is_nullptr)
        return handle.Write(tensor.data);
    else
        return nullptr;
}

// read from GPU cnt elements of type T
template <template <class T> typename TensorT, class T>
auto readTFromGPUOrEmpty(miopen::Handle& handle,
                         miopen::Allocator::ManageDataPtr& gpuPtr,
                         const TensorT<T>& tensor,
                         bool isEmpty) -> decltype(handle.Read<T>(gpuPtr, tensor.GetSize()))
{
    if(!isEmpty)
        return handle.Read<T>(gpuPtr, tensor.GetSize());
    else
        return {};
}

//****************************************************
// FORWARD BASE
//****************************************************
template <class T>
struct verify_rnn_api_base
{
    seqTensor<T> input{};
    seqTensor<T> output{};

    tensor<T> xHiddenState{};
    tensor<T> xCellState{};

    std::vector<T> weights{};
    miopen::RNNDescriptor rnnDesc{};

    bool nohx{};
    bool nocx{};
    bool nohy{};
    bool nocy{};
    bool is_padded_verification{};

    T padding_symbol{};

    verify_rnn_api_base(miopen::RNNDescriptor& pRD,
                        seqTensor<T>& x,
                        seqTensor<T>& y,
                        tensor<T>& hx,
                        tensor<T>& cx,
                        std::vector<T>& w,
                        const bool pnohx = false,
                        const bool pnocx = false,
                        const bool pnohy = false,
                        const bool pnocy = false,
                        T* paddingSymbol = nullptr)
        : input(x),
          output(y),
          xHiddenState(hx),
          xCellState(cx),
          weights(w),
          rnnDesc(pRD),
          nohx(pnohx),
          nocx(pnocx),
          nohy(pnohy),
          nocy(pnocy)
    {
        if(paddingSymbol != nullptr)
        {
            is_padded_verification = input.desc.IsPaddedSeqLayout();
            padding_symbol         = *paddingSymbol;
        }
        else
        {
            is_padded_verification = false;
            padding_symbol         = 0;
        }
    }

    size_t total_GPU_mem_size() {}
    size_t input_GPU_mem_size() {}
    size_t output_GPU_mem_size() {}
    size_t workspace_GPU_mem_size() {}
    size_t reservspace_GPU_mem_size() {}

    void fail(int badtensor) const
    {
        std::cout << "./bin/MIOpenDriver rnn_seq ";

        std::cout << " -F 0 "
                  << " -m ";

        switch(rnnDesc.rnnMode)
        {
        case miopenRNNTANH: std::cout << " tanh "; break;
        case miopenRNNRELU: std::cout << " relu "; break;
        case miopenLSTM: std::cout << " lstm "; break;
        case miopenGRU: std::cout << " gru "; break;
        default: break;
        }

        auto& inLens = input.desc.GetLengths();
        auto& hLens  = xHiddenState.desc.GetLengths();

        std::cout << " --batch_size " << inLens[0] << " --seq_len " << inLens[1] << " --in_vec "
                  << inLens[2] << " --hid_h " << hLens[2] << " --num_layer " << rnnDesc.nLayers
                  << " -r " << rnnDesc.dirMode << " -b " << rnnDesc.biasMode << " -p "
                  << rnnDesc.inputMode << " -a " << rnnDesc.algoMode;

        bool useDropout = !miopen::float_equal(miopen::deref(rnnDesc.dropoutDesc).dropout, 0);

        std::cout << " --io_layout "
                  << miopen::RNNDescriptor::getBaseLayoutFromDataTensor(input.desc)
                  << " --use_dropout " << useDropout;
        if(useDropout)
            std::cout << " --dropout " << miopen::deref(rnnDesc.dropoutDesc).dropout;

        auto& samplesLen = input.desc.GetSequenceLengthsVector();
        std::cout << " --seq_len_array ";
        for(int i = 0; i < inLens[0]; i++)
        {
            if(i < inLens[0] - 1)
            {
                std::cout << samplesLen.at(i) << ",";
            }
            else
            {
                std::cout << samplesLen.at(i);
            }
        }
        std::cout << std::endl;

        if(badtensor >= 0)
        {
            if(badtensor < 3)
            {
                std::cout << "FWD Train LSTM: " << std::endl;
                switch(badtensor)
                {
                case(0): std::cout << "Output tensor report." << std::endl; break;
                case(1): std::cout << "Hidden state tensor report." << std::endl; break;
                case(2): std::cout << "Cell state tensor report." << std::endl; break;
                default: break;
                }
            }
            else if(badtensor < 6)
            {
                std::cout << "BWD Train LSTM: " << std::endl;
                switch(badtensor)
                {
                case(3): std::cout << "Output tensor output report." << std::endl; break;
                case(4): std::cout << "Hidden state tensor report." << std::endl; break;
                case(5): std::cout << "Cell state tensor report." << std::endl; break;
                default: break;
                }
            }
            else if(badtensor == 6)
            {
                std::cout << "WRW Train LSTM " << std::endl;
            }
        }
    }
};

//****************************************************
// RNN TRAIN
//****************************************************

template <class T>
struct rnn_ref
{
    struct FwdResult
    {
        std::vector<T> y;
        std::vector<T> hy;
        std::vector<T> cy;
        FwdResult(std::vector<T> fwd_y,
                  std::vector<T> fwd_hy,
                  std::vector<T> fwd_cy,
                  bool nohy,
                  bool nocy)
            : y(std::move(fwd_y)),
              hy(nohy ? std::vector<T>{} : std::move(fwd_hy)),
              cy(nocy ? std::vector<T>{} : std::move(fwd_cy))
        {
        }
    };

    struct BwdResult
    {
        std::vector<T> din;
        std::vector<T> dhx;
        std::vector<T> dcx;
        BwdResult(std::vector<T> bwd_din,
                  std::vector<T> bwd_dhx,
                  std::vector<T> bwd_dcx,
                  bool nodhx,
                  bool nodcx)
            : din(std::move(bwd_din)),
              dhx(nodhx ? std::vector<T>{} : std::move(bwd_dhx)),
              dcx(nodcx ? std::vector<T>{} : std::move(bwd_dcx))
        {
        }
    };

    struct WrwResult
    {
        std::vector<T> dwei;

        WrwResult(std::vector<T> wrw_dwei) : dwei(std::move(wrw_dwei)) {}
    };

    virtual size_t getReserveSpaceSize() const = 0;

    virtual size_t getWorkSpaceSize() const = 0;

    virtual FwdResult fwd(const miopen::SeqTensorDescriptor& xDesc,
                          const miopen::SeqTensorDescriptor& yDesc,
                          const std::vector<T>& xData,
                          const std::vector<T>& hxData,
                          const std::vector<T>& cxData,
                          const std::vector<T>& wData,
                          std::vector<T>& reserveSpace,
                          bool nohx,
                          bool nocx,
                          bool nohy,
                          bool nocy) const = 0;

    virtual BwdResult bwd(const miopen::SeqTensorDescriptor& xDesc,
                          const miopen::SeqTensorDescriptor& yDesc,
                          const std::vector<T>& dyData,
                          const std::vector<T>& dhyData,
                          const std::vector<T>& dcyData,
                          const std::vector<T>& hxData,
                          const std::vector<T>& cxData,
                          const std::vector<T>& weiData,
                          std::vector<T>& reserveSpace,
                          std::vector<T>& workSpace,
                          bool nodhx,
                          bool nodcx,
                          bool nodhy,
                          bool nodcy,
                          bool nohx,
                          bool nocx) const = 0;

    virtual WrwResult wrw(const miopen::SeqTensorDescriptor& xDesc,
                          const miopen::SeqTensorDescriptor& dyDesc,
                          const std::vector<T>& xData,
                          const std::vector<T>& hxData,
                          const std::vector<T>& doutData,
                          std::vector<T>& reserveSpace,
                          std::vector<T>& workSpace,
                          bool nohx) const = 0;

    virtual ~rnn_ref(){};
};

template <class T>
struct cpu_rnn_packed_ref : public rnn_ref<T>
{
    using typename rnn_ref<T>::FwdResult;
    using typename rnn_ref<T>::BwdResult;
    using typename rnn_ref<T>::WrwResult;

    // supported only hDesc equal to cDesc
    cpu_rnn_packed_ref(const miopen::RNNDescriptor& rnn,
                       const miopen::SeqTensorDescriptor& maxX,
                       const miopen::TensorDescriptor& hPackedDesc)
        : rnnDesc(rnn)
    {
        assert(checkSeqTensor(maxX));

        dirMode    = rnnDesc.dirMode == miopenRNNbidirection;
        biasMode   = rnnDesc.biasMode == miopenRNNwithBias;
        inputMode  = rnnDesc.inputMode == miopenRNNskip;
        rnnMode    = rnnDesc.rnnMode;
        useDropout = !miopen::float_equal(miopen::deref(rnnDesc.dropoutDesc).dropout, 0);

        std::tie(hiddenLayers, std::ignore, hidVec) = miopen::tien<3>(hPackedDesc.GetLengths());

        outVec = hidVec * (dirMode ? 2 : 1);

        inVec                     = maxX.GetLengths()[2];
        size_t input_batchLen_sum = maxX.GetTotalSequenceLen();

        reserveSpaceSizeCpu = UniRNNCPUReserveSpaceSize(
            rnnMode, rnnDesc.nLayers, input_batchLen_sum, outVec, sizeof(T), useDropout);

        workSpaceSizeCpu = UniRNNCPUWorkSpaceByteSize(
            rnnMode, rnnDesc.nLayers, input_batchLen_sum, rnn.hsize, sizeof(T), dirMode);
    }

    size_t getReserveSpaceSize() const override { return reserveSpaceSizeCpu; }

    size_t getWorkSpaceSize() const override { return workSpaceSizeCpu; }

    FwdResult fwd(const miopen::SeqTensorDescriptor& xDesc,
                  const miopen::SeqTensorDescriptor&,
                  const std::vector<T>& xData,
                  const std::vector<T>& hxData,
                  const std::vector<T>& cxData,
                  const std::vector<T>& wData,
                  std::vector<T>& reserveSpace,
                  bool nohx,
                  bool nocx,
                  bool nohy,
                  bool nocy) const override
    {
        assert(checkSeqTensor(xDesc));

        auto&& handle = get_handle();

        auto batch_seq      = xDesc.GetBatchesPerSequence();
        size_t seq_len      = batch_seq.size();
        size_t total_batchs = std::accumulate(batch_seq.begin(), batch_seq.end(), 0ULL);
        size_t batch_size   = batch_seq.at(0);

        bool is_state_tensor_zip_req = (batch_size != xDesc.GetLengths()[0]);
        bool is_hx_zip_req           = is_state_tensor_zip_req && !nohx;
        bool is_cx_zip_req           = is_state_tensor_zip_req && !nocx;

        std::vector<int> batch_seq_downgrade(batch_seq.cbegin(), batch_seq.cend());

        std::vector<T> hidden_state(UniRNNCPUHiddenStateSize(hiddenLayers, batch_size, hidVec));
        std::vector<T> cell_state(
            UniRNNCPUCellStateSize(rnnDesc.rnnMode, hiddenLayers, batch_size, hidVec));

        std::vector<T> packed_output(UniRNNCPUIOSize(total_batchs, outVec));

        UniformRNNFwdTrainCPUVerify(
            handle,
            useDropout,
            miopen::deref(rnnDesc.dropoutDesc),
            xData,
            wData,        // [ input_state_weight_trans
                          // hidden_state_weight0_trans input1_trans
                          // hidden1_trans ... output_weight;
                          // bidirectional reversed weights ]
            hidden_state, // current/final hidden state
            is_hx_zip_req ? zipStateVectorTensor(
                                hxData, hiddenLayers, xDesc.GetLengths()[0], batch_size, hidVec)
                          : hxData, // initial hidden state
            cell_state,             // current/final cell state
            is_cx_zip_req ? zipStateVectorTensor(
                                cxData, hiddenLayers, xDesc.GetLengths()[0], batch_size, hidVec)
                          : cxData, // initial cell state
            packed_output,
            batch_seq_downgrade, // input batch size
            inVec,               // input data length
            seq_len,             // Number of iterations to unroll over
            dirMode,             // whether using bidirectional net
            biasMode,            // whether using bias
            hiddenLayers,        // 1 by numlayer (number of stacks of hidden layers)
                                 // for unidirection, 2 by numlayer for bidirection
            batch_size,          // equal to input batch size in_n[0]
            hidVec,              // hidden state number
            outVec,              // 1 by hy_h related function for unidirection, 2 by hy_h
                                 // related function for bidirection
            rnnMode,
            inputMode,
            reserveSpace,
            nohx,
            nocx);

        if(is_state_tensor_zip_req)
        {
            return {std::move(packed_output),
                    nohy ? std::move(hidden_state)
                         : zipStateVectorTensor(std::move(hidden_state),
                                                hiddenLayers,
                                                batch_size,
                                                xDesc.GetLengths()[0],
                                                hidVec),
                    nocy ? std::move(cell_state)
                         : zipStateVectorTensor(std::move(cell_state),
                                                hiddenLayers,
                                                batch_size,
                                                xDesc.GetLengths()[0],
                                                hidVec),
                    nohy,
                    nocy};
        }
        else
            return {std::move(packed_output),
                    std::move(hidden_state),
                    std::move(cell_state),
                    nohy,
                    nocy};
    }

    BwdResult bwd(const miopen::SeqTensorDescriptor& xDesc,
                  const miopen::SeqTensorDescriptor&,
                  const std::vector<T>& dyData,
                  const std::vector<T>& dhyData,
                  const std::vector<T>& dcyData,
                  const std::vector<T>& hxData,
                  const std::vector<T>& cxData,
                  const std::vector<T>& weiData,
                  std::vector<T>& reserveSpace,
                  std::vector<T>& workSpace,
                  bool nodhx,
                  bool nodcx,
                  bool nodhy,
                  bool nodcy,
                  bool nohx,
                  bool nocx) const override
    {
        assert(checkSeqTensor(xDesc));

        auto batch_seq    = xDesc.GetBatchesPerSequence();
        size_t seq_len    = batch_seq.size();
        size_t batch_size = batch_seq.at(0);

        bool is_state_tensor_zip_req = (batch_size != xDesc.GetLengths()[0]);
        bool is_dhy_zip_req          = is_state_tensor_zip_req && !nodhy;
        bool is_dcy_zip_req          = is_state_tensor_zip_req && !nodcy;
        bool is_hx_zip_req           = is_state_tensor_zip_req && !nohx;
        bool is_cx_zip_req           = is_state_tensor_zip_req && !nocx;

        size_t total_batchs = std::accumulate(batch_seq.begin(), batch_seq.end(), 0ULL);

        std::vector<int> batch_seq_downgrade(batch_seq.cbegin(), batch_seq.cend());

        std::vector<T> d_hidden_state(UniRNNCPUHiddenStateSize(hiddenLayers, batch_size, hidVec));
        std::vector<T> d_cell_state(
            UniRNNCPUCellStateSize(rnnDesc.rnnMode, hiddenLayers, batch_size, hidVec));
        std::vector<T> packed_dInput(UniRNNCPUIOSize(total_batchs, inVec));

        UniformRNNBwdTrainCPUVerify(
            useDropout,
            miopen::deref(rnnDesc.dropoutDesc),
            packed_dInput, // DX (output)
            weiData,       // [ input_state_weight_trans
                           //   hidden_state_weight0_trans input1_trans
                           //   hidden1_trans ... output_weight;
                           //   bidirectional reversed weights ]
            is_dhy_zip_req ? zipStateVectorTensor(
                                 dhyData, hiddenLayers, xDesc.GetLengths()[0], batch_size, hidVec)
                           : dhyData, // current/final hidden state
            d_hidden_state,           // DHX (output)
            is_hx_zip_req ? zipStateVectorTensor(
                                hxData, hiddenLayers, xDesc.GetLengths()[0], batch_size, hidVec)
                          : hxData, // HX initial hidden state
            is_dcy_zip_req ? zipStateVectorTensor(
                                 dcyData, hiddenLayers, xDesc.GetLengths()[0], batch_size, hidVec)
                           : dcyData, // DCY current/final cell state
            d_cell_state,             // DCX (output)
            is_cx_zip_req ? zipStateVectorTensor(
                                cxData, hiddenLayers, xDesc.GetLengths()[0], batch_size, hidVec)
                          : cxData, // CX
            {},                     // Y
            dyData,                 // DY

            batch_seq_downgrade, // input batch size
            inVec,               // input data length
            seq_len,             // Number of iterations to unroll over
            dirMode,             // whether using bidirectional net
            biasMode,            // whether using bias
            hiddenLayers,        // 1 by numlayer (number of stacks of hidden layers)
                                 // for unidirection, 2 by numlayer for bidirection
            batch_size,          // equal to input batch size in_n[0]
            hidVec,              // hidden state number
            outVec,              // 1 by hy_h related function for unidirection, 2 by
                                 // hy_h related function for bidirection
            rnnMode,
            inputMode,
            reserveSpace,
            workSpace,
            nohx,
            nocx,
            nodhy,
            nodcy);

        if(is_state_tensor_zip_req)
        {
            return {std::move(packed_dInput),
                    nodhx ? std::move(d_hidden_state)
                          : zipStateVectorTensor(std::move(d_hidden_state),
                                                 hiddenLayers,
                                                 batch_size,
                                                 xDesc.GetLengths()[0],
                                                 hidVec),
                    nodcx ? std::move(d_cell_state)
                          : zipStateVectorTensor(std::move(d_cell_state),
                                                 hiddenLayers,
                                                 batch_size,
                                                 xDesc.GetLengths()[0],
                                                 hidVec),
                    nodhx,
                    nodcx};
        }
        else
            return {std::move(packed_dInput),
                    std::move(d_hidden_state),
                    std::move(d_cell_state),
                    nodhx,
                    nodcx};
    }

    WrwResult wrw(const miopen::SeqTensorDescriptor& xDesc,
                  const miopen::SeqTensorDescriptor&,
                  const std::vector<T>& xData,
                  const std::vector<T>& hxData,
                  const std::vector<T>& doutData,
                  std::vector<T>& reserveSpace,
                  std::vector<T>& workSpace,
                  bool nohx) const override
    {
        assert(checkSeqTensor(xDesc));

        auto batch_seq = xDesc.GetBatchesPerSequence();

        std::vector<int> batch_seq_downgrade(batch_seq.cbegin(), batch_seq.cend());

        size_t seq_len    = batch_seq.size();
        size_t batch_size = batch_seq.at(0);

        bool is_state_tensor_zip_req = (batch_size != xDesc.GetLengths()[0]);
        bool is_hx_zip_req           = is_state_tensor_zip_req && !nohx;

        std::vector<T> dwei_data(UniRNNCPUWeightSize(
            rnnMode, rnnDesc.nLayers, hidVec, inVec, biasMode, inputMode, dirMode));

        UniformRNNBwdWeightCPUVerify(
            useDropout,
            xData,
            dwei_data, // (output) [ input_state_weight_trans
                       // hidden_state_weight0_trans
                       // input1_trans hidden1_trans ...
                       // output_weight; bidirectional
                       // reversed weights ]
            is_hx_zip_req ? zipStateVectorTensor(
                                hxData, hiddenLayers, xDesc.GetLengths()[0], batch_size, hidVec)
                          : hxData, // initial hidden state
            doutData,
            batch_seq_downgrade, // input batch size
            inVec,               // input data length
            seq_len,             // Number of iterations to unroll over
            dirMode,             // whether using bidirectional net
            biasMode,            // whether using bias
            hiddenLayers,        // 1 by numlayer (number of stacks of hidden
                                 // layers) for unidirection, 2 by numlayer for
                                 // bidirection
            batch_size,          // equal to input batch size in_n[0]
            hidVec,              // hidden state number
            outVec,              // 1 by hy_h related function for unidirection, 2
                                 // by hy_h related function for bidirection
            rnnMode,
            inputMode,
            reserveSpace,
            workSpace,
            nohx);

        return {std::move(dwei_data)};
    }

private:
    bool checkSeqTensor(const miopen::SeqTensorDescriptor& desc) const
    {
        bool ret = true;
        ret &= desc.IsPacked();
        ret &= miopenRNNDataSeqMajorNotPadded ==
               miopen::RNNDescriptor::getBaseLayoutFromDataTensor(desc);
        return ret;
    }

    // to remove zero size samples
    std::vector<T> zipStateVectorTensor(const std::vector<T>& data,
                                        size_t nLayers,
                                        size_t inBatchSize,
                                        size_t outBatchSize,
                                        size_t vecSize) const
    {
        std::vector<T> ret(nLayers * outBatchSize * vecSize, static_cast<T>(0));
        size_t copy_size = std::min(inBatchSize, outBatchSize) * vecSize;
        for(size_t i = 0; i < nLayers; i++)
            std::copy_n(data.begin() + i * inBatchSize * vecSize,
                        copy_size,
                        ret.begin() + i * outBatchSize * vecSize);
        return ret;
    }

    const miopen::RNNDescriptor rnnDesc{};

    bool dirMode{};
    bool biasMode{};
    bool inputMode{};
    bool useDropout{};
    miopenRNNMode_t rnnMode;
    size_t hiddenLayers, hidVec, outVec, inVec;
    size_t reserveSpaceSizeCpu, workSpaceSizeCpu;
};

template <class T>
struct cpu_rnn_universal_ref : rnn_ref<T>
{
    using typename rnn_ref<T>::FwdResult;
    using typename rnn_ref<T>::BwdResult;
    using typename rnn_ref<T>::WrwResult;

    cpu_rnn_universal_ref(const miopen::RNNDescriptor& rnn,
                          const miopen::SeqTensorDescriptor& maxX,
                          const miopen::TensorDescriptor& hPackedDesc)
        : packed_ref(ConstructPackedRNNRef(rnn, maxX, hPackedDesc)), hiddenDesc(hPackedDesc)
    {
    }

    size_t getReserveSpaceSize() const override { return packed_ref.getReserveSpaceSize(); }

    size_t getWorkSpaceSize() const override { return packed_ref.getWorkSpaceSize(); }

    FwdResult fwd(const miopen::SeqTensorDescriptor& xDesc,
                  const miopen::SeqTensorDescriptor& yDesc,
                  const std::vector<T>& xData,
                  const std::vector<T>& hxData,
                  const std::vector<T>& cxData,
                  const std::vector<T>& wData,
                  std::vector<T>& reserveSpace,
                  bool nohx,
                  bool nocx,
                  bool nohy,
                  bool nocy) const override
    {
        if(xDesc.IsPacked() && miopenRNNDataSeqMajorNotPadded ==
                                   miopen::RNNDescriptor::getBaseLayoutFromDataTensor(xDesc))
        {
            return packed_ref.fwd(
                xDesc, yDesc, xData, hxData, cxData, wData, reserveSpace, nohx, nocx, nohy, nocy);
        }
        else
        {
            auto converted_seq_order =
                GetSamplesIndexDescendingOrder(xDesc.GetSequenceLengthsVector());

            // IO
            ////////////////////////////////////////////////////////////////
            seqTensor<T> x_tensor_converted(GetSeqDescriptorLayoutTransform(
                xDesc, miopenRNNDataSeqMajorNotPadded, converted_seq_order));

            const miopen::SeqTensorDescriptor y_converted_desc(GetSeqDescriptorLayoutTransform(
                yDesc, miopenRNNDataSeqMajorNotPadded, converted_seq_order));

            TransformRNNIOLayaoutToTarget(xDesc,
                                          x_tensor_converted.desc,
                                          converted_seq_order,
                                          xData,
                                          x_tensor_converted.data);
            ///////////////////////////////////////////////////////////////

            auto& hid = hiddenDesc.GetLengths();

            std::vector<T> hxData_converted{};
            if(!nohx)
            {
                hxData_converted.resize(hxData.size());
                HiddenTensorReorder(hxData, hxData_converted, converted_seq_order, hid, true);
            }

            std::vector<T> cxData_converted{};
            if(!nocx)
            {
                cxData_converted.resize(cxData.size());
                HiddenTensorReorder(cxData, cxData_converted, converted_seq_order, hid, true);
            }

            auto packed_res = packed_ref.fwd(x_tensor_converted.desc,
                                             y_converted_desc,
                                             x_tensor_converted.data,
                                             hxData_converted,
                                             cxData_converted,
                                             wData,
                                             reserveSpace,
                                             nohx,
                                             nocx,
                                             nohy,
                                             nocy);

            std::vector<int> reverse_order = GetReverseOrderIndex(converted_seq_order);
            // IO
            ////////////////////////////////////////////////////////////////
            seqTensor<T> y_tensor(yDesc);

            TransformRNNIOLayaoutToTarget(
                y_converted_desc, y_tensor.desc, reverse_order, packed_res.y, y_tensor.data);
            ////////////////////////////////////////////////////////////////

            std::vector<T> hyData_converted{};
            if(!nohy)
            {
                hyData_converted.resize(packed_res.hy.size());
                HiddenTensorReorder(packed_res.hy, hyData_converted, reverse_order, hid, true);
            }

            std::vector<T> cyData_converted{};
            if(!nocy)
            {
                cyData_converted.resize(packed_res.cy.size());
                HiddenTensorReorder(packed_res.cy, cyData_converted, reverse_order, hid, true);
            }

            return {y_tensor.data, hyData_converted, cyData_converted, nohy, nocy};
        }
    }

    BwdResult bwd(const miopen::SeqTensorDescriptor& xDesc,
                  const miopen::SeqTensorDescriptor& yDesc,
                  const std::vector<T>& dyData,
                  const std::vector<T>& dhyData,
                  const std::vector<T>& dcyData,
                  const std::vector<T>& hxData,
                  const std::vector<T>& cxData,
                  const std::vector<T>& weiData,
                  std::vector<T>& reserveSpace,
                  std::vector<T>& workSpace,
                  bool nodhx,
                  bool nodcx,
                  bool nodhy,
                  bool nodcy,
                  bool nohx,
                  bool nocx) const override
    {
        if(xDesc.IsPacked() && miopenRNNDataSeqMajorNotPadded ==
                                   miopen::RNNDescriptor::getBaseLayoutFromDataTensor(xDesc))
        {
            return packed_ref.bwd(xDesc,
                                  yDesc,
                                  dyData,
                                  dhyData,
                                  dcyData,
                                  hxData,
                                  cxData,
                                  weiData,
                                  reserveSpace,
                                  workSpace,
                                  nodhx,
                                  nodcx,
                                  nodhy,
                                  nodcy,
                                  nohx,
                                  nocx);
        }
        else
        {
            auto converted_seq_order =
                GetSamplesIndexDescendingOrder(xDesc.GetSequenceLengthsVector());

            // IO
            ////////////////////////////////////////////////////////////////
            seqTensor<T> dy_tensor_converted(GetSeqDescriptorLayoutTransform(
                yDesc, miopenRNNDataSeqMajorNotPadded, converted_seq_order));

            const miopen::SeqTensorDescriptor x_converted_desc(GetSeqDescriptorLayoutTransform(
                xDesc, miopenRNNDataSeqMajorNotPadded, converted_seq_order));

            TransformRNNIOLayaoutToTarget(yDesc,
                                          dy_tensor_converted.desc,
                                          converted_seq_order,
                                          dyData,
                                          dy_tensor_converted.data);
            ///////////////////////////////////////////////////////////////
            auto& hid = hiddenDesc.GetLengths();

            std::vector<T> dhyData_converted{};
            if(!nodhy)
            {
                dhyData_converted.resize(dhyData.size());
                HiddenTensorReorder(dhyData, dhyData_converted, converted_seq_order, hid, true);
            }

            std::vector<T> dcyData_converted{};
            if(!nodcy)
            {
                dcyData_converted.resize(dcyData.size());
                HiddenTensorReorder(dcyData, dcyData_converted, converted_seq_order, hid, true);
            }

            std::vector<T> cxData_converted{};
            if(!nocx)
            {
                cxData_converted.resize(cxData.size());
                HiddenTensorReorder(cxData, cxData_converted, converted_seq_order, hid, true);
            }

            std::vector<T> hxData_converted{};
            if(!nohx)
            {
                hxData_converted.resize(hxData.size());
                HiddenTensorReorder(hxData, hxData_converted, converted_seq_order, hid, true);
            }

            auto packed_res                = packed_ref.bwd(x_converted_desc,
                                             dy_tensor_converted.desc,
                                             dy_tensor_converted.data,
                                             dhyData_converted,
                                             dcyData_converted,
                                             hxData_converted,
                                             cxData_converted,
                                             weiData,
                                             reserveSpace,
                                             workSpace,
                                             nodhx,
                                             nodcx,
                                             nodhy,
                                             nodcy,
                                             nohx,
                                             nocx);
            std::vector<int> reverse_order = GetReverseOrderIndex(converted_seq_order);

            // IO
            ////////////////////////////////////////////////////////////////
            seqTensor<T> dx_tensor(xDesc);

            TransformRNNIOLayaoutToTarget(
                x_converted_desc, dx_tensor.desc, reverse_order, packed_res.din, dx_tensor.data);
            ////////////////////////////////////////////////////////////////

            std::vector<T> dhxData_converted{};
            if(!nodhx)
            {
                dhxData_converted.resize(packed_res.dhx.size());
                HiddenTensorReorder(packed_res.dhx, dhxData_converted, reverse_order, hid, true);
            }

            std::vector<T> dcxData_converted{};
            if(!nodcx)
            {
                dcxData_converted.resize(packed_res.dcx.size());
                HiddenTensorReorder(packed_res.dcx, dcxData_converted, reverse_order, hid, true);
            }

            return {dx_tensor.data, dhxData_converted, dcxData_converted, nodhx, nodcx};
        }
    }

    WrwResult wrw(const miopen::SeqTensorDescriptor& xDesc,
                  const miopen::SeqTensorDescriptor& dyDesc,
                  const std::vector<T>& xData,
                  const std::vector<T>& hxData,
                  const std::vector<T>& dyData,
                  std::vector<T>& reserveSpace,
                  std::vector<T>& workSpace,
                  bool nohx) const override
    {
        if(xDesc.IsPacked() && miopenRNNDataSeqMajorNotPadded ==
                                   miopen::RNNDescriptor::getBaseLayoutFromDataTensor(xDesc))
        {
            return packed_ref.wrw(
                xDesc, dyDesc, xData, hxData, dyData, reserveSpace, workSpace, nohx);
        }
        else
        {
            auto converted_seq_order =
                GetSamplesIndexDescendingOrder(xDesc.GetSequenceLengthsVector());

            // IO
            ////////////////////////////////////////////////////////////////
            seqTensor<T> x_tensor_converted(GetSeqDescriptorLayoutTransform(
                xDesc, miopenRNNDataSeqMajorNotPadded, converted_seq_order));

            seqTensor<T> dy_tensor_converted(GetSeqDescriptorLayoutTransform(
                dyDesc, miopenRNNDataSeqMajorNotPadded, converted_seq_order));

            TransformRNNIOLayaoutToTarget(xDesc,
                                          x_tensor_converted.desc,
                                          converted_seq_order,
                                          xData,
                                          x_tensor_converted.data);

            TransformRNNIOLayaoutToTarget(dyDesc,
                                          dy_tensor_converted.desc,
                                          converted_seq_order,
                                          dyData,
                                          dy_tensor_converted.data);
            ///////////////////////////////////////////////////////////////
            auto& hid = hiddenDesc.GetLengths();

            std::vector<T> hxData_converted{};
            if(!nohx)
            {
                hxData_converted.resize(hxData.size());
                HiddenTensorReorder(hxData, hxData_converted, converted_seq_order, hid, true);
            }

            auto packed_res = packed_ref.wrw(x_tensor_converted.desc,
                                             dy_tensor_converted.desc,
                                             x_tensor_converted.data,
                                             hxData_converted,
                                             dy_tensor_converted.data,
                                             reserveSpace,
                                             workSpace,
                                             nohx);

            return packed_res;
        }
    }

private:
    cpu_rnn_packed_ref<T> packed_ref;
    const miopen::TensorDescriptor hiddenDesc;

    inline cpu_rnn_packed_ref<T>
    ConstructPackedRNNRef(const miopen::RNNDescriptor& rnn,
                          const miopen::SeqTensorDescriptor& maxX,
                          const miopen::TensorDescriptor& hPackedDesc) const
    {
        if(maxX.IsPacked() && miopenRNNDataSeqMajorNotPadded ==
                                  miopen::RNNDescriptor::getBaseLayoutFromDataTensor(maxX))
        {
            return cpu_rnn_packed_ref<T>{rnn, maxX, hPackedDesc};
        }
        else
        {
            return cpu_rnn_packed_ref<T>{
                rnn,
                GetSeqDescriptorLayoutTransform(
                    maxX,
                    miopenRNNDataSeqMajorNotPadded,
                    GetSamplesIndexDescendingOrder(maxX.GetSequenceLengthsVector())),
                hPackedDesc};
        }
    }
};

template <class T>
struct verify_train_rnn : verify_rnn_api_base<T>
{
    using verify_rnn_api_base<T>::input;
    using verify_rnn_api_base<T>::output;
    using verify_rnn_api_base<T>::xHiddenState;
    using verify_rnn_api_base<T>::xCellState;
    using verify_rnn_api_base<T>::weights;
    using verify_rnn_api_base<T>::rnnDesc;

    using verify_rnn_api_base<T>::nohx;
    using verify_rnn_api_base<T>::nocx;
    using verify_rnn_api_base<T>::nohy;
    using verify_rnn_api_base<T>::nocy;

    bool nodhx{};
    bool nodcx{};
    bool nodhy{};
    bool nodcy{};

    using verify_rnn_api_base<T>::is_padded_verification;

    using verify_rnn_api_base<T>::padding_symbol;

    tensor<T> dyHiddenState{};
    tensor<T> dyCellState{};
    seqTensor<T> dOutput{};

    using VerificationObj = std::tuple<std::vector<T>,
                                       std::vector<T>,
                                       std::vector<T>,
                                       std::vector<T>,
                                       std::vector<T>,
                                       std::vector<T>,
                                       std::vector<T>>;

    VerificationObj result_tuple(const std::vector<T> fwd_y,
                                 const std::vector<T> fwd_hy,
                                 const std::vector<T> fwd_cy,
                                 const std::vector<T> bwd_din,
                                 const std::vector<T> bwd_dhx,
                                 const std::vector<T> bwd_dcx,
                                 const std::vector<T> wrw_dwei) const
    {
        return std::make_tuple(std::move(fwd_y),
                               std::move(fwd_hy),
                               std::move(fwd_cy),
                               std::move(bwd_din),
                               std::move(bwd_dhx),
                               std::move(bwd_dcx),
                               std::move(wrw_dwei));
    }

    verify_train_rnn(miopen::RNNDescriptor& pRD,
                     seqTensor<T>& x,
                     seqTensor<T>& y,
                     seqTensor<T>& dy,
                     tensor<T>& hx,
                     tensor<T>& cx,
                     tensor<T>& dhy,
                     tensor<T>& dcy,
                     std::vector<T>& w,
                     const bool pnohx = false,
                     const bool pnocx = false,
                     const bool pnohy = false,
                     const bool pnocy = false,
                     T* paddingSymbol = nullptr)
        : verify_rnn_api_base<T>(pRD, x, y, hx, cx, w, pnohx, pnocx, pnohy, pnocy, paddingSymbol),
          dyHiddenState(dhy),
          dyCellState(dcy),
          dOutput(dy)
    {
        nodhx = nohx;
        nodcx = nocx;
        nodhy = nohy;
        nodcy = nocy;
    }

    VerificationObj cpu() const
    {
        // auto&& handle = get_handle();

        cpu_rnn_universal_ref<T> refMethod{rnnDesc, input.desc, xHiddenState.desc};

        std::vector<T> reserve_space(refMethod.getReserveSpaceSize());
        std::vector<T> work_space(refMethod.getWorkSpaceSize());

        auto [fwd_y, fwd_hy, fwd_cy] = refMethod.fwd(input.desc,
                                                     output.desc,
                                                     input.data,
                                                     xHiddenState.data,
                                                     xCellState.data,
                                                     weights,
                                                     reserve_space,
                                                     nohx,
                                                     nocx,
                                                     nohy,
                                                     nocy);

        auto [bwd_din, bwd_dhx, bwd_dcx] = refMethod.bwd(input.desc,
                                                         output.desc,
                                                         dOutput.data,
                                                         dyHiddenState.data,
                                                         dyCellState.data,
                                                         xHiddenState.data,
                                                         xCellState.data,
                                                         weights,
                                                         reserve_space,
                                                         work_space,
                                                         nodhx,
                                                         nodcx,
                                                         nodhy,
                                                         nodcy,
                                                         nohx,
                                                         nocx);

        auto wrw_res = refMethod.wrw(input.desc,
                                     output.desc,
                                     input.data,
                                     xHiddenState.data,
                                     dOutput.data,
                                     reserve_space,
                                     work_space,
                                     nohx);

        // if(is_padded_verification)
        //{
        //    std::fill(output_seq.begin(), output_seq.end(), padding_symbol);
        //    ChangeDataPadding(*packed_output, output_seq, batch_seq, batch_seq[0], out_vec, true);
        //}

        return result_tuple(std::move(fwd_y),
                            std::move(fwd_hy),
                            std::move(fwd_cy),
                            std::move(bwd_din),
                            std::move(bwd_dhx),
                            std::move(bwd_dcx),
                            std::move(wrw_res.dwei));
    }

    VerificationObj gpu() const
    {
        auto&& handle = get_handle();

        size_t workSpaceByteSize =
            rnnDesc.GetMaxWorkspaceSize(handle, input.desc, miopenRNNFWDMode_t::miopenRNNTraining);

        size_t reserveSpaceByteSize = rnnDesc.GetMaxReserveSize(handle, input.desc);

        auto workSpace_dev    = handle.Create(workSpaceByteSize);
        auto reserveSpace_dev = handle.Create(reserveSpaceByteSize);

        auto x_dev  = transferTensorToGPUOrNullptr(handle, input, false);
        auto hx_dev = transferTensorToGPUOrNullptr(handle, xHiddenState, nohx);
        auto cx_dev = transferTensorToGPUOrNullptr(handle, xCellState, nocx);

        auto y_dev  = createTensorAtGPUOrNullptr(handle, output, false);
        auto hy_dev = createTensorAtGPUOrNullptr(handle, xHiddenState, nohy);
        auto cy_dev = createTensorAtGPUOrNullptr(handle, xCellState, nocy);

        auto weights_dev = handle.Write(weights);

        rnnDesc.RNNForward(handle,
                           miopenRNNFWDMode_t::miopenRNNTraining,
                           input.desc,
                           x_dev.get(),
                           xHiddenState.desc,
                           hx_dev.get(),
                           hy_dev.get(),
                           xCellState.desc, // cdesc
                           cx_dev.get(),
                           cy_dev.get(),
                           output.desc,
                           y_dev.get(),
                           weights_dev.get(),
                           weights.size() * sizeof(T),
                           workSpace_dev.get(),
                           workSpaceByteSize,
                           reserveSpace_dev.get(),
                           reserveSpaceByteSize);

        size_t workSpace_TCnt    = workSpaceByteSize / sizeof(T);
        size_t reserveSpace_TCnt = (reserveSpaceByteSize + sizeof(T) - 1) / sizeof(T);

        std::vector<T> reserveSpace_fwd_out(reserveSpace_TCnt);
        handle.ReadTo(
            reserveSpace_fwd_out.data(),
            reserveSpace_dev,
            reserveSpaceByteSize); // std::copy(reserveSpace.begin(), reserveSpace.end(), RSVgpu);

        const auto fwd_y  = handle.Read<T>(y_dev, output.GetSize());
        const auto fwd_hy = readTFromGPUOrEmpty(handle, hy_dev, xHiddenState, nohy);
        const auto fwd_cy = readTFromGPUOrEmpty(handle, cy_dev, xCellState, nocy);

        const auto dy_dev  = transferTensorToGPUOrNullptr(handle, dOutput, false);
        const auto dhy_dev = transferTensorToGPUOrNullptr(handle, dyHiddenState, nodhy);
        const auto dcy_dev = transferTensorToGPUOrNullptr(handle, dyCellState, nodcy);

        auto din_dev = createTensorAtGPUOrNullptr(handle, input, false);
        auto dhx_dev = createTensorAtGPUOrNullptr(handle, xHiddenState, nodhx);
        auto dcx_dev = createTensorAtGPUOrNullptr(handle, xCellState, nodcx);

        const auto tmp_din = handle.Read<T>(din_dev, input.GetSize());

        rnnDesc.RNNBackwardData(handle,
                                dOutput.desc,
                                nullptr,
                                dy_dev.get(),
                                xHiddenState.desc,
                                hx_dev.get(),
                                dhy_dev.get(),
                                dhx_dev.get(),
                                xCellState.desc,
                                cx_dev.get(),
                                dcy_dev.get(),
                                dcx_dev.get(),
                                input.desc,
                                din_dev.get(),
                                weights_dev.get(),
                                weights.size() * sizeof(T),
                                workSpace_dev.get(),
                                workSpaceByteSize,
                                reserveSpace_dev.get(),
                                reserveSpaceByteSize);

        const auto bwd_din = handle.Read<T>(din_dev, input.GetSize());
        const auto bwd_dhx = readTFromGPUOrEmpty(handle, dhx_dev, xHiddenState, nodhx);
        const auto bwd_dcx = readTFromGPUOrEmpty(handle, dcx_dev, xCellState, nodcx);

        std::vector<T> workSpace_bwd_out(workSpace_TCnt);
        handle.ReadTo(workSpace_bwd_out.data(), workSpace_dev, workSpaceByteSize);

        auto dweights_dev = handle.Create(weights.size() * sizeof(T));

        rnnDesc.RNNBackwardWeights(handle,
                                   input.desc,
                                   x_dev.get(),
                                   xHiddenState.desc,
                                   hx_dev.get(),
                                   output.desc,
                                   dy_dev.get(),
                                   dweights_dev.get(),
                                   weights.size() * sizeof(T),
                                   workSpace_dev.get(),
                                   workSpaceByteSize,
                                   reserveSpace_dev.get(),
                                   reserveSpaceByteSize);

        const auto wrw_dwei = handle.Read<T>(dweights_dev, weights.size());

        // if(!is_padded_verification)
        //{
        //    MIOPEN_THROW("TODO.");
        //    return result_tuple(fwd_y, fwd_hy, fwd_cy, bwd_din, bwd_dhx, bwd_dcx, wrw_dwei);
        //}

        return result_tuple(fwd_y, fwd_hy, fwd_cy, bwd_din, bwd_dhx, bwd_dcx, wrw_dwei);
    }
};

//****************************************************
// RNN inference fwd
//****************************************************

template <class T>
struct verify_inference_rnn : verify_rnn_api_base<T>
{

    using verify_rnn_api_base<T>::input;
    using verify_rnn_api_base<T>::output;
    using verify_rnn_api_base<T>::xHiddenState;
    using verify_rnn_api_base<T>::xCellState;
    using verify_rnn_api_base<T>::weights;
    using verify_rnn_api_base<T>::rnnDesc;
    using verify_rnn_api_base<T>::nohx;
    using verify_rnn_api_base<T>::nocx;
    using verify_rnn_api_base<T>::nohy;
    using verify_rnn_api_base<T>::nocy;
    using verify_rnn_api_base<T>::is_padded_verification;

    using verify_rnn_api_base<T>::padding_symbol;

    tensor<T> dyHiddenState{};
    tensor<T> dyCellState{};
    seqTensor<T> dOutput{};

    using VerificationObj = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>>;

    VerificationObj
    result_tuple(std::vector<T>&& fwd_y, std::vector<T>&& fwd_hy, std::vector<T>&& fwd_cy)
    {
        return std::make_tuple(fwd_y, fwd_hy, fwd_cy);
    }

    VerificationObj gpu() const
    {
        auto&& handle = get_handle();

        size_t workSpaceByteSize = 0;

        miopenGetRNNTempSpaceSizes(&handle,
                                   &rnnDesc,
                                   &input.desc,
                                   miopenRNNFWDMode_t::miopenRNNInference,
                                   &workSpaceByteSize,
                                   nullptr);

        auto workSpace_dev = handle.Create(workSpaceByteSize);

        auto x_dev  = transferTensorToGPUOrNullptr(handle, input, false);
        auto hx_dev = transferTensorToGPUOrNullptr(handle, xHiddenState, nohx);
        auto cx_dev = transferTensorToGPUOrNullptr(handle, xCellState, nocx);

        auto y_dev  = createTensorAtGPUOrNullptr(handle, output, false);
        auto hy_dev = createTensorAtGPUOrNullptr(handle, xHiddenState, nohy);
        auto cy_dev = createTensorAtGPUOrNullptr(handle, xCellState, nocy);

        auto weights_dev = handle.Write(weights);

        miopenRNNForward(&handle,
                         &rnnDesc,
                         miopenRNNFWDMode_t::miopenRNNInference,
                         &input.desc,
                         x_dev.get(),
                         &xHiddenState.desc,
                         hx_dev.get(),
                         hy_dev.get(),
                         &xCellState.desc,
                         cx_dev.get(),
                         cy_dev.get(),
                         &output.desc,
                         y_dev.get(),
                         weights_dev.get(),
                         weights.size() * sizeof(T),
                         workSpace_dev.get(),
                         workSpaceByteSize,
                         nullptr,
                         0);

        const auto fwd_y  = handle.Read<T>(y_dev, output.GetSize());
        const auto fwd_hy = readTFromGPUOrEmpty<T>(handle, hy_dev, xHiddenState.GetSize(), nohy);
        const auto fwd_cy = readTFromGPUOrEmpty<T>(handle, cy_dev, xCellState.GetSize(), nocy);

        return result_tuple(fwd_y, fwd_hy, fwd_cy);
    }

    size_t total_GPU_mem_size() {}
    size_t input_GPU_mem_size() {}
    size_t output_GPU_mem_size() {}
    size_t workspace_GPU_mem_size() {}
    size_t reservspace_GPU_mem_size() {}
};

template <class T>
constexpr seqTensor<T> build_RNN_seqTensor(miopenRNNBaseLayout_t layout,
                                           int batchSize,
                                           int maxSequenceLen,
                                           int vectorSize,
                                           std::vector<int>& sequenceLenArray,
                                           void* paddingMarker = nullptr)
{
    return {miopen::RNNDescriptor::makeSeqTensorDescriptor(miopen_type<T>{},
                                                           layout,
                                                           maxSequenceLen,
                                                           batchSize,
                                                           vectorSize,
                                                           sequenceLenArray.data(),
                                                           paddingMarker)};
}

constexpr miopenRNNBaseLayout_t rnn_data_layout(int io_layout)
{
    switch(io_layout)
    {
    case 1: return miopenRNNDataSeqMajorNotPadded;
    case 2: return miopenRNNDataSeqMajorPadded;
    case 3: return miopenRNNDataBatchMajorPadded;
    default: MIOPEN_THROW("Incorrect input, unsupported RNNLayout.");
    }
}

inline size_t get_RNN_params_byteSize(miopen::Handle& handle,
                                      miopen::RNNDescriptor& rnnDesc,
                                      miopen::SeqTensorDescriptor& inTensor)
{
    auto& in_lens                     = inTensor.GetLengths();
    const std::vector<size_t> in_dims = {in_lens[0], in_lens[2]};
    miopen::TensorDescriptor baseInputDesc(rnnDesc.dataType, in_dims);
    size_t wei_bytes = 0;

    miopenGetRNNParamsSize(&handle, &rnnDesc, &baseInputDesc, &wei_bytes, rnnDesc.dataType);

    return wei_bytes;
}

template <class T>
struct rnn_seq_api_test_driver : test_driver
{
    std::vector<int> seqLenArray;
    int seqLength{};
    int inVecLen{};
    int hiddenSize{};
    int numLayers{};
    int inputMode{};
    int biasMode{};
    int dirMode{};
    int rnnMode{};
    int algoMode{};
    int batchSize{};
    int useDropout{};
    int io_layout{};

    // Null pointer input
    bool nohx{};
    bool nocx{};
    bool nohy{};
    bool nocy{};

    bool pytorchTensorDescriptorFormat{};

    rnn_seq_api_test_driver() {}

    bool check_GPU_mem_limit(miopen::Handle& handle,
                             miopen::RNNDescriptor& rnnDesc,
                             seqTensor<T>& input,
                             seqTensor<T>& output,
                             tensor<T>& hx,
                             tensor<T>& cx,
                             size_t weightsByteSize,
                             size_t statesSizeInBytes)
    {

        size_t train_workSpace_size, train_reserveSpace_size;
        miopenGetRNNTempSpaceSizes(&handle,
                                   &rnnDesc,
                                   &input.desc,
                                   miopenRNNTraining,
                                   &train_workSpace_size,
                                   &train_reserveSpace_size);

        size_t inference_workSpace_size;
        miopenGetRNNTempSpaceSizes(
            &handle, &rnnDesc, &input.desc, miopenRNNInference, &inference_workSpace_size, nullptr);

        auto tmp_mem =
            std::max(inference_workSpace_size, train_workSpace_size + train_reserveSpace_size);

        size_t total_mem = statesSizeInBytes + tmp_mem +
                           (2 * output.GetSize() + input.GetSize() + weightsByteSize +
                            (nohx ? 0 : 2 * hx.GetSize()) + (nohy ? 0 : 2 * hx.GetSize()) +
                            (nocx ? 0 : 2 * cx.GetSize()) + (nocy ? 0 : 2 * cx.GetSize())) *
                               sizeof(T);

        size_t device_mem = handle.GetGlobalMemorySize();

        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return false;
        }
        return true;
    }

    void fill_buffers(seqTensor<T>& input,
                      seqTensor<T>& dy,
                      tensor<T>& hx,
                      tensor<T>& cx,
                      tensor<T>& dhy,
                      tensor<T>& dcy,
                      std::vector<T>& weights)
    {
        const double scale     = 0.1;
        const double bwd_scale = scale;

        struct scalar_gen_random_float
        {
            double min_val = 0;
            double max_val = 1;

            double operator()() const
            {
                return prng::gen_A_to_B(static_cast<T>(min_val), static_cast<T>(max_val));
            }
        };

        auto gen_positive_value = [=](auto...) {
            return scalar_gen_random_float{std::numeric_limits<T>::epsilon(), 1 * scale}();
        };

        auto gen_positive_value_bwd = [=](auto...) {
            double bwd_max = 1. * scale;
            double bwd_min = std::numeric_limits<T>::epsilon();
            return scalar_gen_random_float{bwd_min, bwd_max}();
        };

        auto fill_array_via_gen = [](auto& dst, size_t dst_sz, double range_l, double range_r) {
            for(size_t it = 0; it < dst_sz; it++)
                dst[it] = prng::gen_A_to_B(static_cast<T>(range_l), static_cast<T>(range_r));
        };
        prng::reset_seed();
        fill_array_via_gen(
            input.data, input.data.size(), std::numeric_limits<T>::epsilon(), 1. * scale);
        prng::reset_seed();
        fill_array_via_gen(
            dy.data, dy.data.size(), std::numeric_limits<T>::epsilon(), 1. * bwd_scale);
        prng::reset_seed();

        const auto hidden_size = hx.desc.GetLengths()[2];
        const double wei_range = sqrt(1. / hidden_size);
        fill_array_via_gen(weights, weights.size(), -wei_range, wei_range);

        if(!nohx)
        {
            hx.generate(gen_positive_value);
        }

        if(!nocx)
        {
            cx.generate(gen_positive_value);
        }

        if(!nohy)
        {
            dhy.generate(gen_positive_value_bwd);
        }

        if(!nocy)
        {
            dcy.generate(gen_positive_value_bwd);
        }
    }

    void args_update()
    {

        if(io_layout == 1 && (!seqLenArray.empty()) &&
           (!std::is_sorted(seqLenArray.begin(), seqLenArray.end(), std::greater<>{})))
        {
            MIOPEN_THROW("Incorrect input, seq_lens should not to increase with "
                         "miopenRNNDataSeqMajorNotPadded layout\n");
        }

        if(!seqLenArray.empty())
        {
            if(seqLenArray.size() < batchSize)
            {

                int padding_val = 0;
                printf("sampl_lens size == %zu is shmaller than time batch_size == %d, padding the "
                       "rest "
                       "of data with %d\n",
                       seqLenArray.size(),
                       batchSize,
                       padding_val);

                std::vector<int> new_seqLenArray(batchSize);

                std::copy_n(seqLenArray.begin(), seqLenArray.size(), new_seqLenArray.begin());
                std::fill_n(new_seqLenArray.begin() + seqLenArray.size(),
                            batchSize - seqLenArray.size(),
                            padding_val);
                seqLenArray = new_seqLenArray;
            }
            size_t seq_max_element = *std::max_element(seqLenArray.begin(), seqLenArray.end());

            if(seqLength < seq_max_element)
                MIOPEN_THROW(
                    "Incorrect input, seq_lens elements should be smaller or equal to seqLength\n");
        }
        else
        {
            printf("Empty batch sequence. Filling uniformly with max_seqLength:%d\n ", seqLength);
            seqLenArray.resize(batchSize, seqLength);
        }
    }

    void run()
    {
        args_update();

        auto&& handle = get_handle();

        miopen::RNNDescriptor rnnDesc{};
        miopen::DropoutDescriptor dropoutDesc{};

        size_t statesSizeInBytes = 0;

        if(useDropout != 0)
        {
            float dropout_rate              = 0.25;
            unsigned long long dropout_seed = 0ULL;
            miopenDropoutGetStatesSize(&handle, &statesSizeInBytes);

            void* dropout_state_buf;
            hipMalloc(static_cast<void**>(&dropout_state_buf), statesSizeInBytes);

            miopenSetDropoutDescriptor(&dropoutDesc,
                                       &handle,
                                       dropout_rate,
                                       dropout_state_buf,
                                       statesSizeInBytes,
                                       dropout_seed,
                                       false,
                                       false,
                                       MIOPEN_RNG_PSEUDO_XORWOW);

            miopenSetRNNDescriptor_V2(&rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      &dropoutDesc,
                                      miopenRNNInputMode_t(inputMode),
                                      miopenRNNDirectionMode_t(dirMode),
                                      miopenRNNMode_t(rnnMode),
                                      miopenRNNBiasMode_t(biasMode),
                                      miopenRNNAlgo_t(algoMode),
                                      type);
        }
        else
        {
            miopenSetRNNDescriptor(&rnnDesc,
                                   hiddenSize,
                                   numLayers,
                                   miopenRNNInputMode_t(inputMode),
                                   miopenRNNDirectionMode_t(dirMode),
                                   miopenRNNMode_t(rnnMode),
                                   miopenRNNBiasMode_t(biasMode),
                                   miopenRNNAlgo_t(algoMode),
                                   type);
        }

        const auto inOut_layout   = rnn_data_layout(io_layout);
        const auto out_vector_len = hiddenSize * ((dirMode != 0) ? 2 : 1);

        T padding_m = static_cast<T>(0);

        seqTensor<T> input = build_RNN_seqTensor<T>(
                         inOut_layout, batchSize, seqLength, inVecLen, seqLenArray, &padding_m),
                     output = build_RNN_seqTensor<T>(inOut_layout,
                                                     batchSize,
                                                     seqLength,
                                                     out_vector_len,
                                                     seqLenArray,
                                                     &padding_m);
        seqTensor<T> dy(output);

        const auto num_hidden_layers = numLayers * ((dirMode != 0) ? 2 : 1);
        tensor<T> hx                 = [=]() {
            if(pytorchTensorDescriptorFormat)
                return tensor<T>(std::vector{num_hidden_layers, batchSize, hiddenSize, 1, 1});
            else
                return tensor<T>(std::vector{num_hidden_layers, batchSize, hiddenSize});
        }();

        tensor<T> cx(hx), dhy(hx), dcy(hx);

        std::vector<T> weights(get_RNN_params_byteSize(handle, rnnDesc, input.desc) / sizeof(T));

        check_GPU_mem_limit(
            handle, rnnDesc, input, output, hx, cx, weights.size() * sizeof(T), statesSizeInBytes);

        fill_buffers(input, dy, hx, cx, dhy, dcy, weights);

        // avoid BWD unexpected fails
        // https://github.com/ROCm/MIOpen/pull/2493#discussion_r1406959588
        if(inVecLen == 1 && hiddenSize == 13 && seqLength == 1 && batchSize == 1)
        {
            tolerance = 110;
        }
        else
        {
            tolerance = 80;
        }

        auto fwdTrain = verify(verify_train_rnn<T>{
            rnnDesc, input, output, dy, hx, cx, dhy, dcy, weights, nohx, nocx, nohy, nocy});
    }
};
