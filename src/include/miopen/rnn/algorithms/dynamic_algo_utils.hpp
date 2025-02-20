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

#include "miopen/rnn/algorithms/default_algo_utils.hpp"

namespace miopen {

namespace rnn_base {

namespace rnn_dynamic {

// 12.5% overhead
constexpr int rounding_limit = 4;

inline size_t getPow2Mask(size_t pow_2)
{
    auto mask = pow_2;
    for(int i = 0; i < rounding_limit; i++)
    {
        mask = (mask >> 1) | pow_2;
    }
    return mask & ~(3);
}

inline size_t getPow2Minimum(size_t pow_2)
{
    for(int i = 0; i < rounding_limit; i++)
    {
        pow_2 >>= 1;
    }
    return pow_2 < 4 ? 4 : pow_2;
}

inline size_t getLowerBoundPow2(size_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
};

// Decomposing a number into a sum of powers of two.
class MaskedPow2Range
{
public:
    class Iterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type   = size_t;
        using reference         = size_t;
        using pointer           = size_t*;
        using value_type        = size_t;

        static Iterator BuildBegin(size_t mask)
        {
            Iterator begin{mask, 0};
            return (mask & 1) != 0u ? begin : ++begin;
        }

        static Iterator BuildEnd(size_t mask) { return {mask, MAX_BIT_POS}; }

        Iterator(size_t range_mask, int bit_position) : mask(range_mask), bitPosition(bit_position)
        {
        }

        size_t operator*() const { return (1LL) << bitPosition; }

        Iterator& operator++()
        {
            bitPosition++;
            while(bitPosition < MAX_BIT_POS && !BitExistenceCheck())
            {
                bitPosition++;
            }
            return *this;
        }

        Iterator& operator--()
        {
            if(bitPosition > 0)
                bitPosition--;

            while(bitPosition > 0 && !BitExistenceCheck())
            {
                bitPosition--;
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const { return bitPosition != other.bitPosition; }

    private:
        size_t mask;
        unsigned int bitPosition;

        bool BitExistenceCheck() const { return (mask & ((1LL) << bitPosition)) != 0u; }

        static constexpr unsigned int MAX_BIT_POS = sizeof(size_t) * 8;
    };

    explicit MaskedPow2Range(size_t range_mask) : mask(range_mask) {}

    Iterator begin() const { return Iterator::BuildBegin(mask); }

    Iterator end() const { return Iterator::BuildEnd(mask); }

    auto rbegin() const { return std::reverse_iterator<Iterator>(end()); }

    auto rend() const { return std::reverse_iterator<Iterator>{begin()}; }

private:
    size_t mask;
};

} // namespace rnn_dynamic

inline std::vector<size_t> roundedDynamicLengths(const SeqTensorDescriptor& desc)
{
    auto src_lens     = desc.GetLengths();
    auto real_seq_len = src_lens[1];

    auto lower_bound_pow2 = rnn_dynamic::getLowerBoundPow2(real_seq_len);
    auto acc_len          = real_seq_len & rnn_dynamic::getPow2Mask(lower_bound_pow2);

    if(real_seq_len > acc_len)
    {
        acc_len += rnn_dynamic::getPow2Minimum(lower_bound_pow2);
    }

    src_lens[1] = acc_len;

    return src_lens;
}

inline SeqTensorDescriptor buildDynamicVirtual(const SeqTensorDescriptor& desc)
{
    std::vector<unsigned int> def_layout{1, 0, 2};
    return {desc.GetType(), def_layout, roundedDynamicLengths(desc), false};
}

inline SeqTensorDescriptor buildRealToDynamicMapTmp(const SeqTensorDescriptor& desc)
{
    std::vector<unsigned int> def_layout{1, 0, 2};

    auto zero_val_padding = [](miopenDataType_t type) {
        std ::vector<char> padding_fill(GetTypeSize(type), 0);
        return padding_fill;
    };

    return {desc.GetType(),
            def_layout,
            desc.GetLengths(),
            desc.GetSequenceLengthsVector(),
            zero_val_padding(desc.GetType()),
            true,
            true};
}

class RNNModuleAlgoDynamic : public RNNForwardDataModularAlgo
{

public:
    RNNModuleAlgoDynamic(const RNNDescriptor& rnnD,
                         const SeqTensorDescriptor& xTDesc,
                         const SeqTensorDescriptor& yTDesc,
                         const TensorDescriptor& hDesc,
                         miopenRNNFWDMode_t mode)
        : RNNForwardDataModularAlgo(RNNModuleAlgoBase::create(
              rnnD, buildDynamicVirtual(xTDesc), buildDynamicVirtual(yTDesc), hDesc, mode)),
          realBatchController(BatchController::Create(xTDesc)),
          realXDesc(xTDesc),
          realYDesc(yTDesc),
          tmpMapXDesc(buildRealToDynamicMapTmp(xTDesc)),
          tmpMapYDesc(buildRealToDynamicMapTmp(yTDesc))
    {
    }

    struct runtimeArgsFwdDynamicExt
    {
        const ConstData_t realX;
        const Data_t tempX;
        const ConstData_t hx;
        const ConstData_t cx;
        const Data_t realY;
        const Data_t tempY;
        const Data_t hy;
        const Data_t cy;
        const ConstData_t w;
        const Data_t workSpace;
        const Data_t reserveSpace;
    };

    runtimeArgsFwdDynamicExt createRuntimeArgsExt(const runtimeArgsFwd& runtimeArgs) const
    {
        const Data_t temp_x =
            moveDataPtr(runtimeArgs.workSpace, workspaceInfo.getBufferSizeImpl(), rnnDesc.dataType);

        const Data_t temp_y = moveDataPtrByte(temp_x, tmpMapXDesc.GetTensorMaxByteSpace());

        return {
            runtimeArgs.x,
            temp_x,
            runtimeArgs.hx,
            runtimeArgs.cx,
            runtimeArgs.y,
            temp_y,
            runtimeArgs.hy,
            runtimeArgs.cy,
            runtimeArgs.w,
            runtimeArgs.workSpace,
            runtimeArgs.reserveSpace,
        };
    }

    auto getTempBuffersSize(const Handle& handle) const
    {
        auto [ws_size, reserve_size] = RNNForwardDataModularAlgo::getTempBuffersSize(handle);

        return std::make_tuple(ws_size + tmpMapXDesc.GetTensorMaxByteSpace() +
                                   tmpMapYDesc.GetTensorMaxByteSpace(),
                               reserve_size);
    }

    static auto getTempBuffersSize(const Handle& handle,
                                   const RNNDescriptor& rnnD,
                                   const SeqTensorDescriptor& xDesc)
    {
        auto y_desc = [](const RNNDescriptor& rnnD, const SeqTensorDescriptor& xDesc) {
            std::vector<size_t> y_lenghts{xDesc.GetLengths()};
            y_lenghts[2] = rnnD.hsize * (rnnD.dirMode == miopenRNNbidirection ? 2 : 1);
            return SeqTensorDescriptor{xDesc.GetType(), y_lenghts};
        }(rnnD, xDesc);

        auto temp_x_desc = buildDynamicVirtual(xDesc);
        auto temp_y_desc = buildDynamicVirtual(y_desc);

        auto [ws_size, reserve_size] =
            RNNForwardDataModularAlgo::getTempBuffersSize(handle, rnnD, temp_x_desc);

        return std::make_tuple(ws_size + temp_x_desc.GetTensorMaxByteSpace() +
                                   temp_y_desc.GetTensorMaxByteSpace(),
                               reserve_size);
    }

    void realXProp(const Handle& handle, const runtimeArgsFwdDynamicExt& runtimeArgsExt) const;

    void realYProp(const Handle& handle, const runtimeArgsFwdDynamicExt& runtimeArgsExt) const;

    void PrepareWriteBuffers(const Handle& handle,
                             const runtimeArgsFwdDynamicExt& runtimeArgsExt,
                             const runtimeArgsFwd& runtimeArgs) const;

    void PropX(const Handle& handle, const runtimeArgsFwd& runtimeArgs) const;

    void PropHiddenY(const Handle& handle,
                     const runtimeArgsFwd& runtimeArgs,
                     size_t layer,
                     SequenceDirection direction) const;

    void PropHyCy(const Handle& handle,
                  const runtimeArgsFwdDynamicExt& runtimeArgs,
                  size_t layer,
                  const SequenceIterator& currentSeq,
                  SequenceDirection direction) const;

    inline size_t getRealTimeSeqSize() const { return realBatchController.size(); }

private:
    const BatchController realBatchController;

    const SeqTensorDescriptor realXDesc;
    const SeqTensorDescriptor realYDesc;
    const SeqTensorDescriptor tmpMapXDesc;
    const SeqTensorDescriptor tmpMapYDesc;
};

class RNNBackwardModuleAlgoDynamic : public RNNBackwardDataModularAlgo
{
    using BaseBWDModuleT = rnn_base::RNNBackwardDataModularAlgo;

public:
    RNNBackwardModuleAlgoDynamic(const RNNDescriptor& rnnD,
                                 const SeqTensorDescriptor& xTDesc,
                                 const SeqTensorDescriptor& yTDesc,
                                 const TensorDescriptor& hDesc,
                                 miopenRNNFWDMode_t mode)
        : BaseBWDModuleT(RNNModuleAlgoBase::create(
              rnnD, buildDynamicVirtual(xTDesc), buildDynamicVirtual(yTDesc), hDesc, mode)),
          realBatchController(BatchController::Create(xTDesc)),
          realDxDesc(xTDesc),
          realDyDesc(yTDesc),
          tmpMapDxDesc(buildRealToDynamicMapTmp(xTDesc)),
          tmpMapDyDesc(buildRealToDynamicMapTmp(yTDesc))
    {
    }

    struct runtimeArgsBwdDynamicExt
    {
        const ConstData_t realDy;
        const Data_t tempDy;
        const ConstData_t dhy;
        const Data_t dhx;
        const ConstData_t cx;
        const ConstData_t dcy;
        const Data_t dcx;
        const Data_t realDx;
        const Data_t tempDx;
        const ConstData_t w;
        const Data_t workSpace;
        const Data_t reserveSpace;
    };

    runtimeArgsBwdDynamicExt createRuntimeArgsExt(const runtimeArgsBwd& runtimeArgs) const
    {
        const Data_t temp_dx =
            moveDataPtr(runtimeArgs.workSpace, workspaceInfo.getBufferSizeImpl(), rnnDesc.dataType);

        const Data_t temp_dy = moveDataPtrByte(temp_dx, tmpMapDxDesc.GetTensorMaxByteSpace());

        return {
            runtimeArgs.dy,
            temp_dy,
            runtimeArgs.dhy,
            runtimeArgs.dhx,
            runtimeArgs.cx,
            runtimeArgs.dcy,
            runtimeArgs.dcx,
            runtimeArgs.dx,
            temp_dx,
            runtimeArgs.w,
            runtimeArgs.workSpace,
            runtimeArgs.reserveSpace,
        };
    }

    auto getTempBuffersSize(const Handle& handle) const
    {
        auto [ws_size, reserve_size] = BaseBWDModuleT::getTempBuffersSize(handle);

        return std::make_tuple(ws_size + tmpMapDxDesc.GetTensorMaxByteSpace() +
                                   tmpMapDyDesc.GetTensorMaxByteSpace(),
                               reserve_size);
    }

    static auto getTempBuffersSize(const Handle& handle,
                                   const RNNDescriptor& rnnD,
                                   const SeqTensorDescriptor& xDesc)
    {
        auto y_desc = [](const RNNDescriptor& rnnD, const SeqTensorDescriptor& xDesc) {
            std::vector<size_t> y_lenghts{xDesc.GetLengths()};
            y_lenghts[2] = rnnD.hsize * (rnnD.dirMode == miopenRNNbidirection ? 2 : 1);
            return SeqTensorDescriptor{xDesc.GetType(), y_lenghts};
        }(rnnD, xDesc);

        auto temp_x_desc = buildDynamicVirtual(xDesc);
        auto temp_y_desc = buildDynamicVirtual(y_desc);

        auto [ws_size, reserve_size] =
            RNNForwardDataModularAlgo::getTempBuffersSize(handle, rnnD, temp_x_desc);

        return std::make_tuple(ws_size + temp_x_desc.GetTensorMaxByteSpace() +
                                   temp_y_desc.GetTensorMaxByteSpace(),
                               reserve_size);
    }

    void realDxProp(const Handle& handle, const runtimeArgsBwdDynamicExt& runtimeArgsExt) const;

    void realDyProp(const Handle& handle, const runtimeArgsBwdDynamicExt& runtimeArgsExt) const;

    void realPropDhy(const Handle& handle,
                     ConstData_t dhy,
                     Data_t workSpace,
                     unsigned int layer,
                     const SequenceIterator& currentSeq,
                     SequenceDirection direction) const;

    void realUpdateHStatePerTimeSeq(const Handle& handle,
                                    ConstData_t dcy,
                                    ConstData_t cx,
                                    Data_t,
                                    Data_t workSpace,
                                    Data_t reserveSpace,
                                    int layer,
                                    const SequenceIterator& seq,
                                    SequenceDirection direction) const;

    void PrepareWriteBuffers(const Handle& handle,
                             const runtimeArgsBwdDynamicExt& runtimeArgsExt) const;

    void PropDx(const Handle& handle,
                ConstData_t w,
                ConstData_t workSpace,
                Data_t dx,
                SequenceDirection direction) const;

    void PropHiddenDy(const Handle& handle,
                      ConstData_t w,
                      Data_t workSpace,
                      Data_t reserveSpace,
                      size_t layer,
                      SequenceDirection direction) const;

    inline size_t getRealTimeSeqSize() const { return realBatchController.size(); }

private:
    BatchController realBatchController;

    SeqTensorDescriptor realDxDesc;
    SeqTensorDescriptor realDyDesc;
    SeqTensorDescriptor tmpMapDxDesc;
    SeqTensorDescriptor tmpMapDyDesc;
};

class RNNBackwardWeiModuleAlgoDynamic : public RNNBackwardWeightsModularAlgo
{
    using BaseBWDModuleT = rnn_base::RNNBackwardWeightsModularAlgo;

public:
    RNNBackwardWeiModuleAlgoDynamic(const RNNDescriptor& rnnD,
                                    const SeqTensorDescriptor& xTDesc,
                                    const SeqTensorDescriptor& yTDesc,
                                    const TensorDescriptor& hDesc,
                                    miopenRNNFWDMode_t mode)
        : BaseBWDModuleT(RNNModuleAlgoBase::create(
              rnnD, buildDynamicVirtual(xTDesc), buildDynamicVirtual(yTDesc), hDesc, mode)),
          realBatchController(BatchController::Create(xTDesc)),
          realXDesc(xTDesc),
          tmpMapXDesc(buildRealToDynamicMapTmp(xTDesc))

    {
    }

    struct runtimeArgsBwWeiDynamicExt
    {
        const ConstData_t realX;
        const Data_t tempX;
        const ConstData_t hx;
        const Data_t dw;
        const ConstData_t backData;
        const ConstData_t forwardData;
        const Data_t freeWorkSpace;
        const size_t freeWorkSpaceSize;
    };

    runtimeArgsBwWeiDynamicExt createRuntimeArgsExt(const runtimeArgsBWWeights& runtimeArgs) const
    {
        const Data_t temp_x         = runtimeArgs.freeWorkSpace;
        const auto temp_x_byte_size = tmpMapXDesc.GetTensorMaxByteSpace();

        const Data_t free_ws = moveDataPtrByte(temp_x, temp_x_byte_size);

        return {runtimeArgs.x,
                temp_x,
                runtimeArgs.hx,
                runtimeArgs.dw,
                runtimeArgs.backData,
                runtimeArgs.forwardData,
                free_ws,
                runtimeArgs.freeWorkSpaceSize - temp_x_byte_size};
    }

    auto getTempBuffersSize(const Handle& handle) const
    {
        auto [ws_size, reserve_size] = BaseBWDModuleT::getTempBuffersSize(handle);

        return std::make_tuple(ws_size + tmpMapXDesc.GetTensorMaxByteSpace() + reserve_size);
    }

    static auto getTempBuffersSize(const Handle& handle,
                                   const RNNDescriptor& rnnD,
                                   const SeqTensorDescriptor& xDesc)
    {
        auto y_desc = [](const RNNDescriptor& rnnD, const SeqTensorDescriptor& xDesc) {
            std::vector<size_t> y_lenghts{xDesc.GetLengths()};
            y_lenghts[2] = rnnD.hsize * (rnnD.dirMode == miopenRNNbidirection ? 2 : 1);
            return SeqTensorDescriptor{xDesc.GetType(), y_lenghts};
        }(rnnD, xDesc);

        auto temp_x_desc = buildDynamicVirtual(xDesc);
        auto temp_y_desc = buildDynamicVirtual(y_desc);

        auto [ws_size, reserve_size] =
            RNNForwardDataModularAlgo::getTempBuffersSize(handle, rnnD, temp_x_desc);

        return std::make_tuple(ws_size + temp_x_desc.GetTensorMaxByteSpace() +
                                   temp_y_desc.GetTensorMaxByteSpace(),
                               reserve_size);
    }

    void HiddenXInputWeights(const Handle& handle,
                             Data_t dw,
                             ConstData_t workSpace,
                             ConstData_t reserveSpace,
                             size_t layer) const;

    void
    PhisXInputWeights(const Handle& handle, Data_t dw, ConstData_t workSpace, ConstData_t x) const;

    void PhisHStateWeights(const Handle& handle,
                           Data_t dw,
                           ConstData_t workSpace,
                           ConstData_t hx,
                           const SequenceIterator& seq,
                           size_t layer,
                           SequenceDirection direction) const;

    void PhisHStateWeights(const Handle& handle,
                           Data_t dw,
                           ConstData_t workSpace,
                           ConstData_t hx,
                           size_t layer,
                           size_t max_seq_len,
                           SequenceDirection direction) const
    {
        if(hx == nullptr)
            return;

        for(auto i = max_seq_len; i > 0; i--)
        {
            const auto seq = SequenceIterator(i - 1, direction, max_seq_len, false);

            PhisHStateWeights(handle, dw, workSpace, hx, seq, layer, direction);
        }
    }

    void realXProp(const Handle& handle, const runtimeArgsBwWeiDynamicExt& runtimeArgsExt) const
    {
        const auto normalized_tensor_size =
            tmpMapXDesc.GetTensorMaxByteSpace() / GetTypeSize(rnnDesc.dataType);

        const auto normalized_desc = miopen::TensorDescriptor(
            rnnDesc.dataType, {1, normalized_tensor_size}, {normalized_tensor_size, 1});

        const float beta = 0.;

        SetTensor(handle, normalized_desc, runtimeArgsExt.tempX, &beta);

        RNNTensorBaseLayoutConverter::ConvertInputTensorGPUData(handle,
                                                                realXDesc,
                                                                runtimeArgsExt.realX,
                                                                tmpMapXDesc,
                                                                runtimeArgsExt.tempX,
                                                                nullptr,
                                                                false);
    }

    inline size_t getRealTimeSeqSize() const { return realBatchController.size(); }

private:
    BatchController realBatchController;

    SeqTensorDescriptor realXDesc;
    SeqTensorDescriptor tmpMapXDesc;
};

} // namespace rnn_base
} // namespace miopen
