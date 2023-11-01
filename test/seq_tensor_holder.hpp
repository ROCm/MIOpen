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

#include "tensor_holder.hpp"
//#include "rnn_util.hpp"

#include <miopen/seq_tensor.hpp>

template <class DataT, class IndexT>
std::vector<DataT> GetReorderedVector(const std::vector<DataT>& src,
                                      const std::vector<IndexT>& resOrder)
{
    if(resOrder.size() != src.size())
        MIOPEN_THROW("data and index size mismatch resOrder.size() != src.size()");

    std::vector<DataT> reordered_result(src.size());
    auto result_iter = reordered_result.begin();

    for(auto i : resOrder)
        *result_iter++ = src[i];

    return reordered_result;
}

struct SeqTensorOffsets
{
    SeqTensorOffsets(const miopen::SeqTensorDescriptor& sDesc)
        : major_dim_part_sum(partSum(sDesc)), desc(sDesc)
    {
    }

    template <typename T>
    size_t operator()(const T pos) const
    {

        size_t pos_offset = 0;

        const auto& lens = desc.GetLengths();

        if(desc.IsPaddedSeqLayout())
        {
            pos_offset = std::accumulate(desc.GetLayoutVector().begin(),
                                         desc.GetLayoutVector().end(),
                                         pos_offset,
                                         [&lens, &pos](auto&& before, auto&& dim) {
                                             return (before * lens[dim]) + pos[dim];
                                         });
        }
        else
        {
            bool is_seq_part_begin            = true;
            size_t saved_offset_before_seqDim = 0;
            for(auto dim : desc.GetLayoutVector())
            {
                if(dim != 0 && dim != 1)
                    pos_offset = (pos_offset * lens[dim]) + pos[dim];
                else
                {
                    // each condition visited only once;
                    if(is_seq_part_begin) // seq block begin - major dim (seq or batch)
                    {
                        saved_offset_before_seqDim =
                            pos_offset * desc.GetTotalSequenceLen() + major_dim_part_sum[pos[dim]];
                        pos_offset = 0;
                    }
                    else // seq block end - minor dim (seq or batch)
                    {
                        const int major_dim_from_minor[] = {1, 0};
                        const auto major_dim_pos         = pos[major_dim_from_minor[dim]];

                        pos_offset *= major_dim_part_sum[major_dim_pos + 1] -
                                      major_dim_part_sum[major_dim_pos];

                        // pos_offset += pos[dim] works only if desc.IsSequenceLengthsSorted() ==
                        // true
                        pos_offset += pos[dim];

                        // add saved offset
                        pos_offset += saved_offset_before_seqDim;
                    }
                    is_seq_part_begin = false;
                }
            }
        }
        return pos_offset;
    }

private:
    std::vector<size_t> partSum(const miopen::SeqTensorDescriptor& sDesc) const
    {
        std::vector<size_t> sum_v{};

        if(!sDesc.IsPaddedSeqLayout())
        {
            const auto& seq_array = sDesc.GetSequenceLengthsVector();
            if(sDesc.GetLayoutVector()[0] == 0)
            {
                sum_v.resize(seq_array.size() + 1);
                sum_v[0] = 0;
                std::partial_sum(seq_array.begin(), seq_array.end(), std::next(sum_v.begin()));
            }
            else
            {
                if(!sDesc.IsSequenceLengthsSorted())
                    MIOPEN_THROW(miopenStatusBadParm,
                                 "Non-sorted SeqMajor tensor is not supported");

                sum_v.resize(seq_array[0] + 1);
                sum_v[0] = 0;
                {
                    std::vector<size_t> batch_per_seq(seq_array[0]);
                    auto batchs_begin   = batch_per_seq.begin();
                    size_t prew_seq_len = 0;
                    for(auto seq_cnt = seq_array.size(); seq_cnt != 0; seq_cnt--)
                    {
                        const auto cur_seq_len = seq_array[seq_cnt - 1];
                        const auto write_n     = cur_seq_len - prew_seq_len;
                        if(write_n > 0)
                        {
                            std::fill_n(batchs_begin, write_n, seq_cnt);
                            batchs_begin += write_n;
                        }
                        prew_seq_len = cur_seq_len;
                    }
                    std::partial_sum(
                        batch_per_seq.begin(), batch_per_seq.end(), std::next(sum_v.begin()));
                }
            }
        }

        return sum_v;
    }
    const std::vector<size_t> major_dim_part_sum;
    const miopen::SeqTensorDescriptor& desc;
};

template <typename T>
void TransformRNNIOLayaoutToTarget(const miopen::SeqTensorDescriptor& srcDesc,
                                   const miopen::SeqTensorDescriptor& dstDesc,
                                   const std::vector<int>& srcToDstSeqMaping,
                                   const std::vector<T>& srcData,
                                   std::vector<T>& dstData)
{
    const auto& maxDimLengths = srcDesc.GetLengths();

    assert(maxDimLengths.size() == 3 && srcDesc.GetLayoutVector()[2] == 2);
    assert(maxDimLengths == dstDesc.GetLengths());

    if(dstDesc.IsPaddingMarkerSpecified() && dstDesc.IsPaddedSeqLayout())
    {
        T paddingSymbol;

        memcpy(&paddingSymbol, dstDesc.GetPaddingMarkerHolder().data(), sizeof(T));

        std::fill(dstData.begin(), dstData.end(), paddingSymbol);
    }

    const auto& srcSeqLengths                  = srcDesc.GetSequenceLengthsVector();
    [[maybe_unused]] const auto& dstSeqLengths = dstDesc.GetSequenceLengthsVector();

    const size_t batch_size = maxDimLengths[0];
    const size_t copy_size  = maxDimLengths[2]; // IO vector size

    const SeqTensorOffsets src_offset_calc(srcDesc);
    const SeqTensorOffsets dst_offset_calc(dstDesc);

    for(size_t batch_it = 0; batch_it < batch_size; batch_it++)
    {
        const size_t src_batch_it = srcToDstSeqMaping[batch_it];
        const size_t dst_batch_it = batch_it;

        assert(dstSeqLengths[dst_batch_it] == srcSeqLengths[src_batch_it]);

        for(size_t seqTime_it = 0; seqTime_it < srcSeqLengths[src_batch_it]; seqTime_it++)
        {

            const size_t src_offset =
                src_offset_calc(std::array<size_t, 3>{src_batch_it, seqTime_it, 0});
            const size_t dst_offset =
                dst_offset_calc(std::array<size_t, 3>{dst_batch_it, seqTime_it, 0});

            std::copy(&srcData[src_offset], &srcData[src_offset + copy_size], &dstData[dst_offset]);
        }
    }
}

inline miopen::SeqTensorDescriptor
GetSeqDescriptorLayoutTransform(const miopen::SeqTensorDescriptor& desc,
                                miopenRNNBaseLayout_t transformLayout,
                                const std::vector<int>& transformSeqOrder)
{
    const auto [layout_dims_order, layout_seq_padding] =
        miopen::RNNDescriptor::convertRNNBaseLayout(transformLayout);

    const std::vector<size_t> transformed_seqLens =
        GetReorderedVector(desc.GetSequenceLengthsVector(), transformSeqOrder);

    return miopen::SeqTensorDescriptor{desc.GetType(),
                                       layout_dims_order,
                                       desc.GetLengths(),
                                       transformed_seqLens,
                                       desc.GetPaddingMarkerHolder(),
                                       true,
                                       layout_seq_padding};
}

template <class T>
struct seqTensor
{
    miopen::SeqTensorDescriptor desc;

    // private:
    std::vector<T> data;
    // public:

    size_t GetDataByteSize() const
    {
        return data.empty() ? desc.GetTensorRealByteSpace() : data.size() * sizeof(T);
    }

    size_t GetSize() const { return desc.GetTensorRealByteSpace() / sizeof(T); }

    std::vector<T>& GetDataPtr() const { return data.data(); }

    seqTensor<T> GetTensorLayoutTransform(miopenRNNBaseLayout_t transformLayout,
                                          std::vector<int>& transformSeqOrder) const
    {
        seqTensor<T> transformed_tensor(
            GetSeqDescriptorLayoutTransform(desc, transformLayout, transformSeqOrder));

        TransformRNNIOLayaoutToTarget(
            desc, transformed_tensor.desc, transformSeqOrder, data, transformed_tensor.data);

        return transformed_tensor;
    }

    // size_t GetNotPaddedDataCnt() { return desc.GetElementCount();}

    seqTensor(const miopen::SeqTensorDescriptor& tensor_desc)
        : desc(tensor_desc), data(desc.GetTensorRealByteSpace() / sizeof(T))
    {
    }

    template <class X>
    seqTensor(const std::vector<X>& dims)
        : desc(miopen_type<T>{}, dims), data(desc.GetTensorRealByteSpace() / sizeof(T))
    {
    }

    seqTensor(miopenDataType_t t,
              miopenTensorLayout_t layout,
              std::size_t batch,
              std::size_t seq,
              std::size_t vector_in)
        : seqTensor(t, layout, {batch, seq, vector_in})
    {
    }

    template <class X>
    seqTensor(miopenDataType_t t, miopenTensorLayout_t layout, const std::vector<X>& dims)
        : desc(t, layout, dims), data(desc.GetTensorRealByteSpace() / sizeof(T))
    {
    }
};
