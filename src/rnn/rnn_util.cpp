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

#include <miopen/rnn.hpp>
#include <miopen/rnn_util.hpp>
#include <algorithm>

namespace miopen {

void RNNTensorPaddingConverter::ConvertTensorData(const Handle& handle,
                                                  const TensorDescriptor& padded_tensor_desc,
                                                  std::vector<int>& bsize_per_time,
                                                  ConstData_t src,
                                                  Data_t dst,
                                                  bool is_src_padded)
{

    const int seq_len = bsize_per_time.size();
    if(seq_len == 0)
        MIOPEN_THROW("Wrong seq_len size");

    auto max_batch_size = bsize_per_time[0];
    auto vector_size    = padded_tensor_desc.GetLengths()[1];

    const std::vector<size_t> padded_stride //= single_desc.GetStrides();
        {static_cast<size_t>(max_batch_size) * vector_size, static_cast<size_t>(vector_size), 1};

    unsigned int left_id    = 0;
    unsigned int src_offset = 0, dst_offset = 0;

    for(int i = 1; i <= seq_len; i++)
    {
        if(i == seq_len || bsize_per_time[left_id] != bsize_per_time[i])
        {
            auto copy_seq_cnt = i - left_id;
            auto copy_bsize   = bsize_per_time[left_id];

            const std::vector<size_t> copy_size{static_cast<size_t>(copy_seq_cnt),
                                                static_cast<size_t>(copy_bsize),
                                                static_cast<size_t>(vector_size)};

            auto packed_desc = miopen::TensorDescriptor(padded_tensor_desc.GetType(), copy_size);
            auto padded_desc =
                miopen::TensorDescriptor(padded_tensor_desc.GetType(), copy_size, padded_stride);

            // Result from GetElementSpace does not include the padding from the last sequence
            // WA: So calculated manually.
            unsigned int WA_padded_ElementSpace = padded_stride[0] * copy_size[0];

            if(is_src_padded)
            {
                CopyTensor(
                    handle, padded_desc, src, packed_desc, dst, src_offset, dst_offset, true);

                src_offset += WA_padded_ElementSpace;
                dst_offset += packed_desc.GetElementSpace();
            }
            else
            {
                CopyTensor(
                    handle, packed_desc, src, padded_desc, dst, src_offset, dst_offset, true);

                src_offset += packed_desc.GetElementSpace();
                dst_offset += WA_padded_ElementSpace;
            }
            left_id = i;
        }
    }
}

std::vector<size_t>
RNNTensorBaseLayoutConverter::GetSamplesDescendingOrder(const SeqTensorDescriptor& desc,
                                                        bool reverse)
{
    const auto sample_count = desc.GetMaxCountOfSequences();

    std::vector<size_t> index_v(sample_count);
    std::iota(index_v.begin(), index_v.end(), 0);

    auto& unsorted_seq_lens = desc.GetSequenceLengthsVector();

    auto seq_len_cmp = [&unsorted_seq_lens](unsigned a_id, unsigned b_id) {
        return unsorted_seq_lens[a_id] > unsorted_seq_lens[b_id];
    };

    std::stable_sort(index_v.begin(), index_v.end(), seq_len_cmp);

    auto get_reverse_index = [](const std::vector<size_t>& base_index) {
        std::vector<size_t> reverse_index(base_index.size());
        unsigned next_rev_index = 0;
        for(auto id : base_index)
            reverse_index[id] = next_rev_index++;
        return reverse_index;
    };

    return !reverse ? index_v : get_reverse_index(index_v);
}

void ReorderTensorGPUData(const Handle& handle,
                          const std::vector<size_t>& tensor_lens,
                          int reordering_dim,
                          const std::vector<size_t>& sample_order,
                          std::vector<size_t> src_stride,
                          std::vector<size_t> dst_stride,
                          ConstData_t src,
                          Data_t dst,
                          miopenDataType_t data_type)
{
    if(tensor_lens[reordering_dim] != sample_order.size())
        MIOPEN_THROW(miopenStatusInternalError, "Wrong tensor lens");

    auto get_single_samlpe_lens = [](const std::vector<size_t>& lens, int reordering_dim) {
        std::vector<size_t> new_lens = lens;
        new_lens[reordering_dim]     = 1;
        return new_lens;
    };

    const std::vector<size_t> copy_size = get_single_samlpe_lens(tensor_lens, reordering_dim);

    const auto src_desc = miopen::TensorDescriptor(data_type, copy_size, src_stride);
    const auto dst_desc = miopen::TensorDescriptor(data_type, copy_size, dst_stride);

    const auto src_sample_stride = src_stride[reordering_dim];
    const auto dst_sample_stride = dst_stride[reordering_dim];
    for(size_t i = 0; i < sample_order.size(); i++)
    {
        const auto dst_offset = i * dst_sample_stride;
        const auto src_offset = sample_order[i] * src_sample_stride;
        CopyTensor(handle, src_desc, src, dst_desc, dst, src_offset, dst_offset, true);
    }
}

void RNNTensorBaseLayoutConverter::ReorderInputTensorGPUData(
    const Handle& handle,
    const SeqTensorDescriptor& padded_tensor_desc,
    const std::vector<size_t>& sample_order,
    const SeqTensorDescriptor& dst_padded_tensor_desc,
    ConstData_t src,
    Data_t dst)
{
    if(!padded_tensor_desc.IsPaddedSeqLayout())
        MIOPEN_THROW(miopenStatusInternalError, "Wrong tensor layout");

    // auto get_single_samlpe_lens = [](const std::vector<size_t>& SeqTensor_lens) {
    //    std::vector<size_t> new_lens = SeqTensor_lens;
    //    new_lens[0]                  = 1;
    //    return new_lens;
    //};
    //
    // const std::vector<size_t> copy_size =
    // get_single_samlpe_lens(padded_tensor_desc.GetLengths());

    const std::vector<size_t> src_stride = padded_tensor_desc.GetPaddedStrides();
    const std::vector<size_t> dst_stride = dst_padded_tensor_desc.GetPaddedStrides();

    ReorderTensorGPUData(handle,
                         padded_tensor_desc.GetLengths(),
                         0,
                         sample_order,
                         src_stride,
                         dst_stride,
                         src,
                         dst,
                         padded_tensor_desc.GetType());

    // const auto src_desc =
    //    miopen::TensorDescriptor(padded_tensor_desc.GetType(), copy_size, src_stride);
    // const auto dst_desc =
    //    miopen::TensorDescriptor(padded_tensor_desc.GetType(), copy_size, dst_stride);
    //
    // const auto src_sample_stride = src_stride[0];
    // const auto dst_sample_stride = dst_stride[0];
    // for(size_t i = 0; i < sample_order.size(); i++)
    //{
    //    const auto dst_offset = i * dst_sample_stride;
    //    const auto src_offset = sample_order[i] * src_sample_stride;
    //    CopyTensor(handle, src_desc, src, dst_desc, dst, src_offset, dst_offset, true);
    //}
}

void RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(const Handle& handle,
                                                              const TensorDescriptor& tensor_desc,
                                                              int reordering_dim,
                                                              std::vector<size_t> sample_order,
                                                              ConstData_t src,
                                                              Data_t dst)
{
    const auto& lens = tensor_desc.GetLengths();
    if(lens[reordering_dim] != sample_order.size())
        MIOPEN_THROW(miopenStatusInternalError, "Wrong tensor lens");

    const std::vector<size_t> src_stride = tensor_desc.GetStrides();
    const std::vector<size_t> dst_stride = tensor_desc.GetStrides();

    ReorderTensorGPUData(handle,
                         lens,
                         reordering_dim,
                         sample_order,
                         src_stride,
                         dst_stride,
                         src,
                         dst,
                         tensor_desc.GetType());
}

void RNNTensorBaseLayoutConverter::ChangeTensorGPUDataPadding(
    const Handle& handle, const SeqTensorDescriptor& tensor_desc, ConstData_t src, Data_t dst)
{
    if(!tensor_desc.IsSequenceLengthsSorted())
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Wrong tensor descriptor, only sorted tensors supported.");
    }

    if(!tensor_desc.IsZeroBytePadding())
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Wrong tensor descriptor, tensors with byte padding not supported.");
    }

    miopenRNNBaseLayout_t data_layout_t = RNNDescriptor::getBaseLayoutFromDataTensor(tensor_desc);

    if(data_layout_t == miopenRNNDataUnknownLayout)
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Wrong tensor descriptor, only Base Layouts supported.");
    }

    bool is_seq_major = data_layout_t == miopenRNNDataSeqMajorNotPadded ||
                        data_layout_t == miopenRNNDataSeqMajorPadded;
    bool is_src_padded = data_layout_t == miopenRNNDataSeqMajorPadded ||
                         data_layout_t == miopenRNNDataBatchMajorPadded;

    const std::vector<size_t>& seq_lens_per_sample = tensor_desc.GetSequenceLengthsVector();

    auto it     = seq_lens_per_sample.begin();
    auto it_end = seq_lens_per_sample.end();

    auto r_it     = seq_lens_per_sample.rbegin();
    auto r_it_end = seq_lens_per_sample.rend();

    const size_t vector_size                = tensor_desc.GetLengths()[2];
    const std::vector<size_t> padded_stride = tensor_desc.GetPaddedStrides();

    auto get_packed_stride = [](const std::vector<size_t>& copy_size,
                                const std::vector<unsigned>& dim_order) {
        std::vector<std::size_t> byte_strides(copy_size.size());
        byte_strides.back() = 1;

        for(size_t i = byte_strides.size() - 1; i > 0; i--)
        {
            auto index_prev     = dim_order[i];
            auto index          = dim_order[i - 1];
            byte_strides[index] = byte_strides[index_prev] * copy_size[index_prev];
        }
        return byte_strides;
    };

    auto get_box_size_batch_major = [](auto& sample_it,
                                       auto& sample_it_end) -> std::tuple<size_t, size_t> {
        auto start_pos      = sample_it;
        size_t box_seq_size = *start_pos, box_batch_size;
        while(sample_it != sample_it_end && *sample_it == box_seq_size)
            sample_it++;
        box_batch_size = std::distance(start_pos, sample_it);
        return {box_seq_size, box_batch_size};
    };

    auto get_box_size_seq_major =
        [](auto& sample_it, auto& sample_it_end, bool is_first) -> std::tuple<size_t, size_t> {
        size_t start_len      = *sample_it,
               box_seq_size   = is_first ? start_len : start_len - *(sample_it - 1),
               box_batch_size = std::distance(sample_it, sample_it_end);
        while(sample_it != sample_it_end && *sample_it == start_len)
            sample_it++;
        return {box_seq_size, box_batch_size};
    };

    size_t src_offset = 0, dst_offset = 0;
    bool first_iter = true;
    while(is_seq_major ? (r_it != r_it_end) : (it != it_end))
    {
        size_t copy_seq_cnt, copy_bsize;

        if(is_seq_major)
            std::tie(copy_seq_cnt, copy_bsize) = get_box_size_seq_major(r_it, r_it_end, first_iter);
        else
            std::tie(copy_seq_cnt, copy_bsize) = get_box_size_batch_major(it, it_end);

        const std::vector<size_t> copy_size{static_cast<size_t>(copy_bsize),
                                            static_cast<size_t>(copy_seq_cnt),
                                            static_cast<size_t>(vector_size)};

        const std::vector<size_t> packed_stride =
            get_packed_stride(copy_size, tensor_desc.GetLayoutVector());

        // Nothing to copy, avoiding error with zero lens in TensorDescriptor
        if(!std::all_of(copy_size.cbegin(), copy_size.cend(), [](size_t x) { return x > 0; }))
            continue;

        const auto packed_desc =
            miopen::TensorDescriptor(tensor_desc.GetType(), copy_size, packed_stride);
        const auto padded_desc =
            miopen::TensorDescriptor(tensor_desc.GetType(), copy_size, padded_stride);

        // Result from GetElementSpace does not include the padding from the last sequence
        // WA: So calculated manually.
        const size_t WA_padded_ElementSpace =
            is_seq_major ? padded_stride[1] * copy_size[1] : padded_stride[0] * copy_size[0];

        if(is_src_padded)
        {
            CopyTensor(handle, padded_desc, src, packed_desc, dst, src_offset, dst_offset, true);

            src_offset += WA_padded_ElementSpace;
            dst_offset += packed_desc.GetElementSpace();
        }
        else
        {
            CopyTensor(handle, packed_desc, src, padded_desc, dst, src_offset, dst_offset, true);

            src_offset += packed_desc.GetElementSpace();
            dst_offset += WA_padded_ElementSpace;
        }
        first_iter = false;
    }
}

void RNNTensorBaseLayoutConverter::ChangePaddedTensorGPUDataLayout(
    const Handle& handle,
    const SeqTensorDescriptor& src_padded_desc,
    ConstData_t src,
    const SeqTensorDescriptor& dst_padded_desc,
    Data_t dst)
{
    if(!src_padded_desc.IsPaddedSeqLayout() || !dst_padded_desc.IsPaddedSeqLayout())
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Wrong tensor descriptor, only padded tensors supported.");
    }

    const auto data_type = src_padded_desc.GetType();

    if(dst_padded_desc.GetType() != data_type)
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Wrong tensor descriptor, Dst data type should match src data type.");
    }

    const std::vector<size_t> copy_size = src_padded_desc.GetLengths();
    if(dst_padded_desc.GetLengths() != copy_size)
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Wrong tensor descriptor, Dst desc size should match Src desc size.");
    }

    const std::vector<size_t> src_stride = src_padded_desc.GetPaddedStrides();
    const std::vector<size_t> dst_stride = dst_padded_desc.GetPaddedStrides();

    auto src_desc = miopen::TensorDescriptor(data_type, copy_size, src_stride);
    auto dst_desc = miopen::TensorDescriptor(data_type, copy_size, dst_stride);

    CopyTensor(handle, src_desc, src, dst_desc, dst, 0, 0, true);
}

void RNNTensorBaseLayoutConverter::ConvertInputTensorGPUData(
    const Handle& handle,
    const SeqTensorDescriptor& src_tensor_desc,
    ConstData_t src,
    const SeqTensorDescriptor& dst_tensor_desc,
    Data_t dst,
    Data_t workspace,
    bool reverse = false)
{
    auto src_layout = RNNDescriptor::getBaseLayoutFromDataTensor(src_tensor_desc);
    auto dst_layout = RNNDescriptor::getBaseLayoutFromDataTensor(dst_tensor_desc);

    if(src_layout == miopenRNNDataBatchMajorPadded)
    {

        if(dst_layout != miopenRNNDataSeqMajorPadded &&
           dst_layout != miopenRNNDataSeqMajorNotPadded)
            MIOPEN_THROW(miopenStatusInternalError, "Wrong layout in Dst tensor.");

        Data_t SeqMajorPadded_ptr;
        SeqTensorDescriptor SeqMajorPadded_desc;

        if(dst_layout == miopenRNNDataSeqMajorPadded)
        {
            SeqMajorPadded_ptr  = dst;
            SeqMajorPadded_desc = dst_tensor_desc;
        }
        else
        {
            SeqMajorPadded_ptr  = workspace;
            SeqMajorPadded_desc = SeqTensorDescriptor(dst_tensor_desc.GetType(),
                                                      dst_tensor_desc.GetLayoutVector(),
                                                      dst_tensor_desc.GetLengths(),
                                                      src_tensor_desc.GetSequenceLengthsVector(),
                                                      {},
                                                      true,
                                                      true);
        }

        ChangePaddedTensorGPUDataLayout(
            handle, src_tensor_desc, src, SeqMajorPadded_desc, SeqMajorPadded_ptr);

        if(dst_layout != miopenRNNDataSeqMajorPadded)
        {
            ConvertInputTensorGPUData(
                handle,
                SeqMajorPadded_desc,
                SeqMajorPadded_ptr,
                dst_tensor_desc,
                dst,
                static_cast<void*>(reinterpret_cast<char*>(workspace) +
                                   SeqMajorPadded_desc.GetTensorMaxByteSpace()),
                reverse);
        }
    }
    else if(src_layout == miopenRNNDataSeqMajorPadded)
    {
        if(dst_layout == miopenRNNDataBatchMajorPadded)
        {
            ChangePaddedTensorGPUDataLayout(handle, src_tensor_desc, src, dst_tensor_desc, dst);
        }
        else if(dst_layout == miopenRNNDataSeqMajorNotPadded)
        {

            bool is_reordering_req = !src_tensor_desc.IsSequenceLengthsSorted();

            const SeqTensorDescriptor sorted_padded_tensor_desc(
                dst_tensor_desc.GetType(),
                dst_tensor_desc.GetLayoutVector(),
                dst_tensor_desc.GetLengths(),
                dst_tensor_desc.GetSequenceLengthsVector(),
                dst_tensor_desc.GetPadding(),
                {},
                true,
                true);

            if(is_reordering_req)
            {
                auto src_sortedOrder = GetSamplesDescendingOrder(src_tensor_desc);

                ReorderInputTensorGPUData(handle,
                                          src_tensor_desc,
                                          src_sortedOrder,
                                          sorted_padded_tensor_desc,
                                          src,
                                          workspace);
            }
            ConstData_t sorted_padded_tensor_data = is_reordering_req ? workspace : src;

            ChangeTensorGPUDataPadding(
                handle, sorted_padded_tensor_desc, sorted_padded_tensor_data, dst);
        }
        else
            MIOPEN_THROW(miopenStatusInternalError, "Unsupported layout.");
    }
    else if(src_layout == miopenRNNDataSeqMajorNotPadded)
    {
        if(dst_layout != miopenRNNDataSeqMajorPadded && dst_layout != miopenRNNDataBatchMajorPadded)
            MIOPEN_THROW(miopenStatusInternalError, "Unsupported layout.");

        bool is_reordering_req = reverse && !dst_tensor_desc.IsSequenceLengthsSorted();

        const SeqTensorDescriptor reordered_padded_tensor_desc(
            src_tensor_desc.GetType(),
            src_tensor_desc.GetLayoutVector(),
            src_tensor_desc.GetLengths(),
            dst_tensor_desc.GetSequenceLengthsVector(),
            dst_tensor_desc.GetPaddingMarkerHolder(),
            true,
            true);

        const SeqTensorDescriptor padded_tensor_desc(src_tensor_desc.GetType(),
                                                     src_tensor_desc.GetLayoutVector(),
                                                     src_tensor_desc.GetLengths(),
                                                     src_tensor_desc.GetSequenceLengthsVector(),
                                                     src_tensor_desc.GetPadding(),
                                                     dst_tensor_desc.GetPaddingMarkerHolder(),
                                                     true,
                                                     true);

        Data_t reordered_tensor_data =
            (dst_layout != miopenRNNDataSeqMajorPadded) ? workspace : dst;
        Data_t padded_data =
            (!is_reordering_req && (dst_layout == miopenRNNDataSeqMajorPadded))
                ? dst
                : (is_reordering_req && (dst_layout != miopenRNNDataSeqMajorPadded)
                       ? static_cast<void*>(reinterpret_cast<char*>(workspace) +
                                            reordered_padded_tensor_desc.GetTensorMaxByteSpace())
                       : workspace);

        FillSeqTensorByPaddingMarker(handle, padded_tensor_desc, padded_data);

        ChangeTensorGPUDataPadding(handle, src_tensor_desc, src, padded_data);

        if(is_reordering_req)
        {
            auto src_changer_order = GetSamplesDescendingOrder(dst_tensor_desc, reverse);

            ReorderInputTensorGPUData(handle,
                                      padded_tensor_desc,
                                      src_changer_order,
                                      reordered_padded_tensor_desc,
                                      padded_data,
                                      reordered_tensor_data);
        }

        if(dst_layout == miopenRNNDataBatchMajorPadded)
        {
            ConvertInputTensorGPUData(
                handle,
                reordered_padded_tensor_desc,
                reordered_tensor_data,
                dst_tensor_desc,
                dst,
                static_cast<void*>(reinterpret_cast<char*>(workspace) +
                                   reordered_padded_tensor_desc.GetTensorMaxByteSpace()),
                reverse);
        }
    }
    else
        MIOPEN_THROW(miopenStatusInternalError, "Unsupported layout.");
}

void FillSeqTensorByPaddingMarker(const Handle& handle,
                                  const SeqTensorDescriptor& desc,
                                  Data_t data)
{
    if(desc.IsPaddingMarkerSpecified())
    {
        if(!desc.IsZeroBytePadding())
            MIOPEN_THROW("Wrong tensor descriptor, tensors with byte padding not supported");

        miopen::SetTensor(handle,
                          TensorDescriptor(desc.GetType(), desc.GetLengths()),
                          data,
                          desc.GetPaddingMarkerHolder().data(),
                          0);
    }
}

} // namespace miopen
