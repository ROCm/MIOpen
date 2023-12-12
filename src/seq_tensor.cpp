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
#include <miopen/seq_tensor.hpp>
#include <miopen/tensor.hpp>

#include <miopen/errors.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>

namespace miopen {

namespace {

template <class T>
bool CheckLengths(const std::vector<T>& lens)
{
    if(lens.empty())
        return false;
    if(!std::all_of(lens.cbegin(), lens.cend(), [](T x) { return x > 0; }))
        return false;
    return true;
}

template <class T>
bool CheckSequenceLengths(const std::vector<T>& lens)
{
    if(lens.empty())
        return false;
    if(!std::all_of(lens.cbegin(), lens.cend(), [](T x) { return x >= 0; }))
        return false;
    return true;
}

std::vector<std::size_t> ConvertLengthsOrThrow(const std::vector<int>& lens_in,
                                               const std::string& err_msg,
                                               bool is_seq_len = false)
{
    if(!is_seq_len)
    {
        if(!CheckLengths(lens_in))
            MIOPEN_THROW(miopenStatusBadParm, err_msg);
    }
    else
    {
        if(!CheckSequenceLengths(lens_in))
            MIOPEN_THROW(miopenStatusBadParm, err_msg);
    }

    std::vector<std::size_t> lens(lens_in.cbegin(), lens_in.cend());
    return lens;
}

} // namespace

SeqTensorDescriptor::SeqTensorDescriptor() : packed(true) {}

// The delegation constructor should be placed above the target constructor in the
// code for better dependency tracking

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::initializer_list<int>& lens_in)
    : SeqTensorDescriptor(t, std::vector<int>(lens_in))
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t, const std::vector<int>& lens_in)
    : SeqTensorDescriptor(t, GetDefaultLayoutVector(lens_in.size()), lens_in, true)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::initializer_list<std::size_t>& lens_in)
    : SeqTensorDescriptor(t, std::vector<std::size_t>(lens_in))
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<std::size_t>& lens_in)
    : SeqTensorDescriptor(t, GetDefaultLayoutVector(lens_in.size()), lens_in, true)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<unsigned int>& layout_in,
                                         const std::vector<int>& lens_in,
                                         bool with_padded_seq_layout)
    : SeqTensorDescriptor(t,
                          layout_in,
                          ConvertLengthsOrThrow(lens_in, "Lengths must be > 0"),
                          with_padded_seq_layout)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<unsigned int>& layout_in,
                                         const std::initializer_list<std::size_t>& lens_in,
                                         bool with_padded_seq_layout)
    : SeqTensorDescriptor(t, layout_in, std::vector<std::size_t>(lens_in), with_padded_seq_layout)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<unsigned int>& layout_in,
                                         const std::vector<std::size_t>& lens_in,
                                         bool with_padded_seq_layout)
    : SeqTensorDescriptor(t, layout_in, lens_in, {}, with_padded_seq_layout)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<unsigned int>& layout_in,
                                         const std::vector<int>& lens_in,
                                         const std::vector<int>& seq_len,
                                         const std::vector<char>& padding_marker_in,
                                         bool use_seq_len,
                                         bool with_padded_seq_layout)
    : SeqTensorDescriptor(t,
                          layout_in,
                          ConvertLengthsOrThrow(lens_in, "Lengths must be > 0"),
                          ConvertLengthsOrThrow(seq_len, "SequenceLengths must be >= 0", true),
                          {},
                          padding_marker_in,
                          use_seq_len,
                          with_padded_seq_layout)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<unsigned int>& layout_in,
                                         const std::vector<std::size_t>& lens_in,
                                         const std::vector<std::size_t>& seq_len,
                                         const std::vector<char>& padding_marker_in,
                                         bool use_seq_len,
                                         bool with_padded_seq_layout)
    : SeqTensorDescriptor(t,
                          layout_in,
                          lens_in,
                          seq_len,
                          {},
                          padding_marker_in,
                          use_seq_len,
                          with_padded_seq_layout)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<unsigned int>& layout_in,
                                         const std::vector<std::size_t>& lens_in,
                                         const std::vector<std::size_t>& padding_in,
                                         bool with_padded_seq_layout)
    : SeqTensorDescriptor(t, layout_in, lens_in, {}, padding_in, {}, false, with_padded_seq_layout)
{
}

SeqTensorDescriptor::SeqTensorDescriptor(miopenDataType_t t,
                                         const std::vector<unsigned int>& layout_in,
                                         const std::vector<std::size_t>& lens_in,
                                         const std::vector<std::size_t>& seq_len,
                                         const std::vector<std::size_t>& padding_in,
                                         const std::vector<char>& padding_marker_in,
                                         bool use_seq_len,
                                         bool with_padded_seq_layout)
    : lens(lens_in),
      padding_marker(padding_marker_in),
      padded_seq_layout(with_padded_seq_layout),
      type(t)
{
    if(lens_in.size() <= 2)
        MIOPEN_THROW(miopenStatusBadParm, "Number of dimensions must be > 2");
    if(!CheckLengths(lens_in))
        MIOPEN_THROW(miopenStatusBadParm, "Lengths must be > 0");

    auto dims = lens_in.size();

    SetDimOrder(layout_in);

    if(padding_in.empty())
    {
        padds = std::vector<std::size_t>(dims, 0);
    }
    else
    {
        if(padding_in.size() != dims)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Lengths and padding number dimensions must be equal");
        }
        else
        {
            padds = padding_in;
        }
    }

    SetSequenceLen(seq_len, use_seq_len);

    UpdatePackedFlag();
}

void SeqTensorDescriptor::SetDimOrder(const std::vector<unsigned int>& dims_order_in)
{
    if(dims_order_in.size() != lens.size())
        MIOPEN_THROW(miopenStatusBadParm, "Lengths and layout number dimensions must be equal");
    dim_order = dims_order_in;
}

const std::vector<unsigned int>& SeqTensorDescriptor::GetLayoutVector() const { return dim_order; };

const std::vector<std::size_t>& SeqTensorDescriptor::GetLengths() const { return lens; }

const std::vector<std::size_t>& SeqTensorDescriptor::GetPadding() const { return padds; }

const std::vector<std::size_t>& SeqTensorDescriptor::GetSequenceLengthsVector() const
{
    return sequence_len;
}

const std::vector<char>& SeqTensorDescriptor::GetPaddingMarkerHolder() const
{
    return padding_marker;
}

std::vector<std::size_t> SeqTensorDescriptor::GetPaddedStrides() const
{
    std::vector<std::size_t> byte_strides(lens.size());
    byte_strides.back() = 1 + padds.back();

    for(size_t i = byte_strides.size() - 1; i > 0; i--)
    {
        auto index_prev     = dim_order[i];
        auto index          = dim_order[i - 1];
        byte_strides[index] = byte_strides[index_prev] * lens[index_prev] + padds[index];
    }

    return byte_strides;
}

bool SeqTensorDescriptor::IsPacked() const { return packed; }
bool SeqTensorDescriptor::IsPaddedSeqLayout() const { return padded_seq_layout; }
bool SeqTensorDescriptor::IsSequenceLengthsSorted() const { return samples_in_descending_order; }
miopenDataType_t SeqTensorDescriptor::GetType() const { return this->type; }

std::size_t SeqTensorDescriptor::GetMaxCountOfSequences() const { return lens[0]; }
std::size_t SeqTensorDescriptor::GetMaxSequenceLength() const { return lens[1]; }

std::size_t SeqTensorDescriptor::GetElementCount() const
{
    size_t acc = GetTotalSequenceLen();

    for(auto it = dim_order.rbegin(); it < dim_order.rend(); it++)
    {
        auto next_pos = *it;
        if(!(next_pos == 1 || next_pos == 0))
            acc *= lens[next_pos];
    }
    return acc;
}

std::size_t SeqTensorDescriptor::GetTensorRealByteSpace() const
{
    if(padded_seq_layout)
        return GetTensorMaxByteSpace();
    else
        return GetTensorRealByteSpaceSeqPacked();
}

std::size_t SeqTensorDescriptor::GetTensorRealByteSpaceSeqPacked() const
{
    size_t acc                  = GetTypeSize(this->type);
    size_t acc_padding_byte_tmp = 0;
    bool is_seq_part            = false;

    for(auto it = dim_order.rbegin(); it < dim_order.rend(); it++)
    {
        auto next_pos = *it;

        if(!(next_pos == 1 || next_pos == 0))
        {
            acc *= lens[next_pos];
            acc_padding_byte_tmp = acc_padding_byte_tmp * lens[next_pos] + padds[next_pos];
        }
        else
        {
            if(!is_seq_part) // seq begin
            {
                acc += acc_padding_byte_tmp;
                acc_padding_byte_tmp = 0;
            }
            else // seq end
            {
                acc_padding_byte_tmp *=
                    (next_pos == 0 ? GetNonEmptySeqCount() : GetLargestSeqLen());
                acc *= GetTotalSequenceLen();
            }
            is_seq_part = !is_seq_part;
        }
    }

    return acc + acc_padding_byte_tmp;
}

std::size_t SeqTensorDescriptor::GetTensorMaxByteSpace() const
{
    auto acc = GetTypeSize(this->type);

    for(auto it = dim_order.rbegin(); it < dim_order.rend(); it++)
    {
        auto next_pos = *it;
        acc           = lens[next_pos] * (acc + padds[next_pos]);
    }
    return acc;
}

bool SeqTensorDescriptor::operator==(const SeqTensorDescriptor& rhs) const
{
    assert(this->lens.size() == rhs.dim_order.size());
    return this->type == rhs.type && this->lens == rhs.lens && this->dim_order == rhs.dim_order;
}

bool SeqTensorDescriptor::operator!=(const SeqTensorDescriptor& rhs) const
{
    return !(*this == rhs);
}

std::size_t SeqTensorDescriptor::GetLargestSeqLen() const
{
    return all_sequences_equal_to_max ? lens[0]
                                      : *std::max_element(sequence_len.begin(), sequence_len.end());
}

std::size_t SeqTensorDescriptor::GetNonEmptySeqCount() const
{
    return (all_sequences_equal_to_max || padded_seq_layout) ? lens[0]
           : samples_in_descending_order
               ? std::distance(sequence_len.begin(),
                               std::lower_bound(sequence_len.begin(), sequence_len.end(), 0))
               : std::count_if(sequence_len.begin(), sequence_len.end(), [](auto const val) {
                     return val > 0;
                 });
}

std::size_t SeqTensorDescriptor::GetTotalSequenceLen() const
{
    return all_sequences_equal_to_max
               ? lens[0] * lens[1]
               : std::accumulate(sequence_len.begin(), sequence_len.end(), std::size_t{0});
}

void SeqTensorDescriptor::SetSequenceLen(const std::vector<std::size_t>& seq_lens)
{
    auto seq_cnt = lens[0];
    if(seq_lens.empty() || seq_cnt != seq_lens.size())
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Size of sequence_len and first dimension size must be equal");
    }

    sequence_len = seq_lens;

    samples_in_descending_order = std ::is_sorted(sequence_len.rbegin(), sequence_len.rend());

    all_sequences_equal_to_max =
        lens[1] == (samples_in_descending_order
                        ? *sequence_len.rbegin()
                        : *std::min_element(sequence_len.begin(), sequence_len.end()));
}

void SeqTensorDescriptor::SetSequenceLen(const std::vector<std::size_t>& seq_lens, bool use_seq_len)
{
    if(use_seq_len)
        return SetSequenceLen(seq_lens);

    sequence_len = std::vector<std::size_t>(lens[0], lens[1]);

    samples_in_descending_order = true;
    all_sequences_equal_to_max  = true;
}

bool SeqTensorDescriptor::IsZeroBytePadding() const
{
    return std::all_of(padds.cbegin(), padds.cend(), [](size_t x) { return x == 0; });
}

bool SeqTensorDescriptor::IsPaddingMarkerSpecified() const { return !padding_marker.empty(); }

void SeqTensorDescriptor::UpdatePackedFlag()
{
    packed = IsZeroBytePadding() && (all_sequences_equal_to_max || !padded_seq_layout);
}

//? move to RNN maybe
std::vector<size_t> SeqTensorDescriptor::GetBatchesPerSequence() const
{
    if(padded_seq_layout)
        MIOPEN_THROW(miopenStatusInternalError, "Only packed SeqTensorDescriptor supported.");

    std::vector<size_t> batches;
    if(all_sequences_equal_to_max)
    {
        batches = std::vector<size_t>(lens[1], lens[0]);
    }
    else
    {
        batches.reserve(sequence_len[0]);
        auto block_begin = sequence_len.rbegin();

        while(block_begin != sequence_len.rend() && *block_begin == 0)
            ++block_begin;

        if(block_begin != sequence_len.rend())
        {
            auto sample_ptr = block_begin;
            auto batch_size = sequence_len.rend() - block_begin;

            batches.insert(batches.end(), *block_begin, batch_size);

            while(sample_ptr != sequence_len.rend())
            {
                if(*sample_ptr != *block_begin)
                {
                    batch_size           = batch_size - (sample_ptr - block_begin);
                    const auto seq_count = *sample_ptr - *block_begin;
                    batches.insert(batches.end(), seq_count, batch_size);

                    block_begin = sample_ptr;
                }
                sample_ptr++;
            }
        }
    }
    return batches;
}

std::string SeqTensorDescriptor::ToString() const
{
    auto coma_fold = [](std::string a, const size_t b) {
        return std::move(a) + ", " + std::to_string(b);
    };

    std::string result = "{";

    if(!this->lens.empty())
    {
        result += "lens{" +
                  std::accumulate(std::next(this->lens.begin()),
                                  this->lens.end(),
                                  std::to_string(lens[0]),
                                  coma_fold) +
                  " }";
        result += ", layout{" +
                  std::accumulate(std::next(this->dim_order.begin()),
                                  this->dim_order.end(),
                                  std::to_string(dim_order[0]),
                                  coma_fold) +
                  " }";

        if(this->padded_seq_layout)
            result += ", padded_seq_layout";

        result += ", padding{" +
                  std::accumulate(std::next(this->padds.begin()),
                                  this->padds.end(),
                                  std::to_string(padds[0]),
                                  coma_fold) +
                  " }";

        if(!this->sequence_len.empty())
        {
            result += ", sequence_len[" +
                      std::accumulate(std::next(this->sequence_len.begin()),
                                      this->sequence_len.end(),
                                      std::to_string(sequence_len[0]),
                                      coma_fold) +
                      " ]";
        }

        if(this->packed)
            result += ", packed";
    }

    return result + "}";
}

std::ostream& operator<<(std::ostream& stream, const SeqTensorDescriptor& t)
{
    return stream << t.ToString();
}

void to_json(nlohmann::json& j, const SeqTensorDescriptor& descriptor)
{
    j = nlohmann::json{
        {"layout", descriptor.dim_order},
        {"padded_seq_layout", descriptor.padded_seq_layout},
        {"lengths", descriptor.lens},
        {"padds", descriptor.padds},
        {"seq_len", descriptor.sequence_len},
        {"all_sequences_equal_to_max", descriptor.all_sequences_equal_to_max},
        {"samples_in_descending_order", descriptor.samples_in_descending_order},
        {"packed", descriptor.packed},
        {"type", descriptor.type},
    };
}

void from_json(const nlohmann::json& j, SeqTensorDescriptor& descriptor)
{
    j.at("layout").get_to(descriptor.dim_order);
    j.at("padded_seq_layout").get_to(descriptor.padded_seq_layout);
    j.at("lengths").get_to(descriptor.lens);
    j.at("padds").get_to(descriptor.padds);
    j.at("seq_len").get_to(descriptor.sequence_len);
    j.at("all_sequences_equal_to_max").get_to(descriptor.all_sequences_equal_to_max);
    j.at("samples_in_descending_order").get_to(descriptor.samples_in_descending_order);
    j.at("packed").get_to(descriptor.packed);
    j.at("type").get_to(descriptor.type);
}

} // namespace miopen
