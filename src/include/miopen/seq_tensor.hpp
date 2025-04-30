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

#include <miopen/miopen.h>

#include <miopen/common.hpp>
#include <miopen/each_args.hpp>
#include <miopen/errors.hpp>
#include <miopen/functional.hpp>
#include <miopen/object.hpp>
#include <miopen/returns.hpp>

#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace miopen {

struct MIOPEN_INTERNALS_EXPORT SeqTensorDescriptor : miopenSeqTensorDescriptor
{
    SeqTensorDescriptor();

    // It is preferable to use constructors with lengths and strides with the std::size_t
    // data type, because in this format the data is stored inside the class

    // The delegation constructor should be placed above the target constructor in the
    // code for better dependency tracking

    SeqTensorDescriptor(miopenDataType_t t, const std::initializer_list<int>& lens_in);
    SeqTensorDescriptor(miopenDataType_t t, const std::vector<int>& lens_in);
    SeqTensorDescriptor(miopenDataType_t t, const std::initializer_list<std::size_t>& lens_in);
    SeqTensorDescriptor(miopenDataType_t t, const std::vector<std::size_t>& lens_in);

    SeqTensorDescriptor(miopenDataType_t t,
                        const std::vector<unsigned int>& layout_in,
                        const std::vector<int>& lens_in,
                        bool with_padded_seq_layout);
    SeqTensorDescriptor(miopenDataType_t t,
                        const std::vector<unsigned int>& layout_in,
                        const std::initializer_list<std::size_t>& lens_in,
                        bool with_padded_seq_layout);
    SeqTensorDescriptor(miopenDataType_t t,
                        const std::vector<unsigned int>& layout_in,
                        const std::vector<std::size_t>& lens_in,
                        bool with_padded_seq_layout);

    SeqTensorDescriptor(miopenDataType_t t,
                        const std::vector<unsigned int>& layout_in,
                        const std::vector<int>& lens_in,
                        const std::vector<int>& seq_len,
                        const std::vector<char>& padding_marker_in,
                        bool use_seq_len,
                        bool with_padded_seq_layout);
    SeqTensorDescriptor(miopenDataType_t t,
                        const std::vector<unsigned int>& layout_in,
                        const std::vector<std::size_t>& lens_in,
                        const std::vector<std::size_t>& seq_len,
                        const std::vector<char>& padding_marker_in,
                        bool use_seq_len,
                        bool with_padded_seq_layout);

    SeqTensorDescriptor(miopenDataType_t t,
                        const std::vector<unsigned int>& layout_in,
                        const std::vector<std::size_t>& lens_in,
                        const std::vector<std::size_t>& padding_in,
                        bool with_padded_seq_layout);

    SeqTensorDescriptor(miopenDataType_t t,
                        const std::vector<unsigned int>& layout_in,
                        const std::vector<std::size_t>& lens_in,
                        const std::vector<std::size_t>& seq_len,
                        const std::vector<std::size_t>& padding_in,
                        const std::vector<char>& padding_marker_in,
                        bool use_seq_len,
                        bool with_padded_seq_layout);

    const std::vector<unsigned int>& GetLayoutVector() const;
    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetPadding() const;
    const std::vector<std::size_t>& GetSequenceLengthsVector() const;
    const std::vector<char>& GetPaddingMarkerHolder() const;

    // Get vector of strides only for padded tensor,
    // if IsPaddedSeqLayout()==false function returns an empty vector
    std::vector<std::size_t> GetPaddedStrides() const;

    bool IsPacked() const;
    bool IsPaddedSeqLayout() const;
    bool IsSequenceLengthsSorted() const;
    bool IsZeroBytePadding() const;
    bool IsPaddingMarkerSpecified() const;

    miopenDataType_t GetType() const;

    std::size_t GetMaxCountOfSequences() const;
    std::size_t GetMaxSequenceLength() const;
    std::size_t GetTotalSequenceLen() const;

    // Calculate the number of tensor significant elements in pieces
    std::size_t GetElementCount() const;
    // Calculating the size of the occupied disk space in bytes
    std::size_t GetTensorRealByteSpace() const;
    // Calculating the maximal tensor size in bytes if all sequences are of the maximum size
    std::size_t GetTensorMaxByteSpace() const;

    bool operator==(const SeqTensorDescriptor& rhs) const;
    bool operator!=(const SeqTensorDescriptor& rhs) const;

    std::vector<size_t> GetBatchesPerSequence() const;

    std::string ToString() const;
    friend std::ostream& operator<<(std::ostream& stream, const SeqTensorDescriptor& t);

    friend void to_json(nlohmann::json& j, const SeqTensorDescriptor& descriptor);
    friend void from_json(const nlohmann::json& j, SeqTensorDescriptor& descriptor);

    void SetDimOrder(const std::vector<unsigned int>& dims_order);

private:
    std::size_t GetTensorRealByteSpaceSeqPacked() const;

    static std::vector<unsigned int> GetDefaultLayoutVector(int dims)
    {
        std::vector<unsigned int> layout_default(dims);
        std::iota(layout_default.begin(), layout_default.end(), 0);
        return layout_default;
    };

    std::size_t GetLargestSeqLen() const;

    std::size_t GetNonEmptySeqCount() const;

    void SetSequenceLen(const std::vector<std::size_t>& seq_lens);
    void SetSequenceLen(const std::vector<std::size_t>& seq_lens, bool use_seq_len);

    void UpdatePackedFlag();

    std::vector<unsigned int> dim_order;

    std::vector<std::size_t> lens;  // length of each dimension
    std::vector<std::size_t> padds; // padding for each dimension

    std::vector<std::size_t>
        sequence_len; // sequence length of each sample, sequence_len.size()=lens[0]

    std::vector<char> padding_marker;

    bool all_sequences_equal_to_max  = false;
    bool samples_in_descending_order = false;
    bool padded_seq_layout           = false;

    bool packed = true;

    miopenDataType_t type = miopenFloat;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenSeqTensorDescriptor, miopen::SeqTensorDescriptor)
