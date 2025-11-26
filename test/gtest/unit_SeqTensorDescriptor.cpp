/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>
#include <miopen/logger.hpp>
#include <miopen/seq_tensor.hpp>

namespace {

struct SeqTensorDescriptorParams
{
    SeqTensorDescriptorParams(miopenDataType_t datatype_in, std::vector<size_t>&& lens_in)
        : datatype(datatype_in), lens(std::move(lens_in))
    {
    }

    SeqTensorDescriptorParams(miopenDataType_t datatype_in,
                              std::vector<size_t>&& lens_in,
                              bool with_padded_seq_layout)
        : datatype(datatype_in), lens(std::move(lens_in)), padded_seq_layout(with_padded_seq_layout)
    {
    }

    miopen::SeqTensorDescriptor GetSeqTensorDescriptor() const
    {
        std::vector<unsigned int> layout_default(lens.size());
        std::iota(layout_default.begin(), layout_default.end(), 0);
        return {datatype, layout_default, lens, padded_seq_layout};
    }

private:
    miopenDataType_t datatype;
    std::vector<size_t> lens;
    bool padded_seq_layout;
};

struct TestCase
{
    SeqTensorDescriptorParams tp;
    size_t max_sequence_length;
    size_t total_sequence_length;
    size_t actual_count_of_sequence;
};

auto GetTestCases()
{
    return std::vector{
        // clang-format off
        TestCase{{miopenHalf, {2, 2, 2}}, 2, 4, 2},
        TestCase{{miopenHalf, {2, 2, 2}, true}, 2, 4, 2},
        TestCase{{miopenHalf, {2, 2, 2, 2}}, 2, 4, 2},
        TestCase{{miopenHalf, {2, 2, 2, 2}, true}, 2, 4, 2},

        TestCase{{miopenHalf, {2, 8, 2, 2}}, 8, 16, 2},
        TestCase{{miopenHalf, {2, 8, 2, 2}, true}, 8, 16, 2},
        TestCase{{miopenHalf, {2, 16, 8, 2, 2}}, 16, 32, 2},
        TestCase{{miopenHalf, {2, 16, 8, 2, 2}, true}, 16, 32, 2},
        // clang-format on
    };
}

struct TestTensorSequence : testing::TestWithParam<TestCase>
{
    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetSeqTensorDescriptor();
        ASSERT_EQ(td.GetTotalSequenceLen(), p.total_sequence_length);
        ASSERT_EQ(td.GetMaxSequenceLength(), p.max_sequence_length);
        ASSERT_EQ(td.GetMaxCountOfSequences(), p.actual_count_of_sequence);
    }
};

using CPU_TestTensorSequence_FP16 = TestTensorSequence;

} // anonymous namespace

TEST_P(CPU_TestTensorSequence_FP16, SeqTensorDescriptor) { RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full, CPU_TestTensorSequence_FP16, testing::ValuesIn(GetTestCases()));
