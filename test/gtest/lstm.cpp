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

#include "lstm_common.hpp"
#include "get_handle.hpp"
#include <miopen/env.hpp>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_DEEPBENCH)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_LSTM_ENABLED)

using TestCase = std::string;
struct LSTMTest : public testing::TestWithParam<std::vector<TestCase>>
{
};

template <class T>
struct lstm_driver : lstm_basic_driver<T>
{
    lstm_driver() : lstm_basic_driver<T>()
    {
        std::vector<int> modes(2, 0);
        modes[1] = 1;
        std::vector<int> defaultBS(1);

        this->add(this->batchSize, "batch-size", this->generate_data(get_lstm_batchSize(), {17}));
        this->add(this->seqLength, "seq-len", this->generate_data(get_lstm_seq_len(), {2}));
        this->add(this->inVecLen, "vector-len", this->generate_data(get_lstm_vector_len()));
        this->add(this->hiddenSize, "hidden-size", this->generate_data(get_lstm_hidden_size()));
        this->add(this->numLayers, "num-layers", this->generate_data(get_lstm_num_layers()));
        this->add(this->nohx, "no-hx", this->flag());
        this->add(this->nodhy, "no-dhy", this->flag());
        this->add(this->nocx, "no-cx", this->flag());
        this->add(this->nodcy, "no-dcy", this->flag());
        this->add(this->nohy, "no-hy", this->flag());
        this->add(this->nodhx, "no-dhx", this->flag());
        this->add(this->nocy, "no-cy", this->flag());
        this->add(this->nodcx, "no-dcx", this->flag());
        this->add(this->flatBatchFill, "flat-batch-fill", this->flag());
        this->add(this->useDropout, "use-dropout", this->generate_data({0}));
        this->add(this->usePadding, "use-padding", this->generate_data({false, true}));

#if(MIO_LSTM_TEST_DEBUG == 3)
        this->biasMode  = 0;
        this->dirMode   = 0;
        this->inputMode = 0;
        this->algoMode  = 0;
#else
        this->add(this->inputMode, "in-mode", this->generate_data(modes));
        this->add(this->biasMode, "bias-mode", this->generate_data(modes));
        this->add(this->dirMode, "dir-mode", this->generate_data(modes));
        this->add(this->algoMode, "algo-mode", this->generate_data(modes));
#endif
        this->add(
            this->batchSeq,
            "batch-seq",
            this->lazy_generate_data(
                [=] { return generate_batchSeq(this->batchSize, this->seqLength); }, defaultBS));
    }
};

int RunLSTMDriver(std::string cmd)
{
    std::vector<std::string> ptrs;
    boost::split(ptrs, cmd, boost::is_any_of(" \t"), boost::token_compress_on);
    ptrs.insert(ptrs.begin(), "test_lstm");
    std::vector<const char*> char_ptrs;
    for(const auto& elem : ptrs)
    {
        char_ptrs.push_back(elem.c_str());
    }

#if(MIO_RNN_TIME_EVERYTHING > 0)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    test_drive<lstm_driver>(char_ptrs.size(), char_ptrs.data());

#if(MIO_RNN_TIME_EVERYTHING > 0)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: RNN test pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif

    auto capture = testing::internal::GetCapturedStderr();
    std::cout << capture;

    exit(0); // NOLINT (concurrency-mt-unsafe)
}

TEST_P(LSTMTest, test_lstm_deepbench_rnn)
{
    if(miopen::IsEnabled(MIOPEN_TEST_DEEPBENCH{}))
    {
        // clang-format off
        RunLSTMDriver("--verbose --batch-size 16 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 64 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 128 --seq-len 25 --vector-len 512 --hidden-size 512 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 16 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 64 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 128 --seq-len 25 --vector-len 1024 --hidden-size 1024 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 16 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 64 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 128 --seq-len 25 --vector-len 2048 --hidden-size 2048 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 16 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 64 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 128 --seq-len 25 --vector-len 4096 --hidden-size 4096 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 8 --seq-len 50 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 16 --seq-len 50 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 50 --vector-len 1536 --hidden-size 1536 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 16 --seq-len 150 --vector-len 256 --hidden-size 256 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 150 --vector-len 256 --hidden-size 256 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        RunLSTMDriver("--verbose --batch-size 64 --seq-len 150 --vector-len 256 --hidden-size 256 --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill");
        // clang-format on
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(LSTMTest, test_lstm_extra)
{
    if(miopen::IsEnabled(MIOPEN_TEST_ALL{}))
    {
        // clang-format off
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dhy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-dhy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-cx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dcy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cx --no-dcy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dhy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-dhy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-cx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dcy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cx --no-dcy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dhx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy --no-dhx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hy --no-cy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-dcx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-cy --no-dcx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dhx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy --no-dhx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hy --no-cy");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-dcx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-cy --no-dcx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 0 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx");
        RunLSTMDriver("--verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0 -dir-mode 1 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx");
        // clang-format on
    }
    else
    {
        GTEST_SKIP();
    }
}

std::vector<std::string> GetTestCases()
{
    std::vector<std::string> test_cases;
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(LSTM, LSTMTest, testing::Values(GetTestCases()));

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
