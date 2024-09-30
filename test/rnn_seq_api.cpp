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

#include "rnn_seq_api.hpp"

template <class T>
struct rnn_seq_driver : rnn_seq_api_test_driver<T>
{
    rnn_seq_driver() : rnn_seq_api_test_driver<T>()
    {
        std::vector<int> modes(2, 0);
        modes[1] = 1;

        this->add(this->inVecLen, "vector-len", this->generate_data(std::vector<int>{1, 7}, 7));
        this->add(this->hiddenSize, "hidden-size", this->generate_data({7, 1, 13}, 13));
        this->add(this->useDropout, "use-dropout", this->generate_data({0, 1}));
        this->add(this->inputMode, "in-mode", this->generate_data(modes));
        this->add(this->biasMode, "bias-mode", this->generate_data({1}));
        this->add(this->dirMode, "dir-mode", this->generate_data(modes));
        this->add(this->rnnMode, "rnn-mode", this->generate_data({2, 1, 3}, 2));
        this->add(this->algoMode, "algo-mode", this->generate_data({0}));
        this->add(this->numLayers, "num-layers", this->generate_data({1, 3}, 3));
        this->add(this->io_layout, "io_layout", this->generate_data({2, 1, 3}, 3));
        this->add(this->batchSize, "batch-size", this->generate_data({1, 4, 6}, 6));
        this->add(this->seqLength, "seq-len", this->generate_data(std::vector<int>{1, 4, 15}, 15));
        this->add(this->seqLenArray,
                  "seqLen-batch",
                  this->generate_data({
                      {1, 15, 14, 15, 14, 1},
                      {1, 0, 3, 4, 2, 0},
                      {1, 2, 3, 4},
                      {4, 3, 2, 1},
                      {4, 4, 4, 4},
                      {1},
                  }));

        this->add(this->nohx, "nohx", this->generate_data({false}));
        this->add(this->nocx, "nocx", this->generate_data({false, true}));
        this->add(this->nohy, "nohy", this->generate_data({false}));
        this->add(this->nocy, "nocy", this->generate_data({false, true}));

        this->add(this->pytorchTensorDescriptorFormat, "pyDescFormat", this->generate_data(modes));
    }

    rnn_seq_driver(bool) : rnn_seq_api_test_driver<T>() {}
    bool is_skip_comb()
    {
        if(!this->seqLenArray.empty())
        {
            if(this->seqLenArray.size() != this->batchSize)
                return true;

            bool is_seqLength_is_max_seq =
                this->seqLength ==
                *std::max_element(this->seqLenArray.begin(), this->seqLenArray.end());

            if(!is_seqLength_is_max_seq)
                return true;
        }

        return false;
    }

    bool is_correct_params()
    {
        if(this->useDropout == 1 && (this->hiddenSize == 1 || this->batchSize == 1))
            return false;

        if(this->inputMode == 1 && this->hiddenSize != this->inVecLen)
            return false;

        if((this->rnnMode != 2) && (!this->nocx || !this->nocy))
            return false;

        if(this->seqLenArray.size() > this->batchSize)
            return false;

        if(this->biasMode && this->nohx)
            return false;

        if(!this->seqLenArray.empty())
        {
            if(this->seqLength <
               *std::max_element(this->seqLenArray.begin(), this->seqLenArray.end()))
                return false;

            if(this->io_layout == 1)
            {
                return std::is_sorted(
                    this->seqLenArray.begin(), this->seqLenArray.end(), std::greater<int>());
            }
        }
        return true;
    }

    void run()
    {

        if(!this->full_set || (is_correct_params() && !is_skip_comb()))
            rnn_seq_api_test_driver<T>::run();
        else
        {
            if(this->verbose)
                std::cout << "Incompatible argument combination, test skipped: "
                          << this->get_command_args() << std::endl;
        }
    }
};

template <class T>
struct lstm_MS_solver : rnn_seq_driver<T>
{
    lstm_MS_solver() : rnn_seq_driver<T>(true)
    {
        std::vector<int> modes(2, 0);
        modes[1] = 1;

        this->add(this->inVecLen, "vector-len", this->generate_data(std::vector<int>{1, 7}, 7));
        this->add(this->hiddenSize, "hidden-size", this->generate_data({13, 1}, 13));
        this->add(this->useDropout, "use-dropout", this->generate_data({0}));
        this->add(this->numLayers, "num-layers", this->generate_data({3}));
        this->add(this->inputMode, "in-mode", this->generate_data({0}));
        this->add(this->biasMode, "bias-mode", this->generate_data(modes));
        this->add(this->dirMode, "dir-mode", this->generate_data({0}));
        this->add(this->rnnMode, "rnn-mode", this->generate_data({2}));
        this->add(this->algoMode, "algo-mode", this->generate_data({0}));

        this->add(this->io_layout, "io_layout", this->generate_data({2}, 2));
        this->add(this->batchSize, "batch-size", this->generate_data({1, 6}, 6));
        this->add(this->seqLength, "seq-len", this->generate_data({38}));
        this->add(this->seqLenArray,
                  "seqLen-batch",
                  this->generate_data({
                      {34, 3, 2, 1},
                      {1, 15, 34, 15, 34, 1},
                      {},
                  }));

        this->add(this->nohx, "nohx", this->generate_data(modes));
        this->add(this->nocx, "nocx", this->generate_data(modes));
        this->add(this->nohy, "nohy", this->generate_data(modes));
        this->add(this->nocy, "nocy", this->generate_data(modes));

        this->add(this->pytorchTensorDescriptorFormat, "pyDescFormat", this->generate_data(modes));
    }

    void run()
    {
        // WA skip this test
        if(this->nohx && this->biasMode == 1)
            return;

        // WA for half verification at gfx90a.
        if(this->type == miopenHalf)
            if(this->nohy && this->nocy)
                return;

        // Optimization of test coverage.
        // Non-[float, Half] types are not used in this code-path and must be tested using another
        // subtest.
        if(this->type == miopenFloat || this->type == miopenHalf)
            rnn_seq_driver<T>::run();
    }
};

int main(int argc, const char* argv[])
{
    test_drive<rnn_seq_driver>(argc, argv);
    test_drive<lstm_MS_solver>(argc, argv);
}
