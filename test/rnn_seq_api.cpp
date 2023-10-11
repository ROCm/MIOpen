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

        std::vector<int> defaultBS{1, 7};

        this->add(this->inVecLen, "vector-len", this->generate_data(std::vector<int>{1, 17}, 17));
        this->add(this->hiddenSize, "hidden-size", this->generate_data({17, 1, 37}, 37));
        this->add(this->numLayers, "num-layers", this->generate_data({1, 7}, 7));
        this->add(this->useDropout, "use-dropout", this->generate_data({0, 1}));

        this->add(this->inputMode, "in-mode", this->generate_data(modes));
        this->add(this->biasMode, "bias-mode", this->generate_data({1}));
        this->add(this->dirMode, "dir-mode", this->generate_data(modes));
        this->add(this->rnnMode, "rnn-mode", this->generate_data({2, 1, 3}));
        this->add(this->algoMode, "algo-mode", this->generate_data({0}));

        this->add(this->io_layout, "io_layout", this->generate_data({2, 3}, 3));
        this->add(this->batchSize, "batch-size", this->generate_data({1, 4, 6}, 17));
        this->add(this->seqLength, "seq-len", this->generate_data(std::vector<int>{1, 4, 20}, 20));
        this->add(this->seqLenArray,
                  "seqLen-batch",
                  this->generate_data({
                      {1, 2, 3, 4},
                      {4, 3, 2, 1},
                      {1, 15, 20, 15, 20, 1},
                      {},
                  }));

        this->add(this->nohx, "nohx", this->generate_data({false}));
        this->add(this->nocx, "nocx", this->generate_data({false}));
        this->add(this->nohy, "nohy", this->generate_data({false}));
        this->add(this->nocy, "nocy", this->generate_data({false}));
    }


    bool is_correct_params() {
        if(!this->seqLenArray.empty())
        {
            if(*std::max_element(this->seqLenArray.begin(), this->seqLenArray.end()) !=
                   this->seqLength ||
               (this->io_layout == 1 && !std::is_sorted(this->seqLenArray.begin(),
                                                        this->seqLenArray.end(),
                                                        std::greater<int>())) ||
               this->seqLenArray.size() != this->batchSize)
                return false;
        }
        if(this->inputMode == 1 && this->hiddenSize != this->inVecLen)
            return false;
        
        return true;

    }


    void run() 
    { 
        if(!this->full_set || is_correct_params())
            rnn_seq_api_test_driver<T>::run();
        else {
            if(this->verbose)
                std::cout << "Incompatible argument combination, test skipped: "
                          << this->get_command_args() << std::endl;
        }
    }

};

int main(int argc, const char* argv[]) { test_drive<rnn_seq_driver>(argc, argv); }
