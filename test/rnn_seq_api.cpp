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

#include "driver.hpp"

template <class T>
struct rnn_seq_driver : test_driver
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
    bool nohx  = false;
    bool nodhy = false;
    bool nocx  = false;
    bool nodcy = false;
    bool nohy  = false;
    bool nodhx = false;
    bool nocy  = false;
    bool nodcx = false;

    rnn_seq_driver()
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
    }

    int call_test_rnn_seq_api(Driver* drv, int argc, char* argv[])
    {
        drv->AddCmdLineArgs();
        int rc = drv->ParseCmdLineArgs(argc, argv);
        if(rc != 0)
        {
            std::cout << "ParseCmdLineArgs() FAILED, rc = " << rc << std::endl;
            return rc;
        }
        drv->GetandSetData();
        rc = drv->AllocateBuffersAndCopy();
        if(rc != 0)
        {
            std::cout << "AllocateBuffersAndCopy() FAILED, rc = " << rc << std::endl;
            return rc;
        }
        bool verifyarg    = (drv->GetInputFlags().GetValueInt("verify") == 1);
        int cumulative_rc = 0; // Do not stop running tests in case of errors.
        auto fargval      = drv->GetInputFlags().GetValueInt("forw");
        if(fargval & 1 || fargval == 0)
        {
            rc = drv->RunForwardGPU();
            cumulative_rc |= rc;
            if(rc != 0)
                std::cout << "RunForwardGPU() FAILED, rc = "
                          << "0x" << std::hex << rc << std::dec << std::endl;
            if(verifyarg) // Verify even if Run() failed.
                cumulative_rc |= drv->VerifyForward();
        }
        if(fargval != 1)
        {
            rc = drv->RunBackwardGPU();
            cumulative_rc |= rc;
            if(rc != 0)
                std::cout << "RunBackwardGPU() FAILED, rc = "
                          << "0x" << std::hex << rc << std::dec << std::endl;
            if(verifyarg) // Verify even if Run() failed.
                cumulative_rc |= drv->VerifyBackward();
        }
        return cumulative_rc;
    }

    bool is_correct_params() {
        if(!seqLenArray.empty())
        {
            if(*std::max_element(seqLenArray.begin(), seqLenArray.end()) != seqLength ||
               (io_layout == 1 &&
                !std::is_sorted(seqLenArray.begin(), seqLenArray.end(), std::greater<int>())) ||
               seqLenArray.size() != batchSize)
                return false;
        }
        if(inputMode == 1 && hiddenSize != inVecLen)
            return false;
        
        return true;

    }

    std::vector<std::string> prepare_strings_for_api_test()
    {
        std::vector<std::string> params;
        params.push_back("MIOpenDriver ");
        switch(this->type)
        {
        case miopenHalf: params.push_back("rnn_seqfp16"); break;
        case miopenFloat: params.push_back("rnn_seq"); break;
        case miopenDouble:
        case miopenDataType_t::miopenBFloat16:
        case miopenDataType_t::miopenInt32:
        case miopenDataType_t::miopenInt8:
        case miopenDataType_t::miopenInt8x4:
        default: return {};
        }

        switch(this->rnnMode)
        {
        case 0: params.push_back("--mode relu"); break;
        case 1: params.push_back("--mode tanh"); break;
        case 2: params.push_back("--mode lstm"); break;
        case 3: params.push_back("--mode gru"); break;
        default: return {};
        }

        params.push_back("--num_layer");
        params.push_back(std::to_string(numLayers));
        params.push_back("--io_layout");
        params.push_back(std::to_string(io_layout));
        params.push_back("--batch_size");
        params.push_back(std::to_string(batchSize));
        params.push_back("--seq_len");
        params.push_back(std::to_string(seqLength));

        if(!seqLenArray.empty())
        {
            params.push_back("--seq_len_array");
            params.push_back(std::accumulate(
                std::next(seqLenArray.begin()),
                seqLenArray.end(),
                std::to_string(seqLenArray[0]),
                [](std::string a, int b) { return std::move(a) + ',' + std::to_string(b); }));
        }


        params.push_back("--hid_h");
        params.push_back(std::to_string(hiddenSize));
        params.push_back("--in_vec");
        params.push_back(std::to_string(inVecLen));
        params.push_back("--bidirection");
        params.push_back(std::to_string(dirMode));
        params.push_back("--bias");
        params.push_back(std::to_string(biasMode));
        params.push_back("--inputmode");
        params.push_back(std::to_string(inputMode));
        params.push_back("--rnnalgo");
        params.push_back(std::to_string(algoMode));
        params.push_back("--use_dropout");
        params.push_back(std::to_string(useDropout));
        if(useDropout)
        {
            params.push_back("--dropout");
            params.push_back("0.1");
        }
        if(no_validate)
        {
            params.push_back("--verify");
            params.push_back("0");
        }

        return params;
    }
    std::vector<char*> convert_to_argv(std::vector<std::string>& strs)
    {
        std::vector<char*> result;

        result.reserve(strs.size());

        std::transform(begin(strs), end(strs), std::back_inserter(result), [](std::string& s) {
            return s.data();
        });
        result.push_back(nullptr);
        return result;
    }

    void print_cmd(std::vector<std::string>& params)
    {
        for(auto& param : params)
            std::cout << param << ' ';
        std::cout << std::endl;
    }

    void run()
    {

        
    }
};

int main(int argc, const char* argv[]) { test_drive<rnn_seq_driver>(argc, argv); }
