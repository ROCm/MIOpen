/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include "pooling_common.hpp"

#define TEST_GET_INPUT_TENSOR 0

template <class T>
struct pooling2d_driver : pooling_driver<T>
{
    std::vector<std::vector<int>> get_2d_pooling_input_shapes()
    {
        return {{1, 19, 1024, 2048},
                {10, 3, 32, 32},
                {5, 32, 8, 8},
                {2, 1024, 12, 12},
                {4, 3, 231, 231},
                {8, 3, 227, 227},
                {1, 384, 13, 13},
                {1, 96, 27, 27},
                {2, 160, 7, 7},
                {1, 192, 256, 512},
                {2, 192, 28, 28},
                {1, 832, 64, 128},
                {1, 256, 56, 56},
                {4, 3, 224, 224},
                {2, 64, 112, 112},
                {2, 608, 4, 4},
                {1, 2048, 11, 11},
                {1, 16, 4096, 4096}};
    }

    pooling2d_driver() : pooling_driver<T>()
    {
#if TEST_GET_INPUT_TENSOR
        std::set<std::vector<int>> in_dim_set = get_inputs(this->batch_factor);
        std::vector<std::vector<int>> in_dim_vec(in_dim_set.begin(), in_dim_set.end());
        this->add(this->in_shape, "input", this->generate_data(in_dim_vec, {16, 32, 8, 8}));
#else
        this->add(
            this->in_shape, "input", this->generate_data_limited(get_2d_pooling_input_shapes(), 9));
#endif
        this->add(this->lens, "lens", this->generate_data({{2, 2}, {3, 3}}));
        this->add(this->strides, "strides", this->generate_data({{2, 2}, {1, 1}}));
        this->add(this->pads, "pads", this->generate_data({{0, 0}, {1, 1}}));
        this->add(this->wsidx, "wsidx", this->generate_data({0, 1}));
    }
};

int main(int argc, const char* argv[]) { test_drive<pooling2d_driver>(argc, argv); }
