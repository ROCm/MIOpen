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

template <class T>
struct pooling3d_driver : pooling_driver<T>
{
    std::vector<std::vector<int>> get_3d_pooling_input_shapes()
    {
        return {{16, 64, 3, 4, 4},
                {16, 32, 4, 9, 9},
                {8, 512, 3, 14, 14},
                {8, 512, 4, 28, 28},
                {16, 64, 56, 56, 56},
                {4, 3, 4, 227, 227},
                {4, 4, 4, 161, 700}};
    }

    pooling3d_driver() : pooling_driver<T>()
    {
        this->add(
            this->in_shape, "input", this->generate_data_limited(get_3d_pooling_input_shapes(), 4));
        this->add(this->lens, "lens", this->generate_data({{2, 2, 2}, {3, 3, 3}}));
        this->add(this->strides, "strides", this->generate_data({{2, 2, 2}, {1, 1, 1}}));
        this->add(this->pads, "pads", this->generate_data({{0, 0, 0}, {1, 1, 1}}));
        this->add(this->wsidx, "wsidx", this->generate_data({1}));
    }
};

int main(int argc, const char* argv[]) { test_drive<pooling3d_driver>(argc, argv); }
