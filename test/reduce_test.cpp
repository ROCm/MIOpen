/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include "reduce_test.hpp"

int main(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);

    bool test_half   = false;
    bool test_double = false;

    test_half = std::any_of(
        as.begin(), as.end(), [](const std::string& elem) { return (elem == "--half"); });

    test_double = std::any_of(
        as.begin(), as.end(), [](const std::string& elem) { return (elem == "--double"); });

    if(test_half)
        test_drive<reduce_driver<half_float::half>>(argc, argv);
    else if(test_double)
        test_drive<reduce_driver<double>>(argc, argv);
    else
        test_drive<reduce_driver<float>>(argc, argv);
};
