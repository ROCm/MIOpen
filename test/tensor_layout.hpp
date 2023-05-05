/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_TENSOR_LAYOUT_HPP
#define GUARD_TENSOR_LAYOUT_HPP

#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include <iterator>
#include <miopen/errors.hpp>

template <typename T>
void tensor_layout_to_strides(const std::vector<T>& len,
                              const std::string& len_layout,
                              const std::string& layout,
                              std::vector<T>& strides)
{
    // Bind the layout and the dimension lengths together into a map.
    std::map<char, T> dim_to_len;
    std::transform(len.begin(),
                   len.end(),
                   len_layout.begin(),
                   std::inserter(dim_to_len, dim_to_len.end()),
                   [](T l, char dim) { return std::make_pair(dim, l); });

    // Now construct the strides according to layout by multiply the
    // dimension lengths together.
    std::transform(len_layout.begin(),
                   len_layout.end(),
                   std::back_inserter(strides),
                   [&layout, &dim_to_len](char cur_layout_char) {
                       auto pos = layout.find(cur_layout_char);
                       if(pos == std::string::npos)
                       {
                           MIOPEN_THROW(std::string("mismatched layout string, unexpect char: ")
                                            .append(1, cur_layout_char));
                       }
                       return std::accumulate(layout.begin() + pos + 1,
                                              layout.end(),
                                              1,
                                              [&dim_to_len](T accumulator, char l) {
                                                  return accumulator * dim_to_len[l];
                                              });
                   });
}

inline std::string tensor_layout_get_default(int size)
{
    if(size == 4)
        return "NCHW";
    if(size == 5)
        return "NCDHW";
    return "";
}

#endif
