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
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <numeric>
#include <string>
#include <iterator>
#include <miopen/errors.hpp>

namespace miopen {

//static void printVector(std::vector<std::size_t> v) 
//{
//    for (int i = 0; i < v.size(); ++i){
//        std::cout << " " << v[i];
//    }
//    std::cout << std::endl;
//}

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

template <class Vector, class Op>
static inline std::vector<int64_t> sort_permutation(const Vector& data, Op op)
{
    std::vector<std::int64_t> result(data.size());
    std::iota(result.begin(), result.end(), 0);
    std::sort(result.begin(), result.end(), [&](auto x, auto y) { return op(data[x], data[y]); });
    return result;
}

static void construct_all_layouts(std::vector<std::string>& all_layouts,
                                  std::string tmp,
                                  const std::string& labels,
                                  std::vector<bool> is_recurrent)
{
    if(tmp.size() == labels.size())
    {
        // std::cout << tmp << std::endl;
        all_layouts.push_back(tmp);
        return;
    }

    for(int i = 0; i < labels.size(); ++i)
    {
        // Skip current label if it already exists
        if(find(tmp.begin(), tmp.end(), labels[i]) != tmp.end())
            continue;

        // Non-ambiguous letter should be in fixed position
        // Give up if the letter is in wrong index
        if(!is_recurrent[i])
        {
            if(tmp.size() != i)
                return;
        }

        tmp += labels[i];
        construct_all_layouts(all_layouts, tmp, labels, is_recurrent);
        tmp.pop_back();
    }
}

template <typename T>
static std::vector<bool> FindRecurrentStrides(const std::vector<T>& strides)
{
    std::unordered_map<T, std::size_t> counts;
    for(int i = 0; i < strides.size(); ++i)
    {
        ++counts[strides[i]];
    }
    std::vector<bool> is_recurrent(strides.size(), false);
    for(int i = 0; i < strides.size(); ++i)
    {
        if(counts[strides[i]] > 1)
        {
            is_recurrent[i] = true;
        }
    }
    return is_recurrent;
}

template <typename T>
static std::vector<std::string> compute_all_layouts(const std::vector<T>& strides,
                                             const std::string& labels)
{
    std::vector<std::string> all_layouts;
    if(labels.size() != strides.size())
    {
        return all_layouts;
    }

    // Copy construct the result string from labels. This allocates the space at one go
    // and is faster than calling push_back in transform.
    auto result          = labels;
    auto p               = sort_permutation(strides, std::greater<>{});
    std::string labels_p = labels;
    std::transform(p.begin(), p.end(), labels_p.begin(), [&](auto i) { return labels[i]; });

    std::vector<bool> is_recurrent = FindRecurrentStrides(strides);
    std::vector<bool> is_recurrent_p(labels.size(), false);
    std::transform(
        p.begin(), p.end(), is_recurrent_p.begin(), [&](auto i) { return is_recurrent[i]; });

    construct_all_layouts(all_layouts, "", labels_p, is_recurrent_p);
    return all_layouts;
}

// check if derived strides == original strides
template <typename T>
static bool is_valid_layout(const std::vector<T>& lens,
                     std::vector<T> strides,
                     std::string layout,
                     const std::string& labels)
{
    std::vector<T> derived_strides;
    tensor_layout_to_strides(lens, labels, layout, derived_strides);
    return derived_strides == strides;
}

template <typename T>
std::vector<std::string> compute_valid_layouts(const std::vector<T>& strides,
                                               const std::vector<T>& lens,
                                               const std::string& labels)
{
    std::vector<std::string> valid_layouts;
    std::vector<std::string> allLayouts = compute_all_layouts(strides, labels);
    for(int i = 0; i < allLayouts.size(); ++i)
    {
        std::cout << " " << allLayouts[i] << " "
                  << is_valid_layout(lens, strides, allLayouts[i], labels) << std::endl;
        if(is_valid_layout(lens, strides, allLayouts[i], labels))
        {
            valid_layouts.push_back(allLayouts[i]);
        }
    }
    return valid_layouts;
}

} // namespace miopen

#endif
