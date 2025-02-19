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

#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <miopen/errors.hpp>

namespace miopen {

template <typename T>
T mean(const std::vector<T>& data)
{
    static_assert(std::is_floating_point_v<T>);
    MIOPEN_THROW_IF(data.size() == 0, "Cannot find Mean of 0 length data");

    T sumOfValues = std::accumulate(data.begin(), data.end(), 0.0);
    return sumOfValues / data.size();
}

template <typename T>
T medianOfSortedData(const std::vector<T>& sortedData)
{
    static_assert(std::is_floating_point_v<T>);
    MIOPEN_THROW_IF(sortedData.size() == 0, "Cannot find Median of 0 length data");

    size_t size = sortedData.size();

    T median = (size % 2 == 0) ? (sortedData[size / 2 - 1] + sortedData[size / 2]) / 2.0
                               : sortedData[size / 2];

    return median;
}

template <typename T>
T median(std::vector<T>& data)
{
    static_assert(std::is_floating_point_v<T>);
    // Note: The data needs to be sorted for other parts of the algorthim
    std::sort(data.begin(), data.end());

    return medianOfSortedData(data);
}

template <typename T>
std::vector<T> medianAbsoluteDeviation(const std::vector<T>& sortedData)
{
    static_assert(std::is_floating_point_v<T>);
    T median = medianOfSortedData(sortedData);

    std::vector<T> absDeviation;
    absDeviation.reserve(sortedData.size());

    std::transform(sortedData.begin(),
                   sortedData.end(),
                   std::back_inserter(absDeviation),
                   [&](auto& value) { return std::abs(value - median); });

    return absDeviation;
}

template <typename T>
std::vector<T> modifiedZScores(const std::vector<T>& sortedData)
{
    static_assert(std::is_floating_point_v<T>);
    T medianValue = medianOfSortedData(sortedData);

    std::vector<T> absolute_deviation = medianAbsoluteDeviation(sortedData);
    T mad                             = median(absolute_deviation);

    // If MAD is 0, then we cannot calcualte the ModifiedZScore
    if(mad == T{0})
    {
        return std::vector<T>(sortedData.size(), 0);
    }
    else
    {
        std::vector<T> modZScores;
        modZScores.reserve(sortedData.size());

        std::transform(sortedData.begin(),
                       sortedData.end(),
                       std::back_inserter(modZScores),
                       [&](auto& value) { return 0.6745 * (value - medianValue) / mad; });

        return modZScores;
    }
}

template <typename T>
T removeHighOutliersAndGetMean(std::vector<T>& data, T z_threshold)
{
    static_assert(std::is_floating_point_v<T>);
    std::sort(data.begin(), data.end());

    std::vector<T> modZScores = modifiedZScores(data);
    std::vector<T> filteredData;

    for(size_t i = 0; i < data.size(); ++i)
    {
        if(modZScores[i] <= z_threshold)
        {
            filteredData.push_back(data[i]);
        }
    }

    return mean(filteredData);
}
} // namespace miopen
