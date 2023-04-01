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

#ifndef MIOPEN_GUARD_MLOPEN_FINDDB_KERNEL_CACHE_KEY_HPP
#define MIOPEN_GUARD_MLOPEN_FINDDB_KERNEL_CACHE_KEY_HPP

#include <miopen/errors.hpp>

#include <string>

namespace miopen {

struct FindDbKCacheKey
{
    std::string algorithm_name = {};
    std::string network_config = {};

    FindDbKCacheKey() = default;

    FindDbKCacheKey(std::string algorithm_name_, std::string network_config_)
        : algorithm_name(algorithm_name_), network_config(network_config_)
    {
        if(!IsValid())
            MIOPEN_THROW("Invalid kernel cache key: " + algorithm_name + ", " + network_config);
    }

    bool IsValid() const { return !algorithm_name.empty() && !network_config.empty(); }
    bool IsUnused() const { return network_config == GetUnusedNetworkConfig(); }

    static FindDbKCacheKey MakeUnused(const std::string& algo_name)
    {
        return {algo_name, GetUnusedNetworkConfig()};
    }

private:
    static const char* GetUnusedNetworkConfig() { return "<unused>"; }
};

} // namespace miopen

#endif
