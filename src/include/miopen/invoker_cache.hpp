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

#pragma once

#include <miopen/errors.hpp>
#include <miopen/invoker.hpp>

#include <boost/optional.hpp>

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace miopen {

class InvokerCache
{
public:
    // network_config, solver_id
    using Key = std::pair<std::string, std::string>;

    boost::optional<const Invoker&> operator[](const Key& key) const;
    // For find 1.0
    boost::optional<const Invoker&> GetFound1_0(const std::string& network_config,
                                                const std::string& algorithm) const;
    void Register(const Key& key, const Invoker& invoker);
    // For find 1.0
    void SetAsFound1_0(const std::string& network_config,
                       const std::string& algorithm,
                       const std::string& solver_id);

private:
    struct Item
    {
        // algorithm -> solver_id
        // for find 1.0
        std::map<std::string, std::string> found_1_0;
        // solver_id -> invoker
        std::map<std::string, Invoker> invokers;
    };

    // network_config -> Item
    std::map<std::string, Item> invokers;
};

} // namespace miopen
