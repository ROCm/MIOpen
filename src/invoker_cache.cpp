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

#include <miopen/invoker_cache.hpp>
#include <miopen/logger.hpp>

namespace miopen {

boost::optional<const Invoker&> InvokerCache::operator[](const Key& key) const
{
    const auto item = invokers.find(key.first);
    if(item == invokers.end())
        return boost::none;
    const auto& item_invokers = item->second.invokers;
    const auto invoker        = item_invokers.find(key.second);
    if(invoker == item_invokers.end())
        return boost::none;
    return invoker->second;
}

boost::optional<const Invoker&> InvokerCache::GetFound1_0(const std::string& network_config,
                                                          const std::string& algorithm) const
{
    const auto item = invokers.find(network_config);
    if(item == invokers.end())
    {
        MIOPEN_LOG_I2("No invokers found for " << network_config);
        return boost::none;
    }
    if(item->second.found_1_0.empty())
    {
        MIOPEN_LOG_I2("Invokers found for " << network_config
                                            << " but there is no find 1.0 result.");
        return boost::none;
    }
    const auto& item_invokers = item->second.invokers;
    const auto& found_1_0_ids = item->second.found_1_0;
    const auto found_1_0_id   = found_1_0_ids.find(algorithm);
    if(found_1_0_id == found_1_0_ids.end())
    {
        MIOPEN_LOG_I2("Invokers found for "
                      << network_config << " but there is no one with an algorithm " << algorithm);
        return boost::none;
    }
    const auto invoker = item_invokers.find(found_1_0_id->second);
    if(invoker == item_invokers.end())
        MIOPEN_THROW("No invoker with solver_id of " + found_1_0_id->second +
                     " was registered for " + network_config);
    return invoker->second;
}

void InvokerCache::Register(const Key& key, const Invoker& invoker)
{
    auto it = invokers.find(key.first);
    if(it != invokers.end())
        it->second.invokers.insert({key.second, invoker});
    auto& item = invokers.insert({key.first, Item{}}).first->second;
    item.invokers.insert({key.second, invoker});
    MIOPEN_LOG_I2("Invoker registered for algorithm " << key.first << " and solver " << key.second);
}

void InvokerCache::SetAsFound1_0(const std::string& network_config,
                                 const std::string& algorithm,
                                 const std::string& solver_id)
{
    const auto item = invokers.find(network_config);
    if(item == invokers.end())
        MIOPEN_THROW("No invoker was registered for " + network_config);

    {
        // Validating at find time
        const auto& item_invokers = item->second.invokers;
        const auto invoker        = item_invokers.find(solver_id);
        if(invoker == item_invokers.end())
            MIOPEN_THROW("No invoker with solver_id of " + solver_id + " was registered for " +
                         network_config);
    }

    item->second.found_1_0[algorithm] = solver_id;
    MIOPEN_LOG_I2("Solver " << solver_id << " registered as find 1.0 best for " << algorithm
                            << " in " << network_config);
}

} // namespace miopen
