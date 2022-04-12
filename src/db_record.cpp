/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <numeric>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

#include <miopen/config.h>
#include <miopen/db_record.hpp>
#include <miopen/logger.hpp>

namespace miopen {

bool DbRecord::SetValues(const std::string& id, const std::string& values)
{
    constexpr auto log_level = MIOPEN_ENABLE_SQLITE ? LoggingLevel::Info2 : LoggingLevel::Info;

    // No need to update the file if values are the same:
    const auto it = map.find(id);
    if(it == map.end() || it->second != values)
    {
        MIOPEN_LOG(log_level,
                   key << ", content " << (it == map.end() ? "inserted" : "overwritten") << ": "
                       << id << ':' << values);
        map[id] = values;
        return true;
    }
    MIOPEN_LOG(log_level, key << ", content is the same, not changed:" << id << ':' << values);
    return false;
}

bool DbRecord::GetValues(const std::string& id, std::string& values) const
{
    const auto it = map.find(id);

    if(it == map.end())
    {
        MIOPEN_LOG_I(key << '=' << id << ':' << "<values not found>");
        return false;
    }

    values = it->second;
    MIOPEN_LOG_I(key << '=' << id << ':' << values);
    return true;
}

bool DbRecord::EraseValues(const std::string& id)
{
    const auto it = map.find(id);
    if(it != map.end())
    {
        MIOPEN_LOG_I(key << ", removed: " << id << ':' << it->second);
        map.erase(it);
        return true;
    }
    MIOPEN_LOG_W(key << ", not found: " << id);
    return false;
}

bool DbRecord::ParseContents(std::istream& contents)
{
    std::string id_and_values;
    int found = 0;

    map.clear();

    while(std::getline(contents, id_and_values, ';'))
    {
        const auto id_size = id_and_values.find(':');

        // Empty VALUES is ok, empty ID is not:
        if(id_size == std::string::npos)
        {
            MIOPEN_LOG_E("Ill-formed file: ID not found; skipped; key: " << key);
            continue;
        }

        const auto id     = id_and_values.substr(0, id_size);
        const auto values = id_and_values.substr(id_size + 1);

        if(map.find(id) != map.end())
        {
            MIOPEN_LOG_E("Duplicate ID (ignored): " << id << "; key: " << key);
            continue;
        }

        map.emplace(id, values);
        ++found;
    }

    return (found > 0);
}

void DbRecord::WriteContents(std::ostream& stream) const
{
    if(map.empty())
        return;

    stream << key << '=';
    WriteIdsAndValues(stream);
}

void DbRecord::WriteIdsAndValues(std::ostream& stream) const
{
    if(map.empty())
        return;

    const auto pairsJoiner = [](const std::string& sum,
                                const std::pair<std::string, std::string>& pair) {
        const auto pair_str = pair.first + ':' + pair.second;
        return sum.empty() ? pair_str : sum + ';' + pair_str;
    };

    stream << std::accumulate(map.begin(), map.end(), std::string(), pairsJoiner) << std::endl;
}

void DbRecord::Merge(const DbRecord& that)
{
    if(key != that.key)
        return;

    for(const auto& that_pair : that.map)
    {
        if(map.find(that_pair.first) != map.end())
            continue;
        map[that_pair.first] = that_pair.second;
    }
}
} // namespace miopen
