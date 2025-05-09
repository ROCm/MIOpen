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
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <numeric>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

#include <miopen/config.h>
#include <miopen/conv_algo_name.hpp>
#include <miopen/db_record.hpp>
#include <miopen/logger.hpp>

// Keep compatibility with old format for reading existing system find-db files.
// To be removed when the system find-db files regenerated in the 2.0 format.
#define WORKAROUND_ISSUE_1987 1

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

#if WORKAROUND_ISSUE_1987
/// Transform find-db (v.1.0) ID:VALUES to the current format.
/// Implementation is intentionally straightforward.
/// Do not include the 1st value from VALUES (solver name) into transformed VALUES.
/// Ignore FdbKCache_Key pair (last two values).
/// Append id (algorithm) to VALUES.
/// Use solver name as ID.
static bool TransformFindDbItem10to20(std::string& id, std::string& values)
{
    MIOPEN_LOG_T("Legacy find-db item: " << id << ':' << values);
    std::size_t pos = values.find(',');
    if(pos == std::string::npos)
        return false;
    const auto solver = values.substr(0, pos);

    const auto time_workspace_pos = pos + 1;
    pos                           = values.find(',', time_workspace_pos);
    if(pos == std::string::npos)
        return false;
    pos = values.find(',', pos + 1);
    if(pos == std::string::npos)
        return false;
    const auto time_workspace = values.substr(time_workspace_pos, pos - time_workspace_pos);

    values = time_workspace + ',' + id;
    id     = solver;
    MIOPEN_LOG_T("Transformed find-db item: " << id << ':' << values);
    return true;
}
#endif

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

        auto id     = id_and_values.substr(0, id_size);
        auto values = id_and_values.substr(id_size + 1);
        boost::trim(values);

#if WORKAROUND_ISSUE_1987
        // Detect legacy find-db item (v.1.0 ID:VALUES) and transform it to the current format.
        // For now, *only* legacy find-db record use convolution algorithm as ID, so if ID is
        // a valid algorithm, then we can safely assume that the item is in legacy format.
        if(IsValidConvolutionDirAlgo(id))
        {
            if(!TransformFindDbItem10to20(id, values))
            {
                MIOPEN_LOG_E("Ill-formed legacy find-db item: " << values);
                continue;
            }
        }
#endif

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
