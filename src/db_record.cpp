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
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>

#include "miopen/errors.hpp"
#include "miopen/db_record.hpp"
#include "miopen/logger.hpp"

#define MIOPEN_LOG_E(...) MIOPEN_LOG(LoggingLevel::Error, __VA_ARGS__)
#define MIOPEN_LOG_W(...) MIOPEN_LOG(LoggingLevel::Warning, __VA_ARGS__)
#define MIOPEN_LOG_I(...) MIOPEN_LOG(LoggingLevel::Info, __VA_ARGS__)

namespace miopen {

bool DbRecord::ParseContents(const std::string& contents)
{
    std::istringstream ss(contents);
    std::string id_and_values;
    int found = 0;

    while(std::getline(ss, id_and_values, ';'))
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

bool DbRecord::StoreValues(const std::string& id, const std::string& values)
{
    // No need to update the file if values are the same:
    const auto it = map.find(id);
    if(it == map.end() || it->second != values)
    {
        MIOPEN_LOG_I("Record under key: " << key << ", content "
                                          << (it == map.end() ? "inserted" : "overwritten") << ": "
                                          << id << ":" << values);
        map[id] = values;
        return true;
    }
    MIOPEN_LOG_I("Record under key: " << key << ", content is the same, not saved:" << id << ":"
                                      << values);
    return false;
}

bool DbRecord::LoadValues(const std::string& id, std::string& values)
{
    const auto it = map.find(id);

    if(it == map.end())
        return false;

    values = it->second;
    MIOPEN_LOG_I("Read record " << key << "=" << id << ":" << values);
    return true;
}

static void Write(std::ostream& stream,
                  const std::string& key,
                  std::unordered_map<std::string, std::string>& map)
{
    stream << key << '=';

    const auto pairsJoiner = [](const std::string& sum,
                                const std::pair<std::string, std::string>& pair) {
        const auto pair_str = pair.first + ":" + pair.second;
        return sum.empty() ? pair_str : sum + ";" + pair_str;
    };

    stream << std::accumulate(map.begin(), map.end(), std::string(), pairsJoiner) << std::endl;
}

static void Copy(std::istream& from, std::ostream& to, std::streamoff count)
{
    constexpr auto buffer_size = 4 * 1024 * 1024;
    char buffer[buffer_size];
    auto left = count;

    while(left > 0 && !from.eof())
    {
        const auto to_read = std::min<std::streamoff>(left, buffer_size);
        from.read(buffer, to_read);
        const auto read = from.gcount();
        to.write(buffer, read);
        left -= read;
    }
}

bool DbRecord::Flush(const RecordPositions* const pos)
{
    assert(pos);
    if(pos->begin < 0 || pos->end < 0)
    {
        std::ofstream file(db_filename, std::ios::app);

        if(!file)
        {
            MIOPEN_LOG_E("File is unwritable.");
            return false;
        }

        (void)file.tellp();
        Write(file, key, map);
    }
    else
    {
        const auto temp_name = db_filename + ".temp";
        std::ifstream from(db_filename, std::ios::ate);

        if(!from)
        {
            MIOPEN_LOG_E("File is unreadable.");
            return false;
        }

        std::ofstream to(temp_name);

        if(!to)
        {
            MIOPEN_LOG_E("Temp file is unwritable.");
            return false;
        }

        const auto from_size = from.tellg();
        from.seekg(std::ios::beg);

        Copy(from, to, pos->begin);
        Write(to, key, map);
        from.seekg(pos->end);
        Copy(from, to, from_size - pos->end);

        from.close();
        to.close();

        std::remove(db_filename.c_str());
        std::rename(temp_name.c_str(), db_filename.c_str());
        /// \todo What if rename fails? Thou shalt not loose the original file.
    }
    return true;
}

void DbRecord::ReadFile(RecordPositions* const pos)
{
    if(pos)
    {
        pos->begin = -1;
        pos->end   = -1;
    }
    map.clear();

    MIOPEN_LOG_I("Looking for key: " << key);

    std::ifstream file(db_filename);

    if(!file)
    {
        MIOPEN_LOG_W("File is unreadable.");
        return;
    }

    int n_line = 0;
    while(true)
    {
        std::string line;
        std::streamoff line_begin = file.tellg();
        if(!std::getline(file, line))
            break;
        ++n_line;
        std::streamoff next_line_begin = file.tellg();

        const auto key_size = line.find('=');
        const bool is_key   = (key_size != std::string::npos && key_size != 0);
        if(!is_key)
        {
            if(!line.empty()) // Do not blame empty lines.
            {
                MIOPEN_LOG_E("Ill-formed record: key not found.");
                MIOPEN_LOG_E(db_filename << "#" << n_line);
            }
            continue;
        }
        const auto current_key = line.substr(0, key_size);

        if(current_key != key)
        {
            continue;
        }
        MIOPEN_LOG_I("Key match: " << current_key);
        const auto contents = line.substr(key_size + 1);

        if(contents.empty())
        {
            MIOPEN_LOG_E("None contents under the key: " << current_key);
            continue;
        }
        MIOPEN_LOG_I("Contents found: " << contents);

        const bool is_parse_ok = ParseContents(contents);

        if(!is_parse_ok)
        {
            MIOPEN_LOG_E("Error parsing payload under the key: " << current_key);
            MIOPEN_LOG_E(db_filename << "#" << n_line);
            MIOPEN_LOG_E(contents);
        }
        // A record with matching key have been found.
        if(pos)
        {
            pos->begin = line_begin;
            pos->end   = next_line_begin;
        }
        break;
    }
}
} // namespace miopen
