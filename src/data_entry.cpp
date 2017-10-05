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
#include "miopen/data_entry.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include "miopen/errors.hpp"

namespace miopen {

enum class LogType
{
    Error   = 31,
    Warning = 33,
    Info = 37,
};

inline const char* GetLogTypeName(LogType type)
{
    return (type == LogType::Warning) ? "Warning"
           : (type == LogType::Info) ? "Info"
           : "Error";
}

/// \todo Better diags everywhere. Print db filename, line number, key, id etc.
/// \todo Reuse existing logging machinery.
static void
Log(std::ostream& stream, LogType type, const std::string& source, const std::string& message)
{
    stream << "[\033[" << static_cast<int>(type) << "m" << GetLogTypeName(type) << "\033[0m]["
           << source << "] " << message << std::endl;
}

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
            Log(std::cerr,
                LogType::Error,
                "DbRecord::ParseContents",
                std::string("Ill-formed file: ID not found; skipped; key: ") + _key);
            continue;
        }

        const auto id     = id_and_values.substr(0, id_size);
        const auto values = id_and_values.substr(id_size + 1);

        if(_content.find(id) != _content.end())
        {
            Log(std::cerr,
                LogType::Error,
                "DbRecord::ParseContents",
                std::string("Duplicate ID (ignored): ") + id + "; key: " + _key);
            continue;
        }

        _content.emplace(id, values);
        ++found;
    }

    return (found > 0);
}

bool DbRecord::Save(const std::string& id, const std::string& values)
{
    if(!_is_content_cached)
    {
        // If there is a record with the same key, we need to find its position in the file.
        // Otherwise the new record with the same key wll be appended and db file become ill-formed.
        ReadIntoCache();
    }
    // No need to update the file if values are the same:
    const auto it = _content.find(id);
    if(it == _content.end() || it->second != values)
    {
        _is_cache_dirty = true;
        _content[id]    = values;
    }
    return true;
}

bool DbRecord::Load(const std::string& id, std::string& values)
{
    if(!_is_content_cached)
        ReadIntoCache();

    const auto it = _content.find(id);

    if(it == _content.end())
        return false;

    values = it->second;
    return true;
}

static void Write(std::ostream& stream,
                  const std::string& key,
                  std::unordered_map<std::string, std::string>& content)
{
    stream << key << '=';

    const auto pairsJoiner = [](const std::string& sum,
                                const std::pair<std::string, std::string>& pair) {
        const auto pair_str = pair.first + ":" + pair.second;
        return sum.empty() ? pair_str : sum + ";" + pair_str;
    };

    stream << std::accumulate(content.begin(), content.end(), std::string(), pairsJoiner)
           << std::endl;
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

std::atomic_int DbRecord::_n_cached_records(0);

void DbRecord::Flush()
{
    if(!_is_cache_dirty)
        return;
    if(_n_cached_records > 1)
    {
        Log(std::cerr,
            LogType::Error,
            "DbRecord::Flush",
            std::string("File update canceled to avoid db corruption. Key: ") + _key);
        return;
    }

    if(_pos_begin < 0 || _pos_end < 0)
    {
        std::ofstream file(_db_filename, std::ios::app);

        if(!file)
        {
            Log(std::cerr, LogType::Error, "DbRecord::Flush", "File is unwritable.");
            return;
        }

        _pos_begin = file.tellp();
        Write(file, _key, _content);
        _pos_end = file.tellp();
    }
    else
    {
        const auto temp_name = _db_filename + ".temp";
        std::ifstream from(_db_filename, std::ios::ate);

        if(!from)
        {
            Log(std::cerr, LogType::Error, "DbRecord::Flush", "File is unreadable.");
            return;
        }

        std::ofstream to(temp_name);

        if(!to)
        {
            Log(std::cerr, LogType::Error, "DbRecord::Flush", "Temp file is unwritable.");
            return;
        }

        const auto from_size = from.tellg();
        from.seekg(std::ios::beg);

        Copy(from, to, _pos_begin);
        Write(to, _key, _content);
        const auto new_end = to.tellp();
        from.seekg(_pos_end);
        Copy(from, to, from_size - _pos_end);

        from.close();
        to.close();

        std::remove(_db_filename.c_str());
        std::rename(temp_name.c_str(), _db_filename.c_str());
        /// \todo What if rename fails? Thou shalt not loose the original file.

        // After successful write, position of the record's end needs to be updated.
        // Position of the beginning remains the same.
        _pos_end = new_end;
    }
    _is_cache_dirty = false;
}

void DbRecord::ReadIntoCache()
{
    ++_n_cached_records;
    _content.clear();
    _is_cache_dirty = false;
    _is_content_cached =
        true; // This is true even if no record found in the db: nothing read <-> nothing cached.

    std::ifstream file(_db_filename);

    if(!file)
    {
        Log(std::cerr, LogType::Warning, "DbRecord::ReadIntoCache", "File is unreadable.");
        return;
    }

    _pos_begin = -1;
    _pos_end   = -1;
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
        if(key_size == std::string::npos || key_size == 0)
        {
            if(!line.empty()) // Do not blame empty lines.
            {
                Log(std::cerr, LogType::Error, "DbRecord::ReadIntoCache", "Ill-formed record: key not found.");
                Log(std::cerr, LogType::Info, "DbRecord::ReadIntoCache", _db_filename + ", line#: " + std::to_string(n_line));
            }
            continue;
        }

        const auto key = line.substr(0, key_size);
        if(key != _key)
            continue;

        const auto contents = line.substr(key_size + 1);
        if(contents.empty())
        {
            Log(std::cerr,
                LogType::Error,
                "DbRecord::ReadIntoCache",
                std::string("None payload under the key: ") + key);
            continue;
        }

        if(ParseContents(contents))
        {
            Log(std::cerr,
                LogType::Error,
                "DbRecord::ReadIntoCache",
                std::string("Error parsing payload under the key:") + key);
        }
        // A record with matching key have been found.
        _pos_begin = line_begin;
        _pos_end   = next_line_begin;
        break;
    }
}
} // namespace miopen
