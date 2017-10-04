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
#include <fstream>
#include <iostream>
#include <sstream>

namespace miopen {

enum class LogType
{
    Error   = 31,
    Warning = 33,
};

const char* GetLogTypeName(LogType type)
{
    switch(type)
    {
    case LogType::Error: return "Error";
    case LogType::Warning: return "Warning";
    default: return "<Unknown importance>";
    }
}

/// \todo Better diags everywhere. Print db filename, line number, key, id etc.
/// \todo Reuse existing logging machinery.
static void
Log(std::ostream& stream, LogType type, const std::string& source, const std::string& message)
{
    stream << "[\033[" << static_cast<int>(type) << "m" << GetLogTypeName(type) << "\033[0m]["
           << source << "] " << message << std::endl;
}

// Record format:
// KEY=[ID:VALUE[;ID:VALUE]*]
//
// KEY - An identifer of a problem description.
// ID  - An identifer of a solution.
// VALUE - Represents a performance config of a solution.
//
// KEY, ID and VALUE shall be ascii strings.
// Any of ";:=" are not allowed.
// Formatting of VALUE is a solution-specific.
// Note: If VALUE is used to represent a set of values,
// then it is recommended to use "," as a separator.

bool DataEntry::ParseEntry(const std::string& entry)
{
    std::istringstream ss(entry);
    std::string id_and_value;
    auto found = 0u;

    while(std::getline(ss, id_and_value, ';'))
    {
        const auto id_size = id_and_value.find(':');

        if (id_size == std::string::npos || id_and_value.size() < id_size + 2)
        {
            Log(std::cerr, LogType::Error, "Database", "ID not found or empty VALUE. Ignored.");
            continue;
        }

        const auto id = id_and_value.substr(0, id_size);
        const auto value = id_and_value.substr(id_size + 1);

        if (_content.find(id) != _content.end())
        {
            Log(std::cerr, LogType::Error, "Database", "Duplicate IDs. Ignored.");
            continue;
        }

        _content.emplace(id, value);
        found++;
    }

    return found > 0;
}

bool DataEntry::Save(const std::string& key, const std::string& value)
{
    /// \todo FIXME fail if record have not been read (from persistent storage) yet.
    /// Duplicates of records may appear in the db otherwise.
    _has_changes  = true;
    _content[key] = value;

    return true;
}

bool DataEntry::Load(const std::string& key, std::string& value) const
{
    /// \todo FXIEM fail if record have not been read (from persistent storage) yet.
    const auto it = _content.find(key);

    if (it == _content.end())
        return false;

    value = it->second;
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

void DataEntry::Flush()
{
    if (!_has_changes)
        return;

    if (_record_begin == -1)
    {
        std::ofstream file(_path, std::ios::app);

        if (!file)
        {
            Log(std::cerr, LogType::Error, "Database", "File is unwritable.");
            return;
        }

        _record_begin = file.tellp();
        Write(file, _entry_key, _content);
        _has_changes = false;
        return;
    }

    const auto temp_name = _path + ".temp";
    std::ifstream from(_path, std::ios::ate);

    if (!from)
    {
        Log(std::cerr, LogType::Error, "Database", "File is unreadable.");
        return;
    }

    std::ofstream to(temp_name);

    if (!to)
    {
        Log(std::cerr, LogType::Error, "Database", "File is unwritable.");
        return;
    }

    const auto from_size = from.tellg();
    from.seekg(std::ios::beg);

    Copy(from, to, _record_begin);
    Write(to, _entry_key, _content);
    from.seekg(_record_end);
    Copy(from, to, from_size - _record_end);

    from.close();
    to.close();

    std::remove(_path.c_str());
    std::rename(temp_name.c_str(), _path.c_str());
}

void DataEntry::ReadFromDisk()
{
    _content.clear();
    _has_changes = false;
    _read        = true;

    std::ifstream file(_path);

    if (!file)
    {
        Log(std::cerr, LogType::Warning, "Database", "File is unreadable.");
        return;
    }

    std::string line;
    std::streamoff line_begin = 0;
    std::streamoff next_line_begin = -1;

    _record_begin = -1;
    _record_end = -1;
    while (true)
    {
        line_begin = file.tellg();
        if (!std::getline(file, line)) {
            _record_begin = -1;
            _record_end = -1;
            break;
        }
        next_line_begin = file.tellg();
        
        const auto key_size = line.find('=');
        if (key_size == std::string::npos || key_size == 0)
        {
            if (line.size() > 0) // Do not blame empty lines.
            {
                Log(std::cerr, LogType::Error, "Database", "None key found.");
            }
            continue;
        }

        const auto key = line.substr(0, key_size);
        if (key != _entry_key)
            continue;

        const auto key_payload = line.substr(key_size + 1);
        if (key_payload.size() < 1)
        {
            Log(std::cerr, LogType::Error, "Database", std::string("None payload under the key: ") + key);
            continue;
        }

        if (ParseEntry(key_payload)) {
            Log(std::cerr, LogType::Error, "Database", std::string("Error parsing payload under the key:") + key);
        }
        // A record with matching key have been found.
        _record_begin = line_begin;
        _record_end = next_line_begin;
        break;
    }
}
}
