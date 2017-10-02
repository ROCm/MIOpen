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
    case LogType::Error:
    {
        static const auto err_str = "Error";
        return err_str;
    }
    case LogType::Warning:
    {
        static const auto warn_str = "Warning";
        return warn_str;
    }
    }
}

static void
Log(std::ostream& stream, LogType type, const std::string& source, const std::string& message)
{
    stream << "[\033[" << static_cast<int>(type) << "m" << GetLogTypeName(type) << "\033[0m]["
           << source << "] " << message << std::endl;
}

DataEntry::DataEntry(const std::string& path, const std::string& entry_key)
    : _path(path), _entry_key(entry_key)
{
}

DataEntry::~DataEntry() { Flush(); }

bool DataEntry::ParseEntry(const std::string& entry)
{
    std::istringstream ss(entry);
    std::string pair;
    auto found = 0u;

    while(std::getline(ss, pair, ';'))
    {
        const auto key_size = pair.find(':');

        if(key_size == std::string::npos || pair.size() < key_size + 2)
        {
            Log(std::cerr, LogType::Error, "Database", "Empty item.");
            continue;
        }

        const auto key   = pair.substr(0, key_size);
        const auto value = pair.substr(key_size + 1);

        if(_content.find(key) != _content.end())
        {
            Log(std::cerr, LogType::Error, "Database", "Duplicate keys.");
            continue;
        }

        _content.emplace(key, value);
        found++;
    }

    return found > 0;
}

bool DataEntry::Save(const std::string& key, const std::string& value)
{
    _has_changes  = true;
    _content[key] = value;

    return true;
}

bool DataEntry::Load(const std::string& key, std::string& value) const
{
    const auto it = _content.find(key);

    if(it == _content.end())
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
    if(!_has_changes)
        return;

    if(_start == -1)
    {
        std::ofstream file(_path, std::ios::app);

        if(!file)
        {
            Log(std::cerr, LogType::Error, "Database", "File is unwritable.");
            return;
        }

        _start = file.tellp();
        Write(file, _entry_key, _content);
        _has_changes = false;
        return;
    }

    const auto temp_name = _path + ".temp";
    std::ifstream from(_path, std::ios::ate);

    if(!from)
    {
        Log(std::cerr, LogType::Error, "Database", "File is unreadable.");
        return;
    }

    std::ofstream to(temp_name);

    if(!to)
    {
        Log(std::cerr, LogType::Error, "Database", "File is unwritable.");
        return;
    }

    const auto from_size = from.tellg();
    from.seekg(std::ios::beg);

    Copy(from, to, _start);
    Write(to, _entry_key, _content);
    Copy(from, to, from_size - from.tellg());

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

    if(!file)
    {
        Log(std::cerr, LogType::Warning, "Database", "File is unreadable.");
        return;
    }

    std::string line;
    std::streamoff start = 0;

    while(std::getline(file, line))
    {
        const auto key_size = line.find('=');

        if(key_size == std::string::npos)
            continue;

        if(line.size() < key_size + 2)
        {
            Log(std::cerr, LogType::Error, "Database", "Empty entry.");
            continue;
        }

        const auto key = line.substr(0, key_size);

        if(key != _entry_key)
            continue;

        const auto entry = line.substr(key_size + 1);

        if(ParseEntry(entry))
        {
            _start = start;
            break;
        }

        Log(std::cerr, LogType::Error, "Database", "Empty entry.");
        start = file.tellg();
    }

    _start = -1;
}
}
