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

#include <miopen/errors.hpp>
#include <miopen/db_record.hpp>
#include <miopen/logger.hpp>

namespace miopen {

bool DbRecord::SetValues(const std::string& id, const std::string& values)
{
    // No need to update the file if values are the same:
    const auto it = map.find(id);
    if(it == map.end() || it->second != values)
    {
        MIOPEN_LOG_I("Record under key: " << key << ", content "
                                          << (it == map.end() ? "inserted" : "overwritten") << ": "
                                          << id << ':' << values);
        map[id] = values;
        return true;
    }
    MIOPEN_LOG_I("Record under key: " << key << ", content is the same, not changed:" << id << ':'
                                      << values);
    return false;
}

bool DbRecord::GetValues(const std::string& id, std::string& values) const
{
    const auto it = map.find(id);

    if(it == map.end())
        return false;

    values = it->second;
    MIOPEN_LOG_I("Read record " << key << '=' << id << ':' << values);
    return true;
}

bool DbRecord::EraseValues(const std::string& id)
{
    const auto it = map.find(id);
    if(it != map.end())
    {
        MIOPEN_LOG_I("Record under key: " << key << ", removed: " << id << ':' << it->second);
        map.erase(it);
        return true;
    }
    MIOPEN_LOG_W("Record under key: " << key << ", not found: " << id);
    return false;
}

bool DbRecord::ParseContents(const std::string& contents)
{
    std::istringstream ss(contents);
    std::string id_and_values;
    int found = 0;

    map.clear();

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

void DbRecord::WriteContents(std::ostream& stream) const
{
    if(map.empty())
        return;

    stream << key << '=';

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

boost::optional<DbRecord> Db::FindRecord(const std::string& key, RecordPositions* pos) const
{
    if(pos)
    {
        pos->begin = -1;
        pos->end   = -1;
    }

    MIOPEN_LOG_I("Looking for key: " << key);

    exclusive_lock lock(mutex());

    std::ifstream file(filename);

    if(!file)
    {
        MIOPEN_LOG_W("File is unreadable.");
        return boost::none;
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
                MIOPEN_LOG_E(filename << "#" << n_line);
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

        DbRecord record(key);
        const bool is_parse_ok = record.ParseContents(contents);

        if(!is_parse_ok)
        {
            MIOPEN_LOG_E("Error parsing payload under the key: " << current_key);
            MIOPEN_LOG_E(filename << "#" << n_line);
            MIOPEN_LOG_E(contents);
        }
        // A record with matching key have been found.
        if(pos)
        {
            pos->begin = line_begin;
            pos->end   = next_line_begin;
        }
        return record;
    }
    // Record was not found
    return boost::none;
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

bool Db::Flush(const DbRecord& record, const RecordPositions* pos) const
{
    assert(pos);

    exclusive_lock lock(mutex());

    if(pos->begin < 0 || pos->end < 0)
    {
        std::ofstream file(filename, std::ios::app);

        if(!file)
        {
            MIOPEN_LOG_E("File is unwritable.");
            return false;
        }

        (void)file.tellp();
        record.WriteContents(file);
    }
    else
    {
        const auto temp_name = filename + ".temp";
        std::ifstream from(filename, std::ios::ate);

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
        record.WriteContents(to);
        from.seekg(pos->end);
        Copy(from, to, from_size - pos->end);

        from.close();
        to.close();

        std::remove(filename.c_str());
        std::rename(temp_name.c_str(), filename.c_str());
        /// \todo What if rename fails? Thou shalt not loose the original file.
    }
    return true;
}

bool Db::StoreRecord(const DbRecord& record) const
{
    MIOPEN_LOG_I("Storing record: " << record.key);
    exclusive_lock lock(mutex());
    RecordPositions pos;
    auto old_record = FindRecord(record.key, &pos);
    return Flush(record, &pos);
}

bool Db::UpdateRecord(DbRecord& record) const
{
    exclusive_lock lock(mutex());
    RecordPositions pos;
    auto old_record = FindRecord(record.key, &pos);
    DbRecord new_record(record);
    if(old_record)
    {
        new_record.Merge(*old_record);
        MIOPEN_LOG_I("Updating record: " << record.key);
    }
    else
    {
        MIOPEN_LOG_I("Storing record: " << record.key);
    }
    bool result = Flush(new_record, &pos);
    if(result)
        record = std::move(new_record);
    return result;
}

bool Db::RemoveRecord(const std::string& key) const
{
    // Create empty record with same key and replace original with that
    // This will remove record
    MIOPEN_LOG_I("Removing record: " << key);
    exclusive_lock lock(mutex());
    RecordPositions pos;
    FindRecord(key, &pos);
    DbRecord empty_record(key);
    return Flush(empty_record, &pos);
}
} // namespace miopen
