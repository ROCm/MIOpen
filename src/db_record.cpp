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

#include "miopen/errors.hpp"
#include "miopen/db_record.hpp"
#include "miopen/logger.hpp"

namespace miopen {

bool DbRecord::ParseContents(const std::string& contents)
{
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(record_format == RecordFormat::CurrentOrMixed || record_format == RecordFormat::Current);
#endif
    std::istringstream ss(contents);
    std::string id_and_values;
    int found = 0;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    bool is_legacy_content_found         = false;
    const bool is_legacy_content_allowed = (record_format == RecordFormat::CurrentOrMixed);
#endif

    while(std::getline(ss, id_and_values, ';'))
    {
        const auto id_size = id_and_values.find(':');

        // Empty VALUES is ok, empty ID is not:
        if(id_size == std::string::npos)
        {
            MIOPEN_LOG_E("Ill-formed file: ID not found; skipped; key: " << key);
            continue;
        }

        const auto id = id_and_values.substr(0, id_size);
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        if(is_legacy_content_allowed)
        {
            // Here we detect actual format of a record.
            if(id == MIOPEN_PERFDB_CONV_LEGACY_ID)
            {
                is_legacy_content_found = true;
                MIOPEN_LOG_I("Legacy content found under key: " << key << " Record is Mixed.");
            }
        }
#endif
        const auto values = id_and_values.substr(id_size + 1);

        if(content.find(id) != content.end())
        {
            MIOPEN_LOG_E("Duplicate ID (ignored): " << id << "; key: " << key);
            continue;
        }

        content.emplace(id, values);
        ++found;
    }

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    if(!is_legacy_content_found)
    {
        record_format = RecordFormat::Current;
    }
    else
    {
        record_format = RecordFormat::Mixed;
    }
#endif
    return (found > 0);
}

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
bool DbRecord::ParseLegacyContents(const std::string& contents)
{
    assert(record_format == RecordFormat::Legacy);
    const auto id = MIOPEN_PERFDB_CONV_LEGACY_ID;
    content.emplace(id, contents);
    return true;
}

static bool isLegacySolver(const std::string& id)
{
    /// \todo Hard-coded for now, quick and dirty.
    return (id == "ConvOclDirectFwd" || id == "ConvOclDirectFwd1x1" || id == "ConvOclDirectFwdC");
}
#endif

bool DbRecord::StoreValues(const std::string& id, const std::string& values)
{
    if(!is_content_cached)
    {
        // If there is a record with the same key, we need to find its position
        // in the db file. Otherwise the new record with the same key will NOT
        // replace the existing record and db file become ill-formed.
        ReadIntoCache();
    }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(record_format != RecordFormat::CurrentOrMixed);
    if((record_format == RecordFormat::Legacy || record_format == RecordFormat::Mixed) &&
       isLegacySolver(id))
    {
        // "__LEGACY__" id shall be replaced by actual legacy SolverId.
        // Values shall be replaced too (we do not want to cope with
        // comparison of legacy values with values in the current format).
        // That is, the whole map entry shall be replaced.
        // Record format becomes Current after that.
        const auto it = content.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
        assert(it != content.end());
        content.erase(it);
        content.emplace(id, values);
        is_cache_dirty = true;
        record_format  = RecordFormat::Current;
        MIOPEN_LOG_I("Legacy content under key: " << key << " replaced by " << id << ":" << values);
        return true;
    }
    else if(record_format == RecordFormat::Legacy && !isLegacySolver(id))
    {
        // Non-legacy SolverId cannot reside in the legacy record by definition.
        // Just add a content to the map and mark record as Mixed.
        assert(content.find(id) == content.end());
        content.emplace(id, values);
        is_cache_dirty = true;
        record_format  = RecordFormat::Mixed;
        MIOPEN_LOG_I("Legacy record under key: " << key << " appended by " << id << ":" << values
                                                 << " and becomes Mixed");
        return true;
    }
    assert((record_format == RecordFormat::Mixed && !isLegacySolver(id)) ||
           record_format == RecordFormat::Current);
#endif
    // No need to update the file if values are the same:
    const auto it = content.find(id);
    if(it == content.end() || it->second != values)
    {
        MIOPEN_LOG_I("Record under key: " << key << ", content "
                                          << (it == content.end() ? "inserted" : "overwritten")
                                          << ": "
                                          << id
                                          << ":"
                                          << values);
        content[id]    = values;
        is_cache_dirty = true;
    }
    else
    {
        MIOPEN_LOG_I("Record under key: " << key << ", content is the same, not saved:" << id << ":"
                                          << values);
    }
    return true;
}

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
bool DbRecord::LoadValues(const std::string& id,
                          std::string& values,
                          DbRecord::ContentFormat& content_format)
#else
bool DbRecord::LoadValues(const std::string& id, std::string& values)
#endif
{
    if(!is_content_cached)
        ReadIntoCache();

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(record_format != RecordFormat::CurrentOrMixed);
    if(record_format == RecordFormat::Legacy)
    {
        if(!isLegacySolver(id))
            return false;
        const auto it = content.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
        if(it == content.end())
            return false;

        values         = it->second;
        content_format = ContentFormat::Legacy;
        MIOPEN_LOG_I("Legacy record read: " << key << ":" << values << " for id: " << id);
        return true;
    }
    else if(record_format == RecordFormat::Mixed)
    {
        if(isLegacySolver(id))
        {
            // contents for legacy solvers may be in either legacy of current format.
            const auto it = content.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
            if(it != content.end())
            {
                values         = it->second;
                content_format = ContentFormat::Legacy;
                MIOPEN_LOG_I("Legacy content read from record (Mixed): " << key << ":" << id << ":"
                                                                         << values);
                return true;
            }
            // Legacy content not found.
            // Content shall be in current format.
            // Fall down.
        }
        // Solver is not legacy one.
        // Content shall be in current format.
        // Fall down.
    }
#endif
    const auto it = content.find(id);

    if(it == content.end())
        return false;

    values = it->second;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    content_format = ContentFormat::Current;
    MIOPEN_LOG_I(
        "Read record " << ((record_format == RecordFormat::Mixed) ? "(Mixed) " : "(Current) ")
                       << key
                       << ":"
                       << id
                       << ":"
                       << values);
#else
    MIOPEN_LOG_I("Read record " << key << ":" << id << ":" << values);
#endif
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

std::atomic_int DbRecord::n_cached_records(0);

void DbRecord::Flush()
{
    if(!is_cache_dirty)
        return;
    if(n_cached_records > 1)
    {
        MIOPEN_LOG_E("File update canceled to avoid db corruption. Key: " << key);
        return;
    }

    if(pos_begin < 0 || pos_end < 0)
    {
        std::ofstream file(db_filename, std::ios::app);

        if(!file)
        {
            MIOPEN_LOG_E("File is unwritable.");
            return;
        }

        pos_begin = file.tellp();
        Write(file, key, content);
        pos_end = file.tellp();
    }
    else
    {
        const auto temp_name = db_filename + ".temp";
        std::ifstream from(db_filename, std::ios::ate);

        if(!from)
        {
            MIOPEN_LOG_E("File is unreadable.");
            return;
        }

        std::ofstream to(temp_name);

        if(!to)
        {
            MIOPEN_LOG_E("Temp file is unwritable.");
            return;
        }

        const auto from_size = from.tellg();
        from.seekg(std::ios::beg);

        Copy(from, to, pos_begin);
        Write(to, key, content);
        const auto new_end = to.tellp();
        from.seekg(pos_end);
        Copy(from, to, from_size - pos_end);

        from.close();
        to.close();

        std::remove(db_filename.c_str());
        std::rename(temp_name.c_str(), db_filename.c_str());
        /// \todo What if rename fails? Thou shalt not loose the original file.

        // After successful write, position of the record's end needs to be updated.
        // Position of the beginning remains the same.
        pos_end = new_end;
    }
    is_cache_dirty = false;
}

void DbRecord::ReadIntoCache()
{
    ++n_cached_records;
    content.clear();
    is_cache_dirty = false;
    is_content_cached =
        true; // This is true even if no record found in the db: nothing read <-> nothing cached.

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    MIOPEN_LOG_I("Looking for key: " << key << ", legacy_key: " << legacy_key);
#else
    MIOPEN_LOG_I("Looking for key: " << key);
#endif

    std::ifstream file(db_filename);

    if(!file)
    {
        MIOPEN_LOG_W("File is unreadable.");
        return;
    }

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    record_format = RecordFormat::Current; // Used if none record found.
#endif
    pos_begin  = -1;
    pos_end    = -1;
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
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        const auto legacy_key_size = is_backward_compatible ? line.find(' ') : std::string::npos;
        const bool is_legacy_key   = (legacy_key_size != std::string::npos && legacy_key_size != 0);
        if(!is_key && !is_legacy_key)
#else
        if(!is_key)
#endif
        {
            if(!line.empty()) // Do not blame empty lines.
            {
                MIOPEN_LOG_E("Ill-formed record: key not found.");
                MIOPEN_LOG_E(db_filename << "#" << n_line);
            }
            continue;
        }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        // Needs to know record format for key compare.
        // Format of a matching record is not yet known because
        // actual compare is not performed yet.
        RecordFormat this_record_format = RecordFormat::Current;
        if(is_backward_compatible)
        {
            // Current format ('=' separator after KEY) takes precedence over
            // legacy conv perf db format (with ' ' separator), because current format is
            // allowed to contain spaces everywhere, while legacy format does
            // not use '='. So:
            this_record_format = (is_key ? RecordFormat::CurrentOrMixed : RecordFormat::Legacy);
        }

        const auto current_key = line.substr(0, is_key ? key_size : legacy_key_size);
#else
        const auto current_key = line.substr(0, key_size);
#endif

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        if(this_record_format == RecordFormat::Legacy)
        {
            if(current_key != legacy_key)
            {
                continue;
            }
        }
        else
        {
            if(current_key != key)
            {
                continue;
            }
        }
        record_format = this_record_format;
        MIOPEN_LOG_I("Key match: "
                     << current_key
                     << " record format: "
                     << ((record_format == RecordFormat::Legacy)
                             ? "Legacy"
                             : (record_format == RecordFormat::CurrentOrMixed)
                                   ? "CurrentOrMixed"
                                   : (record_format == RecordFormat::Mixed) ? "Mixed" : "Current"));
        const auto contents = line.substr((is_key ? key_size : legacy_key_size) + 1);
#else
        if(current_key != key)
        {
            continue;
        }
        MIOPEN_LOG_I(std::string("Key match: " << current_key);
        const auto contents    = line.substr(key_size + 1);
#endif

        if(contents.empty())
        {
            MIOPEN_LOG_E("None contents under the key: " << current_key);
            continue;
        }
        MIOPEN_LOG_I("Contents found: " << contents);

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        const bool is_parse_ok = (record_format == RecordFormat::Legacy)
                                     ? ParseLegacyContents(contents)
                                     : ParseContents(contents);
#else
        const bool is_parse_ok = ParseContents(contents);
#endif

        if(!is_parse_ok)
        {
            MIOPEN_LOG_E("Error parsing payload under the key: " << current_key);
            MIOPEN_LOG_E(db_filename << "#" << n_line);
            MIOPEN_LOG_E(contents);
        }
        // A record with matching key have been found.
        pos_begin = line_begin;
        pos_end   = next_line_begin;
        break;
    }
}
} // namespace miopen
