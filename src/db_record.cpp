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

        if(map.find(id) != map.end())
        {
            MIOPEN_LOG_E("Duplicate ID (ignored): " << id << "; key: " << key);
            continue;
        }

        map.emplace(id, values);
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
    map.emplace(id, contents);
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
        const auto it = map.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
        assert(it != map.end());
        map.erase(it);
        map.emplace(id, values);
        record_format = RecordFormat::Current;
        MIOPEN_LOG_I("Legacy content under key: " << key << " replaced by " << id << ':' << values);
        return true;
    }
    else if(record_format == RecordFormat::Legacy && !isLegacySolver(id))
    {
        // Non-legacy SolverId cannot reside in the legacy record by definition.
        // Just add a content to the map and mark record as Mixed.
        assert(map.find(id) == map.end());
        map.emplace(id, values);
        record_format = RecordFormat::Mixed;
        MIOPEN_LOG_I("Legacy record under key: " << key << " appended by " << id << ':' << values
                                                 << " and becomes Mixed");
        return true;
    }
    assert((record_format == RecordFormat::Mixed && !isLegacySolver(id)) ||
           record_format == RecordFormat::Current);
#endif
    // No need to update the file if values are the same:
    const auto it = map.find(id);
    if(it == map.end() || it->second != values)
    {
        MIOPEN_LOG_I("Record under key: " << key << ", content "
                                          << (it == map.end() ? "inserted" : "overwritten")
                                          << ": "
                                          << id
                                          << ':'
                                          << values);
        map[id] = values;
        return true;
    }
    MIOPEN_LOG_I("Record under key: " << key << ", content is the same, not saved:" << id << ':'
                                      << values);
    return false;
}

bool DbRecord::Erase(const std::string& id)
{
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(record_format != RecordFormat::CurrentOrMixed);
    if((record_format == RecordFormat::Legacy || record_format == RecordFormat::Mixed) &&
       isLegacySolver(id))
    {
        const auto it = map.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
        assert(it != map.end());
        MIOPEN_LOG_I(
            "Legacy content under key: " << key << " removed:" << it->second << ", id: " << id);
        map.erase(it);
        record_format = RecordFormat::Current;
        return true;
    }
    else if(record_format == RecordFormat::Legacy && !isLegacySolver(id))
    {
        // Non-legacy SolverId cannot reside in the legacy record by definition.
        assert(map.find(id) == map.end());
        MIOPEN_LOG_W("Legacy record under key: " << key << ", not found: " << id);
        return false;
    }
    assert((record_format == RecordFormat::Mixed && !isLegacySolver(id)) ||
           record_format == RecordFormat::Current);
#endif
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

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
bool DbRecord::LoadValues(const std::string& id,
                          std::string& values,
                          DbRecord::ContentFormat& content_format)
#else
bool DbRecord::LoadValues(const std::string& id, std::string& values)
#endif
{
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(record_format != RecordFormat::CurrentOrMixed);
    if(record_format == RecordFormat::Legacy)
    {
        if(!isLegacySolver(id))
            return false;
        const auto it = map.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
        if(it == map.end())
            return false;

        values         = it->second;
        content_format = ContentFormat::Legacy;
        MIOPEN_LOG_I("Read record (Legacy): " << legacy_key << " " << values << " for id: " << id);
        return true;
    }
    else if(record_format == RecordFormat::Mixed)
    {
        if(isLegacySolver(id))
        {
            // contents for legacy solvers may be in either legacy of current format.
            const auto it = map.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
            if(it != map.end())
            {
                values         = it->second;
                content_format = ContentFormat::Legacy;
                MIOPEN_LOG_I("Read record (Mixed): " << key << '=' << MIOPEN_PERFDB_CONV_LEGACY_ID
                                                     << ':'
                                                     << values
                                                     << " for id: "
                                                     << id);
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
    const auto it = map.find(id);

    if(it == map.end())
        return false;

    values = it->second;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    content_format = ContentFormat::Current;
    MIOPEN_LOG_I(
        "Read record " << ((record_format == RecordFormat::Mixed) ? "(Mixed) " : "(Current) ")
                       << key
                       << '='
                       << id
                       << ':'
                       << values);
#else
    MIOPEN_LOG_I("Read record " << key << '=' << id << ':' << values);
#endif
    return true;
}

static void Write(std::ostream& stream,
                  const std::string& key,
                  std::unordered_map<std::string, std::string>& map)
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
        MIOPEN_LOG_I("Key match: " << current_key);
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
        if(pos)
        {
            pos->begin = line_begin;
            pos->end   = next_line_begin;
        }
        break;
    }
}
} // namespace miopen
