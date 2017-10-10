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

namespace miopen {

using perfdb::Logger;

#if defined(LOG_I) || defined(LOG_W) || defined(LOG_E)
#error "Error: Unexpected definition(s) of LOG_I, LOG_W, LOG_E macros found."
#endif
#if MIOPEN_DB_RECORD_LOGLEVEL >= 3
#define LOG_I(...)             \
    do                         \
    {                          \
        logger.I(__VA_ARGS__); \
    } while(false)
#else
#define LOG_I(...)
#endif
#if MIOPEN_DB_RECORD_LOGLEVEL >= 2
#define LOG_W(...)             \
    do                         \
    {                          \
        logger.W(__VA_ARGS__); \
    } while(false)
#else
#define LOG_W(...)
#endif
#if MIOPEN_DB_RECORD_LOGLEVEL >= 1
#define LOG_E(...)             \
    do                         \
    {                          \
        logger.E(__VA_ARGS__); \
    } while(false)
#else
#define LOG_E(...)
#endif

bool DbRecord::ParseContents(const std::string& contents)
{
    static const Logger logger("DbRecord::ParseContents");
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(_record_format == RecordFormat::CurrentOrMixed ||
           _record_format == RecordFormat::Current);
#endif
    std::istringstream ss(contents);
    std::string id_and_values;
    int found = 0;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    bool is_legacy_content_found         = false;
    const bool is_legacy_content_allowed = (_record_format == RecordFormat::CurrentOrMixed);
#endif

    while(std::getline(ss, id_and_values, ';'))
    {
        const auto id_size = id_and_values.find(':');

        // Empty VALUES is ok, empty ID is not:
        if(id_size == std::string::npos)
        {
            LOG_E(std::string("Ill-formed file: ID not found; skipped; key: ") + _key);
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
                LOG_I(std::string("Legacy content found under key: ") + _key + " Record is Mixed.");
            }
        }
#endif
        const auto values = id_and_values.substr(id_size + 1);

        if(_content.find(id) != _content.end())
        {
            LOG_E(std::string("Duplicate ID (ignored): ") + id + "; key: " + _key);
            continue;
        }

        _content.emplace(id, values);
        ++found;
    }

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    if(!is_legacy_content_found)
    {
        _record_format = RecordFormat::Current;
    }
    else
    {
        _record_format = RecordFormat::Mixed;
    }
#endif
    return (found > 0);
}

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
bool DbRecord::ParseLegacyContents(const std::string& contents)
{
    assert(_record_format == RecordFormat::Legacy);
    const auto id = MIOPEN_PERFDB_CONV_LEGACY_ID;
    _content.emplace(id, contents);
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
    static const Logger logger("DbRecord::StoreValues");
    if(!_is_content_cached)
    {
        // If there is a record with the same key, we need to find its position
        // in the db file. Otherwise the new record with the same key will NOT
        // replace the existing record and db file become ill-formed.
        ReadIntoCache();
    }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(_record_format != RecordFormat::CurrentOrMixed);
    if((_record_format == RecordFormat::Legacy || _record_format == RecordFormat::Mixed) &&
       isLegacySolver(id))
    {
        // "__LEGACY__" id shall be replaced by actual legacy SolverId.
        // Values shall be replaced too (we do not want to cope with
        // comparison of legacy values with values in the current format).
        // That is, the whole map entry shall be replaced.
        // Record format becomes Current after that.
        const auto it = _content.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
        assert(it != _content.end());
        _content.erase(it);
        _content.emplace(id, values);
        _is_cache_dirty = true;
        _record_format  = RecordFormat::Current;
        LOG_I(std::string("Legacy content under key: ") + _key + " replaced by " + id + ":" +
              values);
        return true;
    }
    else if(_record_format == RecordFormat::Legacy && !isLegacySolver(id))
    {
        // Non-legacy SolverId cannot reside in the legacy record by definition.
        // Just add a content to the map and mark record as Mixed.
        assert(_content.find(id) == _content.end());
        _content.emplace(id, values);
        _is_cache_dirty = true;
        _record_format  = RecordFormat::Mixed;
        LOG_I(std::string("Legacy record under key: ") + _key + " appended by " + id + ":" +
              values + " and becomes Mixed");
        return true;
    }
    assert((_record_format == RecordFormat::Mixed && !isLegacySolver(id)) ||
           _record_format == RecordFormat::Current);
#endif
    // No need to update the file if values are the same:
    const auto it = _content.find(id);
    if(it == _content.end() || it->second != values)
    {
        LOG_I(std::string("Record under key: ") + _key + ", content " +
              (it == _content.end() ? "inserted" : "overwritten") + ": " + id + ":" + values);
        _content[id]    = values;
        _is_cache_dirty = true;
    }
    else
    {
        LOG_I(std::string("Record under key: ") + _key + ", content is the same, not saved:" + id +
              ":" + values);
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
    static const Logger logger("DbRecord::LoadValues");
    if(!_is_content_cached)
        ReadIntoCache();

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    assert(_record_format != RecordFormat::CurrentOrMixed);
    if(_record_format == RecordFormat::Legacy)
    {
        if(!isLegacySolver(id))
            return false;
        const auto it = _content.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
        if(it == _content.end())
            return false;

        values         = it->second;
        content_format = ContentFormat::Legacy;
        LOG_I(std::string("Legacy record read: ") + _key + ":" + values + " for id: " + id);
        return true;
    }
    else if(_record_format == RecordFormat::Mixed)
    {
        if(isLegacySolver(id))
        {
            // contents for legacy solvers may be in either legacy of current format.
            const auto it = _content.find(MIOPEN_PERFDB_CONV_LEGACY_ID);
            if(it != _content.end())
            {
                values         = it->second;
                content_format = ContentFormat::Legacy;
                LOG_I(std::string("Legacy content read from record (Mixed): ") + _key + ":" + id +
                      ":" + values);
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
    const auto it = _content.find(id);

    if(it == _content.end())
        return false;

    values = it->second;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    content_format = ContentFormat::Current;
    LOG_I(std::string("Read record ") +
          ((_record_format == RecordFormat::Mixed) ? "(Mixed) " : "(Current) ") + _key + ":" + id +
          ":" + values);
#else
    LOG_I(std::string("Read record ") + _key + ":" + id + ":" + values);
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

std::atomic_int DbRecord::_n_cached_records(0);

void DbRecord::Flush()
{
    static const Logger logger("DbRecord::Flush");
    if(!_is_cache_dirty)
        return;
    if(_n_cached_records > 1)
    {
        LOG_E(std::string("File update canceled to avoid db corruption. Key: ") + _key);
        return;
    }

    if(_pos_begin < 0 || _pos_end < 0)
    {
        std::ofstream file(_db_filename, std::ios::app);

        if(!file)
        {
            LOG_E("File is unwritable.");
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
            LOG_E("File is unreadable.");
            return;
        }

        std::ofstream to(temp_name);

        if(!to)
        {
            LOG_E("Temp file is unwritable.");
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
    static const Logger logger("DbRecord::ReadIntoCache");
    ++_n_cached_records;
    _content.clear();
    _is_cache_dirty = false;
    _is_content_cached =
        true; // This is true even if no record found in the db: nothing read <-> nothing cached.

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    LOG_I(std::string("Looking for key: ") + _key + ", legacy_key: " + _legacy_key);
#else
    LOG_I(std::string("Looking for key: ") + _key);
#endif

    std::ifstream file(_db_filename);

    if(!file)
    {
        LOG_W("File is unreadable.");
        return;
    }

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    _record_format = RecordFormat::Current; // Used if none record found.
#endif
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
        const bool is_key   = (key_size != std::string::npos && key_size != 0);
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        const auto legacy_key_size = _is_backward_compatible ? line.find(' ') : std::string::npos;
        const bool is_legacy_key   = (legacy_key_size != std::string::npos && legacy_key_size != 0);
        if(!is_key && !is_legacy_key)
#else
        if(!is_key)
#endif
        {
            if(!line.empty()) // Do not blame empty lines.
            {
                LOG_E("Ill-formed record: key not found.");
                LOG_E(_db_filename + "#" + std::to_string(n_line));
            }
            continue;
        }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        // Needs to know record format for key compare.
        // Format of a matching record is not yet known because
        // actual compare is not performed yet.
        RecordFormat this_record_format = RecordFormat::Current;
        if(_is_backward_compatible)
        {
            // Current format ('=' separator after KEY) takes precedence over
            // legacy conv perf db format (with ' ' separator), because current format is
            // allowed to contain spaces everywhere, while legacy format does
            // not use '='. So:
            this_record_format = (is_key ? RecordFormat::CurrentOrMixed : RecordFormat::Legacy);
        }

        const auto key = line.substr(0, is_key ? key_size : legacy_key_size);
#else
        const auto key = line.substr(0, key_size);
#endif

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        if(this_record_format == RecordFormat::Legacy)
        {
            if(key != _legacy_key)
            {
                continue;
            }
        }
        else
        {
            if(key != _key)
            {
                continue;
            }
        }
        _record_format = this_record_format;
        LOG_I(std::string("Key match: ") + key + " record format: " +
              ((_record_format == RecordFormat::Legacy)
                   ? "Legacy"
                   : (_record_format == RecordFormat::CurrentOrMixed)
                         ? "CurrentOrMixed"
                         : (_record_format == RecordFormat::Mixed) ? "Mixed" : "Current"));
        const auto contents = line.substr((is_key ? key_size : legacy_key_size) + 1);
#else
        if(key != _key)
        {
            continue;
        }
        LOG_I(std::string("Key match: ") + key);
        const auto contents    = line.substr(key_size + 1);
#endif

        if(contents.empty())
        {
            LOG_E(std::string("None contents under the key: ") + key);
            continue;
        }
        LOG_I(std::string("Contents found: ") + contents);

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        const bool is_parse_ok = (_record_format == RecordFormat::Legacy)
                                     ? ParseLegacyContents(contents)
                                     : ParseContents(contents);
#else
        const bool is_parse_ok = ParseContents(contents);
#endif

        if(!is_parse_ok)
        {
            LOG_E(std::string("Error parsing payload under the key: ") + key);
            LOG_E(_db_filename + "#" + std::to_string(n_line));
            LOG_E(contents);
        }
        // A record with matching key have been found.
        _pos_begin = line_begin;
        _pos_end   = next_line_begin;
        break;
    }
}
} // namespace miopen
