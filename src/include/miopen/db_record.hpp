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
#ifndef GUARD_MIOPEN_DB_RECORD_HPP_
#define GUARD_MIOPEN_DB_RECORD_HPP_

#include "miopen/config.h"
#include "miopen/logger.hpp"

#include <sstream>
#include <string>
#include <atomic>
#include <unordered_map>
#include <iostream>

namespace miopen {

/// db consists of 0 or more records.
/// Each record is an ASCII text line.
/// Record format:
///   [ KEY "=" ID ":" VALUES { ";" ID ":" VALUES} ]
///
/// KEY - An identifer of a record.
/// ID - Can be considered as a sub-key under which respective VALUES are stored.
/// VALUES - A data associated with specific ID under the KEY. Intended to represent a set of
/// values, hence the name.
///
/// Neither of ";:=" within KEY, ID and VALUES is allowed.
/// There should be none identical KEYs in the same db file.
/// There should be none identical IDs within the same record.
///
/// Intended usage:
/// KEY: A stringized problem config.
/// ID: A symbolic name of the Solver applicable for the KEY (problem config). There could be
/// several Solvers able to handle the same config, so several IDs can be put under a KEY.
/// Format of VALUES stored under each ID is Solver-specific. in other words, how a set of values
/// (or whatever a Solver wants to store in VALUES) is encoded into a string depends on the Solver.
/// Note: If VALUES is used to represent a set of numeric values, then it is recommended to use ","
/// as a separator.

/// Represents a db record associated with specific KEY.
/// Ctor arguments are path to db file and a KEY (or an object able to provide a KEY).
/// Upon construction, allows getting and modifying contents of a record (IDs and VALUES).
/// A record in db file is updated when an instance is destroyed.
///
/// Note: An instance of this class caches position of the found (read) record within db file.
/// This allows to optimize record update (i.e. write to db file).
/// The drawback is: there are implicit dependencies among class instances.
/// An update of db file made by one instance may invalidate cached position
/// stored in another instance. That is why write to db file is disabled when
/// more than 1 instance if this class exist.
///
/// \todo Separate "db file" and "db record" abstractions.
/// \todo Redesign db access and remove the limitation.
/// \todo Check if implementation is MT- and MP-safe.
class DbRecord
{
    private:
    static std::atomic_int n_cached_records;
    // Positions of the record loaded from the db file:
    std::streamoff pos_begin = -1;
    std::streamoff pos_end   = -1;

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
#define MIOPEN_PERFDB_CONV_LEGACY_ID "__LEGACY__"
    enum class RecordFormat
    {
        Current,        // Format as documented above.
        Mixed,          // One legacy content, and >=1 content(s) in current format.
        Legacy,         // KEY in legacy format (x-separated). Exactly one legacy content.
        CurrentOrMixed, // Intermediate status: fomat not yet known, but not Legacy for sure.
    };
    enum class ContentFormat
    {
        Current, // ID:VALUES in current format in both db file and internal map.
        Legacy,  // This kind of content can't appear when record format is Current.
                 //
                 // Format of ID in the db file:
                 // - None ID for legacy records.
                 // - ID == "__LEGACY__" in mixed records.
                 // In the internal map:
                 // - ID is always "__LEGACY__".
                 //
                 // VALUES are always in legacy format (dot-separated).
    };
    RecordFormat record_format = RecordFormat::Current;
#endif
    bool is_content_cached = false;
    bool is_cache_dirty    = false;
    const std::string db_filename;
    const std::string key;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    const std::string legacy_key;
    bool is_backward_compatible = false; // for clang-tidy
#endif
    std::unordered_map<std::string, std::string> content;

    template <class T>
    static // 'static' is for calling from ctor
        std::string
        Serialize(const T& data)
    {
        std::ostringstream ss;
        data.Serialize(ss);
        return ss.str();
    }

    template <class T>
    static // 'static' is for calling from ctor
        std::string
        LegacySerialize(const T& data)
    {
        std::ostringstream ss;
        data.LegacySerialize(ss);
        return ss.str();
    }

    bool ParseLegacyContents(const std::string& contents);
    bool ParseContents(const std::string& contents);
    bool StoreValues(const std::string& id, const std::string& values);
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    bool LoadValues(const std::string& id, std::string& values, ContentFormat& content_format);
#else
    bool LoadValues(const std::string& id, std::string& values);
#endif
    void Flush();
    void ReadIntoCache();

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    DbRecord(const std::string& db_filename_,
             const std::string& key_,
             const std::string& legacy_key_,
             const bool is_backward_compatible_)
        : db_filename(db_filename_),
          key(key_),
          legacy_key(legacy_key_),
          is_backward_compatible(is_backward_compatible_)
#else
    DbRecord(const std::string& db_filename_, const std::string& key_)
        : db_filename(db_filename_), key(key_)
#endif
    {
    }

    public:
    /// T shall provide a db KEY by means of the
    /// "void Serialize(std::ostream&) const"
    /// member function.
    template <class T>
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    DbRecord(const std::string& db_filename_,
             const T& problem_config_,
             const bool is_backward_compatible_ = false)
        : DbRecord(db_filename_,
                   Serialize(problem_config_),
                   is_backward_compatible_ ? LegacySerialize(problem_config_) : "<NONE LEGACY KEY>",
                   is_backward_compatible_)
#else
    DbRecord(const std::string& db_filename_, const T& problem_config_)
        : DbRecord(db_filename_, Serialize(problem_config_))
#endif
    {
    }

    ~DbRecord()
    {
        Flush();
        content.clear();
        if(is_content_cached)
            --n_cached_records;
    }

    /// Obtains values from an object of class T and stores it
    /// in db (in association with id, under the current key).
    /// T shall have the "void Serialize(std::ostream&) const"
    /// member function available.
    template <class T>
    bool Store(const std::string& id, const T& values)
    {
        return StoreValues(id, Serialize(values));
    }

    /// Loads values associated with id under the current key
    /// (KEY:ID:VALUES) and delivers those to a member function
    /// of a class T object. T shall have the
    /// "bool Deserialize(const std::string& str)"
    /// member function available.
    ///
    /// Returns false if there is none KEY:ID:VALUES in the database
    /// or in case of any error, e.g. if VALUES cannot be deserialized
    /// due to incorrect format.
    template <class T>
    bool Load(const std::string& id, T& values)
    {
        std::string s;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        ContentFormat s_format = ContentFormat::Current;
        if(!LoadValues(id, s, s_format))
#else
        if(!LoadValues(id, s))
#endif
            return false;

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
        bool ok = false;
        if(s_format == ContentFormat::Legacy)
        {
            ok = values.LegacyDeserialize(s);
        }
        else
        {
            ok = values.Deserialize(s);
        }
#else
        const bool ok = values.Deserialize(s);
#endif
        if(!ok)
        {
            MIOPEN_LOG(LoggingLevel::Error, "deserialize failed: " << s);
        }
        return ok;
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_DB_RECORD_HPP_
