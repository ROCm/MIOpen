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
#ifndef MIOPEN_DATA_ENTRY_HPP_
#define MIOPEN_DATA_ENTRY_HPP_

#include <sstream>
#include <string>
#include <atomic>
#include <unordered_map>

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
/// There should be none identical KEYs in the db.
/// There should be none identical IDs within the same record.
///
/// Intended usage:
/// KEY: A stringized problem config.
/// ID: A symbolic name of the solution applicable for the KEY (problem config). There could be
/// several solutions
/// for the same problem config, so several IDs can be put under a KEY.
/// Format of VALUES stored under each ID is solution-specific. in other words, how a set of values
/// (or whatever a solution
/// wants to store in VALUES) is encoded into a string depends on the solution.
/// Note: If VALUES is used to represent a set of numeric values, then it is recommended to use ","
/// as a separator.

/// Represents a db record associated with specific KEY.
/// Ctor arguments are path to db file and a KEY (or an object able to provide a KEY).
/// Upon construction, allows getting and modifying contents of a record (IDs and VALUES).
/// A record in db file is updated when an instance is destroyed.
///
/// Note: An instance of this class caches position of the found (read) record in the db file.
/// This allows to optimize record update (i.e. write to db file).
/// The drawback is an implicit dependencies among class instances:
/// update of db file made by one instance may invalidate cached position
/// stored in another instance. That is why write to db file is disabled when
/// more than 1 instance if this class exist.
///
/// \todo Redesign db access and remove the limitation.
/// \todo Check if implementation is MT- and MP-safe.
class DbRecord
{
    private:
    static std::atomic_int _n_cached_records;
    // Positions of the record loaded from the db file:
    std::streamoff _pos_begin = -1;
    std::streamoff _pos_end   = -1;

    bool _is_content_cached = false;
    bool _is_cache_dirty    = false;
    const std::string _db_filename;
    const std::string _key;
    std::unordered_map<std::string, std::string> _content;

    template <class T>
    static // static for calling from ctor
        std::string
        Serialize(const T& data)
    {
        std::ostringstream ss;
        data.Serialize(ss);
        return ss.str();
    }

    bool ParseContents(const std::string& contents);
    bool Save(const std::string& id, const std::string& values);
    bool Load(const std::string& id, std::string& values);
    void Flush();
    void ReadIntoCache();

    DbRecord(const std::string& db_filename, const std::string& key)
        : _db_filename(db_filename), _key(key)
    {
    }

    public:
    /// T shall provide a db KEY by means of the 
    /// "void Serialize(std::ostream&) const"
    /// member function.
    template <class T>
    DbRecord(const std::string& db_filename, const T& problem_config)
        : DbRecord(db_filename, Serialize(problem_config))
    {
    }

    ~DbRecord()
    {
        Flush();
        _content.clear();
        if(_is_content_cached)
            --_n_cached_records;
    }

    /// Obtains values from an object of class T and stores it
    /// in db (in association with id, under the current key).
    /// T shall have the "void Serialize(std::ostream&) const"
    /// member function available.
    template <class T>
    bool Save(const std::string& id, const T& values)
    {
        return Save(id, Serialize(values));
    }

    /// Loads values associated with id under the current key
    /// and delivers those to a member function of a class T object.
    /// T shall have the "bool Deserialize(const std::string& str)"
    /// member function available.
    template <class T>
    bool Load(const std::string& id, T& values)
    {
        std::string s;

        if(!Load(id, s))
            return false;

        return values.Deserialize(s);
    }
};
} // namespace miopen

#endif
