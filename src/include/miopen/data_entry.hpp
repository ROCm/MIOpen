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
#include <unordered_map>

namespace miopen {

class DataEntry
{
    private:
    static bool _instance_loaded;
    bool _read        = false;
    bool _has_changes = false;
    /// Caches position of the found (read) record in the db file
    /// in order to optimize record update (i.e. write to file).
    /// This introduces an implicit dependence between class instances, i.e.
    /// update of db file made by one instance may invalidate cached position
    /// in another instance.
    ///
    /// \todo Redesign db access and remove the limitation that only
    /// one instance of this class is allowed to be loaded.
    std::streamoff _record_begin = -1;
    std::streamoff _record_end   = -1;
    const std::string _path;
    const std::string _entry_key;
    std::unordered_map<std::string, std::string> _content;

    template <class T>
    static // for calling from ctor
        std::string
        Serialize(const T& data)
    {
        std::ostringstream ss;
        data.Serialize(ss);
        return ss.str();
    }

    bool ParseEntry(const std::string& entry);
    bool Save(const std::string& key, const std::string& value);
    bool Load(const std::string& key, std::string& value);
    void Flush();
    void ReadFromDisk();

    public:
    DataEntry(const std::string& path, const std::string& entry_key)
        : _path(path), _entry_key(entry_key)
    {
    }

    // K which shall provide "void Serialize(std::ostream&) const" method.
    template <class K>
    DataEntry(const std::string& path, const K& entry_key) : DataEntry(path, Serialize(entry_key))
    {
    }

    ~DataEntry()
    {
        Flush();
        _instance_loaded = false;
    }

    template <class T>
    bool Save(const std::string& key, const T& value)
    {
        return Save(key, Serialize(value));
    }

    template <class T>
    bool Load(const std::string& key, T& value)
    {
        std::string str;

        if(!Load(key, str))
            return false;

        return value.Deserialize(str);
    }
};
} // namespace miopen

#endif
