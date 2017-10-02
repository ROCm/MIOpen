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
template <class TData>
inline std::string Serialize(const TData& data)
{
    std::ostringstream ss;
    data.Serialize(ss);
    return ss.str();
}

class DataEntry
{
    public:
    DataEntry(const std::string& path, const std::string& entry_key);

    template <class TEntryKey>
    DataEntry(const std::string& path, const TEntryKey& entry_key)
        : DataEntry(path, Serialize(entry_key))
    {
    }

    ~DataEntry();

    inline bool Read() const { return _read; }

    /// Returns true on success
    bool Save(const std::string& key, const std::string& value);

    /// Returns true on success
    template <class TData>
    bool Save(const std::string& key, const TData& value)
    {
        return Save(key, Serialize(value));
    }

    /// Returns true on success
    bool Load(const std::string& key, std::string& value) const;

    /// Returns true on success
    template <class TData>
    bool Load(const std::string& key, TData& value) const
    {
        std::string str;

        if(!Load(key, str))
            return false;

        return value.Deserialize(str);
    }

    void Flush();
    void ReadFromDisk();

    private:
    bool _read            = false;
    bool _has_changes     = false;
    std::streamoff _start = -1;
    const std::string _path;
    const std::string _entry_key;
    std::unordered_map<std::string, std::string> _content;

    bool ParseEntry(const std::string& entry);
};
} // namespace miopen

#endif
