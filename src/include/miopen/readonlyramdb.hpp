/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_GUARD_MLOPEN_READONLYRAMDB_HPP
#define MIOPEN_GUARD_MLOPEN_READONLYRAMDB_HPP

#include <miopen/db_record.hpp>

#include <boost/optional.hpp>

#include <unordered_map>
#include <string>
#include <sstream>

namespace miopen {

namespace debug {
extern bool& rordb_embed_fs_override();
} // namespace debug

class ReadonlyRamDb
{
    public:
    ReadonlyRamDb(std::string path) : db_path(path) {}

    static ReadonlyRamDb& GetCached(const std::string& path, bool warn_if_unreadable);

    boost::optional<DbRecord> FindRecord(const std::string& problem) const
    {
        MIOPEN_LOG_I2("Looking for key " << problem << " in file " << db_path);
        const auto it = cache.find(problem);

        if(it == cache.end())
            return boost::none;

        auto record = DbRecord{problem};

        MIOPEN_LOG_I2("Key match: " << problem);
        MIOPEN_LOG_I2("Contents found: " << it->second.content);

        if(!record.ParseContents(it->second.content))
        {
            MIOPEN_LOG_E("Error parsing payload under the key: "
                         << problem << " form file " << db_path << "#" << it->second.line);
            MIOPEN_LOG_E("Contents: " << it->second.content);
            return boost::none;
        }

        return record;
    }

    template <class TProblem>
    boost::optional<DbRecord> FindRecord(const TProblem& problem) const
    {
        const auto key = DbRecord::Serialize(problem);
        return FindRecord(key);
    }

    template <class TProblem, class TValue>
    bool Load(const TProblem& problem, const std::string& id, TValue& value) const
    {
        const auto record = FindRecord(problem);
        if(!record)
            return false;
        return record->GetValues(id, value);
    }

    private:
    struct CacheItem
    {
        int line;
        std::string content;
    };

    std::string db_path;
    std::unordered_map<std::string, CacheItem> cache;

    ReadonlyRamDb(const ReadonlyRamDb&) = default;
    ReadonlyRamDb(ReadonlyRamDb&&)      = default;
    ReadonlyRamDb& operator=(const ReadonlyRamDb&) = default;
    ReadonlyRamDb& operator=(ReadonlyRamDb&&) = default;

    void Prefetch(bool warn_if_unreadable);
    void ParseAndLoadDb(std::istream& input_stream, bool warn_if_unreadable);
};

} // namespace miopen

#endif
