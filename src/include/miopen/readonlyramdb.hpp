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

class ReadonlyRamDb
{
    public:
    ReadonlyRamDb(std::string path) : db_path(path), arch(""), num_cu(0) {}
    ReadonlyRamDb(std::string path, std::string _arch, std::size_t _num_cu)
        : db_path(path), arch(_arch), num_cu(_num_cu)
    {
    }

    static ReadonlyRamDb& GetCached(const std::string& path,
                                    bool warn_if_unreadable,
                                    const std::string& arch = "",
                                    std::size_t num_cu      = 0);

    boost::optional<DbRecord> FindRecord(const std::string& problem) const
    {
        const auto it = cache.find(problem);

        if(it == cache.end())
            return boost::none;

        auto record = DbRecord{problem};

        if(!record.ParseContents(it->second))
        {
            MIOPEN_LOG_E("Error parsing payload under the key: " << problem << " form file "
                                                                 << db_path
                                                                 << "#"
                                                                 << it->second);
            MIOPEN_LOG_E("Contents: " << it->second);
            return boost::none;
        }
        else
        {
            MIOPEN_LOG_I2("Looking for key " << problem << " in file " << db_path);
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

    std::string db_path;
    std::unordered_map<std::string, std::string> cache;
    std::string arch;
    std::size_t num_cu;

    ReadonlyRamDb(const ReadonlyRamDb&) = default;
    ReadonlyRamDb(ReadonlyRamDb&&)      = default;
    ReadonlyRamDb& operator=(const ReadonlyRamDb&) = default;
    ReadonlyRamDb& operator=(ReadonlyRamDb&&) = default;

    void Prefetch(const std::string& path, bool warn_if_unreadable);
};

struct FindRamDb : ReadonlyRamDb
{
    const std::unordered_map<std::string, std::string>& find_db_init(std::string arch_cu);
    FindRamDb(std::string path, std::string _arch, std::size_t _num_cu)
        : ReadonlyRamDb(":memory:" + path, _arch, _num_cu)
    {
        const auto& m = find_db_init(arch + "_" + std::to_string(num_cu));
        cache         = std::move(m);
    }
    // Override GetCached, since FindRamDb does not have state or init overhead
    static FindRamDb& GetCached(const std::string& path,
                                bool /*warn_if_unreadble*/,
                                const std::string& _arch,
                                const std::size_t _num_cu)
    {
        static auto inst = FindRamDb{path, _arch, _num_cu};
        return inst;
    }
};

struct PerfRamDb : ReadonlyRamDb
{
    const std::unordered_map<std::string, std::string>& perf_db_init(std::string arch_cu);
    PerfRamDb(std::string path, std::string _arch, std::size_t _num_cu)
        : ReadonlyRamDb(":memory:" + path, _arch, _num_cu)
    {
        const auto& m = perf_db_init(arch + "_" + std::to_string(num_cu));
        cache         = std::move(m);
    }

    static PerfRamDb& GetCached(const std::string& path,
                                bool /*warn_if_unreadable*/,
                                const std::string& _arch,
                                const std::size_t _num_cu)
    {
        static auto inst = new PerfRamDb{path, _arch, _num_cu};
        return *inst;
    }
};

} // namespace miopen

#endif
