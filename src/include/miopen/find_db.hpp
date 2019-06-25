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

#ifndef GUARD_MIOPEN_FIND_DB_HPP_
#define GUARD_MIOPEN_FIND_DB_HPP_

#include <miopen/db.hpp>
#include <miopen/db_path.hpp>
#include <miopen/db_record.hpp>
#include <miopen/env.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/readonlyramdb.hpp>

#include <boost/optional.hpp>

#include <functional>
#include <vector>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_DISABLE_FIND_DB)

namespace miopen {

struct Handle;

class FindDbRecord
{
    public:
    static bool enabled;                                  // For unit tests.
    static boost::optional<std::string>& path_override(); /// \todo Remove when #1723 is resolved.

    FindDbRecord(const FindDbRecord&) = delete;
    FindDbRecord& operator=(const FindDbRecord&) = delete;

    template <class TProblemDescription>
    FindDbRecord(Handle& handle, const TProblemDescription& problem)
        : path(path_override() ? *path_override() : GetUserPath(handle)),
          db(!enabled || IsEnabled(MIOPEN_DEBUG_DISABLE_FIND_DB{})
                 ? boost::none
                 : boost::optional<DbClass>{DbClass{
                       {path_override() ? *path_override() : GetInstalledPath(handle), path}}})
    {
        if(!db.is_initialized())
            return;

        content = db->FindRecord(problem);
        in_sync = content.is_initialized();
    }

    ~FindDbRecord()
    {
        if(!db.is_initialized() || !content.is_initialized() || in_sync)
            return;
        if(!db->StoreRecord(content.get()))
            MIOPEN_LOG_E("Failed to store record to find-db at <" << path << ">");
    }

    auto begin() const { return content->As<FindDbData>().begin(); }
    auto begin() { return content->As<FindDbData>().begin(); }
    auto end() const { return content->As<FindDbData>().end(); }
    auto end() { return content->As<FindDbData>().end(); }
    bool empty() const { return !content.is_initialized(); }

    template <class TProblemDescription>
    static std::vector<PerfField> TryLoad(Handle& handle,
                                          const TProblemDescription& problem,
                                          const std::function<void(DbRecord&)>& regenerator)
    {
        auto ret = std::vector<PerfField>{};
        FindDbRecord record{handle, problem};

        if(record.in_sync && !record.CopyValidating(handle, ret))
            return ret;

        MIOPEN_LOG_I("Find-db regenerating.");
        ret.clear();
        record.in_sync = false;
        record.content.emplace(problem);
        regenerator(*record.content);

        for(const auto& pair : record)
            // cppcheck-suppress useStlAlgorithm
            ret.push_back(
                {pair.first, pair.second.solver_id, pair.second.time, pair.second.workspace});

        return ret;
    }

    private:
#if MIOPEN_DEBUG_FIND_DB_CACHING == 1
    using DbClass = DbTimer<MultiFileDb<ReadonlyRamDb, Db, false>>;
#else
    using DbClass = DbTimer<MultiFileDb<Db, Db, false>>;
#endif

    std::string path;
    boost::optional<DbClass> db;
    boost::optional<DbRecord> content{boost::none};
    bool in_sync = false;

    static bool HasKernel(Handle& handle, const FindDbKCacheKey& key);

    static std::string GetInstalledPath(Handle& handle);
    static std::string GetUserPath(Handle& handle);

    // Returns true if rebuild is required
    bool CopyValidating(Handle& handle, std::vector<PerfField>& to) const;

    void LogFindDbItem(bool is_valid, const std::pair<std::string, FindDbData>& pair) const;
};

} // namespace miopen

#endif
