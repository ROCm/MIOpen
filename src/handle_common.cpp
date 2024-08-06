/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/handle.hpp>

#include <miopen/db_preload.hpp>
#include <miopen/errors.hpp>
#include <miopen/find_db.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/ramdb.hpp>
#include <miopen/readonlyramdb.hpp>

#include <mutex>

namespace miopen {
namespace {
template <class Db>
void StartPreloadingDb(DbPreloadStates& states,
                       DbKinds db_kind,
                       const fs::path& path,
                       bool is_system)
{
    if(path.empty())
        return;

    auto task = [db_kind, path, is_system]() {
        if constexpr(std::is_same_v<Db, RamDb>)
        {
            auto db   = std::make_unique<RamDb>(db_kind, path, is_system);
            auto lock = std::unique_lock<LockFile>(db->GetLockFile(), GetDbLockTimeout());

            if(!lock)
                MIOPEN_THROW("Db lock has failed to lock.");
            db->Prefetch();

            return PreloadedDb(std::move(db));
        }
        else
        {
            auto db     = std::make_unique<Db>(db_kind, path);
            std::ignore = is_system;
            db->Prefetch();
            return PreloadedDb(std::move(db));
        }
    };

    auto future = std::async(std::launch::async, std::move(task));

    states.futures.emplace(path, std::move(future));
}
} // namespace

void Handle::TryStartPreloadingDbs()
{
    ExecutionContext ctx{this};

    auto& states = GetDbPreloadStates();

    if(states.started_loading.load(std::memory_order_relaxed))
        return;

    std::unique_lock<std::mutex> lock(states.mutex);

    if(states.started_loading.load(std::memory_order_relaxed))
        return;

    MIOPEN_LOG_I("Preloading dbs");

    auto check = 0;

    // conv perf-db
#if !MIOPEN_DISABLE_SYSDB
    StartPreloadingDb<ReadonlyRamDb>(states, DbKinds::PerfDb, ctx.GetPerfDbPath(), true);
    check++;
#endif
#if !MIOPEN_DISABLE_USERDB
    StartPreloadingDb<RamDb>(states, DbKinds::PerfDb, ctx.GetUserPerfDbPath(), false);
    check++;
#endif

    // conv find-db
#if !MIOPEN_DISABLE_SYSDB
    StartPreloadingDb<ReadonlyRamDb>(
        states, DbKinds::FindDb, FindDbRecord::GetInstalledPath(*this, ""), true);
    check++;
#endif
#if !MIOPEN_DISABLE_USERDB
    StartPreloadingDb<RamDb>(states, DbKinds::FindDb, FindDbRecord::GetUserPath(*this, ""), false);
    check++;
#endif

    // fusion perf-db
#if !MIOPEN_DISABLE_SYSDB
    StartPreloadingDb<ReadonlyRamDb>(states, DbKinds::PerfDb, ctx.GetPerfDbPath("fusion"), true);
    check++;
#endif
#if !MIOPEN_DISABLE_USERDB
    StartPreloadingDb<RamDb>(states, DbKinds::PerfDb, ctx.GetUserPerfDbPath("fusion"), false);
    check++;
#endif

    // fusion find-db
#if !MIOPEN_DISABLE_SYSDB
    StartPreloadingDb<ReadonlyRamDb>(
        states, DbKinds::FindDb, FindDbRecord::GetInstalledPath(*this, "fusion"), true);
    check++;
#endif
#if !MIOPEN_DISABLE_USERDB
    StartPreloadingDb<RamDb>(
        states, DbKinds::FindDb, FindDbRecord::GetUserPath(*this, "fusion"), false);
    check++;
#endif

    if(check != DbPreloadStates::GetNumberOfDbsToLoad())
        // If someone registered more dbs, but forgot to add some specific one to total amount this
        // would crash tests
        MIOPEN_THROW(miopenStatusInternalError,
                     "Update DbPreloadStates::GetNumberOfDbsToLoad() value");

    states.started_loading = true;
}
} // namespace miopen
