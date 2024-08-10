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

#include "miopen/db_record.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/ramdb.hpp"
#include "miopen/readonlyramdb.hpp"
#include <miopen/db_preload.hpp>

#include <miopen/config.h>

#include <chrono>

namespace miopen {
auto GetDbPreloadStates() -> DbPreloadStates&
{
    static DbPreloadStates db_preload_states;
    return db_preload_states;
}

namespace {
template <class Db>
auto GetPreloadedDb(const fs::path& path, DbPreloadStates& states) -> std::unique_ptr<Db>
{
    std::unique_lock<std::mutex> lock{states.mutex, std::defer_lock};

    // Mutex is need to ensure states.futures is not updated while we work
    // so we skip locking if it no more writes can happen
    const auto needs_lock = !states.started_loading.load(std::memory_order_relaxed);

    if(needs_lock)
        lock.lock();

    auto it = states.futures.find(path);

    if(it == states.futures.end())
        return nullptr;

    auto future = std::move(it->second);

    if(needs_lock)
        lock.unlock();

    const auto start = std::chrono::high_resolution_clock::now();
    auto ret         = future.get();
    const auto end   = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I2("GetPreloadedDb time waiting for the db: " << (end - start).count() * .000001f
                                                             << " ms");
    return std::get<std::unique_ptr<Db>>(std::move(ret));
}
} // namespace

template <class Db>
auto MakeDbPreloader(DbKinds db_kind, bool is_system) -> std::function<PreloadedDb(const fs::path&)>
{
    if constexpr(std::is_same_v<Db, RamDb>)
    {
        return [=](const fs::path& path) -> PreloadedDb {
            auto db   = std::make_unique<RamDb>(db_kind, path, is_system);
            auto lock = std::unique_lock<LockFile>(db->GetLockFile(), GetDbLockTimeout());
            if(!lock)
                MIOPEN_THROW("Db lock has failed to lock.");
            db->Prefetch();
            return {std::move(db)};
        };
    }
    else
    {
        return [=](const fs::path& path) -> PreloadedDb {
            auto db     = std::make_unique<Db>(db_kind, path);
            std::ignore = is_system;
            db->Prefetch();
            return {std::move(db)};
        };
    }
}

template auto MakeDbPreloader<RamDb>(DbKinds db_kind, bool is_system)
    -> std::function<PreloadedDb(const fs::path&)>;
template auto MakeDbPreloader<ReadonlyRamDb>(DbKinds db_kind, bool is_system)
    -> std::function<PreloadedDb(const fs::path&)>;

MIOPEN_INTERNALS_EXPORT void
StartPreloadingDb(const fs::path& path,
                  std::function<PreloadedDb(const fs::path&)>&& preloader,
                  DbPreloadStates& states)
{
    if(path.empty())
        return;

    auto future = std::async(std::launch::async, std::move(preloader), path);
    states.futures.emplace(path, std::move(future));
}

MIOPEN_INTERNALS_EXPORT void TryStartPreloadingDbs(const std::function<void()>& preload,
                                                   DbPreloadStates& states)
{
    if(states.started_loading.load(std::memory_order_relaxed))
        return;

    std::unique_lock<std::mutex> lock(states.mutex);

    if(states.started_loading.load(std::memory_order_relaxed))
        return;

    preload();

    states.started_loading.store(true, std::memory_order_relaxed);
}

MIOPEN_INTERNALS_EXPORT auto GetPreloadedRamDb(const fs::path& path, DbPreloadStates& states)
    -> std::unique_ptr<RamDb>
{
    return GetPreloadedDb<RamDb>(path, states);
}

MIOPEN_INTERNALS_EXPORT auto GetPreloadedReadonlyRamDb(const fs::path& path,
                                                       DbPreloadStates& states)
    -> std::unique_ptr<ReadonlyRamDb>
{
    return GetPreloadedDb<ReadonlyRamDb>(path, states);
}

} // namespace miopen
