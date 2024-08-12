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
#include <atomic>
#include <future>
#include <miopen/db_preload.hpp>

#include <miopen/config.h>

#include <chrono>
#include <mutex>

namespace miopen {
auto GetDbPreloadStates() -> DbPreloadStates&
{
    static DbPreloadStates db_preload_states;
    return db_preload_states;
}

template <class Db>
auto DbPreloadStates::GetPreloadedDb(const fs::path& path) -> std::unique_ptr<Db>
{
    std::unique_lock<std::mutex> lock{mutex, std::defer_lock};

    // Mutex is need to ensure states.futures is not updated while we work
    // so we skip locking if it no more writes can happen
    const auto needs_lock = !started_loading.load(std::memory_order_relaxed);

    if(needs_lock)
        lock.lock();

    auto it = futures.find(path);

    if(it == futures.end())
        return nullptr;

    auto& future = it->second;

    if(needs_lock)
        lock.unlock();

    const auto start = std::chrono::high_resolution_clock::now();
    auto ret         = future.get();
    const auto end   = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I2("GetPreloadedDb time waiting for the db: " << (end - start).count() * .000001f
                                                             << " ms");
    return std::get<std::unique_ptr<Db>>(std::move(ret));
}

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
        std::ignore = is_system;

        return [=](const fs::path& path) -> PreloadedDb {
            auto db = std::make_unique<Db>(db_kind, path);
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
DbPreloadStates::StartPreloadingDb(const fs::path& path,
                                   std::function<PreloadedDb(const fs::path&)>&& preloader)
{
    if(path.empty())
        return;

    if(!finish_marker)
    {
        finish_source = std::promise<void>();
        finish_marker = finish_source->get_future();
    }

    threads_left.fetch_add(1, std::memory_order_relaxed);

    auto db_future = std::async(
        std::launch::async,
        [preloader = std::move(preloader), this](auto&& path) mutable {
            const auto left = threads_left.fetch_sub(1, std::memory_order_relaxed) - 1;
            if(left == 0)
                finish_source->set_value_at_thread_exit();
            return preloader(path);
        },
        path);
    futures.emplace(path, std::move(db_future));
}

MIOPEN_INTERNALS_EXPORT void
DbPreloadStates::TryStartPreloadingDbs(const std::function<void()>& preload)
{
    requesters.fetch_add(1, std::memory_order_relaxed);

    if(started_loading.load(std::memory_order_relaxed))
        return;

    std::unique_lock<std::mutex> lock(mutex);

    if(started_loading.load(std::memory_order_relaxed))
        return;

    preload();

    started_loading.store(true, std::memory_order_relaxed);
}

MIOPEN_INTERNALS_EXPORT void DbPreloadStates::WaitForRemainingThreadsIfNeeded()
{
    if(!started_loading.load(std::memory_order_relaxed))
        return;

    const auto requesters_left = requesters.fetch_sub(1, std::memory_order_relaxed) - 1;

    if(requesters_left > 0 || threads_left.load(std::memory_order_relaxed) <= 0 || !finish_marker)
        return;

    std::lock_guard<std::mutex> lock(mutex);

    if(!finish_marker)
        return;

    finish_marker->wait();
    finish_marker.reset();
}

MIOPEN_INTERNALS_EXPORT auto DbPreloadStates::GetPreloadedRamDb(const fs::path& path)
    -> std::unique_ptr<RamDb>
{
    return GetPreloadedDb<RamDb>(path);
}

MIOPEN_INTERNALS_EXPORT auto DbPreloadStates::GetPreloadedReadonlyRamDb(const fs::path& path)
    -> std::unique_ptr<ReadonlyRamDb>
{
    return GetPreloadedDb<ReadonlyRamDb>(path);
}

} // namespace miopen
