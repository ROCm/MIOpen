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

#include <miopen/execution_context.hpp>
#include <miopen/ramdb.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/type_name.hpp>
#include <miopen/db_preload.hpp>

#include <miopen/config.h>

#include <atomic>
#include <chrono>
#include <execution>
#include <future>
#include <mutex>

namespace miopen {
MIOPEN_INTERNALS_EXPORT DbPreloadStates::~DbPreloadStates()
{
    if(preload_thread && preload_thread->joinable())
    {
        preload_stoper.request_stop();
        preload_thread->join();
    }
}

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

    auto future = std::move(it->second);

    if(needs_lock)
        lock.unlock();

    if(!future.valid())
        MIOPEN_THROW(miopenStatusInternalError,
                     "Attempt to reload " + path.string() + " as " + get_type_name<Db>());

    const auto start = std::chrono::high_resolution_clock::now();
    auto ret         = future.get();
    const auto end   = std::chrono::high_resolution_clock::now();
    const auto time  = end - start;
    MIOPEN_LOG_I("GetPreloadedDb time waiting for the db: " << time.count() * .000001f << " ms");
    return std::get<std::unique_ptr<Db>>(std::move(ret));
}

template <class Db>
auto MakeDbPreloader(DbKinds db_kind, bool is_system) -> DbPreloader
{
    if constexpr(std::is_same_v<Db, RamDb>)
    {
        return [=](const stop_token& stop, const fs::path& path) -> PreloadedDb {
            auto db   = std::make_unique<RamDb>(db_kind, path, is_system);
            auto lock = std::unique_lock<LockFile>(db->GetLockFile(), GetDbLockTimeout());
            if(!lock)
                MIOPEN_THROW("Db lock has failed to lock.");
            db->Prefetch(stop);
            return {std::move(db)};
        };
    }
    else
    {
        std::ignore = is_system;

        return [=](const stop_token& stop, const fs::path& path) -> PreloadedDb {
            auto db = std::make_unique<Db>(db_kind, path);
            db->Prefetch(true, stop);
            return {std::move(db)};
        };
    }
}

template auto MakeDbPreloader<RamDb>(DbKinds db_kind, bool is_system) -> DbPreloader;
template auto MakeDbPreloader<ReadonlyRamDb>(DbKinds db_kind, bool is_system) -> DbPreloader;

MIOPEN_INTERNALS_EXPORT void DbPreloadStates::StartPreloadingDb(const fs::path& path,
                                                                DbPreloader&& preloader)
{
    if(path.empty())
        return;

    auto& task = preload_tasks.emplace_back(
        std::bind(std::move(preloader), preload_stoper.get_token(), path));
    futures.emplace(path, task.get_future());
}

MIOPEN_INTERNALS_EXPORT void
DbPreloadStates::TryStartPreloadingDbs(const std::function<void()>& preload)
{
    if(started_loading.load(std::memory_order_relaxed))
        return;

    {
        std::unique_lock<std::mutex> lock(mutex);

        if(started_loading.load(std::memory_order_relaxed))
            return;

        preload_stoper = stop_source();

        preload();

        started_loading.store(true, std::memory_order_relaxed);
        // We have finished updating the map and can allow short-cutting the mutex
    }

    if(!preload_tasks.empty())
    {
        preload_thread = std::thread([tasks = std::move(preload_tasks)]() mutable {
            std::for_each(
                std::execution::par_unseq, tasks.begin(), tasks.end(), [](auto&& task) { task(); });
        });
    }
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
