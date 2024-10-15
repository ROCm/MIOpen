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

#pragma once

#include <miopen/filesystem.hpp>
#include <miopen/ramdb.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/stop_token.hpp>

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <variant>

namespace miopen {

using PreloadedDb = std::variant<std::unique_ptr<RamDb>, std::unique_ptr<ReadonlyRamDb>>;
using DbPreloader = std::function<PreloadedDb(const stop_token&, const fs::path&)>;

struct DbPreloadStates final
{
    DbPreloadStates() = default;
    MIOPEN_INTERNALS_EXPORT ~DbPreloadStates();
    DbPreloadStates(const DbPreloadStates&) = delete;
    auto operator=(const DbPreloadStates&) -> DbPreloadStates& = delete;
    DbPreloadStates(DbPreloadStates&&)                         = delete;
    auto operator=(DbPreloadStates&&) -> DbPreloadStates& = delete;

    MIOPEN_INTERNALS_EXPORT auto GetPreloadedRamDb(const fs::path& path) -> std::unique_ptr<RamDb>;

    MIOPEN_INTERNALS_EXPORT
    auto GetPreloadedReadonlyRamDb(const fs::path& path) -> std::unique_ptr<ReadonlyRamDb>;

    MIOPEN_INTERNALS_EXPORT void StartPreloadingDb(const fs::path& path, DbPreloader&& preloader);

    MIOPEN_INTERNALS_EXPORT void TryStartPreloadingDbs(const std::function<void()>& preload);

private:
    std::mutex mutex;
    std::unordered_map<fs::path, std::future<PreloadedDb>> futures;
    std::optional<std::thread> preload_thread;
    std::vector<std::packaged_task<PreloadedDb(const stop_token&)>> preload_tasks;
    std::atomic<bool> started_loading{false};
    stop_source preload_stoper{nostopstate};

    template <class Db>
    auto GetPreloadedDb(const fs::path& path) -> std::unique_ptr<Db>;
};

template <class Db>
auto MakeDbPreloader(DbKinds db_kind, bool is_system) -> DbPreloader;

extern template auto MakeDbPreloader<RamDb>(DbKinds db_kind, bool is_system) -> DbPreloader;
extern template auto MakeDbPreloader<ReadonlyRamDb>(DbKinds db_kind, bool is_system) -> DbPreloader;

auto GetDbPreloadStates() -> std::shared_ptr<DbPreloadStates>;

} // namespace miopen
