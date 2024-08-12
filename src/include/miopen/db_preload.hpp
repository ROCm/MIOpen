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

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <variant>

namespace miopen {

using PreloadedDb = std::variant<std::unique_ptr<RamDb>, std::unique_ptr<ReadonlyRamDb>>;

struct DbPreloadStates
{
    DbPreloadStates()                                          = default;
    DbPreloadStates(const DbPreloadStates&)                    = delete;
    auto operator=(const DbPreloadStates&) -> DbPreloadStates& = delete;
    DbPreloadStates(DbPreloadStates&&)                         = delete;
    auto operator=(DbPreloadStates&&) -> DbPreloadStates&      = delete;

    MIOPEN_INTERNALS_EXPORT auto GetPreloadedRamDb(const fs::path& path) -> std::unique_ptr<RamDb>;

    MIOPEN_INTERNALS_EXPORT void WaitForRemainingThreadsIfNeeded();

    MIOPEN_INTERNALS_EXPORT
    auto GetPreloadedReadonlyRamDb(const fs::path& path) -> std::unique_ptr<ReadonlyRamDb>;

    MIOPEN_INTERNALS_EXPORT void
    StartPreloadingDb(const fs::path& path,
                      std::function<PreloadedDb(const fs::path&)>&& preloader);

    MIOPEN_INTERNALS_EXPORT void TryStartPreloadingDbs(const std::function<void()>& preload);

private:
    std::mutex mutex;
    std::unordered_map<fs::path, std::future<PreloadedDb>> futures;
    std::atomic<bool> started_loading{false};

    std::optional<std::promise<void>> finish_source{std::nullopt};
    std::optional<std::future<void>> finish_marker{std::nullopt};
    std::atomic<int> threads_left{0};
    std::atomic<int> requesters{0};

    template <class Db>
    auto GetPreloadedDb(const fs::path& path) -> std::unique_ptr<Db>;
};

template <class Db>
auto MakeDbPreloader(DbKinds db_kind,
                     bool is_system) -> std::function<PreloadedDb(const fs::path&)>;

extern template auto MakeDbPreloader<RamDb>(DbKinds db_kind, bool is_system)
    -> std::function<PreloadedDb(const fs::path&)>;
extern template auto MakeDbPreloader<ReadonlyRamDb>(DbKinds db_kind, bool is_system)
    -> std::function<PreloadedDb(const fs::path&)>;

auto GetDbPreloadStates() -> DbPreloadStates&;

} // namespace miopen
