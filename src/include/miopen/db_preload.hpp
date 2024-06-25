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

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <variant>

namespace miopen {

using PreloadedDb = std::variant<std::unique_ptr<RamDb>, std::unique_ptr<ReadonlyRamDb>>;

struct DbPreloadStates
{
    std::mutex mutex;
    std::unordered_map<fs::path, std::future<PreloadedDb>> futures;
    bool started_loading;

    DbPreloadStates()                                          = default;
    DbPreloadStates(const DbPreloadStates&)                    = delete;
    auto operator=(const DbPreloadStates&) -> DbPreloadStates& = delete;
    DbPreloadStates(DbPreloadStates&&)                         = delete;
    auto operator=(DbPreloadStates&&) -> DbPreloadStates&      = delete;
};

auto GetDbPreloadStates() -> DbPreloadStates&;

template <class Db>
inline auto GetPreloadedDb(const fs::path& path) -> std::unique_ptr<Db>
{
    auto& states = GetDbPreloadStates();

    std::unique_lock<std::mutex> lock{states.mutex};

    if(!states.started_loading)
        return nullptr;

    auto it = states.futures.find(path);

    if(it == states.futures.end())
        return nullptr;

    auto future = std::move(it->second);
    lock.unlock();

    const auto start = std::chrono::high_resolution_clock::now();
    auto ret         = it->second.get();
    const auto end   = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I2("GetPreloadedDb time waiting for the db: " << (end - start).count() * .000001f
                                                             << " ms");
    return std::get<std::unique_ptr<Db>>(std::move(ret));
}

} // namespace miopen
