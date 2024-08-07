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

#include <miopen/db_preload.hpp>

#include <miopen/config.h>

#include <chrono>

namespace miopen {

namespace {
// static variable inside of a function is not thread-safe
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
DbPreloadStates db_preload_states;
} // namespace

auto GetDbPreloadStates() -> DbPreloadStates& { return db_preload_states; }

template <class Db>
auto GetPreloadedDb(const fs::path& path) -> std::unique_ptr<Db>
{
    auto& states = GetDbPreloadStates();

    std::unique_lock<std::mutex> lock{states.mutex, std::defer_lock};

    // Mutex is need to ensure states.futures is not updated while we work
    // so we skip locking if it no more writes can happen
    if(!states.started_loading.load(std::memory_order_relaxed))
        lock.lock();

    auto it = states.futures.find(path);

    if(it == states.futures.end())
        return nullptr;

    auto future = std::move(it->second);
    lock.unlock();

    const auto start = std::chrono::high_resolution_clock::now();
    auto ret         = future.get();
    const auto end   = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I2("GetPreloadedDb time waiting for the db: " << (end - start).count() * .000001f
                                                             << " ms");
    return std::get<std::unique_ptr<Db>>(std::move(ret));
}

template auto GetPreloadedDb<RamDb>(const fs::path& path) -> std::unique_ptr<RamDb>;
template auto GetPreloadedDb<ReadonlyRamDb>(const fs::path& path) -> std::unique_ptr<ReadonlyRamDb>;

} // namespace miopen
