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

#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
#include <miopen/sqlite_db.hpp>
#else
#include <miopen/readonlyramdb.hpp>
#include <miopen/ramdb.hpp>
#endif

#include <functional>
#include <optional>

namespace miopen {
struct ExecutionContext;

#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
using PerformanceDb = DbTimer<MultiFileDb<SQLitePerfDb, SQLitePerfDb, true>>;
#else
using PerformanceDb = DbTimer<MultiFileDb<ReadonlyRamDb, RamDb, true>>;
#endif

class [[nodiscard]] DbGetter final
{
public:
    explicit DbGetter(std::function<PerformanceDb()>&& init_);

    DbGetter(const DbGetter&) = delete;
    auto operator=(const DbGetter&) -> DbGetter& = delete;

    [[nodiscard]] auto operator()() -> PerformanceDb&;

private:
    std::function<PerformanceDb()> init;
    std::optional<PerformanceDb> db;
};

[[nodiscard]] MIOPEN_INTERNALS_EXPORT auto MakeConvDbGetter(const ExecutionContext& ctx)
    -> DbGetter;
} // namespace miopen
