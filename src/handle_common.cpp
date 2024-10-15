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

namespace miopen {
namespace {
void PreloadDbPair(DbPreloadStates* states, DbKinds kind, fs::path&& system, fs::path&& user)
{
#if !MIOPEN_DISABLE_SYSDB
    states->StartPreloadingDb(system, MakeDbPreloader<ReadonlyRamDb>(kind, true));
#endif
#if !MIOPEN_DISABLE_USERDB
    states->StartPreloadingDb(user, MakeDbPreloader<RamDb>(kind, false));
#endif
}
} // namespace

void Handle::TryStartPreloadingDbs()
{
    auto const states = GetDbPreloadStates();

    states->TryStartPreloadingDbs([&]() {
        ExecutionContext ctx{this};

        MIOPEN_LOG_I("Preloading dbs");

        // conv find-db
        PreloadDbPair(states.get(),
                      DbKinds::FindDb,
                      FindDbRecord::GetInstalledPath(*this, ""),
                      FindDbRecord::GetUserPath(*this, ""));

        // fusion find-db
        // it uses perf-db from convolution
        PreloadDbPair(states.get(),
                      DbKinds::FindDb,
                      FindDbRecord::GetInstalledPath(*this, "fusion"),
                      FindDbRecord::GetUserPath(*this, "fusion"));

        // conv perf-db
        PreloadDbPair(states.get(), DbKinds::PerfDb, ctx.GetPerfDbPath(), ctx.GetUserPerfDbPath());

        // batchnorm perf-db
        // it doesn't use find-db
        PreloadDbPair(states.get(),
                      DbKinds::PerfDb,
                      ctx.GetPerfDbPath("batchnorm"),
                      ctx.GetUserPerfDbPath("batchnorm"));
    });
}
} // namespace miopen
