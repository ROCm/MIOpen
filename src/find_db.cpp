/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/find_db.hpp>

#include <miopen/handle.hpp>
#include <miopen/finddb_kernel_cache_key.hpp>
#include <miopen/logger.hpp>
#include <miopen/perf_field.hpp>

#include <string>
#include <vector>

namespace miopen {

bool FindDb::enabled = true;

std::string FindDb::GetPath(Handle& handle)
{
    return GetFindDbPath() + "/" + handle.GetDbPathFilename() + ".cd.fdb.txt";
}

bool FindDb::CopyValidating(Handle& handle, std::vector<PerfField>& to) const
{
    auto unbuilt = false;
    auto any     = false;

    for(const auto& pair : record->As<FindDbData>())
    {
        if(loaded && !pair.second.kcache_key.IsUnused())
        {
            const auto is_valid = pair.second.kcache_key.IsValid();

            if(!is_valid ||
               !handle.HasKernel(pair.second.kcache_key.algorithm_name,
                                 pair.second.kcache_key.network_config))
            {
                unbuilt = true;

                MIOPEN_LOG(!is_valid ? LoggingLevel::Info2 : LoggingLevel::Error,
                           "Kernel cache entry not found for solver <"
                               << pair.first
                               << "::"
                               << pair.second.solver_id
                               << "> at network config: "
                               << record->GetKey()
                               << " and kernel cache key: "
                               << pair.second.kcache_key.algorithm_name
                               << ", "
                               << pair.second.kcache_key.network_config);

                for(const auto& pair2 : record->As<FindDbData>())
                    MIOPEN_LOG(
                        !is_valid ? LoggingLevel::Info2 : LoggingLevel::Error,
                        "Find-db record content: <" << pair2.first << "::" << pair2.second.solver_id
                                                    << "> at network config: "
                                                    << pair2.second.kcache_key.network_config
                                                    << " and algorithm name: "
                                                    << pair2.second.kcache_key.algorithm_name);

                MIOPEN_LOG(!is_valid ? LoggingLevel::Info2 : LoggingLevel::Error,
                           "Actual network config used: " << record->GetKey());

                break;
            }
        }

        any = true;
        to.push_back({pair.first, pair.second.solver_id, pair.second.time, pair.second.workspace});
    }

    return !any || unbuilt;
}

} // namespace miopen
