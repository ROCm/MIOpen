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

bool FindDbRecord::enabled = true;

boost::optional<std::string>& FindDbRecord::path_override()
{
    static boost::optional<std::string> data = boost::none;
    return data;
}

std::string FindDbRecord::GetInstalledPath(Handle& handle)
{
    return GetSystemDbPath() + "/" + handle.GetDbBasename() + "." + GetSystemFindDbSuffix() +
           ".fdb.txt";
}

std::string FindDbRecord::GetUserPath(Handle& handle)
{
    return GetUserDbPath() + "/" + handle.GetDbBasename() + "." + GetUserDbSuffix() + ".ufdb.txt";
}

bool FindDbRecord::CopyValidating(Handle& handle, std::vector<PerfField>& to) const
{
    auto unbuilt = false;
    auto any     = false;

    for(const auto& pair : content->As<FindDbData>())
    {
        if(in_sync && !pair.second.kcache_key.IsUnused())
        {
            const auto is_valid = pair.second.kcache_key.IsValid();

            if(!is_valid || !HasKernel(handle, pair.second.kcache_key))
            {
                unbuilt = true;
                LogFindDbItem(is_valid, pair);
                break;
            }

            any = true;
        }
        to.push_back({pair.first, pair.second.solver_id, pair.second.time, pair.second.workspace});
    }

    return !any || unbuilt;
}

void FindDbRecord::LogFindDbItem(bool is_valid,
                                 const std::pair<std::string, FindDbData>& pair) const
{
    const auto log_level = is_valid ? LoggingLevel::Info2 : LoggingLevel::Error;

    MIOPEN_LOG(
        log_level,
        "Kernel cache entry not found for solver <" << pair.first << "::" << pair.second.solver_id
                                                    << "> at network config: "
                                                    << content->GetKey()
                                                    << " and kernel cache key: "
                                                    << pair.second.kcache_key.algorithm_name
                                                    << ", "
                                                    << pair.second.kcache_key.network_config);

    for(const auto& pair2 : content->As<FindDbData>())
        MIOPEN_LOG(log_level,
                   "Find-db record content: <" << pair2.first << "::" << pair2.second.solver_id
                                               << "> at network config: "
                                               << pair2.second.kcache_key.network_config
                                               << " and algorithm name: "
                                               << pair2.second.kcache_key.algorithm_name);
}

bool FindDbRecord::HasKernel(Handle& handle, const FindDbKCacheKey& key)
{
    return handle.HasKernel(key.algorithm_name, key.network_config);
}

} // namespace miopen
