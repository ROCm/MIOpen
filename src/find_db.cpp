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

bool testing_find_db_enabled = true; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

boost::optional<std::string>& testing_find_db_path_override()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static boost::optional<std::string> data = boost::none;
    return data;
}

template <class TDb>
std::string FindDbRecord_t<TDb>::GetInstalledPath(Handle& handle)
{
#if !MIOPEN_DISABLE_SYSDB
    return GetSystemDbPath() + "/" + handle.GetDbBasename() + "." + GetSystemFindDbSuffix() +
           ".fdb.txt";
#else
    (void)(handle);
    return "";
#endif
}

template <class TDb>
std::string FindDbRecord_t<TDb>::GetUserPath(Handle& handle)
{
#if !MIOPEN_DISABLE_USERDB
    return GetUserDbPath() + "/" + handle.GetDbBasename() + "." + GetUserDbSuffix() + ".ufdb.txt";
#else
    (void)(handle);
    return "";
#endif
}

bool CheckInvokerSupport(const std::string& algo)
{
    return algo == "miopenConvolutionFwdAlgoDirect" ||
           algo == "miopenConvolutionBwdDataAlgoDirect" ||
           algo == "miopenConvolutionBwdWeightsAlgoDirect" ||
           algo == "miopenConvolutionFwdAlgoWinograd" ||
           algo == "miopenConvolutionBwdDataAlgoWinograd" ||
           algo == "miopenConvolutionBwdWeightsAlgoWinograd" ||
           algo == "miopenConvolutionFwdAlgoImplicitGEMM" ||
           algo == "miopenConvolutionBwdDataAlgoImplicitGEMM" ||
           algo == "miopenConvolutionBwdWeightsAlgoImplicitGEMM" ||
           algo == "miopenConvolutionFwdAlgoFFT" || algo == "miopenConvolutionBwdDataAlgoFFT" ||
           algo == "miopenConvolutionFwdAlgoGEMM";
}

template <class TDb>
bool FindDbRecord_t<TDb>::Validate(Handle& handle, const NetworkConfig& config) const
{
    auto unbuilt = false;
    auto any     = false;

    for(const auto& pair : content->As<FindDbData>())
    {
        if(in_sync)
        {
            if(CheckInvokerSupport(pair.first))
            {
                if(!handle.GetInvoker(config, {{pair.second.solver_id}}))
                {
                    unbuilt = true;
                    // This is not an logged as error because no error was detected.
                    // Find wasn't executed yet and invokers were not prepared.
                    LogFindDbItem(pair);
                    break;
                }

                any = true;
                continue;
            }

            // Todo: remove when all finds will use invokers
            if(!pair.second.kcache_key.IsUnused())
            {
                const auto is_valid = pair.second.kcache_key.IsValid();

                if(!is_valid || !HasKernel(handle, pair.second.kcache_key))
                {
                    unbuilt = true;
                    LogFindDbItem(pair, !is_valid);
                    break;
                }

                any = true;
            }
        }
    }

    return !any || unbuilt;
}

template <class TDb>
void FindDbRecord_t<TDb>::CopyTo(std::vector<PerfField>& to) const
{
    const auto range = content->As<FindDbData>();
    std::transform(range.begin(), range.end(), std::back_inserter(to), [](const auto& pair) {
        return PerfField{
            pair.first, pair.second.solver_id, pair.second.time, pair.second.workspace};
    });
}

template <class TDb>
void FindDbRecord_t<TDb>::LogFindDbItem(const std::pair<std::string, FindDbData>& pair,
                                        bool log_as_error) const
{
    const auto log_level = log_as_error ? LoggingLevel::Error : LoggingLevel::Info2;

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

template <class TDb>
bool FindDbRecord_t<TDb>::HasKernel(Handle& handle, const FindDbKCacheKey& key)
{
    return handle.HasKernel(key.algorithm_name, key.network_config);
}

template class FindDbRecord_t<FindDb>;
template class FindDbRecord_t<UserFindDb>;

} // namespace miopen
