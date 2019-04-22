/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_PERF_FIELD_HPP_
#define GUARD_MIOPEN_PERF_FIELD_HPP_

#include <miopen/errors.hpp>
#include <miopen/finddb_kernel_cache_key.hpp>
#include <miopen/serializable.hpp>

#include <cstddef>
#include <string>

namespace miopen {

struct PerfField
{
    std::string name;
    std::string solver_id;
    float time;
    std::size_t workspace;

    bool operator<(const PerfField& p) const { return (time < p.time); }
};

struct FindDbData : solver::Serializable<FindDbData>
{
    std::string solver_id;
    float time;
    std::size_t workspace;
    /// kcache_key may have a special value <unused> in network_config. It means that the particular
    /// solver doesn't use kernel cache and doesn't require a validation of built kernel existence.
    FindDbKCacheKey kcache_key;

    FindDbData() : solver_id("<invalid>"), time(-1), workspace(-1) {}

    FindDbData(const std::string& solver_id_,
               float time_,
               std::size_t workspace_,
               const FindDbKCacheKey& kcache_key_)
        : solver_id(solver_id_), time(time_), workspace(workspace_), kcache_key(kcache_key_)
    {
        if(!kcache_key.IsValid())
            MIOPEN_THROW("Invalid kernel cache key: " + kcache_key.algorithm_name + ", " +
                         kcache_key.network_config);
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.solver_id, "solver_id");
        f(self.time, "time");
        f(self.workspace, "workspace");
        f(self.kcache_key.algorithm_name, "kcache_key::algorithm_name");
        f(self.kcache_key.network_config, "kcache_key::network_confing");
    }
};

} // namespace miopen

#endif // GUARD_MIOPEN_PERF_FIELD_HPP_
