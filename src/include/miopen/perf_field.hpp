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
#include <miopen/serializable.hpp>

#include <cstddef>
#include <ostream>
#include <string>

namespace miopen {

struct PerfField
{
    std::string algorithm;
    std::string solver_id;
    float time;
    std::size_t workspace;

    bool operator<(const PerfField& p) const { return (time < p.time); }
};

struct FindDbData : solver::Serializable<FindDbData>
{
    float time;
    std::size_t workspace;
    std::string algorithm;

    FindDbData() : time(-1), workspace(-1), algorithm("<invalid>") {}

    FindDbData(float time_, std::size_t workspace_, const std::string& algorithm_)
        : time(time_), workspace(workspace_), algorithm(algorithm_)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.time, "time");
        f(self.workspace, "workspace");
        f(self.algorithm, "algorithm");
    }

    friend std::ostream& operator<<(std::ostream& os, const FindDbData& obj)
    {
        obj.Serialize(os);
        return os;
    }
};

} // namespace miopen

#endif // GUARD_MIOPEN_PERF_FIELD_HPP_
