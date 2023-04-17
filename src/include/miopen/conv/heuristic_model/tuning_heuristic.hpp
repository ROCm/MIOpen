/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef GAURD_MIOPEN_TUNING_HEURISTIC_HPP_
#define GAURD_MIOPEN_TUNING_HEURISTIC_HPP_

#include <miopen/miopen.h>
#include <miopen/conv/context.hpp>
#include <miopen/solver.hpp>
#include <unordered_map>
#include <typeinfo>
#include <string>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING

namespace miopen {
namespace ai {
namespace tuning {

struct PerfTuningModel
{
    struct impl;
    std::unique_ptr<impl> pImpl;

    PerfTuningModel();
    ~PerfTuningModel();
    PerfTuningModel(PerfTuningModel&&) noexcept;
    PerfTuningModel(const std::string& arch, const std::string& solver);
    PerfTuningModel& operator=(PerfTuningModel&&) noexcept;
    bool ModelSetParams(std::function<bool(int, int)> validator,
                        const std::vector<float>& features) const;
    size_t GetNumParams() const;
};

} // namespace tuning
} // namespace ai
} // namespace miopen
#endif
#endif
