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

#include "miopen/db_record.hpp"
#include "miopen/solver.hpp"
#include "miopen/logger.hpp"

#include <ostream>

#define MIOPEN_LOG_E(...) MIOPEN_LOG(miopen::LoggingLevel::Error, __VA_ARGS__)
#define MIOPEN_LOG_I(...) MIOPEN_LOG(miopen::LoggingLevel::Info, __VA_ARGS__)

namespace miopen {
namespace solver {

std::ostream& operator<<(std::ostream& os, const KernelInfo& k)
{
    os << k.kernel_file << ", " << k.kernel_name << " g_wk={ ";
    for(const auto& size : k.g_wk)
        os << size << ' ';
    os << "}, l_wk={ ";
    for(const auto& size : k.l_wk)
        os << size << ' ';
    return os << "} '" << k.comp_options << '\'';
}

ConvSolution Solver::FindSolution(const ConvolutionContext& context, Db& db) const
{
    std::unique_ptr<PerformanceConfig> config = PerformanceConfigImpl();
    MIOPEN_LOG_I("Finding solution: " << SolverId());
    do
    {
        if(!IsSearchable())
        {
            MIOPEN_LOG_I("Not searchable: " << SolverId());
            break;
        }
        if(db.Load(context, SolverId(), *config))
        {
            MIOPEN_LOG_I("Perf Db: record loaded: " << SolverId());
            if(IsValidPerformanceConfigImpl(context, *config))
            {
                return GetSolution(context, *config);
            }
            MIOPEN_LOG_E("Invalid config loaded from Perf Db: " << SolverId() << ": " << *config);
            break;
        }
        MIOPEN_LOG_I("Perf Db: record NOT found: " << SolverId());
        if(context.do_search)
        {
            MIOPEN_LOG_I("Starting search: " << SolverId());
            if(Search(context, *config))
            {
                db.Store(context, SolverId(), *config);
                return GetSolution(context, *config);
            }
            MIOPEN_LOG_E("Search failed: " << SolverId());
        }
        break;
    } while(false);

    InitPerformanceConfigImpl(context, *config);
    return GetSolution(context, *config);
}

} // namespace solver
} // namespace miopen
