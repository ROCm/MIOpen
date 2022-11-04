/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_HEUR_HPP
#define GUARD_MIOPEN_HEUR_HPP
#include <miopen/handle.hpp>
#include <miopen/convolution.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/conv/heur/metadata.hpp>
#include <miopen/anyramdb.hpp>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>

namespace miopen {

struct ConvHeur
{
    static const nlohmann::json& GetMetadata(const std::string& arch)
    {
        static const nlohmann::json metadata = nlohmann::json::parse(std::ifstream(
            boost::filesystem::path(GetSystemDbPath() + "/" + arch + "_metadata.model")
                .generic_string()));
        return metadata;
    }

    static const std::vector<std::string>& GetSupportedArchs()
    {
        static const std::vector<std::string> supported_archs = {"gfx908"};
        return supported_archs;
    }

    static bool IsHeurApplicable(const std::string& arch, const conv::ProblemDescription& problem)
    {
        if(problem.GetInLayout() != "NCHW" && problem.GetInLayout() != "NCDHW")
            return false;
        if(problem.GetWeightsHeight() != problem.GetWeightsWidth())
            return false;
        if(problem.GetPadH() != problem.GetPadW())
            return false;
        if(problem.GetKernelStrideH() != problem.GetKernelStrideW())
            return false;
        if(problem.GetDilationH() != 1 || problem.GetDilationW() != 1)
            return false;
        const auto& data_type = problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
            return false;
        const auto& supported_archs = GetSupportedArchs();
        if(std::find(supported_archs.begin(), supported_archs.end(), arch) == supported_archs.end())
            return false;
        MIOPEN_LOG_I2("Heuristic is applicable");
        return true;
    }

    std::vector<uint64_t>
    Estimate(const std::string& arch, const conv::ProblemDescription& problem, bool& cached)
    {
        std::string est_name = ":memory:" + arch;
        auto& db             = AnyRamDb::GetCached(est_name);
        auto db_res          = db.FindRecord(problem);
        if(db_res)
        {
            cached = true;
            MIOPEN_LOG_I2("Cached heuristic result found");
            std::vector<uint64_t> db_sol(db_res->size());
            std::transform(db_res->begin(), db_res->end(), db_sol.begin(), [](boost::any id) {
                return boost::any_cast<uint64_t>(id);
            });
            if(miopen::IsLogging(LoggingLevel::Info2))
            {
                std::stringstream ss;
                for(auto& id : db_sol)
                    ss << solver::Id{id}.ToString() << " ID:" << id << ", ";
                MIOPEN_LOG_I2("Cached solvers: " << ss.str());
            }
            return db_sol;
        }
        MIOPEN_LOG_I2("Evaluating Heuristic");

        std::vector<float> features = {
            static_cast<float>(problem.GetInChannels()),
            static_cast<float>(problem.GetInDepth()),
            static_cast<float>(problem.GetInHeight()),
            static_cast<float>(problem.GetInWidth()),
            static_cast<float>(problem.GetWeightsDepth()),
            static_cast<float>(problem.GetWeightsHeight()),
            static_cast<float>(problem.GetWeightsWidth()),
            static_cast<float>(problem.GetOutChannels()),
            static_cast<float>(problem.GetOutDepth()),
            static_cast<float>(problem.GetOutHeight()),
            static_cast<float>(problem.GetOutWidth()),
            static_cast<float>(problem.GetOutBatchSize()),
            static_cast<float>(problem.GetPadD()),
            static_cast<float>(problem.GetPadH()),
            static_cast<float>(problem.GetPadW()),
            static_cast<float>(problem.GetKernelStrideD()),
            static_cast<float>(problem.GetKernelStrideH()),
            static_cast<float>(problem.GetKernelStrideW()),
            static_cast<float>(problem.GetDilationH()),
            static_cast<float>(problem.GetDilationW()),
            static_cast<float>(GetLayoutMap(problem.GetInLayout(), arch)),
            static_cast<float>(GetPrecisionMap(problem.GetInDataType(), arch)),
            static_cast<float>(GetDirectionMap(problem.GetDirection(), arch)),
            static_cast<float>(problem.GetGroupCount())};

        TransformFeatures(features, arch);
        const auto res             = CallModel(features, arch);
        static const auto& solvers = GetSolverMap(arch);

        std::vector<std::pair<int, float>> sort_res(res.size());
        for(auto idx = 0; idx < res.size(); idx++)
            sort_res[idx] = {idx, res[idx]};
        const auto cmp = [](const std::pair<int, float>& a,
                            const std::pair<int, float>& b) -> bool { return a.second > b.second; };

        std::sort(sort_res.begin(), sort_res.end(), cmp);
        // map idx to solver id and then anysolver
        std::vector<uint64_t> sol;
        std::vector<boost::any> any_sol;
        for(const auto& kinder : sort_res)
        {
            const auto id     = kinder.first;
            const auto sol_id = solver::Id{solvers.at(id)};
            if(!sol_id.IsValid())
            {
                MIOPEN_LOG_I2("Invalid solver " << solvers.at(id) << " removed");
                continue;
            }
            sol.push_back(sol_id.Value());
            any_sol.push_back(sol_id.Value());
        }
        db.StoreRecord(problem, any_sol);
        if(miopen::IsLogging(LoggingLevel::Info2))
        {
            std::stringstream ss;
            for(auto& id : sol)
                ss << solver::Id{id}.ToString() << " ID:" << id << ", ";
            MIOPEN_LOG_I2("Heuristic Result: " << ss.str());
        }
        return sol;
    }
};
} // namespace miopen
#endif
