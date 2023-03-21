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

    static std::vector<float> ToFeatures(const std::string& arch,
                                                const conv::ProblemDescription& conv_problem)
    {
        const bool isFwd = conv_problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? conv_problem.GetInChannels() :
                                       conv_problem.GetOutChannels()),
            static_cast<float>(isFwd ? conv_problem.GetInDepth() :
                                       conv_problem.GetOutDepth()),
            static_cast<float>(isFwd ? conv_problem.GetInHeight() :
                                       conv_problem.GetOutHeight()),
            static_cast<float>(isFwd ? conv_problem.GetInWidth() :
                                       conv_problem.GetOutWidth()),
            static_cast<float>(conv_problem.GetWeightsDepth()),
            static_cast<float>(conv_problem.GetWeightsHeight()),
            static_cast<float>(conv_problem.GetWeightsWidth()),
            static_cast<float>(isFwd ? conv_problem.GetOutChannels() :
                                       conv_problem.GetInChannels()),
            static_cast<float>(isFwd ? conv_problem.GetOutDepth() :
                                       conv_problem.GetInDepth()),
            static_cast<float>(isFwd ? conv_problem.GetOutHeight() :
                                       conv_problem.GetInHeight()),
            static_cast<float>(isFwd ? conv_problem.GetOutWidth() :
                                       conv_problem.GetInWidth()),
            static_cast<float>(conv_problem.GetOutBatchSize()),
            static_cast<float>(1), // TunaNet was trained on a dataset of 2D
	   	 		                   // problems where PadD was incorrectly set to 1 
            static_cast<float>(conv_problem.GetPadH()),
            static_cast<float>(conv_problem.GetPadW()),
            static_cast<float>(1), // TunaNet was trained on a dataset of 2D
	   	 		                   // problems where StrideD was incorrectly set to 1 
            static_cast<float>(conv_problem.GetKernelStrideH()),
            static_cast<float>(conv_problem.GetKernelStrideW()),
            static_cast<float>(conv_problem.GetDilationH()),
            static_cast<float>(conv_problem.GetDilationW()),
            static_cast<float>(GetLayoutMap(conv_problem.GetInLayout(), arch)),
            static_cast<float>(GetPrecisionMap(conv_problem.GetInDataType(), arch)),
            static_cast<float>(GetDirectionMap(conv_problem.GetDirection(), arch)),
            static_cast<float>(conv_problem.GetGroupCount())
        };
        return features;
    }

    static bool IsHeurApplicable(const std::string& arch,
                                 const ProblemDescription& problem,
                                 const std::vector<float>& features,
                                 const ConvolutionContext& ctx)
    {
        const auto& conv_problem = problem.conv_problem;
        if(!conv_problem.Is2d())
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Problem not 2D (failed)");
            return false;
        }
        if(conv_problem.GetGroupCount() != 1)
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Group count not 1 (failed)");
            return false;
        }
        if(conv_problem.GetInLayout() != "NCHW" && conv_problem.GetInLayout() != "NCDHW")
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Layout not supported (failed)");
            return false;
        }
        if(conv_problem.GetWeightsHeight() != conv_problem.GetWeightsWidth())
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Filters must be square (fil_h == fil_w) (failed)");
            return false;
        }
        if(conv_problem.GetPadH() != conv_problem.GetPadW())
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Padding must be equal along all axes (failed)");
            return false;
        }
        if(conv_problem.GetKernelStrideH() != conv_problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Stride must be equal along all axes (failed)");
            return false;
        }
        if(conv_problem.GetDilationH() != 1 || conv_problem.GetDilationW() != 1)
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Dilation must be 1 (failed)");
            return false;
        }
        const auto& data_type = conv_problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: Unsupported precision (failed)");
            return false;
        }
        const auto& supported_archs = GetSupportedArchs();
        if(std::find(supported_archs.begin(), supported_archs.end(), arch) == supported_archs.end())
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: GPU not supported (failed)");
            return false;
        }

        // check for outlier configurations where no solver the Heuristic predicts is applicable
        const auto& solver_map    = GetSolverMap(arch);
        size_t applicable_solvers = 0;
        for(const auto& solver_name : solver_map)
        {
            auto solver_id = solver::Id{solver_name.second};
            auto solver    = solver_id.GetSolver();
            if(solver.IsApplicable(ctx))
            {
                applicable_solvers++;
                break;
            }
        }
        if(applicable_solvers == 0)
        {
            MIOPEN_LOG_I2("Heuristic Inapplicable: No solver the heuristic may predict applies (failed)");
            return false;
        }

        // The given problem must be within 3 st. deviations of average problem that the
        // heuristic was trained/evaluated on. Otherwise, the problem is substantially
        // different from the problems the heuristic was trained to handle
        const static std::vector<std::string> feature_names = GetFeatureNames(arch);
        const static std::vector<float> centroids = GetStat("mean", arch);
        const static std::vector<float> deviations = GetStat("std", arch);
        for(size_t i = 0; i < features.size(); ++i) {
            if ((features[i] > centroids[i] + 3 * deviations[i]) ||
                (features[i] < centroids[i] - 3 * deviations[i]))
            {
                std::cout << feature_names[i] << ": ";
                std::cout << centroids[i] - 3 * deviations[i] << " < " << features[i] << " < " << centroids[i] + 3 * deviations[i];
                std::cout << "\n";
                MIOPEN_LOG_I2("Heuristic Inapplicable: Problem is out-of-distribution. (failed)");
                return false;
            }
        }

        MIOPEN_LOG_I2("Heuristic is applicable");
        return true;
    }

    std::vector<uint64_t>
    Estimate(const std::string& arch,
             const conv::ProblemDescription& problem,
             std::vector<float>& features,
             bool& cached)
    {
        std::string est_name = ":memory:" + arch;
        auto& db             = AnyRamDb::GetCached(est_name);
        auto db_res          = db.FindRecord(problem);
        if(db_res)
        {
            cached = true;
            MIOPEN_LOG_I2("Cached heuristic result found");
            std::vector<uint64_t> db_sol(db_res->size());
            // cast returned record to solver ids
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

        TransformFeatures(features, arch);
        std::vector<float> res     = CallModel(features, arch);
        static const auto& solvers = GetSolverMap(arch);

        std::vector<std::pair<int, float>> sort_res(res.size());
        // sorts result based upon magnitude of result in vector, returned from Model,
        // paired with original index (idx). Sort magnitudes in descending order.
        // Greater magnitude = better solver. Indexes (idx), which will be used to map to solvers,
        // with greater corresponding magnitude are at front of the vector so they get priority.
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
