/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <miopen/conv/heur/memref.hpp>
#include <miopen/conv/heur/metadata.hpp>


#include <math.h>

namespace miopen
{
struct ConvHeur
{
    void TransformFeatures(std::vector<float>& features, const Handle& handle, const ProblemDescription& problem)
    {
        const auto& mu  = GetMu(handle, problem);
        const auto& sig = GetSigma(handle, problem);
        for(auto idx = 0; idx < features.size(); ++idx)
            features[idx] = (features[idx] - mu[idx]) / sig[idx];
    }
    bool IsApplicable(const Handle& /*handle*/, const ProblemDescription& problem)
    {
        const auto& p = problem.conv_problem;
        if(!problem.Is2d())
            return false;
        if(!problem.IsFp32())
            return false;
        if(p.GetGroupCount() != 1)
            return false;
        if(p.GetInLayout() != "NCHW")
            return false;
        return true;
    }
    std::vector<uint64_t> Estimate(const Handle& handle, const ProblemDescription& problem)
    {
        if(!IsApplicable(handle, problem))
            return {solver::Id::gemm().Value()}; // the fallback for the fallback
        const auto& p = problem.conv_problem;
        // TODO: define the number of features and a static way to ensure the correct ordering of features
        Tensor2D features{1, 11};
        // Note some features are fed after log2, this results in improved accuracy
        features.d = {static_cast<float>(log2(p.GetInBatchSize())), static_cast<float>(p.GetKernelStrideH()), static_cast<float>(p.GetKernelStrideW()), static_cast<float>(p.GetWeightsHeight()), static_cast<float>(p.GetWeightsWidth()), static_cast<float>(log2(p.GetInChannels())), static_cast<float>(log2(p.GetInHeight())), static_cast<float>(log2(p.GetInWidth())), static_cast<float>(log2(p.GetOutChannels())), static_cast<float>(p.GetPadH()), static_cast<float>(p.GetPadW())};
        TransformFeatures(features.d, handle, problem);
    
        MemRef2D mem_res = CallModel(handle, problem, features);
        const auto solvers = GetSolverMap(handle);
        std::vector<float> res(mem_res.aligned, mem_res.aligned + (mem_res.size0 * mem_res.size1));
        // pointer returned by CallModel is allocated inside the model and needs to be freed once we have a local copy
        delete[]mem_res.aligned;
        std::vector<std::pair<int, float>> sort_res;
        for(auto idx = 0; idx < res.size(); idx++)
            sort_res.push_back({idx, res[idx]});
        const auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) -> bool
        {
            return a.second > b.second;
        };

        std::sort(sort_res.begin(), sort_res.end(), cmp);
        // map idx to solver id and then anysolver
        std::vector<uint64_t> sol;
        for(auto& kinder : sort_res)
        {
            const auto id = kinder.first;
            const auto sol_id = solver::Id{solvers.at(id)};
            sol.push_back(sol_id.Value());
            // sol.push_back(solver::Id{solvers.at(kinder.first)}.GetSolver());
        }
        sol.push_back(solver::Id::gemm().Value()); // incase the predicted solvers are inapplicable
        return sol;
    }
};
} // namespace miopen
#endif
