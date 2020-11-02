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
extern "C" MemRef2D
fwd_gfx906_60_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
struct ConvHeur
{
    void TransformFeatures(std::vector<float>& features, const Handle& handle, const ProblemDescription& problem)
    {
        const auto& mu  = GetMu(handle, problem);
        const auto& sig = GetSig(handle, problem);
        for(auto idx = 0; idx < features.size(); ++idx)
            features[idx] = (features[idx] - mu[idx]) / sig[idx];
    }
    std::vector<solver::AnySolver> Estimate(const Handle& handle, const ProblemDescription& problem)
    {
        const auto& p = problem.conv_problem;
        if(!problem.Is2d())
            return {};
        if(!problem.IsFp32())
            return {};
        if(p.GetGroupCount() != 1)
            return {};
        if(p.GetInLayout() != "NCHW")
            return {};
        // assert(spatial_dim == 2);
        // TODO: define the number of features and a static way to ensure the correct ordering of features
        Tensor2D features{1, 11};
        // Note some features are fed after log2, this results in improved accuracy
        features.d = {static_cast<float>(log2(p.GetInBatchSize())), static_cast<float>(p.GetKernelStrideH()), static_cast<float>(p.GetKernelStrideW()), static_cast<float>(p.GetWeightsHeight()), static_cast<float>(p.GetWeightsWidth()), static_cast<float>(log2(p.GetInChannels())), static_cast<float>(log2(p.GetInHeight())), static_cast<float>(log2(p.GetInWidth())), static_cast<float>(log2(p.GetOutChannels())), static_cast<float>(p.GetPadH()), static_cast<float>(p.GetPadW())};
        TransformFeatures(features.d, handle, problem);
    
        MemRef2D mem_res;
        // TODO: replace with a function that returns this lambda and then call the lambda
        // TODO: free memory when the object is destroyed
        if(handle.GetDeviceName() == "gfx906" && handle.GetMaxComputeUnits() == 60 && problem.direction.IsForward())
        {
            mem_res = fwd_gfx906_60_main_graph(features.data(), features.data(), features.offset, features.size0, features.size1, features.stride0, features.stride1);
            // TODO: check the output size matches expectation
            mem_res.print();
        }
        else
            MIOPEN_THROW(miopenStatusNotImplemented);

        const auto solvers = GetSolverMap(handle, problem);
        std::vector<float> res(mem_res.aligned, mem_res.aligned + (mem_res.size0 * mem_res.size1)); // TODO: free mem_res
        std::vector<std::pair<int, float>> sort_res;
        for(auto idx = 0; idx < res.size(); idx++)
            sort_res.push_back({idx, res[idx]});
        const auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) -> bool
        {
            return a.second > b.second;
        };

        std::sort(sort_res.begin(), sort_res.end(), cmp);
        // map idx to solver id and then anysolver
        std::vector<solver::AnySolver> sol;
        for(auto& kinder : sort_res)
            sol.push_back(solver::Id{solvers.at(kinder.first)}.GetSolver());

        return sol;
    }
};
} // namespace miopen
#endif
