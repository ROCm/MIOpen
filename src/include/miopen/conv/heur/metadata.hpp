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

#ifndef GUARD_MIOPEN_METADATA_HPP_
#define GUARD_MIOPEN_METADATA_HPP_

#include <miopen/handle.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/conv/heur/memref.hpp>

#include <unordered_map>
#include <vector>

#define PPCAT_NX(A, B) A##B

#define PPCAT(A, B) PPCAT_NX(A, B)

namespace miopen
{
const std::unordered_map<int, std::string>& GetSolverMap(const Handle& /*handle*/, const ProblemDescription& problem);
const std::vector<float>& GetMu(const Handle& handle, const ProblemDescription& problem);
const std::vector<float>& GetSigma(const Handle& handle, const ProblemDescription& problem);
const std::vector<std::string>& GetFeatureNames(const Handle& /*handle*/);
MemRef2D CallModel(const Handle& handle, const ProblemDescription& problem, Tensor2D& features);
} // namespace miopen
#endif
