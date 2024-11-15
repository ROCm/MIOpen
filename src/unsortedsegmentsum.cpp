/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/unsortedsegmentsum.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/unsortedsegmentsum/invoke_params.hpp>
#include <miopen/unsortedsegmentsum/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace UnsortedSegmentSum {

miopenStatus_t UnsortedSegmentSumForward(Handle& handle,
                                         const TensorDescriptor& InputDesc,
                                         ConstData_t Input,
                                         const TensorDescriptor& OutputDesc,
                                         Data_t Output,
                                         const TensorDescriptor& SegmentIdsDesc,
                                         ConstData_t segment_ids,
                                         const uint64_t num_segments)
{
    const auto problem = UnsortedSegmentSum::FwdProblemDescription{
        InputDesc, OutputDesc, SegmentIdsDesc, num_segments};

    const auto invoke_params = [&]() {
        auto tmp           = UnsortedSegmentSum::FwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.InputDesc      = &InputDesc;
        tmp.Input          = Input;
        tmp.OutputDesc     = &OutputDesc;
        tmp.Output         = Output;
        tmp.SegmentIdsDesc = &SegmentIdsDesc;
        tmp.segment_ids    = segment_ids;
        tmp.num_segments   = num_segments;
        return tmp;
    }();

    const auto algo = AlgorithmName{"UnsortedSegmentSumForward"};
    const auto solvers =
        solver::SolverContainer<solver::UnsortedSegmentSum::UnsortedSegmentSumForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t UnsortedSegmentSumBackward(Handle& handle,
                                          const TensorDescriptor& OutputGradDesc,
                                          ConstData_t OutputGrad,
                                          const TensorDescriptor& InputGradDesc,
                                          Data_t InputGrad,
                                          const TensorDescriptor& SegmentIdsDesc,
                                          ConstData_t segment_ids,
                                          const uint64_t num_segments)
{
    const auto problem = UnsortedSegmentSum::BwdProblemDescription{
        OutputGradDesc, InputGradDesc, SegmentIdsDesc, num_segments};

    const auto invoke_params = [&]() {
        auto tmp           = UnsortedSegmentSum::BwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.OutputGradDesc = &OutputGradDesc;
        tmp.OutputGrad     = OutputGrad;
        tmp.InputGradDesc  = &InputGradDesc;
        tmp.InputGrad      = InputGrad;
        tmp.SegmentIdsDesc = &SegmentIdsDesc;
        tmp.segment_ids    = segment_ids;
        tmp.num_segments   = num_segments;
        return tmp;
    }();

    const auto algo = AlgorithmName{"UnsortedSegmentSumBackward"};
    const auto solvers =
        solver::SolverContainer<solver::UnsortedSegmentSum::UnsortedSegmentSumBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace UnsortedSegmentSum

} // namespace miopen
