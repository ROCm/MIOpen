/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/names.hpp>

#include <cmath>
#include <sstream>

namespace miopen {

namespace batchnorm {

bool is_fp16(miopenDataType_t type) { return type == miopenHalf; }

bool is_bfp16(miopenDataType_t type) { return type == miopenBFloat16; }

bool is_fp32(miopenDataType_t type) { return (type == miopenFloat); }

bool is_fp16_or_bfp16(miopenDataType_t type) { return is_fp16(type) || is_bfp16(type); }

bool is_fp32_or_fp64(miopenDataType_t type)
{
    return ((type == miopenFloat) || (type == miopenDouble));
}

bool IsOCLInferTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : both FP16
    bool both_fp16 =
        is_fp16(bn_problem.GetXDesc().GetType()) && is_fp16(bn_problem.GetYDesc().GetType());
    // case 2 : both BF16
    bool both_bfp16 =
        is_bfp16(bn_problem.GetXDesc().GetType()) && is_bfp16(bn_problem.GetYDesc().GetType());
    // case 3 : both FP32
    bool both_fp32 =
        is_fp32(bn_problem.GetXDesc().GetType()) && is_fp32(bn_problem.GetYDesc().GetType());

    // OCL supports mixed fp16, bfp16 and pure fp32
    return ((both_fp16 || both_bfp16 || both_fp32) && is_fp32(bn_problem.GetBnScale().GetType()) &&
            is_fp32(bn_problem.GetBnBias().GetType()) &&
            is_fp32(bn_problem.GetBnSMean().GetType()) &&
            is_fp32(bn_problem.GetBnSVar().GetType()));
}

bool IsCKInferTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : mix type
    return ((is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetYDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnScale().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnBias().GetType()) &&
             is_fp32(bn_problem.GetBnSMean().GetType()) &&
             is_fp32(bn_problem.GetBnSVar().GetType())) ||
            // case 2 : fp32 or fp64
            (is_fp32_or_fp64(bn_problem.GetXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetYDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnScale().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnBias().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSMean().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSVar().GetType())));
}

bool IsOCLFwdTrainTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : both FP16
    bool both_fp16 =
        is_fp16(bn_problem.GetXDesc().GetType()) && is_fp16(bn_problem.GetYDesc().GetType());
    // case 2 : both BF16
    bool both_bfp16 =
        is_bfp16(bn_problem.GetXDesc().GetType()) && is_bfp16(bn_problem.GetYDesc().GetType());
    // case 3 : both FP32
    bool both_fp32 =
        is_fp32(bn_problem.GetXDesc().GetType()) && is_fp32(bn_problem.GetYDesc().GetType());

    // OCL supports mixed fp16, bfp16 and pure fp32
    return ((both_fp16 || both_bfp16 || both_fp32) && is_fp32(bn_problem.GetBnScale().GetType()) &&
            is_fp32(bn_problem.GetBnBias().GetType()) &&
            is_fp32(bn_problem.GetBnSMean().GetType()) &&
            is_fp32(bn_problem.GetBnSVar().GetType()));
}

bool IsCKFwdTrainTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : mix type
    return ((is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetYDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnScale().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnBias().GetType()) &&
             is_fp32(bn_problem.GetBnSMean().GetType()) &&
             is_fp32(bn_problem.GetBnSVar().GetType())) ||
            // case 2 : fp32 or fp64
            (is_fp32_or_fp64(bn_problem.GetXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetYDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnScale().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnBias().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSMean().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSVar().GetType())));
}

bool IsOCLBwdTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : both FP16
    bool all_fp16 = is_fp16(bn_problem.GetXDesc().GetType()) &&
                    is_fp16(bn_problem.GetDXDesc().GetType()) &&
                    is_fp16(bn_problem.GetDYDesc().GetType());
    // case 2 : both BF16
    bool all_bfp16 = is_bfp16(bn_problem.GetXDesc().GetType()) &&
                     is_bfp16(bn_problem.GetDXDesc().GetType()) &&
                     is_bfp16(bn_problem.GetDYDesc().GetType());
    // case 3 : both FP32
    bool all_fp32 = is_fp32(bn_problem.GetXDesc().GetType()) &&
                    is_fp32(bn_problem.GetDXDesc().GetType()) &&
                    is_fp32(bn_problem.GetDYDesc().GetType());

    // OCL supports mixed fp16, bfp16 and pure fp32
    return ((all_fp16 || all_bfp16 || all_fp32) && is_fp32(bn_problem.GetBnScale().GetType()) &&
            is_fp32(bn_problem.GetBnBias().GetType()) &&
            is_fp32(bn_problem.GetBnSMean().GetType()) &&
            is_fp32(bn_problem.GetBnSVar().GetType()));
}

bool IsCKBwdTypeValid(const ProblemDescription& bn_problem)
{
    return ((is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
             bn_problem.GetDXDesc().GetType() == miopenFloat &&
             is_fp16_or_bfp16(bn_problem.GetBnScale().GetType()) &&
             bn_problem.GetDYDesc().GetType() == miopenFloat &&
             bn_problem.GetBnSMean().GetType() == miopenFloat &&
             bn_problem.GetBnSVar().GetType() == miopenFloat) ||
            // case 1 : fp32 or fp64
            (is_fp32_or_fp64(bn_problem.GetXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetDXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnScale().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnBias().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSMean().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSVar().GetType())));
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());
    int d                = 1;
    // dimensions
    ss << c;
    ss << "x" << d << "x" << h << "x" << w;
    ss << "x" << n;
    // layout
    ss << "x" << ComputeInLayout();
    ss << "x" << ComputeOutLayout();
    if(direction == Direction::Backward)
    {
        ss << "x" << ComputeDinLayout();
    }
    // data type
    ss << "x" << GetDataTypeName(xDesc.GetType());
    ss << "x" << GetDataTypeName(yOrDyDesc.GetType());
    ss << "x" << GetDataTypeName(scaleDesc.GetType());
    ss << "x" << GetDataTypeName(biasDesc.GetType());
    ss << "x" << GetDataTypeName(sMeanDesc.GetType());
    ss << "x" << GetDataTypeName(sVarianceDesc.GetType());
    if(direction == Direction::Backward)
    {
        ss << "x" << GetDataTypeName(dxDesc.GetType());
    }
    ss << "x" << IsMix();

    // direction
    ss << "x" << GetDirectionStr();
    // save and running
    if(direction == Direction::ForwardTraining)
    {
        ss << "x" << resultsave;
        ss << "x" << resultrunning;
    }
    if(direction == Direction::Backward)
    {
        ss << "x" << useSaved;
    }
    ss << "x" << GetModeStr();
    ss << "x" << activDesc.GetMode();

    return NetworkConfig{ss.str()};
}

} // namespace batchnorm

} // namespace miopen
