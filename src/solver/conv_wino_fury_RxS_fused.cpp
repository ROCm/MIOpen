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

#include <miopen/fusion/solvers.hpp>
#include <miopen/solver.hpp>

namespace miopen {
namespace solver {
namespace fusion {

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoFuryRxSFused<Winodata, Winofilter>::IsApplicable(
    const FusionContext& ctx, const FusionDescription& problem) const
{
    const auto& desc = *problem.fusion_plan_desc;

    if(desc.op_map.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    if(desc.op_map.size() > 1)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;

    // TODO add bias & activation

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto base         = conv::ConvWinoFuryRxS<Winodata, Winofilter>{};
    return base.IsApplicable(ctx, conv_problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoFuryRxSFused<Winodata, Winofilter>::GetWti(const FusionContext& ctx,
                                                         const FusionDescription& problem) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto base         = conv::ConvWinoFuryRxS<Winodata, Winofilter>{};
    return base.GetWti(ctx, conv_problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
size_t
ConvWinoFuryRxSFused<Winodata, Winofilter>::GetWorkspaceSize(const FusionContext& ctx,
                                                             const FusionDescription& problem) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto base         = conv::ConvWinoFuryRxS<Winodata, Winofilter>{};
    return base.GetWorkspaceSize(ctx, conv_problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
bool
ConvWinoFuryRxSFused<Winodata, Winofilter>::MayNeedWorkspace() const
{
    const auto base         = conv::ConvWinoFuryRxS<Winodata, Winofilter>{};
    return base.MayNeedWorkspace();
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoFuryRxSFused<Winodata, Winofilter>::GetSolution(const FusionContext& ctx,
                                                        const FusionDescription& problem) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto base         = conv::ConvWinoFuryRxS<Winodata, Winofilter>{};
    return base.GetSolution(ctx, conv_problem);
}

template struct ConvWinoFuryRxSFused<2, 3>;
// template struct ConvWinoFuryRxSFused<3, 2>;

} // namespace fusion
} // namespace solver
} // namespace miopen
