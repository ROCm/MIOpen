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

#include "miopen/datatype.hpp"
#include <miopen/where/problem_description.hpp>
#include <miopen/where/solvers.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace where {

NetworkConfig BackwardProblemDescription::MakeNetworkConfig() const
{
    tensor_view_t<5> input_grad_tv  = get_inner_expanded_tv<5>(inputGradDesc);
    tensor_view_t<5> other_grad_tv  = get_inner_expanded_tv<5>(otherGradDesc);
    tensor_view_t<5> cond_tv        = get_inner_expanded_tv<5>(conditionDesc);
    tensor_view_t<5> output_grad_tv = get_inner_expanded_tv<5>(outputGradDesc);

    input_grad_tv = broadcast_to(input_grad_tv, output_grad_tv);
    other_grad_tv = broadcast_to(other_grad_tv, output_grad_tv);
    cond_tv       = broadcast_to(cond_tv, output_grad_tv);

    auto cond_contig_size        = solver::where::check_broadcasted_contiguous(cond_tv);
    auto input_grad_contig_size  = solver::where::check_broadcasted_contiguous(input_grad_tv);
    auto other_grad_contig_size  = solver::where::check_broadcasted_contiguous(other_grad_tv);
    auto output_grad_contig_size = solver::where::check_broadcasted_contiguous(output_grad_tv);

    auto is_all_contiguous =
        isTensorViewContiguous(input_grad_tv) && isTensorViewContiguous(other_grad_tv) &&
        isTensorViewContiguous(cond_tv) && isTensorViewContiguous(output_grad_tv);

    bool is_all_broadcasted_contiguous = cond_contig_size && output_grad_contig_size &&
                                         input_grad_contig_size && other_grad_contig_size;

    bool is_condition_broadcasted =
        cond_contig_size && ((input_grad_contig_size % cond_contig_size) == 0 ||
                             (other_grad_contig_size % cond_contig_size) == 0);
    auto output_grad_numel = outputGradDesc.GetElementSize();
    auto outputGrad_dtype = miopen::GetDataType(outputGradDesc.GetType());
    auto inputGrad_dtype  = miopen::GetDataType(inputGradDesc.GetType());

    std::ostringstream ss;

    ss << "is_all_contiguous" << is_all_contiguous;
    ss << "is_all_broadcasted_contiguous" << is_all_broadcasted_contiguous;
    ss << "is_condition_broadcasted" << is_condition_broadcasted;
    ss << "inputGrad_dtype" << inputGrad_dtype;
    ss << "outputGrad_dtype" << outputGrad_dtype;
    ss << "outputGrad_numel" << output_grad_numel;
    ss << "cond_contig_size" << cond_contig_size;
    ss << "input_contig_size" << input_grad_contig_size;
    ss << "other_contig_size" << other_grad_contig_size;
    ss << IsAllPacked();

    return NetworkConfig{ss.str()};
}

} // namespace where

} // namespace miopen
