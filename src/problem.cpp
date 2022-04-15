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

#include <miopen/problem.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/handle.hpp>
#include <miopen/solution.hpp>
#include <miopen/search_options.hpp>

namespace miopen {

std::vector<Solution> Problem::FindSolutions(Handle& handle,
                                             const SearchOptions& options,
                                             std::size_t max_solutions) const
{
    if(!operator_descriptor)
        MIOPEN_THROW(miopenStatusInvalidValue, "Problem operator descriptor has not been set.");

    auto ret = std::vector<Solution>{};

    switch(operator_descriptor->GetPrimitive())
    {
    case solver::Primitive::Convolution:
        ret = FindConvSolutions(handle, options, max_solutions);
        break;
    case solver::Primitive::Activation:
    case solver::Primitive::Batchnorm:
    case solver::Primitive::Pooling:
    default: MIOPEN_THROW(miopenStatusNotImplemented);
    case solver::Primitive::Invalid: MIOPEN_THROW(miopenStatusInvalidValue);
    }

    const auto sorter = [&]() -> std::function<bool(const Solution&, const Solution&)> {
        switch(options.results_order)
        {
        case miopenSearchResultsOrderByTime:
            return [](auto&& l, auto&& r) { return l.GetTime() < r.GetTime(); };
        case miopenSearchResultsOrderByMemory:
            return [](auto&& l, auto&& r) { return l.GetWorkspaceSize() < r.GetWorkspaceSize(); };
        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }
    }();
    std::sort(ret.begin(), ret.end(), sorter);

    return ret;
}

const TensorDescriptor& Problem::GetTensorDescriptorChecked(miopenTensorName_t name,
                                                            const std::string& name_str) const
{
    const auto found = tensor_descriptors.find(name);
    if(found == tensor_descriptors.end())
        MIOPEN_THROW(miopenStatusInvalidValue,
                     "Problem is missing " + name_str + " tensor descriptor.");
    return found->second;
}

conv::ProblemDescription Problem::AsConvolution() const
{
    const auto& conv_desc = *dynamic_cast<ConvolutionDescriptor*>(operator_descriptor.get());

    const auto& x_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    const auto& y_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    const auto conv_dir = static_cast<conv::Direction>(direction);
    return conv_dir == conv::Direction::Forward
               ? conv::ProblemDescription(x_desc, w_desc, y_desc, conv_desc, conv_dir)
               : conv::ProblemDescription(y_desc, w_desc, x_desc, conv_desc, conv_dir);
}

std::vector<Solution> Problem::FindConvSolutions(Handle& handle,
                                                 const SearchOptions& options,
                                                 std::size_t max_solutions) const
{
    auto ret = std::vector<Solution>{};

    const auto& conv_desc = *dynamic_cast<ConvolutionDescriptor*>(operator_descriptor.get());

    if(tensor_descriptors.size() != 3)
        MIOPEN_THROW(miopenStatusInvalidValue,
                     "Convolution problem should have exactly three tensor descriptors.");

    const auto& x_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    const auto& y_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    auto x = handle.Create(x_desc.GetElementSpace());
    auto w = handle.Create(w_desc.GetElementSpace());
    auto y = handle.Create(y_desc.GetElementSpace());

    const auto workspace_max = [&]() {
        switch(direction)
        {
        case miopenProblemDirectionForward:
            if(conv_desc.mode == miopenTranspose)
                return conv_desc.BackwardDataGetWorkSpaceSize(handle, w_desc, x_desc, y_desc);
            return conv_desc.ForwardGetWorkSpaceSize(handle, w_desc, x_desc, y_desc);
        case miopenProblemDirectionBackward:
            if(conv_desc.mode == miopenTranspose)
                return conv_desc.ForwardGetWorkSpaceSize(handle, w_desc, y_desc, x_desc);
            return conv_desc.BackwardDataGetWorkSpaceSize(handle, w_desc, y_desc, x_desc);
        case miopenProblemDirectionBackwardWeight:
            if(conv_desc.mode == miopenTranspose)
                return conv_desc.BackwardWeightsGetWorkSpaceSize(handle, x_desc, y_desc, w_desc);
            return conv_desc.BackwardWeightsGetWorkSpaceSize(handle, y_desc, x_desc, w_desc);
        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }
    }();

    const auto workspace_size = std::min(options.workspace_limit, workspace_max);
    auto workspace            = handle.Create(workspace_size);

    auto find1_solutions = std::vector<miopenConvAlgoPerf_t>{};
    find1_solutions.resize(max_solutions);
    int found;

    switch(direction)
    {
    case miopenProblemDirectionForward: {
        const auto method = conv_desc.mode == miopenTranspose
                                ? &ConvolutionDescriptor::FindConvFwdAlgorithm
                                : &ConvolutionDescriptor::FindConvBwdDataAlgorithm;

        (conv_desc.*method)(handle,
                            x_desc,
                            x.get(),
                            w_desc,
                            w.get(),
                            y_desc,
                            y.get(),
                            max_solutions,
                            &found,
                            find1_solutions.data(),
                            workspace.get(),
                            workspace_size,
                            options.exhaustive_search);
        break;
    }
    case miopenProblemDirectionBackward: {
        const auto method = conv_desc.mode == miopenTranspose
                                ? &ConvolutionDescriptor::FindConvBwdDataAlgorithm
                                : &ConvolutionDescriptor::FindConvFwdAlgorithm;

        (conv_desc.*method)(handle,
                            y_desc,
                            y.get(),
                            w_desc,
                            w.get(),
                            x_desc,
                            x.get(),
                            max_solutions,
                            &found,
                            find1_solutions.data(),
                            workspace.get(),
                            workspace_size,
                            options.exhaustive_search);
        break;
    }
    case miopenProblemDirectionBackwardWeight: {
        decltype(auto) y_desc_ = miopenTranspose ? x_desc : y_desc;
        decltype(auto) y_      = miopenTranspose ? x : y;
        decltype(auto) x_desc_ = miopenTranspose ? y_desc : x_desc;
        decltype(auto) x_      = miopenTranspose ? y : x;

        conv_desc.FindConvBwdWeightsAlgorithm(handle,
                                              y_desc_,
                                              y_.get(),
                                              x_desc_,
                                              x_.get(),
                                              w_desc,
                                              w.get(),
                                              max_solutions,
                                              &found,
                                              find1_solutions.data(),
                                              workspace.get(),
                                              workspace_size,
                                              options.exhaustive_search);
        break;
    }
    }

    ret.reserve(found);

    const auto conv_dir = static_cast<conv::Direction>(direction);
    const auto netcfg   = AsConvolution().BuildConfKey();

    for(auto i = 0; i < found; ++i)
    {
        const auto algo = ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(find1_solutions[i].fwd_algo), conv_dir);

        auto solution = Solution{};
        solution.SetTime(find1_solutions[i].time);
        solution.SetWorkspaceSize(find1_solutions[i].memory);
        solution.SetSolver(handle.GetFound1_0Id(netcfg, AlgorithmName{algo}).value());
        solution.SetProblem(*this);
        ret.emplace_back(std::move(solution));
    }

    return ret;
}

} // namespace miopen
