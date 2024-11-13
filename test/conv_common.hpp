/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#pragma once
#include "test.hpp"
#include "workspace.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>

#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solution.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/any_solver.hpp>

#include <utility>
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <miopen/stringutils.hpp>
#include "tensor_util.hpp"
#include <miopen/algorithm.hpp>
#include "cpu_conv.hpp"
#include "gpu_conv.hpp"
#include "network_data.hpp"
#include "miopen/find_db.hpp"
#include "cpu_bias.hpp"
#include "random.hpp"

#define TEST_DIRECT_SUPPORTED_CONFIG_ONLY (!MIOPEN_USE_ROCBLAS)

using ExecutionContext       = miopen::ExecutionContext;
using ConvProblemDescription = miopen::conv::ProblemDescription;
using Direction              = miopen::conv::Direction;

bool get_handle_xnack();

#if TEST_DIRECT_SUPPORTED_CONFIG_ONLY
static inline bool is_direct_fwd_bwd_data_supported(miopen::Handle& handle,
                                                    const miopen::ConvolutionDescriptor convDesc,
                                                    const miopen::TensorDescriptor& xDesc,
                                                    const miopen::TensorDescriptor& wDesc,
                                                    const miopen::TensorDescriptor& yDesc)
{
    if(convDesc.GetSpatialDimension() != 2)
        return false;

    // Both Fwd and Bwd shall be supported by Direct. Return false otherwise.
    for(int direction = 1; direction >= 0; --direction)
    {
        const auto dir = static_cast<miopen::conv::Direction>(direction);
        const auto problem =
            (dir == miopen::conv::Direction::Forward)
                ? miopen::conv::ProblemDescription{xDesc, wDesc, yDesc, convDesc, dir}
                : miopen::conv::ProblemDescription{yDesc, wDesc, xDesc, convDesc, dir};
        auto ctx                    = miopen::ExecutionContext{};
        ctx.do_search               = false;
        ctx.disable_perfdb_access   = true;
        ctx.general_compile_options = "";
        ctx.SetStream(&handle);
        problem.SetupFloats(ctx);
        if(FindAllDirectSolutions(ctx, problem, {}).empty())
            return false;
    }
    return true;
}

static inline bool is_direct_bwd_wrw_supported(miopen::Handle& handle,
                                               const miopen::ConvolutionDescriptor convDesc,
                                               const miopen::TensorDescriptor& xDesc,
                                               const miopen::TensorDescriptor& wDesc,
                                               const miopen::TensorDescriptor& yDesc)
{
    if(convDesc.GetSpatialDimension() != 2)
        return false;

    const auto problem = miopen::conv::ProblemDescription{
        yDesc, wDesc, xDesc, convDesc, miopen::conv::Direction::BackwardWeights};
    auto ctx = miopen::ExecutionContext{};

    ctx.do_search               = false;
    ctx.general_compile_options = "";
    ctx.disable_perfdb_access   = true;
    ctx.SetStream(&handle);
    problem.SetupFloats(ctx);

    return !FindAllBwdWrW2DSolutions(ctx, problem, {}).empty();
}
#endif

inline miopenTensorLayout_t
StringToLayoutType(std::string layout_str, int tensor_vect, int vector_length)
{
    miopenTensorLayout_t default_layout = miopenTensorNCHW;
    if(tensor_vect == 0)
    {
        if(layout_str == "NCHW")
        {
            return miopenTensorNCHW;
        }
        else if(layout_str == "NHWC")
        {
            return miopenTensorNHWC;
        }
        else if(layout_str == "NDHWC")
        {
            return miopenTensorNDHWC;
        }
        else if(layout_str == "NCDHW")
        {
            return miopenTensorNCDHW;
        }
        else
        {
            MIOPEN_THROW("Non-vectorized tensor only support layout NCHW, NHWC, NCDHW and NDHWC");
            return default_layout;
        }
    }
    else if(tensor_vect == 1)
    {
        if(vector_length == 4)
        {
            return layout_str == "CHWN" ? miopenTensorCHWNc4 : miopenTensorNCHWc4;
        }
        else if(vector_length == 8)
        {
            return layout_str == "CHWN" ? miopenTensorCHWNc8 : miopenTensorNCHWc8;
        }
        else
        {
            MIOPEN_THROW("C-vectorized tensor only support vector length 4 and 8");
            return default_layout;
        }
    }
    else
    {
        MIOPEN_THROW("MIOpen only support Non-vectorized and C-vectorized tensor");
        return default_layout;
    }
}

struct conv_stats
{
    std::string solver_name{};
    miopenConvAlgorithm_t algorithm = static_cast<miopenConvAlgorithm_t>(-1);
};

template <class T, class Tout = T>
tensor<Tout> get_output_tensor(const miopen::ConvolutionDescriptor& filter,
                               const tensor<T>& input,
                               const tensor<T>& weights,
                               const std::string& out_layout)
{

    std::string yLayout = out_layout.empty() ? input.desc.GetLayout_str() : out_layout;
    return tensor<Tout>{filter.GetForwardOutputTensorWithLayout(
        input.desc, weights.desc, yLayout, miopen_type<Tout>{})};
}

enum class ConvApi
{
    Invalid,
    Find_1_0,
    Immediate,
    Find_2_0,
};

struct ProblemDestructor
{
    void operator()(miopenProblem_t ptr) { miopenDestroyProblem(ptr); }
};

struct ProblemWrapper : std::unique_ptr<miopenProblem, ProblemDestructor>
{
    ProblemWrapper(miopenConvolutionDescriptor_t conv, miopenProblemDirection_t direction)
    {
        miopenProblem_t value;
        EXPECT_EQUAL(miopenStatusSuccess, miopenCreateConvProblem(&value, conv, direction));
        this->reset(value);
    }
};

struct FindOptionsDestructor
{
    void operator()(miopenFindOptions_t ptr) { miopenDestroyFindOptions(ptr); }
};

struct FindOptionsWrapper : std::unique_ptr<miopenFindOptions, FindOptionsDestructor>
{
    FindOptionsWrapper()
    {
        miopenFindOptions_t value;
        EXPECT_EQUAL(miopenStatusSuccess, miopenCreateFindOptions(&value));
        this->reset(value);
    }
};

// Convolution test base class
//========================================
template <class T, class Tout = T>
struct conv_base
{
    tensor<T> input;
    tensor<T> weights;
    tensor<Tout> out;
    miopen::ConvolutionDescriptor filter;
    int bias{};
    int search{};
    bool enable_fdb{};
    int conv_spatial_dims{};
    conv_stats* stats{}; // Denotes an object after object construction (never nullptr).
    bool preallocate;

    conv_base(bool preallocate_) : preallocate(preallocate_) {}

    void fail(float = 0) const
    {
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Weights tensor: " << weights.desc.ToString() << std::endl;
        std::cout << "Output tensor: " << out.desc.ToString() << std::endl;
        std::cout << "Filter: " << filter << std::endl;
    }

protected:
    void RunFind2_0(miopenProblem_t problem, const miopenTensorArgument_t* arguments) const
    {
        miopenHandle_t handle = &get_handle();

        constexpr const auto find_limit = 1;
        std::size_t found;

        auto solutions = std::vector<miopenSolution_t>{};
        solutions.resize(find_limit);

        {
            const auto options = MakeOptions();

            if(preallocate)
            {
                for(auto i = 0; i < 3; ++i)
                {
                    miopenSetFindOptionPreallocatedTensor(
                        options.get(), arguments[i].id, arguments[i].buffer);
                }
            }

            EXPECT_EQUAL(
                miopenStatusSuccess,
                miopenFindSolutions(
                    handle, problem, options.get(), solutions.data(), &found, solutions.size()));
        }

        EXPECT_OP(found, >=, 0);

        solutions.resize(found);

        for(const auto& solution : solutions)
        {
            auto workspace_size = std::size_t{};
            EXPECT_EQUAL(miopenStatusSuccess,
                         miopenGetSolutionWorkspaceSize(solution, &workspace_size));

            Workspace wspace{workspace_size};

            EXPECT_EQUAL(
                miopenStatusSuccess,
                miopenRunSolution(handle, solution, 3, arguments, wspace.ptr(), wspace.size()));
        }

        const auto& solution_deref = miopen::deref(solutions.front());

        stats->solver_name = solution_deref.GetSolver().ToString();
        stats->algorithm   = solution_deref.GetSolver().GetAlgo();

        for(const auto& solution : solutions)
        {
            EXPECT_EQUAL(miopenStatusSuccess, miopenDestroySolution(solution));
        }
    }

    ProblemWrapper MakeConvProblem(miopenProblemDirection_t direction,
                                   miopenTensorDescriptor_t x,
                                   miopenTensorDescriptor_t w,
                                   miopenTensorDescriptor_t y)
    {
        auto problem = ProblemWrapper{&filter, direction};

        miopenSetProblemTensorDescriptor(problem.get(), miopenTensorConvolutionX, x);
        miopenSetProblemTensorDescriptor(problem.get(), miopenTensorConvolutionW, w);
        miopenSetProblemTensorDescriptor(problem.get(), miopenTensorConvolutionY, y);

        return problem;
    }

private:
    FindOptionsWrapper MakeOptions() const
    {
        auto search_options = FindOptionsWrapper{};

        EXPECT_EQUAL(miopenStatusSuccess, miopenSetFindOptionTuning(search_options.get(), search));

        return search_options;
    }
};

template <typename T, typename Tout = T>
tensor<Tout> ref_conv_fwd(const tensor<T>& input,
                          const tensor<T>& weights,
                          const tensor<Tout>& out,
                          const miopen::ConvolutionDescriptor& filter,
                          const miopen::Scalar& alpha = miopen::Scalar(1.0),
                          const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    auto rout = out;
    if(filter.mode == miopenTranspose)
    {
        std::fill(rout.begin(), rout.end(), static_cast<Tout>(0));
        bool gpu_ref_used = gpu_ref_convolution_bwd(rout, weights, input, filter);
        if(!gpu_ref_used)
        {
            MIOPEN_LOG_W("GPU reference skipped");
            cpu_convolution_backward_data(filter.GetSpatialDimension(),
                                          rout,
                                          weights,
                                          input,
                                          filter.GetConvPads(),
                                          filter.GetConvStrides(),
                                          filter.GetConvDilations(),
                                          filter.GetGroupCount());
        }
    }
    else
    {
        bool gpu_ref_used = gpu_ref_convolution_fwd(input, weights, rout, filter, alpha, beta);

        if(!gpu_ref_used)
        {
            MIOPEN_LOG_W("GPU reference skipped");
            cpu_convolution_forward(filter.GetSpatialDimension(),
                                    input,
                                    weights,
                                    rout,
                                    filter.GetConvPads(),
                                    filter.GetConvStrides(),
                                    filter.GetConvDilations(),
                                    filter.GetGroupCount());
        }
    }
    return rout;
}

template <typename T, typename Twei, typename Tout>
tensor<Twei> ref_conv_wrw(const tensor<T>& input,
                          const tensor<Twei>& weights,
                          const tensor<Tout>& out,
                          const miopen::ConvolutionDescriptor& filter,
                          const miopen::Scalar& alpha = miopen::Scalar(1.0),
                          const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    auto rwei = weights;
    std::fill(rwei.begin(), rwei.end(), 0);
    bool gpu_ref_used = gpu_ref_convolution_wrw(input, rwei, out, filter, alpha, beta);
    if(!gpu_ref_used)
    {
        MIOPEN_LOG_W("GPU reference skipped");
        cpu_convolution_backward_weight(filter.GetSpatialDimension(),
                                        input,
                                        rwei,
                                        out,
                                        filter.GetConvPads(),
                                        filter.GetConvStrides(),
                                        filter.GetConvDilations(),
                                        filter.GetGroupCount());
    }
    return rwei;
}

template <typename T, typename Tout = T>
tensor<Tout> ref_conv_bwd(const tensor<Tout>& input,
                          const tensor<T>& weights,
                          const tensor<T>& out,
                          const miopen::ConvolutionDescriptor& filter,
                          const miopen::Scalar& alpha = miopen::Scalar(1.0),
                          const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    auto rinput = input;

    std::fill(rinput.begin(), rinput.end(), 0);

    if(filter.mode == miopenTranspose)
    {
        bool gpu_ref_used = gpu_ref_convolution_fwd(out, weights, rinput, filter, alpha, beta);
        if(!gpu_ref_used)
        {
            MIOPEN_LOG_W("GPU reference not run");
            cpu_convolution_forward(filter.GetSpatialDimension(),
                                    out,
                                    weights,
                                    rinput,
                                    filter.GetConvPads(),
                                    filter.GetConvStrides(),
                                    filter.GetConvDilations(),
                                    filter.GetGroupCount());
        }
    }
    else
    {
        bool gpu_ref_used = gpu_ref_convolution_bwd(rinput, weights, out, filter, alpha, beta);
        if(!gpu_ref_used)
        {
            MIOPEN_LOG_W("GPU reference not run");
            cpu_convolution_backward_data(filter.GetSpatialDimension(),
                                          rinput,
                                          weights,
                                          out,
                                          filter.GetConvPads(),
                                          filter.GetConvStrides(),
                                          filter.GetConvDilations(),
                                          filter.GetGroupCount());
        }
    }
    return rinput;
}

template <typename T, typename Tout = T>
tensor<Tout> ref_conv_wrw(const tensor<T>& input,
                          const tensor<Tout>& weights,
                          const tensor<T>& out,
                          const miopen::ConvolutionDescriptor& filter,
                          const miopen::Scalar& alpha = miopen::Scalar(1.0),
                          const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    auto rweights = weights;
    std::fill(rweights.begin(), rweights.end(), 0);

    if(filter.mode == miopenTranspose)
    {
        bool gpu_ref_used = gpu_ref_convolution_wrw(out, rweights, input, filter, alpha, beta);
        if(!gpu_ref_used)
        {
            MIOPEN_LOG_W("GPU reference not run");
            cpu_convolution_backward_weight(filter.GetSpatialDimension(),
                                            out,
                                            rweights,
                                            input,
                                            filter.GetConvPads(),
                                            filter.GetConvStrides(),
                                            filter.GetConvDilations(),
                                            filter.GetGroupCount());
        }
    }
    else
    {
        bool gpu_ref_used = gpu_ref_convolution_wrw(input, rweights, out, filter, alpha, beta);
        if(!gpu_ref_used)
        {
            MIOPEN_LOG_W("GPU reference not run");
            cpu_convolution_backward_weight(filter.GetSpatialDimension(),
                                            input,
                                            rweights,
                                            out,
                                            filter.GetConvPads(),
                                            filter.GetConvStrides(),
                                            filter.GetConvDilations(),
                                            filter.GetGroupCount());
        }
    }
    return rweights;
}

// Mainline convolution tests
//========================================
template <ConvApi api, class T, class Tout = T>
struct verify_forward_conv : conv_base<T, Tout>
{
    using conv_base<T, Tout>::input;
    using conv_base<T, Tout>::weights;
    using conv_base<T, Tout>::out;
    using conv_base<T, Tout>::filter;
    using conv_base<T, Tout>::bias;
    using conv_base<T, Tout>::search;
    using conv_base<T, Tout>::stats;

    using conv_base<T, Tout>::RunFind2_0;
    using conv_base<T, Tout>::MakeConvProblem;

    bool is_vect;

    verify_forward_conv(const tensor<T>& pinput,
                        const tensor<T>& pweights,
                        const tensor<Tout>& pout,
                        const miopen::ConvolutionDescriptor& pfilter,
                        conv_stats& pstats,
                        bool preallocate_,
                        int pbias,
                        int psearch,
                        bool pvect)
        : conv_base<T, Tout>(preallocate_)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
        is_vect = pvect;
        stats   = &pstats;
    }

    tensor<Tout> cpu() const
    {
        auto rout = ref_conv_fwd(input, weights, out, filter);
        if(filter.mode != miopenTranspose)
        {
            bool is_int8   = weights.desc.GetType() == miopenInt8;
            bool is_vect_c = weights.desc.GetVectorLength() > 1;
            rout.par_for_each([&](auto... is) {
                if(is_int8 && !is_vect_c)
                {
                    rout(is...) = Tout(double(rout(is...)) + double(this->bias));
                }
                else if(is_vect_c)
                {
                    for(std::size_t i = 0; i < weights.desc.GetVectorLength(); i++)
                        rout(i, is...) = double(rout(i, is...)) + double(this->bias);
                }
                else
                    rout(is...) = double(rout(is...)) + double(this->bias);
            });
        }

        return rout;
    }

    tensor<Tout> gpu()
    {
        auto&& handle = get_handle();
        auto rout     = out;

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(rout.data);

        /// \section read_solver_name
        /// Find() updates find-db with the most recent information (unless find-db is
        /// disabled). Therefore, after Find(), Immediate mode returns the "best" found solution
        /// as the 1st solution in the list, and we can use Immediate mode to find out
        /// the name of the Solver selected during Find() and then used in Run().
        /// So we use one Immediate mode call during Find mode tests,
        /// to print solver name onto console.
        miopenConvSolution_t selected = {};
        bool fallback_path_taken      = false;
        std::size_t count             = 0;

        Workspace wspace{};

        const auto ctx     = ExecutionContext{&handle};
        const auto problem = ConvProblemDescription{
            input.desc,
            weights.desc,
            rout.desc,
            filter,
            filter.mode != miopenTranspose ? Direction::Forward : Direction::BackwardData};

        switch(api)
        {
        case ConvApi::Immediate:
            if(filter.mode == miopenTranspose)
            {
                if(miopen::debug::testing_find_db_enabled)
                {
                    int ret_algo_count;
                    miopenConvAlgoPerf_t perf;
                    wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

                    filter.FindConvBwdDataAlgorithm(handle,
                                                    input.desc,
                                                    in_dev.get(),
                                                    weights.desc,
                                                    wei_dev.get(),
                                                    rout.desc,
                                                    out_dev.get(),
                                                    1,
                                                    &ret_algo_count,
                                                    &perf,
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    search);
                }
                count = filter.GetSolutionCount(ctx, problem);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }

                const auto solutions =
                    filter.GetSolutions(ctx, problem, count, &fallback_path_taken);
                count = solutions.size();

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }
                selected = solutions.front();

                {
                    const std::size_t ws_size = filter.GetBackwardSolutionWorkspaceSize(
                        handle, input.desc, weights.desc, rout.desc, selected.solution_id);
                    if(ws_size != selected.workspace_size)
                    {
                        std::cout << "WARNING: workspace size mismatch: " << selected.workspace_size
                                  << " != " << ws_size << std::endl;
                    }
                }
                wspace.resize(selected.workspace_size);

                filter.CompileSolution(ctx, problem, selected.solution_id);

                filter.ConvolutionBackwardImmediate(handle,
                                                    input.desc,
                                                    in_dev.get(),
                                                    weights.desc,
                                                    wei_dev.get(),
                                                    rout.desc,
                                                    out_dev.get(),
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    selected.solution_id);
            }
            else
            {
                if(miopen::debug::testing_find_db_enabled)
                {
                    int ret_algo_count;
                    miopenConvAlgoPerf_t perf;
                    wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

                    filter.FindConvFwdAlgorithm(handle,
                                                input.desc,
                                                in_dev.get(),
                                                weights.desc,
                                                wei_dev.get(),
                                                rout.desc,
                                                out_dev.get(),
                                                1,
                                                &ret_algo_count,
                                                &perf,
                                                wspace.ptr(),
                                                wspace.size(),
                                                search);
                }

                count = filter.GetSolutionCount(ctx, problem);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }

                // std::cout << "Forward Conv solutions available: " << count << std::endl;
                auto solutions = filter.GetSolutions(ctx, problem, count, &fallback_path_taken);
                count          = solutions.size();

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }
                selected = std::move(solutions.front());

                {
                    const std::size_t ws_size = filter.GetForwardSolutionWorkspaceSize(
                        handle, weights.desc, input.desc, rout.desc, selected.solution_id);
                    if(ws_size != selected.workspace_size)
                    {
                        std::cout << "WARNING: workspace size mismatch: " << selected.workspace_size
                                  << " != " << ws_size << std::endl;
                    }
                }
                wspace.resize(selected.workspace_size);

                filter.CompileSolution(ctx, problem, selected.solution_id);

                filter.ConvolutionForwardImmediate(handle,
                                                   weights.desc,
                                                   wei_dev.get(),
                                                   input.desc,
                                                   in_dev.get(),
                                                   rout.desc,
                                                   out_dev.get(),
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   selected.solution_id);
            }
            break;
        case ConvApi::Find_1_0:
        case ConvApi::Find_2_0:
            if(weights.desc.GetType() == miopenInt8)
            {

                bool is_transform = (input.desc.GetLengths()[1] % 4 != 0 || is_vect);

                std::vector<std::size_t> in_len(input.desc.GetLengths().begin(),
                                                input.desc.GetLengths().end());
                std::vector<std::size_t> wei_len(weights.desc.GetLengths().begin(),
                                                 weights.desc.GetLengths().end());
                in_len[1]  = ((in_len[1] + 3) / 4) * 4;
                wei_len[1] = ((wei_len[1] + 3) / 4) * 4;

                miopen::TensorDescriptor input_vpad_desc(miopenInt8, in_len);
                miopen::TensorDescriptor weight_vpad_desc(miopenInt8, wei_len);

                auto input_vpad   = tensor<T>{in_len};
                auto weights_vpad = tensor<T>{wei_len};
                auto in_vpad_dev  = handle.Write(input_vpad.data);
                auto wei_vpad_dev = handle.Write(weights_vpad.data);

                if(is_transform)
                {
                    float aph = 1.0;
                    float bta = 0.0;
                    TransformTensor(handle,
                                    &aph,
                                    input.desc,
                                    in_dev.get(),
                                    &bta,
                                    input_vpad_desc,
                                    in_vpad_dev.get());

                    TransformTensor(handle,
                                    &aph,
                                    weights.desc,
                                    wei_dev.get(),
                                    &bta,
                                    weight_vpad_desc,
                                    wei_vpad_dev.get());
                }

                auto in_desc  = (is_transform ? input_vpad_desc : input.desc);
                auto wei_desc = (is_transform ? weight_vpad_desc : weights.desc);

                auto in_buf  = (is_transform ? in_vpad_dev.get() : in_dev.get());
                auto wei_buf = (is_transform ? wei_vpad_dev.get() : wei_dev.get());

                if(api == ConvApi::Find_1_0)
                {
                    wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

                    int ret_algo_count;
                    miopenConvAlgoPerf_t perf;

                    float alpha = 1, beta = 0;

                    filter.FindConvFwdAlgorithm(handle,
                                                in_desc,
                                                in_buf,
                                                wei_desc,
                                                wei_buf,
                                                rout.desc,
                                                out_dev.get(),
                                                1,
                                                &ret_algo_count,
                                                &perf,
                                                wspace.ptr(),
                                                wspace.size(),
                                                search);

                    if(perf.memory > 0)
                    {
                        wspace.resize(perf.memory);
                    }

                    filter.ConvolutionForward(handle,
                                              &alpha,
                                              in_desc,
                                              in_buf,
                                              wei_desc,
                                              wei_buf,
                                              perf.fwd_algo,
                                              &beta,
                                              rout.desc,
                                              out_dev.get(),
                                              wspace.ptr(),
                                              wspace.size());

                    /// \ref read_solver_name
                    auto solutions = filter.GetSolutions(ctx, problem, 1, &fallback_path_taken);
                    count          = solutions.size();

                    if(count > 0)
                        selected = std::move(solutions.front());
                }
                else if(api == ConvApi::Find_2_0)
                {
                    const auto f2_problem = MakeConvProblem(
                        miopenProblemDirectionForward, &in_desc, &wei_desc, &rout.desc);

                    const miopenTensorArgument_t arguments[3] = {
                        {miopenTensorConvolutionX, nullptr, in_buf},
                        {miopenTensorConvolutionW, nullptr, wei_buf},
                        {miopenTensorConvolutionY, nullptr, out_dev.get()},
                    };

                    RunFind2_0(f2_problem.get(), arguments);
                }
                else
                {
                    MIOPEN_THROW(miopenStatusNotImplemented);
                }
            }
            else
            {
                if(api == ConvApi::Find_1_0)
                {
                    wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

                    int ret_algo_count;
                    miopenConvAlgoPerf_t perf;

                    float alpha = 1, beta = 0;

                    if(filter.mode == miopenTranspose)
                    {
                        filter.FindConvBwdDataAlgorithm(handle,
                                                        input.desc,
                                                        in_dev.get(),
                                                        weights.desc,
                                                        wei_dev.get(),
                                                        rout.desc,
                                                        out_dev.get(),
                                                        1,
                                                        &ret_algo_count,
                                                        &perf,
                                                        wspace.ptr(),
                                                        wspace.size(),
                                                        search);

                        if(perf.memory > 0)
                        {
                            wspace.resize(perf.memory);
                        }

                        filter.ConvolutionBackwardData(handle,
                                                       &alpha,
                                                       input.desc,
                                                       in_dev.get(),
                                                       weights.desc,
                                                       wei_dev.get(),
                                                       perf.bwd_data_algo,
                                                       &beta,
                                                       rout.desc,
                                                       out_dev.get(),
                                                       wspace.ptr(),
                                                       wspace.size());
                    }
                    else
                    {
                        filter.FindConvFwdAlgorithm(handle,
                                                    input.desc,
                                                    in_dev.get(),
                                                    weights.desc,
                                                    wei_dev.get(),
                                                    rout.desc,
                                                    out_dev.get(),
                                                    1,
                                                    &ret_algo_count,
                                                    &perf,
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    search);

                        if(perf.memory > 0)
                        {
                            wspace.resize(perf.memory);
                        }

                        filter.ConvolutionForward(handle,
                                                  &alpha,
                                                  input.desc,
                                                  in_dev.get(),
                                                  weights.desc,
                                                  wei_dev.get(),
                                                  perf.fwd_algo,
                                                  &beta,
                                                  rout.desc,
                                                  out_dev.get(),
                                                  wspace.ptr(),
                                                  wspace.size());
                    }

                    /// \ref read_solver_name
                    auto solutions = filter.GetSolutions(ctx, problem, 1, &fallback_path_taken);
                    count          = solutions.size();

                    if(count > 0)
                        selected = std::move(solutions.front());
                }
                else if(api == ConvApi::Find_2_0)
                {
                    const auto f2_problem = MakeConvProblem(
                        miopenProblemDirectionForward, &input.desc, &weights.desc, &rout.desc);

                    const miopenTensorArgument_t arguments[3] = {
                        {miopenTensorConvolutionX, nullptr, in_dev.get()},
                        {miopenTensorConvolutionW, nullptr, wei_dev.get()},
                        {miopenTensorConvolutionY, nullptr, out_dev.get()},
                    };

                    RunFind2_0(f2_problem.get(), arguments);
                }
                else
                {
                    MIOPEN_THROW(miopenStatusNotImplemented);
                }
            }
            break;
        case ConvApi::Invalid: MIOPEN_THROW(miopenStatusInvalidValue);
        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }

        if(count != 0)
        {
            stats->algorithm   = selected.algorithm;
            stats->solver_name = miopen::solver::Id(selected.solution_id).ToString();
            if(fallback_path_taken)
                stats->solver_name += "_fallback";
        }
        rout.data = handle.Read<Tout>(out_dev, rout.data.size());
        return rout;
    }

    void fail(float = 0) const
    {
        std::cout << "Forward convolution: " << stats->solver_name << std::endl;
        this->conv_base<T, Tout>::fail();
    }
};

template <ConvApi api, class T>
struct verify_backward_conv : conv_base<T>
{
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::out;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;
    using conv_base<T>::stats;

    using conv_base<T>::RunFind2_0;
    using conv_base<T>::MakeConvProblem;

    verify_backward_conv(const tensor<T>& pinput,
                         const tensor<T>& pweights,
                         const tensor<T>& pout,
                         const miopen::ConvolutionDescriptor& pfilter,
                         conv_stats& pstats,
                         bool preallocate_,
                         int pbias,
                         int psearch)
        : conv_base<T>(preallocate_)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
        stats   = &pstats;
    }

    tensor<T> cpu() const
    {
        auto rinput = input;

        std::fill(rinput.begin(), rinput.end(), 0);

        if(filter.mode == miopenTranspose)
        {
            bool gpu_ref_used = gpu_ref_convolution_fwd(out, weights, rinput, filter);
            if(!gpu_ref_used)
            {
                MIOPEN_LOG_W("GPU reference not run");
                cpu_convolution_forward(filter.GetSpatialDimension(),
                                        out,
                                        weights,
                                        rinput,
                                        filter.GetConvPads(),
                                        filter.GetConvStrides(),
                                        filter.GetConvDilations(),
                                        filter.GetGroupCount());
            }
        }
        else
        {
            bool gpu_ref_used = gpu_ref_convolution_bwd(rinput, weights, out, filter);
            if(!gpu_ref_used)
            {
                MIOPEN_LOG_W("GPU reference not run");
                cpu_convolution_backward_data(filter.GetSpatialDimension(),
                                              rinput,
                                              weights,
                                              out,
                                              filter.GetConvPads(),
                                              filter.GetConvStrides(),
                                              filter.GetConvDilations(),
                                              filter.GetGroupCount());
            }
        }
        return rinput;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();
        auto rinput   = input;
        std::fill(rinput.begin(), rinput.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Write(weights.data);
        auto in_dev  = handle.Write(rinput.data);

        Workspace wspace{};

        miopenConvSolution_t selected;
        bool fallback_path_taken = false;
        std::size_t count        = 0;

        const auto ctx     = ExecutionContext{&handle};
        const auto problem = ConvProblemDescription{
            out.desc,
            weights.desc,
            rinput.desc,
            filter,
            filter.mode != miopenTranspose ? Direction::BackwardData : Direction::Forward};

        switch(api)
        {
        case ConvApi::Immediate: {
            wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

            int ret_algo_count;
            miopenConvAlgoPerf_t perf;

            if(filter.mode == miopenTranspose)
            {
                if(miopen::debug::testing_find_db_enabled)
                {
                    filter.FindConvFwdAlgorithm(handle,
                                                out.desc,
                                                out_dev.get(),
                                                weights.desc,
                                                wei_dev.get(),
                                                rinput.desc,
                                                in_dev.get(),
                                                1,
                                                &ret_algo_count,
                                                &perf,
                                                wspace.ptr(),
                                                wspace.size(),
                                                search);
                }
                count = filter.GetSolutionCount(ctx, problem);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }

                // std::cout << "backward transpose Conv solutions available: " << count <<
                // std::endl;
                auto solutions = filter.GetSolutions(ctx, problem, count, &fallback_path_taken);
                count          = solutions.size();

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }
                std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
                    return l.time < r.time;
                });
                selected = std::move(solutions.front());

                [[maybe_unused]] std::size_t ws_size = filter.GetForwardSolutionWorkspaceSize(
                    handle, weights.desc, out.desc, rinput.desc, selected.solution_id);

                filter.CompileSolution(ctx, problem, selected.solution_id);

                if(selected.workspace_size > 0)
                {
                    wspace.resize(selected.workspace_size);
                }

                filter.ConvolutionForwardImmediate(handle,
                                                   weights.desc,
                                                   wei_dev.get(),
                                                   out.desc,
                                                   out_dev.get(),
                                                   rinput.desc,
                                                   in_dev.get(),
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   selected.solution_id);
            }
            else
            {
                if(miopen::debug::testing_find_db_enabled)
                {
                    filter.FindConvBwdDataAlgorithm(handle,
                                                    out.desc,
                                                    out_dev.get(),
                                                    weights.desc,
                                                    wei_dev.get(),
                                                    rinput.desc,
                                                    in_dev.get(),
                                                    1,
                                                    &ret_algo_count,
                                                    &perf,
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    search);
                }
                count = filter.GetSolutionCount(ctx, problem);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }

                // std::cout << "Backward Conv solutions available: " << count << std::endl;
                auto solutions = filter.GetSolutions(ctx, problem, count, &fallback_path_taken);
                count          = solutions.size();

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1); // NOLINT (concurrency-mt-unsafe)
                }
                std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
                    return l.time < r.time;
                });
                selected = std::move(solutions.front());

                [[maybe_unused]] std::size_t ws_size = filter.GetBackwardSolutionWorkspaceSize(
                    handle, out.desc, weights.desc, rinput.desc, selected.solution_id);

                filter.CompileSolution(ctx, problem, selected.solution_id);

                if(selected.workspace_size > 0)
                {
                    wspace.resize(selected.workspace_size);
                }

                filter.ConvolutionBackwardImmediate(handle,
                                                    out.desc,
                                                    out_dev.get(),
                                                    weights.desc,
                                                    wei_dev.get(),
                                                    rinput.desc,
                                                    in_dev.get(),
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    selected.solution_id);
            }
            break;
        }
        case ConvApi::Find_1_0: {
            wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

            int ret_algo_count;
            miopenConvAlgoPerf_t perf;

            float alpha = 1, beta = 0;

            if(filter.mode == miopenTranspose)
            {
                filter.FindConvFwdAlgorithm(handle,
                                            out.desc,
                                            out_dev.get(),
                                            weights.desc,
                                            wei_dev.get(),
                                            rinput.desc,
                                            in_dev.get(),
                                            1,
                                            &ret_algo_count,
                                            &perf,
                                            wspace.ptr(),
                                            wspace.size(),
                                            search);

                if(perf.memory > 0)
                {
                    wspace.resize(perf.memory);
                }

                filter.ConvolutionForward(handle,
                                          &alpha,
                                          out.desc,
                                          out_dev.get(),
                                          weights.desc,
                                          wei_dev.get(),
                                          perf.fwd_algo,
                                          &beta,
                                          rinput.desc,
                                          in_dev.get(),
                                          wspace.ptr(),
                                          wspace.size());
            }
            else
            {
                filter.FindConvBwdDataAlgorithm(handle,
                                                out.desc,
                                                out_dev.get(),
                                                weights.desc,
                                                wei_dev.get(),
                                                rinput.desc,
                                                in_dev.get(),
                                                1,
                                                &ret_algo_count,
                                                &perf,
                                                wspace.ptr(),
                                                wspace.size(),
                                                search);

                if(perf.memory > 0)
                {
                    wspace.resize(perf.memory);
                }

                filter.ConvolutionBackwardData(handle,
                                               &alpha,
                                               out.desc,
                                               out_dev.get(),
                                               weights.desc,
                                               wei_dev.get(),
                                               perf.bwd_data_algo,
                                               &beta,
                                               rinput.desc,
                                               in_dev.get(),
                                               wspace.ptr(),
                                               wspace.size());
            }

            /// \ref read_solver_name
            const auto solutions = filter.GetSolutions(ctx, problem, 1, &fallback_path_taken);
            count                = solutions.size();

            if(count > 0)
                selected = std::move(solutions.front());
            break;
        }
        case ConvApi::Find_2_0: {
            const auto f2_problem = MakeConvProblem(
                miopenProblemDirectionBackward, &rinput.desc, &weights.desc, &out.desc);

            const miopenTensorArgument_t arguments[3] = {
                {miopenTensorConvolutionX, nullptr, in_dev.get()},
                {miopenTensorConvolutionW, nullptr, wei_dev.get()},
                {miopenTensorConvolutionY, nullptr, out_dev.get()},
            };

            RunFind2_0(f2_problem.get(), arguments);
            break;
        }
        case ConvApi::Invalid: MIOPEN_THROW(miopenStatusInvalidValue);
        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }

        if(count != 0)
        {
            stats->algorithm   = selected.algorithm;
            stats->solver_name = miopen::solver::Id(selected.solution_id).ToString();
            if(fallback_path_taken)
                stats->solver_name += "_fallback";
        }
        rinput.data = handle.Read<T>(in_dev, rinput.data.size());
        return rinput;
    }

    void fail(float) const
    {
        std::cout << "Backward convolution: " << stats->solver_name << std::endl;
        this->conv_base<T>::fail();
    }
};

template <ConvApi api, class T>
struct verify_backward_weights_conv : conv_base<T>
{
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::out;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;
    using conv_base<T>::stats;

    using conv_base<T>::RunFind2_0;
    using conv_base<T>::MakeConvProblem;

    static constexpr const bool is_conv_wrw_f32 = std::is_same<T, float>::value;

    verify_backward_weights_conv(const tensor<T>& pinput,
                                 const tensor<T>& pweights,
                                 const tensor<T>& pout,
                                 const miopen::ConvolutionDescriptor& pfilter,
                                 conv_stats& pstats,
                                 bool preallocate_,
                                 int pbias,
                                 int psearch)
        : conv_base<T>(preallocate_)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
        stats   = &pstats;
    }

    tensor<T> cpu() const
    {
        auto rweights = weights;
        std::fill(rweights.begin(), rweights.end(), 0);

        if(filter.mode == miopenTranspose)
        {
            bool gpu_ref_used = gpu_ref_convolution_wrw(out, rweights, input, filter);
            if(!gpu_ref_used)
            {
                MIOPEN_LOG_W("GPU reference not run");
                cpu_convolution_backward_weight(filter.GetSpatialDimension(),
                                                out,
                                                rweights,
                                                input,
                                                filter.GetConvPads(),
                                                filter.GetConvStrides(),
                                                filter.GetConvDilations(),
                                                filter.GetGroupCount());
            }
        }
        else
        {
            bool gpu_ref_used = gpu_ref_convolution_wrw(input, rweights, out, filter);
            if(!gpu_ref_used)
            {
                MIOPEN_LOG_W("GPU reference not run");
                cpu_convolution_backward_weight(filter.GetSpatialDimension(),
                                                input,
                                                rweights,
                                                out,
                                                filter.GetConvPads(),
                                                filter.GetConvStrides(),
                                                filter.GetConvDilations(),
                                                filter.GetGroupCount());
            }
        }
        return rweights;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();
        auto rweights = weights;
        std::fill(rweights.begin(), rweights.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Write(rweights.data);
        auto in_dev  = handle.Write(input.data);
        Workspace wspace{};

        miopenConvSolution_t selected;
        bool fallback_path_taken = false;
        std::size_t count        = 0;

        const auto ctx = ExecutionContext{&handle};
        const auto problem =
            ConvProblemDescription{filter.mode != miopenTranspose ? out.desc : input.desc,
                                   rweights.desc,
                                   filter.mode != miopenTranspose ? input.desc : out.desc,
                                   filter,
                                   Direction::BackwardWeights};

        switch(api)
        {
        case ConvApi::Immediate: {
            wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

            int ret_algo_count;
            miopenConvAlgoPerf_t perf;

            if(miopen::debug::testing_find_db_enabled)
            {
                filter.FindConvBwdWeightsAlgorithm(
                    handle,
                    filter.mode == miopenTranspose ? input.desc : out.desc,
                    filter.mode == miopenTranspose ? in_dev.get() : out_dev.get(),
                    filter.mode == miopenTranspose ? out.desc : input.desc,
                    filter.mode == miopenTranspose ? out_dev.get() : in_dev.get(),
                    rweights.desc,
                    wei_dev.get(),
                    1,
                    &ret_algo_count,
                    &perf,
                    wspace.ptr(),
                    wspace.size(),
                    search);
            }

            count = filter.GetSolutionCount(ctx, problem);

            if(count == 0)
            {
                std::cout << "FAILED: Using immediate mode error in GetSolutionCount." << std::endl;
                exit(-1); // NOLINT (concurrency-mt-unsafe)
            }

            // std::cout << "Backward weights conv solutions available: " << count << std::endl;
            auto solutions = filter.GetSolutions(ctx, problem, count, &fallback_path_taken);
            count          = solutions.size();

            if(count == 0)
            {
                std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                          << " Solution count: " << count << std::endl;
                exit(-1); // NOLINT (concurrency-mt-unsafe)
            }
            std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
                return l.time < r.time;
            });
            selected = std::move(solutions.front());

            [[maybe_unused]] std::size_t ws_size = filter.GetWrwSolutionWorkspaceSize(
                handle,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? out.desc : input.desc,
                rweights.desc,
                selected.solution_id);

            filter.CompileSolution(ctx, problem, selected.solution_id);

            if(selected.workspace_size > 0)
            {
                wspace.resize(selected.workspace_size);
            }

            filter.ConvolutionWrwImmediate(
                handle,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? in_dev.get() : out_dev.get(),
                filter.mode == miopenTranspose ? out.desc : input.desc,
                filter.mode == miopenTranspose ? out_dev.get() : in_dev.get(),
                rweights.desc,
                wei_dev.get(),
                wspace.ptr(),
                wspace.size(),
                selected.solution_id);
            break;
        }
        case ConvApi::Find_1_0: {

            wspace.resize(filter.GetWorkSpaceSize(ctx, problem));

            int ret_algo_count;
            miopenConvAlgoPerf_t perf;

            float alpha = 1, beta = 0;
            filter.FindConvBwdWeightsAlgorithm(
                handle,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? in_dev.get() : out_dev.get(),
                filter.mode == miopenTranspose ? out.desc : input.desc,
                filter.mode == miopenTranspose ? out_dev.get() : in_dev.get(),
                rweights.desc,
                wei_dev.get(),
                1,
                &ret_algo_count,
                &perf,
                wspace.ptr(),
                wspace.size(),
                search);

            if(perf.memory > 0)
            {
                wspace.resize(perf.memory);
            }

            filter.ConvolutionBackwardWeights(
                handle,
                &alpha,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? in_dev.get() : out_dev.get(),
                filter.mode == miopenTranspose ? out.desc : input.desc,
                filter.mode == miopenTranspose ? out_dev.get() : in_dev.get(),
                perf.bwd_weights_algo,
                &beta,
                rweights.desc,
                wei_dev.get(),
                wspace.ptr(),
                wspace.size());

            /// \ref read_solver_name
            const auto solutions = filter.GetSolutions(ctx, problem, 1, &fallback_path_taken);
            count                = solutions.size();

            if(count > 0)
                selected = std::move(solutions.front());

            break;
        }
        case ConvApi::Find_2_0: {
            const auto f2_problem = MakeConvProblem(
                miopenProblemDirectionBackwardWeights, &input.desc, &rweights.desc, &out.desc);

            const miopenTensorArgument_t arguments[3] = {
                {miopenTensorConvolutionX, nullptr, in_dev.get()},
                {miopenTensorConvolutionW, nullptr, wei_dev.get()},
                {miopenTensorConvolutionY, nullptr, out_dev.get()},
            };

            RunFind2_0(f2_problem.get(), arguments);
            break;
        }
        case ConvApi::Invalid: MIOPEN_THROW(miopenStatusInvalidValue);
        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }

        if(count != 0)
        {
            stats->algorithm   = selected.algorithm;
            stats->solver_name = miopen::solver::Id(selected.solution_id).ToString();
            if(fallback_path_taken)
                stats->solver_name += "_fallback";
        }
        rweights.data = handle.Read<T>(wei_dev, rweights.data.size());
        return rweights;
    }

    void fail(float) const
    {
        std::cout << "Backward weights convolution: " << stats->solver_name << std::endl;
        this->conv_base<T>::fail();
    }
};
template <class T>
struct verify_forward_conv_int8 : conv_base<T>
{
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;
    using conv_base<T>::stats;
    bool is_vect;

    verify_forward_conv_int8(const tensor<T>& pinput,
                             const tensor<T>& pweights,
                             const miopen::ConvolutionDescriptor& pfilter,
                             conv_stats& pstats,
                             int pbias   = 0,
                             int psearch = 0,
                             bool pvect  = false)
    {
        input   = pinput;
        weights = pweights;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
        is_vect = pvect;
        stats   = &pstats;
    }

    tensor<float> cpu() const
    {
        auto rout = get_output_tensor_int8(filter, input, weights);

        if(filter.mode == miopenConvolution)
        {
            bool gpu_ref_used = gpu_ref_convolution_fwd(input, weights, rout, filter);

            if(!gpu_ref_used)
            {
                MIOPEN_LOG_W("GPU reference skipped");
                cpu_convolution_forward(filter.GetSpatialDimension(),
                                        input,
                                        weights,
                                        rout,
                                        filter.GetConvPads(),
                                        filter.GetConvStrides(),
                                        filter.GetConvDilations(),
                                        filter.GetGroupCount());
            }

            rout.par_for_each(
                [&](auto... is) { rout(is...) = double(rout(is...)) + double(this->bias); });
        }

        return rout;
    }

    tensor<float> gpu() const
    {
        auto&& handle = get_handle();
        auto rout     = get_output_tensor_int8(filter, input, weights);

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(rout.data);

        bool is_transform = (input.desc.GetLengths()[1] % 4 != 0 || is_vect);

        std::vector<std::size_t> in_len(input.desc.GetLengths().begin(),
                                        input.desc.GetLengths().end());
        std::vector<std::size_t> wei_len(weights.desc.GetLengths().begin(),
                                         weights.desc.GetLengths().end());
        in_len[1]  = ((in_len[1] + 3) / 4) * 4;
        wei_len[1] = ((wei_len[1] + 3) / 4) * 4;

        miopen::TensorDescriptor input_vpad_desc(miopenInt8, in_len);
        miopen::TensorDescriptor weight_vpad_desc(miopenInt8, wei_len);

        auto input_vpad   = tensor<T>{in_len};
        auto weights_vpad = tensor<T>{wei_len};
        auto in_vpad_dev  = handle.Write(input_vpad.data);
        auto wei_vpad_dev = handle.Write(weights_vpad.data);

        const auto ctx     = ExecutionContext{&handle};
        const auto problem = ConvProblemDescription{
            is_transform ? weight_vpad_desc : weights.desc,
            is_transform ? input_vpad_desc : input.desc,
            rout.desc,
            filter,
            Direction::Forward,
        };

        if(is_transform)
        {
            float aph = 1.0;
            float bta = 0.0;
            miopen::TransformTensor(
                handle, &aph, input.desc, in_dev.get(), &bta, input_vpad_desc, in_vpad_dev.get());

            miopen::TransformTensor(handle,
                                    &aph,
                                    weights.desc,
                                    wei_dev.get(),
                                    &bta,
                                    weight_vpad_desc,
                                    wei_vpad_dev.get());
        }

        Workspace wspace{filter.GetWorkSpaceSize(ctx, problem)};

        int ret_algo_count;
        miopenConvAlgoPerf_t perf;

        if(miopen::debug::testing_find_db_enabled)
        {
            filter.FindConvFwdAlgorithm(handle,
                                        (is_transform ? input_vpad_desc : input.desc),
                                        (is_transform ? in_vpad_dev.get() : in_dev.get()),
                                        (is_transform ? weight_vpad_desc : weights.desc),
                                        (is_transform ? wei_vpad_dev.get() : wei_dev.get()),
                                        rout.desc,
                                        out_dev.get(),
                                        1,
                                        &ret_algo_count,
                                        &perf,
                                        wspace.ptr(),
                                        wspace.size(),
                                        search);
        }

        auto count = filter.GetSolutionCount(ctx, problem);

        if(count == 0)
        {
            std::cout << "FAILED: Using immediate mode error in GetSolutionCount." << std::endl;
            exit(-1); // NOLINT (concurrency-mt-unsafe)
        }

        // std::cout << "Forward Conv solutions available: " << count << std::endl;
        bool fallback_path_taken = false;
        auto solutions           = filter.GetSolutions(ctx, problem, count, &fallback_path_taken);
        count                    = solutions.size();

        if(count == 0)
        {
            std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                      << " Solution count: " << count << std::endl;
            exit(-1); // NOLINT (concurrency-mt-unsafe)
        }
        std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
            return l.time < r.time;
        });
        auto selected = std::move(solutions.front());

        [[maybe_unused]] std::size_t ws_size =
            filter.GetForwardSolutionWorkspaceSize(handle,
                                                   (is_transform ? weight_vpad_desc : weights.desc),
                                                   (is_transform ? input_vpad_desc : input.desc),
                                                   rout.desc,
                                                   selected.solution_id);

        filter.CompileSolution(ctx, problem, selected.solution_id);

        if(selected.workspace_size > 0)
        {
            wspace.resize(selected.workspace_size);
        }

        filter.ConvolutionForwardImmediate(handle,
                                           (is_transform ? weight_vpad_desc : weights.desc),
                                           (is_transform ? wei_vpad_dev.get() : wei_dev.get()),
                                           (is_transform ? input_vpad_desc : input.desc),
                                           (is_transform ? in_vpad_dev.get() : in_dev.get()),
                                           rout.desc,
                                           out_dev.get(),
                                           wspace.ptr(),
                                           wspace.size(),
                                           selected.solution_id);

        if(count != 0)
        {
            stats->algorithm   = selected.algorithm;
            stats->solver_name = miopen::solver::Id(selected.solution_id).ToString();
            if(fallback_path_taken)
                stats->solver_name += "_fallback";
        }
        rout.data = handle.Read<float>(out_dev, rout.data.size());

        return rout;
    }

    void fail(float = 0) const
    {
        std::cout << "Forward convolution: " << stats->solver_name << std::endl;
        this->conv_base<T>::fail();
    }
};

template <class T, ConvApi api = ConvApi::Find_1_0, class Tout = T>
struct conv_driver : test_driver
{
    tensor<T> input;
    tensor<T> weights;
    miopen::ConvolutionDescriptor filter;
    std::string conv_mode;
    std::string pad_mode;
    std::vector<std::size_t> spatial_dim_elements{};
    std::vector<std::size_t> input_dims{};
    std::vector<std::size_t> weight_tensor_dims{};
    std::vector<std::size_t> filter_dims{};
    std::size_t batch_size{};
    std::size_t input_channels{};
    std::size_t output_channels{};
    std::size_t vector_length{};
    std::size_t tensor_vect{}; // 0: non vectorized, 1: C-vectorized, 2: N-vectorized. keep same
                               // as MIOpenDriver InputFlag "tensor_vect"
    std::string in_layout;
    std::string fil_layout; // keep same as MIOpenDriver argument name
    std::string out_layout;
    std::vector<int> pads_strides_dilations;
    std::vector<int> trans_output_pads;
    int groupCount{};
    bool do_forward          = true;
    bool do_backward_data    = true;
    bool do_backward_weights = true;
    int search               = 0;
    bool gen_float           = false;
    bool enable_fdb          = true;
    std::string output_type  = "";
    bool int8_vectorize      = false;
    bool deterministic       = false;
    bool preallocate         = false;

    std::unordered_map<std::string, miopenConvolutionMode_t> cmode_lookup = {
        {"CONV", miopenConvolution},
        {"TRANS", miopenTranspose},
        {"CONVOLUTION", miopenConvolution},
        {"TRANSPOSE", miopenTranspose}};

    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
        {"DEFAULT", miopenPaddingDefault}};

    std::vector<std::size_t> get_batch_sizes() { return {1, 8, 2, 64, 30, 128, 352, 512}; }

    std::vector<std::vector<std::size_t>> get_2d_spatial_dims()
    {
        return {{14, 14},
                {28, 28},
                {32, 32},
                {7, 7},
                {17, 17},
                {56, 56},
                {55, 55},
                {64, 128},
                {224, 224},
                {1024, 2048},
                {3072, 3072},
                {1, 1},
                {1, 7},
                {7, 1}};
    }

    std::vector<std::vector<std::size_t>> get_2d_filter_dims()
    {
        return {{1, 1}, {3, 3}, {1, 7}, {5, 5}, {7, 1}, {7, 7}, {11, 11}, {2, 2}, {4, 4}};
    }

    std::vector<std::size_t> get_output_channels()
    {
        return {32, 64, 16, 128, 96, 112, 192, 256, 320, 512, 1024};
    }

    std::vector<std::size_t> get_input_channels()
    {
        return {16, 32, 3, 128, 96, 112, 192, 256, 320, 512, 1024};
    }

    std::vector<std::vector<int>> get_2d_pads_strides_dilations()
    {
        return {{0, 0, 1, 1, 1, 1},
                {0, 0, 2, 2, 1, 1},
                {1, 1, 1, 1, 1, 1},
                {1, 1, 2, 2, 1, 1},
                {2, 2, 1, 1, 1, 1},
                {3, 3, 2, 2, 1, 1},
                {0, 0, 1, 1, 2, 2},
                {1, 1, 2, 2, 3, 3},
                {3, 3, 2, 2, 4, 4},
                {0, 0, 1, 1, 1, 2},
                {1, 1, 2, 2, 2, 1}};
    }

    std::vector<std::vector<std::size_t>> get_3d_spatial_dims()
    {
        return {{3, 4, 4},
                {4, 9, 9},
                {3, 14, 14},
                {4, 28, 28},
                {4, 56, 56},
                {4, 161, 700},
                {4, 227, 227},
                {1, 1, 1},
                {1, 2, 2}};
    }

    std::vector<std::vector<std::size_t>> get_3d_filter_dims()
    {
        return {{1, 1, 1},
                {3, 3, 3},
                {3, 5, 5},
                {3, 7, 7},
                {5, 7, 7},
                {3, 11, 11},
                {3, 1, 7},
                {3, 7, 1},
                {3, 5, 20}};
    }

    std::vector<std::vector<int>> get_2d_trans_output_pads() { return {{0, 0}}; }

    std::vector<std::vector<int>> get_3d_pads_strides_dilations()
    {
        return {{0, 0, 0, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 2, 2, 2, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 2, 2, 1, 1, 1},
                {2, 2, 2, 1, 1, 1, 1, 1, 1},
                {3, 3, 3, 2, 2, 2, 1, 1, 1},
                {0, 0, 0, 1, 1, 1, 2, 2, 2},
                {1, 1, 0, 2, 2, 2, 3, 3, 3},
                {3, 3, 3, 2, 2, 2, 4, 4, 4},
                {0, 0, 0, 1, 1, 1, 1, 1, 2},
                {1, 1, 1, 2, 2, 2, 2, 2, 1},
                {2, 2, 2, 1, 1, 1, 4, 4, 3},
                {3, 3, 3, 2, 2, 2, 3, 3, 4}};
    }

    std::vector<std::vector<int>> get_3d_trans_output_pads() { return {{0, 0, 0}}; }

    int get_spatial_dim() const
    {
        for(int i = 2; i < 4; i++)
        {
            if(input_dims.size() == i + 2 and weight_tensor_dims.size() == i + 2 and
               pads_strides_dilations.size() == i * 3ULL and trans_output_pads.size() == i)
                return i;
        }
        std::cout << "FAILED: get_spatial_dim() can't calculate dims count." << std::endl;
        exit(-1); // NOLINT (concurrency-mt-unsafe)
    }

    conv_driver()
    {
        add(conv_mode, "cmode", generate_data({"conv"}));
        add(pad_mode, "pmode", generate_data({"default", "same", "valid"}));
        add(groupCount, "group-count", generate_data({1}));
        add(do_forward, "disable-forward", set_value(false));
        add(do_backward_data, "disable-backward-data", set_value(false));
        add(do_backward_weights, "disable-backward-weights", set_value(false));
        add(search, "search", set_value(1));
        add(gen_float, "generate-float", set_value(true));

        if constexpr(api == ConvApi::Immediate)
            add(enable_fdb, "enable-fdb", generate_data({false, true}));
        else if constexpr(api == ConvApi::Find_2_0)
            add(preallocate, "preallocate", generate_data({false, true}));
    }

    void run()
    {
        if(!input_dims.empty())
            filter.spatialDim = get_spatial_dim();
        else
            filter.spatialDim = filter_dims.size();
        bool is_int8 = (input.desc.GetType() == miopenInt8);

        filter.mode             = cmode_lookup[miopen::ToUpper(conv_mode)];
        filter.paddingMode      = pmode_lookup[miopen::ToUpper(pad_mode)];
        std::size_t spatial_dim = filter.GetSpatialDimension();
        filter.group_count      = std::max(static_cast<int>(groupCount), 1);

        miopenTensorLayout_t input_layout_t =
            StringToLayoutType(in_layout, tensor_vect, vector_length);
        miopenTensorLayout_t weight_layout_t =
            StringToLayoutType(fil_layout, tensor_vect, vector_length);

        if(!input_dims.empty())
        {
            input = tensor<T>{input_layout_t, input_dims}.generate(tensor_elem_gen_integer{17});
            batch_size     = input_dims.at(0);
            input_channels = input_dims.at(1);
            std::copy(input_dims.begin() + 2, input_dims.end(), spatial_dim_elements.begin());
        }
        else if(spatial_dim == 2)
        {
            ///\todo This means input_dims ranged in NCHW way, shall we determine the tensor
            /// dimension via layout string?
            input = tensor<T>{input_layout_t,
                              batch_size,
                              input_channels,
                              spatial_dim_elements.at(0),
                              spatial_dim_elements.at(1)}
                        .generate(tensor_elem_gen_integer{17});
        }
        else if(spatial_dim == 3)
        {
            input = tensor<T>{batch_size,
                              input_channels,
                              spatial_dim_elements.at(0),
                              spatial_dim_elements.at(1),
                              spatial_dim_elements.at(2)}
                        .generate(tensor_elem_gen_integer{17});
        }

        if(!weight_tensor_dims.empty())
        {
            if(fil_layout == "CHWN")
            {
                weights = tensor<T>{weight_layout_t, weight_tensor_dims}.generate(
                    tensor_elem_gen_integer{17});
                output_channels = weight_tensor_dims.at(3);
                std::copy(weight_tensor_dims.begin() + 1,
                          weight_tensor_dims.end() - 1,
                          filter_dims.begin());
            }
            else
            {
                weights = tensor<T>{weight_layout_t, weight_tensor_dims}.generate(
                    tensor_elem_gen_integer{17});
                output_channels = weight_tensor_dims.at(0);
                std::copy(
                    weight_tensor_dims.begin() + 2, weight_tensor_dims.end(), filter_dims.begin());
            }
        }
        else if(spatial_dim == 2)
        {
            if(fil_layout == "NCHW")
            {
                weights = tensor<T>{weight_layout_t,
                                    output_channels,
                                    input_channels / filter.group_count,
                                    filter_dims.at(0),
                                    filter_dims.at(1)}
                              .generate(tensor_elem_gen_integer{17});
            }
            else if(fil_layout == "CHWN")
            {
                weights = tensor<T>{weight_layout_t,
                                    input_channels / filter.group_count,
                                    filter_dims.at(0),
                                    filter_dims.at(1),
                                    output_channels}
                              .generate(tensor_elem_gen_integer{17});
            }
            else
            {
                weights = tensor<T>{output_channels,
                                    input_channels / filter.group_count,
                                    filter_dims.at(0),
                                    filter_dims.at(1)}
                              .generate(tensor_elem_gen_integer{17});
            }
        }
        else if(spatial_dim == 3)
        {
            weights = tensor<T>{output_channels,
                                input_channels / filter.group_count,
                                filter_dims.at(0),
                                filter_dims.at(1),
                                filter_dims.at(2)}
                          .generate(tensor_elem_gen_integer{17});
        }

        if(input.desc.GetNumDims() != in_layout.size() ||
           weights.desc.GetNumDims() != fil_layout.size() ||
           input.desc.GetNumDims() != out_layout.size())
        {
            std::cout << input.desc.GetNumDims() << "," << in_layout.size() << std::endl;
            std::cout << weights.desc.GetNumDims() << "," << fil_layout.size() << std::endl;
            std::cout << input.desc.GetNumDims() << "," << out_layout.size() << std::endl;
            std::cerr << "FAILED: layout not match dimension size!" << std::endl;
            return;
        }

        // reconstruct tensor descriptor(desc) when layout is not the default NCHW layout.
        // by default, this member is constructed when conv2d/3d is constructed (see
        // test_driver::add())
        // but this requires the dimensions come from commandline, which is hard for non-NCHW layout
        if(in_layout != "NCHW" && in_layout != "NCDHW")
        {
            const std::vector<std::size_t> dim_lens = input.desc.GetLengths();
            std::vector<std::size_t> dim_strides;
            miopen::tensor_layout_to_strides(
                dim_lens,
                miopen::tensor_layout_get_default(weights.desc.GetNumDims()),
                in_layout,
                vector_length,
                dim_strides);
            input.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
        }
        if(fil_layout != "NCHW" && fil_layout != "NCDHW" && fil_layout != "CHWN")
        {
            const std::vector<std::size_t> dim_lens = weights.desc.GetLengths();
            std::vector<std::size_t> dim_strides;
            miopen::tensor_layout_to_strides(
                dim_lens,
                miopen::tensor_layout_get_default(weights.desc.GetNumDims()),
                fil_layout,
                vector_length,
                dim_strides);
            weights.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
        }

        if(input.desc.GetNumDims() != 2 + spatial_dim ||
           weights.desc.GetNumDims() != 2 + spatial_dim ||
           pads_strides_dilations.size() != 3 * spatial_dim ||
           trans_output_pads.size() != spatial_dim)
        {
            std::cerr << "FAILED: dimension is wrong!" << std::endl;
            return;
        }

        filter.pads.resize(spatial_dim);
        filter.strides.resize(spatial_dim);
        filter.dilations.resize(spatial_dim);
        filter.trans_output_pads.resize(spatial_dim);
        if(deterministic)
            filter.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);

        std::copy_n(pads_strides_dilations.begin(), spatial_dim, filter.pads.begin());
        std::copy_n(
            pads_strides_dilations.begin() + spatial_dim, spatial_dim, filter.strides.begin());
        std::copy_n(pads_strides_dilations.begin() + 2 * spatial_dim,
                    spatial_dim,
                    filter.dilations.begin());
        std::copy_n(trans_output_pads.begin(), spatial_dim, filter.trans_output_pads.begin());

        std::size_t in_c_len = input.desc.GetLengths()[1];
        std::vector<std::size_t> in_spatial_len(input.desc.GetLengths().begin() + 2,
                                                input.desc.GetLengths().end());
        std::size_t wei_k_len = weights.desc.GetLengths()[0];
        std::size_t wei_c_len = weights.desc.GetLengths()[1];
        std::vector<std::size_t> wei_spatial_len(weights.desc.GetLengths().begin() + 2,
                                                 weights.desc.GetLengths().end());
        if(fil_layout == "CHWN")
        {
            wei_c_len          = weights.desc.GetLengths()[0];
            wei_spatial_len[0] = weights.desc.GetLengths()[1];
            wei_spatial_len[1] = weights.desc.GetLengths()[2];
            wei_k_len          = weights.desc.GetLengths()[3];
        }

        // lack of transposeConv or groupConv for int8 type
        if(is_int8 && filter.mode == miopenTranspose)
        {
            show_command();
            std::cout << "MIOpen doesn't support int8 type transpose convolution." << std::endl;
            return;
        }

        bool is_bfloat16 =
            (input.desc.GetType() == miopenBFloat16 && weights.desc.GetType() == miopenBFloat16);

        if(is_bfloat16 && !(filter.spatialDim == 2))
        {
            show_command();
            std::cout << "Skipped: bfloat16 is supported for 2D conv only" << std::endl;
            return;
        }

        if(((filter.mode == miopenTranspose) &&
            ((filter.group_count == 1 && in_c_len == wei_k_len) ||
             (filter.group_count > 1 && wei_k_len % filter.group_count == 0))) ||
           ((filter.mode == miopenConvolution) &&
            ((filter.group_count == 1 && in_c_len == wei_c_len) ||
             (filter.group_count > 1 && in_c_len % wei_c_len == 0))))
        {
            if(filter.mode == miopenConvolution &&
               (miopen::all_of(filter.GetConvDilations(), [](auto v) { return v == 1; }) ||
                miopen::all_of(wei_spatial_len, [](auto v) { return v == 1; })))
            {
                if(filter.paddingMode == miopenPaddingSame)
                {
                    if(miopen::any_of(filter.GetConvStrides(), [](auto v) { return v == 0; }))
                    {
                        show_command();
                        std::cout << "Skipped: stride[i] == 0" << std::endl;
                        return;
                    }

                    std::vector<std::size_t> pads_(spatial_dim);
                    std::vector<std::ptrdiff_t> out_spatial_len(spatial_dim);

                    for(std::size_t i = 0; i < spatial_dim; ++i)
                    {
                        pads_[i] =
                            (in_spatial_len[i] % filter.GetConvStrides()[i] == 0)
                                ? (std::max(
                                      static_cast<std::ptrdiff_t>(wei_spatial_len[i]) -
                                          static_cast<std::ptrdiff_t>(filter.GetConvStrides()[i]),
                                      static_cast<std::ptrdiff_t>(0)))
                                : (std::max(static_cast<std::ptrdiff_t>(wei_spatial_len[i]) -
                                                static_cast<std::ptrdiff_t>(
                                                    in_spatial_len[i] % filter.GetConvStrides()[i]),
                                            static_cast<std::ptrdiff_t>(0)));

                        filter.pads[i] = pads_[i] / 2;

                        out_spatial_len[i] = miopen::integer_division_ceil(
                            in_spatial_len[i], filter.GetConvStrides()[i]);
                    }

                    if(miopen::any_of(out_spatial_len, [](auto v) { return v <= 0; }))
                    {
                        show_command();
                        std::cout << "Skipped: out_spatial_len[i] <= 0" << std::endl;
                        return;
                    }
                }
                else if(filter.paddingMode == miopenPaddingValid)
                {
                    if(miopen::any_of(filter.GetConvStrides(), [](auto v) { return v == 0; }))
                    {
                        show_command();
                        std::cout << "Skipped: stride[i] == 0" << std::endl;
                        return;
                    }

                    std::vector<ptrdiff_t> out_spatial_len(spatial_dim);

                    for(std::size_t i = 0; i < spatial_dim; ++i)
                    {
                        filter.pads[i] = 0;

                        out_spatial_len[i] = miopen::integer_division_ceil(
                            static_cast<std::ptrdiff_t>(in_spatial_len[i]) -
                                static_cast<std::ptrdiff_t>(wei_spatial_len[i]) + 1,
                            filter.GetConvStrides()[i]);
                    }

                    if(miopen::any_of(out_spatial_len, [](auto v) { return v <= 0; }))
                    {
                        show_command();
                        std::cout << "Skipped: out_spatial_len[i] <= 0" << std::endl;
                        return;
                    }
                }
            }
            if(filter.mode == miopenTranspose)
            {
                for(std::size_t i = 0; i < spatial_dim; ++i)
                {
                    filter.pads[i] = filter.GetConvStrides()[i] - 1;
                }
            }

            if(((filter.mode == miopenTranspose) &&
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(0))) ||
                 (filter.group_count > 1 &&
                  (weights.desc.GetLengths().at(0) % filter.group_count == 0)))) ||
               ((filter.mode == miopenConvolution) &&
                ((weights.desc.GetLayout_str() == "NCHW") ||
                 (weights.desc.GetLayout_str() == "NCHWc")) &&
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1))) ||
                 (filter.group_count > 1 &&
                  (input.desc.GetLengths().at(1) % weights.desc.GetLengths().at(1) == 0)))) ||
               ((filter.mode == miopenConvolution) && (weights.desc.GetLayout_str() == "CHWNc") &&
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(0))) ||
                 (filter.group_count > 1 &&
                  (input.desc.GetLengths().at(1) % weights.desc.GetLengths().at(0) == 0)))))
            {
                auto output = get_output_tensor<T, Tout>(filter, input, weights, out_layout);

                auto gen_positive_value = [=](auto...) {
                    auto data_type = input.desc.GetType();
                    int v_max      = is_int8 ? 16 : (data_type == miopenHalf) ? 4 : 17;
                    return gen_float ? prng::gen_canonical<double>()
                                     : static_cast<double>(prng::gen_A_to_B(1, v_max));
                };

                auto gen_sign_value = [=](auto... is) {
                    auto data_type = input.desc.GetType();
                    int v_max      = is_int8 ? 16 : (data_type == miopenHalf) ? 4 : 17;
                    return gen_float ? prng::gen_A_to_B(-1.0, 1.0)
                                     : static_cast<double>(prng::gen_A_to_B(1, v_max)) *
                                           tensor_elem_gen_checkboard_sign{}(is...);
                };

                auto ctx = miopen::ExecutionContext{&get_handle()};

                bool skip_forward = false;

                bool skip_backward_data    = is_int8;
                bool skip_backward_weights = is_int8;

#if TEST_DIRECT_SUPPORTED_CONFIG_ONLY
                if(input.desc.GetType() == miopenInt8 || input.desc.GetType() == miopenBFloat16)
                {
                    show_command();
                    std::cout << "Direct path doesn't support Int8 or BFloat16 type." << std::endl;
                    return;
                }
                if(input.desc.GetType() == miopenHalf && filter.mode == miopenConvolution)
                {
                    skip_forward = !is_direct_fwd_bwd_data_supported(
                        get_handle(), filter, input.desc, weights.desc, output.desc);

                    skip_backward_data = skip_forward;

                    skip_backward_weights = !is_direct_bwd_wrw_supported(
                        get_handle(), filter, input.desc, weights.desc, output.desc);
                }
#endif

                input.generate(gen_positive_value);
                output.generate(gen_positive_value);
                weights.generate(gen_sign_value);

                size_t total_mem;
                if(is_int8)
                {
                    /// \todo Properly construct the `output` tensor descriptor
                    /// and get rid of this special "int8" stuff.
                    auto output_int8 =
                        get_output_tensor<T, Tout>(filter, input, weights, out_layout);
                    const auto problem        = ConvProblemDescription{input.desc,
                                                                weights.desc,
                                                                std::move(output_int8.desc),
                                                                filter,
                                                                Direction::Forward};
                    const auto workspace_size = filter.GetWorkSpaceSize(ctx, problem);

                    // 4x because assume type is miopenInt8x4
                    total_mem = input.desc.GetNumBytes() + 4 * input.desc.GetNumBytes() +
                                weights.desc.GetNumBytes() + 4 * weights.desc.GetNumBytes() +
                                output_int8.desc.GetNumBytes() + 4 * sizeof(char) * workspace_size;
                }
                else
                {
                    /// \todo Take into account `skip_forward`, `skip_backward_data`,
                    /// `skip_backward_weights` and use this path to compute `total_mem` for int8
                    /// variations.
                    const auto fwd_problem = miopen::conv::ProblemDescription{
                        input.desc,
                        weights.desc,
                        output.desc,
                        filter,
                        filter.mode != miopenTranspose ? Direction::Forward
                                                       : Direction::BackwardData};

                    const auto bwd_problem = miopen::conv::ProblemDescription{
                        output.desc,
                        weights.desc,
                        input.desc,
                        filter,
                        filter.mode != miopenTranspose ? Direction::BackwardData
                                                       : Direction::Forward};

                    const auto wrw_problem = miopen::conv::ProblemDescription{
                        filter.mode != miopenTranspose ? output.desc : input.desc,
                        weights.desc,
                        filter.mode != miopenTranspose ? input.desc : output.desc,
                        filter,
                        Direction::BackwardWeights};

                    const auto workspaces = std::array<std::size_t, 3>{
                        filter.GetWorkSpaceSize(ctx, fwd_problem),
                        filter.GetWorkSpaceSize(ctx, bwd_problem),
                        filter.GetWorkSpaceSize(ctx, wrw_problem),
                    };

                    const auto workspace_size =
                        *std::max_element(workspaces.begin(), workspaces.end());

                    total_mem = input.desc.GetNumBytes() + weights.desc.GetNumBytes() +
                                output.desc.GetNumBytes() +
                                sizeof(char) * workspace_size; // estimate based on backward pass
                }

                size_t device_mem = get_handle().GetGlobalMemorySize();

                if(total_mem >= device_mem)
                {
                    show_command();
                    std::cout << "Config requires " << total_mem
                              << " Bytes to write all necessary tensors to GPU. GPU has "
                              << device_mem << " Bytes of memory." << std::endl;
                    return;
                }

                if(api == ConvApi::Immediate)
                {
                    miopen::debug::testing_find_db_enabled = enable_fdb;
                }

                conv_stats stats;

                if(do_forward && !skip_forward)
                {
                    if(is_int8)
                    {
                        if(output_type == "float")
                        {
                            verify(verify_forward_conv<api, T, float>{
                                input,
                                weights,
                                get_output_tensor<T, float>(filter, input, weights, out_layout),
                                filter,
                                stats,
                                preallocate,
                                0,
                                search,
                                int8_vectorize});
                        }
                        else if(output_type == "int32")
                        {
                            verify(verify_forward_conv<api, T, int>{
                                input,
                                weights,
                                get_output_tensor<T, int>(filter, input, weights, out_layout),
                                filter,
                                stats,
                                preallocate,
                                0,
                                search,
                                int8_vectorize});
                        }
                        else if(output_type == "int8")
                        {
                            verify(verify_forward_conv<api, T, int8_t>{
                                input,
                                weights,
                                get_output_tensor<T, int8_t>(filter, input, weights, out_layout),
                                filter,
                                stats,
                                preallocate,
                                0,
                                search,
                                int8_vectorize});
                        }
                        else
                        {
                            show_command();
                            std::cout << "FAILED: bad output_type: '" << output_type << '\''
                                      << std::endl;
                        }
                    }
                    else
                    {
                        verify(verify_forward_conv<api, T>{
                            input, weights, output, filter, stats, preallocate, 0, search, false});
                    }
                }

                if(do_backward_data && !skip_backward_data)
                {
                    verify(verify_backward_conv<api, T>{
                        input, weights, output, filter, stats, preallocate, 0, search});
                }

                if(do_backward_weights && !skip_backward_weights)
                {
                    output.generate(gen_sign_value);

                    verify(verify_backward_weights_conv<api, T>{
                        input, weights, output, filter, stats, preallocate, 0, search});
                }
            }
        }
    }
};

// CONV BIAS
//==========================
template <class T>
struct verify_backwards_bias
{
    tensor<T> output;
    tensor<T> bias;

    tensor<T> cpu() const
    {
        auto rbias = bias;
        cpu_bias_backward_data(output, rbias);
        return rbias;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rbias    = bias;

        auto out_dev  = handle.Write(output.data);
        auto bias_dev = handle.Write(rbias.data);

        float alpha = 1, beta = 0;
        ConvolutionBackwardBias(
            handle, &alpha, output.desc, out_dev.get(), &beta, rbias.desc, bias_dev.get());

        rbias.data = handle.Read<T>(bias_dev, rbias.data.size());
        return rbias;
    }

    void fail(int = 0) const
    {
        std::cout << "Backwards bias: " << std::endl;
        std::cout << "Output tensor: " << output.desc.ToString() << std::endl;
        std::cout << "Bias tensor: " << bias.desc.ToString() << std::endl;
    }
};

template <class T>
struct conv_bias_driver : test_driver
{
    tensor<T> output;

    int get_spatial_dim() const
    {
        for(int i = 2; i < 4; i++)
        {
            if(output.desc.GetNumDims() == i + 2)
                return i;
        }
        return -1;
    }

    void run()
    {
        std::vector<std::size_t> bias_lens(2 + get_spatial_dim(), 1);
        bias_lens[1] = output.desc.GetLengths()[1];

        tensor<T> bias(bias_lens);

        if(!(bias.desc.GetLengths()[0] == 1 &&
             bias.desc.GetLengths()[1] == output.desc.GetLengths()[0] &&
             std::all_of(bias.desc.GetLengths().begin() + 2,
                         bias.desc.GetLengths().end(),
                         [](auto v) { return v == 1; })))
        {
            return;
        }

        size_t total_mem =
            bias.desc.GetNumBytes() + output.desc.GetNumBytes(); // estimate based on backward pass
        size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        verify(verify_backwards_bias<T>{output, bias});
    }
};
