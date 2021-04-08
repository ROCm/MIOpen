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
#include <miopen/solver.hpp>
#include <miopen/invoke_params.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <miopen/stringutils.hpp>
#include "tensor_util.hpp"
#include <miopen/algorithm.hpp>
#include "cpu_conv.hpp"
#include "network_data.hpp"
#include "miopen/find_db.hpp"
#include "cpu_bias.hpp"

#define TEST_DIRECT_SUPPORTED_CONFIG_ONLY (!MIOPEN_USE_ROCBLAS && !MIOPEN_USE_MIOPENTENSILE)

#define WORKAROUND_MI100_ROM37_HIP_COMPILER_CRASH \
    (HIP_PACKAGE_VERSION_MAJOR == 3 && HIP_PACKAGE_VERSION_MINOR == 7)

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
        auto ctx = miopen::ConvolutionContext{
            xDesc, wDesc, yDesc, convDesc, static_cast<miopen::conv::Direction>(direction)};
        ctx.do_search               = false;
        ctx.save_srch_req           = false;
        ctx.disable_perfdb_access   = true;
        ctx.general_compile_options = "";
        ctx.SetStream(&handle);
        ctx.SetupFloats();
        ctx.DetectRocm();
        if(FindAllDirectSolutions(ctx, {}).empty())
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

    auto ctx = miopen::ConvolutionContext{
        xDesc, wDesc, yDesc, convDesc, miopen::conv::Direction::BackwardWeights};

    ctx.do_search               = false;
    ctx.save_srch_req           = false;
    ctx.general_compile_options = "";
    ctx.disable_perfdb_access   = true;
    ctx.SetStream(&handle);
    ctx.SetupFloats();
    ctx.DetectRocm();

    return !FindAllBwdWrW2DSolutions(ctx, {}).empty();
}
#endif

#if WORKAROUND_MI100_ROM37_HIP_COMPILER_CRASH
static inline bool skip_config(miopen::Handle& handle,
                               const miopen::ConvolutionDescriptor convDesc,
                               const miopen::TensorDescriptor& xDesc,
                               const miopen::TensorDescriptor& wDesc,
                               const miopen::TensorDescriptor& yDesc)
{
    if(convDesc.mode != miopenConvolution)
        return false;

    auto ctx =
        miopen::ConvolutionContext{xDesc, wDesc, yDesc, convDesc, miopen::conv::Direction::Forward};

    ctx.do_search               = false;
    ctx.save_srch_req           = false;
    ctx.general_compile_options = "";
    ctx.disable_perfdb_access   = true;
    ctx.SetStream(&handle);
    ctx.SetupFloats();
    ctx.DetectRocm();

    return ctx.GetStream().GetDeviceName() == "gfx908" && ctx.Is2d() && ctx.IsFp16() &&
           ctx.IsLayoutDefault() && ctx.use_hip_kernels && ctx.group_counts == 1 &&
           ctx.batch_sz == 1 && ctx.n_inputs == 192 && ctx.in_height == 28 && ctx.in_width == 28 &&
           ctx.n_outputs == 1 && ctx.kernel_size_h == 3 && ctx.kernel_size_w == 3 &&
           ctx.pad_w == 1 && ctx.pad_h == 1 && ctx.kernel_stride_w == 1 &&
           ctx.kernel_stride_h == 1 && ctx.kernel_dilation_w == 1 && ctx.kernel_dilation_h == 1;
}
#endif

struct scalar_gen_random_float
{
    double min_val = 0;
    double max_val = 1;

    double operator()() const
    {
        return min_val + (max_val - min_val) * double(std::rand()) / RAND_MAX;
    }
};

struct scalar_gen_random_integer
{
    unsigned long min_val = 1;
    unsigned long max_val = 16;

    double operator()() const
    {
        return static_cast<double>(min_val + std::rand() % (max_val - min_val + 1));
    }
};

struct tensor_elem_gen_one
{
    template <class... Ts>
    double operator()(Ts...) const
    {
        return 1;
    }
};

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

    std::string yLayout =
        out_layout.empty()
            ? input.desc.GetLayout(miopen::tensor_layout_get_default(input.desc.GetSize()))
            : out_layout;
    return tensor<Tout>{filter.GetForwardOutputTensorWithLayout(
        input.desc,
        weights.desc,
        yLayout,
        weights.desc.GetType() == miopenInt8 || weights.desc.GetType() == miopenInt8x4
            ? (std::is_same<Tout, int>{} ? miopenInt32 : miopenFloat)
            : weights.desc.GetType())};
}

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
    bool immed{};
    bool enable_fdb{};
    int conv_spatial_dims{};
    conv_stats* stats{}; // Denotes an object after object construction (never nullptr).

    void fail(float = 0) const
    {
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Weights tensor: " << weights.desc.ToString() << std::endl;
        std::cout << "Output tensor: " << out.desc.ToString() << std::endl;
        std::cout << "Filter: " << filter << std::endl;
    }
};

// Mainline convolution tests
//========================================
template <class T, class Tout = T>
struct verify_forward_conv : conv_base<T, Tout>
{
    using conv_base<T, Tout>::input;
    using conv_base<T, Tout>::weights;
    using conv_base<T, Tout>::out;
    using conv_base<T, Tout>::filter;
    using conv_base<T, Tout>::bias;
    using conv_base<T, Tout>::search;
    using conv_base<T, Tout>::immed;
    using conv_base<T, Tout>::stats;
    bool is_vect;

    verify_forward_conv(const tensor<T>& pinput,
                        const tensor<T>& pweights,
                        const tensor<Tout>& pout,
                        const miopen::ConvolutionDescriptor& pfilter,
                        conv_stats& pstats,
                        int pbias,
                        int psearch,
                        bool pvect,
                        bool pimmed)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
        is_vect = pvect;
        immed   = pimmed;
        stats   = &pstats;
    }

    tensor<Tout> cpu() const
    {
        auto rout = out;

        if(filter.mode == miopenTranspose)
        {
            std::fill(rout.begin(), rout.end(), 0);
            cpu_convolution_backward_data(filter.GetSpatialDimension(),
                                          rout,
                                          weights,
                                          input,
                                          filter.GetConvPads(),
                                          filter.GetConvStrides(),
                                          filter.GetConvDilations(),
                                          filter.GetGroupCount());
        }
        else
        {
            cpu_convolution_forward(filter.GetSpatialDimension(),
                                    input,
                                    weights,
                                    rout,
                                    filter.GetConvPads(),
                                    filter.GetConvStrides(),
                                    filter.GetConvDilations(),
                                    filter.GetGroupCount());

            bool is_int8 =
                weights.desc.GetType() == miopenInt8 || weights.desc.GetType() == miopenInt8x4;
            rout.par_for_each([&](auto... is) {
                if(is_int8)
                    rout(is...) = Tout(double(rout(is...)) + double(this->bias));
                else
                    rout(is...) = double(rout(is...)) + double(this->bias);
            });
        }

        return rout;
    }

    void resize_workspace(miopen::Handle& h,
                          const std::size_t sz,
                          std::vector<char>& ws,
                          miopen::Allocator::ManageDataPtr& ws_dev) const
    {
        ws_dev.reset();
        if(sz > 0)
        {
            ws.resize(sz);
            ws_dev = h.Write(ws);
        }
    }

    tensor<Tout> gpu() const
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
        miopenConvSolution_t selected;
        bool fallback_path_taken = false;
        std::size_t count        = 0;

        std::vector<char> ws;
        miopen::Allocator::ManageDataPtr ws_dev = nullptr;

        if(immed)
        {
            if(filter.mode == miopenTranspose)
            {
                if(miopen::testing_find_db_enabled)
                {
                    int ret_algo_count;
                    miopenConvAlgoPerf_t perf;
                    const size_t workspace_size = filter.BackwardDataGetWorkSpaceSize(
                        handle, weights.desc, input.desc, rout.desc);
                    resize_workspace(handle, workspace_size, ws, ws_dev);

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
                                                    ws_dev.get(),
                                                    workspace_size,
                                                    search);
                }
                count =
                    filter.GetBackwardSolutionCount(handle, input.desc, weights.desc, rout.desc);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1);
                }

                auto solutions = std::vector<miopenConvSolution_t>(count);

                filter.GetBackwardSolutions(handle,
                                            input.desc,
                                            weights.desc,
                                            rout.desc,
                                            count,
                                            &count,
                                            solutions.data(),
                                            &fallback_path_taken);

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1);
                }
                selected = solutions.front();

                {
                    const std::size_t ws_size = filter.GetBackwardSolutionWorkspaceSize(
                        handle, input.desc, weights.desc, rout.desc, selected.solution_id);
                    if(ws_size != selected.workspace_size)
                        std::cout << "WARNING: workspace size mismatch: " << selected.workspace_size
                                  << " != " << ws_size << std::endl;
                }
                resize_workspace(handle, selected.workspace_size, ws, ws_dev);

                filter.CompileBackwardSolution(
                    handle, input.desc, weights.desc, rout.desc, selected.solution_id);

                filter.ConvolutionBackwardImmediate(handle,
                                                    input.desc,
                                                    in_dev.get(),
                                                    weights.desc,
                                                    wei_dev.get(),
                                                    rout.desc,
                                                    out_dev.get(),
                                                    ws_dev.get(),
                                                    selected.workspace_size,
                                                    selected.solution_id);
            }
            else
            {
                if(miopen::testing_find_db_enabled)
                {
                    int ret_algo_count;
                    miopenConvAlgoPerf_t perf;
                    const size_t workspace_size =
                        filter.ForwardGetWorkSpaceSize(handle, weights.desc, input.desc, rout.desc);
                    resize_workspace(handle, workspace_size, ws, ws_dev);

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
                                                ws_dev.get(),
                                                workspace_size,
                                                search);
                }

                count = filter.GetForwardSolutionCount(handle, weights.desc, input.desc, rout.desc);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1);
                }

                // std::cout << "Forward Conv solutions available: " << count << std::endl;
                auto solutions = std::vector<miopenConvSolution_t>(count);

                filter.GetForwardSolutions(handle,
                                           weights.desc,
                                           input.desc,
                                           rout.desc,
                                           count,
                                           &count,
                                           solutions.data(),
                                           &fallback_path_taken);

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1);
                }
                selected = solutions.front();

                {
                    const std::size_t ws_size = filter.GetForwardSolutionWorkspaceSize(
                        handle, weights.desc, input.desc, rout.desc, selected.solution_id);
                    if(ws_size != selected.workspace_size)
                        std::cout << "WARNING: workspace size mismatch: " << selected.workspace_size
                                  << " != " << ws_size << std::endl;
                }
                resize_workspace(handle, selected.workspace_size, ws, ws_dev);

                filter.CompileForwardSolution(
                    handle, weights.desc, input.desc, rout.desc, selected.solution_id);

                filter.ConvolutionForwardImmediate(handle,
                                                   weights.desc,
                                                   wei_dev.get(),
                                                   input.desc,
                                                   in_dev.get(),
                                                   rout.desc,
                                                   out_dev.get(),
                                                   ws_dev.get(),
                                                   selected.workspace_size,
                                                   selected.solution_id);
            }
        }
        else // regular find path for testing
        {
            if(weights.desc.GetType() == miopenInt8 || weights.desc.GetType() == miopenInt8x4)
            {

                bool is_transform = (input.desc.GetLengths()[1] % 4 != 0 || is_vect);

                std::vector<std::size_t> in_len(input.desc.GetLengths().begin(),
                                                input.desc.GetLengths().end());
                std::vector<std::size_t> wei_len(weights.desc.GetLengths().begin(),
                                                 weights.desc.GetLengths().end());
                in_len[1]  = ((in_len[1] + 3) / 4) * 4;
                wei_len[1] = ((wei_len[1] + 3) / 4) * 4;

                miopen::TensorDescriptor input_vpad_desc(is_vect ? miopenInt8x4 : miopenInt8,
                                                         in_len);
                miopen::TensorDescriptor weight_vpad_desc(is_vect ? miopenInt8x4 : miopenInt8,
                                                          wei_len);

                auto input_vpad   = tensor<T>{in_len};
                auto weights_vpad = tensor<T>{wei_len};
                auto in_vpad_dev  = handle.Write(input_vpad.data);
                auto wei_vpad_dev = handle.Write(weights_vpad.data);

                if(is_transform)
                {
                    float aph = 1.0;
                    float bta = 0.0;
                    miopen::TransformTensor(handle,
                                            &aph,
                                            input.desc,
                                            in_dev.get(),
                                            &bta,
                                            input_vpad_desc,
                                            in_vpad_dev.get());

                    miopen::TransformTensor(handle,
                                            &aph,
                                            weights.desc,
                                            wei_dev.get(),
                                            &bta,
                                            weight_vpad_desc,
                                            wei_vpad_dev.get());
                }

                auto in_desc  = (is_transform ? input_vpad_desc : input.desc);
                auto wei_desc = (is_transform ? weight_vpad_desc : weights.desc);

                size_t workspace_size =
                    filter.ForwardGetWorkSpaceSize(handle, wei_desc, in_desc, rout.desc);

                std::vector<char> workspace(workspace_size);
                auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

                int ret_algo_count;
                miopenConvAlgoPerf_t perf;

                float alpha = 1, beta = 0;

                auto in_buf  = (is_transform ? in_vpad_dev.get() : in_dev.get());
                auto wei_buf = (is_transform ? wei_vpad_dev.get() : wei_dev.get());

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
                                            workspace_dev.get(),
                                            workspace_size,
                                            search);

                workspace_dev.reset();
                if(perf.memory > 0)
                {
                    workspace.resize(perf.memory);
                    workspace_dev = handle.Write(workspace);
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
                                          workspace_dev.get(),
                                          workspace_size);

                filter.GetForwardSolutions(handle,
                                           wei_desc,
                                           in_desc,
                                           rout.desc,
                                           1,
                                           &count,
                                           &selected,
                                           &fallback_path_taken); /// \ref read_solver_name
            }
            else
            {
                size_t workspace_size = filter.mode == miopenTranspose
                                            ? filter.BackwardDataGetWorkSpaceSize(
                                                  handle, weights.desc, input.desc, rout.desc)
                                            : filter.ForwardGetWorkSpaceSize(
                                                  handle, weights.desc, input.desc, rout.desc);

                std::vector<char> workspace(workspace_size);
                auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

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
                                                    workspace_dev.get(),
                                                    workspace_size,
                                                    search);

                    workspace_dev.reset();
                    if(perf.memory > 0)
                    {
                        workspace.resize(perf.memory);
                        workspace_dev = handle.Write(workspace);
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
                                                   workspace_dev.get(),
                                                   workspace_size);

                    filter.GetBackwardSolutions(handle,
                                                input.desc,
                                                weights.desc,
                                                rout.desc,
                                                1,
                                                &count,
                                                &selected,
                                                &fallback_path_taken); /// \ref read_solver_name
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
                                                workspace_dev.get(),
                                                workspace_size,
                                                search);

                    workspace_dev.reset();
                    if(perf.memory > 0)
                    {
                        workspace.resize(perf.memory);
                        workspace_dev = handle.Write(workspace);
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
                                              workspace_dev.get(),
                                              workspace_size);

                    filter.GetForwardSolutions(handle,
                                               weights.desc,
                                               input.desc,
                                               rout.desc,
                                               1,
                                               &count,
                                               &selected,
                                               &fallback_path_taken); /// \ref read_solver_name
                }
            }
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

template <class T>
struct verify_backward_conv : conv_base<T>
{
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::out;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;
    using conv_base<T>::immed;
    using conv_base<T>::stats;

    verify_backward_conv(const tensor<T>& pinput,
                         const tensor<T>& pweights,
                         const tensor<T>& pout,
                         const miopen::ConvolutionDescriptor& pfilter,
                         conv_stats& pstats,
                         int pbias,
                         int psearch,
                         bool pimmed)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
        immed   = pimmed;
        stats   = &pstats;
    }

    tensor<T> cpu() const
    {
        auto rinput = input;
        std::fill(rinput.begin(), rinput.end(), 0);

        if(filter.mode == miopenTranspose)
        {
            cpu_convolution_forward(filter.GetSpatialDimension(),
                                    out,
                                    weights,
                                    rinput,
                                    filter.GetConvPads(),
                                    filter.GetConvStrides(),
                                    filter.GetConvDilations(),
                                    filter.GetGroupCount());
        }
        else
        {
            cpu_convolution_backward_data(filter.GetSpatialDimension(),
                                          rinput,
                                          weights,
                                          out,
                                          filter.GetConvPads(),
                                          filter.GetConvStrides(),
                                          filter.GetConvDilations(),
                                          filter.GetGroupCount());
        }
        return rinput;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rinput   = input;
        std::fill(rinput.begin(), rinput.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Write(weights.data);
        auto in_dev  = handle.Write(rinput.data);

        miopenConvSolution_t selected;
        bool fallback_path_taken = false;
        std::size_t count        = 0;

        if(immed)
        {
            size_t workspace_size =
                filter.mode == miopenTranspose
                    ? filter.ForwardGetWorkSpaceSize(handle, weights.desc, out.desc, rinput.desc)
                    : filter.BackwardDataGetWorkSpaceSize(
                          handle, weights.desc, out.desc, rinput.desc);

            std::vector<char> workspace(workspace_size);
            auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

            int ret_algo_count;
            miopenConvAlgoPerf_t perf;

            if(filter.mode == miopenTranspose)
            {
                if(miopen::testing_find_db_enabled)
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
                                                workspace_dev.get(),
                                                workspace_size,
                                                search);
                }
                count = filter.GetForwardSolutionCount(handle, weights.desc, out.desc, rinput.desc);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1);
                }

                // std::cout << "backward transpose Conv solutions available: " << count <<
                // std::endl;
                auto solutions = std::vector<miopenConvSolution_t>(count);

                filter.GetForwardSolutions(handle,
                                           weights.desc,
                                           out.desc,
                                           rinput.desc,
                                           count,
                                           &count,
                                           solutions.data(),
                                           &fallback_path_taken);

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1);
                }
                solutions.resize(count);
                std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
                    return l.time < r.time;
                });
                selected = solutions.front();

                std::size_t ws_size;

                ws_size = filter.GetForwardSolutionWorkspaceSize(
                    handle, weights.desc, out.desc, rinput.desc, selected.solution_id);

                filter.CompileForwardSolution(
                    handle, weights.desc, out.desc, rinput.desc, selected.solution_id);

                workspace_dev.reset();
                if(selected.workspace_size > 0)
                {
                    workspace.resize(selected.workspace_size);
                    workspace_dev = handle.Write(workspace);
                }

                filter.ConvolutionForwardImmediate(handle,
                                                   weights.desc,
                                                   wei_dev.get(),
                                                   out.desc,
                                                   out_dev.get(),
                                                   rinput.desc,
                                                   in_dev.get(),
                                                   workspace_dev.get(),
                                                   ws_size,
                                                   selected.solution_id);
            }
            else
            {
                if(miopen::testing_find_db_enabled)
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
                                                    workspace_dev.get(),
                                                    workspace_size,
                                                    search);
                }
                count =
                    filter.GetBackwardSolutionCount(handle, out.desc, weights.desc, rinput.desc);

                if(count == 0)
                {
                    std::cout << "FAILED: Using immediate mode error in GetSolutionCount."
                              << std::endl;
                    exit(-1);
                }

                // std::cout << "Backward Conv solutions available: " << count << std::endl;
                auto solutions = std::vector<miopenConvSolution_t>(count);

                filter.GetBackwardSolutions(handle,
                                            out.desc,
                                            weights.desc,
                                            rinput.desc,
                                            count,
                                            &count,
                                            solutions.data(),
                                            &fallback_path_taken);

                if(count == 0)
                {
                    std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                              << " Solution count: " << count << std::endl;
                    exit(-1);
                }
                solutions.resize(count);
                std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
                    return l.time < r.time;
                });
                selected = solutions.front();

                std::size_t ws_size;

                ws_size = filter.GetBackwardSolutionWorkspaceSize(
                    handle, out.desc, weights.desc, rinput.desc, selected.solution_id);

                filter.CompileBackwardSolution(
                    handle, out.desc, weights.desc, rinput.desc, selected.solution_id);

                workspace_dev.reset();
                if(selected.workspace_size > 0)
                {
                    workspace.resize(selected.workspace_size);
                    workspace_dev = handle.Write(workspace);
                }

                filter.ConvolutionBackwardImmediate(handle,
                                                    out.desc,
                                                    out_dev.get(),
                                                    weights.desc,
                                                    wei_dev.get(),
                                                    rinput.desc,
                                                    in_dev.get(),
                                                    workspace_dev.get(),
                                                    ws_size,
                                                    selected.solution_id);
            }
        }
        else // Conventional Find conv
        {
            size_t workspace_size =
                filter.mode == miopenTranspose
                    ? filter.ForwardGetWorkSpaceSize(handle, weights.desc, out.desc, rinput.desc)
                    : filter.BackwardDataGetWorkSpaceSize(
                          handle, weights.desc, out.desc, rinput.desc);

            std::vector<char> workspace(workspace_size);
            auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

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
                                            workspace_dev.get(),
                                            workspace_size,
                                            search);

                workspace_dev.reset();
                if(perf.memory > 0)
                {
                    workspace.resize(perf.memory);
                    workspace_dev = handle.Write(workspace);
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
                                          workspace_dev.get(),
                                          workspace_size);

                filter.GetForwardSolutions(handle,
                                           weights.desc,
                                           out.desc,
                                           rinput.desc,
                                           1,
                                           &count,
                                           &selected,
                                           &fallback_path_taken); /// \ref read_solver_name
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
                                                workspace_dev.get(),
                                                workspace_size,
                                                search);

                workspace_dev.reset();
                if(perf.memory > 0)
                {
                    workspace.resize(perf.memory);
                    workspace_dev = handle.Write(workspace);
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
                                               workspace_dev.get(),
                                               workspace_size);

                filter.GetBackwardSolutions(handle,
                                            out.desc,
                                            weights.desc,
                                            rinput.desc,
                                            1,
                                            &count,
                                            &selected,
                                            &fallback_path_taken); /// \ref read_solver_name
            }
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

template <class T>
struct verify_backward_weights_conv : conv_base<T>
{
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::out;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;
    using conv_base<T>::immed;
    using conv_base<T>::stats;
    const bool is_conv_wrw_f32 = std::is_same<T, float>::value;

    verify_backward_weights_conv(const tensor<T>& pinput,
                                 const tensor<T>& pweights,
                                 const tensor<T>& pout,
                                 const miopen::ConvolutionDescriptor& pfilter,
                                 conv_stats& pstats,
                                 int pbias,
                                 int psearch,
                                 bool pimmed)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
        immed   = pimmed;
        stats   = &pstats;
    }

    tensor<T> cpu() const
    {
        auto rweights = weights;
        std::fill(rweights.begin(), rweights.end(), 0);

        if(filter.mode == miopenTranspose)
        {
            cpu_convolution_backward_weight(filter.GetSpatialDimension(),
                                            out,
                                            rweights,
                                            input,
                                            filter.GetConvPads(),
                                            filter.GetConvStrides(),
                                            filter.GetConvDilations(),
                                            filter.GetGroupCount());
        }
        else
        {
            cpu_convolution_backward_weight(filter.GetSpatialDimension(),
                                            input,
                                            rweights,
                                            out,
                                            filter.GetConvPads(),
                                            filter.GetConvStrides(),
                                            filter.GetConvDilations(),
                                            filter.GetGroupCount());
        }
        return rweights;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rweights = weights;
        std::fill(rweights.begin(), rweights.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Write(rweights.data);
        auto in_dev  = handle.Write(input.data);

        miopenConvSolution_t selected;
        bool fallback_path_taken = false;
        std::size_t count        = 0;

        if(immed)
        {
            std::size_t workspace_size = filter.BackwardWeightsGetWorkSpaceSize(
                handle,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? out.desc : input.desc,
                rweights.desc);

            std::vector<char> workspace(workspace_size);
            auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

            int ret_algo_count;
            miopenConvAlgoPerf_t perf;

            if(miopen::testing_find_db_enabled)
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
                    workspace_dev.get(),
                    workspace_size,
                    search);
            }

            count =
                filter.GetWrwSolutionCount(handle,
                                           filter.mode == miopenTranspose ? input.desc : out.desc,
                                           filter.mode == miopenTranspose ? out.desc : input.desc,
                                           rweights.desc);

            if(count == 0)
            {
                std::cout << "FAILED: Using immediate mode error in GetSolutionCount." << std::endl;
                exit(-1);
            }

            // std::cout << "Backward weights conv solutions available: " << count << std::endl;
            auto solutions = std::vector<miopenConvSolution_t>(count);

            filter.GetWrwSolutions(handle,
                                   filter.mode == miopenTranspose ? input.desc : out.desc,
                                   filter.mode == miopenTranspose ? out.desc : input.desc,
                                   rweights.desc,
                                   count,
                                   &count,
                                   solutions.data(),
                                   &fallback_path_taken);

            if(count == 0)
            {
                std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                          << " Solution count: " << count << std::endl;
                exit(-1);
            }
            solutions.resize(count);
            std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
                return l.time < r.time;
            });
            selected = solutions.front();

            std::size_t ws_size;

            ws_size = filter.GetWrwSolutionWorkspaceSize(
                handle,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? out.desc : input.desc,
                rweights.desc,
                selected.solution_id);

            filter.CompileWrwSolution(handle,
                                      filter.mode == miopenTranspose ? input.desc : out.desc,
                                      filter.mode == miopenTranspose ? out.desc : input.desc,
                                      rweights.desc,
                                      selected.solution_id);

            workspace_dev.reset();
            if(selected.workspace_size > 0)
            {
                workspace.resize(selected.workspace_size);
                workspace_dev = handle.Write(workspace);
            }

            filter.ConvolutionWrwImmediate(
                handle,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? in_dev.get() : out_dev.get(),
                filter.mode == miopenTranspose ? out.desc : input.desc,
                filter.mode == miopenTranspose ? out_dev.get() : in_dev.get(),
                rweights.desc,
                wei_dev.get(),
                workspace_dev.get(),
                ws_size,
                selected.solution_id);
        }
        else
        {

            std::size_t workspace_size = filter.BackwardWeightsGetWorkSpaceSize(
                handle,
                filter.mode == miopenTranspose ? input.desc : out.desc,
                filter.mode == miopenTranspose ? out.desc : input.desc,
                rweights.desc);

            std::vector<char> workspace(workspace_size);
            auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

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
                workspace_dev.get(),
                workspace_size,
                search);

            workspace_dev.reset();
            if(perf.memory > 0)
            {
                workspace.resize(perf.memory);
                workspace_dev = handle.Write(workspace);
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
                workspace_dev.get(),
                workspace_size);

            filter.GetWrwSolutions(handle,
                                   filter.mode == miopenTranspose ? input.desc : out.desc,
                                   filter.mode == miopenTranspose ? out.desc : input.desc,
                                   rweights.desc,
                                   1,
                                   &count,
                                   &selected,
                                   &fallback_path_taken); /// \ref read_solver_name
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
            cpu_convolution_forward(filter.GetSpatialDimension(),
                                    input,
                                    weights,
                                    rout,
                                    filter.GetConvPads(),
                                    filter.GetConvStrides(),
                                    filter.GetConvDilations(),
                                    filter.GetGroupCount());

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

        miopen::TensorDescriptor input_vpad_desc(is_vect ? miopenInt8x4 : miopenInt8, in_len);
        miopen::TensorDescriptor weight_vpad_desc(is_vect ? miopenInt8x4 : miopenInt8, wei_len);

        auto input_vpad   = tensor<T>{in_len};
        auto weights_vpad = tensor<T>{wei_len};
        auto in_vpad_dev  = handle.Write(input_vpad.data);
        auto wei_vpad_dev = handle.Write(weights_vpad.data);

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

        size_t workspace_size =
            filter.ForwardGetWorkSpaceSize(handle,
                                           (is_transform ? weight_vpad_desc : weights.desc),
                                           (is_transform ? input_vpad_desc : input.desc),
                                           rout.desc);

        std::vector<char> workspace(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        int ret_algo_count;
        miopenConvAlgoPerf_t perf;

        if(miopen::testing_find_db_enabled)
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
                                        workspace_dev.get(),
                                        workspace_size,
                                        search);
        }

        auto count =
            filter.GetForwardSolutionCount(handle,
                                           (is_transform ? weight_vpad_desc : weights.desc),
                                           (is_transform ? input_vpad_desc : input.desc),
                                           rout.desc);

        if(count == 0)
        {
            std::cout << "FAILED: Using immediate mode error in GetSolutionCount." << std::endl;
            exit(-1);
        }

        // std::cout << "Forward Conv solutions available: " << count << std::endl;
        auto solutions           = std::vector<miopenConvSolution_t>(count);
        bool fallback_path_taken = false;

        filter.GetForwardSolutions(handle,
                                   (is_transform ? weight_vpad_desc : weights.desc),
                                   (is_transform ? input_vpad_desc : input.desc),
                                   rout.desc,
                                   count,
                                   &count,
                                   solutions.data(),
                                   &fallback_path_taken);

        if(count == 0)
        {
            std::cout << "FAILED: Immediate mode has no fallback for this configuration."
                      << " Solution count: " << count << std::endl;
            exit(-1);
        }
        solutions.resize(count);
        std::sort(solutions.begin(), solutions.end(), [](const auto& l, const auto& r) {
            return l.time < r.time;
        });
        auto selected = solutions.front();

        std::size_t ws_size;

        ws_size =
            filter.GetForwardSolutionWorkspaceSize(handle,
                                                   (is_transform ? weight_vpad_desc : weights.desc),
                                                   (is_transform ? input_vpad_desc : input.desc),
                                                   rout.desc,
                                                   selected.solution_id);

        filter.CompileForwardSolution(handle,
                                      (is_transform ? weight_vpad_desc : weights.desc),
                                      (is_transform ? input_vpad_desc : input.desc),
                                      rout.desc,
                                      selected.solution_id);

        workspace_dev.reset();
        if(selected.workspace_size > 0)
        {
            workspace.resize(selected.workspace_size);
            workspace_dev = handle.Write(workspace);
        }

        filter.ConvolutionForwardImmediate(handle,
                                           (is_transform ? weight_vpad_desc : weights.desc),
                                           (is_transform ? wei_vpad_dev.get() : wei_dev.get()),
                                           (is_transform ? input_vpad_desc : input.desc),
                                           (is_transform ? in_vpad_dev.get() : in_dev.get()),
                                           rout.desc,
                                           out_dev.get(),
                                           workspace_dev.get(),
                                           ws_size,
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

template <class T, bool immed_mode = false>
struct conv_driver : test_driver
{
    tensor<T> input;
    tensor<T> weights;
    miopen::ConvolutionDescriptor filter;
    std::string conv_mode;
    std::string pad_mode;
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
    bool immed               = immed_mode;
    bool enable_fdb          = true;

    std::unordered_map<std::string, miopenConvolutionMode_t> cmode_lookup = {
        {"CONV", miopenConvolution},
        {"TRANS", miopenTranspose},
        {"CONVOLUTION", miopenConvolution},
        {"TRANSPOSE", miopenTranspose}};

    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
        {"DEFAULT", miopenPaddingDefault}};

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
            if(input.desc.GetSize() == i + 2 and weights.desc.GetSize() == i + 2 and
               pads_strides_dilations.size() == i * 3 and trans_output_pads.size() == i)
                return i;
        }
        return -1;
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
        if(immed)
        {
            add(enable_fdb, "enable-fdb", generate_data({false, true}));
        }
    }

    void run()
    {
        if(input.desc.GetSize() != in_layout.size() ||
           weights.desc.GetSize() != fil_layout.size() || input.desc.GetSize() != out_layout.size())
        {
            std::cerr << "FAILED: layout not match dimension size!" << std::endl;
            return;
        }

        // reconstruct tensor descriptor(desc) when layout is not the default NCHW layout.
        // by default, this member is constructed when conv2d/3d is constructed (see
        // test_driver::add())
        // but this requires the dimensions come from commandline, which is hard for non-NCHW layout
        if(in_layout != "NCHW" || in_layout != "NCDHW")
        {
            const std::vector<std::size_t> dim_lens = input.desc.GetLengths();
            std::vector<std::size_t> dim_strides;
            miopen::tensor_layout_to_strides(
                dim_lens,
                miopen::tensor_layout_get_default(input.desc.GetSize()),
                in_layout,
                dim_strides);
            input.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
        }
        if(fil_layout != "NCHW" || fil_layout != "NCDHW")
        {
            const std::vector<std::size_t> dim_lens = weights.desc.GetLengths();
            std::vector<std::size_t> dim_strides;
            miopen::tensor_layout_to_strides(
                dim_lens,
                miopen::tensor_layout_get_default(weights.desc.GetSize()),
                fil_layout,
                dim_strides);
            weights.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
        }

        filter.spatialDim       = get_spatial_dim();
        filter.mode             = cmode_lookup[miopen::ToUpper(conv_mode)];
        filter.paddingMode      = pmode_lookup[miopen::ToUpper(pad_mode)];
        std::size_t spatial_dim = filter.GetSpatialDimension();

        if(input.desc.GetSize() != 2 + spatial_dim || weights.desc.GetSize() != 2 + spatial_dim ||
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

        std::copy_n(pads_strides_dilations.begin(), spatial_dim, filter.pads.begin());
        std::copy_n(
            pads_strides_dilations.begin() + spatial_dim, spatial_dim, filter.strides.begin());
        std::copy_n(pads_strides_dilations.begin() + 2 * spatial_dim,
                    spatial_dim,
                    filter.dilations.begin());
        std::copy_n(trans_output_pads.begin(), spatial_dim, filter.trans_output_pads.begin());

        filter.group_count = std::max(static_cast<int>(groupCount), 1);

        std::size_t in_c_len  = input.desc.GetLengths()[1];
        std::size_t wei_k_len = weights.desc.GetLengths()[0];
        std::size_t wei_c_len = weights.desc.GetLengths()[1];

        std::vector<std::size_t> in_spatial_len(input.desc.GetLengths().begin() + 2,
                                                input.desc.GetLengths().end());
        std::vector<std::size_t> wei_spatial_len(weights.desc.GetLengths().begin() + 2,
                                                 weights.desc.GetLengths().end());

        bool is_int8 = (input.desc.GetType() == miopenInt8 || input.desc.GetType() == miopenInt8x4);

        // lack of transposeConv or groupConv for int8 type
        if(is_int8 && (filter.mode == miopenTranspose || filter.group_count > 1))
        {
            show_command();
            std::cout << "MIOpen doesn't support int8 type transpose or group convolution."
                      << std::endl;
            return;
        }

        bool is_bfloat16 =
            (input.desc.GetType() == miopenBFloat16 && weights.desc.GetType() == miopenBFloat16);

        // bfloat16 is not supported for conv3d
        if(is_bfloat16 && !(filter.spatialDim == 2))
            return;

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
                        return;
                    }
                }
                else if(filter.paddingMode == miopenPaddingValid)
                {
                    if(miopen::any_of(filter.GetConvStrides(), [](auto v) { return v == 0; }))
                        return;

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
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1))) ||
                 (filter.group_count > 1 &&
                  (input.desc.GetLengths().at(1) % weights.desc.GetLengths().at(1) == 0)))))
            {
                auto output = get_output_tensor(filter, input, weights, out_layout);

                auto gen_positive_value = [=](auto...) {
                    auto data_type    = input.desc.GetType();
                    std::size_t v_max = is_int8 ? 16 : (data_type == miopenHalf) ? 4 : 16;

                    return gen_float ? scalar_gen_random_float{0, 1}()
                                     : scalar_gen_random_integer{1, v_max}();
                };

                auto gen_sign_value = [=](auto... is) {
                    auto data_type    = input.desc.GetType();
                    std::size_t v_max = is_int8 ? 16 : (data_type == miopenHalf) ? 4 : 16;

                    return gen_float ? scalar_gen_random_float{-1, 1}()
                                     : scalar_gen_random_integer{1, v_max}() *
                                           tensor_elem_gen_checkboard_sign{}(is...);
                };

                auto ctx = miopen::ConvolutionContext(input.desc,
                                                      weights.desc,
                                                      output.desc,
                                                      filter,
                                                      miopen::conv::Direction::Forward);
                ctx.SetStream(&get_handle());

                bool skip_forward = is_int8 && !IsGemmAplicable(ctx);
                if(skip_forward)
                {
                    show_command();
                    std::cout << "This config in int8 type is not supported." << std::endl;
                    return;
                }

                bool skip_backward_data    = is_int8;
                bool skip_backward_weights = is_int8;

#if TEST_DIRECT_SUPPORTED_CONFIG_ONLY
                if(input.desc.GetType() == miopenInt8 || input.desc.GetType() == miopenInt8x4 ||
                   input.desc.GetType() == miopenBFloat16)
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

#if WORKAROUND_MI100_ROM37_HIP_COMPILER_CRASH
                if(skip_config(get_handle(), filter, input.desc, weights.desc, output.desc))
                {
                    skip_forward          = true;
                    skip_backward_data    = true;
                    skip_backward_weights = true;
                }
#endif

                // bwd53 kernel (large images supported) doesnt support stride !=1 and dilation and
                // pad.
                if(filter.GetSpatialDimension() == 2 && in_spatial_len[1] >= 2048 &&
                   ((filter.GetConvStrides()[0] != 1) || (filter.GetConvStrides()[1] != 1) ||
                    (filter.GetConvDilations()[0] != 1) || (filter.GetConvDilations()[1] != 1) ||
                    (filter.GetConvPads()[1] != 0) || (filter.GetConvPads()[0] != 0)))
                {
                    return;
                }

                input.generate(gen_positive_value);
                output.generate(gen_positive_value);
                weights.generate(gen_sign_value);

                auto&& handle = get_handle();

                size_t total_mem;
                if(is_int8)
                {
                    auto output_int8 =
                        get_output_tensor<T, float>(filter, input, weights, out_layout);
                    size_t workspace_size = filter.ForwardGetWorkSpaceSize(
                        handle, weights.desc, input.desc, output_int8.desc);

                    // 4x because assume type is miopenInt8x4
                    total_mem = input.desc.GetNumBytes() + 4 * input.desc.GetNumBytes() +
                                weights.desc.GetNumBytes() + 4 * weights.desc.GetNumBytes() +
                                output_int8.desc.GetNumBytes() + 4 * sizeof(char) * workspace_size;
                }
                else
                {
                    size_t workspace_size_1 =
                        filter.mode == miopenTranspose
                            ? filter.ForwardGetWorkSpaceSize(
                                  handle, weights.desc, output.desc, input.desc)
                            : filter.BackwardDataGetWorkSpaceSize(
                                  handle, weights.desc, output.desc, input.desc);

                    size_t workspace_size_2 =
                        filter.mode == miopenTranspose
                            ? filter.BackwardDataGetWorkSpaceSize(
                                  handle, weights.desc, input.desc, output.desc)
                            : filter.ForwardGetWorkSpaceSize(
                                  handle, weights.desc, input.desc, output.desc);

                    size_t workspace_size_3 = filter.BackwardWeightsGetWorkSpaceSize(
                        handle,
                        filter.mode == miopenTranspose ? input.desc : output.desc,
                        filter.mode == miopenTranspose ? output.desc : input.desc,
                        weights.desc);

                    std::vector<size_t> workspace_sizes = {
                        workspace_size_1, workspace_size_2, workspace_size_3};
                    size_t workspace_size =
                        *std::max_element(workspace_sizes.begin(), workspace_sizes.end());

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

                if(immed)
                {
                    miopen::testing_find_db_enabled = enable_fdb;
                }

                conv_stats stats;

                if(do_forward && !skip_forward)
                {
                    if(is_int8)
                    {
                        verify(verify_forward_conv<T, float>{
                            input,
                            weights,
                            get_output_tensor<T, float>(filter, input, weights, out_layout),
                            filter,
                            stats,
                            0,
                            search,
                            false,
                            immed});
                        verify(verify_forward_conv<T, float>{
                            input,
                            weights,
                            get_output_tensor<T, float>(filter, input, weights, out_layout),
                            filter,
                            stats,
                            0,
                            search,
                            true,
                            immed});
                        verify(verify_forward_conv<T, int>{
                            input,
                            weights,
                            get_output_tensor<T, int>(filter, input, weights, out_layout),
                            filter,
                            stats,
                            0,
                            search,
                            false,
                            immed});
                        verify(verify_forward_conv<T, int>{
                            input,
                            weights,
                            get_output_tensor<T, int>(filter, input, weights, out_layout),
                            filter,
                            stats,
                            0,
                            search,
                            true,
                            immed});
                    }
                    else
                    {
                        verify(verify_forward_conv<T>{
                            input, weights, output, filter, stats, 0, search, false, immed});
                    }
                }

                if(do_backward_data && !skip_backward_data)
                {
                    verify(verify_backward_conv<T>{
                        input, weights, output, filter, stats, 0, search, immed});
                }

                if(do_backward_weights && !skip_backward_weights)
                {
                    output.generate(gen_sign_value);

                    verify(verify_backward_weights_conv<T>{
                        input, weights, output, filter, stats, 0, search, immed});
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
            if(output.desc.GetSize() == i + 2)
                return i;
        }
        return -1;
    }

    void run()
    {
        std::vector<std::size_t> bias_lens(2 + get_spatial_dim(), 1);
        bias_lens[1] = output.desc.GetLengths()[1];

        tensor<T> bias(bias_lens);

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
