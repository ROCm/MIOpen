/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#define MIOPEN

#include <miopen/allocator.hpp>
#include <miopen/db_path.hpp>
#include <miopen/handle.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/mlo_utils.hpp>
#include <miopen/solver.hpp>
#include <miopen/bfloat16.hpp>

#include <half.hpp>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace miopen {
namespace solver {

/*
 * select default configuration if a known configuration has not been found.
 */
LegacyPerformanceConfig
ConvOclDirectFwdLegacyExhaustiveSearch::GetPerformanceConfig(const ConvolutionContext& params) const
{
    //
    LegacyPerformanceConfig result{};
    result.in_tile0 = (params.in_width <= 8)
                          ? 8
                          : (params.in_width <= 16) ? 16 : 32; // size of input data per ALU plane
    result.in_tile1 = (params.in_height <= 8)
                          ? 8
                          : (params.in_height <= 16) ? 16 : 32; // size of input data per ALU plane

    result.out_pix_tile0 =
        std::max(params.kernel_stride_w,
                 ((result.in_tile0 == 8) ? 1 : 2)); // size of ouptput tile per wk-item (ALU))
    result.out_pix_tile1 = std::max(params.kernel_stride_h, ((result.in_tile1 == 8) ? 1 : 2)); //

    result.grp_tile0 = std::max(8, (result.in_tile0 / result.out_pix_tile0));
    result.grp_tile1 = std::max(8, (result.in_tile1 / result.out_pix_tile1));
    result.in_tile0  = result.grp_tile0 * result.out_pix_tile0;
    result.in_tile1  = result.grp_tile1 * result.out_pix_tile1;

    result.n_out_pix_tiles = 8; // # output pixel tiles per wk-item (ALU)
    result.n_in_data_tiles = 2; // # of blocks of different inputs in LDS

    result.n_stacks = 1; // # of diff stacks (part of batch).

    if(params.kernel_size_w == 1 && params.kernel_size_h == 1 &&
       params.group_counts == 1) // Group conv: None 1x1 version yet, fallback to universal kernel.
    {

        // version
        if(params.in_data_type == miopenFloat && params.direction.IsForward() &&
           params.n_inputs % 16 == 0 && params.n_outputs % 16 == 0)
        {
            result.n_in_data_tiles = 128;

            result.n_out_pix_tiles = 32;
            // 0 or 1
            // CHEAT_SHADER_COMPILER =
            result.out_pix_tile0 = 0;

            // int version =
            result.out_pix_tile1 = 1;
        }
        else
        {
            int i_sz             = params.out_height * params.out_width;
            result.out_pix_tile0 = (i_sz & 1) != 0 ? 1 : 2;

            if(params.pad_w > 0 || params.kernel_stride_w > 1)
            {
                if(params.direction.IsForward())
                {
                    result.out_pix_tile0 = (params.out_width & 1) != 0 ? 1 : 2;
                }
                else
                {
                    result.out_pix_tile0 =
                        (((params.out_width & 1) != 0) || ((params.in_width & 1) != 0)) ? 1 : 2;
                }
            }

            result.n_out_pix_tiles = 16;
            result.n_in_data_tiles = 4;
            result.grp_tile0       = 64;
            // int version =
            result.out_pix_tile1 = 0;
        }
    }
    if(!params.do_search) // Prevent spamming durign search.
        MIOPEN_LOG_I2("Returns: " << result);
    return result;
}

/*
 * Measure the current configuration performance.
 */
template <typename Tgpu, class... Solvers>
static int MeasurePerfConfig(const Handle& handle,
                             ConstData_t bot_ocl_buf,
                             Data_t top_ocl_buf,
                             ConstData_t wei_ocl_buf,
                             ConstData_t bias_ocl_buf,
                             double& processing_time,
                             const ConvolutionContext& params,
                             const ConvSolution& kernel_search_result)
{
    if(!kernel_search_result.Succeeded())
    {
        return 1;
    }
#if !MIOPEN_ALLOC_BUFFERS
    if(params.bias && bias_ocl_buf == nullptr)
    {
        MIOPEN_LOG_WE("Legacy search: Bias buffer required");
        return 2;
    }
#endif

    MIOPEN_LOG_I2("Trying " << kernel_search_result.performance_config);
    const auto kernel_params     = kernel_search_result.construction_params[0];
    std::string compiler_options = params.general_compile_options + kernel_params.comp_options;

    try
    {
        Tgpu padding_value = static_cast<Tgpu>(0);
        processing_time    = std::numeric_limits<double>::max();

        auto k = handle.AddKernel("",
                                  "",
                                  kernel_params.kernel_file,
                                  kernel_params.kernel_name,
                                  kernel_params.l_wk,
                                  kernel_params.g_wk,
                                  compiler_options);

        if(params.bias)
        {
            k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
        }
        else
        {
            k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
        }
        processing_time = handle.GetKernelTime();
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE("Status: " << ex.status << ", Message: \"" << ex.what() << '\"');
        return -1;
    }

    MIOPEN_LOG_I2("\t\t\t\t" << processing_time);
    return 0;
}

ConvSolution
ConvOclDirectFwdLegacyExhaustiveSearch::ScreenSolutions(const std::vector<ConvSolution>& solutions,
                                                        const ConvolutionContext& context) const
{
    if(context.IsFp16())
        return SearchImpl<half_float::half>(context, solutions);
    else if(context.IsFp32())
        return SearchImpl<float>(context, solutions);
    else if(context.IsBfp16())
        return SearchImpl<bfloat16>(context, solutions);
    else
    {
        MIOPEN_THROW("Unsupported float_size");
    }
}

template <typename Tgpu>
ConvSolution
ConvOclDirectFwdLegacyExhaustiveSearch::SearchImpl(const ConvolutionContext& params,
                                                   const std::vector<ConvSolution>& solutions) const
{
    ConvSolution result;
    bool is_passed = false;

    double processing_time = std::numeric_limits<double>::max();

#if MIOPEN_ALLOC_BUFFERS
    miopen::Handle profile_h;

    // allocate input/output buffers
    size_t bot_sz = params.bot_sz / sizeof(Tgpu);
    std::vector<Tgpu> bot_sys_buf(bot_sz);
    for(size_t i = 0; i < bot_sz; i++)
    {
        bot_sys_buf[i] = static_cast<Tgpu>(rand() * (1.0 / RAND_MAX));
    }
    auto bot_ocl_buf = profile_h.Write(bot_sys_buf);
    auto bot_ocl_ptr = bot_ocl_buf.get();

    size_t top_sz = params.top_sz / sizeof(Tgpu);
    std::vector<Tgpu> top_sys_buf(top_sz);
    auto top_ocl_buf = profile_h.Write(top_sys_buf);
    auto top_ocl_ptr = top_ocl_buf.get();

    size_t weights_sz = params.weights_sz / sizeof(Tgpu);
    std::vector<Tgpu> wei_sys_buf(weights_sz);
    for(size_t i = 0; i < weights_sz; i++)
    {
        wei_sys_buf[i] = static_cast<Tgpu>((rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
    }
    auto wei_ocl_buf = profile_h.Write(wei_sys_buf);
    auto wei_ocl_ptr = wei_ocl_buf.get();

    std::vector<Tgpu> bias_sys_buf;
    miopen::Allocator::ManageDataPtr bias_ocl_buf = nullptr;
    if(params.bias != 0)
    {
        size_t bias_sz = params.bias_sz / sizeof(Tgpu);
        bias_sys_buf   = std::vector<Tgpu>(bias_sz);
        for(size_t i = 0; i < bias_sz; i++)
        {
            bias_sys_buf[i] = static_cast<Tgpu>(rand() * (1.0 / RAND_MAX));
        }

        bias_ocl_buf = profile_h.Write(bias_sys_buf);
    }
    auto bias_ocl_ptr = bias_ocl_buf.get();
#else
    auto& profile_h = params.GetStream();
    auto bot_ocl_ptr =
        params.direction.IsForward() ? params.GetBufs().io.fwd.x : params.GetBufs().io.bwd.dy;
    auto top_ocl_ptr =
        params.direction.IsForward() ? params.GetBufs().io.fwd.y : params.GetBufs().io.bwd.dx;
    auto wei_ocl_ptr =
        params.direction.IsForward() ? params.GetBufs().io.fwd.w : params.GetBufs().io.bwd.w;
    auto bias_ocl_ptr = params.GetBufs().bias;
#endif
    AutoEnableProfiling enableProfiling{profile_h};

    long long runs_left = solutions.size(), total_runs = solutions.size(), run_counter = 0,
              report_inteval = 5, failed_counter = 0;
    double min_proc_time     = std::numeric_limits<double>::max();

    if(params.kernel_size_w == 1 && params.kernel_size_h == 1 &&
       params.group_counts == 1) // Group conv: None 1x1 version yet, fallback to universal kernel.
    {
        MIOPEN_LOG_W("Searching the best solution in the " << total_runs << " solutions.");

        for(auto& current_solution : solutions)
        {

            const auto ret = MeasurePerfConfig<Tgpu, ConvOclDirectFwd1x1>(profile_h,
                                                                          bot_ocl_ptr,
                                                                          top_ocl_ptr,
                                                                          wei_ocl_ptr,
                                                                          bias_ocl_ptr,
                                                                          processing_time,
                                                                          params,
                                                                          current_solution);
            --runs_left;
            if(ret != 0)
            {
                ++failed_counter;
                run_counter++;
                continue;
            }

            is_passed = true;
            MIOPEN_LOG_T("##(n_current, n_failed, n_runs_total): "
                         << run_counter << " / " << failed_counter << " / " << total_runs
                         << " elapsed_time: " << processing_time << " best_time: "
                         << processing_time << ", " << current_solution.performance_config);

            if(processing_time < min_proc_time)
            {
                MIOPEN_LOG_I('#' << run_counter << ' ' << processing_time << " < " << min_proc_time
                                 << ' ' << current_solution.performance_config);
                min_proc_time = processing_time;
                result        = current_solution;
            }

            if(run_counter % report_inteval == 0)
            {
                MIOPEN_LOG_W("Runs left: " << runs_left << ", "
                                           << "min time so far: " << min_proc_time << ", "
                                           << "curr time: " << processing_time << ' '
                                           << current_solution.performance_config);
            }
            run_counter++;
        }
    }
    else
    {
        MIOPEN_LOG_W("Searching the best solution in the " << total_runs << " solutions.");

        // tile1
        for(auto& current_solution : solutions)
        {

            const auto ret = MeasurePerfConfig<Tgpu, ConvOclDirectFwd>(profile_h,
                                                                       bot_ocl_ptr,
                                                                       top_ocl_ptr,
                                                                       wei_ocl_ptr,
                                                                       bias_ocl_ptr,
                                                                       processing_time,
                                                                       params,
                                                                       current_solution);
            --runs_left;
            if(ret != 0)
            {
                ++failed_counter;
                run_counter++;
                continue;
            }

            is_passed = true;
            MIOPEN_LOG_T("##(n_current, n_failed, n_runs_total): "
                         << run_counter << " / " << failed_counter << " / " << total_runs
                         << " elapsed_time: " << processing_time << " best_time: "
                         << processing_time << ", " << current_solution.performance_config);

            if(processing_time < min_proc_time)
            {
                MIOPEN_LOG_I('#' << run_counter << ' ' << processing_time << " < " << min_proc_time
                                 << ' ' << current_solution.performance_config);
                min_proc_time = processing_time;
                result        = current_solution;
            }

            if(run_counter % report_inteval == 0)
            {
                MIOPEN_LOG_W("Runs left: " << runs_left << ", "
                                           << "min time so far: " << min_proc_time << ", "
                                           << "curr time: " << processing_time << ' '
                                           << current_solution.performance_config);
            }
            run_counter++;
        }
    }

    if(!is_passed)
        MIOPEN_THROW("Search failed for ConvOclDirectFwdLegacyExhaustiveSearch");
    return result;
}

} // namespace solver
} // namespace miopen
