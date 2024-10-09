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

#include <miopen/conv/solvers.hpp>

#include <miopen/allocator.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/db_path.hpp>
#include <miopen/handle.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/bfloat16.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <half/half.hpp>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

/*
 * select default configuration if a known configuration has not been found.
 */
LegacyPerformanceConfig ConvOclDirectFwdLegacyExhaustiveSearch::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    //
    LegacyPerformanceConfig result{};
    result.in_tile0 = (problem.GetInWidth() <= 8)    ? 8
                      : (problem.GetInWidth() <= 16) ? 16
                                                     : 32; // size of input data per ALU plane
    result.in_tile1 = (problem.GetInHeight() <= 8)    ? 8
                      : (problem.GetInHeight() <= 16) ? 16
                                                      : 32; // size of input data per ALU plane

    result.out_pix_tile0 =
        std::max(problem.GetKernelStrideW(),
                 ((result.in_tile0 == 8) ? 1 : 2)); // size of ouptput tile per wk-item (ALU))
    result.out_pix_tile1 =
        std::max(problem.GetKernelStrideH(), ((result.in_tile1 == 8) ? 1 : 2)); //

    result.grp_tile0 = std::max(8, (result.in_tile0 / result.out_pix_tile0));
    result.grp_tile1 = std::max(8, (result.in_tile1 / result.out_pix_tile1));
    result.in_tile0  = result.grp_tile0 * result.out_pix_tile0;
    result.in_tile1  = result.grp_tile1 * result.out_pix_tile1;

    result.n_out_pix_tiles = 8; // # output pixel tiles per wk-item (ALU)
    result.n_in_data_tiles = 2; // # of blocks of different inputs in LDS

    result.n_stacks = 1; // # of diff stacks (part of batch).

    if(problem.GetWeightsWidth() == 1 && problem.GetWeightsHeight() == 1 &&
       problem.GetGroupCount() ==
           1) // Group conv: None 1x1 version yet, fallback to universal kernel.
    {

        // version
        if(problem.GetInDataType() == miopenFloat && problem.IsDirectionForward() &&
           problem.GetInChannels() % 16 == 0 && problem.GetOutChannels() % 16 == 0)
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
            int i_sz             = problem.GetOutHeight() * problem.GetOutWidth();
            result.out_pix_tile0 = (i_sz & 1) != 0 ? 1 : 2;

            if(problem.GetPadW() > 0 || problem.GetKernelStrideW() > 1)
            {
                if(problem.IsDirectionForward())
                {
                    result.out_pix_tile0 = (problem.GetOutWidth() & 1) != 0 ? 1 : 2;
                }
                else
                {
                    result.out_pix_tile0 =
                        (((problem.GetOutWidth() & 1) != 0) || ((problem.GetInWidth() & 1) != 0))
                            ? 1
                            : 2;
                }
            }

            result.n_out_pix_tiles = 16;
            result.n_in_data_tiles = 4;
            result.grp_tile0       = 64;
            // int version =
            result.out_pix_tile1 = 0;
        }
    }
    if(!ctx.do_search) // Prevent spamming durign search.
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
                             const ExecutionContext& ctx,
                             const ProblemDescription& problem,
                             const LegacyPerformanceConfig& config)
{
    ConvSolution kernel_search_result{miopenStatusNotInitialized};

    miopen::each_args(
        [&](auto s) {
            if(!kernel_search_result.Succeeded()) // once
            {
                if(s.IsApplicable(ctx, problem))
                {
                    if(s.IsValidPerformanceConfig(ctx, problem, config))
                    {
                        kernel_search_result = s.GetSolution(ctx, problem, config);
                    }
                }
            }
        },
        Solvers{}...);

    if(!kernel_search_result.Succeeded())
    {
        return 1;
    }
    if(problem.GetBias() && bias_ocl_buf == nullptr)
    {
        MIOPEN_LOG_WE("Legacy search: Bias buffer required");
        return 2;
    }

    MIOPEN_LOG_I2("Trying " << config);
    const auto kernel_params     = kernel_search_result.construction_params[0];
    std::string compiler_options = kernel_params.comp_options;

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

        if(problem.GetBias())
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

LegacyPerformanceConfig
ConvOclDirectFwdLegacyExhaustiveSearch::Search(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx) const
{
    if(problem.IsFp16())
    {
        return SearchImpl<half_float::half>(ctx, problem, invoke_ctx);
    }
    else if(problem.IsFp32())
    {
        return SearchImpl<float>(ctx, problem, invoke_ctx);
    }
    else if(problem.IsBfp16())
    {
        return SearchImpl<bfloat16>(ctx, problem, invoke_ctx);
    }
    else
    {
        MIOPEN_THROW("Unsupported float_size");
    }
}

template <typename Tgpu>
LegacyPerformanceConfig
ConvOclDirectFwdLegacyExhaustiveSearch::SearchImpl(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem,
                                                   const AnyInvokeParams& invoke_ctx) const
{
    LegacyPerformanceConfig result;
    bool is_passed = false;

    double processing_time = std::numeric_limits<double>::max();

    LegacyPerformanceConfig candidate;
    candidate.grp_tile0       = 16;
    candidate.grp_tile1       = 16;
    candidate.in_tile0        = 16;
    candidate.in_tile1        = 16;
    candidate.out_pix_tile0   = 1;
    candidate.out_pix_tile1   = 1;
    candidate.n_out_pix_tiles = 2;
    candidate.n_in_data_tiles = 3;
    candidate.n_stacks        = 1;

    auto& profile_h           = ctx.GetStream();
    const auto& invoke_params = invoke_ctx.CastTo<miopen::conv::DataInvokeParams>();
    const auto bot_ocl_ptr    = invoke_params.tensors.in;
    const auto top_ocl_ptr    = invoke_params.tensors.out;
    const auto wei_ocl_ptr    = invoke_params.tensors.w;
    // There was no place in the source, where it has been actually set to something other than
    // nullptr.
    const auto bias_ocl_ptr = static_cast<Data_t>(nullptr);
    AutoEnableProfiling enableProfiling{profile_h};

    // search loop here
    int grp_tl_ln[4]       = {8, 16, 32};
    int tile_sz1[4]        = {8, 16, 32, 64};
    int tile_sz0[4]        = {8, 16, 32, 64};
    int out_pix_tile_sz[3] = {1, 2, 4};
    int n_out_tiles_rg[5]  = {1, 2, 4, 8};
    int n_in_tiles_rg[3]   = {1, 2, 4};
    int n_in_stacks_sz[2]  = {1, 2};
    int in_tiles[4]        = {64, 128, 256, 2048};

    double min_proc_time = std::numeric_limits<double>::max();

    size_t run_counter = 0, failed_counter = 0;

    int out_pix_tl_cnt = 3; // out_pix_tile_sz[1];
    int n_out_tls      = 4;
    int n_in_tls       = 3;
    int stack_cnt      = std::min(problem.GetBatchSize(), static_cast<std::size_t>(2));
    int n_tile0_sz     = 4;
    int n_tile1_sz     = 4;

    if(problem.GetOutWidth() >= 16)
    {
        tile_sz0[0] = 16;
        tile_sz0[1] = 32;
        n_tile0_sz  = 2;
    }

    if(problem.GetOutHeight() >= 16)
    {
        tile_sz1[0] = 16;
        tile_sz1[1] = 32;
        n_tile1_sz  = 2;
    }

    int n_tiles_cnt = n_tile0_sz * n_tile1_sz;

    size_t report_inteval = 25;

    long long runs_left = 0, total_runs = 0;

    if(problem.GetWeightsWidth() == 1 && problem.GetWeightsHeight() == 1 &&
       problem.GetGroupCount() ==
           1) // Group conv: None 1x1 version yet, fallback to universal kernel.
    {
        MIOPEN_LOG_W("Searching the best solution in the 4 dim space. Please, be patient...");
        int n_grp_tiles0 = 3;
        result.grp_tile1 = 1;
        result.in_tile1  = 1;
        result.in_tile0  = 1;
        report_inteval   = 5;

        // Add 1x1_stride : no padding support yet
        if(problem.GetInDataType() == miopenFloat && problem.IsDirectionForward() &&
           problem.GetInChannels() % 16 == 0 && problem.GetOutChannels() % 16 == 0)
        {

            // unsigned N_LCL_IN_MAPS = result.n_in_data_tiles;
            n_in_tiles_rg[0] = 0;
            n_in_tiles_rg[1] = 3;
            n_in_tls         = 4;

            // int N_LCL_OUT_MAPS = result.n_out_pix_tiles;
            n_out_tiles_rg[0] = 4;
            n_out_tiles_rg[1] = 6;
            // 0 or 1
            out_pix_tl_cnt = 3;
            // unsigned CHEAT_SHADER_COMPILER = result.out_pix_tile0;
            out_pix_tile_sz[0] = 0;
            out_pix_tile_sz[1] = 1;
            n_out_tls          = (n_out_tiles_rg[1] - n_out_tiles_rg[0] + 1);
            n_grp_tiles0       = 1;
            grp_tl_ln[0]       = 64;

            runs_left =
                static_cast<long long>(out_pix_tl_cnt) * n_out_tls * n_in_tls * (n_grp_tiles0 + 1);
            total_runs = runs_left;

            result.out_pix_tile1 = 1;
        }
        else
        {
            int i_sz = problem.GetInWidth() * problem.GetInHeight();
            if(problem.GetKernelStrideW() == 1)
            {
                out_pix_tl_cnt = (i_sz & 1) != 0 ? 1 : (i_sz & 0x3) != 0 ? 2 : 3;
            }
            else
            {
                if(problem.IsDirectionForward())
                {
                    out_pix_tl_cnt = (problem.GetOutWidth() & 1) != 0 ? 1 : 2;
                }
                else
                {
                    out_pix_tl_cnt =
                        (((problem.GetOutWidth() & 1) != 0) || ((problem.GetInWidth() & 1) != 0))
                            ? 1
                            : 2;
                }
            }
            out_pix_tile_sz[0] = 1;
            out_pix_tile_sz[1] = 2;
            out_pix_tile_sz[2] = 4;

            n_out_tiles_rg[0] = 2;
            n_out_tiles_rg[1] = (problem.GetOutChannels() % 64 == 0)   ? 6
                                : (problem.GetOutChannels() % 32 == 0) ? 5
                                                                       : 4;

            n_in_tiles_rg[0] = 2;
            n_in_tiles_rg[1] = (problem.GetInChannels() % 8 == 0) ? 3 : 2;

            grp_tl_ln[0]     = 64;
            grp_tl_ln[1]     = 128;
            grp_tl_ln[2]     = 256;
            n_grp_tiles0     = 3;
            int n_grp_tiles1 = 1;

            int n_grp_tiles = n_grp_tiles1 * n_grp_tiles0;
            n_out_tls       = (n_out_tiles_rg[1] - n_out_tiles_rg[0] + 1);
            n_in_tls        = 2;
            runs_left = static_cast<long long>(n_grp_tiles) * out_pix_tl_cnt * n_out_tls * n_in_tls;
            total_runs = runs_left;

            result.out_pix_tile1 = 0;
        }

        int version = result.out_pix_tile1;

        for(int g0 = 0; g0 < n_grp_tiles0; ++g0)
        {
            result.grp_tile0 = grp_tl_ln[g0];

            // out pix 0
            for(int o_t = n_out_tiles_rg[0]; o_t <= n_out_tiles_rg[1]; ++o_t)
            {
                result.n_out_pix_tiles = (1 << o_t);
                for(int l = 0; l < out_pix_tl_cnt; ++l)
                {
                    result.out_pix_tile0 = out_pix_tile_sz[l];
                    if(version == 0 &&
                       ((result.n_out_pix_tiles == 32 && result.out_pix_tile0 >= 4) ||
                        (result.n_out_pix_tiles == 64 && result.out_pix_tile0 >= 2)))
                    {
                        continue;
                    }

                    for(int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1]; ++i_t)
                    {
                        if(version == 1)
                        {
                            result.n_in_data_tiles = in_tiles[i_t];
                        }
                        else
                        {
                            result.n_in_data_tiles = (1 << i_t);
                        }

                        const auto ret =
                            MeasurePerfConfig<Tgpu, ConvOclDirectFwd1x1>(profile_h,
                                                                         bot_ocl_ptr,
                                                                         top_ocl_ptr,
                                                                         wei_ocl_ptr,
                                                                         bias_ocl_ptr,
                                                                         processing_time,
                                                                         ctx,
                                                                         problem,
                                                                         result);
                        --runs_left;
                        if(ret != 0)
                        {
                            ++failed_counter;
                            continue;
                        }

                        is_passed = true;
                        MIOPEN_LOG_T("##(n_current, n_failed, n_runs_total): "
                                     << run_counter << " / " << failed_counter << " / "
                                     << total_runs << " elapsed_time: " << processing_time
                                     << " best_time: " << processing_time << ", " << result);

                        if(processing_time < min_proc_time)
                        {
                            MIOPEN_LOG_I('#' << run_counter << ' ' << processing_time << " < "
                                             << min_proc_time << ' ' << result);
                            min_proc_time = processing_time;
                            candidate     = result;
                        }

                        if(run_counter % report_inteval == 0)
                        {
                            MIOPEN_LOG_W("Runs left: "
                                         << runs_left << ", "
                                         << "min time so far: " << min_proc_time << ", "
                                         << "curr time: " << processing_time << ' ' << result);
                        }
                        run_counter++;
                    }
                }
            }
        }
    }
    else
    {
        MIOPEN_LOG_W("Searching the best solution in the 9 dim space. Please, be patient...");
        runs_left = static_cast<long long>(n_tiles_cnt) * out_pix_tl_cnt * out_pix_tl_cnt *
                    n_out_tls * n_in_tls * stack_cnt;
        total_runs = runs_left;

        // tile1
        for(int j = 0; j < n_tile1_sz; ++j)
        {
            int tile_sz[3]  = {8, 16, 32};
            result.in_tile1 = tile_sz1[j];
            if(problem.GetOutHeight() * 2 <= result.in_tile1 && result.in_tile1 > tile_sz[0])
            {
                --runs_left;
                continue;
            }

            // tile 0
            for(int i = 0; i < n_tile0_sz; ++i)
            {
                result.in_tile0 = tile_sz0[i];
                if((problem.GetOutWidth() * 2 <= result.in_tile0 && result.in_tile0 > tile_sz[0]))
                {
                    --runs_left;
                    continue;
                }
                if(problem.GetOutHeight() > 16 && problem.GetOutWidth() > 16 &&
                   ((result.in_tile1 == 8 && result.in_tile0 == 8) ||
                    (result.grp_tile0 == 8 && result.grp_tile1 == 8)))
                {
                    --runs_left;
                    continue;
                }
                if(problem.GetOutWidth() > 32 && result.in_tile1 > result.in_tile0)
                {
                    --runs_left;
                    continue;
                }
                // out pix 1

                for(int k = 0; k < out_pix_tl_cnt; ++k)
                {
                    result.out_pix_tile1 = out_pix_tile_sz[k];
                    result.grp_tile1     = result.in_tile1 / result.out_pix_tile1;
                    if(result.out_pix_tile1 > result.in_tile1 || result.grp_tile1 < 8)
                    {
                        --runs_left;
                        continue;
                    }
                    // out pix 0

                    for(int l = 0; l < out_pix_tl_cnt; ++l)
                    {
                        result.out_pix_tile0 = out_pix_tile_sz[l];
                        result.grp_tile0     = result.in_tile0 / result.out_pix_tile0;

                        if(result.out_pix_tile0 > result.in_tile0 || result.grp_tile0 < 8)
                        {
                            --runs_left;
                            continue;
                        }

                        for(int o_t = 0; o_t < n_out_tls; ++o_t)
                        {
                            result.n_out_pix_tiles = n_out_tiles_rg[o_t];
                            if(problem.GetOutChannels() < result.n_out_pix_tiles)
                            {
                                --runs_left;
                                continue;
                            }

                            for(int i_t = 0; i_t < n_in_tls; ++i_t)
                            {
                                result.n_in_data_tiles = n_in_tiles_rg[i_t];
                                if(problem.GetInChannels() < result.n_in_data_tiles)
                                {
                                    --runs_left;
                                    continue;
                                }

                                for(int s = 0; s < stack_cnt; ++s)
                                {

                                    result.n_stacks = n_in_stacks_sz[s];
                                    if(result.n_stacks > problem.GetBatchSize())
                                    {
                                        --runs_left;
                                        continue;
                                    }

                                    if(result.out_pix_tile1 * result.out_pix_tile0 *
                                           result.n_out_pix_tiles * result.n_stacks >=
                                       128)
                                    {
                                        --runs_left;
                                        continue;
                                    }

                                    const auto ret =
                                        MeasurePerfConfig<Tgpu, ConvOclDirectFwd>(profile_h,
                                                                                  bot_ocl_ptr,
                                                                                  top_ocl_ptr,
                                                                                  wei_ocl_ptr,
                                                                                  bias_ocl_ptr,
                                                                                  processing_time,
                                                                                  ctx,
                                                                                  problem,
                                                                                  result);

                                    --runs_left;
                                    if(ret != 0)
                                    {
                                        ++failed_counter;
                                        continue;
                                    }

                                    is_passed = true;
                                    MIOPEN_LOG_T(
                                        "##(n_current, n_failed, n_runs_total): "
                                        << run_counter << " / " << failed_counter << " / "
                                        << total_runs << " elapsed_time: " << processing_time
                                        << " best_time: " << processing_time << ", " << result);

                                    if(processing_time < min_proc_time)
                                    {
                                        MIOPEN_LOG_I('#' << run_counter << ' ' << processing_time
                                                         << " < " << min_proc_time << ' '
                                                         << result);
                                        min_proc_time = processing_time;
                                        candidate     = result;
                                    }

                                    if(run_counter % report_inteval == 0)
                                    {
                                        MIOPEN_LOG_W("Runs left: "
                                                     << runs_left << ", "
                                                     << "min time so far: " << min_proc_time << ", "
                                                     << "curr time: " << processing_time << ' '
                                                     << result);
                                    }
                                    run_counter++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Compare search results vs. default performance config.
    {
        int ret                   = -1;
        double default_time       = std::numeric_limits<double>::max();
        const auto default_config = GetDefaultPerformanceConfig(ctx, problem);
        if(problem.GetWeightsWidth() == 1 && problem.GetWeightsHeight() == 1 &&
           problem.GetGroupCount() ==
               1) // Group conv: None 1x1 version yet, fallback to universal kernel.
        {
            ret = MeasurePerfConfig<Tgpu, ConvOclDirectFwd1x1>(profile_h,
                                                               bot_ocl_ptr,
                                                               top_ocl_ptr,
                                                               wei_ocl_ptr,
                                                               bias_ocl_ptr,
                                                               default_time,
                                                               ctx,
                                                               problem,
                                                               default_config);
        }
        else
        {
            ret = MeasurePerfConfig<Tgpu, ConvOclDirectFwd>(profile_h,
                                                            bot_ocl_ptr,
                                                            top_ocl_ptr,
                                                            wei_ocl_ptr,
                                                            bias_ocl_ptr,
                                                            default_time,
                                                            ctx,
                                                            problem,
                                                            default_config);
        }
        if(ret == 0)
        {
            is_passed = true;
            MIOPEN_LOG_W("Default run, min time so far: " << min_proc_time << ", default time: "
                                                          << default_time << ' ' << default_config);
            if(min_proc_time > default_time)
            {
                MIOPEN_LOG_W("* * * Default time < min time, using default config * * *");
                min_proc_time = default_time;
                candidate     = default_config;
            }
        }
        MIOPEN_LOG_W("...Score: " << (default_time / min_proc_time));
    }
    result = candidate;

    if(!is_passed)
        MIOPEN_THROW("Search failed for ConvOclDirectFwdLegacyExhaustiveSearch");
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
