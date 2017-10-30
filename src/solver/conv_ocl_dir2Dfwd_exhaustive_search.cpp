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

#include "miopen/allocator.hpp"
#include "miopen/db.hpp"
#include "miopen/handle.hpp"
#include "miopen/legacy_exhaustive_search.hpp"
#include "miopen/mlo_utils.hpp"
#include "miopen/solver.hpp"

namespace miopen {
namespace solver {

std::unique_ptr<PerformanceConfig>
ConvOclDirectFwdLegacyExhaustiveSearch::PerformanceConfigImpl() const
{
    return make_unique<LegacyPerformanceConfig>();
}

void LegacyPerformanceConfig::CopyTo(ConvSolution& iud) const
{
    iud.grp_tile0       = grp_tile0;
    iud.grp_tile1       = grp_tile1;
    iud.in_tile0        = in_tile0;
    iud.in_tile1        = in_tile1;
    iud.out_pix_tile0   = out_pix_tile0;
    iud.out_pix_tile1   = out_pix_tile1;
    iud.n_out_pix_tiles = n_out_pix_tiles;
    iud.n_in_data_tiles = n_in_data_tiles;
    iud.n_stacks        = n_stacks;
}

void LegacyPerformanceConfig::Serialize(std::ostream& stream) const
{
    static const auto sep = ',';

    // clang-format off
    stream     << grp_tile1
        << sep << grp_tile0
        << sep << in_tile1
        << sep << in_tile0
        << sep << out_pix_tile1
        << sep << out_pix_tile0
        << sep << n_out_pix_tiles
        << sep << n_in_data_tiles
        << sep << n_stacks; // clang-format on
}

static bool DeserializeField(const char separator, std::istream& from, int& to)
{
    std::string part;

    if(!std::getline(from, part, separator))
        return false;

    const auto start = part.c_str();
    char* end;
    to = std::strtol(start, &end, 10);
    return start != end;
}

bool LegacyPerformanceConfig::Deserialize(const std::string& from)
{
    std::istringstream ss(from);
    LegacyPerformanceConfig temp;
    const char sep = ',';

    const auto succeded = // clang-format off
        DeserializeField(sep, ss, temp.grp_tile1) &&
        DeserializeField(sep, ss, temp.grp_tile0) &&
        DeserializeField(sep, ss, temp.in_tile1) &&
        DeserializeField(sep, ss, temp.in_tile0) &&
        DeserializeField(sep, ss, temp.out_pix_tile1) &&
        DeserializeField(sep, ss, temp.out_pix_tile0) &&
        DeserializeField(sep, ss, temp.n_out_pix_tiles) &&
        DeserializeField(sep, ss, temp.n_in_data_tiles) &&
        DeserializeField(sep, ss, temp.n_stacks); // clang-format on

    if(!succeded)
        return false;

    *this = temp;
    return true;
}

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
bool LegacyPerformanceConfig::LegacyDeserialize(const std::string& from)
{
    std::istringstream ss(from);
    LegacyPerformanceConfig temp;
    const char sep = '.';

    const auto succeded = // clang-format off
        DeserializeField(sep, ss, temp.grp_tile1) &&
        DeserializeField(sep, ss, temp.grp_tile0) &&
        DeserializeField(sep, ss, temp.in_tile1) &&
        DeserializeField(sep, ss, temp.in_tile0) &&
        DeserializeField(sep, ss, temp.out_pix_tile1) &&
        DeserializeField(sep, ss, temp.out_pix_tile0) &&
        DeserializeField(sep, ss, temp.n_out_pix_tiles) &&
        DeserializeField(sep, ss, temp.n_in_data_tiles) &&
        DeserializeField(sep, ss, temp.n_stacks); // clang-format on

    if(!succeded)
        return false;

    *this = temp;
    return true;
}
#endif

/*
* select defult configuration if a known configuration has not been found.
*/
void ConvOclDirectFwdLegacyExhaustiveSearch::InitPerformanceConfigImpl(
    const ConvolutionContext& params, PerformanceConfig& result_) const
{
    auto& result = dynamic_cast<LegacyPerformanceConfig&>(result_);

    //
    result.in_tile0 = (params.in_width <= 8) ? 8 : (params.in_width <= 16)
                                                       ? 16
                                                       : 32; // size of input data per ALU plane
    result.in_tile1 = (params.in_height <= 8) ? 8 : (params.in_height <= 16)
                                                        ? 16
                                                        : 8; // size of input data per ALU plane

    result.grp_tile0 = (result.in_tile0 == 8) ? 8 : 16;
    result.grp_tile1 = (result.in_tile1 == 8) ? 8 : 16;

    result.out_pix_tile0 = 2; // size of ouptput tile per wk-item (ALU))
    result.out_pix_tile1 = 2; //

    result.n_out_pix_tiles = 8; // # output pixel tiles per wk-item (ALU)
    result.n_in_data_tiles = 2; // # of blocks of different inputs in LDS

    result.n_stacks = 1; // # of diff stacks (part of batch).

    if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
    {

        // version
        if(params.forward && params.n_inputs % 16 == 0 && params.n_outputs % 16 == 0 &&
           params.kernel_stride0 == 1 && params.kernel_stride1 == 1)
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
            result.out_pix_tile0 = (i_sz & 1) ? 1 : 2;

            if(params.pad0 > 0 || params.kernel_stride0 > 1)
            {
                if(params.forward)
                {
                    result.out_pix_tile0 = (params.out_width & 1) ? 1 : 2;
                }
                else
                {
                    result.out_pix_tile0 =
                        ((params.out_width & 1) || (params.in_width & 1)) ? 1 : 2;
                }
            }

            result.n_out_pix_tiles = 16;
            result.n_in_data_tiles = 4;
            result.grp_tile0       = 64;
            // int version =
            result.out_pix_tile1 = 0;
        }
    }
}

static const std::vector<std::reference_wrapper<const Solver>>& GetImplementationsToMeasure()
{
    static const std::vector<std::reference_wrapper<const Solver>> implementations = {
        StaticContainer<ConvOclDirectFwd1x1>::Instance(),
        StaticContainer<ConvOclDirectFwdC>::Instance(),
        StaticContainer<ConvOclDirectFwd>::Instance(),
    };

    return implementations;
}

/*
* Measure the current configuration performance.
*/
static int MeasureLoop(Handle* profile_h,
                       Data_t bot_ocl_buf,
                       Data_t top_ocl_buf,
                       Data_t wei_ocl_buf,
                       Data_t bias_ocl_buf,
                       double& processing_time,
                       const ConvolutionContext& params,
                       const LegacyPerformanceConfig& result)
{
    int ret = 0;
    ConvSolution kernel_search_result;
    auto sub_search_params      = params;
    sub_search_params.do_search = false;

    for(const Solver& traits : GetImplementationsToMeasure())
    {
        if(traits.IsApplicable(params))
        {
            kernel_search_result = traits.GetSolution(params, result);

            if(kernel_search_result.Succeeded())
                break;
        }
    }

    if(!kernel_search_result.Succeeded())
    {
        return 1;
    }

    const auto kernel_params     = kernel_search_result.construction_params[0];
    std::string compiler_options = params.general_compile_options + kernel_params.comp_options;

    // Creating OCLKernel obj
    try
    {

        float padding_value = 0;

        double s = 0, e = 0;
        int iter = 1;

        if(profile_h)
        {
            processing_time = std::numeric_limits<float>::max();

            auto k = profile_h->GetKernel("",
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
            processing_time = profile_h->GetKernelTime();
        }
        else
        {
            iter = (params.n_timer_iter <= 0) ? 1 : params.n_timer_iter;

            auto k = params.GetStream().GetKernel("",
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

            params.GetStream().Finish();

            s = miopen_mach_absolute_time();

            for(int i = 0; i < iter && ret == 0; i++)
            {
                if(params.bias)
                {
                    k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
                }
                else
                {
                    k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
                }
            }

            params.GetStream().Finish();
            e = miopen_mach_absolute_time();

            processing_time = subtractTimes(e, s) / iter;
        }
    }
    catch(miopen::Exception&)
    {
        return -1;
    }

    return (ret);
}

void ConvOclDirectFwdLegacyExhaustiveSearch::Search(const ConvolutionContext& params,
                                                    PerformanceConfig& result_) const
{
    auto& result = dynamic_cast<LegacyPerformanceConfig&>(result_);

    miopen::Handle profile_h;
    double processing_time;
    std::string conf_key;
    std::string conf_val;

    int min_grp_tile0 = 16;
    int min_grp_tile1 = 16;
    // tile 0
    int min_in_tile0 = 16;
    // tile 1
    int min_in_tile1 = 16;
    // out pix 0
    int min_out_pix_tile0 = 1;
    // out pix 1
    int min_out_pix_tile1   = 1;
    int min_n_out_pix_tiles = 2;
    int min_n_in_data_tiles = 3;
    int min_n_stacks        = 1;

    // enable profiling for the handle for benchmarking
    profile_h.EnableProfiling();

    profile_h.EnableProfiling();

    // allocate tem input/output buffers
    size_t bot_sz = params.bot_sz / sizeof(float);
    std::vector<float> bot_sys_buf(bot_sz);

    for(int i = 0; i < bot_sz; i++)
    {
        bot_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
    }

    auto bot_ocl_buf = profile_h.Write(bot_sys_buf);

    size_t top_sz = params.top_sz / sizeof(float);
    std::vector<float> top_sys_buf(top_sz);

    auto top_ocl_buf = profile_h.Write(top_sys_buf);

    std::vector<float> random_top_sys_buf(top_sz);
    for(int i = 0; i < top_sz; i++)
    {
        random_top_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
    }

    size_t weights_sz = params.weights_sz / sizeof(float);
    std::vector<float> wei_sys_buf(weights_sz);
    for(int i = 0; i < weights_sz; i++)
    {
        wei_sys_buf[i] = static_cast<float>((rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
    }

    auto wei_ocl_buf = profile_h.Write(wei_sys_buf);

    std::vector<float> bias_sys_buf;
    miopen::Allocator::ManageDataPtr bias_ocl_buf = nullptr;

    if(params.bias)
    {
        size_t bias_sz = params.bias_sz / sizeof(float);
        bias_sys_buf   = std::vector<float>(bias_sz);
        for(int i = 0; i < bias_sz; i++)
        {
            bias_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
        }

        bias_ocl_buf = profile_h.Write(bias_sys_buf);
    }

    // search loop here
    int grp_tl_ln[4]       = {8, 16};
    int tile_sz[3]         = {8, 16, 32};
    int tile_sz1[3]        = {8, 16, 32};
    int tile_sz0[3]        = {8, 16, 32};
    int out_pix_tile_sz[3] = {1, 2, 4};
    int n_out_tiles_rg[5]  = {1, 2, 4, 8};
    int n_in_tiles_rg[3]   = {1, 2, 4};
    int n_in_stacks_sz[2]  = {1, 2};
    int in_tiles[4]        = {64, 128, 256, 2048};

    double min_proc_time = std::numeric_limits<float>::max();

#if 1

    size_t run_counter = 0;
    int n_grp_tiles1   = 2;
    int n_grp_tiles0   = 2;

    int out_pix_tl_cnt = 3; // out_pix_tile_sz[1];
    int n_out_tls      = 4;
    int n_in_tls       = 3;
    int stack_cnt      = std::min(params.batch_sz, 2);
    int n_tile0_sz     = 3;
    int n_tile1_sz     = 3;

    int n_grp_tiles = (n_grp_tiles1 - 1) * n_grp_tiles0;

    int n_tiles_cnt = n_tile0_sz * n_tile1_sz;

    size_t report_inteval = 100;
    //			_n_timer_iter = 250;

    long long runs_left = 0;

    if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
    {

        std::cout << "Searching the best solution in the 4 dim space. Please, be patient it may "
                     "take few minutes."
                  << std::endl;
        result.grp_tile1 = 1;
        result.in_tile1  = 1;
        result.in_tile0  = 1;
        report_inteval   = 5;

        // Add 1x1_stride : no padding support yet
        if(params.forward && params.kernel_stride0 == 1 && params.kernel_stride1 == 1 &&
           params.n_inputs % 16 == 0 && params.n_outputs % 16 == 0)
        {

            // uint N_LCL_IN_MAPS = result.n_in_data_tiles;
            n_in_tiles_rg[0] = 0;
            n_in_tiles_rg[1] = 3;
            n_in_tls         = 4;

            //					int N_LCL_OUT_MAPS = result.n_out_pix_tiles;
            n_out_tiles_rg[0] = 4;
            n_out_tiles_rg[1] = 6;
            // 0 or 1
            out_pix_tl_cnt = 3;
            //					uint CHEAT_SHADER_COMPILER =
            // result.out_pix_tile0;
            out_pix_tile_sz[0] = 0;
            out_pix_tile_sz[1] = 1;
            n_out_tls          = (n_out_tiles_rg[1] - n_out_tiles_rg[0] + 1);
            n_grp_tiles0       = 0;

            runs_left = out_pix_tl_cnt * n_out_tls * n_in_tls * (n_grp_tiles0 + 1);

            result.out_pix_tile1 = 1;
        }
        else
        {
            int i_sz = params.in_width * params.in_height;
            if(params.kernel_stride0 == 1)
            {
                out_pix_tl_cnt = (i_sz & 1) ? 1 : (i_sz & 0x3) ? 2 : 3;
            }
            else
            {
                if(params.forward)
                {
                    out_pix_tl_cnt = (params.out_width & 1) ? 1 : 2;
                }
                else
                {
                    out_pix_tl_cnt = ((params.out_width & 1) || (params.in_width & 1)) ? 1 : 2;
                }
            }
            out_pix_tile_sz[0] = 1;
            out_pix_tile_sz[1] = 2;
            out_pix_tile_sz[2] = 4;

            n_out_tiles_rg[0] = 2;
            n_out_tiles_rg[1] =
                (params.n_outputs % 64 == 0) ? 6 : (params.n_outputs % 32 == 0) ? 5 : 4;

            n_in_tiles_rg[0] = 2;
            n_in_tiles_rg[1] = (params.n_inputs % 8 == 0) ? 3 : 2;

            grp_tl_ln[0] = 64;
            grp_tl_ln[1] = 128;
            grp_tl_ln[2] = 256;
            n_grp_tiles0 = 3;
            n_grp_tiles1 = 1;

            n_grp_tiles = n_grp_tiles1 * n_grp_tiles0;
            n_out_tls   = (n_out_tiles_rg[1] - n_out_tiles_rg[0] + 1);
            n_in_tls    = 2;
            runs_left   = n_grp_tiles * out_pix_tl_cnt * n_out_tls * n_in_tls;

            result.out_pix_tile1 = 0;
        }

        int version = result.out_pix_tile1;

        for(int g0 = 0; g0 <= n_grp_tiles0; ++g0)
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
                        // randomize output
                        profile_h.WriteTo(reinterpret_cast<const void*>(random_top_sys_buf.data()),
                                          top_ocl_buf,
                                          random_top_sys_buf.size() * sizeof(float));

                        const auto ret = MeasureLoop(&profile_h,
                                                     bot_ocl_buf.get(),
                                                     top_ocl_buf.get(),
                                                     wei_ocl_buf.get(),
                                                     bias_ocl_buf.get(),
                                                     processing_time,
                                                     params,
                                                     result);

                        runs_left--;
                        runs_left = (runs_left < 0) ? 0 : runs_left;

                        if(ret != 0)
                        {
                            //          std::cout << "Failed run." << std::endl;
                            continue;
                        }

#if 1
                        if(run_counter % report_inteval == 0)
                        {
                            min_proc_time = (run_counter == 0) ? processing_time : min_proc_time;
                            std::cout << "Runs left : " << runs_left << ", "
                                      << "min time so far : " << min_proc_time << ", "
                                      << "curr time : " << processing_time
#if 1
                                      << ", " << result.grp_tile1 << ", " << result.grp_tile0
                                      << ", " << result.in_tile1 << ", " << result.in_tile0 << ", "
                                      << result.out_pix_tile1 << ", " << result.out_pix_tile0
                                      << ", " << result.n_out_pix_tiles << ", "
                                      << result.n_in_data_tiles << ", " << result.n_stacks
#endif
                                      << std::endl;
                        }

#endif

                        run_counter++;

                        if(min_proc_time > processing_time)
                        {
                            min_proc_time       = processing_time;
                            min_grp_tile0       = result.grp_tile0;
                            min_grp_tile1       = result.grp_tile1;
                            min_in_tile0        = result.in_tile0;
                            min_in_tile1        = result.in_tile1;
                            min_out_pix_tile0   = result.out_pix_tile0;
                            min_out_pix_tile1   = result.out_pix_tile1;
                            min_n_out_pix_tiles = result.n_out_pix_tiles;
                            min_n_in_data_tiles = result.n_in_data_tiles;
                            min_n_stacks        = result.n_stacks;
                        }

                    } // for (int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1]; ++i_t)
                }     // if (result.out_pix_tile0 > result.in_tile0)
            }         // for (int l = 0; l < l_l; ++l)
        }             // for (int g0 = 0; g0 < 2; ++g0)
    }
    else
    {

        std::cout << "Searching the best solution in the 9 dim space. Please, be patient it may "
                     "take few minutes."
                  << std::endl;

        runs_left = n_grp_tiles * n_tiles_cnt * out_pix_tl_cnt * out_pix_tl_cnt * n_out_tls *
                    n_in_tls * stack_cnt;

        for(int g1 = 0; g1 < n_grp_tiles1; g1++)
        {
            result.grp_tile1 = grp_tl_ln[g1];
            for(int g0 = 0; g0 < n_grp_tiles0; ++g0)
            {
                result.grp_tile0 = grp_tl_ln[g0];

                if(result.grp_tile0 < result.grp_tile1)
                {
                    runs_left--;
                    runs_left = (runs_left < 0) ? 0 : runs_left;
                    continue;
                }

                // tile1
                for(int j = 0; j < n_tile1_sz; ++j)
                {
                    result.in_tile1 = tile_sz1[j];
                    if(params.out_height * 2 <= result.in_tile1 && result.in_tile1 > tile_sz[0])
                    {
                        runs_left--;
                        runs_left = (runs_left < 0) ? 0 : runs_left;
                        continue;
                    }

                    // tile 0
                    for(int i = 0; i < n_tile0_sz; ++i)
                    {
                        result.in_tile0 = tile_sz0[i];
                        if((params.out_width * 2 <= result.in_tile0 &&
                            result.in_tile0 > tile_sz[0]))
                        {
                            runs_left--;
                            runs_left = (runs_left < 0) ? 0 : runs_left;
                            continue;
                        }
                        if(params.out_height > 16 && params.out_width > 16 &&
                           ((result.in_tile1 == 8 && result.in_tile0 == 8) ||
                            (result.grp_tile0 == 8 && result.grp_tile1 == 8)))
                        {
                            runs_left--;
                            runs_left = (runs_left < 0) ? 0 : runs_left;
                            continue;
                        }
                        if(params.out_width > 32 && result.in_tile1 > result.in_tile0)
                        {
                            runs_left--;
                            runs_left = (runs_left < 0) ? 0 : runs_left;
                            continue;
                        }
                        // out pix 1

                        for(int k = 0; k < out_pix_tl_cnt; ++k)
                        {
                            result.out_pix_tile1 = out_pix_tile_sz[k];
                            if(result.out_pix_tile1 > result.in_tile1)
                            {
                                runs_left--;
                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                continue;
                            }
                            // out pix 0

                            for(int l = 0; l < out_pix_tl_cnt; ++l)
                            {
                                result.out_pix_tile0 = out_pix_tile_sz[l];

                                if(result.out_pix_tile0 > result.in_tile0)
                                {
                                    runs_left--;
                                    runs_left = (runs_left < 0) ? 0 : runs_left;
                                    continue;
                                }

                                for(int o_t = 0; o_t < n_out_tls; ++o_t)
                                {
                                    result.n_out_pix_tiles = n_out_tiles_rg[o_t];
                                    if(params.n_outputs < result.n_out_pix_tiles)
                                    {
                                        runs_left--;
                                        runs_left = (runs_left < 0) ? 0 : runs_left;
                                        continue;
                                    }

                                    for(int i_t = 0; i_t < n_in_tls; ++i_t)
                                    {
                                        result.n_in_data_tiles = n_in_tiles_rg[i_t];
                                        if(params.n_inputs < result.n_in_data_tiles)
                                        {
                                            runs_left--;
                                            runs_left = (runs_left < 0) ? 0 : runs_left;
                                            continue;
                                        }

                                        for(int s = 0; s < stack_cnt; ++s)
                                        {

                                            result.n_stacks = n_in_stacks_sz[s];
                                            if(result.n_stacks > params.batch_sz)
                                            {
                                                runs_left--;
                                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                                continue;
                                            }

                                            if(result.out_pix_tile1 * result.out_pix_tile0 *
                                                   result.n_out_pix_tiles * result.n_stacks >=
                                               128)
                                            {
                                                runs_left--;
                                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                                continue;
                                            }
#if 0
											if (run_counter != 0 &&
												run_counter % report_inteval == 0)
											{
												std::cout
													<< "Runs left : " << runs_left << ", "
													<< "min time so far : " << min_proc_time << ", "
													<< "curr time : " << processing_time
#if 1
													<< ", " << result.grp_tile1 << ", "
													<< result.grp_tile0 << ", " << result.in_tile1
													<< ", " << result.in_tile0 << ", "
													<< result.out_pix_tile1 << ", "
													<< result.out_pix_tile0 << ", "
													<< result.n_out_pix_tiles << ", "
													<< result.n_in_data_tiles << ", "
													<< result.n_stacks
#endif
													<< std::endl;
										}

#endif

                                            const auto ret = MeasureLoop(&profile_h,
                                                                         bot_ocl_buf.get(),
                                                                         top_ocl_buf.get(),
                                                                         wei_ocl_buf.get(),
                                                                         bias_ocl_buf.get(),
                                                                         processing_time,
                                                                         params,
                                                                         result);

                                            runs_left--;
                                            runs_left = (runs_left < 0) ? 0 : runs_left;

                                            if(ret != 0)
                                            {
                                                //				std::cout << "Failed
                                                //run."
                                                //<< std::endl;
                                                continue;
                                            }

#if 1
                                            if(run_counter % report_inteval == 0)
                                            {
                                                min_proc_time = (run_counter == 0) ? processing_time
                                                                                   : min_proc_time;

                                                std::cout
                                                    << "Runs left : " << runs_left << ", "
                                                    << "min time so far : " << min_proc_time << ", "
                                                    << "curr time : " << processing_time
#if 1
                                                    << ", " << result.grp_tile1 << ", "
                                                    << result.grp_tile0 << ", " << result.in_tile1
                                                    << ", " << result.in_tile0 << ", "
                                                    << result.out_pix_tile1 << ", "
                                                    << result.out_pix_tile0 << ", "
                                                    << result.n_out_pix_tiles << ", "
                                                    << result.n_in_data_tiles << ", "
                                                    << result.n_stacks
#endif
                                                    << std::endl;
                                            }

#endif

                                            run_counter++;

                                            if(min_proc_time > processing_time)
                                            {
                                                min_proc_time       = processing_time;
                                                min_grp_tile0       = result.grp_tile0;
                                                min_grp_tile1       = result.grp_tile1;
                                                min_in_tile0        = result.in_tile0;
                                                min_in_tile1        = result.in_tile1;
                                                min_out_pix_tile0   = result.out_pix_tile0;
                                                min_out_pix_tile1   = result.out_pix_tile1;
                                                min_n_out_pix_tiles = result.n_out_pix_tiles;
                                                min_n_in_data_tiles = result.n_in_data_tiles;
                                                min_n_stacks        = result.n_stacks;
                                            }

                                        } // for (int s = 0; s < 3; ++s)
                                    } // for (int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1];
                                      // ++i_t)
                                }     // if (result.out_pix_tile0 > result.in_tile0)
                            }         // for (int l = 0; l < l_l; ++l)
                        }             // for (int k = 0; k < k_l; ++k)
                    }                 // for (int i = 0; i < 3; ++i)
                }                     // for (int j = 0; j < 3; ++j)
            }                         // for (int g0 = 0; g0 < 2; ++g0)
        }                             // for (int g1 = 0; g1 < 2; g1++)
    }

    std::cout << std::endl << "Score: " << min_proc_time << std::endl;
#endif
    result.grp_tile0       = min_grp_tile0;
    result.grp_tile1       = min_grp_tile1;
    result.in_tile0        = min_in_tile0;
    result.in_tile1        = min_in_tile1;
    result.out_pix_tile0   = min_out_pix_tile0;
    result.out_pix_tile1   = min_out_pix_tile1;
    result.n_out_pix_tiles = min_n_out_pix_tiles;
    result.n_in_data_tiles = min_n_in_data_tiles;
    result.n_stacks        = min_n_stacks;

    profile_h.EnableProfiling(false);
}

} // namespace solver
} // namespace miopen
