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

#include <sstream>
#include <limits>
#include <cassert>

#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1U_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1U_SEARCH_OPTIMIZED)

namespace miopen {
namespace solver {

static inline bool UseSubsample(const ConvolutionContext& c)
{
    return (c.kernel_stride0 > 1 || c.kernel_stride1 > 1) && c.direction.IsForward();
}

static inline bool UseUpsample(const ConvolutionContext& c)
{
    return (c.kernel_stride0 > 1 || c.kernel_stride1 > 1) && c.direction.IsBackwardData();
}

/// After 2x subsampling kernel, image size on asm kernel input becomes 4x (2*2) smaller.
/// As padding = 0, we can simply re-use output image size (no computations required).
/// \note For backward convolutions input image size is held in
/// out_height/out_width and vice versa.
static inline int AsmImgHeight(const ConvolutionContext& c)
{
    return UseSubsample(c) ? c.out_height : c.in_height;
}

static inline int AsmImgWidth(const ConvolutionContext& c)
{
    return UseSubsample(c) ? c.out_width : c.in_width;
}

/// \todo move to separate header and use in other solvers.
template <int L, int H>
inline static bool IsTwoPower(const int v)
{
    static_assert(L <= H, "L <= H");
    if(((v - 1) & v) != 0)
        return false;
    return L <= v && v <= H;
}

template <int L, int H>
inline static bool NextTwoPower(int& v)
{
    static_assert((((L - 1) & L) == 0), "L is not power of 2");
    static_assert((((H - 1) & H) == 0), "H is not power of 2");
    assert((IsTwoPower<L, H>(v)));
    if(v == H)
    {
        v = L;
        return true;
    }
    v *= 2;
    return false;
}

template <int L, int H>
inline static bool IsLinear(const int v)
{
    static_assert(L <= H, "L <= H");
    return L <= v && v <= H;
}

template <int L, int H>
inline static bool NextLinear(int& v)
{
    assert((IsLinear<L, H>(v)));
    if(H == v)
    {
        v = L;
        return true;
    }
    ++v;
    return false;
}

// This range is like regular range [0,4,8...32], but 1 is used instead of 0.
inline static bool Is_1_4_8_12__32(const int& v)
{
    return v == 1 || (v % 4 == 0 && IsLinear<1, 8>(v / 4));
}

inline static bool Next_1_4_8_12__32(int& v)
{
    assert(Is_1_4_8_12__32(v));
    int tmp        = v / 4;
    const bool ret = NextLinear<0, 8>(tmp);
    v              = ret ? 1 : tmp * 4;
    return ret;
}

inline static bool Is_1_4(const int& v) { return v == 1 || v == 4; }

bool PerformanceConfigConvAsm1x1U::SetNextValue()
{
    // Increment with wrap-around:
    do
    {
        if(!NextLinear<1, 4>(read_size))
            break;
        if(!Next_1_4_8_12__32(k_mult))
            break;
        if(!NextLinear<1, 16>(chunks_per_wave))
            break;
        if(!NextTwoPower<1, 64>(chunk_size))
            break;
        if(!NextLinear<1, 8>(n_blocks_per_wave))
            break;
        if(!NextLinear<1, 8>(waves_in_group))
            break;
        // All the fields of performance config have wrapped around.
        return false;
    } while(false);
    return true;
}

PerformanceConfigConvAsm1x1U::PerformanceConfigConvAsm1x1U(int read_size_,
                                                           int k_mult_,
                                                           int chunks_per_wave_,
                                                           int chunk_size_,
                                                           int n_blocks_per_wave_,
                                                           int waves_in_group_,
                                                           bool use_spare_set_)
    : read_size(read_size_),
      k_mult(k_mult_),
      chunks_per_wave(chunks_per_wave_),
      chunk_size(chunk_size_),
      n_blocks_per_wave(n_blocks_per_wave_),
      waves_in_group(waves_in_group_),
      use_spare_set(use_spare_set_)
{
}

inline bool PerformanceConfigConvAsm1x1U::
operator==(const PerformanceConfigConvAsm1x1U& other) const
{
    // clang-format off
    return read_size == other.read_size
        && k_mult == other.k_mult
        && chunks_per_wave == other.chunks_per_wave
        && chunk_size == other.chunk_size
        && n_blocks_per_wave == other.n_blocks_per_wave
        && waves_in_group == other.waves_in_group
        && use_spare_set == other.use_spare_set; // clang-format on
}

bool PerformanceConfigConvAsm1x1U::IsValidValue() const
{
    // clang-format off
    return IsLinear<1,4>(read_size)
        && Is_1_4_8_12__32(k_mult)
        && IsLinear<1,16>(chunks_per_wave)
        && IsTwoPower<1,64>(chunk_size)
        && IsLinear<1,8>(n_blocks_per_wave)
        && IsLinear<1,8>(waves_in_group); // clang-format on
}

bool PerformanceConfigConvAsm1x1U::IsValidForProblem(const ConvolutionContext& config) const
{
    if(!IsValidValue())
        return false;
    if(!(read_size <= chunks_per_wave))
        return false;
    if(!(waves_in_group <= config.n_inputs))
        return false;
    if(!(k_mult <= config.n_outputs))
        return false;
    const int in_gprs  = chunks_per_wave * n_blocks_per_wave;
    const int acc_gprs = in_gprs * k_mult;
    const int vgprs    = 4 + 2 * in_gprs + acc_gprs;
    if(!(vgprs < 256))
        return false;
    const int max_waves_per_CU = (256 / vgprs) * 4;
    if(!(max_waves_per_CU >= waves_in_group))
        return false;
    const int sgprs = 24 + 2 * k_mult;
    if(!(sgprs < 102)) /// \todo This is valid for Gfx8 and Gfx9. Check for newer parts.
        return false;
    const int total_n_blocks = (config.batch_sz + GetNPerGpr() - 1) / GetNPerGpr();
    if(!(n_blocks_per_wave <= total_n_blocks))
        return false;
    const int img_hw       = config.out_height * config.out_width;
    const int total_chunks = (img_hw + chunk_size - 1) / chunk_size;
    if(!(chunks_per_wave <= total_chunks))
        return false;
    if(config.direction.IsBackwardData() && !(config.n_outputs % k_mult == 0))
        return false;
    return true;
}

bool PerformanceConfigConvAsm1x1U::IsValid(const ConvolutionContext& config) const
{
    if(!IsValidForProblem(config))
        return false;
    if(!miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1U_SEARCH_OPTIMIZED{})) // clang-format off
    {
        // Narrow search space in optimized mode.
        if (! ((use_spare_set ? Is_1_4(k_mult) : IsTwoPower<16,32>(k_mult))
            && IsLinear<1,8>(chunks_per_wave)
            && (use_spare_set ? Is_1_4(chunk_size) : IsTwoPower<16,64>(chunk_size))
            && IsLinear<1,4>(n_blocks_per_wave)
            && IsLinear<1,4>(waves_in_group)))
            return false;
    } // clang-format on
    return true;
}

void PerformanceConfigConvAsm1x1U::EuristicInit(const ConvolutionContext& config)
{
    read_size         = 4;
    k_mult            = 16;
    chunks_per_wave   = 1;
    chunk_size        = 16;
    n_blocks_per_wave = 1;
    waves_in_group    = 1;

    if(!IsValidForProblem(config))
    {
        MIOPEN_LOG_I("!IsValidForProblem(): " << ToString() << ". Conservative re-init...");
        read_size         = 1;
        k_mult            = 1;
        chunks_per_wave   = 1;
        chunk_size        = 1;
        n_blocks_per_wave = 1;
        waves_in_group    = 1;
        assert(IsValidForProblem(config));
    }
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceConfigConvAsm1x1U::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvAsm1x1U
ConvAsm1x1U::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvAsm1x1U pp;
    pp.EuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsm1x1U::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                           const PerformanceConfigConvAsm1x1U& c) const
{
    return c.IsValidValue() && c.IsValidForProblem(problem);
}

bool ConvAsm1x1U::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.use_asm_kernels)
    {
        return false;
    }
    if(!(params.rmv == rocm_meta_version::V3 || params.rmv == rocm_meta_version::AMDHSA_1_0))
    {
        return false;
    }
    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos && name.find("gfx9") == std::string::npos)
    {
        return false;
    }
    assert(params.weights_layout.length() == 0); // _weights_layout is not supported yet
    // clang-format off
    bool ok = (params.pad0 == 0         // -q  pad_w
        && params.pad1 == 0             // -p  pad_h
        && params.kernel_stride0 <= 2   // -u  stride_w
        && params.kernel_stride0 == params.kernel_stride1
        && params.kernel_size0 == 1     // -x  S wei_w
        && params.kernel_size1 == 1     // -y  R wei_h
        && params.kernel_dilation0 == 1
        && params.kernel_dilation1 == 1
        && params.bias == 0
        && params.float_size == 32
        && params.in_layout == "NCHW");
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    if (miopen::IsEnabled(MIOPEN_DEBUG_FIND_FIRST_CONV{})
        && params.kernel_stride0 > 1)
    {
        /// Disabled asm_1x1u for stride=2 due to the overhead of
        /// Up/Subsampler and SetTensor for UpSampler. (Refer to issue #940).
        return false;
    }
    /// \todo Ilya: The checks below look adequate but needs to be double-checked.
    {
        const long input_line_size = 4 * params.in_width;
        const long input_feature_map_size = input_line_size * params.in_height;
        const long input_stack_size = input_feature_map_size * params.n_inputs;
        if (! (input_stack_size < (1U << 24)))
            return false;
    }
    {
        const long output_line_size = 4 * params.out_width;
        const long output_feature_map_size = output_line_size * params.out_height;
        const long output_stack_size = output_feature_map_size * params.n_outputs;
        if (! (output_stack_size < (1U << 24)))
            return false;
    }
    // Check limits:
    auto h_w = static_cast<long>(AsmImgHeight(params)) * AsmImgWidth(params);
    const auto r_s     = static_cast<long>(params.kernel_size1) * params.kernel_size0;
    const auto c_h_w   = static_cast<long>(params.n_inputs) * h_w;    // C*H*W
    const auto k_h_w   = static_cast<long>(params.n_outputs) * h_w;   // K*H*W
    const auto n_c_h_w = static_cast<long>(params.batch_sz) * c_h_w;  // N*C*H*W
    const auto n_k_h_w = static_cast<long>(params.batch_sz) * k_h_w;  // N*K*H*W
    const auto c_k_r_s = static_cast<long>(params.n_inputs) * params.n_outputs * r_s; // C*K*R*S
    ok = params.batch_sz < std::pow(2, 16)      // -n   N batch_size
         && params.n_inputs < std::pow(2, 16)   // -c   C input_channels
         && params.n_outputs < std::pow(2, 16)  // -k   K output_channels
         && c_h_w < std::pow(2, 24)
         && k_h_w < std::pow(2, 24)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29); // clang-format on
    return ok;
}

bool ConvAsm1x1U::IsFast(const ConvolutionContext&) const { return true; }

static int divide_round_plus_inf(const int x, const int y)
{
    assert(x >= 0 && y > 0);
    if(x % y != 0)
        return x / y + 1;
    return x / y;
}

ConvSolution ConvAsm1x1U::GetSolution(const ConvolutionContext& params,
                                      const PerformanceConfigConvAsm1x1U& config,
                                      const bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;

    std::ostringstream options;

    result.workspce_sz = 0;

    KernelInfo kernel;

    if(UseSubsample(params) || UseUpsample(params))
    {
        // subsampled input, in_height equals to image size after downsampling
        int in_batch_stride = AsmImgWidth(params) * AsmImgHeight(params) *
                              (UseSubsample(params) ? params.n_inputs : params.n_outputs);
        int write_unit =
            (AsmImgWidth(params) % 4 == 0) ? 4 : (AsmImgWidth(params) % 3 == 0)
                                                     ? 3
                                                     : (AsmImgWidth(params) % 2 == 0) ? 2 : 1;

        int n_grp0_size0 = 256;

        const auto subsample_kernel_compilation_options =
            " -DUPSAMPLE" + std::string(" -DMLO_GRP0_SZ0=") + std::to_string(n_grp0_size0) +
            std::string(" -DMLO_GRP0_SZ1=1 ") + std::string(" -DMLO_GRP0_SZ2=1 ") +
            std::string(" -DMLO_FILTER0_STRIDE0=") + std::to_string(params.kernel_stride0) +
            std::string(" -DMLO_FILTER0_STRIDE1=") + std::to_string(params.kernel_stride1) +
            std::string(" -DMLO_WRITE_UNIT=") + std::to_string(write_unit) +
            std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(params.out_channel_stride) +
            std::string(" -DMLO_OUT_STRIDE=") + std::to_string(params.out_stride) +
            std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(in_batch_stride) +
            std::string(" -DMLO_IN0_BATCH_STRIDE=") + std::to_string(params.in_batch_stride) +
            std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(params.out_batch_stride) +
            std::string(" -DMLO_IN0_CHANNEL_STRIDE=") + std::to_string(params.in_channel_stride) +
            std::string(" -DMLO_IN0_STRIDE=") + std::to_string(params.in_stride) +
            params.general_compile_options;

        kernel.l_wk.push_back(n_grp0_size0);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        // output is number of subsampled input maps
        size_t gbl_wk0 = (in_batch_stride / write_unit);
        size_t gbl_wk1 = params.batch_sz;
        size_t gbl_wk2 = 1;

        kernel.g_wk.push_back(gbl_wk0);
        kernel.g_wk.push_back(gbl_wk1);
        kernel.g_wk.push_back(gbl_wk2);

        kernel.kernel_file = "MIOpenUtilKernels3.cl";

        if(UseSubsample(params))
            kernel.kernel_name = "SubSample";
        else
            kernel.kernel_name = "UpSample";

        kernel.comp_options = subsample_kernel_compilation_options;

        assert(params.out_data_type == "FP16" || params.out_data_type == "FP32" ||
               params.out_data_type == "FP64");
        int data_len =
            (params.out_data_type == "FP16" ? 2 : (params.out_data_type == "FP32" ? 4 : 8));
        result.workspce_sz = in_batch_stride * params.batch_sz * data_len;
    }

    GenerateClangDefsym(options, "stride_h", 1);
    GenerateClangDefsym(options, "stride_w", 1);
    GenerateClangDefsym(options, "img_h", AsmImgHeight(params)); // H
    GenerateClangDefsym(options, "img_w", AsmImgWidth(params));  // W

    // Note that params.n_outputs and params.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "batch_size", params.batch_sz);       // N
    GenerateClangDefsym(options, "input_channels", params.n_inputs);   // C
    GenerateClangDefsym(options, "output_channels", params.n_outputs); // K
    GenerateClangDefsym(options, "wei_h", params.kernel_size1);        // R
    GenerateClangDefsym(options, "wei_w", params.kernel_size0);        // S
    GenerateClangDefsym(options, "pad_h", params.pad1);
    GenerateClangDefsym(options, "pad_w", params.pad0);
    GenerateClangDefsym(options, "weights_layout", params.direction.IsForward() ? 0 : 1);
    GenerateClangDefsym(
        options, "ROCM_METADATA_VERSION", (params.rmv == rocm_meta_version::V3) ? 3 : 4);

    const PerformanceConfigConvAsm1x1U* pcfg = &config;
    PerformanceConfigConvAsm1x1U fromEnv;
    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1U_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValidValue())
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1U_PERF_VALS: "
                                 "Bad format or invalid for the problem config: "
                                 << s);
                }
                else
                {
                    MIOPEN_LOG_I("Overridden from env: " << fromEnv.ToString());
                    pcfg = &fromEnv;
                }
            }
        }
    }

    GenerateClangDefsym(options, "read_size", pcfg->GetReadSize());
    GenerateClangDefsym(options, "k_mult", pcfg->GetKMult());
    GenerateClangDefsym(options, "chunks_per_wave", pcfg->GetChunksPerWave());
    GenerateClangDefsym(options, "chunk_size", pcfg->GetChunkSize());
    GenerateClangDefsym(options, "n_blocks_per_wave", pcfg->GetNBlocksPerWave());
    GenerateClangDefsym(options, "waves_in_group", pcfg->GetWavesInGroup());

    KernelInfo kinfo;
    kinfo.comp_options = options.str();

    kinfo.l_wk.clear(); // workgroupsize
    kinfo.l_wk.push_back(64 * pcfg->GetWavesInGroup());
    kinfo.l_wk.push_back(1);
    kinfo.l_wk.push_back(1);

    kinfo.g_wk.clear(); // gridsize
    const int hw_per_wave = pcfg->GetChunksPerWave() * pcfg->GetChunkSize();

    kinfo.g_wk.push_back(
        kinfo.l_wk[0] *
        divide_round_plus_inf(AsmImgHeight(params) * AsmImgWidth(params), hw_per_wave));

    kinfo.g_wk.push_back(divide_round_plus_inf(params.n_outputs, pcfg->GetKMult()));
    const int n_images_per_wave = pcfg->GetNBlocksPerWave() * pcfg->GetNPerGpr();
    kinfo.g_wk.push_back(divide_round_plus_inf(params.batch_sz, n_images_per_wave));

    kinfo.kernel_file = "conv1x1u.s";
    kinfo.kernel_name = "gcnAsmConv1x1U";

    if(UseSubsample(params))
        result.construction_params.push_back(kernel);

    result.construction_params.push_back(kinfo);

    if(UseUpsample(params))
        result.construction_params.push_back(kernel);
    return result;
}

int ConvAsm1x1U::RunAndMeasureSolution(miopen::Handle& profile_h,
                                       Data_t bot_ocl_buf,
                                       Data_t top_ocl_buf,
                                       Data_t wei_ocl_buf,
                                       Data_t bias_ocl_buf,
                                       const ConvolutionContext& params,
                                       const ConvSolution& solution,
                                       float& elapsed_time) const
{
    assert(bias_ocl_buf == nullptr);
    (void)bias_ocl_buf;
    KernelInfo k_info;

    if(UseSubsample(params))
        k_info = solution.construction_params[1];
    else if(UseUpsample(params))
        k_info = solution.construction_params[0];
    else
        k_info = solution.construction_params[0];

#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        // ConvolutionContext::general_compile_options is for OpenCL kernels
        // and thus not applicable for assembly.
        auto kernel = profile_h.AddKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);

        int unused       = 0;
        int* return_addr = nullptr;
        auto n_groups =
            static_cast<int>(params.GetStream().GetMaxComputeUnits()); // kernel needs int32

        kernel(params.batch_sz,      // N
               params.n_inputs,      // C
               AsmImgHeight(params), // H
               AsmImgWidth(params),  // W
               params.n_outputs,     // K
               n_groups,             // n_groups
               unused,
               unused,
               bot_ocl_buf,
               wei_ocl_buf,
               top_ocl_buf,
               return_addr);

        elapsed_time = profile_h.GetKernelTime();
    }
#ifdef NDEBUG
    catch(miopen::Exception&)
    {
        return -1;
    }
#endif
    return 0;
}

PerformanceConfigConvAsm1x1U ConvAsm1x1U::Search(const ConvolutionContext& context) const
{
    if(UseSubsample(context) || UseUpsample(context))
        return GenericSearch(*this, context, SearchTweak::OverrideXBufferSizeByWorkspaceSize);
    else
        return GenericSearch(*this, context);
}

} // namespace solver
} // namespace miopen
