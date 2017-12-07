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
#include <iterator>
#include <chrono>

#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1WRW_PERF_VALS)

namespace miopen {

/// \todo Factor out this (to generic search implementation)
class Timer
{
    public:
    Timer(){};
    void start() { st = std::chrono::steady_clock::now(); }
    float elapsed_ms()
    {
        capture();
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st)
            .count();
    }

    private:
    void capture() { et = std::chrono::steady_clock::now(); }
    std::chrono::time_point<std::chrono::steady_clock> st;
    std::chrono::time_point<std::chrono::steady_clock> et;
};

namespace solver {

/// \todo Factor out this (to generic search implementation)
class VirtualIteratorWrW1x1;

/// \todo Factor out this (to generic search implementation)
///
/// This container (together with corresponding iterator) provies access
/// to a set of performance configs which, by definition, must be
/// suitable for the given problem config.
///
/// It does not hold values themselves as these would take too much memory
/// but can be easily computed (and that is what iterator actually does).
///
/// The container holds problem config information instead. This info
/// is required for advancing the iterator to the next valid configuration.
/// Also it provides const_iterator, begin() and end().
class VirtualContainer1x1WrW
{
    // Valid iterator shall denote an element of a container.
    const ConvolutionContext& config;
    friend class VirtualIteratorWrW1x1;

    public:
    using const_iterator = VirtualIteratorWrW1x1;
    VirtualContainer1x1WrW(const ConvolutionContext& config_) : config(config_) {}
    VirtualIteratorWrW1x1 begin() const;
    VirtualIteratorWrW1x1 end() const;
};

// Iterator shall advance to the next valid config, i.e. the one which
// satisfies PerformanceConfig.IsValid(ProblemConfig)
class VirtualIteratorWrW1x1
    : public std::iterator<std::input_iterator_tag, PerformanceConfigConvAsmBwdWrW1x1>
{
    value_type v; // PerformanceConfigConvAsmBwdWrW1x1
    const VirtualContainer1x1WrW* container;

    static const value_type& GetMinValue();
    static const value_type& GetOutOfRangeValue();

    /// Implements begin()
    VirtualIteratorWrW1x1(const VirtualContainer1x1WrW* container_)
        : v(GetMinValue()), container(container_)
    {
        if(!IsValid())
            Next();
    }
    friend class VirtualContainer1x1WrW; // Passes itself to private ctor in order to construct
                                         // begin().
    void Next();
    bool IsValid();

    public:
    /// Implementes end() and also serves as a default ctor.
    VirtualIteratorWrW1x1() : v(GetOutOfRangeValue()), container(nullptr) {}

    bool operator!=(VirtualIteratorWrW1x1 const& other) const;
    const value_type& operator*() const { return v; }
    const value_type* operator->() const { return &v; }
    VirtualIteratorWrW1x1& operator++()
    {
        Next();
        return *this;
    }
};

inline VirtualIteratorWrW1x1 VirtualContainer1x1WrW::begin() const { return {this}; }

inline VirtualIteratorWrW1x1 VirtualContainer1x1WrW::end() const { return {}; }

const VirtualIteratorWrW1x1::value_type& VirtualIteratorWrW1x1::GetMinValue()
{
    static const value_type val(1, 1, 1, 1, 1, 1);
    return val;
}

const VirtualIteratorWrW1x1::value_type& VirtualIteratorWrW1x1::GetOutOfRangeValue()
{
    static const value_type val(-1, -1, -1, -1, -1, -1);
    return val;
}

inline bool VirtualIteratorWrW1x1::IsValid()
{
    if(!container)
        return false;
    return v.IsValid(container->config);
}

inline bool VirtualIteratorWrW1x1::operator!=(VirtualIteratorWrW1x1 const& other) const
{
    return !(v.IsEqual(other.v) && container == other.container);
}

inline static bool Inc_1_2_4_8_16(int& v)
{
    assert(v == 1 || v == 2 || v == 4 || v == 8 || v == 16);
    if(v == 16)
    {
        v = 1;
        return true;
    }
    v = v * 2;
    return false;
}

inline static bool Is_1_2_4_8_16(const int& v)
{
    return v == 1 || v == 2 || v == 4 || v == 8 || v == 16;
}

inline static bool Inc_1_2_4(int& v)
{
    assert(v == 1 || v == 2 || v == 4);
    if(v == 4)
    {
        v = 1;
        return true;
    }
    v = v * 2;
    return false;
}

inline static bool Is_1_2_4(const int& v) { return v == 1 || v == 2 || v == 4; }

void VirtualIteratorWrW1x1::Next()
{
    if(container == nullptr)
    {
        v = GetOutOfRangeValue();
        return;
    }
    do
    {
        // Increment with wrap-around:
        do
        {
            if(!Inc_1_2_4_8_16(v.c_per_gpr))
                break;
            if(!Inc_1_2_4_8_16(v.c_mult))
                break;
            if(!Inc_1_2_4_8_16(v.k_per_gpr))
                break;
            if(!Inc_1_2_4_8_16(v.k_mult))
                break;
            if(++v.read_size <= 4)
                break;
            v.read_size = 1;
            if(!Inc_1_2_4(v.n_per_gpr))
                break;
            // All the fields (components) of performance confic have wrapped around.
            // The next one is not the min (in the allowed range) but a one beyond the end.
            // Iterator is useless from now.
            v         = GetOutOfRangeValue();
            container = nullptr;
            return;
        } while(false);
    } while(!IsValid());
}

PerformanceConfigConvAsmBwdWrW1x1::PerformanceConfigConvAsmBwdWrW1x1(
    int c_per_gpr_, int c_mult_, int k_per_gpr_, int k_mult_, int read_size_, int n_per_gpr_)
    : c_per_gpr(c_per_gpr_),
      c_mult(c_mult_),
      k_per_gpr(k_per_gpr_),
      k_mult(k_mult_),
      read_size(read_size_),
      n_per_gpr(n_per_gpr_)
{
}

inline bool
PerformanceConfigConvAsmBwdWrW1x1::IsEqual(const PerformanceConfigConvAsmBwdWrW1x1& other) const
{
    // clang-format off
    return c_per_gpr == other.c_per_gpr
        && c_mult == other.c_mult
        && k_per_gpr == other.k_per_gpr
        && k_mult == other.k_mult
        && read_size == other.read_size
        && n_per_gpr == other.n_per_gpr; // clang-format on
}

bool PerformanceConfigConvAsmBwdWrW1x1::IsValidRange() const
{
    // clang-format off
    return Is_1_2_4_8_16(c_per_gpr)
        && Is_1_2_4_8_16(c_mult)
        && Is_1_2_4_8_16(k_per_gpr)
        && Is_1_2_4_8_16(k_mult)
        && (1 <= read_size && read_size <= 4)
        && Is_1_2_4(n_per_gpr); // clang-format on
}

bool PerformanceConfigConvAsmBwdWrW1x1::IsValid(const ConvolutionContext& config) const
{
    if(!IsValidRange())
        return false;
    assert((GetChunkSize() * c_per_gpr) == 16);
    if(!(k_per_gpr <= c_per_gpr))
        return false;
    if(c_mult > 1 || k_mult > 1)
    {
        assert(c_per_gpr * c_mult != 0);
        if(!(config.n_outputs % (c_per_gpr * c_mult) == 0))
            return false;
        assert(k_per_gpr * k_mult != 0);
        if(!(config.n_inputs % (k_per_gpr * k_mult) == 0))
            return false;
    }
    if(!(c_mult * k_mult * k_per_gpr + 9 + (c_mult + k_mult) * read_size * GetPipeDepth() <= 256))
    {
        return false;
    }
    return true;
}

void PerformanceConfigConvAsmBwdWrW1x1::EuristicInit(const ConvolutionContext& config)
{
    read_size = 4;
    n_per_gpr = (config.batch_sz >= 4 && (config.out_height * config.out_width) <= 128) ? 4 : 1;

    const auto c_k_256 = config.n_outputs * config.n_inputs / 256; // C*K/256
    if(c_k_256 < 2)
    {
        c_per_gpr = 1;
        c_mult    = 1;
        k_per_gpr = 1;
        k_mult    = 1;
    }
    else if(c_k_256 < (2 * 4))
    {
        c_per_gpr = 1;
        c_mult    = 2;
        k_per_gpr = 1;
        k_mult    = 2;
    }
    else if(c_k_256 < (2 * 4 * 4))
    {
        c_per_gpr = 2;
        c_mult    = 2;
        k_per_gpr = 2;
        k_mult    = 2;
    }
    else if(c_k_256 < (2 * 4 * 4 * 4))
    {
        c_per_gpr = 2;
        c_mult    = 4;
        k_per_gpr = 2;
        k_mult    = 4;
    }
    else
    {
        c_per_gpr = 4;
        c_mult    = 4;
        k_per_gpr = 4;
        k_mult    = 4;
    }

    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        c_per_gpr = 4;
        c_mult    = 1;
        k_per_gpr = 4;
        k_mult    = 1;
        assert(IsValid(config));
    }
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceConfigConvAsmBwdWrW1x1::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvAsmBwdWrW1x1
ConvAsmBwdWrW1x1::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvAsmBwdWrW1x1 pp;
    pp.EuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsmBwdWrW1x1::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                                const PerformanceConfigConvAsmBwdWrW1x1& c) const
{
    return c.IsValidRange() && c.IsValid(problem);
}

bool ConvAsmBwdWrW1x1::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.assembler_available)
    {
        return false;
    }

    if(params.n_passes)
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
    bool ok = params.pad0 == 0          // -q  pad_w
        && params.pad1 == 0             // -p  pad_h
        && params.kernel_stride0 == 1   // -u  stride_w
        && params.kernel_stride1 == 1   // -v  stride_h
        && params.kernel_size0 == 1     // -x  S wei_w
        && params.kernel_size1 == 1     // -y  R wei_h
        && params.kernel_dilation0 == 1
        && params.kernel_dilation1 == 1
        && params.bias == 0
        && params.in_layout == "NCHW";
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    // Check limits:
    const auto h_w     = static_cast<long>(params.out_height) * params.out_width;
    const auto r_s     = static_cast<long>(params.kernel_size1) * params.kernel_size0;
    const auto c_h_w   = static_cast<long>(params.n_outputs) * h_w;   // C*H*W
    const auto k_h_w   = static_cast<long>(params.n_inputs) * h_w;    // K*H*W
    const auto n_c_h_w = static_cast<long>(params.batch_sz) * c_h_w;  // N*C*H*W
    const auto n_k_h_w = static_cast<long>(params.batch_sz) * k_h_w;  // N*K*H*W
    const auto c_k_r_s = static_cast<long>(params.n_outputs) * params.n_inputs * r_s; // C*K*R*S
    ok = params.batch_sz < std::pow(2, 16)      // -n   N batch_size
         && params.n_outputs < std::pow(2, 16)  // -c   C input_channels
         && params.n_inputs < std::pow(2, 16)   // -k   K output_channels
         && c_h_w < std::pow(2, 24)
         && k_h_w < std::pow(2, 24)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29); // clang-format on
    return ok;
}

bool ConvAsmBwdWrW1x1::IsFast(const ConvolutionContext&) const { return true; }

static int divide_round_plus_inf(const int x, const int y)
{
    assert(x >= 0 && y > 0);
    if(x % y != 0)
        return x / y + 1;
    return x / y;
}

ConvSolution ConvAsmBwdWrW1x1::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfigConvAsmBwdWrW1x1& config,
                                           const bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;
    std::ostringstream options;
    GenerateClangDefsym(options, "batch_size", params.batch_sz); // N
    GenerateClangDefsym(options, "img_h", params.out_height);    // H
    GenerateClangDefsym(options, "img_w", params.out_width);     // W
    // Note that params.n_outputs and params.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "input_channels", params.n_outputs); // C
    GenerateClangDefsym(options, "output_channels", params.n_inputs); // K
    GenerateClangDefsym(options, "wei_h", params.kernel_size1);       // R
    GenerateClangDefsym(options, "wei_w", params.kernel_size0);       // S
    GenerateClangDefsym(options, "pad_h", params.pad1);
    GenerateClangDefsym(options, "pad_w", params.pad0);
    GenerateClangDefsym(options, "stride_h", params.kernel_stride1);
    GenerateClangDefsym(options, "stride_w", params.kernel_stride0);
    GenerateClangDefsym(options, "weights_layout", 0);
    GenerateClangDefsym(options, "reverse_weights", 0);
    GenerateClangDefsym(
        options, "ROCM_METADATA_VERSION", (params.rmv == rocm_meta_version::V3) ? 3 : 4);
    // Perf tune:
    GenerateClangDefsym(options, "do_not_use_default_perf_params", 1);

    const PerformanceConfigConvAsmBwdWrW1x1* pcfg = &config;
    PerformanceConfigConvAsmBwdWrW1x1 fromEnv;
    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1WRW_PERF_VALS{});
        if(p_asciz)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(params))
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_GCN_ASM_DIRECT_1X1WRW_PERF_VALS: "
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
    GenerateClangDefsym(options, "n_per_gpr", pcfg->GetNPerGpr());
    GenerateClangDefsym(options, "pipe_depth", pcfg->GetPipeDepth());
    GenerateClangDefsym(options, "c_per_gpr", pcfg->GetCPerGpr());
    GenerateClangDefsym(options, "c_mult", pcfg->GetCMult());
    GenerateClangDefsym(options, "k_per_gpr", pcfg->GetKPerGpr());
    GenerateClangDefsym(options, "k_mult", pcfg->GetKMult());
    GenerateClangDefsym(options, "read_size", pcfg->GetReadSize());
    GenerateClangDefsym(options, "chunk_size", pcfg->GetChunkSize());
    GenerateClangDefsym(options, "hw_per_gpr", pcfg->GetHwPerGpr());

    KernelInfo kernel;

    kernel.comp_options = options.str();

    kernel.l_wk.clear(); // workgroupsize
    kernel.l_wk.push_back(64);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.clear(); // gridsize
    kernel.g_wk.push_back(64);
    kernel.g_wk.push_back(
        divide_round_plus_inf(params.n_outputs, pcfg->GetCPerGpr() * pcfg->GetCMult()));
    kernel.g_wk.push_back(
        divide_round_plus_inf(params.n_inputs, pcfg->GetKPerGpr() * pcfg->GetKMult()));

    kernel.kernel_file = "conv1x1wrw.s";
    kernel.kernel_name = "gcnAsmConv1x1WrW";

    result.construction_params.push_back(kernel);
    result.workspce_sz = 0;
    return result;
}

/// \todo Factor out this (to generic search implementation)
class Heartbeat1x1WrW
{
    size_t n_within_beat;
    size_t n_best;
    float best_time; // within beat
    float elapsed_cumulative;
    miopen::Timer timer;
    PerformanceConfigConvAsmBwdWrW1x1 best_config;

    void Continue()
    {
        best_time     = std::numeric_limits<float>::max();
        n_within_beat = 0;
        timer.start();
    }

    public:
    Heartbeat1x1WrW() : n_within_beat(), n_best(), best_time(), elapsed_cumulative() {}

    void Start()
    {
        elapsed_cumulative = 0.0f;
        best_config        = PerformanceConfigConvAsmBwdWrW1x1();
        Continue();
    }

    void Monitor(const bool is_recent_failed,
                 const float recent_time,
                 const size_t n_recent,
                 const float total_best,
                 size_t n_failed,
                 size_t n_total,
                 const PerformanceConfigConvAsmBwdWrW1x1& recent_config)
    {
        ++n_within_beat;
        if(!is_recent_failed && (recent_time < best_time))
        {
            best_time   = recent_time;
            n_best      = n_recent;
            best_config = recent_config;
        }
        const float elapsed = timer.elapsed_ms();
        if(elapsed > 3000)
        {
            elapsed_cumulative += elapsed;
            const float eta_sec =
                n_recent ? ((n_total - n_recent) * (elapsed_cumulative / n_recent) / 1000)
                         : 0.0f; // paraniod
            MIOPEN_LOG_W(n_recent << '/' << n_failed << '/' << n_total << ' ' << total_best
                                  << ", best within recent "
                                  << n_within_beat
                                  << ": "
                                  << best_time
                                  << " #"
                                  << n_best
                                  << ' '
                                  << best_config
                                  << ", ETA:"
                                  << eta_sec
                                  << " sec.");
            Continue();
        }
    }
};

static int RunSolution(miopen::Handle& profile_h,
                       Data_t bot_ocl_buf,
                       Data_t top_ocl_buf,
                       Data_t wei_ocl_buf,
                       const ConvolutionContext& params,
                       const ConvSolution& solution,
                       float& elapsed_time)
{
    (void)params; // -warning
    const KernelInfo k_info = solution.construction_params[0];
#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        // ConvolutionContext::general_compile_options is for OpenCL kernels
        // and thus not applicable for assembly.
        auto kernel = profile_h.GetKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);
        int unused       = 0;
        int* return_addr = nullptr;

        kernel(unused, // N
               unused, // C
               unused, // H
               unused, // W
               unused, // K
               unused, // n_groups
               unused,
               unused,
               top_ocl_buf,
               wei_ocl_buf,
               bot_ocl_buf,
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

static void
InitRandomly(std::vector<float>& vec, const double offset = 0.0, const double factor = 1.0)
{
    float* p = vec.data();
    for(int i = 0; i < vec.size(); ++i)
        *p++ = static_cast<float>((rand() * (1.0 / RAND_MAX) + offset) * factor);
}

PerformanceConfigConvAsmBwdWrW1x1 ConvAsmBwdWrW1x1::Search(const ConvolutionContext& params) const
{
    PerformanceConfigConvAsmBwdWrW1x1 best_config;
    miopen::Handle profile_h;
    profile_h.EnableProfiling(true);

    // Allocate buffers, init input buffers.
    std::vector<float> bot(params.bot_sz / sizeof(float));
    std::vector<float> top(params.top_sz / sizeof(float));
    std::vector<float> wei(params.weights_sz / sizeof(float));
    InitRandomly(bot);
    InitRandomly(top);
    auto bot_ocl_buf = profile_h.Write(bot);
    auto top_ocl_buf = profile_h.Write(top);
    auto wei_ocl_buf = profile_h.Write(wei);

    int n_runs_total = 0;
    const VirtualContainer1x1WrW all_configs(params);
    {
        for(const auto& dummy : all_configs)
        {
            ++n_runs_total;
            (void)dummy;
        }
    }
    MIOPEN_LOG_W("Searching the best solution among " << n_runs_total << "...");
    bool is_passed   = false; // left false only if all iterations failed.
    float best_time  = std::numeric_limits<float>::max();
    size_t n_failed  = 0;
    size_t n_current = 0;
    size_t n_best    = 0;
    Heartbeat1x1WrW heartbeat;
    heartbeat.Start();
    for(const auto& current_config : all_configs)
    {
        float elapsed_time;
        MIOPEN_LOG_I2('#' << n_current << '/' << n_failed << '/' << n_runs_total << ' '
                          << current_config);
        // Smooth the jitter of the measured time.:
        // If 1st probe isn't worse than the best one by 5%,
        // then re-run 4 more times and compute average time,
        // and decide using average vs. the best.
        auto ret = RunSolution(profile_h,
                               bot_ocl_buf.get(),
                               top_ocl_buf.get(),
                               wei_ocl_buf.get(),
                               params,
                               GetSolution(params, current_config, true),
                               elapsed_time);
        if(ret == 0)
        {
            if(elapsed_time / best_time < 1.05f)
            {
                MIOPEN_LOG_I2("Finding average for: " << elapsed_time << " / " << best_time << " = "
                                                      << (elapsed_time / best_time));
                float temp;
                for(int i = 0; i < 4; ++i)
                {
                    ret = RunSolution(profile_h,
                                      bot_ocl_buf.get(),
                                      top_ocl_buf.get(),
                                      wei_ocl_buf.get(),
                                      params,
                                      GetSolution(params, current_config, true),
                                      temp);
                    if(ret != 0)
                    {
                        break;
                    }
                    elapsed_time += temp;
                }
                if(ret == 0)
                {
                    is_passed = true;
                    elapsed_time /= 5;
                    if(elapsed_time < best_time)
                    {
                        MIOPEN_LOG_I('#' << n_current << '/' << n_failed << '/' << n_runs_total
                                         << ' '
                                         << elapsed_time
                                         << " < "
                                         << best_time
                                         << ' '
                                         << current_config);
                        best_config = current_config;
                        best_time   = elapsed_time;
                        n_best      = n_current;
                    }
                    else
                    {
                        MIOPEN_LOG_I2(
                            "Average is not better: " << elapsed_time << " >= " << best_time);
                    }
                }
            }
        }

        if(ret != 0)
        {
            MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                             << " Failed rc="
                             << ret);
            ++n_failed;
        }
        heartbeat.Monitor(
            ret != 0, elapsed_time, n_current, best_time, n_failed, n_runs_total, current_config);
        ++n_current;
    }

    profile_h.EnableProfiling(false);
    MIOPEN_LOG_W("Done: " << n_runs_total << '/' << n_failed << '/' << n_runs_total << ", best #"
                          << n_best
                          << ' '
                          << best_time
                          << ' '
                          << best_config);
    if(!is_passed)
        MIOPEN_THROW("Search failed for PerformanceConfigConvAsmBwdWrW1x1");
    return best_config;
}

} // namespace solver
} // namespace miopen
