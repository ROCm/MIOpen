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

#define MIOPEN_GCN_ASM_DIRECT_3X3WRW_SEARCH_LWC_FIXED 0

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_SEARCH_QUICK)

namespace miopen {

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

static bool IsReverseInOutAllowed(const ConvolutionContext& config)
{
    return config.kernel_stride0 == 1 && config.kernel_stride1 == 1;
}

class VirtualIterator;

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
class VirtualContainer
{
    // Valid iterator shall denote an element of a container.
    const ConvolutionContext& config;
    friend class VirtualIterator;

    public:
    using const_iterator = VirtualIterator;
    VirtualContainer(const ConvolutionContext& config_) : config(config_) {}
    VirtualIterator begin() const;
    VirtualIterator end() const;
};

// Iterator shall advance to the next valid config, i.e. the one which
// satisfies PerformanceConfig.IsValid(ProblemConfig)
class VirtualIterator
    : public std::iterator<std::input_iterator_tag, PerformanceConfigAsmDirect3x3WrW>
{
    value_type v; // PerformanceConfigAsmDirect3x3WrW
    const VirtualContainer* container;

    static const value_type& GetMinValue();
    static const value_type& GetOutOfRangeValue();

    /// Implements begin()
    VirtualIterator(const VirtualContainer* container_) : v(GetMinValue()), container(container_)
    {
        if(!IsValid())
            Next();
    }
    friend class VirtualContainer; // Passes itself to private ctor in order to construct begin().
    void Next();
    bool IsValid();

    public:
    /// Implementes end() and also serves as a default ctor.
    VirtualIterator() : v(GetOutOfRangeValue()), container(nullptr) {}

    bool operator!=(VirtualIterator const& other) const;
    const value_type& operator*() const { return v; }
    const value_type* operator->() const { return &v; }
    VirtualIterator& operator++()
    {
        Next();
        return *this;
    }
};

inline VirtualIterator VirtualContainer::begin() const { return {this}; }

inline VirtualIterator VirtualContainer::end() const { return {}; }

const VirtualIterator::value_type& VirtualIterator::GetMinValue()
{
    static const value_type val(0, 0, 8, 1, 1, 1);
    return val;
}

const VirtualIterator::value_type& VirtualIterator::GetOutOfRangeValue()
{
    static const value_type val(-1, -1, -1, -1, -1, -1);
    return val;
}

inline bool VirtualIterator::IsValid()
{
    if(!container)
        return false;
    return v.IsValid(container->config);
}

inline bool VirtualIterator::operator!=(VirtualIterator const& other) const
{
    return !(v.IsEqual(other.v) && container == other.container);
}

void VirtualIterator::Next()
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
#if MIOPEN_GCN_ASM_DIRECT_3X3WRW_SEARCH_LWC_FIXED == 0
            if(!miopen::IsEnabled(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_SEARCH_QUICK{}))
            {
                // (0 <= limit_wave_cnt && limit_wave_cnt <= 9)
                if(++v.limit_wave_cnt <= 9)
                    break;
            }
#endif
            v.limit_wave_cnt = 0;
            // (0 <= reverse_inout && reverse_inout <= 1)
            if(++v.reverse_inout <= 1)
                break;
            v.reverse_inout = 0;
            // (8 == chunk_size || 16 == chunk_size)
            if((v.chunk_size += 8) <= 16)
                break;
            v.chunk_size = 8;
            // (1 == k_per_wave || 2 == k_per_wave || 4 == k_per_wave || 8 == k_per_wave)
            if(1 == v.k_per_wave)
            {
                v.k_per_wave = 2;
                break;
            }
            if(2 == v.k_per_wave)
            {
                v.k_per_wave = 4;
                break;
            }
            if(4 == v.k_per_wave)
            {
                v.k_per_wave = 8;
                break;
            }
            v.k_per_wave = 1;
            // (1 <= pipe_lines_depth && pipe_lines_depth <= 16)
            if(++v.pipe_lines_depth <= 16)
                break;
            v.pipe_lines_depth = 1;
            // (1 <= n_per_group && n_per_group <= 8);
            if(++v.n_per_group <= 8)
                break;
            v.n_per_group = 1;
            // All the fields (components) of performance confic have wrapped around.
            // The next one is not the min (in the allowed range) but a one beyond the end.
            // Iterator is useless from now.
            v         = GetOutOfRangeValue();
            container = nullptr;
            return;
        } while(false);
    } while(!IsValid());
}

PerformanceConfigAsmDirect3x3WrW::PerformanceConfigAsmDirect3x3WrW(
    int lwc, int rio, int csz, int kpw, int pld, int npg)
    : limit_wave_cnt(lwc),
      reverse_inout(rio),
      chunk_size(csz),
      k_per_wave(kpw),
      pipe_lines_depth(pld),
      n_per_group(npg)
{
}

inline bool
PerformanceConfigAsmDirect3x3WrW::IsEqual(const PerformanceConfigAsmDirect3x3WrW& other) const
{
    return limit_wave_cnt == other.limit_wave_cnt && reverse_inout == other.reverse_inout &&
           chunk_size == other.chunk_size && k_per_wave == other.k_per_wave &&
           pipe_lines_depth == other.pipe_lines_depth && n_per_group == other.n_per_group;
}

bool PerformanceConfigAsmDirect3x3WrW::IsValidRange() const
{
    return (0 <= limit_wave_cnt && limit_wave_cnt <= 9) &&
           (0 <= reverse_inout && reverse_inout <= 1) && (8 == chunk_size || 16 == chunk_size) &&
           (1 == k_per_wave || 2 == k_per_wave || 4 == k_per_wave || 8 == k_per_wave) &&
           (1 <= pipe_lines_depth && pipe_lines_depth <= 16) &&
           (1 <= n_per_group && n_per_group <= 8);
}

bool PerformanceConfigAsmDirect3x3WrW::IsValid(const ConvolutionContext& config) const
{
    if(!IsValidRange())
        return false;
    assert(chunk_size != 0);
    if((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        return false;
    if((reverse_inout ? config.n_inputs : config.n_outputs) % GetCPerWave() != 0)
        return false;
    if(!(chunk_size * k_per_wave <= 64))
        return false;
    if((reverse_inout ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        return false;
    if(!(n_per_group <= config.batch_sz))
        return false;
    if(!(1 <= pipe_lines_depth && pipe_lines_depth <= std::min(config.out_height, 16)))
        return false;
    if(reverse_inout && !IsReverseInOutAllowed(config))
        return false;

    {
        const int accums_cnt =
            (config.kernel_size0 * config.kernel_size1 * GetCPerWave() * k_per_wave * chunk_size) /
            64;
        assert(chunk_size);
        int gprs_per_line_in = (config.out_width + chunk_size - 1) / chunk_size;
        if(chunk_size != 16)
        {
            assert(chunk_size - config.pad0);
            gprs_per_line_in =
                (config.out_width + chunk_size - config.pad0 - 1) / (chunk_size - config.pad0);
        }
        assert(config.kernel_stride0);
        gprs_per_line_in += gprs_per_line_in % config.kernel_stride0;
        const int gprs_per_line_out =
            (gprs_per_line_in > 1) ? gprs_per_line_in / config.kernel_stride0 : 1;

        const int lines_in = pipe_lines_depth + config.kernel_size1 - 1;
        assert(config.kernel_stride1);
        const int lines_out =
            (pipe_lines_depth + config.kernel_stride1 - 1) / config.kernel_stride1;
        const int vgprs =
            accums_cnt + lines_in * gprs_per_line_in + lines_out * gprs_per_line_out + 6;
        if(!(vgprs <= 256))
            return false;
        if(n_per_group > 4)
            if(!(vgprs <= 128))
                return false;
        if(limit_wave_cnt != 0 && limit_wave_cnt * 4 < n_per_group)
            return false;
        const int lds_size = (n_per_group - 1) * 64 /*wavesize*/ * sizeof(float) * accums_cnt;
        if(!(lds_size <= 65536))
            return false;

        const int unroll_factor = pipe_lines_depth * (pipe_lines_depth + 2);
        const int steps         = std::max(0, config.out_height - 1 - pipe_lines_depth);
        assert(unroll_factor);
        const int loops   = pipe_lines_depth + unroll_factor + steps % unroll_factor + 1;
        const int m_instr = 3 + (gprs_per_line_in + 3) / 4;
        const int v_instr =
            (k_per_wave * config.kernel_size1 * gprs_per_line_out * config.kernel_size0 * 4) / 3;
        const int total = loops * (m_instr + v_instr); // instructions
        if(total >= 32000)                             // Estimation, a bit smaller than 32K.
            return false;
    }
    return true;
}

void PerformanceConfigAsmDirect3x3WrW::EuristicInit(const ConvolutionContext& config)
{
    limit_wave_cnt = 0;

    chunk_size = (config.out_width < 48) ? 8 : 16;
    if((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        chunk_size = 16; // Fixup for correctness

    reverse_inout = 0;
    if(IsReverseInOutAllowed(config) && ((config.n_outputs % 4 != 0) || (config.out_width < 8)))
        reverse_inout = 1;

    const auto c_k = config.n_outputs * config.n_inputs; // C*K
    if(c_k < 256)
        k_per_wave = 1;
    else if(c_k < 16384)
        k_per_wave = 2;
    else // C*K >= 16k
        k_per_wave = ((chunk_size == 8) ? 2 : 4);
    while((reverse_inout ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        k_per_wave /= 2; // Fixup for correctness

    if(c_k <= 512)
        n_per_group = 8;
    else if(c_k <= 4096)
        n_per_group = 4;
    else if(c_k <= 8192)
        n_per_group = 2;
    else
        n_per_group = 1;
    if(n_per_group > config.batch_sz)
        n_per_group = config.batch_sz; // n_per_group should never be > batch size.
    if(config.out_width >= 256 &&
       n_per_group > 4) // when width >= 256, n_per_group should not be > 4.
        n_per_group = 4;

    pipe_lines_depth = (config.out_height <= 1) ? 1 : 2;
    if((config.out_height < 8) && (config.out_width < 64))
    {
        pipe_lines_depth = config.out_height; // Special case.
    }

    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        limit_wave_cnt   = 0;
        reverse_inout    = 0;
        chunk_size       = 16; // CPerWave() = 4;
        k_per_wave       = 1;
        pipe_lines_depth = 2;
        n_per_group      = 1;
        if(config.n_outputs % 4 != 0)
        {
            /// (1) If reverse is Off, then both (C % c_per_wave) and (K % k_per_wave) must be 0.
            /// Toggling reverse swaps C and K in the condition above.
            /// (2) From the other hand, IsApplicable() ensures that either C or K is evenly
            /// divisable by 4.
            /// (3) We just set k_per_wave=1, c_per_wave=4. Therefore, (1) always can be satisfied
            /// here. If (C % c_per_wave) is not zero, just push reverse button so K and C will
            /// swap.
            ///
            /// \note C (input channels) resides in n_outputs, K (output channels) - in n_inputs,
            /// because that's how reverse convolutions are handled in MIOpen.
            reverse_inout = 1;
        }
        assert(IsValid(config));
    }
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceConfigAsmDirect3x3WrW::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigAsmDirect3x3WrW
ConvAsmBwdWrW3x3::GetPerformanceConfig(const ConvolutionContext& params) const
{
    std::string s;
    PerformanceConfigAsmDirect3x3WrW pp;
    const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS{});
    if(p_asciz)
    {
        s = std::string(p_asciz);
    }
    if(!s.empty()) // Otherwise, nothing is set in env -> nothing to parse.
    {
        static const std::string h("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: ");
        if(!pp.Deserialize(s))
        {
            MIOPEN_THROW(h + "Bad format:" + s);
        }
        if(!pp.IsValid(params))
        {
            MIOPEN_THROW(h + "Out of range of invalid for the problem config:" + s);
        }
        MIOPEN_LOG_I("From env: " << pp.ToString());
    }
    else
    {
        pp.EuristicInit(params);
    }
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsmBwdWrW3x3::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                                const PerformanceConfigAsmDirect3x3WrW& c) const
{
    return c.IsValidRange() && c.IsValid(problem);
}

bool ConvAsmBwdWrW3x3::IsApplicable(const ConvolutionContext& params) const
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
    bool ok = params.pad0 == 1                   // -q  pad_w
              && params.pad1 == 1                // -p  pad_h
              && (params.kernel_stride0 <= 2)    // -u  stride_w
              && (params.kernel_stride1 <= 2)    // -v  stride_h
              && params.kernel_size0 == 3        // -x  S wei_w
              && params.kernel_size1 == 3        // -y  R wei_h
              && params.kernel_dilation0 == 1 && params.kernel_dilation1 == 1 && params.bias == 0 &&
              params.in_layout == "NCHW";
    // && _weights_layout == "KCHW"
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    // Check limits:
    const auto h_w     = static_cast<long>(params.out_height) * params.out_width;
    const auto r_s     = static_cast<long>(params.kernel_size1) * params.kernel_size0;
    const auto c_h_w   = static_cast<long>(params.n_outputs) * h_w;   // C*H*W
    const auto k_h_w   = static_cast<long>(params.n_inputs) * h_w;    // K*H*W
    const auto c_r_s   = static_cast<long>(params.n_outputs) * r_s;   // C*R*S
    const auto k_r_s   = static_cast<long>(params.n_inputs) * r_s;    // K*R*S
    const auto n_c_h_w = static_cast<long>(params.batch_sz) * c_h_w;  // N*C*H*W
    const auto n_k_h_w = static_cast<long>(params.batch_sz) * k_h_w;  // N*K*H*W
    const auto c_k_r_s = static_cast<long>(params.n_outputs) * k_r_s; // C*K*R*S
    // clang-format off
    ok = params.out_width > 0
         && params.out_width <= 512
         && (IsReverseInOutAllowed(params)
                ? ((params.n_outputs % 4 == 0) || (params.n_inputs % 4 == 0))
                : (params.n_outputs % 4 == 0))
         && params.out_height < std::pow(2, 16) // -H   H img_h
         && params.batch_sz < std::pow(2, 16)   // -n   N batch_size
         && params.n_outputs < std::pow(2, 16)  // -c   C input_channels
         && params.n_inputs < std::pow(2, 16)   // -k   K output_channels
         && c_h_w < std::pow(2, 22)
         && k_h_w < std::pow(2, 22)
         && c_r_s < std::pow(2, 22)
         && k_r_s < std::pow(2, 22)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29);                                    // clang-format on
    return ok;
}

bool ConvAsmBwdWrW3x3::IsFast(const ConvolutionContext&) const { return true; }

ConvSolution ConvAsmBwdWrW3x3::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfigAsmDirect3x3WrW& config) const
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
    GenerateClangDefsym(options, "limit_wave_cnt", config.GetLimitWaveCnt());
    GenerateClangDefsym(options, "chunk_size", config.GetChunkSize());
    GenerateClangDefsym(options, "c_per_wave", config.GetCPerWave());
    GenerateClangDefsym(options, "k_per_wave", config.GetKPerWave());
    GenerateClangDefsym(options, "n_per_group", config.GetNPerGroup());
    GenerateClangDefsym(options, "pipe_lines_depth", config.GetPipeLinesDepth());
    GenerateClangDefsym(options, "reverse_inout", config.GetReverseInout());
    // Debugging:
    GenerateClangDefsym(options, "enable_debug_output", 0);

    KernelInfo kernel;

    kernel.comp_options = options.str();

    kernel.l_wk.clear(); // workgroupsize
    kernel.l_wk.push_back(64 * config.GetNPerGroup());
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.clear(); // gridsize
    kernel.g_wk.push_back(64 * config.GetNPerGroup());

    if(config.GetReverseInout() == 0)
    {
        kernel.g_wk.push_back(params.n_outputs / config.GetCPerWave());
        kernel.g_wk.push_back(params.n_inputs / config.GetKPerWave());
    }
    else
    {
        kernel.g_wk.push_back(params.n_outputs / config.GetKPerWave());
        kernel.g_wk.push_back(params.n_inputs / config.GetCPerWave());
    }

    kernel.kernel_file = "conv3x3wrw.s";
    kernel.kernel_name = "gcnAsmConv3x3WrW";

    result.construction_params.push_back(kernel);
    result.workspce_sz = 0;
    return result;
}

class Heartbeat
{
    size_t n_within_beat;
    size_t n_best;
    float best_time; // within beat
    float elapsed_cumulative;
    miopen::Timer timer;
    PerformanceConfigAsmDirect3x3WrW best_config;

    void Continue()
    {
        best_time     = std::numeric_limits<float>::max();
        n_within_beat = 0;
        timer.start();
    }

    public:
    Heartbeat() : n_within_beat(), n_best(), best_time(), elapsed_cumulative() {}

    void Start()
    {
        elapsed_cumulative = 0.0f;
        best_config        = PerformanceConfigAsmDirect3x3WrW();
        Continue();
    }

    void Monitor(const float recent_time,
                 const size_t n_recent,
                 const float total_best,
                 size_t n_failed,
                 size_t n_total,
                 const PerformanceConfigAsmDirect3x3WrW& recent_config)
    {
        ++n_within_beat;
        if(recent_time < best_time)
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
        auto n_groups =
            static_cast<int>(params.GetStream().GetMaxComputeUnits()); // kernel needs int32

        kernel(params.batch_sz,   // N
               params.n_outputs,  // C
               params.out_height, // H
               params.out_width,  // W
               params.n_inputs,   // K
               n_groups,          // n_groups
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

PerformanceConfigAsmDirect3x3WrW ConvAsmBwdWrW3x3::Search(const ConvolutionContext& params) const
{
    PerformanceConfigAsmDirect3x3WrW best_config;
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
    const VirtualContainer all_configs(params);
    {
        for(const auto& dummy : all_configs)
        {
            ++n_runs_total;
            (void)dummy;
        }
    }
    MIOPEN_LOG_W("Searching the best solution among " << n_runs_total << "...");
    bool is_passed   = false;
    float best_time  = std::numeric_limits<float>::max();
    size_t n_failed  = 0;
    size_t n_current = 0;
    size_t n_best    = 0;
    Heartbeat heartbeat;
    heartbeat.Start();
    for(const auto& current_config : all_configs)
    {
        float elapsed_time;
        MIOPEN_LOG_I2('#' << n_current << '/' << n_failed << '/' << n_runs_total << ' '
                          << current_config);
        const auto ret = RunSolution(profile_h,
                                     bot_ocl_buf.get(),
                                     top_ocl_buf.get(),
                                     wei_ocl_buf.get(),
                                     params,
                                     GetSolution(params, current_config),
                                     elapsed_time);
        if(ret == 0)
        {
            is_passed = true;
            if(elapsed_time < best_time)
            {
                MIOPEN_LOG_I('#' << n_current << '/' << n_failed << '/' << n_runs_total << ' '
                                 << elapsed_time
                                 << " < "
                                 << best_time
                                 << ' '
                                 << current_config);
                best_config = current_config;
                best_time   = elapsed_time;
                n_best      = n_current;
            }
        }
        else
        {
            MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                             << " Failed rc="
                             << ret);
            ++n_failed;
        }
        heartbeat.Monitor(
            elapsed_time, n_current, best_time, n_failed, n_runs_total, current_config);
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
        MIOPEN_THROW("Search failed for PerformanceConfigAsmDirect3x3WrW");
    return best_config;
}

} // namespace solver
} // namespace miopen
