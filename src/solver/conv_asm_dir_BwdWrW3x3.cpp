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

#include <unordered_map>

#include "miopen/solver.hpp"
#include "miopen/gcn_asm_utils.hpp"
#include "miopen/env.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS)

namespace miopen {
namespace solver {

struct PerfParamsAsmDirect3x3WrW
{
    int limit_wave_cnt;
    int chunk_size; // 16 or 8. Lower values increase register pressure
    int c_per_wave; // should be (64 / chunk_size)
    int k_per_wave; // 1, 2, 4, 8 and chunk_size * k_per_wave <= 64. Higher values increase register
                    // preasure
    int n_per_group;      // 1..8 and n_per_group <= batch_size
    int pipe_lines_depth; // 1..8 and pipe_lines_depth <= img_h. Higher values increase register
                          // pressure
    int reverse_inout;    // 0 or 1
    PerfParamsAsmDirect3x3WrW()
        : limit_wave_cnt(0),
          chunk_size(16),
          c_per_wave(4),
          k_per_wave(4),
          n_per_group(1),
          pipe_lines_depth(2),
          reverse_inout(0)
    {
    }
};
inline int PopIntFromString(std::string& s, size_t digits)
{
    const auto val = std::stoi(s.substr(0, digits));
    s              = s.substr(digits);
    return val;
}

static void ParsePerfParamsAsmDirect3x3WrW(const std::string& s, PerfParamsAsmDirect3x3WrW& pp)
{
    if(s.size() != 9)
    {
        MIOPEN_THROW("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: bad format.");
    }

    std::string temp    = s;
    pp.limit_wave_cnt   = PopIntFromString(temp, 2); // two digits
    pp.reverse_inout    = PopIntFromString(temp, 1);
    pp.chunk_size       = PopIntFromString(temp, 2); // two digits
    pp.k_per_wave       = PopIntFromString(temp, 1);
    pp.pipe_lines_depth = PopIntFromString(temp, 2);
    pp.n_per_group      = PopIntFromString(temp, 1);
    // Check if values are wrong.
    if(!((0 <= pp.limit_wave_cnt && pp.limit_wave_cnt <= 10) &&
         (0 <= pp.reverse_inout && pp.reverse_inout <= 1) &&
         (8 == pp.chunk_size || 16 == pp.chunk_size) &&
         (1 == pp.k_per_wave || 2 == pp.k_per_wave || 4 == pp.k_per_wave || 8 == pp.k_per_wave) &&
         (1 <= pp.pipe_lines_depth && pp.pipe_lines_depth <= 16) &&
         (1 <= pp.n_per_group && pp.n_per_group <= 8)))
    {
        MIOPEN_THROW("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: out of range.");
    }
}

static std::string FormPerfParamsAsmDirect3x3WrW(int limit_wave_cnt,
                                                 int reverse_inout,
                                                 int chunk_size,
                                                 int k_per_wave,
                                                 int pipe_lines_depth,
                                                 int n_per_group)
{
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << limit_wave_cnt;
    oss << std::setfill('0') << std::setw(1) << reverse_inout;
    oss << std::setfill('0') << std::setw(2) << chunk_size;
    oss << std::setfill('0') << std::setw(1) << k_per_wave;
    oss << std::setfill('0') << std::setw(2) << pipe_lines_depth;
    oss << std::setfill('0') << std::setw(1) << n_per_group;
    return oss.str();
}

static PerfParamsAsmDirect3x3WrW
mloComputePerfParamsAsmDirect3x3WrW(const ConvolutionContext& params)
{
    /// LUT entry/env.var format: 8 decimal ASCII digits, left to right:
    /// limit_wave_cnt   [00..10]
    /// reverse_inout    [0..1]
    /// chunk_size       {08,16}
    /// k_per_wave       {1,2,4,8}
    /// pipe_lines_depth [1..16]
    /// n_per_group      [1..8]
    /// \note chunk_size is not in included in the format, but computed.

    /// Optimal values in LUT were found on Gfx8 with 56 CUs (R9 Fury).
    /// \todo Test on devices with 64 CUs (e.g. R9 Nano, Vega10) and expand
    /// implementation if optimal values are different.
    static const std::unordered_map<std::string, std::string> perf_vals_map({
        //              W    H    c    n    k    dir CUs               lwc[2] rio csz[2] kpw pld npg
        {MakeKeyWHCNKD(13, 13, 192, 128, 384, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 192, 128, 384, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 256, 128, 256, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 256, 128, 256, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 256, 128, 384, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 256, 128, 384, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 384, 128, 256, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 384, 128, 256, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 384, 128, 384, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 2, 1)},
        {MakeKeyWHCNKD(13, 13, 384, 128, 384, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 8, 2, 1)},
        {MakeKeyWHCNKD(14, 14, 128, 8, 256, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 2, 1)},
        {MakeKeyWHCNKD(14, 14, 512, 8, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 4, 1)},
        {MakeKeyWHCNKD(14, 14, 512, 16, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 4, 1)},
        {MakeKeyWHCNKD(14, 14, 512, 16, 512, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 4, 1)},
        {MakeKeyWHCNKD(14, 14, 512, 32, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 4, 1)},
        {MakeKeyWHCNKD(14, 14, 512, 64, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 4, 1)},
        {MakeKeyWHCNKD(16, 16, 256, 8, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 2, 1)},
        {MakeKeyWHCNKD(27, 27, 128, 8, 128, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 3, 1)},
        {MakeKeyWHCNKD(28, 28, 256, 8, 512, 0), FormPerfParamsAsmDirect3x3WrW(4, 1, 8, 2, 2, 1)},
        {MakeKeyWHCNKD(28, 28, 256, 16, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 2, 3, 1)},
        {MakeKeyWHCNKD(28, 28, 256, 32, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 4, 1)},
        {MakeKeyWHCNKD(28, 28, 256, 64, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 4, 1)},
        {MakeKeyWHCNKD(28, 28, 512, 32, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 4, 1)},
        {MakeKeyWHCNKD(28, 28, 512, 64, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 4, 1)},
        {MakeKeyWHCNKD(54, 54, 64, 8, 64, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 16, 2, 2, 4)},
        {MakeKeyWHCNKD(54, 54, 64, 8, 64, 0, 64), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 2, 3, 2)},
        {MakeKeyWHCNKD(56, 56, 64, 16, 192, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 2, 4)},
        {MakeKeyWHCNKD(56, 56, 64, 32, 192, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 4, 4)},
        {MakeKeyWHCNKD(56, 56, 256, 32, 256, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 2, 4, 1)},
        {MakeKeyWHCNKD(56, 56, 256, 64, 256, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 2, 4, 1)},
        {MakeKeyWHCNKD(60, 6, 64, 16, 128, 0), FormPerfParamsAsmDirect3x3WrW(4, 0, 16, 2, 6, 1)},
        {MakeKeyWHCNKD(60, 6, 64, 16, 128, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 2, 2, 1)},
        {MakeKeyWHCNKD(112, 112, 64, 8, 128, 0), FormPerfParamsAsmDirect3x3WrW(3, 0, 16, 4, 2, 2)},
        {MakeKeyWHCNKD(112, 112, 64, 8, 128, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 16, 4, 2, 1)},
        {MakeKeyWHCNKD(112, 112, 64, 16, 128, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 2, 4)},
        {MakeKeyWHCNKD(112, 112, 64, 16, 128, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 3, 1)},
        {MakeKeyWHCNKD(112, 112, 64, 32, 128, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 2, 4)},
        {MakeKeyWHCNKD(112, 112, 64, 32, 128, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 16, 4, 3, 1)},
        {MakeKeyWHCNKD(112, 112, 64, 64, 128, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 2, 4)},
        {MakeKeyWHCNKD(112, 112, 256, 8, 512, 0), FormPerfParamsAsmDirect3x3WrW(0, 1, 16, 4, 2, 1)},
        {MakeKeyWHCNKD(120, 12, 32, 16, 64, 0), FormPerfParamsAsmDirect3x3WrW(3, 1, 16, 2, 1, 4)},
        {MakeKeyWHCNKD(120, 12, 32, 16, 64, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 2, 2, 2)},
        {MakeKeyWHCNKD(224, 224, 3, 8, 64, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 16, 1, 2, 4)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(224, 224, 3, 16, 64, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 16, 1, 5, 4)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(240, 24, 16, 16, 32, 0), FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 1, 8)},
        {MakeKeyWHCNKD(240, 24, 16, 16, 32, 0, 64),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 2, 1, 8)},
        {MakeKeyWHCNKD(13, 13, 384, 64, 256, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 8, 11, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(13, 13, 256, 50, 384, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 8, 11, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(13, 13, 384, 50, 384, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 8, 11, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(13, 13, 384, 50, 256, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 8, 11, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(28, 28, 64, 32, 64, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 2, 2, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(28, 28, 64, 32, 96, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 2, 5, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 160, 32, 160, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 11, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 160, 32, 192, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 8, 5, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 192, 32, 256, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 3, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(512, 256, 64, 1, 192, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 16, 4, 1, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(256, 128, 96, 1, 128, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 1, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(256, 128, 128, 1, 192, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 1, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 256, 16, 256, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 8, 11, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(12, 12, 512, 128, 1024, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 8, 11, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(12, 12, 1024, 128, 1024, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 8, 11, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(7, 7, 192, 128, 384, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 7, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(7, 7, 160, 128, 320, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 7, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 160, 128, 320, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 5, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 144, 128, 288, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 5, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 128, 128, 256, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 4, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 112, 128, 224, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 1, 8, 4, 5, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(14, 14, 96, 128, 208, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 7, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(56, 56, 64, 128, 192, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 16, 4, 4, 1)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(28, 28, 128, 128, 192, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 2, 2)}, /// \todo Find opt values for 56CUs
        {MakeKeyWHCNKD(28, 28, 96, 128, 128, 0),
         FormPerfParamsAsmDirect3x3WrW(0, 0, 8, 4, 2, 2)}, /// \todo Find opt values for 56CUs
    });

    std::string s;
    PerfParamsAsmDirect3x3WrW pp;
    const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS{});
    if(p_asciz)
    {
        s = std::string(p_asciz);
    }
    if(!s.empty())
    { // Parse and check non-empty string from env.
        ParsePerfParamsAsmDirect3x3WrW(s, pp);
        if(((params.n_outputs % (64 / pp.chunk_size) != 0) &&
            (params.n_inputs % (64 / pp.chunk_size) != 0)) ||
           ((pp.reverse_inout ? params.n_outputs : params.n_inputs) % pp.k_per_wave != 0) ||
           !(pp.n_per_group <= params.batch_sz) ||
           !(1 <= pp.pipe_lines_depth && pp.pipe_lines_depth <= std::min(params.in_height, 16)))
        {
            MIOPEN_THROW(
                "MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: incorrect for the problem config.");
        }
    }
    else
    {
        // Try to get values from LUT. If not found, use heuristic algorithm.
        // At first, try to find numCUs-specific values.
        const auto numCUs = static_cast<int>(params.GetStream().GetMaxComputeUnits());
        auto key          = MakeKeyWHCNKD(params.in_width,
                                 params.in_height,
                                 params.n_outputs,
                                 params.batch_sz,
                                 params.n_inputs,
                                 0,
                                 numCUs);
        auto found = perf_vals_map.find(key);
        if(found == perf_vals_map.end())
        { // numCUs-specific values not found, try to find "universal" ones.
            key = MakeKeyWHCNKD(params.in_width,
                                params.in_height,
                                params.n_outputs,
                                params.batch_sz,
                                params.n_inputs,
                                0);
            found = perf_vals_map.find(key);
        }
        if(found != perf_vals_map.end())
        {
            s = found->second;
            ParsePerfParamsAsmDirect3x3WrW(s, pp);
            /// \todo Copy-paste from above. Generalize.
            if(((params.n_outputs % (64 / pp.chunk_size) != 0) &&
                (params.n_inputs % (64 / pp.chunk_size) != 0)) ||
               ((pp.reverse_inout ? params.n_outputs : params.n_inputs) % pp.k_per_wave != 0) ||
               !(pp.n_per_group <= params.batch_sz) ||
               !(1 <= pp.pipe_lines_depth && pp.pipe_lines_depth <= std::min(params.in_height, 16)))
            {
                MIOPEN_THROW("mloComputePerfParamsAsmDirect3x3WrW: LUT entry: incorrect for the "
                             "problem config.");
            }
        }
        else
        {
            {
                auto& v = pp.chunk_size;
                v       = (params.in_width < 48) ? 8 : 16;
                if((params.n_outputs % (64 / v) != 0) && (params.n_inputs % (64 / v) != 0))
                {
                    v = 16; // Fixup for correctness
                }
            }
            {
                auto& v = pp.reverse_inout;
                if((params.n_outputs % 4 != 0) || (params.in_width < 8))
                {
                    v = 1;
                }
                else
                {
                    v = 0;
                }
            }
            const auto c_k = params.n_outputs * params.n_inputs; // C*K
            {
                auto& v = pp.k_per_wave;
                if(c_k < 256)
                {
                    v = 1;
                }
                else if(c_k < 16384)
                {
                    v = 2;
                }
                else
                { // C*K >= 16k
                    v = (pp.chunk_size == 8) ? 2 : 4;
                }
                while((pp.reverse_inout ? params.n_outputs : params.n_inputs) % v != 0)
                {
                    v /= 2; // Fixup for correctness
                }
            }
            {
                auto& v = pp.n_per_group;
                if(c_k <= 512)
                {
                    v = 8;
                }
                else if(c_k <= 4096)
                {
                    v = 4;
                }
                else if(c_k <= 8192)
                {
                    v = 2;
                }
                else
                {
                    v = 1;
                }
                if(v > params.batch_sz)
                {
                    v = params.batch_sz; // Fixup for correctness
                }
            }
            {
                auto& v = pp.pipe_lines_depth;
                v       = (params.in_height <= 1) ? 1 : 2;
                if((params.in_height < 8) && (params.in_width < 64))
                {
                    v = params.in_height; // Special case.
                }
            }
        }
    }
    pp.c_per_wave = 64 / pp.chunk_size;
    return pp;
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

    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos && name.find("gfx9") == std::string::npos)
    {
        return false;
    }
    assert(params.weights_layout.length() == 0); // _weights_layout is not supported yet
    bool ok = params.pad0 == 1                   // -q     pad_w
              && params.pad1 == 1                // -p     pad_h
              && params.kernel_stride0 == 1      // -u     stride_w
              && params.kernel_stride1 == 1      // -v     stride_h
              && params.kernel_size0 == 3        // -x   S wei_w
              && params.kernel_size1 == 3        // -y   R wei_h
              && params.in_layout == "NCHW";
    // && _weights_layout == "KCHW"
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    // Check limits:
    const auto h_w     = static_cast<long>(params.in_height) * params.in_width;
    const auto r_s     = static_cast<long>(params.kernel_size1) * params.kernel_size0;
    const auto c_h_w   = static_cast<long>(params.n_outputs) * h_w;   // C*H*W
    const auto k_h_w   = static_cast<long>(params.n_inputs) * h_w;    // K*H*W
    const auto c_r_s   = static_cast<long>(params.n_outputs) * r_s;   // C*R*S
    const auto k_r_s   = static_cast<long>(params.n_inputs) * r_s;    // K*R*S
    const auto n_c_h_w = static_cast<long>(params.batch_sz) * c_h_w;  // N*C*H*W
    const auto n_k_h_w = static_cast<long>(params.batch_sz) * k_h_w;  // N*K*H*W
    const auto c_k_r_s = static_cast<long>(params.n_outputs) * k_r_s; // C*K*R*S
    ok                 = params.in_width > 0 && params.in_width <= 256 &&
         params.in_height < std::pow(2, 16)    // -H   H img_h
         && params.batch_sz < std::pow(2, 16)  // -n   N batch_size
         && params.n_outputs < std::pow(2, 16) // -c   C input_channels
         && params.n_inputs < std::pow(2, 16)  // -k   K output_channels
         && ((params.n_outputs % 4 == 0) || (params.n_inputs % 4 == 0)) &&
         c_h_w < std::pow(2, 22) && k_h_w < std::pow(2, 22) && c_r_s < std::pow(2, 22) &&
         k_r_s < std::pow(2, 22) && n_c_h_w < std::pow(2, 29) && n_k_h_w < std::pow(2, 29) &&
         c_k_r_s < std::pow(2, 29);
    if(!ok)
    {
        return false;
    }
    // Check other constraints:
    const PerfParamsAsmDirect3x3WrW pp = mloComputePerfParamsAsmDirect3x3WrW(params);
    if(pp.reverse_inout == 0)
    {
        ok = (params.n_outputs % pp.c_per_wave) == 0 && (params.n_inputs % pp.k_per_wave) == 0;
    }
    else
    {
        ok = (params.n_outputs % pp.k_per_wave) == 0 && (params.n_inputs % pp.c_per_wave) == 0;
    }
    return ok;
}

bool ConvAsmBwdWrW3x3::IsFast(const ConvolutionContext&) const { return true; }

ConvSolution ConvAsmBwdWrW3x3::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfig&) const
{
    ConvSolution result;
    std::ostringstream options;
    GenerateClangDefsym(options, "batch_size", params.batch_sz); // N
    GenerateClangDefsym(options, "img_h", params.in_height);     // H
    GenerateClangDefsym(options, "img_w", params.in_width);      // W
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
    // Perf tune:
    const PerfParamsAsmDirect3x3WrW pp = mloComputePerfParamsAsmDirect3x3WrW(params);
    GenerateClangDefsym(options, "limit_wave_cnt", pp.limit_wave_cnt);
    GenerateClangDefsym(options, "chunk_size", pp.chunk_size);
    GenerateClangDefsym(options, "c_per_wave", pp.c_per_wave);
    GenerateClangDefsym(options, "k_per_wave", pp.k_per_wave);
    GenerateClangDefsym(options, "n_per_group", pp.n_per_group);
    GenerateClangDefsym(options, "pipe_lines_depth", pp.pipe_lines_depth);
    GenerateClangDefsym(options, "reverse_inout", pp.reverse_inout);
    // Debugging:
    GenerateClangDefsym(options, "enable_debug_output", 0);

    KernelInfo kernel;

    kernel.comp_options = options.str();

    kernel.l_wk.clear(); // workgroupsize
    kernel.l_wk.push_back(64 * pp.n_per_group);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.clear(); // gridsize
    kernel.g_wk.push_back(64 * pp.n_per_group);

    if(pp.reverse_inout == 0)
    {
        kernel.g_wk.push_back(params.n_outputs / pp.c_per_wave);
        kernel.g_wk.push_back(params.n_inputs / pp.k_per_wave);
    }
    else
    {
        kernel.g_wk.push_back(params.n_outputs / pp.k_per_wave);
        kernel.g_wk.push_back(params.n_inputs / pp.c_per_wave);
    }

    kernel.kernel_file = "conv3x3wrw.s";
    kernel.kernel_name = "gcnAsmConv3x3WrW";

    result.construction_params.push_back(kernel);
    result.workspce_sz = 0;
    return result;
}
} // namespace solver
} // namespace miopen
