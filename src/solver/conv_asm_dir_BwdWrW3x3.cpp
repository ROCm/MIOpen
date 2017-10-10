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
#include <unordered_map>

#include "miopen/solver.hpp"
#include "miopen/gcn_asm_utils.hpp"
#include "miopen/env.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS)

namespace miopen {
namespace solver {

static bool IsReverseInOutAllowed(const ConvolutionContext& config)
{
    return config.kernel_stride0 == 1 && config.kernel_stride1 == 1;
}

class Asm3x3WrwPerformanceConfig : public PerformanceConfig
{
    int limit_wave_cnt;   // [0..10]
    int reverse_inout;    // [0..1], 1 is allowed for stride=1x1 only.
    int chunk_size;       // {16,8}, Smaller values increase register pressure.
    int k_per_wave;       // {1,2,4,8} && ((chunk_size * k_per_wave) <= 64).
                          // Higher values increase register pressure.
    int pipe_lines_depth; // [1..16] && (pipe_lines_depth <= img_h).
                          // Higher values increase register pressure.
    int n_per_group;      // [1..8] && (n_per_group <= batch_size).

    // Values are within allowed range.
    bool IsValidRange() const;
    // Values are valid for specific problem config.
    bool IsValid(const ConvolutionContext& config) const;
    // 
    void EuristicInit(const ConvolutionContext& config);

    Asm3x3WrwPerformanceConfig(int limit_wave_cnt_, int reverse_inout_,
                               int chunk_size_, int k_per_wave_,
                               int pipe_lines_depth_, int n_per_group_)
    : limit_wave_cnt(limit_wave_cnt_), reverse_inout(reverse_inout_),
      chunk_size(chunk_size_), k_per_wave(k_per_wave_),
      pipe_lines_depth(pipe_lines_depth_), n_per_group(n_per_group_)
    {}
    std::string ToString() const;
    int GetCPerWave() const { assert(chunk_size != 0); return 64/chunk_size; }
    
    public:
    Asm3x3WrwPerformanceConfig()
    : Asm3x3WrwPerformanceConfig(0,0,16,4,2,1)
    {}
    void Serialize(std::ostream&) const override;
    bool Deserialize(const std::string& str) override;
    
    friend class ConvAsmBwdWrW3x3;
};

bool Asm3x3WrwPerformanceConfig::IsValidRange() const
{
    return (0 <= limit_wave_cnt && limit_wave_cnt <= 10) &&
           (0 <= reverse_inout && reverse_inout <= 1) &&
           (8 == chunk_size || 16 == chunk_size) &&
           (1 == k_per_wave || 2 == k_per_wave || 4 == k_per_wave || 8 == k_per_wave) &&
           (1 <= pipe_lines_depth && pipe_lines_depth <= 16) &&
           (1 <= n_per_group && n_per_group <= 8);
}

bool Asm3x3WrwPerformanceConfig::IsValid(const ConvolutionContext& config) const
{
    assert(chunk_size != 0);
    if ((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        return false;
    if ((reverse_inout ? config.n_inputs : config.n_outputs) % GetCPerWave() != 0)
        return false;
    if ((reverse_inout ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        return false;
    if (!(n_per_group <= config.batch_sz))
        return false;
    if (!(1 <= pipe_lines_depth && pipe_lines_depth <= std::min(config.out_height, 16))) // FIXME out_? What if stride != 1?
        return false;
    if (reverse_inout && !IsReverseInOutAllowed(config))
        return false;
    if (config.out_width >= 256 && n_per_group > 4) // when width >= 256, n_per_group should NOT be > 4. // FIXME out_? What if stride != 1?
        return false;
    return true;        
}

void Asm3x3WrwPerformanceConfig::EuristicInit(const ConvolutionContext& config)
{
    // chunk_size
    chunk_size = (config.out_width < 48) ? 8 : 16;
    if((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        chunk_size = 16; // Fixup for correctness
    // reverse_inout
    reverse_inout = 0;
    if(IsReverseInOutAllowed(config) && ((config.n_outputs % 4 != 0) || (config.out_width < 8)))
        reverse_inout = 1;
    // k_per_wave
    const auto c_k = config.n_outputs * config.n_inputs; // C*K
    if(c_k < 256)
        k_per_wave = 1;
    else if(c_k < 16384)
        k_per_wave = 2;
    else // C*K >= 16k
        k_per_wave = ((chunk_size == 8) ? 2 : 4);
    while((reverse_inout ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        k_per_wave /= 2; // Fixup for correctness
    // n_per_group
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
    if(config.out_width >= 256 && n_per_group > 4) // when width >= 256, n_per_group should not be > 4.
        n_per_group = 4;
    // pipe_lines_depth
    pipe_lines_depth = (config.out_height <= 1) ? 1 : 2;
    if((config.out_height < 8) && (config.out_width < 64))
    {
        pipe_lines_depth = config.out_height; // Special case.
    }
    assert(IsValidRange());
    if (!IsValid(config))
    {
        limit_wave_cnt = 0;
        reverse_inout = 0;
        chunk_size = 16; //CPerWave() = 4;
        k_per_wave = 1;
        pipe_lines_depth = 2;
        n_per_group = 1;
        if (config.n_outputs % 4 != 0) {
            /// (1) If reverse is Off, then both (C % c_per_wave) and (K % k_per_wave) must be 0.
            /// Toggling reverse swaps C and K in the condition above.
            /// (2) From the other hand, IsApplicable() ensures that either C or K is evenly divisable by 4.
            /// (3) We just set k_per_wave=1, c_per_wave=4. Therefore, (1) always can be satisfied here.
            /// If (C % c_per_wave) is not zero, just push reverse button so K and C will swap.
            ///
            /// \note C (input channels) resides in n_outputs, K (output channels) - in n_inputs,
            /// because that's how reverse convolutions are handled in MIOpen.
            reverse_inout = 1;
        }
        assert(IsValid(config));
    }
}

static
bool DeserializeField(char separator, std::istream& from, int& to)
{
    std::string part;

    if(!std::getline(from, part, separator))
        return false;

    char* end;
    to = std::strtol(part.c_str(), &end, 10);
    return part.c_str() == end;
}

void Asm3x3WrwPerformanceConfig::Serialize(std::ostream& stream) const
{
    static const auto sep = ','; // clang-format off
    stream << limit_wave_cnt
        << sep << reverse_inout
        << sep << chunk_size
        << sep << k_per_wave
        << sep << pipe_lines_depth
        << sep << n_per_group; // clang-format on
}

bool Asm3x3WrwPerformanceConfig::Deserialize(const std::string& str)
{
    Asm3x3WrwPerformanceConfig out;
    {
        std::istringstream tmp(str);
        const auto ok = // clang-format off
            DeserializeField(',', tmp, out.limit_wave_cnt) &&
            DeserializeField(',', tmp, out.reverse_inout) &&
            DeserializeField(',', tmp, out.chunk_size) &&
            DeserializeField(',', tmp, out.k_per_wave) &&
            DeserializeField(',', tmp, out.pipe_lines_depth) &&
            DeserializeField(',', tmp, out.n_per_group); // clang-format on

        if(!ok)
            return false;
    }
    *this = out;
    return true;
}

std::string Asm3x3WrwPerformanceConfig::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

std::unique_ptr<PerformanceConfig> ConvAsmBwdWrW3x3::PerformanceConfigImpl() const
{
    return make_unique<Asm3x3WrwPerformanceConfig>();
}

void ConvAsmBwdWrW3x3::InitPerformanceConfigImpl(const ConvolutionContext& params,
                                                 PerformanceConfig& result) const
{
    static const std::unordered_map<std::string, std::string> perf_vals_map({
        // clang-format off
        //            W    H    c    n    k {u  v} dir {CUs}            lwc[2] rio csz[2] kpw pld[2] npg
        {MakeLutKey(  7,   7, 160, 128, 320, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  7, 1).ToString()},                                    
        {MakeLutKey(  7,   7, 192, 128, 384, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  7, 1).ToString()},                                    
        {MakeLutKey(  7,   7, 512,  16, 512, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  7, 1).ToString()},                                    
        {MakeLutKey( 12,  12, 512, 128,1024, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 8, 11, 1).ToString()},                                    
        {MakeLutKey( 12,  12,1024, 128,1024, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 8, 11, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 192, 128, 384, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 192, 128, 384, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 256,  50, 384, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 8, 11, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 256, 128, 256, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 256, 128, 256, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 256, 128, 384, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 256, 128, 384, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 384,  50, 256, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 8, 11, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 384,  50, 384, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 8, 11, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 384,  64, 256, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 8, 11, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 384, 128, 256, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 384, 128, 256, 0, 64),       Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 384, 128, 384, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 13,  13, 384, 128, 384, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0,  8, 8,  2, 1).ToString()},                                    
        {MakeLutKey( 14,  14,  96, 128, 208, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  7, 2).ToString()},                                    
        {MakeLutKey( 14,  14, 112, 128, 224, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  5, 2).ToString()}, /// \todo Find opt values for 56CUs
        {MakeLutKey( 14,  14, 128,   8, 256, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 128,  32, 192, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  3, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 128, 128, 256, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 144, 128, 288, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  5, 2).ToString()},                                    
        {MakeLutKey( 14,  14, 160,  32, 160, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4, 11, 2).ToString()},                                    
        {MakeLutKey( 14,  14, 160,  32, 192, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 8,  5, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 160, 128, 320, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  5, 2).ToString()},                                    
        {MakeLutKey( 14,  14, 192,  32, 256, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  3, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 256,  16, 256, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 8, 11, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 256,  16, 256, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0,  8, 8,  7, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 256,  32, 256, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0,  8, 8,  4, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 512,   8, 512, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 512,  16, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 512,  16, 512, 0, 64),       Asm3x3WrwPerformanceConfig(0, 1,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 512,  32, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 14,  14, 512,  64, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 16,  16, 256,   8, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  2, 1).ToString()},                                    
        {MakeLutKey( 27,  27, 128,   8, 128, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  3, 1).ToString()}, /// \todo Find opt values for 56CUs
        {MakeLutKey( 28,  28,  64,  32,  64, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 2,  2, 2).ToString()},                                    
        {MakeLutKey( 28,  28,  64,  32,  96, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 2,  5, 2).ToString()},                                    
        {MakeLutKey( 28,  28,  96,  32,  96, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  3, 1).ToString()},                                    
        {MakeLutKey( 28,  28,  96, 128, 128, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  2, 2).ToString()},                                    
        {MakeLutKey( 28,  28, 128,  16, 128, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  3, 1).ToString()},                                    
        {MakeLutKey( 28,  28, 128,  32, 160, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  3, 1).ToString()},                                    
        {MakeLutKey( 28,  28, 128, 128, 192, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  2, 2).ToString()},                                    
        {MakeLutKey( 28,  28, 256,   8, 512, 0),           Asm3x3WrwPerformanceConfig(4, 1,  8, 2,  2, 1).ToString()},                                    
        {MakeLutKey( 28,  28, 256,  16, 512, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 2,  3, 1).ToString()},                                    
        {MakeLutKey( 28,  28, 256,  32, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 28,  28, 256,  64, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 28,  28, 512,  32, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 28,  28, 512,  64, 512, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 54,  54,  64,   8,  64, 0),           Asm3x3WrwPerformanceConfig(0, 1, 16, 2,  2, 4).ToString()},                                    
        {MakeLutKey( 54,  54,  64,   8,  64, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0,  8, 2,  3, 2).ToString()},                                    
        {MakeLutKey( 56,  56,  64,  16,  64, 2, 2, 0),     Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  3, 2).ToString()},                                    
        {MakeLutKey( 56,  56,  64,  16, 192, 0),           Asm3x3WrwPerformanceConfig(0, 0,  8, 4,  2, 4).ToString()},                                    
        {MakeLutKey( 56,  56,  64,  32, 192, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  4, 4).ToString()},                                    
        {MakeLutKey( 56,  56,  64, 128, 192, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  4, 1).ToString()},                                    
        {MakeLutKey( 56,  56, 256,  32, 256, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 2,  4, 1).ToString()},                                    
        {MakeLutKey( 56,  56, 256,  64, 256, 0),           Asm3x3WrwPerformanceConfig(0, 1,  8, 2,  4, 1).ToString()},                                    
        {MakeLutKey( 60,   6,  64,  16, 128, 0),           Asm3x3WrwPerformanceConfig(4, 0, 16, 2,  6, 1).ToString()},                                    
        {MakeLutKey( 60,   6,  64,  16, 128, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0, 16, 2,  2, 1).ToString()},                                    
        {MakeLutKey(112, 112,  64,   8, 128, 0),           Asm3x3WrwPerformanceConfig(3, 0, 16, 4,  2, 2).ToString()},                                    
        {MakeLutKey(112, 112,  64,   8, 128, 0, 64),       Asm3x3WrwPerformanceConfig(0, 1, 16, 4,  2, 1).ToString()},                                    
        {MakeLutKey(112, 112,  64,  16, 128, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  2, 4).ToString()},                                    
        {MakeLutKey(112, 112,  64,  16, 128, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  3, 1).ToString()},                                    
        {MakeLutKey(112, 112,  64,  32, 128, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  2, 4).ToString()},                                    
        {MakeLutKey(112, 112,  64,  32, 128, 0, 64),       Asm3x3WrwPerformanceConfig(0, 1, 16, 4,  3, 1).ToString()},                                    
        {MakeLutKey(112, 112,  64,  64, 128, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  2, 4).ToString()},                                    
        {MakeLutKey(112, 112, 256,   8, 512, 0),           Asm3x3WrwPerformanceConfig(0, 1, 16, 4,  2, 1).ToString()},                                    
        {MakeLutKey(120,  12,  32,  16,  64, 0),           Asm3x3WrwPerformanceConfig(3, 1, 16, 2,  1, 4).ToString()},                                    
        {MakeLutKey(120,  12,  32,  16,  64, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0, 16, 2,  2, 2).ToString()},                                    
        {MakeLutKey(224, 224,   3,   8,  64, 0, 64),       Asm3x3WrwPerformanceConfig(0, 1, 16, 1,  2, 4).ToString()}, /// \todo Find opt values for 56CUs
        {MakeLutKey(224, 224,   3,  16,  64, 0, 64),       Asm3x3WrwPerformanceConfig(0, 1, 16, 1,  5, 4).ToString()}, /// \todo Find opt values for 56CUs
        {MakeLutKey(240,  24,  16,  16,  32, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  1, 8).ToString()},                                    
        {MakeLutKey(240,  24,  16,  16,  32, 0, 64),       Asm3x3WrwPerformanceConfig(0, 0, 16, 2,  1, 8).ToString()},                                    
        {MakeLutKey(256, 128,  96,   1, 128, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  1, 1).ToString()},                                    
        {MakeLutKey(256, 128, 128,   1, 192, 0),           Asm3x3WrwPerformanceConfig(0, 0, 16, 4,  1, 1).ToString()},                                    
        {MakeLutKey(512, 256,  64,   1, 192, 0),           Asm3x3WrwPerformanceConfig(0, 1, 16, 4,  1, 1).ToString()},
        // clang-format on
    });

    std::string s;
    Asm3x3WrwPerformanceConfig pp;
    const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS{});
    if(p_asciz)
    {
        s = std::string(p_asciz);
    }
    if(!s.empty()) // Otherwise, nothing is set in env -> nothing to parse.
    {
        if (!pp.Deserialize(s))
        {
            MIOPEN_THROW("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: Bad format (hint: lwc,rio,csz,kpw,lpd,npg)");
        }
        if(!pp.IsValidRange())
        {
            MIOPEN_THROW("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: Out of range.");
        }
        if (!pp.IsValid(params))
        {
            MIOPEN_THROW("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: Incorrect for the problem config.");
        }
    }
    else
    {
        // Try to get values from LUT. If not found, use heuristic algorithm.
        // At first, try to find numCUs-specific values.
        const auto numCUs = static_cast<int>(params.GetStream().GetMaxComputeUnits());
        auto key          = MakeLutKey(params.out_width,
                              params.out_height,
                              params.n_outputs,
                              params.batch_sz,
                              params.n_inputs,
                              params.kernel_stride0,
                              params.kernel_stride1,
                              0,
                              numCUs);
        auto found = perf_vals_map.find(key);
        if(found == perf_vals_map.end())
        { // numCUs-specific values not found, try to find "universal" ones.
            key = MakeLutKey(params.out_width,
                             params.out_height,
                             params.n_outputs,
                             params.batch_sz,
                             params.n_inputs,
                             params.kernel_stride0,
                             params.kernel_stride1,
                             0);
            found = perf_vals_map.find(key);
        }
        if(found != perf_vals_map.end())
        {
            s = found->second;
            if (!pp.Deserialize(s))
            {
                MIOPEN_THROW("ConvAsmBwdWrW3x3: LUT entry: Bad format (hint: lwc,rio,csz,kpw,lpd,npg)");
            }
            if(!pp.IsValidRange())
            {
                MIOPEN_THROW("ConvAsmBwdWrW3x3: LUT entry: Out of range.");
            }
            if (!pp.IsValid(params))
            {
                MIOPEN_THROW("ConvAsmBwdWrW3x3: LUT entry: Incorrect for the problem config.");
            }
        }
        else
        {
            pp.EuristicInit(params);
        }
    }
    dynamic_cast<Asm3x3WrwPerformanceConfig&>(result) = pp;
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
    bool ok = params.pad0 == 1                   // -q  pad_w
              && params.pad1 == 1                // -p  pad_h
              && (params.kernel_stride0 <= 2)    // -u  stride_w
              && (params.kernel_stride1 <= 2)    // -v  stride_h
              && params.kernel_size0 == 3        // -x  S wei_w
              && params.kernel_size1 == 3        // -y  R wei_h
              && params.in_layout == "NCHW";
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
         && c_k_r_s < std::pow(2, 29); // clang-format on
    if(!ok)
    {
        return false;
    }
    return true;
}

bool ConvAsmBwdWrW3x3::IsFast(const ConvolutionContext&) const { return true; }

ConvSolution ConvAsmBwdWrW3x3::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfig& config) const
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
    // Perf tune:
    const auto& pp = dynamic_cast<const Asm3x3WrwPerformanceConfig&>(config);
    GenerateClangDefsym(options, "limit_wave_cnt", pp.limit_wave_cnt);
    GenerateClangDefsym(options, "chunk_size", pp.chunk_size);
    GenerateClangDefsym(options, "c_per_wave", pp.GetCPerWave());
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
        kernel.g_wk.push_back(params.n_outputs / pp.GetCPerWave());
        kernel.g_wk.push_back(params.n_inputs / pp.k_per_wave);
    }
    else
    {
        kernel.g_wk.push_back(params.n_outputs / pp.k_per_wave);
        kernel.g_wk.push_back(params.n_inputs / pp.GetCPerWave());
    }

    kernel.kernel_file = "conv3x3wrw.s";
    kernel.kernel_name = "gcnAsmConv3x3WrW";

    result.construction_params.push_back(kernel);
    result.workspce_sz = 0;
    return result;
}

void ConvAsmBwdWrW3x3::Search(const ConvolutionContext& params, PerformanceConfig& config) const
{
#if 0
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

    size_t localMemSize = profile_h.GetLocalMemorySize();
    profile_h.EnableProfiling();

    // const auto hw_wave_sz = 64;
    const auto dev_local_mem_sz = localMemSize; // in bytes

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
    int n_out_tiles_rg[2]  = {1, 8};
    int n_in_tiles_rg[2]   = {1, 4};
    int n_in_stacks_sz[3]  = {1, 2, 4};
    int in_tiles[4]        = {64, 128, 256, 2048};
    /*
    std::vector<int> v_tile_sz;
    std::vector<int> v_out_pix_tile_sz;
    std::vector<int> v_n_out_tiles_rg;
    std::vector<int> v_n_in_tiles_rg;
    std::vector<int> v_n_in_stacks_sz;
    */
    //

    double min_proc_time = std::numeric_limits<float>::max();

#if 1

    size_t run_counter = 0;
    int n_grp_tiles1   = 2;
    int n_grp_tiles0   = 2;

    int out_pix_tl_cnt = 3; // out_pix_tile_sz[1];
    int n_out_tls      = n_out_tiles_rg[1];
    int stack_cnt      = 2;
    int n_tile0_sz     = 3;
    int n_tile1_sz     = 3;

    n_out_tls = std::min(params.n_outputs, n_out_tls);

    if(params.in_width <= 8)
    {
        n_tile0_sz       = 1;
        n_in_tiles_rg[1] = 16;
    }
    else if(params.in_width <= 16)
    {
        n_tile0_sz       = 1;
        tile_sz0[0]      = 16;
        n_in_tiles_rg[1] = 8;
    }
    else if(params.in_width <= 32)
    {
        n_tile0_sz  = 2;
        tile_sz0[0] = 16;
        tile_sz0[1] = 32;
    }

    if(params.in_height <= 8)
    {
        n_tile1_sz       = 1;
        n_in_tiles_rg[1] = 16;
    }
    else if(params.in_height <= 16)
    {
        n_tile1_sz       = 1;
        tile_sz1[0]      = 16;
        n_in_tiles_rg[1] = 8;
    }
    else if(params.in_width <= 32)
    {
        n_tile1_sz  = 2;
        tile_sz1[0] = 16;
        tile_sz1[1] = 32;
    }

    bool unaligned = (params.out_height < 8 || params.out_width < 8 ||
                      (params.out_height > 8 && params.out_height < 16) ||
                      (params.out_width > 8 && params.out_width < 16) ||
                      (params.out_height > 16 && params.out_height < 32) ||
                      (params.out_width > 16 && params.out_width < 32));

    if(unaligned)
    {
        out_pix_tile_sz[1] = 6;
        out_pix_tl_cnt     = out_pix_tile_sz[1];
    }

    int n_grp_tiles = n_grp_tiles1 * n_grp_tiles0;

    int n_tiles_cnt = n_tile0_sz * n_tile1_sz;
    n_grp_tiles = (params.out_height > 16 && params.out_width > 16) ? n_grp_tiles - 1 : n_grp_tiles;
    n_tiles_cnt = (params.out_height > 16 && params.out_width > 16) ? n_tiles_cnt - 1 : n_tiles_cnt;
    size_t report_inteval = 100;
    //			_n_timer_iter = 250;

    long long runs_left = 0;

    if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
    {
        grp_tl_ln[0] = 64;
        grp_tl_ln[1] = 128;
        grp_tl_ln[2] = 256;
        grp_tl_ln[3] = 512;
        n_grp_tiles1 = 1;
        n_grp_tiles0 = 4;

        tile_sz1[0] = 1;
        tile_sz0[0] = 4;
        n_tile0_sz = n_tile1_sz = 1;
        n_tiles_cnt             = n_tile0_sz * n_tile1_sz;
        out_pix_tile_sz[0]      = (unaligned) ? 0 : out_pix_tile_sz[0];
        out_pix_tile_sz[1]      = 1;
        n_out_tiles_rg[1]       = 16;
        n_in_tiles_rg[1]        = 8;
        stack_cnt               = 3;
        out_pix_tl_cnt          = out_pix_tile_sz[1];
        n_out_tls               = n_out_tiles_rg[1];
        n_grp_tiles             = n_grp_tiles1 * n_grp_tiles0;

        report_inteval = 20;
    }

    if(params.kernel_size0 == 1 && params.kernel_size1 == 1 && params.n_outputs % 4 == 0 &&
       params.n_inputs % 4 == 0)
    {

        std::cout << "Searching the best solution in the 4 dim space. Please, be patient it may "
                     "take few minutes."
                  << std::endl;
        result.grp_tile1 = 1;
        result.in_tile1  = 1;
        result.in_tile0  = 1;
        report_inteval   = 4;

        // Add 1x1_stride : no padding support yet
        if(params.forward && params.kernel_stride0 == 1 && params.kernel_stride1 == 1 &&
           params.n_inputs % 16 == 0 && params.n_outputs % 16 == 0)
        {

            // uint N_LCL_IN_MAPS = result.n_in_data_tiles;
            n_in_tiles_rg[0] = 0;
            n_in_tiles_rg[1] = 3;
            int n_in_tls     = 4;

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

            n_grp_tiles  = n_grp_tiles1 * n_grp_tiles0;
            n_out_tls    = (n_out_tiles_rg[1] - n_out_tiles_rg[0] + 1);
            int n_in_tls = 2;
            runs_left    = n_grp_tiles * out_pix_tl_cnt * n_out_tls * n_in_tls;

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

                        if(ret != 0)
                        {
                            std::cout << "Failed run." << std::endl;
                            runs_left--;
                            runs_left = (runs_left < 0) ? 0 : runs_left;
                            continue;
                        }

                        if(run_counter != 0 && run_counter % report_inteval == 0)
                        {
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

                        run_counter++;
                        runs_left--;
                        runs_left = (runs_left < 0) ? 0 : runs_left;
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
                    n_in_tiles_rg[1] * stack_cnt;

        for(int g1 = 0; g1 < n_grp_tiles1; g1++)
        {
            result.grp_tile1 =
                (params.kernel_size0 == 1 && params.kernel_size1 == 1) ? 1 : grp_tl_ln[g1];
            for(int g0 = 0; g0 < n_grp_tiles0; ++g0)
            {
                result.grp_tile0 = grp_tl_ln[g0];

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

                        for(int k = (unaligned) ? out_pix_tile_sz[0] : 0; k < out_pix_tl_cnt; ++k)
                        {
                            result.out_pix_tile1 = (unaligned) ? k : out_pix_tile_sz[k];
                            if(result.out_pix_tile1 > result.in_tile1)
                            {
                                runs_left--;
                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                continue;
                            }
                            // out pix 0

                            for(int l = (unaligned) ? out_pix_tile_sz[0] : 0; l < out_pix_tl_cnt;
                                ++l)
                            {
                                result.out_pix_tile0 =
                                    (params.kernel_size0 == 1 && params.kernel_size1 == 1)
                                        ? 4
                                        : (unaligned) ? l : out_pix_tile_sz[l];

                                if(result.out_pix_tile0 > result.in_tile0)
                                {
                                    runs_left--;
                                    runs_left = (runs_left < 0) ? 0 : runs_left;
                                    continue;
                                }

                                int o_l = n_out_tiles_rg[1];
                                for(int o_t = n_out_tiles_rg[0]; o_t <= o_l; ++o_t)
                                {
                                    result.n_out_pix_tiles = o_t;
                                    if(params.n_outputs < result.n_out_pix_tiles)
                                    {
                                        runs_left--;
                                        runs_left = (runs_left < 0) ? 0 : runs_left;
                                        continue;
                                    }
#if 1
                                    if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
                                    {
                                        int N4S = 1;

                                        int MAP_SZ4 =
                                            (params.in_width * params.in_height + N4S * 4 - 1) /
                                            (N4S * 4);

                                        int GRP_SZ          = result.grp_tile0;
                                        int N_MAPS_PERGROUP = 1;
                                        int exchange_step;

                                        if(MAP_SZ4 <= GRP_SZ / 2)
                                        {
                                            N_MAPS_PERGROUP   = GRP_SZ / MAP_SZ4;
                                            int lcl_mem_avial = (result.grp_tile0 <= 192)
                                                                    ? (dev_local_mem_sz / 4) / 2
                                                                    : (dev_local_mem_sz / 4);

                                            exchange_step =
                                                lcl_mem_avial / (N_MAPS_PERGROUP * MAP_SZ4 * 4);
                                            exchange_step = std::min(
                                                std::min(exchange_step, result.n_out_pix_tiles),
                                                N_MAPS_PERGROUP);
                                            if(exchange_step < result.n_out_pix_tiles)
                                            {
                                                auto tmp_stp = static_cast<int>(std::ceil(
                                                    std::sqrt(static_cast<float>(exchange_step))));
                                                n_in_tiles_rg[0] = tmp_stp;
                                                n_in_tiles_rg[1] = exchange_step;
                                            }
                                            else
                                            {
                                                n_in_tiles_rg[0] = 1;
                                                n_in_tiles_rg[1] = 1;
                                            }
                                        }
                                    }
#endif
                                    for(int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1]; ++i_t)
                                    {
                                        result.n_in_data_tiles = i_t;
                                        if(params.n_inputs < result.n_in_data_tiles)
                                        {
                                            runs_left--;
                                            runs_left = (runs_left < 0) ? 0 : runs_left;
                                            continue;
                                        }

                                        for(int s = 0; s < stack_cnt; ++s)
                                        {

                                            result.n_stacks = n_in_stacks_sz[s];
                                            if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
                                            {
                                            }
                                            else
                                            {
                                                int alu_tile0 = std::max(
                                                    1, result.in_tile0 / result.out_pix_tile0);
                                                int alu_tile1 = std::max(
                                                    1, result.in_tile1 / result.out_pix_tile1);
                                                int alu_tiles_sz = (alu_tile0 * alu_tile1);
                                                int n_alus_total =
                                                    (result.grp_tile0 * result.grp_tile1);

                                                if(alu_tiles_sz >
                                                   n_alus_total /* || result.n_in_data_tiles*result.n_out_pix_tiles*result.out_pix_tile1*result.out_pix_tile0 > 240*/)
                                                {
                                                    runs_left--;
                                                    runs_left = (runs_left < 0) ? 0 : runs_left;
                                                    continue;
                                                }
                                            }

                                            if(result.n_stacks > params.batch_sz)
                                            {
                                                runs_left--;
                                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                                continue;
                                            }
                                            const auto ret = MeasureLoop(&profile_h,
                                                                         bot_ocl_buf.get(),
                                                                         top_ocl_buf.get(),
                                                                         wei_ocl_buf.get(),
                                                                         bias_ocl_buf.get(),
                                                                         processing_time,
                                                                         params,
                                                                         result);

                                            if(ret != 0)
                                            {
                                                std::cout << "Failed run." << std::endl;
                                                runs_left--;
                                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                                continue;
                                            }

                                            if(run_counter != 0 &&
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

                                            run_counter++;
                                            runs_left--;
                                            runs_left = (runs_left < 0) ? 0 : runs_left;
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
#endif
}

} // namespace solver
} // namespace miopen
