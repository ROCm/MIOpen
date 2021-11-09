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

#ifndef GUARD_MIOPEN_SOLVER_HPP_
#define GUARD_MIOPEN_SOLVER_HPP_

#include <miopen/config.h>

#include <miopen/conv_solution.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/rocm_features.hpp>
#include <miopen/type_name.hpp>
#include <miopen/miopen.h>
#include <miopen/buffer_info.hpp>

#include <memory>
#include <string>
#include <vector>
#include <ostream>
#include <algorithm>
#include <initializer_list>

namespace miopen {

namespace debug {

/// If set to true, then always enable ConvDirectNaive* solver, regardless of environment value
/// MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_* that control enable/disable of these solvers.
/// Currently used during driver using naive kernel as gpu reference.
extern bool
    AlwaysEnableConvDirectNaive; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug

struct AnyInvokeParams;

namespace solver {
/// \todo Move wave_size into abstraction wich represent GPU information
const int wave_size = 64;
template <class Solver>
std::string ComputeSolverDbId(Solver)
{
    const auto& const_name = get_type_name<Solver>();
    auto idx               = const_name.find_last_of(':');
    auto name              = const_name.substr(idx + 1);
    std::replace(name.begin(), name.end(), ',', '-');
    name.erase(std::remove(name.begin(), name.end(), ' '), name.end());

    return name;
}

// This will retrieve the id of the solver to write to the database. By
// default it uses the class name. If the class is renamed, this function can
// overriden to keep the name to avoid DB corruption.
template <class Solver>
const std::string& SolverDbId(Solver solver)
{
    static const auto result = ComputeSolverDbId(solver);
    return result;
}

/// Base class for problem solvers.
///
/// Solvers are to be instantiated as const objects and shall not have any variable
/// internal state. Any non-const state information, if required, to be stored in the
/// solver-specific context objects.
///
/// There could be multiple solvers of the same algorithm for a problem config.
template <class Context>
struct SolverBase
{

    /// Initializes performance config to the default values.
    /// The function may involve some heuristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    /// Only implemented by SearchableSolvers
    /// PerformanceConfig GetPerformanceConfig(const ConvolutionContext&) const;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    /// Only implemented by SearchableSolvers
    /// bool IsValidPerformanceConfig(const Context&, const PerformanceConfig&) const
    /// {
    ///     return true; // Do not check by default.
    /// }

    /// Returns true if solution can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    ///
    /// Every SolverBase which IsApplicable() for some problem config must be able to
    /// GetPerformanceConfig() so that GetSolution() would return valid
    /// solution for a problem (i.e. convolution). In other words, if a Solution
    /// says "I'm suitable" for a problem, it agrees to solve that problem correctly.
    bool IsApplicable(const Context&) const { return false; }

    /// [Informative as of Sep 2020] The minimum requirement for Dynamic Solvers:
    /// Batch size and input picture size (N, W, H) must NOT be compiled into the
    /// kernel(s) that consist a Solution. These must go into the kernel as a
    /// run-time parameters.
    bool IsDynamic() const { return false; }

    /// [Informative as of Sep 2020] Returns an approximated value of the expected
    /// WTI or -2.0 when this value can't be computed. Tips:
    /// * Value 1.0 corresponds to the 100% utilization of HW capabilities as
    ///   if Direct computational algorithm is used.
    /// * [Notice] WTI may exceed 1.0 for highly optimized algorithms like Winograd.
    /// * @see https://github.com/ROCmSoftwarePlatform/MIOpen/issues/410
    float GetWti(const Context&) const { return -2.0; }

    // Returns the workspace size required by the solver for a given ConvolutionContext
    size_t GetWorkspaceSize(const Context&) const { return 0; };

    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    /// ConvSolution GetSolution(const ConvolutionContext& params) const;

    /// Searchable solvers provide a GetSolution that takes a Context and PerformanceConfig
    /// ConvSolution GetSolution(const ConvolutionContext& params,
    ///                          const PerformanceConfig& config) const;
};

struct PerformanceConfigConvAsm3x3U : Serializable<PerformanceConfigConvAsm3x3U>
{
    int limit_wave_cnt;        // [0..9]
    int filters_per_wave;      // [1..8]
    int output_lines_per_wave; // [1..8]

    PerformanceConfigConvAsm3x3U(int lwc, int fpw, int olpw);
    PerformanceConfigConvAsm3x3U() : PerformanceConfigConvAsm3x3U(-1, -1, -1) {}
    PerformanceConfigConvAsm3x3U(bool) : PerformanceConfigConvAsm3x3U(0, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.filters_per_wave, "filters_per_wave");
        f(self.output_lines_per_wave, "output_lines_per_wave");
    }

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvAsm3x3U& other) const;
    std::string ToString() const;
};

struct ConvAsm3x3U : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    PerformanceConfigConvAsm3x3U GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvAsm3x3U&) const;
    PerformanceConfigConvAsm3x3U Search(const ConvolutionContext&,
                                        const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsm3x3U& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConfigConvAsm1x1U : Serializable<PerformanceConfigConvAsm1x1U>
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int read_size;        // [1..4]            <same>          <same>
    int k_mult;           // 1,[4,8,12..32]    2^n[8..32]      1,4
    int chunks_per_wave;  // [1..16]           [1..8]          <same>
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     1,4
    int n_mult;           // [1..8]            [1..4]          <same>
    int c_mult;           // 2^n[1..32]        2^n[1..4]       <same>
    int waves_c_in_group; // [1..8]            [1..4]          <same>
    int waves_k_in_group; // 1,[2,4,8]         1,[2,4,8]       <same>
    bool use_spare_set;

    PerformanceConfigConvAsm1x1U(int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1U(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.read_size, "read_size");
        f(self.k_mult, "k_mult");
        f(self.chunks_per_wave, "chunks_per_wave");
        f(self.chunk_size, "chunk_size");
        f(self.n_mult, "n_mult");
        f(self.c_mult, "c_mult");
        f(self.waves_c_in_group, "waves_c_in_group");
        f(self.waves_k_in_group, "waves_k_in_group");
    }

    // clang-format off
    int GetReadSize() const { return read_size; }
    int GetKMult() const { return k_mult; }
    int GetChunksPerWave() const { return chunks_per_wave; }
    int GetChunkSize() const { return chunk_size; }
    int GetNMult() const { return n_mult; }
    int GetCMult() const { return c_mult; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const { assert(chunk_size); return 64 / chunk_size; }
    // clang-format on

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvAsm1x1U& other) const;
    std::string ToString() const;
};

struct ConvAsm1x1U : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvAsm1x1U GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvAsm1x1U&) const;
    PerformanceConfigConvAsm1x1U Search(const ConvolutionContext&,
                                        const AnyInvokeParams& invoke_ctx) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsm1x1U& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConfigConvBiasActivAsm1x1U : PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvBiasActivAsm1x1U& other) const;
};

struct ConvBiasActivAsm1x1U : ConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U GetPerformanceConfig(const ConvolutionContext&) const;

    PerformanceConfigConvBiasActivAsm1x1U Search(const ConvolutionContext&,
                                                 const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsm1x1U& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConfigConvAsm1x1UV2 : Serializable<PerformanceConfigConvAsm1x1UV2>
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     <same>
    int dwords_per_ld;    // [1..4]            1,2,3           <same>
    int k_mult;           // [1..32]           8,16            1,2,3,4
    int c_mult;           // [1..32]           2^n[1..4]       <same>
    int n_mult;           // [1..32]           1,2             <same>
    int w_mult;           // [1..32]           1,2             <same>
    int h_mult;           // [1..32]           1,2             <same>
    int h_per_chunk;      // 2^n[1..64]        [2,4,8]         <same>
    int waves_k_in_group; // [1..8]            2,4             <same>
    int waves_c_in_group; // [1..8]            1,2             <same>
    bool use_spare_set;

    PerformanceConfigConvAsm1x1UV2(int, int, int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1UV2()
        : PerformanceConfigConvAsm1x1UV2(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1UV2(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.dwords_per_ld, "dwords_per_ld");
        f(self.k_mult, "k_mult");
        f(self.c_mult, "c_mult");
        f(self.n_mult, "n_mult");
        f(self.w_mult, "w_mult");
        f(self.h_mult, "h_mult");
        f(self.h_per_chunk, "h_per_chunk");
        f(self.waves_k_in_group, "waves_k_in_group");
        f(self.waves_c_in_group, "waves_c_in_group");
    }

    // clang-format off
    int GetChunkSize() const { return chunk_size; }
    int GetDwordsPerLd() const { return dwords_per_ld; }
    int GetCMult() const { return c_mult; }
    int GetKMult() const { return k_mult; }
    int GetNMult() const { return n_mult; }
    int GetWMult() const { return w_mult; }
    int GetHMult() const { return h_mult; }
    int GetHPerChunk() const { return h_per_chunk; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const { assert(chunk_size); return 64 / chunk_size; }
    // clang-format on

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvAsm1x1UV2& other) const;
    std::string ToString() const;
};

struct ConvAsm1x1UV2 : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvAsm1x1UV2 GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvAsm1x1UV2&) const;
    PerformanceConfigConvAsm1x1UV2 Search(const ConvolutionContext&,
                                          const AnyInvokeParams& invoke_ctx) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsm1x1UV2& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvAsm5x10u2v2f1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvAsm5x10u2v2b1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvAsm7x7c3h224w224k64u2v2p3q3f1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwd11x11 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwdGen : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct PerformanceImplicitGemm : Serializable<PerformanceImplicitGemm>
{
    int BPerBlock; // 2^n[8..16]
    int KPerBlock; // 2^n[32..128]
    int EPerBlock; // 2^n[4..16]

    int GemmNRepeat; // == 2

    int GemmMPerThreadSubC; // 2^n[2..4]
    int GemmNPerThreadSubC; // 2^n[2..4]

    int GemmMLevel0Cluster; // 2^n[1..4]
    int GemmNLevel0Cluster; // 2^n[1..4]
    int GemmMLevel1Cluster; // 2^n[1..4]
    int GemmNLevel1Cluster; // 2^n[1..4]

    int InBlockCopyClusterLengths_E;  // 2^n[4..16]
    int InBlockCopyClusterLengths_B;  // 2^n[8..16]
    int InBlockCopyClusterLengths_N1; // 2^n[1..2]
    int InBlockCopyClusterLengths_N2; // 2^n[1..4]

    int WeiBlockCopyClusterLengths_E; // 2^n[1..4]
    int WeiBlockCopyClusterLengths_K; // 2^n[16..128]

    bool use_spare_set;

    PerformanceImplicitGemm(
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, bool);

    PerformanceImplicitGemm()
        : PerformanceImplicitGemm(
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemm(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BPerBlock, "BPerBlock");
        f(self.KPerBlock, "KPerBlock");
        f(self.EPerBlock, "EPerBlock");
        f(self.GemmNRepeat, "GemmNRepeat");
        f(self.GemmMPerThreadSubC, "GemmMPerThreadSubC");
        f(self.GemmNPerThreadSubC, "GemmNPerThreadSubC");
        f(self.GemmMLevel0Cluster, "GemmMLevel0Cluster");
        f(self.GemmNLevel0Cluster, "GemmNLevel0Cluster");
        f(self.GemmMLevel1Cluster, "GemmMLevel1Cluster");
        f(self.GemmNLevel1Cluster, "GemmNLevel1Cluster");
        f(self.InBlockCopyClusterLengths_E, "InBlockCopyClusterLengths_E");
        f(self.InBlockCopyClusterLengths_N1, "InBlockCopyClusterLengths_N1");
        f(self.InBlockCopyClusterLengths_B, "InBlockCopyClusterLengths_B");
        f(self.InBlockCopyClusterLengths_N2, "InBlockCopyClusterLengths_N2");
        f(self.WeiBlockCopyClusterLengths_E, "WeiBlockCopyClusterLengths_E");
        f(self.WeiBlockCopyClusterLengths_K, "WeiBlockCopyClusterLengths_K");
    }

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& ctx) const;
    bool operator==(const PerformanceImplicitGemm& other) const;
    std::string ToString() const;
};

struct PerformanceImplicitGemmV4R1 : public PerformanceImplicitGemm
{
    PerformanceImplicitGemmV4R1(int a,
                                int b,
                                int c,
                                int d,
                                int e,
                                int f,
                                int g,
                                int h,
                                int i,
                                int j,
                                int k,
                                int l,
                                int m,
                                int n,
                                int o,
                                int p,
                                bool q)
        : PerformanceImplicitGemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
    {
    }

    PerformanceImplicitGemmV4R1()
        : PerformanceImplicitGemmV4R1(
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R1(bool spare) : PerformanceImplicitGemm(spare) {}

    bool IsValid(const ConvolutionContext& ctx) const;
};

struct PerformanceImplicitGemmV4R4Fwd : Serializable<PerformanceImplicitGemmV4R4Fwd>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmV4R4Fwd(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4Fwd(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4Fwd(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4Fwd() : PerformanceImplicitGemmV4R4Fwd(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4Fwd(bool spare);

    bool operator==(const PerformanceImplicitGemmV4R4Fwd& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    std::string ToString() const;
};

struct PerformanceImplicitGemmV4R4WrW : Serializable<PerformanceImplicitGemmV4R4WrW>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmV4R4WrW(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4WrW(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4WrW(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4WrW() : PerformanceImplicitGemmV4R4WrW(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4WrW(bool spare);

    bool operator==(const PerformanceImplicitGemmV4R4WrW& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    std::string ToString() const;
};

struct PerformanceImplicitGemmBwdDataV1R1 : Serializable<PerformanceImplicitGemmBwdDataV1R1>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmBwdDataV1R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV1R1()
        : PerformanceImplicitGemmBwdDataV1R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV1R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV1R1(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmBwdDataV1R1(bool spare);

    bool operator==(const PerformanceImplicitGemmBwdDataV1R1& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    std::string ToString() const;
};

struct PerformanceImplicitGemmBwdDataV4R1 : Serializable<PerformanceImplicitGemmBwdDataV4R1>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmBwdDataV4R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV4R1()
        : PerformanceImplicitGemmBwdDataV4R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV4R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV4R1(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmBwdDataV4R1(bool spare);

    bool operator==(const PerformanceImplicitGemmBwdDataV4R1& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    std::string ToString() const;
};

struct PerformanceImplicitGemmBwdDataV4R1Xdlops
    : Serializable<PerformanceImplicitGemmBwdDataV4R1Xdlops>
{
    int GemmNPerBlock; // 2^n[8..16]
    int GemmMPerBlock; // 2^n[32..128]
    int GemmKPerBlock; // 2^n[4..16]

    int GemmKPACKSize; // 2^[1..4]

    int GemmMPerWave;
    int GemmNPerWave;

    // GemmAThreadCopyMoreGemmK is currently a fix value, is untunable
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    bool use_spare_set;
    PerformanceImplicitGemmBwdDataV4R1Xdlops(int, int, int, int, int, int, bool, bool, bool);

    PerformanceImplicitGemmBwdDataV4R1Xdlops();
    PerformanceImplicitGemmBwdDataV4R1Xdlops(bool spare);
    PerformanceImplicitGemmBwdDataV4R1Xdlops(
        int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceImplicitGemmBwdDataV4R1Xdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    bool operator==(const PerformanceImplicitGemmBwdDataV4R1Xdlops& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmKPACKSize, "GemmKPACKSize");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    bool IsReallyValid(const ConvolutionContext& ctx) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;
    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    std::string ToString() const;
};

struct ConvHipImplicitGemmV4R1Fwd : SolverBase<ConvolutionContext>
{
    PerformanceImplicitGemmV4R1 GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmV4R1& c) const;

    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmV4R1& config,
                             bool disableConfigOverrideFromEnv = false) const;

    PerformanceImplicitGemmV4R1 Search(const ConvolutionContext&,
                                       const AnyInvokeParams& invoke_ctx) const;
};

struct ConvHipImplicitGemmV4R4Fwd : SolverBase<ConvolutionContext>
{
    static std::tuple<int, int, int> CalculateGemmSize(const ConvolutionContext& ctx);
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceImplicitGemmV4R4Fwd GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmV4R4Fwd& config) const;
    PerformanceImplicitGemmV4R4Fwd Search(const ConvolutionContext&,
                                          const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmV4R4Fwd& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConvMlirIgemm : Serializable<PerformanceConvMlirIgemm>
{
    int BlockSize;
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerThread;
    int GemmNPerThread;
    bool use_spare_set;

    /// \ref https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1154
    static const PerformanceConvMlirIgemm& MlirHeuristicInitRequest();

    PerformanceConvMlirIgemm(int, int, int, int, int, int, bool);

    PerformanceConvMlirIgemm(int a, int b, int c, int d, int e, int f)
        : PerformanceConvMlirIgemm(a, b, c, d, e, f, false)
    {
    }

    PerformanceConvMlirIgemm() : PerformanceConvMlirIgemm(-1, -1, -1, -1, -1, -1, false) {}

    PerformanceConvMlirIgemm(bool spare);

    bool operator==(const PerformanceConvMlirIgemm& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    bool IsValid(const ConvolutionContext& ctx) const;
    bool SetNextValue(const ConvolutionContext& config);
    std::string ToString() const;
};

struct ConvMlirIgemmFwd : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceConvMlirIgemm GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceConvMlirIgemm& config) const;
    PerformanceConvMlirIgemm Search(const ConvolutionContext&,
                                    const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConvMlirIgemm& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConvMlirIgemmXdlops : Serializable<PerformanceConvMlirIgemmXdlops>
{
    int GemmMPerBlock; // 2^n[32..128]
    int GemmNPerBlock; // 2^n[8..16]
    int GemmKPerBlock; // 2^n[4..16]
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPACKSize; // 2^[1..4]

    // GemmAThreadCopyMoreGemmK is currently a fix value, is untunable
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    bool use_spare_set;

    /// \ref https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1154
    static const PerformanceConvMlirIgemmXdlops& MlirHeuristicInitRequest();

    PerformanceConvMlirIgemmXdlops(int, int, int, int, int, int, bool, bool, bool);

    PerformanceConvMlirIgemmXdlops();
    PerformanceConvMlirIgemmXdlops(bool spare);
    PerformanceConvMlirIgemmXdlops(int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceConvMlirIgemmXdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    bool operator==(const PerformanceConvMlirIgemmXdlops& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPACKSize, "GemmKPACKSize");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    bool IsValid(const ConvolutionContext& ctx) const;
    bool SetNextValue(const ConvolutionContext& config);
    std::string ToString() const;
};

struct ConvMlirIgemmFwdXdlops : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
    PerformanceConvMlirIgemmXdlops GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceConvMlirIgemmXdlops& config) const;
    PerformanceConvMlirIgemmXdlops Search(const ConvolutionContext&,
                                          const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConvMlirIgemmXdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceImplicitGemmV4R4GenXdlopsFwdFp32
    : Serializable<PerformanceImplicitGemmV4R4GenXdlopsFwdFp32>
{
    int GemmMPerBlock; // 2^n[32..128]
    int GemmNPerBlock; // 2^n[8..16]
    int GemmKPerBlock; // 2^n[4..16]

    int GemmMPerWave; // [4, 16, 32, 64]
    int GemmNPerWave; // [4, 16, 32, 64]

    bool use_spare_set;

    PerformanceImplicitGemmV4R4GenXdlopsFwdFp32(int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4GenXdlopsFwdFp32()
        : PerformanceImplicitGemmV4R4GenXdlopsFwdFp32(-1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4GenXdlopsFwdFp32(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
    }

    void HeuristicInit(const ConvolutionContext& ctx);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& ctx) const;
    bool operator==(const PerformanceImplicitGemmV4R4GenXdlopsFwdFp32& other) const;
    std::string ToString() const;

    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct ConvHipImplicitGemmV4R4WrW : SolverBase<ConvolutionContext>
{
    static std::tuple<int, int, int> CalculateGemmSize(const ConvolutionContext& ctx);
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceImplicitGemmV4R4WrW GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmV4R4WrW& config) const;
    PerformanceImplicitGemmV4R4WrW Search(const ConvolutionContext&,
                                          const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmV4R4WrW& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvMlirIgemmWrW : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceConvMlirIgemm GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceConvMlirIgemm& config) const;
    PerformanceConvMlirIgemm Search(const ConvolutionContext&,
                                    const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConvMlirIgemm& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvMlirIgemmWrWXdlops : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceConvMlirIgemmXdlops GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceConvMlirIgemmXdlops& config) const;
    PerformanceConvMlirIgemmXdlops Search(const ConvolutionContext&,
                                          const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConvMlirIgemmXdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceImplicitGemmXdlops : Serializable<PerformanceImplicitGemmXdlops>
{
    int BPerBlock; // 2^n[8..16]
    int KPerBlock; // 2^n[32..128]
    int EPerBlock; // 2^n[4..16]
    int EBlocks;   // 2*n[1..64]
    int EPACKSize; // 2*n[1..4] // 1 - fp32; 2,4 - bfp16; 4 - fp16

    int GemmMPerWave;
    int GemmNPerWave;

    int InBlockCopyClusterLengths_E; // 2^n[4..16]
    int InBlockCopyClusterLengths_B; // 2^n[8..16]

    int WeiBlockCopyClusterLengths_E; // 2^n[1..4]
    int WeiBlockCopyClusterLengths_K; // 2^n[16..128]

    bool use_spare_set;

    PerformanceImplicitGemmXdlops(int, int, int, int, int, int, int, int, int, int, int, bool);

    PerformanceImplicitGemmXdlops()
        : PerformanceImplicitGemmXdlops(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmXdlops(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BPerBlock, "BPerBlock");
        f(self.KPerBlock, "KPerBlock");
        f(self.EPerBlock, "EPerBlock");
        f(self.EBlocks, "EBlocks");
        f(self.EPACKSize, "EPACKSize");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.InBlockCopyClusterLengths_E, "InBlockCopyClusterLengths_E");
        f(self.InBlockCopyClusterLengths_B, "InBlockCopyClusterLengths_B");
        f(self.WeiBlockCopyClusterLengths_E, "WeiBlockCopyClusterLengths_E");
        f(self.WeiBlockCopyClusterLengths_K, "WeiBlockCopyClusterLengths_K");
    }

    void HeuristicInit(const ConvolutionContext& ctx);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& ctx) const;
    bool operator==(const PerformanceImplicitGemmXdlops& other) const;
    std::string ToString() const;
};

struct PerformanceImplicitGemmForwardV4R4Xdlops
    : Serializable<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    PerformanceImplicitGemmForwardV4R4Xdlops(int, int, int, int, int, int, bool, bool, int);
    PerformanceImplicitGemmForwardV4R4Xdlops();
    PerformanceImplicitGemmForwardV4R4Xdlops(bool) : PerformanceImplicitGemmForwardV4R4Xdlops() {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    bool operator==(const PerformanceImplicitGemmForwardV4R4Xdlops& other) const;
    std::string ToString() const;

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    bool IsReallyValid(const ConvolutionContext& ctx) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct PerformanceImplicitGemmForwardV4R5Xdlops
    : Serializable<PerformanceImplicitGemmForwardV4R5Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    bool use_spare_set;

    PerformanceImplicitGemmForwardV4R5Xdlops(int, int, int, int, int, int, bool, bool, int, bool);
    PerformanceImplicitGemmForwardV4R5Xdlops();
    PerformanceImplicitGemmForwardV4R5Xdlops(bool spare);

    PerformanceImplicitGemmForwardV4R5Xdlops(
        int a, int b, int c, int d, int e, int f, bool g, bool h, int i)
        : PerformanceImplicitGemmForwardV4R5Xdlops(a, b, c, d, e, f, g, h, i, false)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    bool operator==(const PerformanceImplicitGemmForwardV4R5Xdlops& other) const;
    std::string ToString() const;

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    bool IsReallyValid(const ConvolutionContext& ctx) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    : Serializable<PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    int GemmMFactor;
    int GemmNFactor;
    int GemmKFactor;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool, int);
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm();
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm(bool)
        : PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm()
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmMFactor, "GemmMFactor");
        f(self.GemmNFactor, "GemmNFactor");
        f(self.GemmKFactor, "GemmKFactor");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    bool operator==(const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm& other) const;
    std::string ToString() const;

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    bool IsReallyValid(const ConvolutionContext& ctx) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct PerformanceImplicitGemmBwdV1R1Xdlops : Serializable<PerformanceImplicitGemmBwdV1R1Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    PerformanceImplicitGemmBwdV1R1Xdlops(int, int, int, int, int, int, bool, bool);
    PerformanceImplicitGemmBwdV1R1Xdlops();
    PerformanceImplicitGemmBwdV1R1Xdlops(bool) : PerformanceImplicitGemmBwdV1R1Xdlops() {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    bool operator==(const PerformanceImplicitGemmBwdV1R1Xdlops& other) const;
    std::string ToString() const;

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    bool IsReallyValid(const ConvolutionContext& ctx) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct ConvHipImplicitGemmForwardV4R4Xdlops : SolverBase<ConvolutionContext>
{
    static std::tuple<int, int, int, int> CalculateGemmSize(const ConvolutionContext& ctx);
    PerformanceImplicitGemmForwardV4R4Xdlops
    GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmForwardV4R4Xdlops& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmForwardV4R4Xdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;

    PerformanceImplicitGemmForwardV4R4Xdlops Search(const ConvolutionContext&,
                                                    const AnyInvokeParams& invoke_ctx) const;
};

struct ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm : SolverBase<ConvolutionContext>
{
    static std::tuple<int, int, int, int, int, int, int> CalculateGemmSize(
        const ConvolutionContext& ctx, int GemmMFactor, int GemmNFactor, int GemmKFactor);
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool
    IsValidPerformanceConfig(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm& config,
                             bool disableConfigOverrideFromEnv = false) const;

    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    Search(const ConvolutionContext&, const AnyInvokeParams& invoke_ctx) const;
};

struct ConvHipImplicitGemmForwardV4R5Xdlops : SolverBase<ConvolutionContext>
{
    PerformanceImplicitGemmForwardV4R5Xdlops
    GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmForwardV4R5Xdlops& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmForwardV4R5Xdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;
    PerformanceImplicitGemmForwardV4R5Xdlops Search(const ConvolutionContext&,
                                                    const AnyInvokeParams& invoke_ctx) const;
};

struct PerformanceImplicitGemmV4R4GenXdlopsWrWFp32
    : Serializable<PerformanceImplicitGemmV4R4GenXdlopsWrWFp32>
{
    int GemmMPerBlock; // 2^n[32..128]
    int GemmNPerBlock; // 2^n[8..16]
    int GemmKPerBlock; // 2^n[4..16]
    int GemmKBlocks;   // 2^n[1..64]

    int GemmMPerWave; // [4, 16, 32, 64]
    int GemmNPerWave; // [4, 16, 32, 64]

    bool use_spare_set;

    PerformanceImplicitGemmV4R4GenXdlopsWrWFp32(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4GenXdlopsWrWFp32()
        : PerformanceImplicitGemmV4R4GenXdlopsWrWFp32(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4GenXdlopsWrWFp32(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmKBlocks, "GemmKBlocks");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
    }

    void HeuristicInit(const ConvolutionContext& ctx);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& ctx) const;
    bool operator==(const PerformanceImplicitGemmV4R4GenXdlopsWrWFp32& other) const;
    std::string ToString() const;

    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct ConvHipImplicitGemmV4R1WrW : SolverBase<ConvolutionContext>
{
    PerformanceImplicitGemmV4R1 GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmV4R1& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmV4R1& config,
                             bool disableConfigOverrideFromEnv = false) const;

    PerformanceImplicitGemmV4R1 Search(const ConvolutionContext&,
                                       const AnyInvokeParams& invoke_ctx) const;
};

struct ConvHipImplicitGemmBwdDataV1R1 : SolverBase<ConvolutionContext>
{
    static std::tuple<int, int, int> CalculateGemmSize(const ConvolutionContext& ctx);
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceImplicitGemmBwdDataV1R1 GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmBwdDataV1R1& config) const;
    PerformanceImplicitGemmBwdDataV1R1 Search(const ConvolutionContext&,
                                              const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmBwdDataV1R1& config,
                             bool disableConfigOverrideFromEnv = false) const;
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;
};

struct ConvMlirIgemmBwd : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceConvMlirIgemm GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceConvMlirIgemm& config) const;
    PerformanceConvMlirIgemm Search(const ConvolutionContext&,
                                    const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConvMlirIgemm& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvMlirIgemmBwdXdlops : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceConvMlirIgemmXdlops GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceConvMlirIgemmXdlops& config) const;
    PerformanceConvMlirIgemmXdlops Search(const ConvolutionContext&,
                                          const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConvMlirIgemmXdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvHipImplicitGemmBwdDataV4R1 : SolverBase<ConvolutionContext>
{
    static int CalculateNumberOfGemm(const ConvolutionContext& ctx);
    static std::tuple<int, int, int> CalculateGemmSize(const ConvolutionContext& ctx, int gemm_id);
    bool IsApplicable(const ConvolutionContext& ctx) const;
    PerformanceImplicitGemmBwdDataV4R1 GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmBwdDataV4R1& config) const;
    PerformanceImplicitGemmBwdDataV4R1 Search(const ConvolutionContext&,
                                              const AnyInvokeParams& invoke_ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmBwdDataV4R1& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvHipImplicitGemmBwdDataV4R1Xdlops : SolverBase<ConvolutionContext>
{
    static int CalculateNumberOfGemm(const ConvolutionContext& ctx);
    static std::tuple<int, int, int, int> CalculateGemmSize(const ConvolutionContext& ctx,
                                                            int gemm_id);
    PerformanceImplicitGemmBwdDataV4R1Xdlops
    GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmBwdDataV4R1Xdlops& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmBwdDataV4R1Xdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;
    PerformanceImplicitGemmBwdDataV4R1Xdlops Search(const ConvolutionContext&,
                                                    const AnyInvokeParams& invoke_ctx) const;
};

struct ConvHipImplicitGemmBwdDataV1R1Xdlops : SolverBase<ConvolutionContext>
{
    static std::tuple<int, int, int, int> CalculateGemmSize(const ConvolutionContext& ctx);
    PerformanceImplicitGemmBwdV1R1Xdlops GetPerformanceConfig(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmBwdV1R1Xdlops& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmBwdV1R1Xdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;
    PerformanceImplicitGemmBwdV1R1Xdlops Search(const ConvolutionContext& ctx,
                                                const AnyInvokeParams& invoke_ctx) const;
};

struct ConvAsmImplicitGemmV4R1DynamicFwd : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct ConvAsmImplicitGemmV4R1DynamicFwd_1x1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct ConvAsmImplicitGemmV4R1DynamicWrw : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct ConvAsmImplicitGemmGTCDynamicWrwXdlops : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct ConvAsmImplicitGemmV4R1DynamicBwd : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext&) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext&) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdXdlops : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct ConvAsmImplicitGemmGTCDynamicBwdXdlops : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
struct ConvOclDirectFwdLegacyExhaustiveSearch : SolverBase<ConvolutionContext>
{
    LegacyPerformanceConfig GetPerformanceConfig(const ConvolutionContext&) const;
    LegacyPerformanceConfig Search(const ConvolutionContext&,
                                   const AnyInvokeParams& invoke_ctx) const;

    private:
    template <typename Tgpu>
    LegacyPerformanceConfig SearchImpl(const ConvolutionContext&) const;
};

struct ConvOclDirectFwd : ConvOclDirectFwdLegacyExhaustiveSearch
{
    bool IsApplicable(const ConvolutionContext& params) const;

    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& searched_params) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const LegacyPerformanceConfig&) const;

    protected:
    bool IsApplicableBase(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwdFused : ConvOclDirectFwd
{
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& searched_params) const;
};

struct ConvOclDirectFwd1x1 : ConvOclDirectFwdLegacyExhaustiveSearch
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& searched_params) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const LegacyPerformanceConfig&) const
    {
        return true;
    }
};

struct ConvBinWinograd3x3U : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvBinWinogradRxS : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct PerformanceConfigConvBinWinogradRxSf3x2
    : Serializable<PerformanceConfigConvBinWinogradRxSf3x2>
{
    int n_groups;
    PerformanceConfigConvBinWinogradRxSf3x2(int n_groups_);
    PerformanceConfigConvBinWinogradRxSf3x2() : PerformanceConfigConvBinWinogradRxSf3x2(-1) {}
    PerformanceConfigConvBinWinogradRxSf3x2(bool) : PerformanceConfigConvBinWinogradRxSf3x2(1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_groups, "n_groups");
    }
    int GetNGroups() const { return n_groups; }

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvBinWinogradRxSf3x2& other) const;
    std::string ToString() const;
};

struct ConvBinWinogradRxSf3x2 : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvBinWinogradRxSf3x2 GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvBinWinogradRxSf3x2&) const;
    PerformanceConfigConvBinWinogradRxSf3x2 Search(const ConvolutionContext&,
                                                   const AnyInvokeParams& invoke_ctx) const;

    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvBinWinogradRxSf3x2& config,
                             bool disableConfigOverrideFromEnv = false) const;
    static size_t GetNGroups(const size_t group_conv, const size_t grid_group_size)
    {
        assert(group_conv != 0);
        return grid_group_size / group_conv;
    }
};

struct PerformanceConfigConvBinWinogradRxSf2x3
    : Serializable<PerformanceConfigConvBinWinogradRxSf2x3>
{
    int n_groups;
    PerformanceConfigConvBinWinogradRxSf2x3(int n_groups_);
    PerformanceConfigConvBinWinogradRxSf2x3() : PerformanceConfigConvBinWinogradRxSf2x3(-1) {}
    PerformanceConfigConvBinWinogradRxSf2x3(bool) : PerformanceConfigConvBinWinogradRxSf2x3(1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_groups, "n_groups");
    }
    int GetNGroups() const { return n_groups; }

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvBinWinogradRxSf2x3& other) const;
    std::string ToString() const;
};

struct ConvBinWinogradRxSf2x3 : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvBinWinogradRxSf2x3 GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvBinWinogradRxSf2x3&) const;
    PerformanceConfigConvBinWinogradRxSf2x3 Search(const ConvolutionContext&,
                                                   const AnyInvokeParams& invoke_ctx) const;

    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsDynamic() const { return true; }
    float GetWti(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvBinWinogradRxSf2x3& config,
                             bool disableConfigOverrideFromEnv = false) const;
    static size_t GetNGroups(const size_t group_conv, const size_t grid_group_size)
    {
        assert(group_conv != 0);
        return grid_group_size / group_conv;
    }
};

struct ConvBinWinogradRxSf2x3g1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsDynamic() const { return true; }
    float GetWti(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvBinWinogradRxSf2x3g1Fused : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvBinWinogradRxSFused : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvMPBidirectWinograd : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsDynamic() const { return true; }
    size_t GetWorkspaceSize(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;

    // kernel_file_name for solver identification
    static std::string GetSolverFileNames(int id)
    {
        static const std::string names[3] = {"xform_bidirect_winograd_data.s",
                                             "xform_bidirect_winograd_filter.s",
                                             "xform_bidirect_winograd_out.s"};
        return names[id];
    }

    static std::string GetSolverKernelNames(int id)
    {
        static const std::string name_suffix =
            '_' + std::to_string(WinoDataH) + '_' + std::to_string(WinoDataW) + '_' +
            std::to_string(WinoFilterH) + '_' + std::to_string(WinoFilterW);
        static const std::string names[3] = {
            "miopenGcnAsmMPBidirectWinogradXformData" + name_suffix,
            "miopenGcnAsmMPBidirectWinogradXformFilter" + name_suffix,
            "miopenGcnAsmMPBidirectWinogradXformOut" + name_suffix};
        return names[id];
    }

    static int GetSolverWinoXformHWSize() { return WinoDataH + WinoFilterH - 1; }
};
extern template struct ConvMPBidirectWinograd<2, 3>;
extern template struct ConvMPBidirectWinograd<3, 3>;
extern template struct ConvMPBidirectWinograd<4, 3>;
extern template struct ConvMPBidirectWinograd<5, 3>;
extern template struct ConvMPBidirectWinograd<6, 3>;

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvMPBidirectWinograd_xdlops : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;

    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmForwardV4R4Xdlops& c) const
    {
        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsValidPerformanceConfig(
            GetTransformedConvContext(ctx), c);
    }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>()
                   .GetWorkspaceSize(ctx) +
               ConvHipImplicitGemmForwardV4R4Xdlops{}.GetWorkspaceSize(
                   GetTransformedConvContext(ctx));
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmForwardV4R4Xdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;

    ConvolutionContext GetTransformedConvContext(const ConvolutionContext&) const;

    // kernel_file_name for solver identification
    static std::string GetSolverFileNames(int id)
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverFileNames(id);
    }

    static std::string GetSolverKernelNames(int id)
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverKernelNames(id);
    }

    static int GetSolverWinoXformHWSize()
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverWinoXformHWSize();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops
    GetPerformanceConfig(const ConvolutionContext& ctx) const
    {
        return ConvHipImplicitGemmForwardV4R4Xdlops{}.GetPerformanceConfig(
            GetTransformedConvContext(ctx));
    }
    bool IsThisSolverDynamic() const { return true; }

    bool IsDynamic() const
    {
        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsDynamic() &&
               ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}
                   .IsDynamic() &&
               IsThisSolverDynamic();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops Search(const ConvolutionContext&,
                                                    const AnyInvokeParams&) const;
};

extern template struct ConvMPBidirectWinograd_xdlops<2, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<3, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<4, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<5, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<6, 3>;

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvWinograd3x3MultipassWrW : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsDynamic() const { return true; }
    size_t GetWorkspaceSize(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;

    // kernel_file_name for solver identification
    static std::string GetSolverFileNames(int id)
    {
        static const std::string names[3] = {"xform_data.s", "xform_filter.s", "xform_out.s"};
        return names[id];
    }
    static std::string GetSolverKernelNames(int id)
    {
        static const std::string name_suffix =
            '_' + std::to_string(WinoDataH) + '_' + std::to_string(WinoDataW) + '_' +
            std::to_string(WinoFilterH) + '_' + std::to_string(WinoFilterW);
        static const std::string names[3] = {"miopenGcnAsmWinogradXformData" + name_suffix,
                                             "miopenGcnAsmWinogradXformFilter" + name_suffix,
                                             "miopenGcnAsmWinogradXformOut" + name_suffix};

        return names[id];
    }
    static int GetGroupCountMult() { return 4; }

    static int GetSolverWinoXformHWSize(const miopen::ConvolutionContext& ctx, int id)
    {
        if(id == 0)
            return WinoDataH + (WinoFilterH - 1) * (WinoDataH == 7 ? 2 : ctx.kernel_stride_h);
        else
            return WinoDataW + (WinoFilterW - 1) * (WinoDataW == 7 ? 2 : ctx.kernel_stride_w);
    }

    private:
    InvokerFactory PrepareInvokerFactory(const ConvolutionContext& params, std::size_t ws_sz) const;
};

extern template struct ConvWinograd3x3MultipassWrW<3, 2>;
extern template struct ConvWinograd3x3MultipassWrW<3, 3>;
extern template struct ConvWinograd3x3MultipassWrW<3, 4>;
extern template struct ConvWinograd3x3MultipassWrW<3, 5>;
extern template struct ConvWinograd3x3MultipassWrW<3, 6>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2, 1, 1>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3, 1, 1>;
extern template struct ConvWinograd3x3MultipassWrW<5, 3>;
extern template struct ConvWinograd3x3MultipassWrW<5, 4>;

struct PerformanceConfigAsmDirect3x3WrW : Serializable<PerformanceConfigAsmDirect3x3WrW>
{
    int limit_wave_cnt;   // [0..9]
    int reverse_inout;    // [0..1], 1 is allowed for stride=1x1 only.
    int chunk_size;       // {16,8}, Smaller values increase register pressure.
    int k_per_wave;       // {1,2,4,8} && ((chunk_size * k_per_wave) <= 64).
                          // Higher values increase register pressure.
    int pipe_lines_depth; // [1..16] && (pipe_lines_depth <= img_h).
                          // Higher values increase register pressure.
    int n_per_group;      // [1..8] && (n_per_group <= batch_size).

    PerformanceConfigAsmDirect3x3WrW(int lwc, int rio, int csz, int kpw, int pld, int npg);
    PerformanceConfigAsmDirect3x3WrW() : PerformanceConfigAsmDirect3x3WrW(-1, -1, -1, -1, -1, -1) {}
    PerformanceConfigAsmDirect3x3WrW(bool) : PerformanceConfigAsmDirect3x3WrW(0, 0, 8, 1, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.reverse_inout, "reverse_inout");
        f(self.chunk_size, "chunk_size");
        f(self.k_per_wave, "k_per_wave");
        f(self.pipe_lines_depth, "pipe_lines_depth");
        f(self.n_per_group, "n_per_group");
    }

    // clang-format off
    int GetLimitWaveCnt() const { return limit_wave_cnt; }
    int GetReverseInout() const { return reverse_inout; }
    int GetChunkSize() const { return chunk_size; }
    int GetKPerWave() const { return k_per_wave; }
    int GetPipeLinesDepth() const { return pipe_lines_depth; }
    int GetNPerGroup() const { return n_per_group; }
    int GetCPerWave() const { assert(chunk_size); return 64 / chunk_size; } // clang-format on

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigAsmDirect3x3WrW& other) const;
    std::string ToString() const;
};

struct ConvAsmBwdWrW3x3 : SolverBase<ConvolutionContext>
{
    PerformanceConfigAsmDirect3x3WrW GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigAsmDirect3x3WrW&) const;
    PerformanceConfigAsmDirect3x3WrW Search(const ConvolutionContext&,
                                            const AnyInvokeParams& invoke_ctx) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigAsmDirect3x3WrW& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConfigConvAsmBwdWrW1x1 : Serializable<PerformanceConfigConvAsmBwdWrW1x1>
{

    int chunk_size;    // {1,2,4,8,16}
    int c_per_gpr;     // {1,2,4,8,16}
    int c_mult;        // {1,2,4,8,16}
    int k_per_gpr;     // {1,2,4,8,16}
    int k_mult;        // {1,2,4,8,16}
    int n_per_gpr;     // {1,2,4}
    int n_part_cnt;    // [1..8]
    int read_size;     // [1..4]
    int short_store;   // {0,1}
    int data_prefetch; // [0..4]
    bool use_spare_set;

    /// The following conditions must be met.
    ///
    /// Shader design-related constraints:
    /// - (A) (chunk_size * c_per_gpr) == 16
    /// - (B) k_per_gpr <= c_per_gpr
    /// - (C) (c_mult > 1 || k_mult > 1)
    ///         ? ((fwd_C % (c_per_gpr * c_mult) == 0) && (fwd_K % (k_per_gpr * k_mult) == 0))
    ///         : (true)
    ///
    /// Resource-related constraints:
    /// - (D) c_mult * k_mult * k_per_gpr + 9 + (c_mult + k_mult) * read_size * pipe_depth <= 256
    ///
    /// Where:
    /// - fwd_C := Num input channels for forward convolution (-c).
    ///   For backward, this is actually n_outputs.
    /// - fwd_K := Num output channels for forward convolution (-k).
    ///   For backward, this is actually n_inputs.

    PerformanceConfigConvAsmBwdWrW1x1(int chunk_size_,
                                      int c_per_gpr_,
                                      int c_mult_,
                                      int k_per_gpr_,
                                      int k_mult_,
                                      int n_per_gpr_,
                                      int n_part_cnt_,
                                      int read_size_,
                                      int short_store_,
                                      int data_prefetch_,
                                      bool);
    PerformanceConfigConvAsmBwdWrW1x1()
        : PerformanceConfigConvAsmBwdWrW1x1(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsmBwdWrW1x1(bool spare)
        : PerformanceConfigConvAsmBwdWrW1x1(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.c_per_gpr, "c_per_gpr");
        f(self.c_mult, "c_mult");
        f(self.k_per_gpr, "k_per_gpr");
        f(self.k_mult, "k_mult");
        f(self.n_per_gpr, "n_per_gpr");
        f(self.n_part_cnt, "n_part_cnt");
        f(self.read_size, "read_size");
        f(self.short_store, "short_store");
        f(self.data_prefetch, "data_prefetch");
    }

    // clang-format off
    int GetChunkSize() const { return chunk_size; }
    int GetCPerGpr() const { return c_per_gpr; }
    int GetCMult() const { return c_mult; }
    int GetKPerGpr() const { return k_per_gpr; }
    int GetKMult() const { return k_mult; }
    int GetNPerGpr() const { return n_per_gpr; }
    int GetNPartCnt() const { return n_part_cnt; }
    int GetHWPerGpr() const {   assert(c_per_gpr); assert(n_per_gpr); assert(chunk_size);
                                return wave_size / (c_per_gpr * n_per_gpr * chunk_size); } // "hw" stands for "height-and-width".
    int GetReadSize() const { return read_size; }
    int GetShortStore() const {return short_store; }
    int GetDataPrefetch() const { return data_prefetch; }
    // clang-format on

    void HeuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvAsmBwdWrW1x1& other) const;
    std::string ToString() const;
};

struct ConvAsmBwdWrW1x1 : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvAsmBwdWrW1x1 GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvAsmBwdWrW1x1&) const;
    PerformanceConfigConvAsmBwdWrW1x1 Search(const ConvolutionContext&,
                                             const AnyInvokeParams& invoke_ctx) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsmBwdWrW1x1& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

/// N_BATCH_LOOPS - {1,2,4,8,16} Num batches processed in single workitem.
///     Required workspace size depends on it. However there is a restriction in the internal
///     Solver API that this shouldn't be so. Therefore the family of Solvers created.
///     Each Solver in the family has constant value of this parameter.
template <int N_BATCH_LOOPS>
struct PerformanceConfigConvOclBwdWrw2
    : Serializable<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>
{
    // Num waves involved a workgroup.
    int n_waves = -1; // {1,2,4,8}
    // Num values to read in a workitem (read_unit).
    int read_size = -1; // [6..12]
    // Num of output channels (top/bottom layer in forward/backward direction)
    // that share the same input channel in single workgroup.
    // Also represents number of output channels in single tile.
    int n_out_channels_per_tile = -1; // {1,2,4,8}
    // How many tiles of output channels are processed in a single workgroup?
    // n_out_channels_in_lcl * n_out_channels_tiles = total number of
    // output channels processed in single workgroup.
    int n_out_channels_tiles = -1; // {1,2,4,8}
    // Num of output rows processed in a single iteration of loop in a workitem
    // (N_ALIGNED_OUT_SCAN_BLK).
    int n_out_rows_in_lcl = -1; // [2..11]

    PerformanceConfigConvOclBwdWrw2(int nw, int rs, int nocpt, int noct, int noril)
        : n_waves(nw),
          read_size(rs),
          n_out_channels_per_tile(nocpt),
          n_out_channels_tiles(noct),
          n_out_rows_in_lcl(noril)
    {
    }
    PerformanceConfigConvOclBwdWrw2() {}
    PerformanceConfigConvOclBwdWrw2(bool) : PerformanceConfigConvOclBwdWrw2(1, 6, 1, 1, 2) {}
    // spare_set is not used in this solver.

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_waves, "n_waves");
        f(self.read_size, "read_size");
        f(self.n_out_channels_per_tile, "n_out_channels_per_tile");
        f(self.n_out_channels_tiles, "n_out_channels_tiles");
        f(self.n_out_rows_in_lcl, "n_out_rows_in_lcl");
    }

    // clang-format off
    int GetNumWaves() const { return n_waves; }
    int GetReadSize() const { return read_size; }
    int GetNumOutChannelsPerTile() const { return n_out_channels_per_tile; }
    int GetNumOutChannelTiles() const { return n_out_channels_tiles; }
    int GetNumOutRowsPerIterPerWork() const { return n_out_rows_in_lcl; } // clang-format on

    void HeuristicInit(const ConvolutionContext& params);
    bool IsValidValue() const;
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValid(const ConvolutionContext& params) const;
    bool operator==(const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& other) const;
    std::string ToString() const;
};

template <int N_BATCH_LOOPS>
struct ConvOclBwdWrW2 : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
    GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&) const;
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS> Search(const ConvolutionContext&,
                                                          const AnyInvokeParams& invoke_ctx) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config,
                             bool disableConfigOverrideFromEnv = false) const;

    protected:
    bool IsApplicableBase(const ConvolutionContext& params) const;
};

extern template struct ConvOclBwdWrW2<1>;
extern template struct ConvOclBwdWrW2<2>;
extern template struct ConvOclBwdWrW2<4>;
extern template struct ConvOclBwdWrW2<8>;
extern template struct ConvOclBwdWrW2<16>;

/// A separate solver from ConvOclBwdWrW2 to disable auto-tuning for certain configs.
/// Basically, this is *hack* for non-group 3x3 and 1x1 cases.
/// It is assumed that Solutions provided by the ConvOclBwdWrW2 solver
/// would never beat 3x3 and 1x1 assembly WrW kernels, even after tuning.
struct ConvOclBwdWrW2NonTunable : ConvOclBwdWrW2<1>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;

    private:
    // This function dervied from ConvOclBwdWrW2 is declared private
    // so that this solver is not marked searchable/tunable.
    template <int N_BATCH_LOOPS>
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvOclBwdWrW53 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclBwdWrW1x1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const;
};

struct fft : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct PerformanceImplicitGemmWrwV4R4Xdlops : Serializable<PerformanceImplicitGemmWrwV4R4Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmK;
    bool use_spare_set;

    PerformanceImplicitGemmWrwV4R4Xdlops(int, int, int, int, int, int, bool, bool, bool);
    PerformanceImplicitGemmWrwV4R4Xdlops();
    PerformanceImplicitGemmWrwV4R4Xdlops(bool spare);
    PerformanceImplicitGemmWrwV4R4Xdlops(int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceImplicitGemmWrwV4R4Xdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmK, "GemmBThreadCopyMoreGemmK");
    }

    bool operator==(const PerformanceImplicitGemmWrwV4R4Xdlops& other) const;
    std::string ToString() const;

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    bool IsReallyValid(const ConvolutionContext& ctx) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;

    std::tuple<int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ConvolutionContext& ctx) const;
    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct ConvHipImplicitGemmWrwV4R4Xdlops : SolverBase<ConvolutionContext>
{
    PerformanceImplicitGemmWrwV4R4Xdlops GetPerformanceConfig(const ConvolutionContext& ctx) const;
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmWrwV4R4Xdlops& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmWrwV4R4Xdlops& config,
                             bool disableConfigOverrideFromEnv = false) const;

    PerformanceImplicitGemmWrwV4R4Xdlops Search(const ConvolutionContext&,
                                                const AnyInvokeParams& invoke_ctx) const;
};

struct PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    : Serializable<PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    int GemmMFactor;
    int GemmNFactor;
    int GemmKTotalFactor;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmK;

    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool);
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm();
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm(bool)
        : PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm()
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmMFactor, "GemmMFactor");
        f(self.GemmNFactor, "GemmNFactor");
        f(self.GemmKTotalFactor, "GemmKTotalFactor");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmK, "GemmBThreadCopyMoreGemmK");
    }

    bool operator==(const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm& other) const;
    std::string ToString() const;

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    bool IsReallyValid(const ConvolutionContext& ctx) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;
    int CalculateGemmKBlocks(const ConvolutionContext& ctx) const;

    std::tuple<int, int, int, int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ConvolutionContext& ctx) const;
    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext& ctx) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const;
};

struct ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm : SolverBase<ConvolutionContext>
{
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    GetPerformanceConfig(const ConvolutionContext& ctx) const;
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm& c) const;
    bool IsApplicable(const ConvolutionContext& ctx) const;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm& config,
                             bool disableConfigOverrideFromEnv = false) const;

    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    Search(const ConvolutionContext&, const AnyInvokeParams& invoke_ctx) const;
};

struct PerformanceConvCkIgemmFwdV6r1DlopsNchw : Serializable<PerformanceConvCkIgemmFwdV6r1DlopsNchw>
{
    int ck_tunable_list_id;

    PerformanceConvCkIgemmFwdV6r1DlopsNchw(int a) : ck_tunable_list_id(a) {}

    PerformanceConvCkIgemmFwdV6r1DlopsNchw() : PerformanceConvCkIgemmFwdV6r1DlopsNchw(-1) {}

    PerformanceConvCkIgemmFwdV6r1DlopsNchw(bool) : PerformanceConvCkIgemmFwdV6r1DlopsNchw(0) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.ck_tunable_list_id, "ck_tunable_list_id");
    }

    bool SetNextValue(const ConvolutionContext&);
    bool IsValid(const ConvolutionContext&) const;
    bool operator==(const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config) const
    {
        return ck_tunable_list_id == config.ck_tunable_list_id;
    }
};

struct ConvCkIgemmFwdV6r1DlopsNchw : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext&) const;
    std::size_t GetWorkspaceSize(const ConvolutionContext&) const;
    bool IsDynamic() const { return true; }
    PerformanceConvCkIgemmFwdV6r1DlopsNchw GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const;
    PerformanceConvCkIgemmFwdV6r1DlopsNchw Search(const ConvolutionContext&,
                                                  const AnyInvokeParams&) const;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const PerformanceConvCkIgemmFwdV6r1DlopsNchw&,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvDirectNaiveConvFwd : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled due to MIOpenGemm or OCL compiler issues.
    float GetWti(const ConvolutionContext&) const { return 0.01; }
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct ConvDirectNaiveConvBwd : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled due to MIOpenGemm or OCL compiler issues.
    float GetWti(const ConvolutionContext&) const { return 0.01; }
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct ConvDirectNaiveConvWrw : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled due to MIOpenGemm or OCL compiler issues.
    float GetWti(const ConvolutionContext&) const { return 0.01; }
    ConvSolution GetSolution(const ConvolutionContext& ctx) const;
};

struct GemmFwdBase : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsDynamic() const { return true; }
    float GetWti(const ConvolutionContext& ctx) const { return GetWti(ctx, ctx.conv_problem); }
    float GetWti(const ExecutionContext& context, const conv::ProblemDescription& problem) const;
};

struct GemmFwd1x1_0_2 : GemmFwdBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmFwd1x1_0_1_int8 : GemmFwdBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmFwd1x1_0_1 : GemmFwdBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmFwdRest : GemmFwdBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmBwdBase : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsDynamic() const { return true; }
    float GetWti(const ConvolutionContext& ctx) const { return GetWti(ctx, ctx.conv_problem); }
    float GetWti(const ExecutionContext& context, const conv::ProblemDescription& problem) const;
};

struct GemmBwd1x1_stride2 : GemmBwdBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmBwd1x1_stride1 : GemmBwdBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmBwdRest : GemmBwdBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmWrwBase : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsDynamic() const { return true; }
    float GetWti(const ConvolutionContext& ctx) const { return GetWti(ctx, ctx.conv_problem); }
    float GetWti(const ExecutionContext& context, const conv::ProblemDescription& problem) const;
};

struct GemmWrw1x1_stride1 : GemmWrwBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmWrwUniversal : GemmWrwBase
{
    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        return GetWorkspaceSize(ctx, ctx.conv_problem);
    }

    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        return IsApplicable(ctx, ctx.conv_problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx) const
    {
        return GetSolution(ctx, ctx.conv_problem);
    }

    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct PerformanceConfigAsmImplicitGemmGTC : Serializable<PerformanceConfigAsmImplicitGemmGTC>
{
    std::string direction;
    std::string tensor_layout;
    miopenDataType_t precision;
    int nxb;
    int nxe;

    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;

    int wave_tile_m;
    int wave_tile_n;
    int wave_tile_k;
    int wave_step_m;
    int wave_step_n;
    int wave_repeat_m;
    int wave_repeat_n;

    int multihead;
    int vector_store;
    int gemm_k_global_split;
    int merge_e;
    int tensor_a_pass_through;

    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;

    bool use_spare_set;
    int index;

    PerformanceConfigAsmImplicitGemmGTC(std::string dir,
                                        std::string layout,
                                        miopenDataType_t prec,
                                        int b,
                                        int e,
                                        int mpb,
                                        int npb,
                                        int kpb,
                                        int wtm,
                                        int wtn,
                                        int wtk,
                                        int wsm,
                                        int wsn,
                                        int wrm,
                                        int wrn,
                                        int mh,
                                        int vs,
                                        int gks,
                                        int me,
                                        int pta,
                                        std::initializer_list<int> ta_t,
                                        std::initializer_list<int> ta_c,
                                        std::initializer_list<int> tb_t,
                                        std::initializer_list<int> tb_c,
                                        bool spare = false);
    PerformanceConfigAsmImplicitGemmGTC()
        : PerformanceConfigAsmImplicitGemmGTC("fwd",
                                              "nchw",
                                              miopenFloat,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTC("fwd",
                                              "nchw",
                                              miopenFloat,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        std::string prec_string = self.precision == miopenFloat
                                      ? "fp32"
                                      : (self.precision == miopenHalf ? "fp16" : "bf16");
        f(self.direction, "dir");
        f(self.tensor_layout, "lyt");
        f(prec_string, "pre");
        f(self.nxb, "nxb");
        f(self.nxe, "nxe");
        f(self.gemm_m_per_block, "mpb");
        f(self.gemm_n_per_block, "npb");
        f(self.gemm_k_per_block, "kpb");

        f(self.wave_tile_m, "wtm");
        f(self.wave_tile_n, "wtn");
        f(self.wave_tile_k, "wtk");
        f(self.wave_step_m, "wsm");
        f(self.wave_step_n, "wsn");
        f(self.wave_repeat_m, "wrm");
        f(self.wave_repeat_n, "wrn");

        f(self.multihead, "mh");
        f(self.vector_store, "vs");
        f(self.gemm_k_global_split, "gks");
        f(self.merge_e, "me");
        f(self.tensor_a_pass_through, "pta");

        f(self.tensor_a_thread_lengths[0], "ta0");
        f(self.tensor_a_thread_lengths[1], "ta1");
        f(self.tensor_a_thread_lengths[2], "ta2");
        f(self.tensor_a_thread_lengths[3], "ta3");

        f(self.tensor_a_cluster_lengths[0], "ca0");
        f(self.tensor_a_cluster_lengths[1], "ca1");
        f(self.tensor_a_cluster_lengths[2], "ca2");
        f(self.tensor_a_cluster_lengths[3], "ca3");

        f(self.tensor_b_thread_lengths[0], "tb0");
        f(self.tensor_b_thread_lengths[1], "tb1");
        f(self.tensor_b_thread_lengths[2], "tb2");
        f(self.tensor_b_thread_lengths[3], "tb3");

        f(self.tensor_b_cluster_lengths[0], "cb0");
        f(self.tensor_b_cluster_lengths[1], "cb1");
        f(self.tensor_b_cluster_lengths[2], "cb2");
        f(self.tensor_b_cluster_lengths[3], "cb3");
        f(self.index, "index");
    }

    // Chilrden must provide support for ComputedContainer.
    void HeuristicInit(const ConvolutionContext&) = delete;
    bool SetNextValue(const ConvolutionContext&)  = delete;
    bool IsValidValue() const                     = delete;
    bool IsValid(const ConvolutionContext&) const = delete;

    bool IsDefaultConstructed() const;
    bool operator==(const PerformanceConfigAsmImplicitGemmGTC& other) const;
    void CopyParameters(const PerformanceConfigAsmImplicitGemmGTC& other);
    std::string ToString() const;
    std::string ToKernelName(const ConvolutionContext& ctx) const;
    int BlockSize() const;
};

struct PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           miopenFloat,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           miopenFloat,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC : SolverBase<ConvolutionContext>
{
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC&) const;
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    Search(const ConvolutionContext&, const AnyInvokeParams& invoke_ctx) const;

    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;

    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           miopenFloat,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           miopenFloat,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }
    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
};

struct ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC : SolverBase<ConvolutionContext>
{
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC&) const;
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    Search(const ConvolutionContext&, const AnyInvokeParams& invoke_ctx) const;

    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;

    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC("fwd",
                                                           "nchw",
                                                           miopenFloat,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC("fwd",
                                                           "nchw",
                                                           miopenFloat,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    void HeuristicInit(const ConvolutionContext& ctx);
    bool SetNextValue(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext& ctx) const;
    size_t ComputeKernelOccupancy() const;
    private:
    void SetParamsForKSplit(const ConvolutionContext& ctx, const size_t& occupancy);
};

struct ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC : SolverBase<ConvolutionContext>
{
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC&) const;
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    Search(const ConvolutionContext&, const AnyInvokeParams& invoke_ctx) const;

    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const;

    bool IsApplicable(const ConvolutionContext& ctx) const;
    bool IsDynamic() const { return true; }
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct AnySolver;

} // namespace solver
} // namespace miopen

struct mlo_construct_direct2D_fusion : mlo_construct_base
{
    mlo_construct_direct2D_fusion(miopen::conv::Direction dir, bool do_bias = false)
        : mlo_construct_base(dir, do_bias)
    {
    }
    mlo_construct_direct2D_fusion(const miopen::TensorDescriptor& in,
                                  const miopen::TensorDescriptor& weights,
                                  const miopen::TensorDescriptor& out,
                                  const miopen::ConvolutionDescriptor& conv,
                                  miopen::conv::Direction dir,
                                  bool do_bias = false)
        : mlo_construct_base(in, weights, out, conv, dir, do_bias)
    {
    }

    bool IsAutoTuneEnabled() const { return _search_params.do_search; }

    inline void mloCopyTo(miopen::ConvolutionContext& params) const /// TODO: get rid of this
    {
        params = _search_params;
    }
    miopen::solver::ConvSolution FindSolution(const std::vector<miopen::solver::AnySolver>& solvers,
                                              const miopen::AnyInvokeParams& invoke_ctx);
};

#endif // GUARD_MIOPEN_SOLVER_HPP_
