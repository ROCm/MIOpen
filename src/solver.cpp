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

#include <miopen/solver.hpp>
#include <miopen/conv_algo_name.hpp>

#include <miopen/db.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/par_for.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/timer.hpp>

#include <boost/range/adaptor/transformed.hpp>
#include <ostream>

namespace miopen {
namespace solver {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_COMPILE_PARALLEL_LEVEL)

std::ostream& operator<<(std::ostream& os, const KernelInfo& k)
{
    os << k.kernel_file << ", " << k.kernel_name << " g_wk={ ";
    for(const auto& size : k.g_wk)
        os << size << ' ';
    os << "}, l_wk={ ";
    for(const auto& size : k.l_wk)
        os << size << ' ';
    return os << "} '" << k.comp_options << '\'';
}

std::vector<Program> PrecompileKernels(const Handle& h, const std::vector<KernelInfo>& kernels)
{
    CompileTimer ct;
    std::vector<Program> programs(kernels.size());

    // clang-format off
    par_for_strided(kernels.size(),
                    max_threads{Value(MIOPEN_COMPILE_PARALLEL_LEVEL{}, 20)},
                    [&](auto i) {
                        const KernelInfo& k = kernels[i];
                        programs[i]         = h.LoadProgram(k.kernel_file, k.comp_options, false, "");
                    });
    // clang-format on
    ct.Log("PrecompileKernels");
    return programs;
}

void PrecompileSolutions(const Handle& h, const std::vector<const ConvSolution*>& sols)
{
    // Find all kernels that need to be compiled from the solutions
    std::vector<KernelInfo> kernels;
    for(auto&& sol : sols)
    {
        if(!sol->Succeeded())
            continue;
        for(auto&& kernel : sol->construction_params)
        {
            if(h.HasProgram(kernel.kernel_file, kernel.comp_options))
                continue;
            kernels.push_back(kernel);
        }
    }

    // Precompile the kernels in parallel, but dont add them to the cache
    std::vector<Program> programs = PrecompileKernels(h, kernels);

    // Add programs to the cache
    for(std::size_t i = 0; i < programs.size(); i++)
    {
        const KernelInfo& k = kernels[i];
        h.AddProgram(programs[i], k.kernel_file, k.comp_options);
    }
}

std::ostream& operator<<(std::ostream& os, const ConvSolution& s)
{
    auto strings =
        s.construction_params | boost::adaptors::transformed([](auto k) { return k.kernel_name; });
    os << s.solver_id << ": " << JoinStrings(strings, "/");
    return os;
}

struct IdRegistryData
{
    std::unordered_map<uint64_t, std::string> value_to_str;
    std::unordered_map<std::string, uint64_t> str_to_value;
    std::unordered_map<uint64_t, AnySolver> value_to_solver;
    std::unordered_map<uint64_t, miopenConvAlgorithm_t> value_to_algo;
};

struct SolverRegistrar
{
    SolverRegistrar(IdRegistryData& registry);
};

static auto& IdRegistry()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static auto data            = IdRegistryData{};
    static const auto registrar = SolverRegistrar{data};
    (void)registrar; // clang-tidy
    return data;
}

const std::unordered_map<uint64_t, AnySolver>& GetMapValueToAnySolver()
{
    return IdRegistry().value_to_solver;
}

Id::Id(uint64_t value_) : value(value_)
{
    is_valid = (IdRegistry().value_to_str.find(value) != IdRegistry().value_to_str.end());
}

Id::Id(const std::string& str) : Id(str.c_str()) {}

Id::Id(const char* str)
{
    const auto it = IdRegistry().str_to_value.find(str);
    is_valid      = (it != IdRegistry().str_to_value.end());
    value         = is_valid ? it->second : invalid_value;
}

std::string Id::ToString() const
{
    if(!IsValid())
        return "INVALID_SOLVER_ID_" + std::to_string(value);
    return IdRegistry().value_to_str[value];
}

AnySolver Id::GetSolver() const
{
    const auto it = IdRegistry().value_to_solver.find(value);
    return it != IdRegistry().value_to_solver.end() ? it->second : AnySolver{};
}

std::string Id::GetAlgo(conv::Direction dir) const
{
    return ConvolutionAlgoToDirectionalString(GetAlgo(), dir);
}

miopenConvAlgorithm_t Id::GetAlgo() const
{
    const auto it = IdRegistry().value_to_algo.find(value);
    if(it == IdRegistry().value_to_algo.end())
        MIOPEN_THROW(miopenStatusInternalError);
    return it->second;
}

inline bool Register(IdRegistryData& registry,
                     uint64_t value,
                     const std::string& str,
                     miopenConvAlgorithm_t algo)
{
    if(value == Id::invalid_value)
    {
        MIOPEN_LOG_E(Id::invalid_value << " is special id value for invalid solver (" << str
                                       << ")");
        return false;
    }

    if(registry.value_to_str.find(value) != registry.value_to_str.end())
    {
        MIOPEN_LOG_E("Registered duplicate ids: [" << value << "]" << str << " and ["
                                                   << registry.value_to_str.find(value)->first
                                                   << "]"
                                                   << registry.value_to_str.find(value)->second);
        return false;
    }

    if(registry.str_to_value.find(str) != registry.str_to_value.end())
    {
        MIOPEN_LOG_E("Registered duplicate ids: [" << value << "]" << str << " and ["
                                                   << registry.str_to_value.find(str)->second
                                                   << "]"
                                                   << registry.str_to_value.find(str)->first);
        return false;
    }

    registry.value_to_str.emplace(value, str);
    registry.str_to_value.emplace(str, value);
    registry.value_to_algo.emplace(value, algo);
    return true;
}

template <class TSolver>
inline void
RegisterWithSolver(IdRegistryData& registry, uint64_t value, TSolver, miopenConvAlgorithm_t algo)
{
    if(Register(registry, value, SolverDbId(TSolver{}), algo))
        registry.value_to_solver.emplace(value, TSolver{});
}

inline SolverRegistrar::SolverRegistrar(IdRegistryData& registry)
{
    // When solver gets removed its registration line should be replaced with ++id to keep
    // backwards compatibility. New solvers should only be added to the end of list unless it is
    // intended to reuse an id of a removed solver.

    uint64_t id = 0; // 0 is reserved for invalid value.

    // IMPORTANT: New solvers should be added to the end of the function!

    RegisterWithSolver(registry, ++id, ConvAsm3x3U{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvAsm1x1U{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvAsm1x1UV2{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvBiasActivAsm1x1U{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvAsm5x10u2v2f1{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvAsm5x10u2v2b1{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(
        registry, ++id, ConvAsm7x7c3h224w224k64u2v2p3q3f1{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclDirectFwd11x11{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclDirectFwdGen{}, miopenConvolutionAlgoDirect);
    ++id; // removed ConvOclDirectFwd3x3
    RegisterWithSolver(registry, ++id, ConvOclDirectFwd{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclDirectFwdFused{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclDirectFwd1x1{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvBinWinograd3x3U{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(registry, ++id, ConvBinWinogradRxS{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(registry, ++id, ConvAsmBwdWrW3x3{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvAsmBwdWrW1x1{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<1>{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<2>{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<4>{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<8>{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<16>{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2NonTunable{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW53{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW1x1{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmV4R1Fwd{}, miopenConvolutionAlgoImplicitGEMM);
    ++id; // removed solver ConvHipImplicitGemmV4Fwd
    ++id; // removed solver ConvHipImplicitGemmV4_1x1
    ++id; // removed solver ConvHipImplicitGemmV4R4FwdXdlops
    ++id; // removed solver ConvHipImplicitGemmV4R4Xdlops_1x1
    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmV4R1WrW{}, miopenConvolutionAlgoImplicitGEMM);
    ++id; // removed solver ConvHipImplicitGemmV4WrW

    // Several ids w/o solver for immediate mode
    Register(registry, ++id, "gemm", miopenConvolutionAlgoGEMM);
    RegisterWithSolver(registry, ++id, fft{}, miopenConvolutionAlgoFFT);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<3, 4>{}, miopenConvolutionAlgoWinograd);
    ++id; // Id for ConvSCGemmFGemm.
    RegisterWithSolver(registry, ++id, ConvBinWinogradRxSf3x2{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<3, 5>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<3, 6>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<3, 2>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<3, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<7, 2>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<7, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<7, 2, 1, 1>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<7, 3, 1, 1>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<1, 1, 7, 2>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<1, 1, 7, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<5, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvWinograd3x3MultipassWrW<5, 4>{}, miopenConvolutionAlgoWinograd);

    ++id; // removed solver ConvHipImplicitGemmV4R4WrWXdlops
    ++id; // removed solver ConvHipImplicitGemmV4R4GenFwdXdlops
    ++id; // removed solver ConvHipImplicitGemmV4R4GenWrWXdlops

    RegisterWithSolver(registry, ++id, ConvBinWinogradRxSf2x3{}, miopenConvolutionAlgoWinograd);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmV4R4Fwd{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmBwdDataV1R1{}, miopenConvolutionAlgoImplicitGEMM);
    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmBwdDataV4R1{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmBwdDataV1R1Xdlops{}, miopenConvolutionAlgoImplicitGEMM);

    ++id; // removed solver ConvHipImplicitGemmV4R4GenXdlopsFwdFp32
    ++id; // removed solver ConvHipImplicitGemmV4R4GenXdlopsWrWFp32

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmBwdDataV4R1Xdlops{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmV4R4WrW{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvAsmImplicitGemmV4R1DynamicFwd{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvAsmImplicitGemmV4R1DynamicFwd_1x1{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmForwardV4R4Xdlops{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvAsmImplicitGemmV4R1DynamicBwd{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvAsmImplicitGemmV4R1DynamicWrw{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd<2, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd<3, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd<4, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd<5, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd<6, 3>{}, miopenConvolutionAlgoWinograd);

    RegisterWithSolver(registry,
                       ++id,
                       ConvAsmImplicitGemmGTCDynamicWrwXdlops{},
                       miopenConvolutionAlgoImplicitGEMM);
    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmWrwV4R4Xdlops{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(registry,
                       ++id,
                       ConvAsmImplicitGemmGTCDynamicFwdXdlops{},
                       miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd_xdlops<2, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd_xdlops<3, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd_xdlops<4, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd_xdlops<5, 3>{}, miopenConvolutionAlgoWinograd);
    RegisterWithSolver(
        registry, ++id, ConvMPBidirectWinograd_xdlops<6, 3>{}, miopenConvolutionAlgoWinograd);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmForwardV4R5Xdlops{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(registry,
                       ++id,
                       ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm{},
                       miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(registry,
                       ++id,
                       ConvAsmImplicitGemmGTCDynamicBwdXdlops{},
                       miopenConvolutionAlgoImplicitGEMM);
    RegisterWithSolver(registry,
                       ++id,
                       ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm{},
                       miopenConvolutionAlgoImplicitGEMM);
    RegisterWithSolver(registry, ++id, ConvBinWinogradRxSf2x3g1{}, miopenConvolutionAlgoWinograd);

    RegisterWithSolver(registry, ++id, ConvDirectNaiveConvFwd{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvDirectNaiveConvBwd{}, miopenConvolutionAlgoDirect);
    RegisterWithSolver(registry, ++id, ConvDirectNaiveConvWrw{}, miopenConvolutionAlgoDirect);

    RegisterWithSolver(registry, ++id, GemmFwd1x1_0_1{}, miopenConvolutionAlgoGEMM);
    RegisterWithSolver(registry, ++id, GemmFwd1x1_0_1_int8{}, miopenConvolutionAlgoGEMM);
    RegisterWithSolver(registry, ++id, GemmFwd1x1_0_2{}, miopenConvolutionAlgoGEMM);
    RegisterWithSolver(registry, ++id, GemmFwdRest{}, miopenConvolutionAlgoGEMM);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmMlirCppFwd{}, miopenConvolutionAlgoImplicitGEMM);
    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmMlirCppBwd{}, miopenConvolutionAlgoImplicitGEMM);
    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmMlirCppWrW{}, miopenConvolutionAlgoImplicitGEMM);

    RegisterWithSolver(registry, ++id, GemmBwd1x1_stride2{}, miopenConvolutionAlgoGEMM);
    RegisterWithSolver(registry, ++id, GemmBwd1x1_stride1{}, miopenConvolutionAlgoGEMM);
    RegisterWithSolver(registry, ++id, GemmBwdRest{}, miopenConvolutionAlgoGEMM);

    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmMlirBinFwd{}, miopenConvolutionAlgoImplicitGEMM);
    RegisterWithSolver(
        registry, ++id, ConvHipImplicitGemmMlirBinBwd{}, miopenConvolutionAlgoImplicitGEMM);
    // IMPORTANT: New solvers should be added to the end of the function!
}

} // namespace solver
} // namespace miopen
