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

#include <miopen/db.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/stringutils.hpp>

#include <boost/range/adaptor/transformed.hpp>
#include <ostream>

namespace miopen {
namespace solver {

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
};

struct SolverRegistrar
{
    SolverRegistrar(IdRegistryData& registry);
};

static auto& IdRegistry()
{
    static auto data            = IdRegistryData{};
    static const auto registrar = SolverRegistrar{data};
    (void)registrar; // clang-tidy
    return data;
}

Id::Id(uint64_t value_) : value(value_)
{
    if(IdRegistry().value_to_str.find(value_) == IdRegistry().value_to_str.end())
        value = invalid;
}

Id::Id(const std::string& str)
{
    const auto it = IdRegistry().str_to_value.find(str);
    value         = it != IdRegistry().str_to_value.end() ? it->second : invalid;
}

Id::Id(const char* str)
{
    const auto it = IdRegistry().str_to_value.find(str);
    value         = it != IdRegistry().str_to_value.end() ? it->second : invalid;
}

std::string Id::ToString() const
{
    if(!IsValid())
        return "invalid solver::Id";
    return IdRegistry().value_to_str[value];
}

AnySolver Id::GetSolver() const
{
    const auto it = IdRegistry().value_to_solver.find(value);
    return it != IdRegistry().value_to_solver.end() ? it->second : AnySolver{};
}

inline bool Register(IdRegistryData& registry, uint64_t value, const std::string& str)
{
    if(value == Id::invalid)
    {
        MIOPEN_LOG_E(Id::invalid << " is special id value for invalid solver (" << str << ")");
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
    return true;
}

template <class TSolver>
inline void RegisterWithSolver(IdRegistryData& registry, uint64_t value, TSolver)
{
    if(Register(registry, value, SolverDbId(TSolver{})))
        registry.value_to_solver.emplace(value, TSolver{});
}

inline SolverRegistrar::SolverRegistrar(IdRegistryData& registry)
{
    // When solver gets removed its registration line should be replaced with ++id to keep
    // backwards compatibility. New solvers should only be added to the end of list unless it is
    // intended to reuse an id of a removed solver.

    uint64_t id = 0;
    RegisterWithSolver(registry, ++id, ConvAsm3x3U{});
    RegisterWithSolver(registry, ++id, ConvAsm1x1U{});
    RegisterWithSolver(registry, ++id, ConvAsm1x1UV2{});
    RegisterWithSolver(registry, ++id, ConvBiasActivAsm1x1U{});
    RegisterWithSolver(registry, ++id, ConvAsm5x10u2v2f1{});
    RegisterWithSolver(registry, ++id, ConvAsm5x10u2v2b1{});
    RegisterWithSolver(registry, ++id, ConvAsm7x7c3h224w224k64u2v2p3q3f1{});
    RegisterWithSolver(registry, ++id, ConvOclDirectFwd11x11{});
    RegisterWithSolver(registry, ++id, ConvOclDirectFwdGen{});
    RegisterWithSolver(registry, ++id, ConvOclDirectFwd3x3{});
    RegisterWithSolver(registry, ++id, ConvOclDirectFwd{});
    RegisterWithSolver(registry, ++id, ConvOclDirectFwdFused{});
    RegisterWithSolver(registry, ++id, ConvOclDirectFwd1x1{});
    RegisterWithSolver(registry, ++id, ConvBinWinograd3x3U{});
    RegisterWithSolver(registry, ++id, ConvBinWinogradRxS{});
    RegisterWithSolver(registry, ++id, ConvAsmBwdWrW3x3{});
    RegisterWithSolver(registry, ++id, ConvAsmBwdWrW1x1{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<1>{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<2>{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<4>{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<8>{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2<16>{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW2NonTunable{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW53{});
    RegisterWithSolver(registry, ++id, ConvOclBwdWrW1x1{});

    // Several ids w/o solver for immediate mode
    Register(registry, ++id, "gemm");
    Register(registry, ++id, "fft");
}

} // namespace solver
} // namespace miopen
