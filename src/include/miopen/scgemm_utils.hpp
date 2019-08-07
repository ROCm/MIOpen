/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_SCGEMM_UTILS_HPP_
#define GUARD_MIOPEN_SCGEMM_UTILS_HPP_

#include <miopen/handle.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/scgemm_param.hpp>
#include <miopen/scgemm/scgemm.hpp>

#define MIOPEN_SCGEMM_OP_TYPE_TEMPLATE(OTYPE, RTYPE)                                     \
    template <>                                                                          \
    struct scgemm_op_type<OTYPE>                                                         \
    {                                                                                    \
        using routine_type = RTYPE;                                                      \
        static auto GetSCGemmRoutines() -> decltype(std::vector<routine_type>());        \
        static int Routine2Int(routine_type r) { return static_cast<int>(r); };          \
        static routine_type Int2Routine(int i) { return static_cast<routine_type>(i); }; \
    };                                                                                   \
    extern template struct scgemm_op_type<OTYPE>;

namespace miopen {

template <SCGemmOpType T>
struct scgemm_op_type
{
    using routine_type = int;
    static auto GetSCGemmRoutines() -> decltype(std::vector<routine_type>())
    {
        return std::vector<routine_type>();
    };
    static int Routine2Int(routine_type r) { return r; };
    static routine_type Int2Routine(int i) { return i; };
};

MIOPEN_SCGEMM_OP_TYPE_TEMPLATE(SCGemmOpFConv, scgemm::scgemm_conv_routine_t)
MIOPEN_SCGEMM_OP_TYPE_TEMPLATE(SCGemmOpFGemm, scgemm::scgemm_gemm_routine_t)

size_t
GetSCGemmConvFwdWorkSpaceSize(const ConvolutionContext& params, SCGemmOpType type, int routine);

size_t GetMaximumSCGemmConvFwdWorkSpaceSize(const ConvolutionContext& params);

size_t
GetSCGemmConvFwdAuxBufferSize(const ConvolutionContext& params, SCGemmOpType type, int routine);

size_t GetMaximumSCGemmConvFwdAuxBufferSize(const ConvolutionContext& params, SCGemmOpType type);

void CompiledSCGemmKernelParams(const ConvolutionContext& params,
                                SCGemmKernelParams& scgParams,
                                uint32_t mask = 0);

void CompiledSCGemmKernelParamsFromSolution(const miopen::solver::ConvSolution& solution,
                                            const ConvolutionContext& params,
                                            SCGemmKernelParams& scgParams,
                                            uint32_t mask = 0);

float CallSCGemm(miopen::Handle& handle,
                 const ConvolutionContext& ctx,
                 ConstData_t src,
                 Data_t dst,
                 ConstData_t wei,
                 ConstData_t bias,
                 Data_t workspace,
                 std::vector<KernelInvoke>& kernels,
                 uint32_t mask = 0,
                 float coef    = 1.0f);

} // namespace miopen
#endif
