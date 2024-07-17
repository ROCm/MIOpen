
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/batchnorm/solvers.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/batchnorm_infer.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_BN_INFER)

namespace miopen {
namespace solver {
namespace batchnorm {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

#endif

bool BnCKFwdInference::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_BN_INFER))
        return false;
    if(!bn_problem.IsLayoutNHWC())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;
    if(!bn_problem.Is2D())
        return false;
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardInference)
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf:
        return (CheckBnCKFwdApplicability<F16, F16, F32, F16, F16, F32>(bn_problem) != -1);
    case miopenFloat:
        return (CheckBnCKFwdApplicability<F32, F32, F32, F32, F32, F32>(bn_problem) != -1);
    case miopenDouble:
        return (CheckBnCKFwdApplicability<F64, F64, F64, F64, F64, F64>(bn_problem) != -1);
    case miopenBFloat16:
        return (CheckBnCKFwdApplicability<BF16, BF16, F32, BF16, BF16, F32>(bn_problem) != -1);
    case miopenInt64:
    case miopenInt32:
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8: break;
    }
#endif
    return false;
}

ConvSolution BnCKFwdInference::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(bn_problem.GetXDesc().GetType())
            {
            case miopenHalf:
                InitInvokerFactoryBnCKFwdInferenceNHWC<F16, F16, F32, F16, F16, F32>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenFloat:
                InitInvokerFactoryBnCKFwdInferenceNHWC<F32, F32, F32, F32, F32, F32>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenDouble:
                InitInvokerFactoryBnCKFwdInferenceNHWC<F64, F64, F64, F64, F64, F64>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenBFloat16:
                InitInvokerFactoryBnCKFwdInferenceNHWC<BF16, BF16, F32, BF16, BF16, F32>(
                    handle, primitive_parameters, bn_problem);
                break;
            case miopenInt8:
            case miopenInt32:
            case miopenInt64:
            case miopenFloat8:
            case miopenBFloat8:
            default: MIOPEN_THROW("Unsupported datatype");
            }
        };
    };
    return result;
#else
    return {};
#endif
}

} // namespace batchnorm
} // namespace solver
} // namespace miopen
