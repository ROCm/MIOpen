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
#include <miopen/check_numerics.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/miopen.h>
#include <miopen/softmax.hpp>
#include <miopen/softmax/invoke_params.hpp>
#include <miopen/softmax/solvers.hpp>
#include <miopen/tensor.hpp>

#include <nlohmann/json.hpp>

namespace miopen {

extern "C" miopenStatus_t miopenCreateSoftmaxDescriptor(miopenSoftmaxDescriptor_t* softmaxDesc)
{
    MIOPEN_LOG_FUNCTION(softmaxDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(softmaxDesc);
        desc       = new miopen::SoftmaxDescriptor();
    });
}

extern "C" miopenStatus_t miopenSetSoftmaxDescriptor(miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float alpha,
                                                     float beta,
                                                     miopenSoftmaxAlgorithm_t algorithm,
                                                     miopenSoftmaxMode_t mode)
{

    MIOPEN_LOG_FUNCTION(softmaxDesc, alpha, beta, algorithm, mode);
    return miopen::try_(
        [&] { miopen::deref(softmaxDesc).SetParams(alpha, beta, algorithm, mode); });
}

extern "C" miopenStatus_t miopenGetSoftmaxDescriptor(const miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float* alpha,
                                                     float* beta,
                                                     miopenSoftmaxAlgorithm_t* algorithm,
                                                     miopenSoftmaxMode_t* mode)
{
    MIOPEN_LOG_FUNCTION(softmaxDesc);
    return miopen::try_([&] {
        *alpha     = miopen::deref(softmaxDesc).GetAlpha();
        *beta      = miopen::deref(softmaxDesc).GetBeta();
        *algorithm = miopen::deref(softmaxDesc).GetAlgorithm();
        *mode      = miopen::deref(softmaxDesc).GetMode();
    });
}

std::ostream& operator<<(std::ostream& stream, const SoftmaxDescriptor& x)
{
    stream << "softmax,"
           << "alpha" << x.GetAlpha() << ",beta" << x.GetBeta() << ",algorithm" << x.GetAlgorithm()
           << ",mode" << x.GetMode() << ",";

    return stream;
}

void to_json(nlohmann::json& json, const SoftmaxDescriptor& descriptor)
{
    json = nlohmann::json{
        {"alpha", descriptor.GetAlpha()},
        {"beta", descriptor.GetBeta()},
        {"algorithm", descriptor.GetAlgorithm()},
        {"mode", descriptor.GetMode()},
    };
}

void from_json(const nlohmann::json& json, SoftmaxDescriptor& descriptor)
{
    json.at("alpha").get_to(descriptor.alpha);
    json.at("beta").get_to(descriptor.beta);
    json.at("algorithm").get_to(descriptor.algorithm);
    json.at("mode").get_to(descriptor.mode);
}

miopenStatus_t SoftmaxForward(const Handle& handle,
                              const void* alpha,
                              const void* beta,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              miopenSoftmaxAlgorithm_t algorithm,
                              miopenSoftmaxMode_t mode,
                              int x_offset,
                              int y_offset)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto problem = softmax::ProblemDescription{alpha, beta, xDesc, yDesc, algorithm, mode};
    const auto invoke_params =
        softmax::InvokeParams{alpha, beta, xDesc, x, yDesc, y, algorithm, mode, x_offset, y_offset};
    const auto algo = AlgorithmName{"Softmax"};
    const auto solvers =
        solver::SolverContainer<solver::softmax::AttnSoftmax, solver::softmax::Softmax>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxBackward(const Handle& handle,
                               const void* alpha,
                               const TensorDescriptor& yDesc,
                               ConstData_t y,
                               const TensorDescriptor& dyDesc,
                               ConstData_t dy,
                               const void* beta,
                               const TensorDescriptor& dxDesc,
                               Data_t dx,
                               miopenSoftmaxAlgorithm_t algorithm,
                               miopenSoftmaxMode_t mode,
                               int y_offset,
                               int dy_offset,
                               int dx_offset)
{
    if(dx == nullptr || y == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto problem =
        softmax::ProblemDescription{alpha, beta, yDesc, dyDesc, dxDesc, algorithm, mode};
    const auto invoke_params = softmax::InvokeParams{alpha,
                                                     beta,
                                                     yDesc,
                                                     y,
                                                     dyDesc,
                                                     dy,
                                                     dxDesc,
                                                     dx,
                                                     algorithm,
                                                     mode,
                                                     y_offset,
                                                     dy_offset,
                                                     dx_offset};
    const auto algo          = AlgorithmName{"Softmax"};
    const auto solvers       = solver::SolverContainer<solver::softmax::Softmax>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxForward_V3(Handle& handle,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& outputDesc,
                                 Data_t output,
                                 uint32_t dim,
                                 miopenSoftmaxAlgorithm_t algorithm)
{
    const auto problem = softmax::ProblemDescription{inputDesc, outputDesc, dim, algorithm};

    const auto invoke_params = [&]() {
        auto tmp      = softmax::InvokeParams{};
        tmp.type      = InvokeType::Run;
        tmp.xdxDesc   = inputDesc;
        tmp.yDesc     = outputDesc;
        tmp.x         = input;
        tmp.forward_y = output;
        tmp.dim       = dim;
        tmp.algorithm = algorithm;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SoftmaxV3Forward"};
    const auto solvers = solver::SolverContainer<solver::softmax::SoftmaxV3Forward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxBackward_V3(Handle& handle,
                                  const TensorDescriptor& outputDesc,
                                  ConstData_t output,
                                  const TensorDescriptor& outputGradDesc,
                                  ConstData_t outputGrad,
                                  const TensorDescriptor& inputGradDesc,
                                  Data_t inputGrad,
                                  uint32_t dim,
                                  miopenSoftmaxAlgorithm_t algorithm)
{
    const auto problem =
        softmax::ProblemDescription{outputDesc, outputGradDesc, inputGradDesc, dim, algorithm};

    const auto invoke_params = [&]() {
        auto tmp       = softmax::InvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.yDesc      = outputDesc;
        tmp.backward_y = output;
        tmp.dyDesc     = outputGradDesc;
        tmp.dy         = outputGrad;
        tmp.xdxDesc    = inputGradDesc;
        tmp.dx         = inputGrad;
        tmp.dim        = dim;
        tmp.algorithm  = algorithm;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SoftmaxV3Backward"};
    const auto solvers = solver::SolverContainer<solver::softmax::SoftmaxV3Backward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
