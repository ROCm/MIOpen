/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

// TODO: I don't think we need a separate file for this, we should move this into
// batchnorm_functions.hpp or activation functions.hpp
#ifndef BNORM_SPATIAL_ACTIVATION_FUNCTIONS_HPP
#define BNORM_SPATIAL_ACTIVATION_FUNCTIONS_HPP

#include "configuration.hpp"

namespace miopen {
namespace batchnorm {

template <typename FpPrecType,
          miopen::neuron_op_type NrnOpType = miopen::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::pasthru>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return tmp;
}

template <typename FpPrecType,
          miopen::neuron_op_type NrnOpType = miopen::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return max(tmp, static_cast<FpPrecType>(0.));
}

template <typename FpPrecType,
          miopen::neuron_op_type NrnOpType = miopen::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clipped_relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const&)
{
    return min(_alpha, max(tmp, static_cast<FpPrecType>(0.)));
}

template <typename FpPrecType,
          miopen::neuron_op_type NrnOpType = miopen::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clamp>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const& _beta)
{
    return max(static_cast<FpPrecType>(_alpha), min(static_cast<FpPrecType>(_beta), tmp));
}

} // namespace batchnorm
} // namespace miopen

#endif
