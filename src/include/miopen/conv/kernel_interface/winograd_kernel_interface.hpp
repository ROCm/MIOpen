/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/conv/problem_description.hpp>

namespace miopen {
namespace conv {

enum class WinoShaderFlagsV40 : uint64_t {
    F_REVERSE_R                  = 1ULL << 0,
    F_REVERSE_S                  = 1ULL << 1,
    F_FLIP_K_C                   = 1ULL << 2, // Deprecated
    F_DENORMS_RND_ENABLE         = 1ULL << 3,
    F_MALL_READ_CACHE_ENABLE     = 1ULL << 4,
    F_ADDR_INDIRECT              = 1ULL << 6,
    F_BIAS                       = 1ULL << 7,
    F_LEAKY_RELU                 = 1ULL << 8, // Deprecated
    F_NKCHR_STRIDES              = 1ULL << 9,
    F_GROUPED_CONVOLUTION        = 1ULL << 10,
    F_FORCE_FILTER_TRAVERSE_MODE = 1ULL << 11,
    F_FILTER_TRAVERSE_DUAL       = 1ULL << 12,
    F_TENSOR_OFFSETS             = 1ULL << 13,
    F_USE_ACTIVATION_MODE        = 1ULL << 14,
    F_USE_EXTENDED_FLAGS_64      = 1ULL << 15,
};

inline WinoShaderFlagsV40 operator|(WinoShaderFlagsV40 lhs, WinoShaderFlagsV40 rhs)
{
    using T = std::underlying_type_t<WinoShaderFlagsV40>;
    return static_cast<WinoShaderFlagsV40>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline WinoShaderFlagsV40 operator|=(WinoShaderFlagsV40 lhs, WinoShaderFlagsV40 rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

inline std::ostream& operator<<(std::ostream& s, const WinoShaderFlagsV40& flags)
{
    using T = std::underlying_type_t<WinoShaderFlagsV40>;
    s << "0x" << std::hex << static_cast<T>(flags) << std::dec;
    return s;
}

enum class WinoShaderActivationModeV40_t : uint8_t {
    IDENTITY    = 0, // no activation, alpha and beta are ignored
    LEAKY_RELU  = 1, // ReLU, beta field is ignored
    SIGMOID     = 2, // sigmoid, alpha and beta fields are ignored
    SCALED_TANH = 3, // parametric tanh function
};

inline std::ostream& operator<<(std::ostream& s, const WinoShaderActivationModeV40_t& mode)
{
    using T = std::underlying_type_t<WinoShaderActivationModeV40_t>;
    s << static_cast<T>(mode);
    return s;
}

struct WinoShaderArgsV40
{
    // Main convolution parameters
    uint32_t N;           // batch size
    uint32_t C;           // number of input channels in each filter group
    uint32_t H;           // input height
    uint32_t W;           // input width
    uint32_t K;           // number of output channels in each filter group
    uint32_t R;           // filter height
    uint32_t S;           // filter width
    int32_t pad_h;        // padding in h dimension
    int32_t pad_w;        // padding in w dimension
    uint32_t out_h;       // output height
    uint32_t out_w;       // output width
    uint32_t G;           // number of filter groups

    // Data layout related parameters
    uint32_t d_N_stride; // stride in number of elements of the N dimension of the input data buffer
    uint32_t d_C_stride; // stride in number of elements of the C dimension of the input data buffer
    uint32_t d_H_stride; // stride in number of elements of the H dimension of the input data buffer
    uint32_t d_G_stride; // stride in number of elements of the G dimension of the input data buffer

    uint32_t f_K_stride; // stride in number of elements of the K dimension of the filter buffer
    uint32_t f_C_stride; // stride in number of elements of the C dimension of the filter buffer
    uint32_t f_R_stride; // stride in number of elements of the R dimension of the filter buffer
    uint32_t f_G_stride; // stride in number of elements of the G dimension of the filter buffer

    uint32_t o_N_stride; // stride in number of elements of the N dimension of the output buffer
    uint32_t o_K_stride; // stride in number of elements of the K dimension of the output buffer
    uint32_t o_H_stride; // stride in number of elements of the H dimension of the output buffer
    uint32_t o_G_stride; // stride in number of elements of the G dimension of the output buffer

    // Fused activation parameters
    float alpha;          // activation parameter alpha
    float beta;           // activation parameter beta
    WinoShaderActivationModeV40_t activation_mode;  // activation mode

    // Other shader parameters
    uint32_t n_groups;    // number of shader groups
    WinoShaderFlagsV40 flags;   // shader flags
    uint8_t sync_limit;  // maximum number of sync attempts
    uint8_t sync_period; // synchronization period

    bool SetConvParams(const ProblemDescription& problem);
    void SetStrides(const ProblemDescription& problem);
    void SetActivParams(WinoShaderActivationModeV40_t mode, float alpha, float beta) noexcept;
    void SetShaderParams(uint32_t n_groups, WinoShaderFlagsV40 flags, uint8_t sync_limit, uint8_t sync_period) noexcept;  
};

} // namespace conv
} // namespace miopen
