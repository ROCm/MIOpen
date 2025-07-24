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

#include <type_traits>
#include <ostream>
#include <stdint.h>

#include <miopen/miopen.h>

namespace miopen {

namespace conv {
struct ProblemDescription;
} // namespace conv

enum class WinoShaderFlagsV2 : uint64_t
{
    F_REVERSE_R                  = 1ULL << 0,
    F_REVERSE_S                  = 1ULL << 1,
    F_FLIP_K_C                   = 1ULL << 2, // Deprecated
    F_DENORMS_RND_ENABLE         = 1ULL << 3,
    F_MALL_READ_CACHE_ENABLE     = 1ULL << 4,
    F_ACC_PRE_ACTIVATION_MODE    = 1ULL << 5,
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

inline WinoShaderFlagsV2 operator&(WinoShaderFlagsV2 lhs, WinoShaderFlagsV2 rhs)
{
    using T = std::underlying_type_t<WinoShaderFlagsV2>;
    return static_cast<WinoShaderFlagsV2>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline WinoShaderFlagsV2 operator|(WinoShaderFlagsV2 lhs, WinoShaderFlagsV2 rhs)
{
    using T = std::underlying_type_t<WinoShaderFlagsV2>;
    return static_cast<WinoShaderFlagsV2>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline WinoShaderFlagsV2 operator|=(WinoShaderFlagsV2& lhs, WinoShaderFlagsV2 rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

inline std::ostream& operator<<(std::ostream& s, WinoShaderFlagsV2 flags)
{
    using T = std::underlying_type_t<WinoShaderFlagsV2>;
    s << "0x" << std::hex << static_cast<T>(flags) << std::dec;
    return s;
}

enum class WinoShaderActivationModeV2_t : uint8_t
{
    IDENTITY    = 0, // y = x                       no activation, alpha and beta are ignored
    LEAKY_RELU  = 1, // y = x >= 0 ? x : alpha * x  beta is ignored
    SIGMOID     = 2, // y = 1 / (1 + e^-x)          alpha and beta fields are ignored
    SCALED_TANH = 3, // y = alpha * tanh(beta * x), where tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    RELU        = 4  // y = max(0, x)               alpha and beta fields are ignored
};

inline std::ostream& operator<<(std::ostream& s, const WinoShaderActivationModeV2_t& mode)
{
    s << static_cast<unsigned>(mode);
    return s;
}

struct WinoShaderArgsV2
{
    // Main convolution parameters
    uint32_t N;     // batch size
    uint32_t Cg;    // number of input channels in each filter group
    uint32_t H;     // input height
    uint32_t W;     // input width
    uint32_t Kg;    // number of output channels in each filter group
    uint32_t R;     // filter height
    uint32_t S;     // filter width
    int32_t pad_h;  // padding in h dimension
    int32_t pad_w;  // padding in w dimension
    uint32_t out_h; // output height
    uint32_t out_w; // output width
    uint32_t G;     // number of filter groups

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
    WinoShaderActivationModeV2_t activation_mode; // activation mode

    // Other shader parameters
    uint32_t n_groups;         // number of shader groups
    WinoShaderFlagsV2 flags64; // shader flags
    uint8_t sync_limit;        // maximum number of sync attempts
    uint8_t sync_period;       // synchronization period

    bool SetConvParams(const conv::ProblemDescription& problem);
    void SetStrides(const conv::ProblemDescription& problem);
    void SetActivParams(miopenActivationMode_t mode);
    void SetShaderParams(uint32_t n_groups,
                         WinoShaderFlagsV2 flags,
                         uint8_t sync_limit,
                         uint8_t sync_period) noexcept;

    // Template is used to catch -Wshift-count-overflow
    /// \todo Move to a utility header
    template <uint32_t exp, typename T = uint32_t>
    static constexpr T PowOf2() noexcept
    {
        return static_cast<T>(1) << exp;
    }

    bool dimsFit16bit() const noexcept
    {
        // clang-format off
        return N < PowOf2<16>()
            && G < PowOf2<16>()
           && Cg < PowOf2<16>()
           && Kg < PowOf2<16>()
       && Cg * G < PowOf2<16>()
       && Kg * G < PowOf2<16>()
            && H < PowOf2<16>()
            && W < PowOf2<16>()
            && R < PowOf2<16>()
            && S < PowOf2<16>()
        && out_h < PowOf2<16>()
        && out_w < PowOf2<16>() - 3;
        // clang-format on
    }

    bool R_S_fit16bit() const noexcept
    {
        // clang-format off
        return R < PowOf2<16>()
            && S < PowOf2<16>();
        // clang-format on
    }

    bool R_S_fit3x3() const noexcept
    {
        // clang-format off
        return R <= 3U
            && S <= 3U;
        // clang-format on
    }

    bool batchTensorSizesFit31bits() const noexcept
    {
        // clang-format off
        // Convert everything to 64 bit
        uint64_t N_  = N;
        uint64_t Cg_ = Cg;
        uint64_t H_  = H;
        uint64_t W_  = W;
        uint64_t Kg_ = Kg;
        uint64_t R_  = R;
        uint64_t S_  = S;
        uint64_t OH_ = out_h;
        uint64_t OW_ = out_w;
        uint64_t G_  = G;
        // proceed avoiding overflows and assuming dimsFit16bit() passes
        uint64_t KCR =  (((Kg_ * G_ - 1) * Cg_ * G_) + 1) * R_;
        uint64_t NCH =  (((N_       - 1) * Cg_ * G_) + 1) * H_;
        uint64_t NKOH = (((N_       - 1) * Kg_ * G_) + 1) * OH_;
        return KCR        < PowOf2<28>()
            && KCR  * S_  < PowOf2<28>()
            && NCH        < PowOf2<31>()
            && NCH  * W_  < PowOf2<31>()
            && NKOH       < PowOf2<31>()
            && NKOH * OW_ < PowOf2<31>();
        // clang-format on
    }

    bool paddedSizesFit16bits() const noexcept
    {
        // clang-format off
        return static_cast<int64_t>(pad_h) + H <= PowOf2<16, int64_t>()
            && static_cast<int64_t>(pad_w) + W <= PowOf2<16, int64_t>()
            && std::abs(static_cast<int64_t>(pad_h)) + out_h + R <= PowOf2<16, int64_t>()
            && std::abs(static_cast<int64_t>(pad_w)) + out_w + S <= PowOf2<16, int64_t>();
        // clang-format on
    }
};

} // namespace miopen
