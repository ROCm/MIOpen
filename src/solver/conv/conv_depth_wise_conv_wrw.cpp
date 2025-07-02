/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/datatype.hpp>
#include <ostream>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/array.hpp>
#include <ck/library/utility/device_memory.hpp>
#include <ck/library/utility/host_tensor.hpp>
#include <ck/library/utility/host_tensor_generator.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTH_WISE_CONV_WRW)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;
using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

// Hash combining utility
template <typename T>
inline void hash_combine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for arrays
template <typename T, ck::index_t N>
std::size_t hash_array(const ck::Array<T, N>& arr) {
    std::size_t seed = 0;
    for (const auto& elem : arr) {
        hash_combine(seed, elem);
    }
    return seed;
}

size_t GetStringHash(std::string str)
{
    std::hash<std::string> hasher;
    size_t hashValue = hasher(str);

    return hashValue;
}

struct CacheData
{
    size_t hashcode;
    size_t kernelhash;
    uint   split_k;
};

static std::unordered_map<size_t, CacheData> wrw_cache =
{
    {0x1d8e031b28fe3461, {0x1d8e031b28fe3461, 0x8ada6ba6c0a8262f, 0x20}},
    {0x0e92b755dc1549cc, {0x0e92b755dc1549cc, 0x9edc399f25f4e3cf, 0x1}},
    {0x37896c980651dd6a, {0x37896c980651dd6a, 0x3b6498916e34a84c, 0x4}},
    {0x25d1a9f7bf77cdb1, {0x25d1a9f7bf77cdb1, 0x9edc399f25f4e3cf, 0x1}},
    {0x2dcf0679e65a2176, {0x2dcf0679e65a2176, 0xb2b47d69d3efff96, 0x8}},
    {0x7cda3462205873e9, {0x7cda3462205873e9, 0x8caaa26caf0cf637, 0x4}},
    {0x4b30467a1ed12e94, {0x4b30467a1ed12e94, 0x634867055bc5ccb3, 0x10}},
    {0xe51eab72929667c7, {0xe51eab72929667c7, 0x7cdccadfae7e4912, 0x20}},
    {0x862a553bc1b6135f, {0x862a553bc1b6135f, 0x7cdccadfae7e4912, 0x20}},
    {0xc6075c461b1f4387, {0xc6075c461b1f4387, 0x18c034c0dd790bc9, 0x1}},
    {0xfcaf93b538f3b837, {0xfcaf93b538f3b837, 0x55335f15571bd2d2, 0x1}},
    {0xce0513f63f76d7e2, {0xce0513f63f76d7e2, 0xb62b34a2c839689b, 0x1}},
    {0x6de0f5c2e31130ad, {0x6de0f5c2e31130ad, 0x6b529345f00e682f, 0x1}},
    {0x7fd11c3a9a196bdb, {0x7fd11c3a9a196bdb, 0xd8cef6f43ec57670, 0x20}},
};

struct PerfArgs
{
    ck::index_t block_size;
    ck::index_t tile_w;
    ck::index_t tile_h;
    ck::index_t filter_size;
    ck::index_t dilation_w;
    ck::index_t dilation_h;
    ck::index_t stride_w;
    ck::index_t stride_h;
    ck::index_t pad_w;
    ck::index_t pad_h;
    ck::index_t n_batch;
    ck::index_t num_wave_per_tile;
    ck::index_t in_scalar_per_vector;
    ck::index_t out_scalar_per_vector;
    ck::index_t dst_scalar_per_vector;
    ck::index_t w_split;
    bool require_padding;
};

//                                              b_s, t_w, t_h, f_s, d_w, d_h, s_w, s_h, p_w, p_h, n_b, n_w, i_s, o_s, d_s, w_s,   r_p
static const std::vector<PerfArgs> perf_arg = {{256,  28,  28,   5,   1,   1,   1,   1,   2,   2,   2,   1,   2,   2,   2,   1,   false},
                                               {256,  14,  14,   5,   1,   1,   1,   1,   2,   2,   8,   1,   2,   2,   8,   1,   false},
                                               { 64,   7,   7,   5,   1,   1,   1,   1,   2,   2,  16,   1,   1,   1,   8,   1,   false},
                                               {128,  56,  56,   5,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   1,   false},
                                               {256,  14,  14,   5,   1,   1,   2,   2,   2,   2,   8,   1,   2,   1,   8,   1,   false},
                                               {256, 112, 112,   3,   1,   1,   1,   1,   1,   1,   1,   4,   8,   8,   1,   1,   false},
                                               {128,  56,  56,   3,   1,   1,   1,   1,   1,   1,   1,   2,   4,   4,   1,   1,   false},
                                               {256,  28,  28,   3,   1,   1,   1,   1,   1,   1,   2,   1,   4,   4,   2,   1,   false},
                                               { 64,  14,  14,   3,   1,   1,   1,   1,   1,   1,   4,   1,   2,   2,   4,   1,   false},
                                               { 64,   7,   7,   3,   1,   1,   1,   1,   1,   1,  16,   1,   1,   1,   8,   1,   false},
                                               {256, 112, 112,   3,   1,   1,   2,   2,   1,   1,   1,   4,   8,   4,   1,   1,   false},
                                               {256,  28,  28,   3,   1,   1,   2,   2,   1,   1,   2,   1,   4,   2,   2,   1,   false},
                                               // 28 x 5 x 1
                                               { 64,  28,  28,   5,   1,   1,   1,   1,   2,   2,   2,   1,   2,   2,   2,   1,   false},
                                               { 64,  28,  28,   5,   1,   1,   1,   1,   2,   2,   2,   1,   4,   4,   2,   1,   false},
                                               {256,  28,  28,   5,   1,   1,   1,   1,   2,   2,   2,   1,   4,   4,   2,   1,   false},
                                               // 14 x 5 x 1
                                               {256,  14,  14,   5,   1,   1,   1,   1,   2,   2,   4,   1,   2,   2,   4,   1,   false},
                                               {256,  14,  14,   5,   1,   1,   1,   1,   2,   2,   2,   1,   2,   2,   2,   1,   false},
                                               { 64,  14,  14,   5,   1,   1,   1,   1,   2,   2,   4,   1,   2,   2,   4,   1,   false},
                                               { 64,  14,  14,   5,   1,   1,   1,   1,   2,   2,   2,   1,   2,   2,   2,   1,   false},
                                               // 7 x 5 x 1
                                               { 64,   7,   7,   5,   1,   1,   1,   1,   2,   2,   8,   1,   1,   1,   8,   1,   false},
                                               { 64,   7,   7,   5,   1,   1,   1,   1,   2,   2,   4,   1,   1,   1,   4,   1,   false},
                                               { 64,   7,   7,   5,   1,   1,   1,   1,   2,   2,   2,   1,   1,   1,   2,   1,   false},
                                               // 56 x 5 x 2
                                               {128,  56,  56,   5,   1,   1,   2,   2,   2,   2,   2,   2,   4,   2,   2,   1,   false},
                                               {128,  56,  56,   5,   1,   1,   2,   2,   2,   2,   2,   2,   4,   2,   2,   2,   false},
                                               {128,  56,  56,   5,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   false},
                                               { 64,  56,  56,   5,   1,   1,   2,   2,   2,   2,   2,   1,   2,   2,   2,   2,   false},
                                               // 14 x 5 x 2
                                               {256,  14,  14,   5,   1,   1,   2,   2,   2,   2,   4,   1,   2,   1,   4,   1,   false},
                                               { 64,  14,  14,   5,   1,   1,   2,   2,   2,   2,   8,   1,   2,   1,   8,   1,   false},
                                               { 64,  14,  14,   5,   1,   1,   2,   2,   2,   2,   4,   1,   2,   1,   4,   1,   false},
                                               // 112 x 3 x 1
                                               {256, 112, 112,   3,   1,   1,   1,   1,   1,   1,   2,   4,   4,   4,   2,   2,   false},
                                               {256, 112, 112,   3,   1,   1,   1,   1,   1,   1,   2,   4,   2,   2,   2,   4,   false},
                                               {256, 112, 112,   3,   1,   1,   1,   1,   1,   1,   2,   4,   2,   2,   2,   2,   false},
                                               // 56 x 3 x 1
                                               {256,  56,  56,   3,   1,   1,   1,   1,   1,   1,   2,   4,   4,   4,   2,   1,   false},
                                               {128,  56,  56,   3,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   false},
                                               {128,  56,  56,   3,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   4,   false},
                                               {256,  56,  56,   3,   1,   1,   1,   1,   1,   1,   2,   4,   2,   2,   2,   2,   false},
                                               // 28 x 3 x 1
                                               {256,  28,  28,   3,   1,   1,   1,   1,   1,   1,   2,   1,   2,   2,   2,   1,   false},
                                               {128,  28,  28,   3,   1,   1,   1,   1,   1,   1,   2,   1,   4,   4,   2,   1,   false},
                                               { 64,  28,  28,   3,   1,   1,   1,   1,   1,   1,   2,   1,   4,   4,   2,   1,   false},
                                               // 14 x 3 x 1
                                               { 64,  14,  14,   3,   1,   1,   1,   1,   1,   1,   8,   1,   2,   2,   8,   1,   false},
                                               { 64,  14,  14,   3,   1,   1,   1,   1,   1,   1,   2,   1,   2,   2,   2,   1,   false},
                                               {128,  14,  14,   3,   1,   1,   1,   1,   1,   1,   4,   1,   2,   2,   4,   1,   false},
                                               {256,  14,  14,   3,   1,   1,   1,   1,   1,   1,   4,   1,   2,   2,   4,   1,   false},
                                               // 7 x 3 x 1
                                               { 64,   7,   7,   3,   1,   1,   1,   1,   1,   1,   8,   1,   1,   1,   8,   1,   false},
                                               { 64,   7,   7,   3,   1,   1,   1,   1,   1,   1,   4,   1,   1,   1,   4,   1,   false},
                                               { 64,   7,   7,   3,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1,   2,   1,   false},
                                               // 112 x 3 x 2
                                               {256, 112, 112,   3,   1,   1,   2,   2,   1,   1,   2,   4,   4,   2,   2,   2,   false},
                                               {256, 112, 112,   3,   1,   1,   2,   2,   1,   1,   2,   4,   2,   2,   2,   2,   false},
                                               {256, 112, 112,   3,   1,   1,   2,   2,   1,   1,   2,   4,   2,   2,   2,   4,   false},
                                               // 28 x 3 x 3
                                               { 64,  28,  28,   3,   1,   1,   2,   2,   1,   1,   2,   1,   4,   2,   2,   1,   false},
                                               {128,  28,  28,   3,   1,   1,   2,   2,   1,   1,   2,   1,   4,   2,   2,   1,   false},
                                               { 64,  28,  28,   3,   1,   1,   2,   2,   1,   1,   4,   1,   4,   2,   4,   1,   false},
                                               {128,  28,  28,   3,   1,   1,   2,   2,   1,   1,   4,   1,   4,   2,   4,   1,   false},
                                               {256,  28,  28,   3,   1,   1,   2,   2,   1,   1,   4,   1,   4,   2,   4,   1,   false}};

static std::string GetTypeString(const PerfArgs& arg)
{
    auto str = std::stringstream();

    // clang-format off
    str << "DeviceGroupedConvBwdWeightDlV4<"
        << 2 << ", "   // NDimSpatial
        << arg.block_size << ", "
        << "GNHWC" << ", "
        << "GKYXC" << ", "
        << "GNHWK" << ", "
        << "BlockTileSize<" << arg.tile_h << ", " << arg.tile_w << ">, "
        << "FilterSize<" << arg.filter_size << ","<< arg.filter_size << ">, "
        << "Dilation<" << arg.dilation_h << ", " << arg.dilation_w << ">, "
        << "Stride<" << arg.stride_h << ", " << arg.stride_w<< ">, "
        << "Pad<" << arg.pad_h << ", " << arg.pad_w<< ">, "
        << "NBatch: " << arg.n_batch << ", "
        << "NumWavePerTile: " << arg.num_wave_per_tile << ", "
        << "InScalarPerVector: " << arg.in_scalar_per_vector << ", "
        << "OutScalarPerVector: " << arg.out_scalar_per_vector << ", "
        << "DstScalarPerVector: " << arg.dst_scalar_per_vector << ", "
        << "RequirePadding: " << arg.require_padding << ", "
        << "WSplit: " << arg.w_split << ">"
        << std::endl;
    // clang-format on

    return str.str();
}

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        hi      = ProblemInterpreter::GetInputHeightHi(problem);
        wi      = ProblemInterpreter::GetInputWidthWi(problem);
        n       = ProblemInterpreter::GetBatchN(problem);
        k       = ProblemInterpreter::GetOutputChannelK(problem);
        c       = ProblemInterpreter::GetInputChannelC(problem);
        ho      = ProblemInterpreter::GetOutputHeightHo(problem);
        wo      = ProblemInterpreter::GetOutputWidthWo(problem);
        sy      = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
        sx      = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
        dy      = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
        dx      = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
        ply     = ProblemInterpreter::GetInputLeftPadH(problem);
        plx     = ProblemInterpreter::GetInputLeftPadW(problem);
        pry     = ProblemInterpreter::GetAdjustedInputRightPadH(problem);
        prx     = ProblemInterpreter::GetAdjustedInputRightPadW(problem);
        fy      = ProblemInterpreter::GetFilterHeightY(problem);
        fx      = ProblemInterpreter::GetFilterWidthX(problem);
        g       = ProblemInterpreter::GetGroupCountG(problem);
        di      = ProblemInterpreter::GetInputDepthDi(problem);
        dout    = ProblemInterpreter::GetOutputDepthDo(problem);
        z       = ProblemInterpreter::GetFilterDepthZ(problem);
        c_per_g = c / g;
        k_per_g = k / g;

        // in_lengths  = {g,                           n,                 c_per_g, hi,           wi     };
        // in_strides  = {n * hi * wi * c_per_g,       hi * wi * c_per_g, 1,       wi * c_per_g, c_per_g};
        // out_lengths = {g,                           n,                 k_per_g, ho,           wo     };
        // out_strides = {n * ho * wo * k_per_g,       ho * wo * k_per_g, 1,       wo * k_per_g, k_per_g};
        // wei_lengths = {g,                           k_per_g,           c_per_g, fy,           fx     };
        // wei_strides = {k_per_g * fy * fx * c_per_g, fy * fx * c_per_g, 1,       fx * c_per_g, c_per_g};

        in_lengths  = {g,                 n,                     c_per_g, hi,           wi     };
        in_strides  = {hi * wi * c_per_g, g * hi * wi * c_per_g, 1,       wi * c_per_g, c_per_g};
        out_lengths = {g,                 n,                     k_per_g, ho,           wo     };
        out_strides = {ho * wo * k_per_g, g * ho * wo * k_per_g, 1,       wo * k_per_g, k_per_g};
        wei_lengths = {g,                 k_per_g,               c_per_g, fy,           fx     };
        wei_strides = {fy * fx * c_per_g, g * fy * fx * c_per_g, 1,       fx * c_per_g, c_per_g};

        bias_lengths = {g, 1, k_per_g, 1, 1};
        bias_strides = {k_per_g, 0, 1, 0, 0};

        padding_left  = {ply, plx};
        padding_right = {pry, prx};
        stride        = {sy, sx};
        dilation      = {dy, dx};
    }

    size_t GetParamHash() const
    {
        size_t seed = 0;
        // Combine hashes of each parameter
        hash_combine(seed, hash_array(in_lengths));
        hash_combine(seed, hash_array(in_strides));
        hash_combine(seed, hash_array(out_lengths));
        hash_combine(seed, hash_array(out_strides));
        hash_combine(seed, hash_array(wei_lengths));
        hash_combine(seed, hash_array(wei_strides));
        hash_combine(seed, hash_array(bias_lengths));
        hash_combine(seed, hash_array(bias_strides));
        hash_combine(seed, hash_array(stride));
        hash_combine(seed, hash_array(dilation));
        hash_combine(seed, hash_array(padding_left));
        hash_combine(seed, hash_array(padding_right));

        ck::Array<ck::index_t, 5> others = {c, k, di, dout, z};
        hash_combine(seed, hash_array(others));

        return seed;
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;
    ~CKArgs()                        = default;

    // std::size_t GetFlops() const
    // {
    //     // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
    //     return static_cast<std::size_t>(2) * G * N * K * C *
    //         std::accumulate(std::next(std::begin(out_lens), 3),
    //                         std::end(out_lens),
    //                         static_cast<std::size_t>(1), std::multiplies<>()) *
    //         std::accumulate(std::next(std::begin(wei_lens), 3),
    //                         std::end(wei_lens),
    //                         static_cast<std::size_t>(1), std::multiplies<>());
    // }

        int hi;
        int wi;
        int n;
        int k;
        int c;
        int ho;
        int wo;
        int sy;
        int sx;
        int dy;
        int dx;
        int ply;
        int plx;
        int pry;
        int prx;
        int fy;
        int fx;
        int g;
        int di;
        int dout;
        int z;
        int c_per_g;
        int k_per_g;

        ck::Array<ck::index_t, 5> in_lengths;
        ck::Array<ck::index_t, 5> in_strides;
        ck::Array<ck::index_t, 5> out_lengths;
        ck::Array<ck::index_t, 5> out_strides;
        ck::Array<ck::index_t, 5> wei_lengths;
        ck::Array<ck::index_t, 5> wei_strides;

        ck::Array<ck::index_t, 5> bias_lengths;
        ck::Array<ck::index_t, 5> bias_strides;

        ck::Array<ck::index_t, 2> padding_left;
        ck::Array<ck::index_t, 2> padding_right;
        ck::Array<ck::index_t, 2> stride;
        ck::Array<ck::index_t, 2> dilation;
};

static bool IsSupportedArgument(const PerfArgs& arg,
                                const CKArgs& ck_arg,
                                const ck::index_t k_batch)
{
    constexpr ck::index_t spatial_offset = 3;
    constexpr ck::index_t wave_size = 64;
    // In
    const ck::index_t hi        = ck_arg.in_lengths[spatial_offset + 0];
    const ck::index_t wi        = ck_arg.in_lengths[spatial_offset + 1];
    const ck::index_t wi_stride = ck_arg.in_strides[spatial_offset + 1];
    const ck::index_t n         = ck_arg.in_lengths[1];
    // Out
    const ck::index_t wo        = ck_arg.out_lengths[spatial_offset + 1];
    const ck::index_t wo_stride = ck_arg.out_strides[spatial_offset + 1];
    // Wei
    const ck::index_t filter_y = ck_arg.wei_lengths[spatial_offset + 0];
    const ck::index_t filter_x = ck_arg.wei_lengths[spatial_offset + 1];
    const ck::index_t filter_k = ck_arg.wei_lengths[1];
    const ck::index_t filter_c = ck_arg.wei_lengths[2];

    const ck::index_t tile_h     = arg.tile_h;
    const ck::index_t tile_w     = arg.tile_w;
    const ck::index_t pad_h      = arg.pad_h;
    const ck::index_t pad_w      = arg.pad_w;
    const ck::index_t stride_h   = arg.stride_h;
    const ck::index_t stride_w   = arg.stride_w;
    const ck::index_t dilation_h = arg.dilation_h;
    const ck::index_t dilation_w = arg.dilation_w;

    const ck::index_t num_tile_per_block = arg.block_size / wave_size / arg.num_wave_per_tile;

    if(filter_k != 1 || filter_c != 1)
    {
        return false;
    }
    if(n % (k_batch * arg.n_batch * num_tile_per_block) != 0)
    {
        return false;
    }
    if (arg.require_padding == false)
    {
        if(hi != tile_h || wi != tile_w)
        {
            return false;
        }
    }
    if(filter_y != arg.filter_size || filter_x != arg.filter_size)
    {
        return false;
    }
    if(pad_h != ck_arg.padding_left[0] || pad_w != ck_arg.padding_left[1])
    {
        return false;
    }
    if(pad_h != ck_arg.padding_right[0])
    {
        if(stride_h == 2 && pad_h > 0)
        {
            ck::index_t alter_pad_h = pad_h;
            if((tile_h + pad_h + pad_h - ((arg.filter_size - 1) * dilation_h + 1)) % stride_h != 0)
            {
                alter_pad_h = pad_h - 1;
            }
            if(alter_pad_h != ck_arg.padding_right[0])
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    if(pad_w != ck_arg.padding_right[1])
    {
        if(stride_w == 2 && pad_w > 0)
        {
            ck::index_t alter_pad_w = pad_w;
            if((tile_w + pad_w + pad_w - ((arg.filter_size - 1) * dilation_w + 1)) % stride_w != 0)
            {
                alter_pad_w = pad_w - 1;
            }
            if(alter_pad_w != ck_arg.padding_right[1])
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    if(stride_h != ck_arg.stride[0] || stride_w != ck_arg.stride[1])
    {
        return false;
    }
    if(dilation_h != ck_arg.dilation[0] ||
        dilation_w != ck_arg.dilation[1])
    {
        return false;
    }
    if(arg.in_scalar_per_vector > 1)
    {
        if(wi % arg.in_scalar_per_vector != 0)
        {
            return false;
        }
        if(wi_stride != 1)
        {
            return false;
        }
    }
    if(arg.out_scalar_per_vector > 1)
    {
        if(wo % arg.out_scalar_per_vector != 0)
        {
            return false;
        }
        if(wo_stride != 1)
        {
            return false;
        }
    }
    return true;
}

static bool GetSupportedSolutionCount(const ProblemDescription& problem)
{
    const auto& ck_args = CKArgs{problem};
    uint solutionCount = 0;

    for (int i = 0; i < perf_arg.size(); i++)
    {
        bool is_supported = IsSupportedArgument(perf_arg[i],
                                                ck_args,
                                                1); // splitk
        if (is_supported)
        {
            solutionCount++;
        }
    }

    return solutionCount;
}

size_t ConvDepthWiseConvWrw::GetWorkspaceSize(const ExecutionContext&,
                                              const ProblemDescription& problem) const
{
    // Since the best split_k is not known here, always treating it > 1.
    auto ck_args = CKArgs{problem};

    return ck::math::integer_least_multiple(
                    sizeof(float) * ck_args.wei_lengths[0] * ck_args.wei_lengths[1] *
                    ck_args.wei_lengths[2] * ck_args.wei_lengths[3] *
                    ck_args.wei_lengths[4],
                    128);
}

bool ConvDepthWiseConvWrw::IsApplicable(const ExecutionContext& ctx,
                                        const ProblemDescription& problem) const
{
    if(!miopen::debug::AlwaysEnableConvDirectNaive)
    {
        if(env::disabled(MIOPEN_DEBUG_CONV_DEPTH_WISE_CONV_WRW))
            return false;
        if(!ctx.use_hip_kernels)
            return false;
    }

    if(!ConvDirectNaiveConvIsApplicableByKernelType(ctx, problem))
        return false;

    if(!problem.IsLayoutDefault() && !problem.IsLayoutNHWC())
        return false;

    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16() || problem.IsFp8() ||
         problem.IsBfp8()))
        return false;

    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
    {
        auto test_cast = [&](const TensorDescriptor& desc) {
            if(desc.GetCastType())
            {
                const auto cast_type = *desc.GetCastType();
                if(cast_type == miopenFloat8_fnuz || cast_type == miopenBFloat8_fnuz)
                    return false;
            }
            // all tested tensors must have cast type set
            return true;
        };
        if(test_cast(problem.GetIn()))
            return false;
        if(test_cast(problem.GetOut()))
            return false;
    }

    if (GetSupportedSolutionCount(problem) == 0)
        return false;

    return true;
}

InvokerFactory MakeImplDepthWiseConvWrwInvokerFactory(const ProblemDescription& problem,
                                                      const CKArgs& ck_args,
                                                      const ck::Array<ck::index_t, 5>& new_in_lengths,
                                                      const ck::Array<ck::index_t, 5>& new_out_lengths,
                                                      const ck::Array<ck::index_t, 5>& acc_strides,
                                                      const uint split_k)
{
    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            {
                if (split_k > 1)
                {
                    hipMemsetAsync(data_ctx.workSpace, 0, data_ctx.workSpaceSize, handle.GetStream());
                }
                handle.Run(kernels[0])(static_cast<const InDataType*>(tensors.x),
                                    static_cast<const WeiDataType*>(split_k > 1 ? nullptr : tensors.dw),
                                    static_cast<const OutDataType*>(tensors.dy),
                                    static_cast<const AccDataType*>(split_k > 1 ? data_ctx.workSpace : nullptr),
                                    split_k > 1 ? new_in_lengths : ck_args.in_lengths,
                                    ck_args.in_strides,
                                    ck_args.wei_lengths,
                                    split_k > 1 ? acc_strides : ck_args.wei_strides,
                                    split_k > 1 ? new_out_lengths : ck_args.out_lengths,
                                    ck_args.out_strides,
                                    split_k);
            }
            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
            }

            if (split_k > 1)
            {
                handle.Run(kernels[1])(tensors.dw,
                                    data_ctx.workSpace,
                                    ck_args.wei_strides,
                                    acc_strides);
            }

            if(handle.IsProfilingEnabled())
            {
                if (split_k > 1)
                {
                    elapsed += handle.GetKernelTime();
                }

                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

ConvSolution ConvDepthWiseConvWrw::GetSolution(const ExecutionContext& ctx,
                                               const ProblemDescription& problem) const
{
    ConvSolution result;

    if(problem.Is2d())
    {
        auto ck_args = CKArgs{problem};

        // TODO: choose the best perf_arg !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        uint best_perf_arg_index = perf_arg.size();

        // TODO: choose the best split_k !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        uint split_k = 1;
        float best_avg_time = 3.4e+30;

        // const size_t argsHash = ck_args.GetParamHash();
        // const auto it = wrw_cache.find(argsHash);

        // if (it != wrw_cache.end())
        // {
        //     const CacheData cd = it->second;

        //     split_k = cd.split_k;

        //     std::cout << "Find cached solution " << std::hex << std::setw(16) << std::setfill('0') << cd.hashcode
        //         << ", kernal hash:" << std::hex << std::setw(16) << std::setfill('0') << cd.kernelhash
        //         <<", split_k:"<< split_k << std::endl;;

        //     for (uint i = 0; i < perf_arg.size(); i++)
        //     {
        //         if (IsSupportedArgument(perf_arg[i], ck_args, split_k) && (GetStringHash(GetTypeString(perf_arg[i])) == cd.kernelhash))
        //         {
        //             best_perf_arg_index = i;
        //             std::cout << "Find best cached kernel!!!" << std::endl;
        //             break;
        //         }
        //     }
        // }
        // else
        // {
        //     for (uint i = 0; i < perf_arg.size(); i++)
        //     {
        //         if (IsSupportedArgument(perf_arg[i], ck_args, split_k))
        //         {
        //             best_perf_arg_index = i;
        //             std::cout << "best_perf_arg_index = " << best_perf_arg_index << ", split_k = " << split_k << std::endl;
        //             break;
        //         }
        //     }
        // }

        // if (best_perf_arg_index == perf_arg.size())
        // {
        //     std::cout << "Argument is not supported!!!" << std::endl;
        //     return result;
        // }
        std::array<uint, 6> split_k_array = {1, 2, 4, 8, 16, 32};
        uint cold_niters = 5;
        uint n_repeat = 10;

        for (uint i = 0; i < perf_arg.size(); i++)
        {
            for (uint j = 0; j < split_k_array.size(); j++)
            {
                if (IsSupportedArgument(perf_arg[i], ck_args, split_k_array[j]))
                {
                    std::vector<KernelInfo> params;

                    ck::Array<ck::index_t, 5> acc_strides = {ck_args.wei_lengths[1] * ck_args.wei_lengths[2] * ck_args.wei_lengths[3] * ck_args.wei_lengths[4],
                                                            ck_args.wei_lengths[2] * ck_args.wei_lengths[3] * ck_args.wei_lengths[4],
                                                            ck_args.wei_lengths[3] * ck_args.wei_lengths[4],
                                                            ck_args.wei_lengths[4],
                                                            1};
                    ck::Array<ck::index_t, 5> new_in_lengths = ck_args.in_lengths;
                    ck::Array<ck::index_t, 5> new_out_lengths = ck_args.out_lengths;
                    new_in_lengths.At(1) /= split_k_array[j];
                    new_out_lengths.At(1) /= split_k_array[j];

                    size_t block_size0 = perf_arg[i].block_size;
                    size_t grid_size0  = static_cast<size_t>(ck_args.in_lengths[0]);

                    KernelInfo kernel0_info, kernel1_info;

                    kernel0_info.kernel_file = "device_grouped_conv_bwd_weight_dl_v4.cpp";
                    kernel0_info.kernel_name = "kernel_grouped_conv_bwd_weight_dl_v4_run";

                    kernel0_info.l_wk = {block_size0, 1, 1};
                    kernel0_info.g_wk = {grid_size0 * block_size0, split_k_array[j], 1};

                    kernel0_info.comp_options = ck_utility::get_ck_common_compiler_flag(ctx.GetStream())
                        + ctx.general_compile_options
                        + " -DCK_PARAM_BLOCKSIZE=" + std::to_string(perf_arg[i].block_size)
                        + " -DCK_PARAM_TILE_W=" + std::to_string(perf_arg[i].tile_w)
                        + " -DCK_PARAM_TILE_H=" + std::to_string(perf_arg[i].tile_h)
                        + " -DCK_PARAM_FILTERSIZE=" + std::to_string(perf_arg[i].filter_size)
                        + " -DCK_PARAM_PROBLEM_CONV_DILATION_W=" + std::to_string(perf_arg[i].dilation_w)
                        + " -DCK_PARAM_PROBLEM_CONV_DILATION_H=" + std::to_string(perf_arg[i].dilation_h)
                        + " -DCK_PARAM_PROBLEM_CONV_STRIDE_W=" + std::to_string(perf_arg[i].stride_w)
                        + " -DCK_PARAM_PROBLEM_CONV_STRIDE_H=" + std::to_string(perf_arg[i].stride_h)
                        + " -DCK_PARAM_PROBLEM_CONV_PAD_W=" + std::to_string(perf_arg[i].pad_w)
                        + " -DCK_PARAM_PROBLEM_CONV_PAD_H=" + std::to_string(perf_arg[i].pad_h)
                        + " -DCK_PARAM_NBATCH=" + std::to_string(perf_arg[i].n_batch)
                        + " -DCK_PARAM_NUMWAVEPERTILE=" + std::to_string(perf_arg[i].num_wave_per_tile)
                        + " -DCK_PARAM_INSCALARPERVECTOR=" + std::to_string(perf_arg[i].in_scalar_per_vector)
                        + " -DCK_PARAM_OUTSCALARPERVECTOR=" + std::to_string(perf_arg[i].out_scalar_per_vector)
                        + " -DCK_PARAM_DSTSCALARPERVECTOR=" + std::to_string(perf_arg[i].dst_scalar_per_vector)
                        + " -DCK_PARAM_REQUIREPADDING=" + std::to_string(perf_arg[i].require_padding)
                        + " -DCK_PARAM_WSPLIT=" + std::to_string(perf_arg[i].w_split)
                        ;

                    kernel1_info.kernel_file = "device_grouped_conv_bwd_weight_dl_v4.cpp";
                    kernel1_info.kernel_name = "kernel_grouped_conv_bwd_weight_elementwise_run";

                    kernel1_info.comp_options = kernel0_info.comp_options;

                    size_t block_size1 = perf_arg[i].filter_size * perf_arg[i].filter_size;
                    size_t grid_size1  = ck_args.in_lengths[0];

                    kernel1_info.l_wk = {block_size1, 1, 1};
                    kernel1_info.g_wk = {grid_size1 * block_size1, 1, 1};

                    size_t workspace_sz = GetWorkspaceSize(ctx, problem);

                    InvokerFactory factory = MakeImplDepthWiseConvWrwInvokerFactory(problem,
                                                                                    ck_args,
                                                                                    new_in_lengths,
                                                                                    new_out_lengths,
                                                                                    acc_strides,
                                                                                    split_k_array[j]);

                    params.push_back(kernel0_info);
                    if (split_k_array[j] > 1)
                    {
                        params.push_back(kernel1_info);
                    }

                    // Start runtime tuning.
                    Tensor<InDataType> in_g_n_c_wis(std::initializer_list<ck::index_t>{ck_args.g, ck_args.n, ck_args.c_per_g, ck_args.hi, ck_args.wi});
                    Tensor<WeiDataType> wei_g_k_c_xs(std::initializer_list<ck::index_t>{ck_args.g, ck_args.k_per_g, ck_args.c_per_g, ck_args.fy, ck_args.fx});
                    Tensor<OutDataType> out_g_n_k_wos(std::initializer_list<ck::index_t>{ck_args.g, ck_args.n, ck_args.k_per_g, ck_args.ho, ck_args.wo});

                    in_g_n_c_wis.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 0.2});
                    out_g_n_k_wos.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.1, 0.1});

                    DeviceMem in_device_buf(sizeof(InDataType)   * in_g_n_c_wis.mDesc.GetElementSpaceSize());
                    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_g_k_c_xs.mDesc.GetElementSpaceSize());
                    DeviceMem out_device_buf(sizeof(OutDataType) * out_g_n_k_wos.mDesc.GetElementSpaceSize());
                    DeviceMem workSpace_device_buf(workspace_sz);

                    in_device_buf.ToDevice(in_g_n_c_wis.mData.data());
                    out_device_buf.ToDevice(out_g_n_k_wos.mData.data());

                    auto& handle = ctx.GetStream();
                    auto invoker = handle.PrepareInvoker(factory, params);
                    const auto invoke_ctx = miopen::conv::WrWInvokeParams{{problem.GetOut(),
                                                                            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                                                            problem.GetIn(),
                                                                            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                                                            problem.GetWeights(),
                                                                            static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer())},
                                                                            static_cast<AccDataType*>(workSpace_device_buf.GetDeviceBuffer()),
                                                                            workspace_sz,
                                                                            false};

                    float avg_time = 0.0f;

                    // Warm up.
                    for (uint m = 0; m < cold_niters; m++)
                    {
                        invoker(handle, invoke_ctx);
                    }

                    // Benchmark.
                    for (uint n = 0; n < n_repeat; n++)
                    {
                        invoker(handle, invoke_ctx);
                        avg_time += handle.GetKernelTime();
                    }

                    avg_time /= n_repeat;

                    std::cout << "current_index = " << i << ", split_k = " << split_k_array[j] << ", avg_time = " << avg_time << std::endl;

                    if (best_avg_time > avg_time)
                    {
                        best_perf_arg_index = i;
                        split_k = split_k_array[j];
                        best_avg_time = avg_time;
                        result.construction_params.clear();
                        result.construction_params.push_back(kernel0_info);
                        if (split_k > 1)
                        {
                            result.construction_params.push_back(kernel1_info);
                        }
                        result.workspace_sz = workspace_sz;
                        result.invoker_factory = MakeImplDepthWiseConvWrwInvokerFactory(problem,
                                                                                        ck_args,
                                                                                        new_in_lengths,
                                                                                        new_out_lengths,
                                                                                        acc_strides,
                                                                                        split_k);
                    }
                }
            }
        }
        std::cout << "best_perf_arg_index = " << best_perf_arg_index << ", split_k = " << split_k << std::endl;
    }
    else
    {
        // result = conv_internal::GetConv3DWRWSolution(ctx, problem);
    }
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
