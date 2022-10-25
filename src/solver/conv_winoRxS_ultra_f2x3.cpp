/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/sequences.hpp>
#include <miopen/stringutils.hpp>

#include <boost/any.hpp>

#include <tuple>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_ULTRA_RXS_F2X3_CONST)

namespace miopen {
namespace solver {

namespace {

constexpr uint64_t group_size = 64;
constexpr uint64_t o_tile_W   = 2;
constexpr uint64_t o_tile_H   = 2;
constexpr uint64_t d_tile_W   = 4;
constexpr uint64_t d_tile_H   = 4;

// step is alwas based on the output tile size
constexpr uint64_t o_tile_step_W = o_tile_W;
constexpr uint64_t o_tile_step_H = o_tile_H;
constexpr uint64_t d_tile_step_W = o_tile_W;
constexpr uint64_t d_tile_step_H = o_tile_H;

//
// Number of tile lanes (QWORDs for packed clip bits)
//
constexpr uint64_t d_clip_tiles_QW = group_size * d_tile_W / (sizeof(uint64_t) * CHAR_BIT);
constexpr uint64_t o_clip_tiles_QW = group_size * o_tile_W / (sizeof(uint64_t) * CHAR_BIT);

struct work_info
{
    int64_t d_load_offset_addr;
    int64_t o_store_offset_addr;
    uint64_t step_1_pos;
    uint64_t step_2_pos;
    uint64_t d_clip[d_clip_tiles_QW][d_tile_H];
    uint64_t o_clip[o_clip_tiles_QW][o_tile_H];
};

enum struct flush_control
{
    FLUSH_NONE,
    FLUSH_N,
};

struct WinogradUltraDescription
{
    uint32_t N;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t K;
    uint32_t R;
    uint32_t S;
    int32_t pad_H;
    int32_t pad_W;
    uint32_t out_H;
    uint32_t out_W;
    uint32_t d_N_pitch;
    uint32_t d_C_pitch;
    uint32_t d_H_pitch;
    uint32_t d_W_pitch;
    uint32_t o_N_pitch;
    uint32_t o_K_pitch;
    uint32_t o_H_pitch;
    uint32_t o_W_pitch;
    int32_t d_step_1_pitch;
    int32_t d_step_2_pitch;
    int32_t o_step_1_pitch;
    int32_t o_step_2_pitch;
    flush_control flush;
    uint32_t flags;

    WinogradUltraDescription() = delete;

    WinogradUltraDescription(const ProblemDescription& problem)
    {
        constexpr unsigned F_REVERSE_R = 1 << 0;
        constexpr unsigned F_REVERSE_S = 1 << 1;
        constexpr unsigned F_FLIP_K_C  = 1 << 2;

        const auto desc = UnifiedDescriptionConv2d(problem);
        N               = desc.N;
        C               = desc.C;
        K               = desc.K;
        out_H           = desc.out_h;
        out_W           = desc.out_w;
        R               = desc.R;
        S               = desc.S;
        pad_H           = desc.pad_h;
        pad_W           = desc.pad_w;

        flags = problem.direction.IsForward() ? 0 : F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
        H     = problem.in_height;
        W     = problem.in_width;
        C     = C / problem.group_counts;
        K     = K / problem.group_counts;

        const auto d_buf = BuffInfo(GetGroupConvLayout(GetMemLayout_t(problem.in_layout), true),
                                    N,
                                    C,
                                    H,
                                    W,
                                    problem.group_counts,
                                    GetTypeSize(problem.in_data_type));
        const auto o_buf = BuffInfo(GetGroupConvLayout(GetMemLayout_t(problem.out_layout), true),
                                    N,
                                    K,
                                    out_H,
                                    out_W,
                                    problem.group_counts,
                                    GetTypeSize(problem.out_data_type));

        const unsigned tiles_n_row    = (out_W + o_tile_step_W - 1) / o_tile_step_W;
        const unsigned tiles_n_column = (out_H + o_tile_step_H - 1) / o_tile_step_H;

        d_N_pitch = d_buf.byte_stride.nk;
        d_C_pitch = d_buf.byte_stride.c;
        d_H_pitch = d_buf.byte_stride.h;
        d_W_pitch = d_buf.byte_stride.w;

        d_step_1_pitch = d_tile_step_H * d_H_pitch - tiles_n_row * d_tile_step_W * d_W_pitch;
        d_step_2_pitch = d_N_pitch - tiles_n_column * d_tile_step_H * d_H_pitch;

        o_N_pitch = o_buf.byte_stride.nk;
        o_K_pitch = o_buf.byte_stride.c;
        o_H_pitch = o_buf.byte_stride.h;
        o_W_pitch = o_buf.byte_stride.w;

        o_step_1_pitch = o_tile_step_H * o_H_pitch - tiles_n_row * o_tile_step_W * o_W_pitch;
        o_step_2_pitch = o_N_pitch - tiles_n_column * o_tile_step_H * o_H_pitch;

        flush = d_step_2_pitch >= std::pow(2, 23)   ? flush_control::FLUSH_N
                : o_step_2_pitch >= std::pow(2, 23) ? flush_control::FLUSH_N
                                                    : flush_control::FLUSH_NONE;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const WinogradUltraDescription& desc)
{
    // clang-format off
        stream << "N=" << desc.N << " C=" << desc.C << " H=" << desc.H << " W=" << desc.W
               << " K=" << desc.K << " R=" << desc.R << " S=" << desc.S 
               << " pad_H=" << desc.pad_H << " pad_W=" << desc.pad_W
               << " out_H=" << desc.out_H << " out_W=" << desc.out_W
               << " d_N_pitch=" << desc.d_N_pitch << " d_C_pitch=" << desc.d_C_pitch
               << " d_H_pitch=" << desc.d_H_pitch << " d_W_pitch=" << desc.d_W_pitch
               << " o_N_pitch=" << desc.o_N_pitch << " o_K_pitch=" << desc.o_K_pitch
               << " o_H_pitch=" << desc.o_H_pitch << " o_W_pitch=" << desc.o_W_pitch
               << " d_step_1_pitch=" << desc.d_step_1_pitch
               << " d_step_2_pitch=" << desc.d_step_2_pitch
               << " o_step_1_pitch=" << desc.o_step_1_pitch
               << " o_step_2_pitch=" << desc.o_step_2_pitch
               << " flags=" << desc.flags;
    // clang-format on

    stream << " flush=";
    switch(desc.flush)
    {
    case flush_control::FLUSH_N: stream << "FLUSH_N"; break;
    case flush_control::FLUSH_NONE: stream << "FLUSH_NONE"; break;
    }

    return stream;
}

inline void WU_control_make_3x3_w_info(const WinogradUltraDescription& desc,
                                       std::vector<work_info>& w_info)
{
    //
    // We assume the filter position is controlled by the LEFT pads and output sizes only here
    //
    // If the output size needed to be conputed based on the input size, filter size and left/right
    // pads, it is supposed to be done somewhere outside
    //

    int64_t o_cur_w = 0;
    int64_t o_cur_h = 0;
    int64_t cur_n   = 0;

    while((o_cur_w < desc.out_W) && (o_cur_h < desc.out_H) && (cur_n < desc.N))
    {
        bool flush_tail   = false;
        work_info cur_w_i = {};
        int64_t d_cur_w   = o_cur_w - desc.pad_W;
        int64_t d_cur_h   = o_cur_h - desc.pad_H;

        cur_w_i.d_load_offset_addr =
            d_cur_w * desc.d_W_pitch + d_cur_h * desc.d_H_pitch + cur_n * desc.d_N_pitch;
        cur_w_i.o_store_offset_addr =
            o_cur_w * desc.o_W_pitch + o_cur_h * desc.o_H_pitch + cur_n * desc.o_N_pitch;

        for(unsigned n_tile = 0; n_tile < group_size; n_tile++)
        {
            for(unsigned i = 0; i < d_tile_W; i++)
            {
                for(unsigned j = 0; j < d_tile_H; j++)
                {
                    unsigned k = n_tile * d_tile_W / (sizeof(uint64_t) * CHAR_BIT);

                    // clang-format off
                    cur_w_i.d_clip[k][j] <<= 1;
                    cur_w_i.d_clip[k][j] |= static_cast<uint64_t>(
                                            (d_cur_w + i < 0) || (desc.W <= d_cur_w + i) ||
                                            (d_cur_h + j < 0) || (desc.H <= d_cur_h + j) ||
                                            (cur_n < 0) || (desc.N <= cur_n) || flush_tail);
                    // clang-format on
                }
            }
            for(unsigned i = 0; i < o_tile_W; i++)
            {
                for(unsigned j = 0; j < o_tile_H; j++)
                {
                    unsigned k = n_tile * o_tile_W / (sizeof(uint64_t) * CHAR_BIT);

                    // clang-format off
                    cur_w_i.o_clip[k][j] <<= 1;
                    cur_w_i.o_clip[k][j] |= static_cast<uint64_t>(
                                            (o_cur_w + i < 0) || (desc.out_W <= o_cur_w + i) ||
                                            (o_cur_h + j < 0) || (desc.out_H <= o_cur_h + j) ||
                                            (cur_n < 0) || (desc.N <= cur_n) || flush_tail);
                    // clang-format on
                }
            }

            cur_w_i.step_1_pos <<= 1;
            cur_w_i.step_2_pos <<= 1;

            if(!flush_tail)
            {
                d_cur_w += d_tile_step_W;
                o_cur_w += o_tile_step_W;

                if(desc.out_W <= o_cur_w)
                {
                    cur_w_i.step_1_pos |= 1;

                    o_cur_w = 0;
                    d_cur_w = o_cur_w - desc.pad_W;

                    o_cur_h += o_tile_step_H;
                    d_cur_h += d_tile_step_H;
                }
                if(desc.out_H <= o_cur_h)
                {
                    cur_w_i.step_2_pos |= 1;

                    o_cur_h = 0;
                    d_cur_h = o_cur_h - desc.pad_H;

                    cur_n += 1;

                    if(desc.flush == flush_control::FLUSH_N)
                        flush_tail = true;
                }
            }
        }
        w_info.push_back(cur_w_i);
    }
}

inline void WU_control_w_info_bit_encode(const std::vector<work_info>& w_info,
                                         std::vector<uint32_t>& gpu_control)
{
    for(auto i = 0; i < w_info.size(); i++)
    {
        std::array<uint32_t, group_size> block = {0};
        work_info w_i                          = w_info[i];

        for(auto j = 0; j < 32; j++)
        {
            uint64_t qword;
            bool bit_reverse;

            if(j == 0)
            {
                qword       = w_i.d_load_offset_addr;
                bit_reverse = false;
            }
            else if(j == 1)
            {
                qword       = w_i.o_store_offset_addr;
                bit_reverse = false;
            }
            else if(j == 2)
            {
                qword       = w_i.step_1_pos;
                bit_reverse = true;
            }
            else if(j == 3)
            {
                qword       = w_i.step_2_pos;
                bit_reverse = true;
            }
            else if(j >= 4 && j < 4 + d_clip_tiles_QW * d_tile_H)
            {
                unsigned k  = j - 4;
                qword       = w_i.d_clip[k / d_tile_H][k % d_tile_H];
                bit_reverse = true;
            }
            else if(j >= 4 + d_clip_tiles_QW * d_tile_H &&
                    j < 4 + d_clip_tiles_QW * d_tile_H + o_clip_tiles_QW * o_tile_H)
            {
                unsigned k  = j - 4 - d_clip_tiles_QW * d_tile_H;
                qword       = w_i.o_clip[k / o_tile_H][k % o_tile_H];
                bit_reverse = true;
            }
            else if(j == 24)
            {
                qword       = i;
                bit_reverse = false;
            }
            else
            {
                qword       = 0;
                bit_reverse = false;
            }

            for(auto k = 0; k < group_size; k++)
            {
                auto idx = bit_reverse ? group_size - 1 - k : k;
                block[idx] <<= 1;
                block[idx] |= (qword & 1);
                qword >>= 1;
            }
        }

        gpu_control.insert(gpu_control.end(), block.begin(), block.end());
    }
}

inline void WU_control_make_3x3(const WinogradUltraDescription& desc,
                                std::vector<uint32_t>& gpu_control,
                                unsigned n_groups,
                                unsigned intl_factor)
{
    std::vector<work_info> w_info;
    WU_control_make_3x3_w_info(desc, w_info);

    std::vector<work_info> w_info_intl;
    for(int i = 0; i < w_info.size(); i += intl_factor * n_groups)
        for(int k = 0; k < intl_factor; k++)
            for(int j = k; j < intl_factor * n_groups && i + j < w_info.size(); j += intl_factor)
                w_info_intl.push_back(w_info[i + j]);

    WU_control_w_info_bit_encode(w_info_intl, gpu_control);
}

#if MIOPEN_BACKEND_HIP
inline bool IsShaderContraintsMet(const int R,
                                  const int S,
                                  const int,
                                  const int,
                                  const int C,
                                  const int K,
                                  const int H,
                                  const int W,
                                  const int OH,
                                  const int OW,
                                  const int,
                                  const ExecutionContext& ctx,
                                  const ProblemDescription& problem)
{
    // Padding for bwd data shall not be negative.
    /// \todo Either remove WrW related code or re-use function from RxS
    if(problem.direction.IsBackwardData())
    {
        if(!(0 <= problem.GetBackwardPadW() && problem.GetBackwardPadW() < std::pow(2, 16)))
            return false;
        if(!(0 <= problem.GetBackwardPadH() && problem.GetBackwardPadH() < std::pow(2, 16)))
            return false;
    }
    const auto grid_workgroup_count_x = ctx.GetStream().GetMaxHardwareComputeUnits();
    if(!problem.IsLayoutDefault())
    {
        return false;
    }

    constexpr auto ELEM_SZ    = static_cast<size_t>(sizeof(half_float::half));
    constexpr auto D_W_PITCH  = ELEM_SZ * 1;
    constexpr auto O_W_PITCH  = ELEM_SZ * 1;
    const auto D_H_PITCH      = D_W_PITCH * W;
    const auto O_H_PITCH      = O_W_PITCH * OW;
    const auto D_C_PITCH      = D_H_PITCH * H;
    const auto O_K_PITCH      = O_H_PITCH * OH;
    const auto D_N_PITCH      = D_C_PITCH * C;
    const auto O_N_PITCH      = O_K_PITCH * K;
    const auto TILES_N_ROW    = (OW + o_tile_step_W - 1) / o_tile_step_W;
    const auto TILES_N_COLUMN = (OH + o_tile_step_H - 1) / o_tile_step_H;

    const int64_t D_STEP_1_PITCH =
        d_tile_step_H * D_H_PITCH - TILES_N_ROW * d_tile_step_W * D_W_PITCH;
    const int64_t O_STEP_1_PITCH =
        o_tile_step_H * O_H_PITCH - TILES_N_ROW * o_tile_step_W * O_W_PITCH;
    const int64_t D_STEP_2_PITCH = D_N_PITCH - TILES_N_COLUMN * d_tile_step_H * D_H_PITCH;
    const int64_t O_STEP_2_PITCH = O_N_PITCH - TILES_N_COLUMN * o_tile_step_H * O_H_PITCH;

    // clang-format off
    return C <= 240
        && K <= 16
        && S <= 3
        && R <= 3
        && D_H_PITCH < std::pow(2, 16)
        && O_H_PITCH < std::pow(2, 16)
        && D_C_PITCH < std::pow(2, 30)
        && O_K_PITCH < std::pow(2, 30)
        && D_STEP_1_PITCH < std::pow(2, 18)
        && O_STEP_1_PITCH < std::pow(2, 18)
        && D_STEP_2_PITCH < std::pow(2, 30)
        && O_STEP_2_PITCH < std::pow(2, 30)
        && grid_workgroup_count_x < std::pow(2, 16);
    // clang-format on
}
#endif

template <typename T>
void* CopyDataToSymbol(const Handle& handle,
                       const Kernel& kernel,
                       const std::vector<T>& data,
                       const std::string& name)
{
#if MIOPEN_BACKEND_HIP
    const auto module    = kernel.program.GetModule();
    const auto data_size = data.size() * sizeof(T);

    size_t dev_buf_sz;
    hipDeviceptr_t dev_buf_ptr;
    auto status = hipModuleGetGlobal(&dev_buf_ptr, &dev_buf_sz, module, name.c_str());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed hip to get control_buf info");

    if(dev_buf_sz < data_size)
        MIOPEN_THROW("Buffer size is than required");

    handle.Copy(static_cast<const void*>(data.data()), dev_buf_ptr, data_size);

    return static_cast<void*>(dev_buf_ptr);
#else
    std::ignore = handle;
    std::ignore = kernel;
    std::ignore = data;
    std::ignore = name;
    return nullptr;
#endif
}

} // namespace

bool ConvBinWinogradUltraRxSf2x3Const::IsApplicable(const ExecutionContext& ctx,
                                                    const ProblemDescription& problem) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_ULTRA_RXS_F2X3_CONST{}))
        return false;

#if MIOPEN_BACKEND_HIP
    if(!problem.Is2d())
        return false;
    if(!problem.IsFp16())
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!ctx.rmv.IsV3())
        return false;
    if(problem.direction.IsBackwardWrW())
        return false;

    const auto name = ctx.GetStream().GetDeviceName();
    if(!StartsWith(name, "gfx10"))
        return false;

    // clang-format off
    if (!( problem.kernel_stride_w == 1
        && problem.kernel_stride_w == problem.kernel_stride_h
        && problem.kernel_dilation_w == 1
        && problem.kernel_dilation_h == 1
        && problem.bias == 0
        && problem.group_counts == 1
        && problem.in_layout == "NCHW"))
        return false;
    // clang-format on

    const auto n_inputs_per_group  = problem.n_inputs / problem.group_counts,
               n_outputs_per_group = problem.n_outputs / problem.group_counts;

    return IsShaderContraintsMet(problem.kernel_size_h, // RxS
                                 problem.kernel_size_w,
                                 problem.kernel_stride_h,
                                 problem.kernel_stride_w,
                                 n_inputs_per_group,  // C
                                 n_outputs_per_group, // K
                                 problem.in_height,   // HxW
                                 problem.in_width,
                                 problem.out_height, // OHxOW
                                 problem.out_width,
                                 problem.batch_sz, // N
                                 ctx,
                                 problem);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution ConvBinWinogradUltraRxSf2x3Const::GetSolution(const ExecutionContext& ctx,
                                                           const ProblemDescription& problem) const
{
    const unsigned n_groups = ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto desc         = WinogradUltraDescription(problem);
    const auto intl_factor  = 1;

    uint64_t reserved_offset = 0;
    int* reserved_ptr        = nullptr;
    float relu_alpha         = 1.0;

    std::vector<uint32_t> control_buf;
    WU_control_make_3x3(desc, control_buf, n_groups, intl_factor);

    const unsigned n_works      = control_buf.size() / group_size;
    const size_t control_buf_sz = control_buf.size() * sizeof(decltype(control_buf)::value_type);

    const size_t wg_size = 256;

    KernelInfo kernel;

    kernel.g_wk.push_back(wg_size * n_groups * problem.group_counts);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const std::string kernel_name    = "miopenSp3AsmConv_Ultra_v1_0_14_gfx10";
    const std::string kernel_file    = "Conv_Winograd_Ultra_v1_0_14";
    const std::string kernel_postfix = "_fp16_pk_stride1";

    std::stringstream ss;
    // clang-format off
    ss << "_" << desc.N << "x" << desc.H << "x" << desc.W
       << "_" << desc.out_H << "x" << desc.out_W
       << "_" << desc.pad_H << "x" << desc.pad_W
       << "_" << n_groups << "_" << intl_factor;
    // clang-format on

    kernel.kernel_name = kernel_name + kernel_postfix + ss.str();
    kernel.kernel_file = kernel_file + kernel_postfix + ".s";

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
        {"control_buf_alloc_type", 1},
        {kbp::Option{}, "mcumode"},
        {kbp::Option{}, "mwavefrontsize64"},

        // Control buffer is part of kernel and depends on the following parameters.
        // The following unused compile options ensure the uniqueness of the control buffer
        // and the correct program caching.
        {"control_buf_sz", control_buf_sz},
        {"hash_N", desc.N},
        {"hash_H", desc.H},
        {"hash_W", desc.W},
        {"hash_out_H", desc.out_H},
        {"hash_out_W", desc.out_H},
        {"hash_pad_H", desc.pad_H},
        {"hash_pad_W", desc.pad_W},
        {"hash_n_groups", n_groups},
        {"hash_intl_factor", intl_factor}};
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    ConvSolution solution;

    solution.construction_params.push_back(kernel);

    solution.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels.front();
        const auto control_buf_ptr =
            CopyDataToSymbol(ctx.GetStream(), kern, control_buf, "control_buf");

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& invoke_params = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& tensors       = invoke_params.tensors;

            MIOPEN_LOG_I2(desc << " n_groups=" << n_groups << " n_works=" << n_works);

            handle.Run(kern)(desc.C,
                             desc.K,
                             n_groups,
                             n_works,
                             desc.d_C_pitch,
                             desc.d_H_pitch,
                             desc.d_step_1_pitch,
                             desc.d_step_2_pitch,
                             desc.o_K_pitch,
                             desc.o_H_pitch,
                             desc.o_step_1_pitch,
                             desc.o_step_2_pitch,
                             tensors.in,
                             tensors.out,
                             control_buf_ptr,
                             tensors.w,
                             reserved_ptr, // Unused bias_addr.
                             relu_alpha,
                             desc.flags,
                             desc.R,
                             desc.S,
                             reserved_offset,
                             reserved_offset,
                             reserved_offset,
                             reserved_offset,
                             reserved_offset);
        };
    };

    return solution;
}

} // namespace solver
} // namespace miopen
