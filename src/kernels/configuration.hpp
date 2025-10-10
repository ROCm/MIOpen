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

// This file will read the environment variables (pre-defined macros during compiling)
// and transfer them into config structs
#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "default_configurations.hpp"

#include "vector_types.hpp"
#include "bfloat16_dev.hpp"
#include "miopen_type_traits.hpp"

// miopen root configs
namespace miopen {

enum class type_strategy : int
{
    fp16,
    fp32,
    fpmix,
    bfpmix,
};

enum class neuron_op_type : int
{
    pasthru      = 0, // x
    relu         = 3, // max(0, x)
    clipped_relu = 7, // min(alpha, max(0, x))
    clamp        = 10,
    total        = 11,
};

namespace detail {

template <int LayoutNHWC,
          int SaveMeanVariance,
          int RunningResult,
          int UseFp16,
          int UseFp32,
          int UseFpmix,
          int UseBfpmix,
          int UseAMDGCN,
          int NrnOpId>
struct proto_config
{
    static_assert(LayoutNHWC == 0 || LayoutNHWC == 1,
                  "LayoutNHWC (MIO_LAYOUT_NHWC) must be 0 or 1");
    static_assert(SaveMeanVariance == 0 || SaveMeanVariance == 1,
                  "SaveMeanVariance must be 0 or 1");
    static_assert(RunningResult == 0 || RunningResult == 1, "SaveMeanVariance must be 0 or 1");
    static_assert(UseFp16 == 0 || UseFp16 == 1, "UseFp16 must be 0 or 1");
    static_assert(UseFp32 == 0 || UseFp32 == 1, "UseFp32 must be 0 or 1");
    static_assert(UseFpmix == 0 || UseFpmix == 1, "UseFpmix must be 0 or 1");
    static_assert(UseBfpmix == 0 || UseBfpmix == 1, "UseBfpmix must be 0 or 1");
    static_assert((UseFp16 + UseFp32 + UseFpmix + UseBfpmix) == 1,
                  "only one of these configs can and must be chosen.");
    static_assert(UseAMDGCN == 0 || UseAMDGCN == 1, "UseAMDGCN must be 0 or 1");
    static_assert(NrnOpId >= 0 && NrnOpId <= 10,
                  "NrnOpId can only be interger between 0-10 (inclusive)");

    static constexpr bool layout_nhwc        = static_cast<bool>(LayoutNHWC);
    static constexpr bool save_mean_variance = static_cast<bool>(SaveMeanVariance);
    static constexpr bool running_result     = static_cast<bool>(RunningResult);
    static constexpr type_strategy input_type_strategy =
        UseFp16 ? type_strategy::fp16
                : (UseFp32 ? type_strategy::fp32
                           : (UseFpmix ? type_strategy::fpmix : type_strategy::bfpmix));
    static constexpr bool use_amdgnc = UseAMDGCN;
    static constexpr auto neuron_op  = static_cast<neuron_op_type>(NrnOpId);
};
} // namespace detail

using config = detail::proto_config<MIO_LAYOUT_NHWC,
                                    MIO_SAVE_MEAN_VARIANCE,
                                    MIO_RUNNING_RESULT,
                                    MIOPEN_USE_FP16,
                                    MIOPEN_USE_FP32,
                                    MIOPEN_USE_FPMIX,
                                    MIOPEN_USE_BFPMIX,
                                    MIOPEN_USE_AMDGCN,
                                    MIOPEN_NRN_OP_ID>;

} // namespace miopen

// miopen batchnorm configs
namespace miopen {
namespace batchnorm {

enum class architecture : int
{
    unknown,
    gfx103x,
    gfx110x,
    gfx115x,
    gfx120x,
};

namespace detail {

// TODO: why this is here, becasue before c++ 20, double is not supported to be template parameter
struct half_max
{
    static constexpr double value = HALF_MAX;
};

// TODO: why this is here, becasue before c++ 20, double is not supported to be template parameter
struct flt_max
{
    static constexpr double value = FLT_MAX;
};

template <int Grp0, int Grp1, int Grp2>
struct launch_dimension
{
    static_assert(Grp0 >= 0, "MIO_BN_GRP0 should be always >= 0");
    static_assert(Grp1 >= 0, "MIO_BN_GRP1 should be always >= 0");
    static_assert(Grp2 >= 0, "MIO_BN_GRP2 should be always >= 0");
    static constexpr unsigned int grp0 = static_cast<unsigned int>(Grp0);
    static constexpr unsigned int grp1 = static_cast<unsigned int>(Grp1);
    static constexpr unsigned int grp2 = static_cast<unsigned int>(Grp2);
};

template <int Gfx103x, int Gfx110x, int Gfx120x, int Gfx115x>
struct architecture_switch
{
    static_assert(Gfx103x == 0 || Gfx103x == 1, "Gfx103x must be 0 or 1");
    static_assert(Gfx110x == 0 || Gfx110x == 1, "Gfx110x must be 0 or 1");
    static_assert(Gfx120x == 0 || Gfx120x == 1, "Gfx120x must be 0 or 1");
    static_assert(Gfx115x == 0 || Gfx115x == 1, "Gfx115x must be 0 or 1");
    static_assert(Gfx103x + Gfx110x + Gfx120x + Gfx115x == 1 ||
                      Gfx103x + Gfx110x + Gfx120x + Gfx115x == 0,
                  "only one of these configs can be chosen.");
    static constexpr architecture value =
        static_cast<bool>(Gfx103x)
            ? architecture::gfx103x
            : (static_cast<bool>(Gfx110x)
                   ? architecture::gfx110x
                   : (static_cast<bool>(Gfx120x)
                          ? architecture::gfx120x
                          : (static_cast<bool>(Gfx115x) ? architecture::gfx115x
                                                        : architecture::unknown)));
};

template <typename MiopenConfig,
          typename HalfMax,
          typename FltMax,
          typename LaunchDim,
          typename Architecture,
          int Variant,
          int NCHW,
          int MaxN,
          int C,
          int HW,
          int NHW,
          int CHW,
          int Vectorize,
          int StashMethod,
          int LoopUnrollMaxN,
          int LoopUnrollMaxHW,
          int LDSGCNSize,
          int LDSSize,
          int UseNodpp>
struct proto_config
{
    static_assert(Vectorize == 0 || Vectorize == 1, "Vectorize must be 0 or 1");
    static_assert(UseNodpp == 0 || UseNodpp == 1, "UseNodpp must be 0 or 1");
    static_assert(NCHW >= 0, "MIO_BN_NCHW should be always >= 0");
    static_assert(MaxN >= 0, "MIO_BN_MAXN should be always >= 0");
    static_assert(C >= 0, "MIO_BN_C should be always >= 0");
    static_assert(HW >= 0, "MIO_BN_HW should be always >= 0");
    static_assert(NHW >= 0, "MIO_BN_NHW should be always >= 0");
    static_assert(CHW >= 0, "MIO_BN_CHW should be always >= 0");

    static constexpr auto input_type_strategy = MiopenConfig::input_type_strategy;
    using fp_type                             = typename std::conditional<
        input_type_strategy == type_strategy::fp16 || input_type_strategy == type_strategy::fpmix,
        _Float16,
        typename std::conditional<input_type_strategy == type_strategy::fp32, float, ushort>::
            type>::type;
    using fp_prec_type  = float;
    using fp_accum_type = float;
    static constexpr double epsilon =
        input_type_strategy == type_strategy::fp16 ? 0.0001 : 0.000001;
    static constexpr fp_type max_val =
        input_type_strategy == type_strategy::fp16
            ? HalfMax::value
            : (input_type_strategy == type_strategy::fp32
                   ? FltMax::value
                   : 0); // TODO: not sure if 0 should be the default value of this.
    static constexpr auto launch_dim           = LaunchDim{};
    static constexpr unsigned int nchw         = static_cast<unsigned int>(NCHW);
    static constexpr unsigned int max_n        = static_cast<unsigned int>(MaxN);
    static constexpr unsigned int c            = static_cast<unsigned int>(C);
    static constexpr unsigned int hw           = static_cast<unsigned int>(HW);
    static constexpr unsigned int nhw          = static_cast<unsigned int>(NHW);
    static constexpr unsigned int chw          = static_cast<unsigned int>(CHW);
    static constexpr bool vectorize            = static_cast<bool>(Vectorize);
    static constexpr int stash_method          = StashMethod;
    static constexpr int loop_unroll_max_n     = LoopUnrollMaxN;
    static constexpr int loop_unroll_max_hw    = LoopUnrollMaxHW;
    static constexpr unsigned int lds_gcn_size = static_cast<unsigned int>(LDSGCNSize);
    static constexpr unsigned int lds_size     = static_cast<unsigned int>(LDSSize);
    static constexpr bool use_nodpp =
        input_type_strategy == type_strategy::fpmix ? false : static_cast<bool>(UseNodpp);
    static constexpr int variant      = Variant;
    static constexpr auto target_arch = Architecture::value;
#ifdef __AMDGCN__
    static constexpr bool use_amdgnc =
        MiopenConfig::use_amdgnc &&
        !(target_arch == architecture::gfx103x || target_arch == architecture::gfx110x ||
          target_arch == architecture::gfx120x || target_arch == architecture::gfx115x) &&
        !(use_nodpp && (variant != 0));
#else
    static constexpr bool use_amdgnc = false;
#endif
    static constexpr unsigned int vec_size = vectorize ? 4 : 1;
    static constexpr unsigned int vec_size_x =
        vectorize && MiopenConfig::layout_nhwc ? vec_size : 1;
    static constexpr unsigned int vec_size_y =
        vectorize && !MiopenConfig::layout_nhwc ? vec_size : 1;

    using fp_prec_c_type =
        typename std::conditional<vectorize && MiopenConfig::layout_nhwc,
                                  typename mapped_vector_type<fp_prec_type, vec_size>::type,
                                  fp_prec_type>::type;

    using fp_prec_ls_type =
        typename std::conditional<vectorize,
                                  typename mapped_vector_type<fp_prec_type, vec_size>::type,
                                  fp_prec_type>::type;

    using fp_c_type =
        typename std::conditional<vectorize && MiopenConfig::layout_nhwc,
                                  typename mapped_vector_type<fp_type, vec_size>::type,
                                  fp_type>::type;

    using fp_ls_type = typename std::
        conditional<vectorize, typename mapped_vector_type<fp_type, vec_size>::type, fp_type>::type;

    using fp_accum_c_type =
        typename std::conditional<vectorize && MiopenConfig::layout_nhwc,
                                  typename mapped_vector_type<fp_accum_type, vec_size>::type,
                                  fp_accum_type>::type;

    using fp_accum_ls_type =
        typename std::conditional<vectorize,
                                  typename mapped_vector_type<fp_accum_type, vec_size>::type,
                                  fp_accum_type>::type;
};

} // namespace detail

using config = miopen::batchnorm::detail::proto_config<
    miopen::config,
    miopen::batchnorm::detail::half_max,
    miopen::batchnorm::detail::flt_max,
    miopen::batchnorm::detail::launch_dimension<MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2>,
    miopen::batchnorm::detail::
        architecture_switch<MIO_BN_GFX103X, MIO_BN_GFX110X, MIO_BN_GFX120X, MIO_BN_GFX115X>,
    MIO_BN_VARIANT,
    MIO_BN_NCHW,
    MIO_BN_MAXN,
    MIO_BN_C,
    MIO_BN_HW,
    MIO_BN_NHW,
    MIO_BN_CHW,
    MIO_BN_VECTORIZE,
    MIO_BN_STASH_METHOD,
    MIO_BN_LOOP_UNROLL_MAXN,
    MIO_BN_LOOP_UNROLL_MAXHW,
    MIO_BN_LDSGCN_SIZE,
    MIO_BN_LDS_SIZE,
    MIO_BN_NODPP>;

} // namespace batchnorm

} // namespace miopen

#endif // CONFIGURATION_HPP
