/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include <miopen/conv/solvers.hpp>
#include <algorithm>
#include <numeric>
#include <functional>

namespace miopen {
namespace solver {
namespace conv {

PerformanceConfigAsmImplicitGemmGTC::PerformanceConfigAsmImplicitGemmGTC(
    std::string dir,
    std::string layout,
    std::string prec,
    int b,
    int e,
    int mpb,
    int npb,
    int kpb,
    int wtm,
    int wtn,
    int wtk,
    int wsm,
    int wsn,
    int wrm,
    int wrn,
    int mh,
    int vs,
    int gks,
    int me,
    int pta,
    std::initializer_list<int> ta_t,
    std::initializer_list<int> ta_c,
    std::initializer_list<int> tb_t,
    std::initializer_list<int> tb_c,
    bool spare)
    : direction(dir),
      tensor_layout(layout),
      precision(prec),
      nxb(b),
      nxe(e),

      gemm_m_per_block(mpb),
      gemm_n_per_block(npb),
      gemm_k_per_block(kpb),

      wave_tile_m(wtm),
      wave_tile_n(wtn),
      wave_tile_k(wtk),
      wave_step_m(wsm),
      wave_step_n(wsn),
      wave_repeat_m(wrm),
      wave_repeat_n(wrn),
      multihead(mh),
      vector_store(vs),
      gemm_k_global_split(gks),
      merge_e(me),
      tensor_a_pass_through(pta),
      tensor_a_thread_lengths(ta_t),
      tensor_a_cluster_lengths(ta_c),
      tensor_b_thread_lengths(tb_t),
      tensor_b_cluster_lengths(tb_c),
      use_spare_set(spare),
      index(0)
{
}

PerformanceConfigAsmImplicitGemmGTC::PerformanceConfigAsmImplicitGemmGTC(
    std::string dir,
    std::string layout,
    miopenDataType_t prec,
    int b,
    int e,
    int mpb,
    int npb,
    int kpb,
    int wtm,
    int wtn,
    int wtk,
    int wsm,
    int wsn,
    int wrm,
    int wrn,
    int mh,
    int vs,
    int gks,
    int me,
    int pta,
    std::initializer_list<int> ta_t,
    std::initializer_list<int> ta_c,
    std::initializer_list<int> tb_t,
    std::initializer_list<int> tb_c,
    bool spare)
    : direction(dir),
      tensor_layout(layout),
      precision(prec == miopenFloat ? "fp32" : (prec == miopenHalf ? "fp16" : "bf16")),
      nxb(b),
      nxe(e),

      gemm_m_per_block(mpb),
      gemm_n_per_block(npb),
      gemm_k_per_block(kpb),

      wave_tile_m(wtm),
      wave_tile_n(wtn),
      wave_tile_k(wtk),
      wave_step_m(wsm),
      wave_step_n(wsn),
      wave_repeat_m(wrm),
      wave_repeat_n(wrn),
      multihead(mh),
      vector_store(vs),
      gemm_k_global_split(gks),
      merge_e(me),
      tensor_a_pass_through(pta),
      tensor_a_thread_lengths(ta_t),
      tensor_a_cluster_lengths(ta_c),
      tensor_b_thread_lengths(tb_t),
      tensor_b_cluster_lengths(tb_c),
      use_spare_set(spare),
      index(0)
{
}

bool PerformanceConfigAsmImplicitGemmGTC::IsDefaultConstructed() const
{
    int default_lengths[4] = {1, 1, 1, 1};
    // clang-format off
    return direction == "fwd"
        && tensor_layout == "nchw"
        && precision == "fp32"
        && nxb == 1
        && nxe == 1
        && gemm_m_per_block == 1
        && gemm_n_per_block == 1
        && gemm_k_per_block == 1
        && wave_tile_m == 1
        && wave_tile_n == 1
        && wave_tile_k == 1
        && wave_step_m == 1
        && wave_step_n == 1
        && wave_repeat_m == 1
        && wave_repeat_n == 1
        && multihead == 1
        && vector_store == 1
        && gemm_k_global_split == 1
        && merge_e == 1
        && tensor_a_pass_through == 1
        && std::equal(std::begin(tensor_a_thread_lengths),  std::end(tensor_a_thread_lengths),  std::begin(default_lengths))
        && std::equal(std::begin(tensor_a_cluster_lengths), std::end(tensor_a_cluster_lengths), std::begin(default_lengths))
        && std::equal(std::begin(tensor_b_thread_lengths),  std::end(tensor_b_thread_lengths),  std::begin(default_lengths))
        && std::equal(std::begin(tensor_b_cluster_lengths), std::end(tensor_b_cluster_lengths), std::begin(default_lengths))
        && index == 0;
    // clang-format on
}

bool PerformanceConfigAsmImplicitGemmGTC::operator==(
    const PerformanceConfigAsmImplicitGemmGTC& other) const
{
    // clang-format off
    return direction == other.direction
        && tensor_layout == other.tensor_layout
        && precision == other.precision
        && nxb == other.nxb
        && nxe == other.nxe
        && gemm_m_per_block == other.gemm_m_per_block
        && gemm_n_per_block == other.gemm_n_per_block
        && gemm_k_per_block == other.gemm_k_per_block
        && wave_tile_m == other.wave_tile_m
        && wave_tile_n == other.wave_tile_n
        && wave_tile_k == other.wave_tile_k
        && wave_step_m == other.wave_step_m
        && wave_step_n == other.wave_step_n
        && wave_repeat_m == other.wave_repeat_m
        && wave_repeat_n == other.wave_repeat_n
        && multihead == other.multihead
        && vector_store == other.vector_store
        && ((gemm_k_global_split == 0 && other.gemm_k_global_split == 0)
            || (gemm_k_global_split > 0 && other.gemm_k_global_split > 0))
        && merge_e == other.merge_e
        && tensor_a_pass_through == other.tensor_a_pass_through
        && std::equal(std::begin(tensor_a_thread_lengths),  std::end(tensor_a_thread_lengths),  std::begin(other.tensor_a_thread_lengths))
        && std::equal(std::begin(tensor_a_cluster_lengths), std::end(tensor_a_cluster_lengths), std::begin(other.tensor_a_cluster_lengths))
        && std::equal(std::begin(tensor_b_thread_lengths),  std::end(tensor_b_thread_lengths),  std::begin(other.tensor_b_thread_lengths))
        && std::equal(std::begin(tensor_b_cluster_lengths), std::end(tensor_b_cluster_lengths), std::begin(other.tensor_b_cluster_lengths));
    // clang-format on
}

void PerformanceConfigAsmImplicitGemmGTC::CopyParameters(
    const PerformanceConfigAsmImplicitGemmGTC& other)
{
    // only copy parameters except spare/index, in case we break the search state
    direction             = other.direction;
    tensor_layout         = other.tensor_layout;
    precision             = other.precision;
    nxb                   = other.nxb;
    nxe                   = other.nxe;
    gemm_m_per_block      = other.gemm_m_per_block;
    gemm_n_per_block      = other.gemm_n_per_block;
    gemm_k_per_block      = other.gemm_k_per_block;
    wave_tile_m           = other.wave_tile_m;
    wave_tile_n           = other.wave_tile_n;
    wave_tile_k           = other.wave_tile_k;
    wave_step_m           = other.wave_step_m;
    wave_step_n           = other.wave_step_n;
    wave_repeat_m         = other.wave_repeat_m;
    wave_repeat_n         = other.wave_repeat_n;
    multihead             = other.multihead;
    vector_store          = other.vector_store;
    gemm_k_global_split   = other.gemm_k_global_split;
    merge_e               = other.merge_e;
    tensor_a_pass_through = other.tensor_a_pass_through;
    std::copy(std::begin(other.tensor_a_thread_lengths),
              std::end(other.tensor_a_thread_lengths),
              std::begin(tensor_a_thread_lengths));
    std::copy(std::begin(other.tensor_a_cluster_lengths),
              std::end(other.tensor_a_cluster_lengths),
              std::begin(tensor_a_cluster_lengths));
    std::copy(std::begin(other.tensor_b_thread_lengths),
              std::end(other.tensor_b_thread_lengths),
              std::begin(tensor_b_thread_lengths));
    std::copy(std::begin(other.tensor_b_cluster_lengths),
              std::end(other.tensor_b_cluster_lengths),
              std::begin(tensor_b_cluster_lengths));
}

std::string PerformanceConfigAsmImplicitGemmGTC::ToString() const
{
    std::ostringstream ss;
    // ss << ToKernelName();
    Serialize(ss);
    if(gemm_k_global_split != 0)
        ss << "[" << gemm_k_global_split << "]";
    return ss.str();
}

std::string PerformanceConfigAsmImplicitGemmGTC::ToKernelName(const ExecutionContext& ctx) const
{
    std::ostringstream kernel_name;
    const auto device_name = ctx.GetStream().GetDeviceName();
    std::string gtc_str =
        device_name == "gfx908" ? "_gtcx_" : (device_name == "gfx90a" ? "_gtcx2_" : "_gtcx3_");
    kernel_name << "igemm_" << direction << gtc_str << tensor_layout << "_" << precision << "_bx"
                << nxb << "_ex" << nxe << "_bt" << gemm_m_per_block << "x" << gemm_n_per_block
                << "x" << gemm_k_per_block << "_wt" << wave_tile_m << "x" << wave_tile_n << "x"
                << wave_tile_k << "_ws" << wave_step_m << "x" << wave_step_n << "_wr"
                << wave_repeat_m << "x" << wave_repeat_n << "_ta" << tensor_a_thread_lengths[0]
                << "x" << tensor_a_thread_lengths[1] << "x" << tensor_a_thread_lengths[2] << "x"
                << tensor_a_thread_lengths[3] << "_" << tensor_a_cluster_lengths[0] << "x"
                << tensor_a_cluster_lengths[1] << "x" << tensor_a_cluster_lengths[2] << "x"
                << tensor_a_cluster_lengths[3] << "_tb" << tensor_b_thread_lengths[0] << "x"
                << tensor_b_thread_lengths[1] << "x" << tensor_b_thread_lengths[2] << "x"
                << tensor_b_thread_lengths[3] << "_" << tensor_b_cluster_lengths[0] << "x"
                << tensor_b_cluster_lengths[1] << "x" << tensor_b_cluster_lengths[2] << "x"
                << tensor_b_cluster_lengths[3];

    if(tensor_a_pass_through != 0)
        kernel_name << "_pta";
    if(multihead != 0)
        kernel_name << "_mh";
    if(merge_e != 0)
        kernel_name << "_me";
    if(vector_store != 0)
        kernel_name << "_vs" + std::to_string(vector_store);
    if(gemm_k_global_split != 0)
        kernel_name << "_gkgs";

    return kernel_name.str();
}

int PerformanceConfigAsmImplicitGemmGTC::BlockSize() const
{
    return std::accumulate(std::begin(tensor_a_cluster_lengths),
                           std::end(tensor_a_cluster_lengths),
                           1,
                           std::multiplies<int>());
}

// Vector c type perform config

PerformanceConfigAsmImplicitGemmGTCvector::PerformanceConfigAsmImplicitGemmGTCvector(
    std::string dir,
    std::string layout,
    std::string prec,
    int b,
    int e,
    int mpb,
    int npb,
    int kpb,
    int lgtm,
    int lgtn,
    int lgpwm,
    int lgpwn,
    int lgrm,
    int lgrn,
    int vec_c,
    std::initializer_list<int> ta_t,
    std::initializer_list<int> ta_c,
    std::initializer_list<int> tb_t,
    std::initializer_list<int> tb_c,
    bool spare)
    : direction(dir),
      tensor_layout(layout),
      precision(prec),
      nxb(b),
      nxe(e),

      gemm_m_per_block(mpb),
      gemm_n_per_block(npb),
      gemm_k_per_block(kpb),

      lanegroup_tile_m(lgtm),
      lanegroup_tile_n(lgtn),
      lanegroup_wave_m(lgpwm),
      lanegroup_wave_n(lgpwn),
      lanegroup_repeat_m(lgrm),
      lanegroup_repeat_n(lgrn),
      vector_c(vec_c),
      tensor_a_thread_lengths(ta_t),
      tensor_a_cluster_lengths(ta_c),
      tensor_b_thread_lengths(tb_t),
      tensor_b_cluster_lengths(tb_c),
      use_spare_set(spare),
      index(0)
{
}

PerformanceConfigAsmImplicitGemmGTCvector::PerformanceConfigAsmImplicitGemmGTCvector(
    std::string dir,
    std::string layout,
    miopenDataType_t prec,
    int b,
    int e,
    int mpb,
    int npb,
    int kpb,
    int lgtm,
    int lgtn,
    int lgpwm,
    int lgpwn,
    int lgrm,
    int lgrn,
    int vec_c,
    std::initializer_list<int> ta_t,
    std::initializer_list<int> ta_c,
    std::initializer_list<int> tb_t,
    std::initializer_list<int> tb_c,
    bool spare)
    : direction(dir),
      tensor_layout(layout),
      precision((prec == miopenHalf && vec_c == 4) ? "Halfx4" : "Halfx8"),
      nxb(b),
      nxe(e),

      gemm_m_per_block(mpb),
      gemm_n_per_block(npb),
      gemm_k_per_block(kpb),

      lanegroup_tile_m(lgtm),
      lanegroup_tile_n(lgtn),
      lanegroup_wave_m(lgpwm),
      lanegroup_wave_n(lgpwn),
      lanegroup_repeat_m(lgrm),
      lanegroup_repeat_n(lgrn),
      vector_c(vec_c),
      tensor_a_thread_lengths(ta_t),
      tensor_a_cluster_lengths(ta_c),
      tensor_b_thread_lengths(tb_t),
      tensor_b_cluster_lengths(tb_c),
      use_spare_set(spare),
      index(0)
{
}

bool PerformanceConfigAsmImplicitGemmGTCvector::IsDefaultConstructed() const
{
    int default_lengths[4] = {1, 1, 1, 1};
    // clang-format off
    return direction == "fwd"
        && tensor_layout == "nchwc"
        && precision == "Halfx4"
        && nxb == 1
        && nxe == 1
        && gemm_m_per_block == 1
        && gemm_n_per_block == 1
        && gemm_k_per_block == 1
        && lanegroup_tile_m == 1
        && lanegroup_tile_n == 1
        && lanegroup_wave_m == 1
        && lanegroup_wave_n == 1
        && lanegroup_repeat_m == 1
        && lanegroup_repeat_n == 1
        && vector_c == 1
        && std::equal(std::begin(tensor_a_thread_lengths),  std::end(tensor_a_thread_lengths),  std::begin(default_lengths))
        && std::equal(std::begin(tensor_a_cluster_lengths), std::end(tensor_a_cluster_lengths), std::begin(default_lengths))
        && std::equal(std::begin(tensor_b_thread_lengths),  std::end(tensor_b_thread_lengths),  std::begin(default_lengths))
        && std::equal(std::begin(tensor_b_cluster_lengths), std::end(tensor_b_cluster_lengths), std::begin(default_lengths))
        && index == 0;
    // clang-format on
}

bool PerformanceConfigAsmImplicitGemmGTCvector::operator==(
    const PerformanceConfigAsmImplicitGemmGTCvector& other) const
{
    // clang-format off
    return direction == other.direction
        && tensor_layout == other.tensor_layout
        && precision == other.precision
        && nxb == other.nxb
        && nxe == other.nxe
        && gemm_m_per_block == other.gemm_m_per_block
        && gemm_n_per_block == other.gemm_n_per_block
        && gemm_k_per_block == other.gemm_k_per_block
        && lanegroup_tile_m == other.lanegroup_tile_m
        && lanegroup_tile_n == other.lanegroup_tile_n
        && lanegroup_wave_m == other.lanegroup_wave_m
        && lanegroup_wave_n == other.lanegroup_wave_n
        && lanegroup_repeat_m == other.lanegroup_repeat_m
        && lanegroup_repeat_n == other.lanegroup_repeat_n
        && vector_c == other.vector_c
        && std::equal(std::begin(tensor_a_thread_lengths),  std::end(tensor_a_thread_lengths),  std::begin(other.tensor_a_thread_lengths))
        && std::equal(std::begin(tensor_a_cluster_lengths), std::end(tensor_a_cluster_lengths), std::begin(other.tensor_a_cluster_lengths))
        && std::equal(std::begin(tensor_b_thread_lengths),  std::end(tensor_b_thread_lengths),  std::begin(other.tensor_b_thread_lengths))
        && std::equal(std::begin(tensor_b_cluster_lengths), std::end(tensor_b_cluster_lengths), std::begin(other.tensor_b_cluster_lengths));
    // clang-format on
}

void PerformanceConfigAsmImplicitGemmGTCvector::CopyParameters(
    const PerformanceConfigAsmImplicitGemmGTCvector& other)
{
    // only copy parameters except spare/index, in case we break the search state
    direction          = other.direction;
    tensor_layout      = other.tensor_layout;
    precision          = other.precision;
    nxb                = other.nxb;
    nxe                = other.nxe;
    gemm_m_per_block   = other.gemm_m_per_block;
    gemm_n_per_block   = other.gemm_n_per_block;
    gemm_k_per_block   = other.gemm_k_per_block;
    lanegroup_tile_m   = other.lanegroup_tile_m;
    lanegroup_tile_n   = other.lanegroup_tile_n;
    lanegroup_wave_m   = other.lanegroup_wave_m;
    lanegroup_wave_n   = other.lanegroup_wave_n;
    lanegroup_repeat_m = other.lanegroup_repeat_m;
    lanegroup_repeat_n = other.lanegroup_repeat_n;
    vector_c           = other.vector_c;
    std::copy(std::begin(other.tensor_a_thread_lengths),
              std::end(other.tensor_a_thread_lengths),
              std::begin(tensor_a_thread_lengths));
    std::copy(std::begin(other.tensor_a_cluster_lengths),
              std::end(other.tensor_a_cluster_lengths),
              std::begin(tensor_a_cluster_lengths));
    std::copy(std::begin(other.tensor_b_thread_lengths),
              std::end(other.tensor_b_thread_lengths),
              std::begin(tensor_b_thread_lengths));
    std::copy(std::begin(other.tensor_b_cluster_lengths),
              std::end(other.tensor_b_cluster_lengths),
              std::begin(tensor_b_cluster_lengths));
}

std::string PerformanceConfigAsmImplicitGemmGTCvector::ToString() const
{
    std::ostringstream ss;
    // ss << ToKernelName();
    Serialize(ss);
    return ss.str();
}

std::string
PerformanceConfigAsmImplicitGemmGTCvector::ToKernelName(const ExecutionContext& ctx) const
{
    std::ostringstream kernel_name;
    const auto device_name    = ctx.GetStream().GetDeviceName();
    std::string gtc_str       = device_name == "gfx1030" ? "_gtcn2_" : "_gtcn_";
    std::string vec_precision = precision == "Halfx4" ? "fp16x4" : "fp16x8";
    kernel_name << "igemm_" << direction << gtc_str << tensor_layout << "_" << vec_precision
                << "_bx" << nxb << "_ex" << nxe << "_bt" << gemm_m_per_block << "x"
                << gemm_n_per_block << "x" << gemm_k_per_block << "_lt" << lanegroup_tile_m << "x"
                << lanegroup_tile_n << "_lw" << lanegroup_wave_m << "x" << lanegroup_wave_n << "_lr"
                << lanegroup_repeat_m << "x" << lanegroup_repeat_n << "_ta"
                << tensor_a_thread_lengths[0] << "x" << tensor_a_thread_lengths[1] << "x"
                << tensor_a_thread_lengths[2] << "x" << tensor_a_thread_lengths[3] << "_"
                << tensor_a_cluster_lengths[0] << "x" << tensor_a_cluster_lengths[1] << "x"
                << tensor_a_cluster_lengths[2] << "x" << tensor_a_cluster_lengths[3] << "_tb"
                << tensor_b_thread_lengths[0] << "x" << tensor_b_thread_lengths[1] << "x"
                << tensor_b_thread_lengths[2] << "x" << tensor_b_thread_lengths[3] << "_"
                << tensor_b_cluster_lengths[0] << "x" << tensor_b_cluster_lengths[1] << "x"
                << tensor_b_cluster_lengths[2] << "x" << tensor_b_cluster_lengths[3];

    return kernel_name.str();
}

int PerformanceConfigAsmImplicitGemmGTCvector::BlockSize() const
{
    return std::accumulate(std::begin(tensor_a_cluster_lengths),
                           std::end(tensor_a_cluster_lengths),
                           1,
                           std::multiplies<int>());
}

} // namespace conv
} // namespace solver
} // namespace miopen
