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
.if ROCM_METADATA_VERSION == 4
.hsa_code_object_version 2,1
.hsa_code_object_isa
.endif

.text
.p2align 8

.include "rocm_version.inc"
.include "inst_wrappers.inc"
.include "gpr_alloc.inc"
.include "common.inc"


// kernarg layout:
// dwords 0 	uint32_t N;
// dwords 1 	uint32_t C;
// dwords 2 	uint32_t H;
// dwords 3 	uint32_t W;
//
// dwords 4 	uint32_t K;
// dwords 5 	uint32_t n_groups;
// dwords 6 	uint32_t flags;
// dwords 7 	uint32_t reserved;
//
// dwords 8:9	uint64_t  data_addr;
// dwords 10:11	uint64_t  filter_addr;
// dwords 12:13 uint64_t  output_addr;
// dwords 14:15	uint64_t  return_addr;
//
// dwords 16	uint32_t  R;	// filter height
// dwords 17	uint32_t  S;	// filter width
// dwords 18	int32_t   pad_h;	// padding
// dwords 19	int32_t   pad_w;	// padding
//
// dwords 20	uint32_t  out_h;	// output height
// dwords 21	uint32_t  out_w;	// output width
//
// dwords 22:23	uint64_t bias_addr;
// dwords 24	float RELU_alpha;
//
// dwords 25	uint32_t d_N_stride;
// dwords 26	uint32_t d_C_stride;
// dwords 27	uint32_t d_H_stride;
// dwords 28	uint32_t d_W_stride;
//
// dwords 29	uint32_t f_K_stride;
// dwords 30	uint32_t f_C_stride;
// dwords 31	uint32_t f_R_stride;
// dwords 32	uint32_t f_S_stride;
//
// dwords 33	uint32_t o_N_stride;
// dwords 34	uint32_t o_K_stride;
// dwords 35	uint32_t o_H_stride;
// dwords 36	uint32_t o_W_stride;
.set KERNEL_ARGUMENTS_SIZE, (36+1)*4


default read_size, 1
default elem_size, 4
default acc_type, TYPE_FP32
default buf_type, TYPE_FP32


static_assert(xformx_f_size <= 6)
static_assert(xformy_f_size <= 6)
static_assert(xformx_o_size == 1 || xformx_o_size == 3 || xformx_o_size == 5 || xformx_o_size == 7)
static_assert(xformy_o_size == 1 || xformy_o_size == 3 || xformy_o_size == 5 || xformy_o_size == 7)
static_assert(fdilation_w == fdilation_h)

static_assert(acc_type == TYPE_FP32)
static_assert(buf_type == TYPE_FP32 || buf_type == TYPE_FP16 || buf_type == TYPE_BFP16)
.if(buf_type == TYPE_FP32)
    elem_size = 4
    lds_elem_size = 4
.elseif (buf_type == TYPE_FP16 || buf_type == TYPE_BFP16)
    static_assert(read_size == 1)
    elem_size = 2
    lds_elem_size = 4
.endif

fdilation = fdilation_w
out_points = xformx_o_size * xformy_o_size

static_assert(read_size == 1) // TODO: remove restriction
.GPR_ALLOC_BEGIN
// initial state
// s[0:1] - kernarg address
// s2 - wg x (1 wg per CU)
kernarg = 0
gid_x = 2
stmp = 3
.SGPR_ALLOC_FROM 4
// following sgprs should be allocated in strict sequence to follow kernarg layout
.SGPR_ALLOC N
.SGPR_ALLOC C
.SGPR_ALLOC H
.SGPR_ALLOC W

.SGPR_ALLOC K
.SGPR_ALLOC n_groups
.SGPR_ALLOC flags
.SGPR_ALLOC unused1

.SGPR_ALLOC d_addr, 2
.SGPR_ALLOC unused2, 2 // filter_addr
.SGPR_ALLOC o_addr, 2
.SGPR_ALLOC dbg_addr, 2

.SGPR_ALLOC R
.SGPR_ALLOC S
.SGPR_ALLOC pad_h
.SGPR_ALLOC pad_w

.SGPR_ALLOC out_h
.SGPR_ALLOC out_w

.SGPR_ALLOC unused3, 2 // bias_addr
.SGPR_ALLOC unused4 // RELU_alpha

.SGPR_ALLOC d_N_stride
.SGPR_ALLOC d_C_stride
.SGPR_ALLOC d_H_stride
.SGPR_ALLOC d_W_stride

.SGPR_ALLOC f_K_stride
.SGPR_ALLOC f_C_stride
.SGPR_ALLOC f_R_stride
.SGPR_ALLOC f_S_stride

.SGPR_ALLOC o_N_stride
.SGPR_ALLOC o_C_stride
.SGPR_ALLOC o_H_stride
.SGPR_ALLOC o_W_stride

// end of kernarg extent

.SGPR_ALLOC wave_id
.SGPR_ALLOC soff
.SGPR_RESERVE_VCC

.VGPR_ALLOC_FROM 0
.VGPR_ALLOC tid
.VGPR_ALLOC vtmp
accums_cnt = read_size * xformx_d_size * xformy_d_size
.VGPR_ALLOC accums, accums_cnt
.VGPR_ALLOC voff_d
.VGPR_ALLOC voff_o
.VGPR_ALLOC vcur_n
.VGPR_ALLOC vcur_c
.VGPR_ALLOC vcur_tile


.GPR_ALLOC_END

.macro kernel_begin  x_o_size, y_o_size, x_f_size, y_f_size
    .globl miopenGcnAsmWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
    .type miopenGcnAsmWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size,@function
    .if ROCM_METADATA_VERSION == 4
        .amdgpu_hsa_kernel miopenGcnAsmWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
    .endif
    miopenGcnAsmWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size:
.endm

kernel_begin  %xformx_o_size, %xformy_o_size, %xformx_f_size, %xformy_f_size

.if ROCM_METADATA_VERSION == 4
.include "xform_kd_cov2.inc"
.endif

    s_load_dwordx16 s[N:dbg_addr+1], s[kernarg:kernarg+1], 0x0
    s_load_dwordx16 s[R:f_R_stride], s[kernarg:kernarg+1], 0x4 * 16
    s_load_dwordx4 s[f_S_stride:o_H_stride], s[kernarg:kernarg+1], 0x4 * 32
    s_load_dword   s[o_W_stride], s[kernarg:kernarg+1], 0x4 * 36

    v_lshrrev_b32 v[vtmp], 6, v[tid]
    //v_readfirstlane_b32 s[wave_id], v[vtmp]
    s_mov_b32 s[wave_id], s[gid_x]
    v_and_b32 v[tid], 0x3f, v[tid]

    s_waitcnt 0

    .GPR_REUSE out_h, const8_0
    .GPR_REUSE pad_w, const0_25
    s_mov_b32 s[const0_25], 0.25
    s_mov_b32 s[const8_0], 8.0

    // compute addresses
    v_mul_lo_u32 v[vcur_tile], 0+read_size, v[tid]
    s_mul_i32 s[stmp], read_size * wave_size, s[wave_id]
    v_add_u32 v[vcur_tile], s[stmp], v[vcur_tile]
    .GPR_REUSE pad_h, tiles
    s_mul_i32 s[tiles], s[N], s[K]
    s_cmp_lt_u32 s[d_N_stride], s[d_C_stride]
    s_cbranch_scc1 CN_layout
    NC_layout:
    u_div v[vcur_tile], s[K], v[vcur_n], vtmp, unused2, dbg_addr
    v_mul_lo_u32 v[vtmp], s[K], v[vcur_n]
    v_sub_u32 v[vcur_c], v[vcur_tile], v[vtmp]
    s_branch CN_layout_end
    CN_layout:
    u_div v[vcur_tile], s[N], v[vcur_c], vtmp, unused2, dbg_addr
    v_mul_lo_u32 v[vtmp], s[N], v[vcur_c]
    v_sub_u32 v[vcur_n], v[vcur_tile], v[vtmp]
    CN_layout_end:
    v_mul_lo_u32 v[vtmp], s[d_C_stride], v[vcur_c]
    v_mul_lo_u32 v[voff_d], s[d_N_stride], v[vcur_n]
    v_add_u32 v[voff_d], v[vtmp], v[voff_d]
    v_mul_lo_u32 v[vtmp], s[o_C_stride], v[vcur_c]
    v_mul_lo_u32 v[voff_o], s[o_N_stride], v[vcur_n]
    v_add_u32 v[voff_o], v[vtmp], v[voff_o]

    s_mov_b32 s[soff], 0
    .GPR_REUSE o_W_stride, buf_step
    s_mul_i32 s[buf_step], elem_size, s[tiles]

    // mask out of range tiles with read_size granularity
    v_cmpx_lt_i32 vcc, v[vcur_tile], s[tiles]

    // construct descriptors
    .GPR_REUSE d_addr, d_desc //s[4]
    .GPR_REUSE o_addr, o_desc //s[4]
    .GPR_REUSE  R, s2_tmp
    .GPR_INVALIDATE unused
    .GPR_INVALIDATE S
    .GPR_INVALIDATE dbg_addr
    s_mov_b32 s[d_desc+3], 0x00020000
    s_mov_b32 s[o_desc+3], 0x00020000
    s_mul_i32 s[d_desc+2], xformx_d_size * xformy_d_size, s[d_W_stride]
    s_min_i32 s[stmp], s[o_C_stride], s[o_N_stride]
    s_mul_i32 s[o_desc+2], s[stmp], s[tiles]


    i=0
    .rept xformx_d_size * xformy_d_size
        acc = accums + read_size * i
        .if (elem_size == 2 && read_size == 1)
            buffer_load_short_d16 v[acc], v[voff_d], s[d_desc:d_desc+3], s[soff] offen
        .elseif read_size == 1
            buffer_load_dword v[acc], v[voff_d], s[d_desc:d_desc+3], s[soff] offen
        .elseif read_size == 2
            buffer_load_dwordx2 v[acc:acc+1], v[voff_d], s[d_desc:d_desc+3], s[soff] offen
        .elseif read_size == 3
            buffer_load_dwordx3 v[acc:acc+2], v[voff_d], s[d_desc:d_desc+3], s[soff] offen
        .elseif read_size == 4
            buffer_load_dwordx4 v[acc:acc+3], v[voff_d], s[d_desc:d_desc+3], s[soff] offen
        .endif
        s_add_u32 s[soff], s[soff], s[buf_step]
        i=i+1
    .endr

    s_waitcnt 0

    .if(buf_type != TYPE_FP32)
        static_assert(read_size == 1)
        .rept i
            //if acc_type == buf_type do nothing
            v_reg_data_type_convert v[accums + i - 1], acc_type, v[accums + i - 1], buf_type
            i = i - 1
        .endr
    .endif

    // inplace xform that could store output in lower or upper addresses
    .macro m_xform_out o_size, f_size, f_dil, lower
        .if \o_size == 3 && \f_size == 2 && \f_dil == 1
            v_add_f32 v[vtmp], v[dx1], v[dx2]
            v_sub_f32 v[dx1], v[dx1], v[dx2]
            v_add_f32 v[dx2], v[vtmp], v[dx3]
            v_add_f32 v[dx0], v[dx0], v[vtmp]
        .elseif \o_size == 3 && \f_size == 3 && \f_dil == 1
            v_add_f32 v[vtmp], v[dx1], v[dx2]
            v_sub_f32 v[dx1], v[dx1], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[dx3]
            v_fma_f32 v[dx4], v[dx3], 4.0, v[dx4]
            v_add_f32 v[dx0], v[vtmp], v[dx0]
            v_fma_f32 v[dx1], v[dx3], 2.0, v[dx1]
            v_add_f32 v[dx2], v[dx4], v[vtmp]
        .elseif \o_size == 3 && \f_size == 4 && \f_dil == 1
            v_add_f32 v[vtmp], v[dx1], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[vtmp]
            v_add_f32 v[dx5], v[dx5], v[vtmp]
            v_sub_f32 v[dx1], v[dx1], v[dx2]
            v_add_f32 v[vtmp], v[dx3], v[dx4]
            v_sub_f32 v[dx3], v[dx3], v[dx4]
            v_add_f32 v[dx0], v[vtmp], v[dx0]
            v_fma_f32 v[dx1], v[dx3], 2.0, v[dx1]
            v_fma_f32 v[dx2], v[vtmp], 4.0, v[dx5]
        .elseif \o_size == 3 && \f_size == 5 && \f_dil == 1
            v_add_f32 v[vtmp], v[dx1], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[dx5]
            v_fma_f32 v[dx1], v[dx5], 0.5, v[dx1]
            v_fma_f32 v[dx6], v[dx5], s[const0_25], v[dx6]
            v_sub_f32 v[dx1], v[dx1], v[dx2]
            v_add_f32 v[dx0], v[vtmp], v[dx0]
            v_add_f32 v[dx6], v[vtmp], v[dx6]
            v_sub_f32 v[dx5], v[dx3], v[dx4]
            v_add_f32 v[vtmp], v[dx3], v[dx4]
            v_add_f32 v[dx0], v[dx0], v[vtmp]
            v_fma_f32 v[dx1], v[dx5], 2.0, v[dx1]
            v_fma_f32 v[dx2], v[vtmp], 4.0, v[dx6]
        .elseif \o_size == 3 && \f_size == 6 && \f_dil == 1
            v_add_f32 v[vtmp], v[dx5], v[dx6]
            v_add_f32 v[dx0], v[vtmp], v[dx0]
            v_fma_f32 v[dx7], v[vtmp], s[const0_25], v[dx7]
            v_add_f32 v[vtmp], v[dx1], v[dx2]
            v_add_f32 v[dx0], v[vtmp], v[dx0]
            v_add_f32 v[dx7], v[vtmp], v[dx7]
            v_sub_f32 v[vtmp], v[dx5], v[dx6]
            v_fma_f32 v[dx1], v[vtmp], 0.5, v[dx1]
            v_sub_f32 v[dx1], v[dx1], v[dx2]
            v_sub_f32 v[dx2], v[dx3], v[dx4]
            v_add_f32 v[vtmp], v[dx3], v[dx4]
            v_add_f32 v[dx0], v[dx0], v[vtmp]
            v_fma_f32 v[dx1], v[dx2], 2.0, v[dx1]
            v_fma_f32 v[dx2], v[vtmp], 4.0, v[dx7]
        .elseif \o_size == 7 && \f_size == 2 && \f_dil == 1
             v_fma_f32 v[vtmp], v[dx3], s[const8_0], v[dx4]

             v_add_f32 v[dx4], v[dx7], v[dx6]
             v_add_f32 v[dx4], v[dx4], v[dx5]
             v_sub_f32 v[dx5], v[dx6], v[dx7]
             v_add_f32 v[dx6], v[dx6], v[dx7]
             v_add_f32 v[dx6], v[dx6], v[dx8]

             v_add_f32 v[dx8], v[dx1], v[dx2]
             v_sub_f32 v[dx7], v[dx1], v[dx2]

             v_add_f32 v[dx0], v[dx0], v[dx3]
             v_add_f32 v[dx0], v[dx0], v[dx8]

             v_fma_f32 v[dx2], v[dx3], 4.0, v[dx8]
             v_fma_f32 v[dx1], v[dx3], 2.0, v[dx7]

             v_add_f32 v[dx3], v[dx7], v[vtmp]

        .elseif \o_size == 7 && \f_size == 3 && \f_dil == 1

             v_fma_f32 v[dx10], v[dx9], 4.0, v[dx10]
             v_mov_b32 v[vtmp], v[dx6]

             v_add_f32 v[dx7],  v[dx7], v[dx8]
             v_add_f32 v[dx6],  v[dx7], v[dx10]


             v_add_f32 v[vtmp],  v[vtmp], v[dx9]

             v_mov_b32 v[dx10], v[dx5]

             v_fma_f32 v[dx9], v[dx9], 2.0, v[dx7]
             v_fma_f32 v[dx5], v[dx8], -2.0, v[dx9]

             v_mov_b32 v[dx9], v[dx4]

             v_add_f32 v[dx4], v[vtmp], v[dx7]

             v_add_f32 v[vtmp], v[dx9], v[dx3]
             v_add_f32 v[dx8], v[dx1], v[dx2]
             v_sub_f32 v[dx7], v[dx1], v[dx2]

             v_add_f32 v[dx0], v[dx0], v[vtmp]
             v_add_f32 v[dx0], v[dx0], v[dx8]

             v_fma_f32 v[dx2], v[vtmp], 4.0, v[dx8]

             v_sub_f32 v[vtmp], v[dx3], v[dx9]

             v_fma_f32 v[dx1], v[vtmp], 2.0, v[dx7]
             v_fma_f32 v[dx3], v[vtmp], s[const8_0], v[dx7]
             v_add_f32 v[dx3], v[dx3], v[dx10]
        .elseif \o_size == 3 && \f_size == 2 && \f_dil == 2
            v_add_f32 v[dx0], v[dx0], v[dx2]
            v_add_f32 v[dx1], v[dx1], v[dx3]
            v_add_f32 v[dx2], v[dx2], v[dx4]
        .elseif \o_size == 3 && \f_size == 3 && \f_dil == 2
            v_add_f32 v[dx0], v[dx0], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[dx4]

            v_add_f32 v[dx1], v[dx1], v[dx3]
            v_add_f32 v[dx1], v[dx1], v[dx5]

            v_sub_f32 v[dx2], v[dx2], v[dx4]
            v_add_f32 v[dx2], v[dx2], v[dx6]
        .elseif \o_size == 3 && \f_size == 4 && \f_dil == 2
            v_add_f32 v[dx0], v[dx0], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[dx4]
            v_add_f32 v[dx0], v[dx0], v[dx6]

            v_add_f32 v[dx1], v[dx1], v[dx3]
            v_add_f32 v[dx1], v[dx1], v[dx5]
            v_add_f32 v[dx1], v[dx1], v[dx7]

            v_sub_f32 v[dx2], v[dx2], v[dx4]
            v_mac_f32 v[dx2], 2.0, v[dx6]
            v_add_f32 v[dx2], v[dx2], v[dx8]
        .elseif \o_size == 3 && \f_size == 5 && \f_dil == 2
            v_add_f32 v[dx0], v[dx0], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[dx4]
            v_add_f32 v[dx0], v[dx0], v[dx6]
            v_add_f32 v[dx0], v[dx0], v[dx8]

            v_add_f32 v[dx1], v[dx1], v[dx3]
            v_add_f32 v[dx1], v[dx1], v[dx5]
            v_add_f32 v[dx1], v[dx1], v[dx7]
            v_add_f32 v[dx1], v[dx1], v[dx9]

            v_sub_f32 v[dx2], v[dx2], v[dx4]
            v_mac_f32 v[dx2], 2.0, v[dx6]
            v_mac_f32 v[dx2],-2.0, v[dx8]
            v_add_f32 v[dx2], v[dx2], v[dx10]
        .elseif \o_size == 3 && \f_size == 6 && \f_dil == 2
            v_add_f32 v[dx0], v[dx0], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[dx4]
            v_add_f32 v[dx0], v[dx0], v[dx6]
            v_add_f32 v[dx0], v[dx0], v[dx8]
            v_add_f32 v[dx0], v[dx0], v[dx10]

            v_add_f32 v[dx1], v[dx1], v[dx3]
            v_add_f32 v[dx1], v[dx1], v[dx5]
            v_add_f32 v[dx1], v[dx1], v[dx7]
            v_add_f32 v[dx1], v[dx1], v[dx9]
            v_add_f32 v[dx1], v[dx1], v[dx11]

            v_sub_f32 v[dx2], v[dx2], v[dx4]
            v_mac_f32 v[dx2], 2.0, v[dx6]
            v_mac_f32 v[dx2],-2.0, v[dx8]
            v_mac_f32 v[dx2], 0.5, v[dx10]
            v_add_f32 v[dx2], v[dx2], v[dx12]
        .elseif \o_size == 5 && \f_size == 3 && \f_dil == 1

            v_mul_f32 v[vtmp], 0.0625, v[dx5]
            v_add_f32 v[dx6], v[dx6], v[vtmp]
            v_add_f32 v[dx0], v[dx0], v[dx5]
            v_add_f32 v[vtmp], v[dx1], v[dx2]
            v_sub_f32 v[dx1], v[dx1], v[dx2]
            v_add_f32 v[dx2], v[dx3], v[dx4]
            v_sub_f32 v[dx3], v[dx3], v[dx4]
            v_add_f32 v[dx4], v[vtmp], v[dx2]
            v_add_f32 v[dx0], v[dx0], v[dx4]
            v_mul_f32 v[dx4], 16.0, v[dx2]
            v_add_f32 v[dx4], v[dx4], v[vtmp]
            v_add_f32 v[dx4], v[dx6], v[dx4]
            v_fma_f32 v[dx2], 4.0, v[dx2], v[vtmp]
            v_mul_f32 v[dx6], 0.25, v[dx5]
            v_add_f32 v[dx2], v[dx2], v[dx6]
            v_fma_f32 v[vtmp], 2.0, v[dx3], v[dx1]
            v_mul_f32 v[dx3], 8.0, v[dx3]
            v_add_f32 v[dx3], v[dx3], v[dx1]
            v_fma_f32 v[dx1], 0.5, v[dx5], v[vtmp]
            v_mul_f32 v[dx5], 0.125, v[dx5]
            v_add_f32 v[dx3], v[dx5], v[dx3]

        .elseif \o_size == 5 && \f_size == 4 && \f_dil == 1

            v_add_f32 v[vtmp], v[dx1], v[dx2]
            v_sub_f32 v[dx1],  v[dx1], v[dx2]
            v_add_f32 v[dx2], v[dx3], v[dx4]
            v_sub_f32 v[dx3], v[dx3], v[dx4]
            v_add_f32 v[dx4], v[dx5], v[dx6]
            v_sub_f32 v[dx5], v[dx5], v[dx6]
            v_add_f32 v[dx6], v[vtmp], v[dx2]
            v_add_f32 v[dx0], v[dx6],v[dx0]
            v_add_f32 v[dx0], v[dx0],v[dx4]
            v_mul_f32 v[dx6], 0.0625, v[dx4]
            v_add_f32 v[dx7], v[dx6], v[dx7]
            v_mul_f32 v[dx6], 16.0, v[dx2]
            v_add_f32 v[dx7], v[dx6], v[dx7]
            v_fma_f32 v[dx2], 4.0, v[dx2], v[vtmp]
            v_mul_f32 v[dx6], 0.25, v[dx4]
            v_add_f32 v[dx2], v[dx2], v[dx6]
            v_add_f32 v[dx4], v[vtmp], v[dx7]
            v_mul_f32 v[dx6], 0.125, v[dx5]
            v_mul_f32 v[dx7], 8.0, v[dx3]
            v_add_f32 v[dx7], v[dx7], v[dx6]
            v_fma_f32 v[dx6], 2.0, v[dx3], v[dx1]
            v_add_f32 v[dx3], v[dx1], v[dx7]
            v_fma_f32 v[dx1], 0.5, v[dx5], v[dx6]

        .elseif \o_size == 1 || \f_size == 1
            //nop
        .else
            static_assert(0)
        .endif

        .if \lower
            .irp ii,0,1,2,3,4,5,6
                .if(\ii < \o_size)
                    v_mov_b32 v[ox\ii], v[dx\ii]
                .endif
            .endr
        .else
            .irp ii,6,5,4,3,2,1,0
                .if(\ii < \o_size)
                    v_mov_b32 v[ox\ii], v[dx\ii]
                .endif
            .endr
        .endif
    .endm


    // backtransform each column
    i=0
    .rept xformx_d_size
        dx0 = accums + read_size * i
        ox0 = dx0 + read_size * xformx_d_size * (xformy_d_size - xformy_o_size)
        .irp ii,1,2,3,4,5,6,7,8,9,10,11,12
            dx\ii = dx0 + read_size * xformx_d_size * \ii
            ox\ii = ox0 + read_size * xformx_d_size * \ii
        .endr

        m_xform_out xformy_o_size, xformy_f_size, fdilation, 0
        i=i+1
    .endr

    // compute output offset
    s_mov_b32 s[soff], 0
    s_mov_b32 s[buf_step], elem_size * out_points

    // backtransform each row
    i=0

    .rept xformy_o_size
        dx0 = accums + read_size * xformx_d_size * (xformy_d_size - xformy_o_size) + read_size * xformx_d_size * i
        ox0 = accums + i * xformx_o_size
        .irp ii,1,2,3,4,5,6,7,8,9,10,11,12
            dx\ii = dx0 + read_size * \ii
            ox\ii = ox0 + \ii
        .endr
        m_xform_out xformx_o_size, xformx_f_size, fdilation, 1
        i=i+1
    .endr

    out_reg_id = 0

    .if(elem_size == 2)
        .rept out_points
            v_reg_data_type_convert v[accums + out_reg_id], buf_type, v[accums + out_reg_id], acc_type, v[vtmp], s[s2_tmp:s2_tmp+1]
            buffer_store_short v[accums + out_reg_id], v[voff_o], s[o_desc:o_desc+3], s[soff], offen offset:0+elem_size*out_reg_id
            out_reg_id = out_reg_id + 1
        .endr
    .else
        .rept (out_points / 4)
            buffer_store_dwordx4 v[accums+out_reg_id:accums+out_reg_id+3], v[voff_o], s[o_desc:o_desc+3], s[soff], offen offset:0+out_reg_id*elem_size
            out_reg_id = out_reg_id + 4
        .endr

        .rept (out_points % 4)
            buffer_store_dword v[accums+out_reg_id], v[voff_o], s[o_desc:o_desc+3], s[soff], offen offset:0+out_reg_id*elem_size
            out_reg_id = out_reg_id + 1
        .endr
    .endif
    s_add_u32 s[soff], s[buf_step], s[soff]

    s_endpgm

.Lfunc_end0:

.include "xform_metadata.inc"

.altmacro
.macro METADATA_WRAPPER sc, vc, wg_x, lds_size, kernarg_size, kernel_suf
    KERNEL_DESCRIPTOR_COV3 <miopenGcnAsmWinogradXformOut\kernel_suf>
    METADATA \sc, \vc, \wg_x, \lds_size, \kernarg_size, <miopenGcnAsmWinogradXformOut\kernel_suf>
.endm

.macro kernel_end x_o_size, y_o_size, x_f_size, y_f_size
    .size miopenGcnAsmWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size, .Lfunc_end0 - miopenGcnAsmWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
    METADATA_WRAPPER %.AUTO_SGPR_COUNT, %.AUTO_VGPR_COUNT, %(64), %.AUTO_LDS_BYTE_SIZE, %KERNEL_ARGUMENTS_SIZE, _\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
.endm

kernel_end %xformx_o_size, %xformy_o_size, %xformx_f_size, %xformy_f_size
