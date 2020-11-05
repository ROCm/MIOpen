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
.include "rocm_version.inc"
.include "gpr_alloc.inc"
.include "inst_wrappers.inc"
.include "utilities.inc"
.include "conv_common.inc"

.altmacro
// Kernel performs winograd transformation for either (input) data, filter or output
//
// non-transformed data should have [H][W] as the lowest dims for better performance
// transformed data is stored in GCNHW or GNCHW layouts

// limits:
// N, C, H, W, K, R, S, pad_*, out_*, n_groups < 2^16
// n_groups < 2^10
// total number of tiles < 2^32
// (data transform only) filter or output should be covered by a single winograd tile or
// winograd transform should by symmetrical wrt filter and output tile step
// winograd tile sizes (including input and transformed) should be <= 8
// input/output W stride, H stride  < 2^23
// filter R stride, S stride  < 2^23
// intput tensor size < 2^31 (2GiB)
// transformed layout - GCNHW or GNCHW
// mirroring supported only for filter transform, filter size should be equal to filter tile

// kernarg layout:

// dwords 0:1   uint64_t src_addr
// dwords 2:3   uint64_t dst_addr
//
//            dim sizes of the non-transformed data
// dwords 4     uint32_t G
// dwords 5     uint32_t M1 // highest between N and C
// dwords 6     uint32_t M0 // lowest between N and C
// dwords 7     uint32_t H
// dwords 8     uint32_t W
//
// dwords 9     uint32_t n_groups
// dwords 10    uint32_t flags // 1 - swap_NC, 2 - single otile_w, 4 - single otile_h
//
// dwords 11    int32_t pad_h // convolution padding
// dwords 12    int32_t pad_w // convolution padding
//
// dwords 13    uint32_t tiles_h // number of transformed tiles in h dim
// dwords 14    uint32_t tiles_w // number of transformed tiels in w dim
//
//            strides of the non-transformed data         
// dwords 15    uint32_t stride_G
// dwords 16    uint32_t stride_M1 // highest between N and C
// dwords 17    uint32_t stride_M0 // lowest between N and C
// dwords 18    uint32_t stride_H
// dwords 19    uint32_t stride_W
.set KERNEL_ARGUMENTS_SIZE, (19+1)*4+6*8

default pipe_config, 0
default use_exec_for_vmem, 1 // todo test and remove

.if pipe_config == 0
    r_pipe_depth  = 8
    w_pipe_depth  = 8
    lds_buffers   = 4
    rw_pipe_shift = 11
    n_accums = r_pipe_depth * 8
    // interleave_w_xform = 1
.elseif pipe_config == 1
    assert(0)
.else
    assert(0)
.endif


phase_offset = r_pipe_depth
static_assert( r_pipe_depth % lds_buffers == 0 )
static_assert( w_pipe_depth % lds_buffers == 0 )
static_assert( rw_pipe_shift < 30 )

// external flags, provided by host
.set L_F_SWAP_NC, 0
.set L_F_SINGLE_OTILE_W, 1
.set L_F_SINGLE_OTILE_H, 2
.set external_flags_mask, 0x7
// internal flags
.set L_F_ODD_PAD_W, 3
.set L_F_ODD_PAD_H, 4

static_assert(src_type == TYPE_FP32 || src_type == TYPE_FP16 || src_type == TYPE_BFP16)
static_assert(dst_type == TYPE_FP32 || dst_type == TYPE_FP16 || dst_type == TYPE_BFP16)


.macro s_swap s0, s1, stmp
    s_mov_b32 \stmp, \s0
    s_mov_b32 \s0, \s1
    s_mov_b32 \s1, \stmp
.endm
.macro v_cvt dst, src, dst_type, src_type
    .if \dst_type == \src_type
        // no op
    .elseif \dst_type == TYPE_FP32 && \src_type == TYPE_FP16
        v_cvt_f32_f16 \dst, \src
    .elseif \dst_type == TYPE_FP32 && \src_type == TYPE_BFP16
        v_lshlrev_b32 \dst, 16, \src
    .elseif \dst_type == TYPE_FP16 && \src_type == TYPE_FP32
        v_cvt_f16_f32 \dst, \src
    .elseif \dst_type == TYPE_BFP16 && \src_type == TYPE_FP32
        v_lshrrev_b32 \dst, 16, \src // TODO: add rounding, etc
    .else
        assert(0) // unknown type or convertion is not implemented
    .endif
.endm
.macro v_cvt_array base_v, n, dst_type, src_type
    cvt_i = 0 // TODO: switch i to local scope
    .rept \n
        v_cvt v[\base_v + cvt_i], v[\base_v + cvt_i], \dst_type, \src_type
        cvt_i = cvt_i + 1
    .endr
.endm
.macro elemsize size, type
    .if (\type == TYPE_FP32)
        \size = 4
    .elseif (\type == TYPE_FP16 || \type == TYPE_BFP16)
        \size = 2
    .endif
.endm

elemsize src_elem_size, src_type
elemsize dst_elem_size, dst_type

lds_elem_size = 4

waves_in_group = 8
tiles_per_wave = 8
tiles_per_group = waves_in_group * tiles_per_wave / 2
r_slot_size = xformy_x_size
w_slot_size = xformx_x_size

.text
.p2align 8

static_assert(xformx_d_size <= 8)
static_assert(xformy_d_size <= 8)
static_assert(xformx_f_size <= 8)
static_assert(xformy_f_size <= 8)
static_assert(xformx_x_size <= 8)
static_assert(xformy_x_size <= 8)

static_assert(fdilation_w * ddilation_w * stride_w <= 2)
static_assert(fdilation_h * ddilation_h * stride_h <= 2)

.if xform_filter
    x_elem_size     = dst_elem_size
    in_tile_width   = xformx_f_size
    in_tile_height  = xformy_f_size
    out_tile_width  = xformx_x_size
    out_tile_height = xformy_x_size
.elseif xform_data
    x_elem_size = dst_elem_size
    in_tile_width   = xformx_d_size
    in_tile_height  = xformy_d_size
    out_tile_width  = xformx_x_size
    out_tile_height = xformy_x_size
.elseif xform_output
    x_elem_size = src_elem_size
    in_tile_width   = xformx_x_size
    in_tile_height  = xformy_x_size
    out_tile_width  = xformx_o_size
    out_tile_height = xformy_o_size
.endif


.GPR_ALLOC_BEGIN
// initial state
// s[0:1] - kernarg address
// s2 - wg x (1 wg per CU)
kernarg = 0
gid_x = 2

.SGPR_ALLOC_FROM 4
// following sgprs should be allocated in strict sequence to follow kernarg layout
.SGPR_ALLOC src_addr, 2
.SGPR_ALLOC dst_addr, 2

.SGPR_ALLOC G
.SGPR_ALLOC M1
.SGPR_ALLOC M0
.SGPR_ALLOC H
.SGPR_ALLOC W

.SGPR_ALLOC n_groups
.SGPR_ALLOC flags

.SGPR_ALLOC pad_h
.SGPR_ALLOC pad_w

.SGPR_ALLOC tiles_h
.SGPR_ALLOC tiles_w

.SGPR_ALLOC stride_G
.SGPR_ALLOC stride_M1
.SGPR_ALLOC stride_M0
.SGPR_ALLOC stride_H
.SGPR_ALLOC stride_W

// end of kernarg extent
.if .SGPR_NEXT_FREE % 4
    .SGPR_ALLOC_ONCE div_pw
.endif
.if .SGPR_NEXT_FREE % 4
    .SGPR_ALLOC_ONCE div_ph
.endif
.if .SGPR_NEXT_FREE % 4
    .SGPR_ALLOC_ONCE div_m0
.endif
.SGPR_ALLOC d_desc, 4
.SGPR_ALLOC o_desc, 4
.SGPR_ALLOC stmp, 4
.SGPR_ALLOC tile_w_mask, 2
.SGPR_ALLOC_ONCE div_pw
.SGPR_ALLOC_ONCE div_ph
.SGPR_ALLOC_ONCE div_m0
.SGPR_ALLOC div_m1
.SGPR_ALLOC neg_pw
.SGPR_ALLOC neg_ph
.SGPR_ALLOC neg_m0
.SGPR_ALLOC neg_m1
.SGPR_ALLOC tile_step_x
.SGPR_ALLOC tile_step_y

.SGPR_ALLOC sync_target
.SGPR_ALLOC total_tiles
.SGPR_ALLOC write_wave
.SGPR_ALLOC shift_cnt
.SGPR_ALLOC phase_cnt
.SGPR_ALLOC x_TW_stride // tw - coordinate inside transformed tile
.SGPR_ALLOC x_TH_stride // th - coordinate inside transformed tile
.SGPR_ALLOC addr_step
.SGPR_ALLOC base_tile
.SGPR_ALLOC tiles_step
.SGPR_RESERVE_VCC

.VGPR_ALLOC_FROM 0
.VGPR_ALLOC tid
.VGPR_ALLOC lds_addr, 8
.VGPR_ALLOC dummy_volatile
.VGPR_ALLOC invalid
.VGPR_ALLOC acc, n_accums

// s only
.if .VGPR_NEXT_FREE % 2
    .VGPR_ALLOC_ONCE n8_g
.endif
.VGPR_ALLOC n8base_addr, 8
.VGPR_ALLOC n4hibase_addr, 8
vtmp_size = 8
.VGPR_ALLOC vtmp, vtmp_size
.VGPR_ALLOC_ONCE n8_g
.VGPR_ALLOC n8_w
.VGPR_ALLOC n4hi_w
.VGPR_ALLOC n8_w_prev
.VGPR_ALLOC n4hi_w_prev
.VGPR_ALLOC n8_h
.VGPR_ALLOC n8_pw // w in padded space
.VGPR_ALLOC n8_ph // h in padded space
.VGPR_ALLOC n8_m0
.VGPR_ALLOC n8_m1
.VGPR_ALLOC w_const // 0 to 7 pattern
.VGPR_ALLOC w_bconst // (0 to 7) * s_W_stride pattern
.VGPR_ALLOC tile_step_x8


// transformed wave only
.VGPR_ALLOC x_cur_tile


.LDS_ALLOC_FROM 0
lds_buf_size = tiles_per_group * 8 * 8 * lds_elem_size
.LDS_ALLOC lds_terminated, 4
.LDS_ALLOC lds_sync_cnts, 4*32
.LDS_ALLOC lds_buf,  lds_buf_size * lds_buffers


.GPR_ALLOC_END

.macro _winograd_xform_base o_size, f_size
    .if xform_data && \o_size == 2 && \f_size == 2
        v_sub_f32 v[d0], v[d0], v[d1]
        v_sub_f32 v[d2], v[d2], v[d1]
    .elseif xform_filter && \o_size == 2 && \f_size == 2
        v_mov_b32 v[d2], v[d1]
        v_add_f32 v[d1], v[d0], v[d2]
    .elseif xform_output && \o_size == 2 && \f_size == 2
        v_add_f32 v[d0], v[d0], v[d1]
        v_add_f32 v[d1], v[d1], v[d2]
    
    .elseif (xform_data && \o_size == 2 && \f_size == 3) || (xform_data && \o_size == 3 && \f_size == 2)
        v_sub_f32 v[d0], v[d0], v[d2]
        v_sub_f32 v[d3], v[d3], v[d1]
        v_sub_f32 v[t0], v[d2], v[d1]
        v_add_f32 v[d1], v[d1], v[d2]
        v_mov_b32 v[d2], v[t0]
    .elseif xform_filter && \o_size == 2 && \f_size == 3
        v_mov_b32 v[d3], v[d2]
        v_add_f32 v[t0], v[d0], v[d2]
        v_sub_f32 v[d2], v[t0], v[d1] div:2
        v_add_f32 v[d1], v[t0], v[d1] div:2
    .elseif xform_output && \o_size == 2 && \f_size == 3
        v_add_f32 v[d0], v[d1], v[d0]
        v_add_f32 v[d0], v[d2], v[d0]
        v_add_f32 v[d1], v[d3], v[d1]
        v_sub_f32 v[d1], v[d1], v[d2]
        //{ 1, 1, 1, 0 },
        //{ 0, 1,-1, 1 }


    .elseif (xform_data && \o_size == 2 && \f_size == 4) || (xform_data && \o_size == 4 && \f_size == 2)
        v_sub_f32 v[d0], v[d0], v[d2]
        v_sub_f32 v[d4], v[d4], v[d2]
        v_fma_f32 v[t0],-2.0, v[d1], v[d3]
        v_fma_f32 v[t1], 2.0, v[d1], v[d3]
        v_sub_f32 v[d3], v[d3], v[d1]
        v_fma_f32 v[d0], 2.0, v[d0], v[d3]
        v_fma_f32 v[d4],-2.0, v[d3], v[d4]
        v_sub_f32 v[d1], v[t0], v[d2]
        v_mul_f32 v[d2],-3.0, v[d2]
        v_add_f32 v[d2], v[d2], v[t1]
        //d0 { 2,-1,-2, 1, 0 },
        //d1 { 0,-2,-1, 1, 0 },
        //d2 { 0, 2,-3, 1, 0 },
        //d3 { 0,-1, 0, 1, 0 },
        //d4 { 0, 2,-1,-2, 1 },
    .elseif xform_filter && \o_size == 2 && \f_size == 4
        v_mov_b32 v[d4], v[d3]
        v_add_f32 v[t0], v[d0], v[d2]
        v_add_f32 v[t1], v[d1], v[d3]
        v_fma_f32 v[t2], 4.0, v[d2], v[d0]
        v_fma_f32 v[t3], 4.0, v[d3], v[d1]
        v_mul_f32 v[d0], 0.5, v[d0]
        v_add_f32 v[d1], v[t0], v[t1]
        v_mul_f32 v[d1],-0.5, v[d1]
        v_sub_f32 v[d2], v[t1], v[t0]
        v_mul_f32 v[d2], 0.16666666667, v[d2]
        v_fma_f32 v[d3], 2.0, v[t3], v[t2]
        v_mul_f32 v[d3], 0.16666666667, v[d3]
        //{ 0.5, 0, 0, 0 },
        //{ -0.5,   -0.5,  -0.5,  -0.5 },  {-1, -1, -1, -1} / 2
        //{-1.0/6, 1.0/6,-1.0/6, 1.0/6 },  {-1,  1, -1,  1} / 6
        //{ 1.0/6, 1.0/3, 2.0/3, 4.0/3 },  { 1,  2,  4,  8} / 6
        //{ 0, 0, 0, 1 }},
    .elseif xform_output && \o_size == 2 && \f_size == 4
        v_add_f32 v[d4], v[d1], v[d4]
        v_fma_f32 v[d4], 2.0, v[d3], v[d4]
        v_add_f32 v[d3], v[d3], v[d2]
        v_add_f32 v[d0], v[d0], v[d1]
        v_add_f32 v[d0], v[d0], v[d3]
        v_sub_f32 v[d1], v[d4], v[d2]
        //{ 1, 1, 1, 1, 0 },
        //{ 0, 1,-1, 2, 1 }

    .elseif xform_filter && \o_size == 4 && \f_size == 2
        v_sub_f32 v[d2], v[d1], v[d0]
        v_mul_f32 v[d2], 0.16666666667, v[d2]
        v_fma_f32 v[d3], 2.0, v[d1], v[d0]
        v_mul_f32 v[d3], 0.16666666667, v[d3]
        v_mov_b32 v[d4], v[d1]
        v_add_f32 v[d1], v[d1], v[d0]
        v_mul_f32 v[d0], 0.5, v[d0]
        v_mul_f32 v[d1],-0.5, v[d1]
        //{ 0.5, 0 },
        //{-0.5,-0.5 },
        //{-1.0/6, 1.0/6 },
        //{ 1.0/6, 1.0/3 },
        //{ 0, 1 }
    .elseif xform_output && \o_size == 4 && \f_size == 2
        v_add_f32 v[t0], v[d1], v[d2]
        v_add_f32 v[d0], v[d0], v[d3]
        v_add_f32 v[d0], v[d0], v[t0]
        v_sub_f32 v[t1], v[d1], v[d2]
        v_fma_f32 v[d1], 2.0, v[d3], v[t1]
        v_fma_f32 v[d2], 4.0, v[d3], v[t0]
        v_mul_f32 v[d3], 8.0, v[d3]
        v_add_f32 v[t1], v[t1], v[d4]
        v_add_f32 v[d3], v[d3], v[t1]
        //{ 1, 1, 1, 1, 0 },
        //{ 0, 1,-1, 2, 0 },
        //{ 0, 1, 1, 4, 0 },
        //{ 0, 1,-1, 8, 1 }
    
    
    .elseif xform_filter && \o_size == 3 && \f_size == 2
        v_mov_b32 v[d3], v[d1]
        v_sub_f32 v[d2], v[d0], v[d1] div:2
        v_add_f32 v[d1], v[d0], v[d1] div:2
    .elseif xform_output && \o_size == 3 && \f_size == 2
        v_add_f32 v[t0], v[d1], v[d2]
        v_sub_f32 v[d1], v[d1], v[d2]
        v_add_f32 v[d2], v[t0], v[d3]
        v_add_f32 v[d0], v[d0], v[t0]
    
    
    .elseif xform_data && \o_size == 3 && \f_size == 3
        v_sub_f32 v[d0], v[d0], v[d2] mul:2
        v_sub_f32 v[d4], v[d4], v[d2]
        v_fma_f32 v[t0], -2.0, v[d1], v[d3]
        v_fma_f32 v[t1],  2.0, v[d1], v[d3]
        v_sub_f32 v[d3], v[d3], v[d1]
        v_add_f32 v[d0], v[d0], v[d3]
        v_mac_f32 v[d4], -2.0, v[d3]
        v_sub_f32 v[d1], v[t0], v[d2]
        v_mul_f32 v[t0], -3.0, v[d2]
        v_add_f32 v[d2], v[t0], v[t1]
    .elseif xform_filter && \o_size == 3 && \f_size == 3
        v_mov_b32 v[d4], v[d2]
        v_fma_f32 v[d3], 2.0, v[d1], v[d0]
        v_mac_f32 v[d3], 4.0, v[d2]
        v_mul_f32 v[d3], 0.16666666667, v[d3] // 1/6
        v_add_f32 v[t0], v[d0], v[d2]
        v_mul_f32 v[d0], 0.5, v[d0]
        v_sub_f32 v[d2], v[d1], v[t0]
        v_mul_f32 v[d2], 0.16666666667, v[d2] // 1/6
        v_add_f32 v[d1], v[d1], v[t0]
        v_mul_f32 v[d1], -0.5, v[d1]
    .elseif xform_output && \o_size == 3 && \f_size == 3
        v_add_f32 v[t0], v[d1], v[d2]
        v_sub_f32 v[d1], v[d1], v[d2]
        v_add_f32 v[d0], v[d0], v[d3]
        v_fma_f32 v[d4], v[d3], 4.0, v[d4]
        v_add_f32 v[d0], v[t0], v[d0]
        v_fma_f32 v[d1], v[d3], 2.0, v[d1]
        v_add_f32 v[d2], v[d4], v[t0]
        //{ 1, 1, 1, 1, 0 },
        //{ 0, 1,-1, 2, 0 },
        //{ 0, 1, 1, 4, 1 }
    
    
    .elseif (xform_data && \o_size == 4 && \f_size == 3) || (xform_data && \o_size == 3 && \f_size == 4)
        v_fma_f32 v[d0], 4.0, v[d0], v[d4]
        v_mac_f32 v[d0], -5.0, v[d2]
        v_fma_f32 v[d5], 4.0, v[d1], v[d5]
        v_mac_f32 v[d5], -5.0, v[d3]
    
        v_sub_f32 v[t0], v[d3], v[d1] mul:2
        v_sub_f32 v[t1], v[d4], v[d2]
        v_fma_f32 v[t2], -4.0, v[d2], v[d4]
            
        v_fma_f32 v[d1], -4.0, v[d1], v[d3]
        v_sub_f32 v[d2], v[t2], v[d1]
        v_add_f32 v[d1], v[t2], v[d1]
        v_add_f32 v[d3], v[t1], v[t0]
        v_sub_f32 v[d4], v[t1], v[t0]
    .elseif xform_filter && \o_size == 4 && \f_size == 3
        v_add_f32 v[t2], v[d0], v[d2]
        v_mul_f32 v[d0], 0.25, v[d0]
        v_mov_b32 v[d5], v[d2]
        v_fma_f32 v[d3], 0.5, v[d1], v[d0]
        v_add_f32 v[d3], v[d3], v[d2]
        v_mov_b32 v[t0], 0.16666666667
        v_mul_f32 v[t1], -0.16666666667, v[d1]
        v_fma_f32 v[d4], v[t0], v[d3], v[t1]
        v_sub_f32 v[d3], v[d4], v[t1]
        v_fma_f32 v[d1], neg(v[t0]), v[t2], v[t1]
        v_fma_f32 v[d2], -2.0, v[t1], v[d1]
        //{ 0.25, 0, 0 },
		//{-1.0/6,-1.0/6, -1.0/6 },   {-1, -1, -1} / 6
		//{-1.0/6, 1.0/6, -1.0/6 },   {-1,  1, -1} / 6
		//{ 1.0/24, 1.0/12, 1.0/6 },  { 1,  2,  4} / 24
		//{ 1.0/24,-1.0/12, 1.0/6 },  { 1, -2,  4} / 24
		//{ 0, 0, 1 },
    .elseif xform_output && \o_size == 4 && \f_size == 3
        v_add_f32 v[t0], v[d1], v[d2]
        v_add_f32 v[t1], v[d3], v[d4]
        v_sub_f32 v[t2], v[d1], v[d2]
        v_sub_f32 v[t3], v[d3], v[d4]
        v_add_f32 v[d0], v[t0], v[d0]
        v_add_f32 v[d0], v[t1], v[d0]
        v_fma_f32 v[d1], 2.0, v[t3], v[t2]
        v_fma_f32 v[d2], 4.0, v[t1], v[t0]
        v_mul_f32 v[t3], 8.0, v[t3]
        v_add_f32 v[d3], v[t2], v[d5]
        v_add_f32 v[d3], v[t3], v[d3]
        //{ 1, 1, 1, 1, 1, 0 },
        //{ 0, 1,-1, 2,-2, 0 },
        //{ 0, 1, 1, 4, 4, 0 },
        //{ 0, 1,-1, 8,-8, 1 }

    .elseif xform_filter && \o_size == 3 && \f_size == 4
        v_mov_b32 v[d5], v[d3]
        v_add_f32 v[t0], v[d0], v[d2]
        v_add_f32 v[t1], v[d1], v[d3]
        v_fma_f32 v[t2], 4.0, v[d2], v[d0]
        v_fma_f32 v[t3], 4.0, v[d3], v[d1]
        v_mul_f32 v[d0], 0.25, v[d0]
        v_add_f32 v[d1], v[t0], v[t1]
        v_mul_f32 v[d1], -0.16666666667, v[d1]
        v_sub_f32 v[d2], v[t1], v[t0]
        v_mul_f32 v[d2], 0.16666666667, v[d2]
        v_fma_f32 v[d3], 0.5, v[t2], v[t3]
        v_fma_f32 v[d4],-0.5, v[t2], v[t3]
        v_mul_f32 v[d3], 0.08333333333, v[d3]
        v_mul_f32 v[d4],-0.08333333333, v[d4]
        //d0 { 0.25, 0, 0, 0 },
        //d1 {-1.0/6, -1.0/6, -1.0/6,-1.0/6 },  {-1, -1, -1, -1} / 6
        //d2 {-1.0/6,  1.0/6, -1.0/6, 1.0/6 },  {-1,  1, -1,  1} / 6
        //d3 { 1.0/24, 1.0/12, 1.0/6, 1.0/3 },  { 1,  2,  4,  8} / 24
        //d4 { 1.0/24,-1.0/12, 1.0/6,-1.0/3 },  { 1, -2,  4, -8} / 24
        //d5 { 0, 0, 0, 1 },
        
    .elseif xform_output && \o_size == 3 && \f_size == 4
        v_add_f32 v[t0], v[d1], v[d2]
        v_add_f32 v[t1], v[d3], v[d4]
        v_sub_f32 v[t2], v[d1], v[d2]
        v_sub_f32 v[t3], v[d3], v[d4]
        v_add_f32 v[d0], v[t0], v[d0]
        v_add_f32 v[d0], v[t1], v[d0]
        v_fma_f32 v[d1], 2.0, v[t3], v[t2]
        v_add_f32 v[t0], v[t0], v[d5]
        v_fma_f32 v[d2], 4.0, v[t1], v[t0]
        //{ 1, 1, 1, 1, 1, 0 },
		//{ 0, 1,-1, 2,-2, 0 },
		//{ 0, 1, 1, 4, 4, 1 }
    
    .elseif (xform_data && \o_size == 5 && \f_size == 3) || (xform_data && \o_size == 3 && \f_size == 5)
        v_mov_b32 v[t3], 1.5
        v_mov_b32 v[t4], -2.5
        v_mov_b32 v[t5], -3.5
        v_mov_b32 v[t6], -4.5
        v_mov_b32 v[t7], -5.0
    
        v_fma_f32 v[d0], 4.0, v[d0], v[d4]
        v_fma_f32 v[d0], v[t7], v[d2], v[d0] //fmac -5 // D0 = 0.5 * d0 - D5
        v_fma_f32 v[d6], 4.0, v[d2], v[d6] //fmac
        v_fma_f32 v[d6], v[t7], v[d4], v[d6] //fmac -5 // D6 = d6 - 0.5 * D5
    
        v_sub_f32 v[t0], v[d4], v[d2]
        v_fma_f32 v[t1], -4.0, v[d2], v[d4]
        v_fma_f32 v[d2], -2.0, v[d1], v[d5]
        v_fma_f32 v[d2], v[t5], v[d3], v[d2] //fmac -3.5
        v_fma_f32 v[d2], neg(v[t3]), v[t1], v[d2] //fmac -1.5
        v_fma_f32 v[t1], 0.5, v[t1], v[d5]
        v_fma_f32 v[t1], v[t6], v[d3], v[t1] //fmac -4.5 // D1 = 2 * d1 + t1
        v_fma_f32 v[t2], -2.0, v[d3], v[d5]
        v_fma_f32 v[t2], v[t3], v[t0], v[t2] //fmac 1.5 // D3 = d1 + t2
    
        v_sub_f32 v[d4], v[d5], v[d1]
        v_fma_f32 v[d4], v[t4], v[t0], v[d4] //fmac -2.5
        v_fma_f32 v[d5], 4.0, v[d1], v[d5] //fmac
        v_fma_f32 v[d5], v[t7], v[d3], v[d5] //fmac -5
        
        v_fma_f32 v[d0], 0.5, v[d0], neg(v[d5])
        v_fma_f32 v[d6], -0.5, v[d5], v[d6] //fmac
        v_add_f32 v[d3], v[d1], v[t2]
        v_fma_f32 v[d1], 2.0, v[d1], v[t1]
    
        //     0   1    2      3    4      5  6
        //	 
        //0  { 2, -4, -2.5,    5,  0.5,   -1, 0 },
        //5  { 0,  4,    0,   -5,    0,    1, 0 },
        //6  { 0, -2,    4,  2.5,   -5, -0.5, 1 }},
        //   
        //1  { 0,  2,   -2, -4.5,  0.5,    1, 0 },
        //2  { 0, -2,    6, -3.5, -1.5,    1, 0 },
        //   
        //3  { 0,  1, -1.5,   -2,  1.5,    1, 0 },
        //4  { 0, -1,  2.5,    0, -2.5,    1, 0 },
        //
        // 21 + 5c
        //d0 = 4*d0 + d4
        //d0 += -5 * d2 // d0 = 0.5 * d0 - d5
        //d6 += 4 * d2
        //d6 += -5 * d4 // d6 += -0.5 * d5
        //
        //t0 = d4 - d2
        //t1 = -4 * d2 + d4
        //d2 = -2 * d1 + d5
        //d2 += -3.5 * d3
        //d2 += -1.5 * t1
        //t1 = 0.5 * t1 + d5
        //t1 += -4.5 * d3 // d1 = 2 * d1 + t1
        //
        //t3 = -2 * d3 + d5
        //t3 += 1.5 * t0 // d3 = d1 + t3
        //d4 = d5 - d1
        //d4 += -2.5 * t0
        //
        //d5 += 4 * d1
        //d5 += -5 * d3
        //
        //d0 = 0.5 * d0 - d5
        //d6 += -0.5 * d5
        //d3 = d1 + t3
        //d1 = 2 * d1 + t1
    .elseif xform_filter && \o_size == 5 && \f_size == 3
        //d0{  0.5,       0,       0    },
        //d1{ -1.0/3,  -1.0/3,  -1.0/3  }, {1, 1, 1} * -1/3
        //d2{  1.0/9,  -1.0/9,   1.0/9  }, [1, -1, 1} * 1/9
        //d3{  1.0/36,  1.0/18,  1.0/9  }, {1/4, 1/2, 1/1} * 1/9
        //d4{ -1.0/60,  1.0/30, -1.0/15 }, {-1/4, 1/2, -1/1} * 1/15
        //d5{ 32.0/45, 16.0/45,  8.0/45 }, {4, 2, 1} * 8/45
        //d6{    0,       0,       1    }},
    
        v_add_f32 v[t0], v[d0], v[d2]
    
        v_mov_b32 v[d6], v[d2]
        v_fma_f32 v[d5], 2.0, v[d1], v[d2]
        v_fma_f32 v[d5], 4.0, v[d0], v[d5]
        v_mul_f32 v[d5], 0.17777777778, v[d5]
    
        v_mul_f32 v[d0], 0.5, v[d0]
        v_sub_f32 v[d4], v[d1], v[d0]
        v_fma_f32 v[d4], -0.5, v[d4], v[d2]
        v_add_f32 v[d3], v[d1], v[d4]
    
        v_mul_f32 v[d4], -0.0666666667, v[d4]
        v_mul_f32 v[d3], 0.1111111111, v[d3]
    
        v_sub_f32 v[d2], v[t0], v[d1]
        v_add_f32 v[d1], v[t0], v[d1]
    
        v_mul_f32 v[d1], -0.3333333333, v[d1]
        v_mul_f32 v[d2], 0.1111111111, v[d2]
    .elseif xform_output && \o_size == 5 && \f_size == 3
        v_add_f32 v[t0], v[d1], v[d2]
        v_add_f32 v[t1], v[d3], v[d4]
        v_sub_f32 v[t2], v[d1], v[d2]
        v_sub_f32 v[t3], v[d3], v[d4]
        v_add_f32 v[d0], v[d5], v[d0]
        v_add_f32 v[d0], v[t0], v[d0]
        v_add_f32 v[d0], v[t1], v[d0]
        
        v_fma_f32 v[d1], 0.5, v[d5], v[t2]
        v_fma_f32 v[d1], 2.0, v[t3], v[d1]
    
        v_mul_f32 v[d5], 0.25, v[d5]
        v_add_f32 v[d2], v[d5], v[t0]
        v_mul_f32 v[t1], 4.0, v[t1]
        v_add_f32 v[d2], v[t1], v[d2]
    
        v_mul_f32 v[d5], 0.5, v[d5]
        v_add_f32 v[d3], v[t2], v[d5]
        v_mul_f32 v[t3], 8.0, v[t3]
        v_add_f32 v[d3], v[t3], v[d3]
    
        v_fma_f32 v[d4], 0.5, v[d5], v[d6]
        v_add_f32 v[d4], v[t0], v[d4]
        v_fma_f32 v[d4], 4.0, v[t1], v[d4]
        //    0  1  2   3   4    5   6
        //0 { 1, 1, 1,  1,  1,   1,  0 },   0 { 1,  t0,    t1,     1,   0 }
        //1 { 0, 1,-1,  2, -2, 1/2,  0 },   1 { 0,  t2,   2*t3,   1/2,  0 }
        //2 { 0, 1, 1,  4,  4, 1/4,  0 },   2 { 0,  t0,   4*t1,   1/4,  0 }
        //3 { 0, 1,-1,  8, -8, 1/8,  0 },   3 { 0,  t2,   8*t3,   1/8,  0 }
        //4 { 0, 1, 1, 16, 16, 1/16, 1 }    4 { 0,  t0,  16*t1,  1/16,  1 }
    
    .elseif (xform_data && \o_size == 6 && \f_size == 3) || (xform_data && \o_size == 3 && \f_size == 6)
        v_mov_b32 v[t7], 5.25
        v_sub_f32 v[t0], v[d4], v[d2]
        v_sub_f32 v[d0], v[d0], v[d6]
        v_fma_f32 v[d0], v[t7], v[t0], v[d0] //fmac 5.25
        v_sub_f32 v[t0], v[d3], v[d5]
        v_sub_f32 v[d7], v[d7], v[d1]
        v_fma_f32 v[d7], v[t7], v[t0], v[d7] //fmac 5.25
    
        v_mov_b32 v[t7], -4.25
        v_add_f32 v[t0], v[d1], v[d5]
        v_fma_f32 v[t0], v[t7], v[d3], v[t0] //fmac -4.25
        v_add_f32 v[t1], v[d2], v[d6]
        v_fma_f32 v[t1], v[t7], v[d4], v[t1] //fmac -4.25 // d1 = t1 + t0; d2 = t1 - t0
    
        v_mov_b32 v[t6], 0.25
        v_fma_f32 v[t3], v[t6], v[d2], v[d6]
        v_mov_b32 v[t7], -1.25
        v_fma_f32 v[t3], v[t7], v[d4], v[t3] //fmac -1.25
        v_fma_f32 v[t2], 4.0, v[d5], v[d1]
        v_mov_b32 v[t7], -5.0
        v_fma_f32 v[t2], v[t7], v[d3], v[t2] //fmac -5.0 // d3 = 0.5 * t2 + t3; d4 = -0.5 * t2 + t3
    
        v_fma_f32 v[t4], 4.0, v[d2], v[d6]
        v_fma_f32 v[t4], v[t7], v[d4], v[t4] //fmac -5.0
        v_fma_f32 v[t5], 4.0, v[d1], v[d5]
        v_fma_f32 v[t5], v[t7], v[d3], v[t5] //fmac -5.0
    
        v_fma_f32 v[d5],  0.5, v[t5], v[t4]
        v_fma_f32 v[d6], -0.5, v[t5], v[t4]
        v_fma_f32 v[d3],  0.5, v[t2], v[t3]
        v_fma_f32 v[d4], -0.5, v[t2], v[t3]
        v_add_f32 v[d1], v[t1], v[t0]
        v_sub_f32 v[d2], v[t1], v[t0]
        // 24 + 5c
        //t0 = d4 - d2
        //d0 = d0 - d6
        //d0 += 5.25 * t0
        //t0 = d3 - d5
        //d7 = d7 - d1
        //d7 += 5.25 * t0
        //
        //t0 = d1 + d5
        //t0 += -4.25 * d3
        //t1 = d2 + d6
        //t1 += -4.25 * d4 // d1 = t1 + t0; d2 = t1 - t0
        //
        //t3 = d6 + 0.25 * d2
        //t3 += -1.25 * d4
        //t2 = d1 + 4 * d5
        //t2 += -5 * d3 // d3 = 0.5 * t2 + t3;  d4 = -0.5 * t2 + t3
        //
        //t4 = d6 + 4 * d2
        //t4 += -5 * d4
        //t5 = d5 + 4 * d1
        //t5 += -5 * d3
        //
        //d5 = t4 + 0.5 * t5
        //d6 = t4 - 0.5 * t5
        //d3 = 0.5 * t2 + t3
        //d4 = -0.5 * t2 + t3
        //d1 = t1 + t0
        //d2 = t1 - t0
        //    0    1      2      3      4      5   6  7
        //0 { 1,    0, -5.25,     0,  5.25,    0, -1, 0 },
        //1 { 0,    1,     1, -4.25, -4.25,    1,  1, 0 },
        //2 { 0,   -1,     1,  4.25, -4.25,   -1,  1, 0 },
        //3 { 0,  0.5,  0.25,  -2.5, -1.25,    2,  1, 0 },
        //4 { 0, -0.5,  0.25,   2.5, -1.25,   -2,  1, 0 },
        //5 { 0,    2,     4,  -2.5,    -5,  0.5,  1, 0 },
        //6 { 0,   -2,     4,   2.5,    -5, -0.5,  1, 0 },
        //7 { 0,   -1,     0,  5.25,     0,-5.25,  0, 1 }},
    .elseif xform_filter && \o_size == 6 && \f_size == 3
        //0 {     1,       0,      0    },
        //1 {  -2.0/9,  -2.0/9, -2.0/9  }, {1,  1,  1} *-2/9
        //2 {  -2.0/9,   2.0/9, -2.0/9  }, {1, -1,  1} *-2/9
        //3 {  1.0/90,   1.0/45, 2.0/45 }, {1,  2,  4} * 1/90
        //4 {  1.0/90,  -1.0/45, 2.0/45 }, {1  -2,  4} * 1/90
        //5 { 32.0/45,  16.0/45, 8.0/45 }, {4,  2,  1} * 16/90
        //6 { 32.0/45, -16.0/45, 8.0/45 }, {4, -2,  1} * 16/90
        //7 {    0,        0,      1    }},
    
        v_mov_b32 v[d7], v[d2]
    
        v_fma_f32 v[t0], 4.0, v[d0], v[d2]
        v_fma_f32 v[d6],-2.0, v[d1], v[t0]
        v_mul_f32 v[d6], 0.17777777777, v[d6]
        v_fma_f32 v[d5], 2.0, v[d1], v[t0]
        v_mul_f32 v[d5], 0.17777777777, v[d5]
    
        v_fma_f32 v[t0], 4.0, v[d2], v[d0]
        v_fma_f32 v[d4],-2.0, v[d1], v[t0]
        v_mul_f32 v[d4], 0.01111111111, v[d4]
        v_fma_f32 v[d3], 2.0, v[d1], v[t0]
        v_mul_f32 v[d3], 0.01111111111, v[d3]
    
        v_add_f32 v[t0], v[d0], v[d2]
        v_sub_f32 v[d2], v[t0], v[d1]
        v_mul_f32 v[d2],-0.22222222222, v[d2]
        v_add_f32 v[d1], v[t0], v[d1]
        v_mul_f32 v[d1],-0.22222222222, v[d1]
    .elseif xform_output && \o_size == 6 && \f_size == 3
        v_add_f32 v[t0], v[d1], v[d2]
        v_add_f32 v[t1], v[d3], v[d4]
        v_add_f32 v[t2], v[d5], v[d6]
        v_add_f32 v[t3], v[t1], v[t2]
        v_add_f32 v[d0], v[t0], v[d0]
        v_add_f32 v[d0], v[t3], v[d0]
        v_sub_f32 v[t3], v[d1], v[d2]
        v_sub_f32 v[t4], v[d3], v[d4]
        v_sub_f32 v[t5], v[d5], v[d6]
    
        v_fma_f32 v[d1], 0.5, v[t5], v[t3]
        v_fma_f32 v[d1], 2.0, v[t4], v[d1]
    
        v_mul_f32 v[t2], 0.25, v[t2]
        v_add_f32 v[d2], v[t0], v[t2]
        v_mul_f32 v[t1], 4.0, v[t1]
        v_add_f32 v[d2], v[t1], v[d2]
    
        v_mul_f32 v[t5], 0.125, v[t5]
        v_add_f32 v[d3], v[t3], v[t5]
        v_mul_f32 v[t4], 8.0, v[t4]
        v_add_f32 v[d3], v[t4], v[d3]
        
        v_mul_f32 v[t2], 0.25, v[t2]
        v_add_f32 v[d4], v[t0], v[t2]
        v_fma_f32 v[d4], 4.0, v[t1], v[d4]
    
        v_mul_f32 v[t5], 0.25, v[t5]
        v_add_f32 v[d5], v[d7], v[t5]
        v_add_f32 v[d5], v[t3], v[d5]
        v_fma_f32 v[d5], 4.0, v[t4], v[d5]
    
        //    0  1  2   3    4    5      6    7
        //0 { 1, 1, 1,  1,   1,   1,     1,   0 },   0 { 1,  t0,   t1,     t2,   0 }
        //1 { 0, 1,-1,  2,  -2,  1/2,  -1/2,  0 },   1 { 0,  t3,  2*t4,   t5/2,  0 }
        //2 { 0, 1, 1,  4,   4,  1/4,   1/4,  0 },   2 { 0,  t0,  4*t1,   t2/4,  0 }  
        //3 { 0, 1,-1,  8,  -8,  1/8,  -1/8,  0 },   3 { 0,  t3,  8*t4,   t5/8,  0 }
        //4 { 0, 1, 1, 16,  16,  1/16,  1/16, 0 },   4 { 0,  t0, 16*t1,  t2/16,  0 }
        //5 { 0, 1,-1, 32, -32,  1/32, -1/32, 1 }    5 { 0,  t3, 32*t4,  t5/32,  1 }
    
    .elseif \o_size == 1
        .irp i,1,2,3,4,5,6,7
            .if xform_output && (\i < \f_size)
                v_add_f32 v[d0], v[d0], v[d\i]
            .endif
        .endr
    .elseif \f_size == 1
        .irp i,1,2,3,4,5,6,7
            .if xform_filter && (\i < \o_size)
                v_mov_b32 v[d\i], v[d0]
            .endif
        .endr
    .else
        static_assert(0)
    .endif
.endm

.macro _enumerate_d x_size, base_gpr, step, off
    .irp i,0,1,2,3,4,5,6,7
        _off = \step * \i + \off
        .if \i < (\x_size)
            d\i = \base_gpr + _off
        .else
            d\i = -1
        .endif
    .endr
.endm

.macro winograd_xform o_size, f_size, x_size, ddil, odd_flag, fdil, strd, base_gpr, vtmp, mirror
    .irp i,0,1,2,3,4,5,6,7
        .if \i < vtmp_size
            t\i = \vtmp + \i
        .endif
    .endr
    .if \mirror
        static_assert(xform_filter)
        i = 0
        .rept \f_size / 2
            v_swap_b32 v[\base_gpr + i], v[\base_gpr + \f_size - i - 1]
            i = i + 1
        .endr
    .endif

    .if \fdil == 2
        o2 = \o_size / 2
        o1 = \o_size - o2
        .if xform_filter // copy filter
            i = 2 * \f_size - 1
            .rept 2 * \f_size - 1
                v_mov_b32 v[\base_gpr + i], v[\base_gpr + i / 2]
                i = i - 1
            .endr
        .endif
        _enumerate_d \x_size, \base_gpr, 2, 0
        _winograd_xform_base o1, \f_size
        _enumerate_d \x_size, \base_gpr, 2, 1
        _winograd_xform_base o2, \f_size
    .elseif \strd == 2
        f2 = \f_size / 2
        f1 = \f_size - f2
        _enumerate_d \x_size, \base_gpr, 2, 0
        _winograd_xform_base \o_size, f1
        _enumerate_d \x_size, \base_gpr, 2, 1
        _winograd_xform_base \o_size, f2
        .if xform_output // merge output
            i = 0
            .rept \o_size
                v_add_f32 v[\base_gpr + i], v[\base_gpr + 2 * i], v[\base_gpr + 2 * i + 1]
                i = i + 1;
            .endr
        .endif
        //v_add_f32 v[\base_gpr + \x_size - 1], 1.0, v[\base_gpr]
    .elseif \ddil == 2
        f2 = \f_size / 2
		f1 = \f_size - f2
		o2even = \o_size / 2
		o1even = \o_size - o2even
		o1odd = \o_size / 2
		o2odd = \o_size - o1odd

        s_bitcmp1_b32 s[flags], \odd_flag
        s_cbranch_scc1 ddil_odd_start\@
ddil_even_start\@:

        .if xform_data // copy input data
            i = \x_size
            .rept \x_size
                v_mov_b32 v[\base_gpr +  i], v[\base_gpr + (i + 1) / 2]
                i = i - 1
            .endr
        .endif
        _enumerate_d \x_size, \base_gpr, 2, 0
        _winograd_xform_base o1even, f1
        _enumerate_d \x_size, \base_gpr, 2, 1
        _winograd_xform_base o2even, f2
        s_branch ddil_end\@
ddil_odd_start\@:
        .if xform_data // copy input data
            i = \x_size
            .rept \x_size
                v_mov_b32 v[\base_gpr +  i], v[\base_gpr + (i + 0) / 2]
                i = i - 1
            .endr
        .endif
        _enumerate_d \x_size, \base_gpr, 2, 0
        _winograd_xform_base o1odd, f1
        _enumerate_d \x_size, \base_gpr, 2, 1
        _winograd_xform_base o2odd, f2
        .if xform_output
            i = 0
            .rept \o_size / 2
                v_swap_b32 v[\base_gpr + 2 * i], v[\base_gpr + 2 * i + 1]
                i = i + 1
            .endr
        .endif

ddil_end\@:

    .else
        _enumerate_d \x_size, \base_gpr, 1, 0
        _winograd_xform_base \o_size, \f_size
    .endif

.endm

.macro kernel_begin  x_o_size, y_o_size, x_f_size, y_f_size
    .if (xform_filter)
        .globl miopenGcnAsmMPAnydirectWinogradXformFilter_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
        .p2align 8
        .type miopenGcnAsmMPAnydirectWinogradXformFilter_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size,@function
        miopenGcnAsmMPAnydirectWinogradXformFilter_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size:
    .elseif (xform_data)
        .globl miopenGcnAsmMPAnydirectWinogradXformData_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
        .p2align 8
        .type miopenGcnAsmMPAnydirectWinogradXformData_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size,@function
        miopenGcnAsmMPAnydirectWinogradXformData_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size:
    .elseif (xform_output)
        .globl miopenGcnAsmMPAnydirectWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
        .p2align 8
        .type miopenGcnAsmMPAnydirectWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size,@function
        miopenGcnAsmMPAnydirectWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size:
    .endif
.endm

kernel_begin  %xformx_o_size, %xformy_o_size, %xformx_f_size, %xformy_f_size

    
//ASMBEGIN

    s_load_dwordx16 s[src_addr:stride_G], s[kernarg:kernarg+1], 0x0
    s_load_dwordx4 s[stride_M1:stride_W], s[kernarg:kernarg+1], 0x4 * 16

    
    // init sync counters and determine wave type (read/write)
    v_mov_b32 v[invalid], 0x80000000
    s_lshl_b32 s[shift_cnt], 1, rw_pipe_shift
    s_mov_b32 s[phase_cnt], 0
    v_cmp_lt_u32 vcc, wave_size, v[tid]
    s_cbranch_vccnz skip_lds_init
    v_lshlrev_b32 v[vtmp], 2, v[tid]
    v_mov_b32 v[vtmp+1], 0
    ds_write_b32 v[vtmp], v[vtmp+1] // init sync counters with 0
skip_lds_init:
    v_cmp_le_u32 vcc, wave_size * waves_in_group / 2, v[tid]
    s_cmp_eq_u32 vcc_lo, 0
    s_cselect_b32 s[write_wave], 0, 1
    s_mov_b32 s[sync_target], 0
    s_barrier
    s_waitcnt 0

    // set flags and swap N and C if needed
    s_and_b32 s[flags], external_flags_mask, s[flags] // clear internally set flags (if any)
    s_and_b32 s[stmp], 0x1, s[pad_w]
    s_lshl_b32 s[stmp], s[stmp], L_F_ODD_PAD_W
    s_add_u32 s[flags], s[stmp], s[flags]
    s_and_b32 s[stmp], 0x1, s[pad_h]
    s_lshl_b32 s[stmp], s[stmp], L_F_ODD_PAD_H
    s_add_u32 s[flags], s[stmp], s[flags]
    s_bitcmp1_b32 s[flags], L_F_SWAP_NC
    s_cbranch_scc0 skip_swap_nc
    s_swap s[M1], s[M0], s[stmp]
    s_swap s[stride_M1], s[stride_M0], s[stmp]
skip_swap_nc:

    .if xform_filter
        s_mov_b32 s[tile_step_x], xformx_f_size
        s_mov_b32 s[tile_step_y], xformy_f_size
    .elseif xform_data
        s_bitcmp1_b32 s[flags], L_F_SINGLE_OTILE_W
        s_cselect_b32 s[tile_step_x], (xformx_f_size * fdilation_w) / ddilation_w, (xformx_o_size * stride_w) / ddilation_w
        s_bitcmp1_b32 s[flags], L_F_SINGLE_OTILE_H
        s_cselect_b32 s[tile_step_y], (xformy_f_size * fdilation_h) / ddilation_h, (xformy_o_size * stride_h) / ddilation_h
        s_ashr_i32 s[pad_w], s[pad_w], ddilation_w - 1
        s_ashr_i32 s[pad_h], s[pad_h], ddilation_h - 1
    .elseif xform_output
        s_mov_b32 s[tile_step_x], xformx_o_size
        s_mov_b32 s[tile_step_y], xformy_o_size
    .endif

    // base_tile id
    s_mul_i32 s[base_tile], tiles_per_group, s[gid_x]
    s_mul_i32 s[tiles_step], tiles_per_group, s[n_groups]


    // early exit
    err = stmp+3
    s_mov_b32 s[err], 0
    s_mul_i32 s[stmp], s[tiles_w], s[tiles_h] // known to fit in 32 bit
    s_mul_i32 s[stmp+1], s[M0], s[M1] // known to fit in 32 bit
    s_mul_hi_u32 s[stmp+2], s[stmp], s[stmp+1]
    s_cmp_gt_u32 s[stmp+2], 0
    s_cmov_b32 s[err], 1
    s_mul_i32 s[stmp], s[stmp], s[stmp+1] // tiles_w * tiles_h * M0 * M1
    s_mul_hi_u32 s[stmp+2], s[stmp], s[G]
    s_cmp_gt_u32 s[stmp+2], 0
    s_cmov_b32 s[err], 1
    s_mul_i32 s[total_tiles], s[stmp], s[G] // total number of tiles
    s_cmp_ge_u32 s[base_tile], s[total_tiles]
    s_cmov_b32 s[err], 1
    u16limit = stmp+2
    s_mov_b32 s[u16limit], 1<<16
    s_cmp_ge_u32 s[H], s[u16limit]
    s_cmov_b32 s[err], 1
    s_cmp_ge_u32 s[W], s[u16limit]
    s_cmov_b32 s[err], 1
    s_cmp_ge_u32 s[M0], s[u16limit]
    s_cmov_b32 s[err], 1
    s_cmp_ge_u32 s[M1], s[u16limit]
    s_cmov_b32 s[err], 1
    s_cmp_ge_u32 s[G], s[u16limit]
    s_cmov_b32 s[err], 1

    s_ashr_i32 s[stmp], s[pad_w], 16 // check if high bits are all 1 or 0
    s_addk_i32 s[stmp], 1 
    s_cmp_gt_u32 s[stmp], 1
    s_cmov_b32 s[err], 1

    s_ashr_i32 s[stmp], s[pad_h], 16 // check if high bits are all 1 or 0
    s_addk_i32 s[stmp], 1 
    s_cmp_gt_u32 s[stmp], 1
    s_cmov_b32 s[err], 1

    s_cmp_ge_u32 s[tiles_step], s[u16limit]
    s_cmov_b32 s[err], 1
    s_cmp_eq_u32 s[err], 1
    s_cbranch_scc1 endpgm
    .GPR_INVALIDATE err
    .GPR_INVALIDATE u16limit

    // construct buffer descriptors
    // size covers whole buffer
    s_mov_b64 s[d_desc:d_desc+1], s[src_addr:src_addr+1]
    s_mov_b64 s[o_desc:o_desc+1], s[dst_addr:dst_addr+1]
    s_mov_b32 s[d_desc+3], 0x00020000
    s_mov_b32 s[o_desc+3], 0x00020000
    s_mul_i32 s[stmp], s[H], s[W]
    s_mul_i32 s[stmp], s[stmp], s[M0]
    s_mul_i32 s[stmp], s[stmp], s[M1]
    s_mul_i32 s[stmp], s[stmp], s[G]
    s_mul_i32 s[stmp+1], xformx_x_size * xformy_x_size, s[total_tiles]
    .if xform_output
        s_mul_i32 s[d_desc+2], src_elem_size, s[stmp+1]
        s_mul_i32 s[o_desc+2], dst_elem_size, s[stmp]
    .else
        s_mul_i32 s[d_desc+2], src_elem_size, s[stmp]
        s_mul_i32 s[o_desc+2], dst_elem_size, s[stmp+1]
    .endif

    s_cmp_eq_u32 s[write_wave], 1
    s_cbranch_scc1 lab_write_wave


    .macro send_token sync_slot
        ds_append v[dummy_volatile] offset:lds_sync_cnts + 4 * \sync_slot
    .endm

    .macro wait_token sync_slot, target
        s_mov_b32 m0, 0x20000 + lds_sync_cnts + 4 * \sync_slot
        s_nop 0
        lab_sync\@:
        v_cmp_eq_u32 vcc, src_lds_direct, s[\target]
        s_nop 5
        s_cbranch_vccz lab_sync\@
    .endm

    .macro update_sync_target target
        s_add_u32 s[\target], waves_in_group * wave_size, s[\target]
    .endm

    .macro init_x_lds_addr num_addr
        t  = vtmp
        tl = vtmp + 1
        h  = vtmp + 2
        v_and_b32 v[t], 0x1f, v[tid]
        v_lshlrev_b32 v[tl], 2, v[t]
        v_lshlrev_b32 v[t], 8, v[t]
        v_bfe_u32 v[h], v[tid], 5, 3
        v_lshl_add_u32 v[lds_addr], v[h], 5, v[t]
        i = 0
        .rept \num_addr - 1
            v_add_u32 v[lds_addr + i + 1], 4, v[lds_addr + i]
            v_xor_b32 v[lds_addr + i], v[tl], v[lds_addr + i]
            i = i + 1
        .endr
        v_xor_b32 v[lds_addr + i], v[tl], v[lds_addr + i]
    .endm

    .macro init_s_lds_addr num_addr
        t  = vtmp
        tl = vtmp + 1
        w  = vtmp + 2
        hs = stmp
        v_and_b32 v[w], 0x7, v[tid]
        v_lshrrev_b32 v[t], 3, v[tid]
        v_and_b32 v[t], 0x1f, v[t]
        v_lshlrev_b32 v[t], 6, v[t]
        v_bfe_u32 v[tl], v[t], 4, 7
        v_add_lshl_u32 v[lds_addr], v[t], v[w], 2
        s_lshl_b32 s[hs], 1, 5
        i = 0
        .rept \num_addr - 1
            v_add_u32 v[lds_addr + i + 1], s[hs], v[lds_addr + i]
            v_xor_b32 v[lds_addr + i], v[tl], v[lds_addr + i]
            i = i + 1
        .endr
        v_xor_b32 v[lds_addr + i], v[tl], v[lds_addr + i]
    .endm

    .macro init_x_mem_addr
        v_and_b32 v[x_cur_tile], 0x1f, v[tid]
        v_add_u32 v[x_cur_tile], s[base_tile], v[x_cur_tile]
        s_mul_i32 s[x_TW_stride], s[G], s[total_tiles]
        s_mul_i32 s[x_TW_stride], x_elem_size, s[x_TW_stride]
        s_mul_i32 s[x_TH_stride], xformx_x_size, s[x_TW_stride]
        s_mul_i32 s[addr_step], tiles_per_group * x_elem_size, s[n_groups]
        v_mul_u32_u24 v[n8base_addr], x_elem_size, v[x_cur_tile]
        v_bfe_u32 v[vtmp], v[tid], 5, 3
        v_mad_u64_u32 v[n8base_addr:n8base_addr+1], vcc, v[vtmp], s[x_TH_stride], v[n8base_addr:n8base_addr+1]
        v_cmp_lt_u32 vcc, v[vtmp], xformy_x_size
        v_cndmask_b32 v[n8base_addr], v[invalid], v[n8base_addr], vcc
    .endm

    .macro init_s_mem_addr
        // compute constants
        v_and_b32 v[w_const], 7, v[tid]
        v_mul_u32_u24 v[w_bconst], s[stride_W], v[w_const]
        .if xform_output
            v_cmp_lt_u32 s[tile_w_mask:tile_w_mask+1], v[w_const], out_tile_width
        .else
            v_cmp_lt_u32 s[tile_w_mask:tile_w_mask+1], v[w_const], in_tile_width
        .endif
        v_lshlrev_b32 v[tile_step_x8], 3, s[tile_step_x]

        // compute div magic numbers
        s_mul_i32 s[stmp],   s[tile_step_x], s[tiles_w]
        s_mul_i32 s[stmp+1], s[tile_step_y], s[tiles_h]
        s_sub_i32 s[neg_m0], 0, s[M0]
        s_sub_i32 s[neg_m1], 0, s[M1]
        s_sub_i32 s[neg_pw], 0, s[stmp]
        s_sub_i32 s[neg_ph], 0, s[stmp+1]
        v_writelane_b32 v[vtmp], s[M0],     0
        v_writelane_b32 v[vtmp], s[M1],     1
        v_writelane_b32 v[vtmp], s[stmp],   2
        v_writelane_b32 v[vtmp], s[stmp+1], 3
        ceil_2_32_div_u16 v[vtmp], v[vtmp], vtmp+1, stmp
        v_readlane_b32 s[div_m0], v[vtmp], 0
        v_readlane_b32 s[div_m1], v[vtmp], 1
        v_readlane_b32 s[div_pw], v[vtmp], 2
        v_readlane_b32 s[div_ph], v[vtmp], 3

        // compute initial indices
        phase = vtmp+1
        local_tile = vtmp+2
        v_and_b32 v[phase], 7, v[tid]
        v_bfe_u32 v[local_tile], v[tid], 3, 5
        v_mul_u32_u24 v[n8_pw], s[tiles_step], v[phase]
        v_add_u32 v[n8_pw], v[local_tile], v[n8_pw]
        v_add_u32 v[n8_pw], s[base_tile], v[n8_pw]
        .GPR_INVALIDATE phase
        .GPR_INVALIDATE local_tile
        v_mul_u32_u24 v[n8_pw], s[tile_step_x], v[n8_pw]
        v_mov_b32 v[n8_ph], 0
        v_mov_b32 v[n8_m0],  0
        v_mov_b32 v[n8_m1],  0
        v_mov_b32 v[n8_g],  0
    .endm

    .macro norm_basen_i24 carry, x, rcp_magic, base, neg_base
        v_mul_hi_u32 \carry, \rcp_magic, \x
        v_cmp_eq_u32 vcc, 1, \base
        v_cndmask_b32 \carry, \carry, \x, vcc
        v_mad_i32_i24 \x, \carry, \neg_base, \x
    .endm
    .macro splitn8_inplace n8, n4hi
        v_mov_b32 \n4hi, \n8
        v_mov_b32 \n4hi, \n8 row_shl:4 bank_mask:0x5
        v_mov_b32 \n8,   \n8 row_shr:4 bank_mask:0xA
    .endm
    .macro splitn8 n4lo, n4hi, n8
        v_mov_b32 \n4hi, \n8
        v_mov_b32 \n4lo, \n8
        v_mov_b32 \n4hi, \n8 row_shl:4 bank_mask:0x5
        v_mov_b32 \n4lo,   \n8 row_shr:4 bank_mask:0xA
    .endm
    .macro unpack_valn8 dst, off, lane, n4lo, n4hi=-1
        q = \lane % 4
        .if \lane < 4
            v_add_u32 \dst, \n4lo, \off quad_perm:[q,q,q,q]
        .else
            v_add_u32 \dst, \n4hi, \off quad_perm:[q,q,q,q]
        .endif
    .endm

    .macro normalize_s_indices g, m1, m0, h, w, pw, ph, vtmp
        v_mul_hi_u32 v[\vtmp], s[div_pw], \pw
        v_mad_i32_i24 \pw, v[\vtmp], s[neg_pw], \pw
        v_mad_i32_i24 \ph, v[\vtmp], s[tile_step_y], \ph
        v_mul_hi_u32 v[\vtmp], s[div_ph], \ph
        v_mad_i32_i24 \ph, v[\vtmp], s[neg_ph], \ph
        v_add_u32 \m0, v[\vtmp], \m0
        norm_basen_i24 v[\vtmp], \m0, s[div_m0], s[M0], s[neg_m0]
        v_add_u32 \m1, v[\vtmp], \m1
        norm_basen_i24 v[\vtmp], \m1, s[div_m1], s[M1], s[neg_m1]
        v_add_u32 \g, v[\vtmp], \g
        .if xform_data
            v_subrev_u32 \w, s[pad_w], \pw
            v_subrev_u32 \h, s[pad_h], \ph
        .else
            v_mov_b32 \w, \pw
            v_mov_b32 \h, \ph
        .endif
    .endm

    .macro compute_h_tile_base_addr addr_name, num, g, m1, m0, h, w, vtmp
        // compute base tile addr
        v_mul_i32_i24 v[\vtmp], s[stride_W], \w
        v_mad_i32_i24 v[\vtmp], s[stride_H], \h, v[\vtmp]
        v_mad_u64_u32 v[\vtmp:\vtmp+1], vcc, s[stride_M0], \m0, v[\vtmp:\vtmp+1]
        v_mad_u64_u32 v[\vtmp:\vtmp+1], vcc, s[stride_M1], \m1, v[\vtmp:\vtmp+1]
        v_mad_u64_u32 v[\vtmp:\vtmp+1], vcc, s[stride_G], \g, v[\vtmp:\vtmp+1]
        v_cmp_lt_u32 vcc, \g, s[G]
        v_cndmask_b32 v[\addr_name], v[invalid], v[\vtmp], vcc

        //compute and mask addr for each h
        i = 0
        .rept \num - 1
            v_add_u32 v[\addr_name + i + 1], s[stride_H], v[\addr_name + i]
            v_cmp_lt_u32 vcc, v[n8_h], s[H]
            v_cndmask_b32 v[\addr_name + i], v[invalid], v[\addr_name + i], vcc
            v_add_u32 v[n8_h], 1, v[n8_h]
            i = i + 1
        .endr
        v_cmp_lt_u32 vcc, v[n8_h], s[H]
        v_cndmask_b32 v[\addr_name + i], v[invalid], v[\addr_name + i], vcc
    .endm

    .macro vmem_loads vdst, addr, num, elem_size
        i = 0
        .rept \num
            .if \elem_size == 2
                buffer_load_ushort v[\vdst + i], v[\addr + i], s[d_desc:d_desc+3], 0, offen
            .else
                buffer_load_dword v[\vdst + i], v[\addr + i], s[d_desc:d_desc+3], 0, offen
            .endif
            i = i + 1
        .endr
    .endm

    .macro vmem_stores vsrc, addr, num, elem_size
        i = 0
        .rept \num
            .if \elem_size == 2
                buffer_store_short v[\vsrc+i], v[\addr + i], s[o_desc:o_desc+3], 0, offen
            .else
                buffer_store_dword v[\vsrc+i], v[\addr + i], s[o_desc:o_desc+3], 0, offen
            .endif
            i = i + 1
        .endr
    .endm


lab_read_wave:

    .if xform_output
        loads_per_phase = in_tile_width
        lds_per_phase   = out_tile_width
        init_x_lds_addr lds_per_phase
        init_x_mem_addr
    .else
        loads_per_phase = in_tile_height
        lds_per_phase   = out_tile_height
        init_s_lds_addr lds_per_phase
        init_s_mem_addr
    .endif

    // enter r_loop
    s_lshr_b32 s[shift_cnt], s[shift_cnt], 1
    s_branch r_loop_entrance


r_loop:
    .macro rx_phase_macro phase
        w_token = (\phase + 0)& 1
        s_token = (\phase + 1) & 1
        u_token = (\phase + 1) & 1
        xform_slot = \phase
        load_slot  = \phase
        ldsv_slot  = \phase
        lds_slot   = \phase % lds_buffers

        xform_base = acc + 8 * xform_slot
        load_base  = acc + 8 * load_slot
        lds_gpr    = acc + 8 * ldsv_slot
        lds_off    = lds_buf + lds_buf_size * lds_slot

        // 1rx. xform
        s_wait (r_pipe_depth-1) * loads_per_phase, 0
        v_cvt_array xform_base, loads_per_phase, TYPE_FP32, src_type
        winograd_xform xformx_o_size, xformx_f_size, xformx_x_size, ddilation_w, L_F_ODD_PAD_W, fdilation_w, stride_w, xform_base, vtmp, xform_mirror


        // 2rx. wait token, send token
        .if \phase == 0
            r_loop_entrance:
        .endif
        s_addk_i32 s[phase_cnt], 1
        wait_token w_token, sync_target
        .if u_token
            update_sync_target sync_target
        .endif
        send_token s_token
        

        // 3rx. lds write
        i = 0
        .rept lds_per_phase
            ds_write_b32 v[lds_addr + i], v[lds_gpr + i], offset:0+lds_off
            i = i + 1
        .endr


        // 4rx. check exit
        s_cmp_ge_u32 s[base_tile], s[total_tiles]
        s_cbranch_scc0 r_skip_epilogue\@
        s_mov_b32 s[tiles_step], 0
        s_lshr_b32 s[shift_cnt], s[shift_cnt], 1
        s_cmp_eq_u32 s[shift_cnt], 0
        s_cbranch_scc1 r_loop_end // exit
        s_waitcnt 0
        s_branch r_phase_end\@ //skip loads and addr computations
        r_skip_epilogue\@:


        // 5rx. buffer load
        i = 0
        .rept loads_per_phase - 1
            v_add_u32 v[n8base_addr + i + 1], s[x_TW_stride], v[n8base_addr + i]
            i = i + 1
        .endr
        vmem_loads load_base, n8base_addr, loads_per_phase, src_elem_size
        s_add_u32 s[base_tile], s[tiles_step], s[base_tile]
        v_add_u32 v[n8base_addr], s[addr_step], v[n8base_addr]

        r_phase_end\@:
    .endm

    .macro r_phase_macro phase
        w_token = \phase & 1
        s_token = (\phase + 1) & 1
        u_token = (\phase + 1) & 1
        xform_slot = \phase
        load_slot  = \phase
        ldsv_slot  = \phase
        lds_slot   = \phase % lds_buffers

        xform_base = acc + 8 * xform_slot
        load_base  = acc + 8 * load_slot
        lds_gpr    = acc + 8 * ldsv_slot
        lds_off    = lds_buf + lds_buf_size * lds_slot


        // 1r. xform
        // zeroing OOB columns and convert to fp32
        unpack_valn8 v[vtmp], v[w_const], \phase, v[n8_w_prev], v[n4hi_w_prev]
        v_cmp_lt_u32 vcc, v[vtmp], s[W]
        s_wait (r_pipe_depth-1) * loads_per_phase, 0
        i = 0
        .rept loads_per_phase
            v_cndmask_b32 v[xform_base + i], 0, v[xform_base + i], vcc
            i = i + 1
        .endr
        v_cvt_array xform_base, loads_per_phase, TYPE_FP32, src_type
        winograd_xform xformy_o_size, xformy_f_size, xformy_x_size, ddilation_h, L_F_ODD_PAD_H, fdilation_h, stride_h, xform_base, vtmp, xform_mirror


        // 2r. wait token, send token
        s_addk_i32 s[phase_cnt], 1
        wait_token w_token, sync_target
        .if u_token
            update_sync_target sync_target
        .endif
        send_token s_token
        

        // 3r. lds write
        i = 0
        .rept lds_per_phase
            ds_write_b32 v[lds_addr + i], v[lds_gpr + i], offset:0+lds_off
            i = i + 1
        .endr


        // 4r. check exit
        s_cmp_ge_u32 s[base_tile], s[total_tiles]
        s_cbranch_scc0 r_skip_epilogue\@
        s_mov_b32 s[tiles_step], 0
        s_lshr_b32 s[shift_cnt], s[shift_cnt], 1
        s_cmp_eq_u32 s[shift_cnt], 0
        s_cbranch_scc1 r_loop_end // exit
        .if \phase == 7
            splitn8 v[n8_w_prev], v[n4hi_w_prev], v[n8_w]
        .endif
        s_waitcnt 0
        s_branch r_phase_end\@ //skip loads and addr computations
        r_skip_epilogue\@:


        // 5r. buffer load
        i = 0
        .rept loads_per_phase
            unpack_valn8 v[vtmp + i], v[w_bconst], \phase, v[n8base_addr + i], v[n4hibase_addr + i]
            i = i + 1
        .endr
        .if use_exec_for_vmem
            s_mov_b64 exec, s[tile_w_mask:tile_w_mask+1]
        .endif
        vmem_loads load_base, vtmp, loads_per_phase, src_elem_size
        .if use_exec_for_vmem
            s_mov_b64 exec, -1
        .endif
        s_add_u32 s[base_tile], s[tiles_step], s[base_tile]

        
        // 6r. compute addr for next 8 phases
        .if \phase == 7
            // preserve index data for the next unroll to check OOB columns
            splitn8 v[n8_w_prev], v[n4hi_w_prev], v[n8_w]

            r_loop_entrance:
            // compute addr
            normalize_s_indices v[n8_g], v[n8_m1], v[n8_m0], v[n8_h], v[n8_w], v[n8_pw], v[n8_ph], vtmp
            compute_h_tile_base_addr n8base_addr, loads_per_phase, v[n8_g], v[n8_m1], v[n8_m0], v[n8_h], v[n8_w], vtmp
            i = 0
            .rept loads_per_phase
                splitn8_inplace v[n8base_addr + i], v[n4hibase_addr + i]
                i = i + 1
            .endr

            // compute next indices
            v_mad_u32_u24 v[n8_pw], v[tile_step_x8], s[tiles_step], v[n8_pw]
        .endif

        r_phase_end\@:
    .endm

    phase = 0
    .rept r_pipe_depth
        .if xform_output
            rx_phase_macro phase
        .else
            r_phase_macro phase
        .endif
        phase = phase + 1
    .endr

    s_branch r_loop
r_loop_end:

s_branch endpgm




lab_write_wave:
    
    .if xform_output
        lds_per_phase    = in_tile_height
        stores_per_phase = out_tile_height
        init_s_lds_addr lds_per_phase
        init_s_mem_addr
    .else
        lds_per_phase    = in_tile_width
        stores_per_phase = out_tile_width
        init_x_lds_addr lds_per_phase
        init_x_mem_addr
    .endif


w_loop:
    .macro ws_phase_macro phase
        w_token = \phase & 1
        s_token = (\phase + 1) & 1
        u_token = (\phase + 1) & 1
        xform_slot = (\phase + 5) % w_pipe_depth
        store_slot = (\phase + 5) % w_pipe_depth
        ldsv_slot  = (\phase + 6) % w_pipe_depth
        lds_slot   = (\phase + 2) % lds_buffers

        xform_base = acc + 8 * xform_slot
        store_base = acc + 8 * store_slot
        lds_gpr    = acc + 8 * ldsv_slot
        lds_off    = lds_buf + lds_buf_size * lds_slot

        // 1ws. wait token, send token
        s_addk_i32 s[phase_cnt], 1
        wait_token w_token, sync_target
        .if u_token
            update_sync_target sync_target
        .endif
        s_wait , 0
        send_token s_token

        // 2ws. lds read
        i = 0
        .rept lds_per_phase
            ds_read_b32 v[lds_gpr + i], v[lds_addr + i] offset:0+lds_off
            i = i + 1
        .endr
        
        
        // 3ws. check prologue finished
        s_lshr_b32 s[shift_cnt], s[shift_cnt], 1
        s_cmp_lg_u32 s[shift_cnt], 0
        s_cbranch_scc1 w_phase_end\@


        // 4ws. xform
        winograd_xform xformy_o_size, xformy_f_size, xformy_x_size, ddilation_h, L_F_ODD_PAD_H, fdilation_h, stride_h, xform_base, vtmp, xform_mirror


        // 5ws. compute addr for next 8 phases
        .if \phase == (rw_pipe_shift % w_pipe_depth)
            normalize_s_indices v[n8_g], v[n8_m1], v[n8_m0], v[n8_h], v[n8_w], v[n8_pw], v[n8_ph], vtmp
            compute_h_tile_base_addr n8base_addr, stores_per_phase, v[n8_g], v[n8_m1], v[n8_m0], v[n8_h], v[n8_w], vtmp
            splitn8_inplace v[n8_w], v[n4hi_w]
            i = 0
            .rept stores_per_phase
                splitn8_inplace v[n8base_addr + i], v[n4hibase_addr + i]
                i = i + 1
            .endr

            // compute next indices
            v_mad_u32_u24 v[n8_pw], v[tile_step_x8], s[tiles_step], v[n8_pw]
        .endif

        
        // 6ws. store
        addr_lane_off = (32*w_pipe_depth - rw_pipe_shift) % w_pipe_depth
        addr_lane = (addr_lane_off + \phase) % w_pipe_depth
        unpack_valn8 v[vtmp], v[w_const], addr_lane, v[n8_w], v[n4hi_w]
        v_cmp_lt_u32 s[stmp:stmp+1], v[vtmp], s[W]
        s_and_b64 s[stmp:stmp+1], s[tile_w_mask:tile_w_mask+1], s[stmp:stmp+1]
        i = 0
        .rept stores_per_phase
            unpack_valn8 v[vtmp + i], v[w_bconst], addr_lane, v[n8base_addr + i], v[n4hibase_addr + i]
            v_cndmask_b32 v[vtmp + i], v[invalid], v[vtmp + i], s[stmp:stmp+1]
            i = i + 1
        .endr
        .if use_exec_for_vmem
            s_mov_b64 exec, s[tile_w_mask:tile_w_mask+1]
        .endif
        v_cvt_array store_base, stores_per_phase, dst_type, TYPE_FP32
        vmem_stores store_base, vtmp, stores_per_phase, dst_elem_size
        .if use_exec_for_vmem
            s_mov_b64 exec, -1
        .endif

        // 7ws. check exit
        s_add_u32 s[base_tile], s[tiles_step], s[base_tile]
        s_cmp_ge_u32 s[base_tile], s[total_tiles]
        s_cbranch_scc1 w_loop_end
        w_phase_end\@:
    .endm

    .macro w_phase_macro phase
        w_token = \phase & 1
        s_token = (\phase + 1) & 1
        u_token = (\phase + 1) & 1
        xform_slot = (\phase + 5) % w_pipe_depth
        store_slot = (\phase + 5) % w_pipe_depth
        ldsv_slot  = (\phase + 6) % w_pipe_depth
        lds_slot   = (\phase + 2) % lds_buffers

        xform_base = acc + 8 * xform_slot
        store_base = acc + 8 * store_slot
        lds_gpr    = acc + 8 * ldsv_slot
        lds_off    = lds_buf + lds_buf_size * lds_slot

        // 1w. wait token, send token
        s_addk_i32 s[phase_cnt], 1
        wait_token w_token, sync_target
        .if u_token
            update_sync_target sync_target
        .endif
        s_wait , 0
        send_token s_token

        // 2w. lds read
        i = 0
        .rept lds_per_phase
            ds_read_b32 v[lds_gpr + i], v[lds_addr + i] offset:0+lds_off
            i = i + 1
        .endr
        
        
        // 3w. check prologue finished
        s_lshr_b32 s[shift_cnt], s[shift_cnt], 1
        s_cmp_lg_u32 s[shift_cnt], 0
        s_cbranch_scc1 w_phase_end\@


        // 4w. xform
        winograd_xform xformx_o_size, xformx_f_size, xformx_x_size, ddilation_w, L_F_ODD_PAD_W, fdilation_w, stride_w, xform_base, vtmp, xform_mirror


        // 5w. store
        v_cmp_lt_u32 vcc, v[x_cur_tile], s[total_tiles]
        v_cndmask_b32 v[vtmp], v[invalid], v[n8base_addr], vcc
        i = 0
        .rept stores_per_phase - 1
            v_add_u32 v[vtmp + i + 1], s[x_TW_stride], v[vtmp + i]
            i = i + 1
        .endr
        v_add_u32 v[x_cur_tile], s[tiles_step], v[x_cur_tile]
        v_add_u32 v[n8base_addr], s[addr_step], v[n8base_addr]
        v_cvt_array store_base, stores_per_phase, dst_type, TYPE_FP32
        vmem_stores store_base, vtmp, stores_per_phase, dst_elem_size
        

        // 6w. check exit
        s_add_u32 s[base_tile], s[tiles_step], s[base_tile]
        s_cmp_ge_u32 s[base_tile], s[total_tiles]
        s_cbranch_scc1 w_loop_end
        w_phase_end\@:
    .endm

    phase = 0
    .rept w_pipe_depth
        .if xform_output
            ws_phase_macro phase
        .else
            w_phase_macro phase
        .endif
        phase = phase + 1
    .endr
    s_branch w_loop
w_loop_end:

endpgm:
s_endpgm
.rept 64
    s_nop 0
.endr

.Lfunc_end0:


.macro KERNEL_DESCRIPTOR_COV3 kernel_name
.rodata
.p2align 6
.amdhsa_kernel \kernel_name
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 0
        .amdhsa_system_sgpr_workgroup_id_z 0
        .amdhsa_system_vgpr_workitem_id 0
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
.end_amdhsa_kernel
.endm

.altmacro

.macro METADATA sc, vc, wg_x, lds_size, kernarg_size, kernel_name
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: \kernel_name
    .symbol: \kernel_name\().kd
    .sgpr_count: \sc
    .vgpr_count: \vc
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: \kernarg_size
    .kernarg_segment_align: 8
    .group_segment_fixed_size: \lds_size
    .private_segment_fixed_size: 0
    .reqd_workgroup_size: [ \wg_x, 1, 1 ]
    .max_flat_workgroup_size: \wg_x
    .wavefront_size: 64
    .args:
    - { .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .name: src_addr, .address_space: global, .is_const: false }
    - { .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .name: dst_addr, .address_space: global, .is_const: false }
    - { .size: 4, .offset:  16, .value_kind: by_value, .value_type: i32, .name: G }
    - { .size: 4, .offset:  20, .value_kind: by_value, .value_type: i32, .name: M1 }
    - { .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32, .name: M0 }
    - { .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32, .name: H }
    - { .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32, .name: W }
    - { .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32, .name: n_groups }
    - { .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32, .name: flags }
    - { .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32, .name: pad_h }
    - { .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32, .name: pad_w }
    - { .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32, .name: tiles_h }
    - { .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32, .name: tiles_w }
    - { .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32, .name: stride_G }
    - { .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32, .name: stride_M1 }
    - { .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32, .name: stride_M0 }
    - { .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32, .name: stride_H }
    - { .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32, .name: stride_W }
    - { .size: 8, .offset:  80, .value_kind: hidden_global_offset_x, .value_type: i64 }
    - { .size: 8, .offset:  88, .value_kind: hidden_global_offset_y, .value_type: i64 }
    - { .size: 8, .offset:  96, .value_kind: hidden_global_offset_z, .value_type: i64 }
    - { .size: 8, .offset: 104, .value_kind: hidden_none,   .value_type: i8 }
    - { .size: 8, .offset: 112, .value_kind: hidden_none,   .value_type: i8 }
    - { .size: 8, .offset: 120, .value_kind: hidden_none,   .value_type: i8 }
...
.end_amdgpu_metadata
.endm // METADATA


.altmacro
.macro METADATA_WRAPPER sc, vc, wg_x, lds_size, kernarg_size, kernel_suf
    .if (xform_filter)
        KERNEL_DESCRIPTOR_COV3 <miopenGcnAsmMPAnydirectWinogradXformFilter\kernel_suf>
        METADATA \sc, \vc, \wg_x, \lds_size, \kernarg_size, <miopenGcnAsmMPAnydirectWinogradXformFilter\kernel_suf>
    .elseif (xform_data)
        KERNEL_DESCRIPTOR_COV3 <miopenGcnAsmMPAnydirectWinogradXformData\kernel_suf>
        METADATA \sc, \vc, \wg_x, \lds_size, \kernarg_size, <miopenGcnAsmMPAnydirectWinogradXformData\kernel_suf>
    .elseif (xform_output)
        KERNEL_DESCRIPTOR_COV3 <miopenGcnAsmMPAnydirectWinogradXformOut\kernel_suf>
        METADATA \sc, \vc, \wg_x, \lds_size, \kernarg_size, <miopenGcnAsmMPAnydirectWinogradXformOut\kernel_suf>
    .endif
.endm

.macro kernel_end x_o_size, y_o_size, x_f_size, y_f_size
    .if (xform_filter)
        .size miopenGcnAsmMPAnydirectWinogradXformFilter_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size, .Lfunc_end0 - miopenGcnAsmMPAnydirectWinogradXformFilter_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
    .elseif (xform_data)
        .size miopenGcnAsmMPAnydirectWinogradXformData_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size, .Lfunc_end0 - miopenGcnAsmMPAnydirectWinogradXformData_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
    .elseif (xform_output)
        .size miopenGcnAsmMPAnydirectWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size, .Lfunc_end0 - miopenGcnAsmMPAnydirectWinogradXformOut_\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
    .endif
    METADATA_WRAPPER %.AUTO_SGPR_COUNT, %.AUTO_VGPR_COUNT, %(512), %.AUTO_LDS_BYTE_SIZE, %KERNEL_ARGUMENTS_SIZE, _\y_o_size\()_\x_o_size\()_\y_f_size\()_\x_f_size
.endm

kernel_end %xformx_o_size, %xformy_o_size, %xformx_f_size, %xformy_f_size

