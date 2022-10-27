/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
; generated by igemm_codegen.py (2461eab400b8d4378cb16e464421a920037d1b0f)
;
.macro .v_u32_div v_q, v_n, v_d, v_tmp4, s_tmp4
    v_cvt_f32_u32     v[\v_tmp4+0],   v[\v_d]
    v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
    v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
    v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
    v_mul_lo_u32      v[\v_tmp4+1],   v[\v_d],      v[\v_tmp4+0]
    v_mul_hi_u32      v[\v_tmp4+2],   v[\v_d],      v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
    v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
    v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
    v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
    v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
    v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
    v_mul_hi_u32      v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_n]
    v_mul_lo_u32      v[\v_tmp4+1],   v[\v_tmp4+0],   v[\v_d]
    v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_n],      v[\v_tmp4+1]
    v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], v[\v_n],      v[\v_tmp4+1]
    v_cmp_ge_u32      s[\s_tmp4+2:\s_tmp4+3], v[\v_tmp4+2],   v[\v_d]
    v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
    s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
    v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
    v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
    v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
    v_cmp_ne_i32      vcc,          0,          v[\v_d]
    v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_u32_div_vs v_q, v_n, s_d, v_tmp4, s_tmp4
    v_cvt_f32_u32     v[\v_tmp4+0],   s[\s_d]
    v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
    v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
    v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
    v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],      v[\v_tmp4+0]
    v_mul_hi_u32      v[\v_tmp4+2],   s[\s_d],      v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
    v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
    v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
    v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
    v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
    v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
    v_mul_hi_u32      v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_n]
    v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],     v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_n],      v[\v_tmp4+1]
    v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], v[\v_n],      v[\v_tmp4+1]
    v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d],    v[\v_tmp4+2]
    v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
    s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
    v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
    v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
    v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
    v_cmp_ne_i32      vcc,          s[\s_d],   0
    v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4
    v_cvt_f32_u32     v[\v_tmp4+0],   s[\s_d]
    v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
    v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
    v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
    v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],      v[\v_tmp4+0]
    v_mul_hi_u32      v[\v_tmp4+2],   s[\s_d],      v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
    v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
    v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
    v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
    v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
    v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
    v_mul_hi_u32      v[\v_tmp4+0],   s[\s_n],   v[\v_tmp4+0]
    v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],     v[\v_tmp4+0]
    v_sub_co_u32      v[\v_tmp4+2],   vcc,        s[\s_n],      v[\v_tmp4+1]
    v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], s[\s_n],      v[\v_tmp4+1]
    v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d],    v[\v_tmp4+2]
    v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
    s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
    v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
    v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
    v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
    v_cmp_ne_i32      vcc,          s[\s_d],   0
    v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_u32_div_rem v_r, v_q, v_n, v_d, v_tmp4, s_tmp4
    .v_u32_div \v_q, \v_n, \v_d, \v_tmp4, \s_tmp4
    v_mul_lo_u32 v[\v_tmp4], v[\v_d], v[\v_q]
    v_sub_u32 v[\v_r], v[\v_n], v[\v_tmp4]
.endm

.macro .v_u32_div_rem_vs v_r, v_q, v_n, s_d, v_tmp4, s_tmp4
    .v_u32_div_vs \v_q, \v_n, \s_d, \v_tmp4, \s_tmp4
    v_mul_lo_u32 v[\v_tmp4], s[\s_d], v[\v_q]
    v_sub_u32 v[\v_r], v[\v_n], v[\v_tmp4]
.endm

.macro .v_u32_div_rem_ss s_r, s_q, s_n, s_d, v_q, v_tmp4, s_tmp4
    .v_u32_div_ss \v_q, \s_n, \s_d, \v_tmp4, \s_tmp4
    v_readfirstlane_b32 s[\s_q], v[\v_q]
    s_mul_i32 s[\s_tmp4], s[\s_d], s[\s_q]
    s_sub_i32 s[\s_r], s[\s_n], s[\s_tmp4]
.endm

.macro .mdiv_u32_ss s_quot s_numer s_magic s_shift s_tmp
    s_mul_hi_u32 s[\s_tmp], s[\s_magic], s[\s_numer]
    s_add_u32 s[\s_tmp], s[\s_tmp], s[\s_numer]
    s_lshr_b32 s[\s_quot], s[\s_tmp], s[\s_shift]
.endm

.macro .mdiv_u32_rem_ss s_rem s_quot s_numer s_magic s_shift s_denom s_tmp
    .mdiv_u32_ss \s_quot,\s_numer,\s_magic,\s_shift,\s_tmp
    s_mul_i32 s[\s_tmp], s[\s_denom], s[\s_quot]
    s_sub_u32 s[\s_rem], s[\s_numer], s[\s_tmp]
.endm

.macro .mdiv_u32_vs v_quot v_numer s_magic s_shift v_tmp
    v_mul_hi_u32 v[\v_tmp], s[\s_magic], v[\v_numer]
    v_add_u32 v[\v_tmp], v[\v_tmp], v[\v_numer]
    v_lshrrev_b32 v[\v_quot], s[\s_shift], v[\v_tmp]
.endm

.macro .mdiv_u32_rem_vs v_rem v_quot v_numer s_magic s_shift s_denom v_tmp
    .mdiv_u32_vs \v_quot,\v_numer,\s_magic,\s_shift,\v_tmp
    v_mul_lo_u32 v[\v_tmp], s[\s_denom], v[\v_quot]
    v_sub_u32 v[\v_rem], v[\v_numer], v[\v_tmp]
.endm

.macro .v_clear_acc_c a, num
    _a = \a
    .rept \num
        v_accvgpr_write_b32 a[_a], 0
        _a = _a + 1
    .endr
.endm

.macro .v_clear_nc vid, num
    _v = \vid
    .rept \num
        v_mov_b32 v[_v], 0
        _v = _v + 1
    .endr
.endm

;----------------------------------------------------------
; starting of kernel igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs
; tensor_layout              : 'nhwc'
; gemm_m_per_block           : 32
; gemm_n_per_block           : 64
; gemm_k_per_block           : 16
; wave_tile_m                : 16
; wave_step_m                : 1
; wave_repeat_m              : 1
; wave_tile_n                : 16
; wave_step_n                : 1
; wave_repeat_n              : 2
; wave_tile_k                : 4
; tensor_a_thread_lengths    : [1, 1, 1, 2]
; tensor_a_cluster_lengths   : [1, 16, 1, 16]
; tensor_b_thread_lengths    : [1, 1, 1, 4]
; tensor_b_cluster_lengths   : [1, 16, 1, 16]
; direction                  : 'wrw'
; precision                  : 'fp32'
; nxb                        : 0
; nxe                        : 0
; gemm_k_global_split        : 1
; 
; block_size                 : 256
; lds_total                  : 8192
; lds_buffer_num             : 1
; 
.set k_p_in, 0
.set k_p_wei, 8
.set k_p_out, 16
.set k_hi, 24
.set k_wi, 28
.set k_n, 32
.set k_k, 36
.set k_c, 40
.set k_ho, 44
.set k_wo, 48
.set k_stride_h, 52
.set k_stride_w, 56
.set k_dilation_h, 60
.set k_dilation_w, 64
.set k_pad_h, 68
.set k_pad_w, 72
.set k_y, 76
.set k_x, 80
.set k_gemm_k_global_split, 84
.set k_group, 88
.set k_pack_0, 92
.set k_end, 96

.set s_ka, 0
.set s_bx, 2
.set s_by, 3
.set s_bz, 4
.set s_p_in, 8
.set s_p_wei, 12
.set s_p_out, 16
.set s_hi, 20
.set s_wi, 21
.set s_n, 22
.set s_k, 23
.set s_c, 24
.set s_ho, 25
.set s_wo, 26
.set s_stride_h, 27
.set s_stride_w, 28
.set s_dilation_h, 29
.set s_dilation_w, 30
.set s_pad_h, 31
.set s_pad_w, 32
.set s_y, 33
.set s_x, 34
.set s_gemmk_split, 35
.set s_group, 36
.set s_gemmk_per_wg, 37
.set s_ho_x_stride_h, 38
.set s_wo_x_stride_w, 39
.set s_in_stride_wi, 40
.set s_in_stride_hi, 41
.set s_in_stride_n, 42
.set s_out_stride_wo, 43
.set s_out_stride_ho, 44
.set s_out_stride_n, 45
.set s_wei_stride_k, 46
.set s_ec_padded, 47
.set s_in_stride_n_n, 48
.set s_out_stride_n_n, 49
.set s_move_slice_n, 50
.set s_move_slice_n_dsho, 51
.set s_move_slice_n_dswo, 52
.set s_dim_b, 53
.set s_block_gtc_ie, 54
.set s_block_gtc_ik, 55
.set s_block_gtc_iec, 56
.set s_block_gtc_in, 57
.set s_block_gtc_ig, 58
.set s_knum, 1
.set s_gemm_k_num_n1, 0
.set s_gemm_k_num_dsho, 59
.set s_gemm_k_num_dswo, 60
.set s_kitr, 3
.set s_in_offset, 61
.set s_out_offset, 61
.set s_sub_n, 61
.set s_in_stride_move_n, 62
.set s_out_stride_move_n, 63
.set s_k_padded, 64
.set s_c_padded, 65
.set s_out_move_step, 66
.set s_in_move_step, 67
.set s_tmp, 68
.set s_end, 74

.set v_c, 0  ; coalescing:8, needed:0, resuable:21
.set v_a, 0
.set v_b, 2
.set v_gld_a, 6
.set v_gld_b, 8
.set v_sst_a_os, 12
.set v_sst_b_os, 13
.set v_sld_a_os, 14
.set v_sld_b_os, 15
.set v_in_os, 16
.set v_in_os_base, 17
.set v_gtc_in, 18
.set v_out_os, 19
.set v_out_os_base, 20
.set v_co_sst, 21
.set v_co_sld, 22
.set v_wei_os, 23
.set v_wei_c_flag, 24
.set v_gtc_iec, 25
.set v_wei_ie, 26
.set v_gtc_ic, 27
.set v_gtc_ie, 28
.set v_gtc_ik, 29
.set v_gtc_inb_a, 30
.set v_gemm_in, 31
.set v_gemm_im, 32
.set v_wei_ic, 33
.set v_wei_iec, 34
.set v_co_sub_m_index, 35
.set v_co_sub_n_index, 36
.set v_cur_k, 37
.set v_tmp, 38
.set v_end, 56

.set a_c, 48
.set a_end, 56

.text
.globl igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs
.p2align 8
.type igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs,@function
igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],       s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],      s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],      s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx16 s[s_hi+0:s_hi+15],        s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dwordx2  s[s_group+0:s_group+1],      s[s_ka+0:s_ka+1],    0+k_group

    ; input, thread(1,nb,1,c): 1x1x1x4, cluster(1,nb,1,ec): 1x16x1x16
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_iec], 15, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_iec], 2, v[v_gtc_iec]
    v_lshrrev_b32 v[v_tmp], 4, v[v_tmp]
    v_and_b32 v[v_gtc_inb_a], 15, v[v_tmp]

    ; output, thread(1,nb,1,k): 1x1x1x2, cluster(1,nb,1,k) 1x16x1x16
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_ik], 15, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_ik], 1, v[v_gtc_ik]

    s_mov_b32 s[s_p_in+3], 0x27000
    s_mov_b32 s[s_p_wei+2], 0xffffffff
    s_mov_b32 s[s_p_wei+3], 0x27000
    s_mov_b32 s[s_p_out+3], 0x27000
    s_waitcnt lgkmcnt(0)

    ; calculate index
    ; s_lshr_b32 s[s_sub_n], s[s_n], s[s_gemmk_split]
    s_mul_i32 s[s_in_stride_wi], s[s_c], s[s_group]
    s_mul_i32 s[s_tmp+2], s[s_wi], s[s_in_stride_wi]
    s_mul_i32 s[s_in_stride_n], s[s_hi], s[s_tmp+2]
    s_mov_b32 s[s_wei_stride_k], s[s_c]
    s_mul_i32 s[s_out_stride_wo], s[s_k], s[s_group]
    s_mul_i32 s[s_tmp+2], s[s_wo], s[s_out_stride_wo]
    s_mul_i32 s[s_out_stride_n], s[s_ho], s[s_tmp+2]
    s_mul_i32 s[s_dim_b], s[s_ho], s[s_wo]
    s_mul_i32  s[s_tmp], s[s_n], s[s_in_stride_n]
    s_mul_i32  s[s_tmp+1], s[s_n], s[s_out_stride_n]
    s_lshl_b32 s[s_tmp+4], s[s_tmp], 2
    s_lshl_b32 s[s_tmp+5], s[s_tmp+1], 2
    s_mul_i32 s[s_tmp], s[s_by], s[s_tmp+4]
    s_mul_hi_u32 s[s_tmp+1], s[s_by], s[s_tmp+4]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]
    s_mul_i32 s[s_tmp], s[s_by], s[s_tmp+5]
    s_mul_hi_u32 s[s_tmp+1], s[s_by], s[s_tmp+5]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]
    ; compute start point
    s_mul_i32 s[s_sub_n], s[s_bz], s[s_gemmk_per_wg]
    v_add_u32 v[v_gtc_inb_a], v[v_gtc_inb_a], s[s_sub_n]
    ; pad gemm_m if needed
    s_add_u32 s[s_tmp], 31, s[s_k]
    s_lshr_b32 s[s_tmp], s[s_tmp], 5
    s_lshl_b32 s[s_k_padded], s[s_tmp], 5
    ; pad c for 1x1 cases
    s_add_u32 s[s_tmp], 63, s[s_c]
    s_lshr_b32 s[s_tmp], s[s_tmp], 6
    s_lshl_b32 s[s_c_padded], s[s_tmp], 6

    ; add block i_n
    ; gemm_m_per_block:32, gemm_n_per_block:64
    s_lshr_b32 s[0], s[s_c_padded], 6
    s_lshr_b32 s[s_tmp], s[s_k_padded], 5
    s_mul_i32 s[1], s[0], s[s_tmp]
    ;s_lshl_b32 s[s_tmp+3], 1, s[s_gemmk_split]
    ;s_sub_u32 s[s_tmp+3], s[s_tmp+3], 1
    ;s_and_b32 s[s_block_gtc_in], s[s_bx], s[s_tmp+3]
    ;s_mul_i32 s[s_block_gtc_in], s[s_block_gtc_in], s[s_sub_n]
    ;s_lshr_b32 s[s_bx], s[s_bx], s[s_gemmk_split]
    .v_u32_div_rem_ss s_tmp+4, s_block_gtc_ig, s_bx, 1, v_tmp+5, v_tmp, s_tmp
    s_mov_b32 s[s_bx], s[s_tmp+4]
    .v_u32_div_rem_ss s_tmp+4, s_tmp+5, s_bx, 0, v_tmp+5, v_tmp, s_tmp
    ; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im
    s_lshl_b32 s[s_block_gtc_ik], s[s_tmp+5], 5
    s_lshl_b32 s[s_block_gtc_iec], s[s_tmp+4], 6

    ; config for output and input range
    s_mul_i32 s[s_p_out+2], s[s_n], s[s_out_stride_n]
    s_lshl_b32 s[s_p_out+2], s[s_p_out+2], 2
    s_mul_i32 s[s_p_in+2], s[s_n], s[s_in_stride_n]
    s_lshl_b32 s[s_p_in+2], s[s_p_in+2], 2
    v_add_u32 v[v_gtc_ic], s[s_block_gtc_iec], v[v_gtc_iec]
    ; calculate input offset
    s_lshl_b32 s[s_block_gtc_ig], s[s_block_gtc_ig], 2
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_c]
    s_sub_u32 s[s_p_in+2], s[s_p_in+2], s[s_tmp]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]

    v_mul_lo_u32 v[v_tmp], v[v_gtc_inb_a], s[s_in_stride_wi]
    v_add_lshl_u32 v[v_in_os], v[v_tmp], v[v_gtc_ic], 2


    ; load input
    buffer_load_dwordx4 v[v_gld_b+0:v_gld_b+0+3], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:0

    ; calculate out offset
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_k]
    s_sub_u32 s[s_p_out+2], s[s_p_out+2], s[s_tmp]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]

    v_add_u32 v[v_cur_k], s[s_block_gtc_ik], v[v_gtc_ik]
    ;s_mul_i32 s[s_tmp], s[s_out_stride_n], s[s_block_gtc_in]
    ;v_add_lshl_u32 v[v_tmp+1], v[v_cur_k], s[s_tmp], 2
    v_mul_lo_u32 v[v_tmp], v[v_gtc_inb_a], s[s_out_stride_wo]
    v_add_lshl_u32 v[v_out_os], v[v_tmp], v[v_cur_k], 2


    ; load output
    buffer_load_dwordx2 v[v_gld_a+0:v_gld_a+0+1], v[v_out_os], s[s_p_out:s_p_out+3], 0 offen offset:0

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get source matrix gemm index, k_pack:1, v_pack:1, k_pack_per_thread:1
    v_and_b32 v[v_gemm_in], 15, v[v_tmp+5]           ; block_n index 
    v_and_b32 v[v_gemm_im], 15, v[v_tmp+5]           ; block_m index 
    v_lshrrev_b32 v[v_tmp+5], 4, v[v_tmp+5]
    v_and_b32 v[v_tmp + 0], 3, v[v_tmp+5]          ; block_k_per_wave index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 0], 6, v[v_gemm_in]
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp + 0], 5, v[v_gemm_im]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_and_b32 v[v_tmp + 2], 1, v[v_tmp+5]  ; waves_per_n index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 2], 4, v[v_gemm_in]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_and_b32 v[v_tmp + 3], 1, v[v_tmp+5]  ; waves_per_m index
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp + 3], 4, v[v_gemm_im]

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get dst matrix gemm index
    v_and_b32 v[v_tmp+0], 15, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 4, v[v_tmp+5]
    v_and_b32 v[v_tmp+1], 3, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_mov_b32 v[v_co_sst], v[v_tmp+0]
    v_lshlrev_b32 v[v_co_sld], 2, v[v_tmp+1]
    v_and_b32 v[v_tmp+0], 1, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_and_b32 v[v_tmp+1], 1, v[v_tmp+5]
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+0], 4, v[v_co_sst]
    v_lshl_or_b32 v[v_co_sld], v[v_tmp+1], 4, v[v_co_sld]

    ; LDS store, in: 1,nb,1,ec: 1x1x1x4, 1x16x1x16
    v_sub_i32 v[v_gtc_inb_a], v[v_gtc_inb_a], s[s_sub_n]
    v_lshl_or_b32 v[v_tmp], v[v_gtc_inb_a], 6, v[v_gtc_iec]
    v_lshlrev_b32 v[v_sst_b_os], 2, v[v_tmp]
    v_add_u32 v[v_sst_b_os], 2048, v[v_sst_b_os]

    ; LDS store, out: 1,nb,1,k: 1x1x1x2, 1x16x1x16
    v_lshl_or_b32 v[v_tmp], v[v_gtc_inb_a], 5, v[v_gtc_ik]
    v_lshlrev_b32 v[v_sst_a_os], 2, v[v_tmp]

    ; LDS load
    v_lshlrev_b32 v[v_sld_b_os], 2, v[v_gemm_in]
    v_lshlrev_b32 v[v_sld_a_os], 2, v[v_gemm_im]
    v_add_u32 v[v_sld_b_os], 2048, v[v_sld_b_os]

    v_mov_b32 v[v_gemm_in], v[v_co_sst]
    v_mov_b32 v[v_gemm_im], v[v_co_sld]
    ; init_co_lds_offset for xdlops
    v_lshrrev_b32 v[v_tmp], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp],  3 v[v_tmp]   ; thread id of lanegroup_m_per_cluster
    v_lshlrev_b32 v[v_co_sst], 2, v[v_tmp]
    v_lshrrev_b32 v[v_tmp+2], 4, v[v_gemm_im]  ; thread id of waves_per_m
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+2], 4, v[v_co_sst]
    v_lshrrev_b32 v[v_tmp], 2, v[v_co_sst]
    v_lshlrev_b32 v[v_tmp+1], 2, v[v_gemm_in]   ; implicit transpose with m granularity:4 while store
    v_lshl_or_b32 v[v_co_sst], v[v_tmp], 8, v[v_tmp+1]
    v_lshlrev_b32 v[v_co_sst], 2, v[v_co_sst]
    v_lshlrev_b32 v[v_co_sld], 4, v[0]
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:32x64 sub_m_index:[0, 4, 8, 12]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:1, l_mw:1, l_mb:1, l_mt:4 | n_mc:4, n_ml:1, n_mv:2
    ; nd_stride:[4, 4, 1, 1, 1, 1, 2, 1]
    v_lshrrev_b32 v[v_co_sub_m_index], 6, v[0]   ; get tid along m
    v_and_b32 v[v_tmp+0], 3, v[v_co_sub_m_index]                   ; => x_mc
    v_lshlrev_b32 v[v_co_sub_m_index], 2, v[v_tmp+0]      ; => accumulate x_mc
    ; init_co_sub_n_index xdlops
    v_and_b32 v[v_co_sub_n_index], 63, v[0]

    ; weight offset
    s_mul_i32 s[s_tmp+2], s[s_k], s[s_wei_stride_k]
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+2]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+2]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp]
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], s[s_tmp+1]

    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_ik], 2
    s_mul_i32 s[s_tmp], s[s_wei_stride_k], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_wei_stride_k], s[s_tmp+3]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp]
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], s[s_tmp+1]

    ; compute v_co_sub_n_index along ec : 64
    v_and_b32 v[v_wei_iec], 63, v[v_co_sub_n_index]     ; => EC

    ; compute wei_ic and set wei_flag
    v_add_u32 v[v_wei_ic], v[v_wei_iec], s[s_block_gtc_iec]
    v_cmp_gt_u32 vcc, s[s_c], v[v_wei_ic]
    v_cndmask_b32 v[v_wei_c_flag],  0, 1, vcc
    ; compute wei offset
    v_mov_b32 v[v_wei_os], v[v_wei_ic]
    ; add i_k
    v_mul_lo_u32 v[v_tmp], s[s_wei_stride_k], v[v_co_sub_m_index]
    v_add_u32 v[v_wei_os], v[v_wei_os], v[v_tmp]
    v_lshlrev_b32 v[v_wei_os], 2, v[v_wei_os]
    ; move slice step for output tensor
    s_mul_i32 s[s_tmp], s[s_k], s[s_group]
    s_mul_i32 s[s_tmp+1], s[s_c], s[s_group]
    s_lshl_b32 s[s_out_move_step], s[s_tmp], 6
    s_lshl_b32 s[s_in_move_step], s[s_tmp+1], 6
    ; move slice stride
    s_lshl_b32 s[s_wei_stride_k], s[s_wei_stride_k], 2
    s_add_i32 s[s_knum], s[s_gemmk_per_wg], 15
    s_lshr_b32 s[s_knum], s[s_knum], 4
    s_lshl_b32 s[s_knum], s[s_knum], 4

    ; start MFMA loop, 16x16 wave tile with 1x2 repeat, 1x1 step, k_pack:1
    s_waitcnt vmcnt(1)
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+0:v_gld_b+0+3] 

    s_waitcnt vmcnt(0)
    ds_write_b64 v[v_sst_a_os], v[v_gld_a+0:v_gld_a+0+1] 

    .v_clear_nc a_c, 8
    ; make sure acc WAR harzard, at least 1 nop for src_c
    s_sub_i32 s[s_kitr], s[s_knum], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs_mfma_end

    v_add_u32 v[v_in_os], v[v_in_os], s[s_in_move_step]
    v_add_u32 v[v_out_os], v[v_out_os], s[s_out_move_step]
    s_waitcnt lgkmcnt(0)
    s_barrier
L_igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs_mfma_body:
    ; do fma accumulate with unroll 16
    ds_read_b32 v[v_a], v[v_sld_a_os] 
    ds_read_b32 v[v_b], v[v_sld_b_os] 
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:128
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    buffer_load_dwordx4 v[v_gld_b+0:v_gld_b+0+3], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:0
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:512 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_b+2], v[v_sld_b_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_b+3], v[v_sld_b_os] offset:1152 ; load i_k:1 into local buffer 1, repeat 1
    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4
    buffer_load_dwordx2 v[v_gld_a+0:v_gld_a+0+1], v[v_out_os], s[s_p_out:s_p_out+3], 0 offen offset:0
    ds_read_b32 v[v_a], v[v_sld_a_os] offset:1024 ; load i_k:2 into local buffer 0, repeat 0
    ds_read_b32 v[v_b], v[v_sld_b_os] offset:2048 ; load i_k:2 into local buffer 0, repeat 0
    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+1], v[v_b+2], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_add_u32 v[v_in_os], v[v_in_os], s[s_in_move_step]
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:2176 ; load i_k:2 into local buffer 0, repeat 1
    ds_read_b32 v[v_b+2], v[v_sld_b_os] offset:3072 ; load i_k:3 into local buffer 1, repeat 0
    s_waitcnt lgkmcnt(4)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b+3], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4
    v_add_u32 v[v_out_os], v[v_out_os], s[s_out_move_step]
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:1536 ; load i_k:3 into local buffer 1, repeat 0
    ds_read_b32 v[v_b+3], v[v_sld_b_os] offset:3200 ; load i_k:3 into local buffer 1, repeat 1
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_waitcnt vmcnt(1)
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+0:v_gld_b+0+3]
    s_waitcnt lgkmcnt(4)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    s_waitcnt vmcnt(0)
    ds_write_b64 v[v_sst_a_os], v[v_gld_a+0:v_gld_a+0+1]
    s_barrier
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+1], v[v_b+2], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    s_sub_i32 s[s_kitr], s[s_kitr], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs_mfma_finishing
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b+3], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_branch L_igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs_mfma_body
L_igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs_mfma_finishing:
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b+3], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4

L_igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs_mfma_end:
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b32 v[v_a], v[v_sld_a_os] 
    ds_read_b32 v[v_b], v[v_sld_b_os] 
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:128
    ; k iteration : 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:512 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_b+2], v[v_sld_b_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_b+3], v[v_sld_b_os] offset:1152 ; load i_k:1 into local buffer 1, repeat 1

    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4
    ds_read_b32 v[v_a], v[v_sld_a_os] offset:1024 ; load i_k:2 into local buffer 0, repeat 0
    ds_read_b32 v[v_b], v[v_sld_b_os] offset:2048 ; load i_k:2 into local buffer 0, repeat 0

    ; k iteration : 4
    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+1], v[v_b+2], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:2176 ; load i_k:2 into local buffer 0, repeat 1
    ds_read_b32 v[v_b+2], v[v_sld_b_os] offset:3072 ; load i_k:3 into local buffer 1, repeat 0

    s_waitcnt lgkmcnt(4)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b+3], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:1536 ; load i_k:3 into local buffer 1, repeat 0
    ds_read_b32 v[v_b+3], v[v_sld_b_os] offset:3200 ; load i_k:3 into local buffer 1, repeat 1

    ; k iteration : 8
    s_waitcnt lgkmcnt(4)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4

    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4

    ; k iteration : 12
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+1], v[v_b+2], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4

    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b+3], v[a_c+4:a_c+7]     ; repeat:0x1, step:0x0, num_a_c:4

    s_nop 9
    ; coalescing store, mapping:mt_m:32, mt_n:64, wt_m:16, wt_n:16, ws:4, r_m:1, r_n:2, s_m:1, s_n:1 | 16x16x4, lanegroup_m_tcbw:4x4x1x1, lanegroup_n_tcbw:1x16x1x1
    ; coalescing_groups:1, num_dword_per_group:8
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:32x64 sub_m_index:[0, 4, 8, 12]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:1, l_mw:1, l_mb:1, l_mt:4 | n_mc:4, n_ml:1, n_mv:2
    ; nd_stride:[4, 1, 1, 1, 1, 2, 1]
    ; start group 0, i_g_mr:0, i_g_ms:0, i_g_mw:0, i_g_mb:0, i_g_mt:0, m index start from 0
    s_barrier
    ds_write_b128 v[v_co_sst], v[a_c:a_c+3]    ; idword:0(0,0),  0x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+4:a_c+4+3] offset:512   ; idword:32(0,32),  0x32 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:1, i_ns:0, i_nw:0
    s_mov_b32 s[s_tmp], 0   ; i_m:0(i_m0:0,i_m1:0)
    v_add_u32 v[v_cur_k], s[s_block_gtc_ik], v[v_co_sub_m_index]
    v_mov_b32 v[v_tmp], v[v_cur_k]
    s_waitcnt lgkmcnt(0)
    s_barrier
    ;   load from lds, i_ssgroup:0, num_sld_per_ssgroup:2
    ds_read_b128 v[v_c:v_c+3], v[v_co_sld] 
    ds_read_b128 v[v_c+4:v_c+4+3], v[v_co_sld] offset:4096
    v_cmpx_eq_u32 vcc, 1, v[v_wei_c_flag]
    ;   store to global, m index start from 0, m0:0, m1:0
    s_waitcnt lgkmcnt(1)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b32 s[s_tmp], s[s_wei_stride_k]   ; i_m:1(i_m0:0,i_m1:1)
    v_add_u32 v[v_tmp], 1, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 2, s[s_wei_stride_k]   ; i_m:2(i_m0:0,i_m1:2)
    v_add_u32 v[v_tmp], 2, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+2], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 3, s[s_wei_stride_k]   ; i_m:3(i_m0:0,i_m1:3)
    v_add_u32 v[v_tmp], 3, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 16, s[s_wei_stride_k]   ; i_m:16(i_m0:0,i_m1:16)
    v_add_u32 v[v_tmp], 16, v[v_cur_k]
    s_waitcnt lgkmcnt(0)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+4], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 17, s[s_wei_stride_k]   ; i_m:17(i_m0:0,i_m1:17)
    v_add_u32 v[v_tmp], 17, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+5], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 18, s[s_wei_stride_k]   ; i_m:18(i_m0:0,i_m1:18)
    v_add_u32 v[v_tmp], 18, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+6], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 19, s[s_wei_stride_k]   ; i_m:19(i_m0:0,i_m1:19)
    v_add_u32 v[v_tmp], 19, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+7], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b64 exec, -1

L_igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs_out:
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 56
    .amdhsa_next_free_sgpr 74
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_tg_split 0
    .amdhsa_accum_offset 48
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs
    .symbol: igemm_wrw_gtcx2_nhwc_fp32_bx0_ex0_bt32x64x16_wt16x16x4_ws1x1_wr1x2_ta1x1x1x2_1x16x1x16_tb1x1x1x4_1x16x1x16_gkgs.kd
    .sgpr_count: 80
    .vgpr_count: 56
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n_         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y_         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: gemm_k_global_split, .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
