/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2021 Advanced Micro Devices, Inc.
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
; generated by igemm_codegen.py (a3229bb2a2624f0dc8e4fbd118817745667e66ac)
;
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
; starting of kernel igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8
; tensor_layout              : nchw
; gemm_m_per_block           : 16
; gemm_n_per_block           : 32
; gemm_k_per_block           : 16
; wave_tile_m                : 8
; wave_step_m                : 2
; wave_repeat_m              : 1
; wave_tile_n                : 32
; wave_step_n                : 1
; wave_repeat_n              : 1
; wave_tile_k                : 4
; tensor_a_thread_lengths    : [1, 2, 1, 2]
; tensor_a_cluster_lengths   : [1, 8, 1, 8]
; tensor_b_thread_lengths    : [1, 2, 4, 1]
; tensor_b_cluster_lengths   : [1, 8, 1, 8]
; direction                  : bwd
; precision                  : fp16
; nxb                        : 1
; nxe                        : 1
; 
; block_size                 : 64
; lds_total                  : 4096
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
.set k_dtile_iy, 84
.set k_dtile_ix, 88
.set k_dtile_dy, 92
.set k_dtile_dx, 96
.set k_dtile_y, 100
.set k_dtile_x, 104
.set k_dtile_h, 108
.set k_dtile_w, 112
.set k_dslice_y, 116
.set k_dslice_x, 120
.set k_dslice_h, 124
.set k_dslice_w, 128
.set k_dslice_h_left, 132
.set k_dslice_w_left, 136
.set k_group, 140
.set k_magic_0, 144
.set k_magic_1, 148
.set k_magic_2, 152
.set k_magic_3, 156
.set k_magic_4, 160
.set k_magic_5, 164
.set k_magic_6, 168
.set k_shift_pack_0, 172
.set k_shift_pack_1, 176
.set k__pack_0, 180
.set k_end, 184

.set s_ka, 0
.set s_bx, 2
.set s_p_in, 4
.set s_p_wei, 8
.set s_p_out, 12
.set s_hi, 16
.set s_wi, 17
.set s_n, 18
.set s_k, 19
.set s_c, 20
.set s_ho, 21
.set s_wo, 22
.set s_stride_h, 23
.set s_stride_w, 24
.set s_dilation_h, 25
.set s_dilation_w, 26
.set s_pad_h, 27
.set s_pad_w, 28
.set s_y, 29
.set s_x, 30
.set s_dtile_iy, 31
.set s_dtile_ix, 32
.set s_dtile_dy, 33
.set s_dtile_dx, 34
.set s_dtile_y, 35
.set s_dtile_x, 36
.set s_dtile_h, 37
.set s_dtile_w, 38
.set s_dslice_y, 39
.set s_dslice_x, 40
.set s_dslice_h, 41
.set s_dslice_w, 42
.set s_dslice_h_left, 43
.set s_dslice_w_left, 44
.set s_group, 45
.set s_out_stride_k, 37
.set s_out_stride_k0, 46
.set s_out_stride_n, 38
.set s_out_stride_n0, 47
.set s_in_stride_c, 48
.set s_in_stride_n, 45
.set s_wei_stride_c, 49
.set s_wei_stride_c0, 50
.set s_wei_stride_k, 51
.set s_wei_stride_k0, 52
.set s_stride_dslice_hw, 41
.set s_stride_dslice_yx, 29
.set s_dslice_dim_b, 41
.set s_out_stride_k_k1, 23
.set s_wei_stride_k_k1, 25
.set s_move_slice_k_k1, 27
.set s_block_gtc_ig, 53
.set s_block_gtc_ic, 54
.set s_block_gtc_in0, 55
.set s_block_gtc_in1b, 56
.set s_knum, 1
.set s_gemm_k_num_k1, 2
.set s_out_stride_k_save, 39
.set s_wei_stride_k_save, 40
.set s_dtile_dy_neg, 33
.set s_dtile_dx_neg, 34
.set s_kitr, 3
.set s_out_offset, 57
.set s_wei_offset, 62
.set s_magic_0, 54
.set s_magic_1, 55
.set s_magic_2, 14
.set s_magic_3, 15
.set s_magic_4, 3
.set s_magic_5, 10
.set s_magic_6, 11
.set s_shift_pack_0, 6
.set s_shift_pack_1, 7
.set s_tmp, 62
.set s_end, 68

.set v_c, 0  ; coalescing:8, needed:2, resuable:32
.set v_a, 2
.set v_b, 10
.set v_gld_a, 14
.set v_gld_b, 18
.set v_sst_a_os, 26
.set v_sst_b_os, 27
.set v_sld_a_os, 28
.set v_sld_b_os, 29
.set v_out_iho, 30
.set v_out_iwo, 31
.set v_out_dslice_ih, 32
.set v_out_dslice_iw, 33
.set v_out_os, 34
.set v_out_os_base, 35
.set v_wei_iy, 36
.set v_wei_ix, 37
.set v_dtile_iy, 38
.set v_dtile_ix, 39
.set v_wei_os, 40
.set v_wei_os_base, 41
.set v_out_flag, 42
.set v_co_sst, 43
.set v_co_sld, 44
.set v_in_flag, 45
.set v_in_os, 46
.set v_gtc_ik1, 47
.set v_move_slice_k_ik1, 47
.set v_gtc_ic0, 48
.set v_gtc_ic1, 49
.set v_gtc_ik0, 50
.set v_gtc_ik1e, 51
.set v_gtc_in0, 52
.set v_gtc_in1b, 53
.set v_gtc_in1, 54
.set v_gemm_in, 55
.set v_gemm_im, 56
.set v_in_in0, 57
.set v_in_in1b, 58
.set v_in_in1, 59
.set v_in_ihi, 60
.set v_in_iwi, 61
.set v_in_dslice_ih, 62
.set v_in_dslice_iw, 63
.set v_co_sub_m_index, 64
.set v_co_sub_n_index, 65
.set v_tmp, 66
.set v_end, 72

.set a_c, 0
.set a_end, 8

.text
.globl igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8
.p2align 8
.type igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8,@function
igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8:
    ; unmerge_sub_k:16, unmerge_sub_k1:16, unmerge_sub_n:32, unmerge_sub_n1:8
    ; gemm_m_unmerge_cluster:0, gemm_n_unmerge_cluster:0, gemm_k_unmerge_cluster:0
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],       s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],      s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],      s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx16 s[s_hi+0:s_hi+15],        s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dwordx8  s[s_dtile_ix+0:s_dtile_ix+7],   s[s_ka+0:s_ka+1],    0+k_dtile_ix
    s_load_dwordx4  s[s_dslice_x+0:s_dslice_x+3],   s[s_ka+0:s_ka+1],    0+k_dslice_x
    s_load_dwordx2  s[s_dslice_w_left+0:s_dslice_w_left+1],   s[s_ka+0:s_ka+1],    0+k_dslice_w_left
    s_load_dwordx2 s[s_magic_0+0:s_magic_0+1],   s[s_ka+0:s_ka+1],    0+k_magic_0
    s_load_dwordx2 s[s_magic_2+0:s_magic_2+1],   s[s_ka+0:s_ka+1],    0+k_magic_2
    s_load_dword   s[s_magic_4],   s[s_ka+0:s_ka+1],    0+k_magic_4
    s_load_dwordx2 s[s_magic_5+0:s_magic_5+1],   s[s_ka+0:s_ka+1],    0+k_magic_5
    s_load_dwordx2 s[s_shift_pack_0+0:s_shift_pack_0+1],   s[s_ka+0:s_ka+1],    0+k_shift_pack_0

    ; output, thread(k0,k1e,n0,n1b): 1x2x4x1, cluster(k0,k1e,n0,n1b): 1x8x1x8
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_in1b], 7, v[v_tmp]
    v_lshrrev_b32 v[v_tmp], 3, v[v_tmp]
    v_mov_b32 v[v_gtc_in0], 0
    v_and_b32 v[v_gtc_ik1e], 7, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_ik1e], 1, v[v_gtc_ik1e]
    v_lshrrev_b32 v[v_tmp], 3, v[v_tmp]
    v_mov_b32 v[v_gtc_ik0], 0

    ; wei, thread(k0,k1e,c0,c1): 1x2x1x2, cluster(k0,k1e,c0,c1): 1x8x1x8
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_ic1], 7, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_ic1], 1, v[v_gtc_ic1]
    v_lshrrev_b32 v[v_tmp], 3, v[v_tmp]
    v_mov_b32 v[v_gtc_ic0], 0

    s_waitcnt lgkmcnt(0)

    ; calculate index ...

    ; initialize the strides
    s_mul_i32 s[s_out_stride_k],      s[s_ho],       s[s_wo]
    s_mul_i32 s[s_tmp],      s[s_k],       s[s_out_stride_k]
    s_mul_i32 s[s_out_stride_n],      s[s_group],        s[s_tmp]
    s_mul_i32 s[s_in_stride_c],       s[s_hi],       s[s_wi]
    s_mul_i32 s[s_tmp],       s[s_c],        s[s_in_stride_c]
    s_mul_i32 s[s_in_stride_n],       s[s_group],        s[s_tmp]
    s_mul_i32 s[s_wei_stride_c],      s[s_y],        s[s_x]
    s_mul_i32 s[s_wei_stride_k],      s[s_c],        s[s_wei_stride_c]
    s_mul_i32 s[s_stride_dslice_hw],  s[s_dslice_h], s[s_dslice_w]
    s_mov_b32 s[s_out_stride_k_save], s[s_out_stride_k]
    s_mov_b32 s[s_wei_stride_k_save], s[s_wei_stride_k]
    ; pad b into multiplier of nxb
    s_mov_b32 s[s_dslice_dim_b], s[s_stride_dslice_hw]
    s_lshl_b32 s[s_out_stride_n0], s[s_out_stride_n], 3
    s_mul_i32 s[s_dtile_dy_neg], -1, s[s_dtile_dy]
    s_mul_i32 s[s_dtile_dx_neg], -1, s[s_dtile_dx]

    ; k1e transform
    v_mov_b32 v[v_gtc_ik1], v[v_gtc_ik1e]

    ; gemm_m_per_block:16, gemm_n_per_block:32
    s_mul_i32 s[s_tmp], s[s_dslice_dim_b], s[s_n]
    s_mul_i32 s[s_tmp+1], s[s_tmp], s[s_c]
    s_lshr_b32 s[0], s[s_tmp+1], 9
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080010 ; offset:16, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_block_gtc_ig,s_bx,s_magic_2,s_tmp+3,0,s_tmp
    s_mov_b32 s[s_bx], s[s_tmp+4]
    s_mul_i32 s[s_tmp], s[s_dslice_dim_b], s[s_n]
    s_lshr_b32 s[0], s[s_tmp], 5
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080018 ; offset:24, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_tmp+5,s_bx,s_magic_3,s_tmp+3,0,s_tmp
    s_mov_b64 s[0:1], s[s_magic_0+0:s_magic_0+1]
    ; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im
    s_lshl_b32 s[s_block_gtc_ic], s[s_tmp+5], 4
    s_mov_b32 s[s_tmp+5], s[s_dslice_dim_b] ; total number of n1b
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080000 ; offset:0, width:8
    .mdiv_u32_rem_ss s_block_gtc_in1b,s_block_gtc_in0,s_tmp+4,s_magic_4,s_tmp+3,s_tmp+5,s_tmp
    s_lshl_b32 s[s_block_gtc_in1b], s[s_block_gtc_in1b], 3
    s_lshl_b32 s[s_block_gtc_in0], s[s_block_gtc_in0], 2

    ; n1b transform
    v_add_u32 v[v_tmp+5], s[s_block_gtc_in1b], v[v_gtc_in1b]
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080008 ; offset:8, width:8
    .mdiv_u32_rem_vs v_tmp+4,v_gtc_in1,v_tmp+5,s_magic_5,s_tmp+3,s_dslice_dim_b,v_tmp
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080010 ; offset:16, width:8
    .mdiv_u32_rem_vs v_out_dslice_iw,v_out_dslice_ih,v_tmp+4,s_magic_6,s_tmp+3,s_dslice_w,v_tmp

    ; iHTildaLeft, iWTildaLeft
    v_add_u32 v[v_out_dslice_ih], s[s_dslice_h_left], v[v_out_dslice_ih]
    v_add_u32 v[v_out_dslice_iw], s[s_dslice_w_left], v[v_out_dslice_iw]
    v_mov_b32 v[v_out_iho], v[v_out_dslice_ih]
    v_mov_b32 v[v_out_iwo], v[v_out_dslice_iw]

    s_mov_b64 s[2:3], s[s_magic_5+0:s_magic_5+1]
    ; calculate output offset
    s_mul_i32 s[s_tmp+5], s[s_k], s[s_out_stride_k]
    s_lshl_b32 s[s_block_gtc_ig], s[s_block_gtc_ig], 1
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+5]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+5]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]
    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_in0], 4
    s_mul_i32 s[s_tmp], s[s_out_stride_n], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_out_stride_n], s[s_tmp+3]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]

    v_mov_b32 v[v_tmp], v[v_gtc_ik1]
    v_mul_lo_u32 v[v_tmp], s[s_out_stride_k], v[v_tmp]
    v_mov_b32 v[v_tmp+1], v[v_gtc_in1]
    v_mul_lo_u32 v[v_tmp+1], s[s_out_stride_n], v[v_tmp+1]
    v_add_lshl_u32 v[v_out_os_base], v[v_tmp], v[v_tmp+1], 1
    ; from ho, wo, os_base, compute final offset
    v_mad_u32_u24 v[v_tmp], s[s_wo], v[v_out_iho], v[v_out_iwo]
    v_lshl_add_u32 v[v_out_os], v[v_tmp], 1, v[v_out_os_base]
    v_cmp_gt_u32 vcc, s[s_ho], v[v_out_iho]
    v_cndmask_b32 v[v_out_flag], 0, 1, vcc
    v_cmp_gt_u32 vcc, s[s_wo], v[v_out_iwo]
    v_cndmask_b32 v[v_out_flag], 0, v[v_out_flag], vcc

    s_lshl_b32 s[s_out_stride_k_save], s[s_out_stride_k_save], 1
    s_lshl_b32 s[s_out_stride_n0], s[s_out_stride_n0], 1

    s_mul_i32 s[s_out_offset+0], 2, s[s_out_stride_n0]
    s_mul_i32 s[s_out_offset+1], 3, s[s_out_stride_n0]
    s_add_u32 s[s_out_offset+2], s[s_out_stride_k_save], s[s_out_stride_n0]
    s_add_u32 s[s_out_offset+3], s[s_out_stride_k_save], s[s_out_offset+0]
    s_add_u32 s[s_out_offset+4], s[s_out_stride_k_save], s[s_out_offset+1]
    s_mov_b32 s[s_p_out+2], 0xffffffff
    s_mov_b32 s[s_p_out+3], 0x27000
    ; load output
    .v_clear_nc v_gld_b, 8
    v_cmp_eq_u32 vcc, 1, v[v_out_flag]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_load_short_d16 v[v_gld_b+0], v[v_out_os], s[s_p_out:s_p_out+3], 0 offen offset:0
    buffer_load_short_d16 v[v_gld_b+1], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_stride_n0] offen offset:0
    buffer_load_short_d16 v[v_gld_b+2], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+0] offen offset:0
    buffer_load_short_d16 v[v_gld_b+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+1] offen offset:0
    buffer_load_short_d16 v[v_gld_b+4], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_stride_k_save] offen offset:0
    buffer_load_short_d16 v[v_gld_b+5], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+2] offen offset:0
    buffer_load_short_d16 v[v_gld_b+6], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+3] offen offset:0
    buffer_load_short_d16 v[v_gld_b+7], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+4] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]

    ; calculate wei offset
    s_mul_i32 s[s_tmp+2], s[s_k], s[s_wei_stride_k]
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+2]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+2]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp]
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], s[s_tmp+1]
    v_mov_b32 v[v_dtile_iy], s[s_dtile_iy]
    v_mov_b32 v[v_dtile_ix], s[s_dtile_ix]
    v_mov_b32 v[v_wei_iy], v[v_dtile_iy]
    v_mov_b32 v[v_wei_ix], v[v_dtile_ix]
    v_mov_b32 v[v_tmp], v[v_gtc_ic1]
    v_add_u32 v[v_tmp+5], s[s_block_gtc_ic], v[v_tmp]
    v_mul_lo_u32 v[v_tmp], s[s_wei_stride_c], v[v_tmp+5]
    v_mov_b32 v[v_tmp+1], v[v_gtc_ik1]
    v_mul_lo_u32 v[v_tmp+1], s[s_wei_stride_k], v[v_tmp+1]
    v_add_lshl_u32 v[v_wei_os_base], v[v_tmp], v[v_tmp+1], 1
    ; from y, x, os_base, compute final offset
    v_mad_u32_u24 v[v_tmp], v[v_wei_iy], s[s_x], v[v_wei_ix]
    v_lshl_add_u32 v[v_wei_os], v[v_tmp], 1, v[v_wei_os_base]

    s_lshl_b32 s[s_wei_stride_k_save], s[s_wei_stride_k_save], 1

    
    s_mov_b32 s[s_p_wei+2], 0xffffffff
    s_mov_b32 s[s_p_wei+3], 0x27000
    ; load weight
    .v_clear_nc v_gld_a, 2
    buffer_load_dword v[v_gld_a+0], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    buffer_load_dword v[v_gld_a+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k_save] offen offset:0

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get source matrix gemm index
    v_and_b32 v[v_gemm_in], 3, v[v_tmp+5]           ; block_n index 
    v_and_b32 v[v_gemm_im], 3, v[v_tmp+5]           ; block_m index 
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_and_b32 v[v_tmp + 0], 7, v[v_tmp+5]          ; block_n_per_wave index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 0], 2, v[v_gemm_in]
    v_lshrrev_b32 v[v_tmp+5], 3, v[v_tmp+5]
    v_and_b32 v[v_tmp + 1], 1, v[v_tmp+5]          ; block_m_per_wave index
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp + 1], 2, v[v_gemm_im]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]

    ; LDS store, out: k0,k1e,n0,n1b: 1x2x4x1, 1x8x1x8, order:4
    v_lshlrev_b32 v[v_tmp], 2, v[v_gtc_in1b]
    v_lshrrev_b32 v[v_tmp+1], 2, v[v_gtc_ik1e]
    v_lshl_add_u32 v[v_tmp], v[v_tmp+1], 7, v[v_tmp]
    v_and_b32 v[v_tmp+1], 3, v[v_gtc_ik1e]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+1]
    v_lshlrev_b32 v[v_sst_b_os], 1, v[v_tmp]
    v_add_u32 v[v_sst_b_os], 512, v[v_sst_b_os]

    ; LDS store, wei: k0,k1e,c0,c1: 1x2x1x2, 1x8x1x8, order:0
    v_lshlrev_b32 v[v_tmp], 2, v[v_gtc_ic1]
    v_lshrrev_b32 v[v_tmp+1], 2, v[v_gtc_ik1e]
    v_lshl_add_u32 v[v_tmp], v[v_tmp+1], 6, v[v_tmp]
    v_and_b32 v[v_tmp+1], 3, v[v_gtc_ik1e]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+1]
    v_lshlrev_b32 v[v_sst_a_os], 1, v[v_tmp]

    ; LDS load
    v_lshlrev_b32 v[v_sld_b_os], 3, v[v_gemm_in]
    v_lshlrev_b32 v[v_sld_a_os], 3, v[v_gemm_im]
    v_add_u32 v[v_sld_b_os], 512, v[v_sld_b_os]

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get dst matrix gemm index
    v_and_b32 v[v_tmp+0], 3, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_and_b32 v[v_tmp+2], 7, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 3, v[v_tmp+5]
    v_and_b32 v[v_tmp+3], 1, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_mov_b32 v[v_gemm_in], v[v_tmp+0]
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp+2], 2, v[v_gemm_in]
    v_lshlrev_b32 v[v_gemm_im], 2, v[v_tmp+3]

    ; init_co_lds_offset for xdlops
    v_lshrrev_b32 v[v_tmp], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp], 0, v[v_tmp]   ; thread id of lanegroup_m_per_cluster
    v_lshlrev_b32 v[v_co_sst], 2, v[v_tmp]
    v_lshrrev_b32 v[v_tmp+1], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp+1], 1  , v[v_tmp+1]   ; thread id of block_m_per_lanegroup
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+1], 2, v[v_co_sst]
    v_lshrrev_b32 v[v_tmp], 2, v[v_co_sst]
    v_lshlrev_b32 v[v_tmp+1], 2, v[v_gemm_in]   ; implicit transpose with m granularity:4 while store
    v_lshl_or_b32 v[v_co_sst], v[v_tmp], 7, v[v_tmp+1]
    v_lshlrev_b32 v[v_co_sst], 1, v[v_co_sst]
    v_lshlrev_b32 v[v_co_sld], 3, v[0]
    ; init_co_sub_m_index xdlops, block_size:64, macro-tile:16x32 sub_m_index:[0, 4]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:2, l_mw:1, l_mb:1, l_mt:4 | n_mc:1, n_ml:2, n_mv:1
    ; nd_stride:[1, 2, 1, 1, 2, 1, 1]
    v_lshrrev_b32 v[v_co_sub_m_index], 5, v[0]   ; get tid along m
    v_and_b32 v[v_tmp+0], 1, v[v_co_sub_m_index]                   ; => x_ml
    v_mov_b32 v[v_co_sub_m_index], v[v_tmp+0]      ; => accumulate x_ml
    v_lshlrev_b32 v[v_co_sub_m_index], 2, v[v_co_sub_m_index]
    ; init_co_sub_n_index xdlops
    v_and_b32 v[v_co_sub_n_index], 31, v[0]

    ; input offset
    s_mul_i32 s[s_tmp+2], s[s_c], s[s_in_stride_c]
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+2]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+2]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]
    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_in0], 4
    s_mul_i32 s[s_tmp], s[s_in_stride_n], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_in_stride_n], s[s_tmp+3]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]

    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_ic], 1
    s_mul_i32 s[s_tmp], s[s_in_stride_c], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_in_stride_c], s[s_tmp+3]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]

    ; compute v_co_sub_n_index along n0 x n1b : 4x8
    v_and_b32 v[v_in_in1b], 7, v[v_co_sub_n_index]     ; => N1B
    v_lshrrev_b32 v[v_in_in0], 3, v[v_co_sub_n_index]  ; => N0
    ;   compute from n1b
    v_add_u32 v[v_tmp+5], s[s_block_gtc_in1b], v[v_in_in1b]
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080008 ; offset:8, width:8
    .mdiv_u32_rem_vs v_tmp+4,v_in_in1,v_tmp+5,2,s_tmp+3,s_dslice_dim_b,v_tmp
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080010 ; offset:16, width:8
    .mdiv_u32_rem_vs v_in_dslice_iw,v_in_dslice_ih,v_tmp+4,3,s_tmp+3,s_dslice_w,v_tmp

    v_add_u32 v[v_in_dslice_ih], s[s_dslice_h_left], v[v_in_dslice_ih]
    v_add_u32 v[v_in_dslice_iw], s[s_dslice_w_left], v[v_in_dslice_iw]

    ; dslice_h,dslice_y -> hip,  dslice_w,dslicw_x -> wip
    s_mul_i32 s[s_tmp], s[s_dtile_iy], s[s_dilation_h]
    v_mul_lo_u32 v[v_tmp], s[s_stride_h], v[v_in_dslice_ih]
    v_add_u32 v[v_tmp], s[s_tmp], v[v_tmp]
    s_mul_i32 s[s_tmp+1], s[s_dtile_ix], s[s_dilation_w]
    v_mul_lo_u32 v[v_tmp+1], s[s_stride_w], v[v_in_dslice_iw]
    v_add_u32 v[v_tmp+1], s[s_tmp+1], v[v_tmp+1]
    ; v_tmp: hip, v_tmp+1: wip

    ; hip->h, wip->w
    v_sub_i32 v[v_in_ihi], v[v_tmp], s[s_pad_h]
    v_sub_i32 v[v_in_iwi], v[v_tmp+1], s[s_pad_w]

    v_cmp_gt_u32 vcc, s[s_hi], v[v_in_ihi]
    v_cndmask_b32 v[v_in_flag], 0, 1, vcc
    v_cmp_gt_u32 vcc, s[s_wi], v[v_in_iwi]
    v_cndmask_b32 v[v_in_flag], 0, v[v_in_flag], vcc

    ; add in_in0, in_in1
    v_lshl_or_b32 v[v_tmp+1], v[v_in_in0], 3, v[v_in_in1]
    v_mul_lo_u32 v[v_in_os], s[s_in_stride_n], v[v_tmp+1]
    ; add i_c
    v_mul_lo_u32 v[v_tmp], s[s_in_stride_c], v[v_co_sub_m_index]
    v_add_u32 v[v_in_os], v[v_in_os], v[v_tmp]
    ; add hi, wi
    v_mul_lo_u32 v[v_tmp+1], s[s_wi], v[v_in_ihi]
    v_add3_u32 v[v_in_os], v[v_in_os], v[v_tmp+1], v[v_in_iwi]
    v_lshlrev_b32 v[v_in_os], 1, v[v_in_os]

    ; move slice stride
    s_mov_b32 s[s_tmp+5], 16
    s_mov_b32 s[s_move_slice_k_k1], s[s_tmp+5]

    s_mov_b32 s[s_p_in+2], 0xffffffff
    s_mov_b32 s[s_p_in+3], 0x27000
    s_mul_i32 s[s_out_stride_k_k1], s[s_move_slice_k_k1], s[s_out_stride_k]  ; might be 0 or larger
    s_mul_i32 s[s_wei_stride_k_k1], s[s_move_slice_k_k1], s[s_c]  ; might be 0 or larger
    s_lshl_b32 s[s_out_stride_k_k1], s[s_out_stride_k_k1], 1
    s_lshl_b32 s[s_wei_stride_k_k1], s[s_wei_stride_k_k1], 1
    s_lshl_b32 s[s_out_stride_k], s[s_out_stride_k], 1
    s_lshl_b32 s[s_wei_stride_k], s[s_wei_stride_k], 1
    s_lshl_b32 s[s_in_stride_c], s[s_in_stride_c], 1
    s_mov_b32 s[s_knum], s[s_k]

    ; start MFMA loop, 8x32 wave tile with 1x1 repeat, 2x1 step
    s_waitcnt vmcnt(2)
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+0] offset:0
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+0+1] offset:64
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+2] offset:128
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+2+1] offset:192
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+4] offset:2
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+4+1] offset:66
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+6] offset:130
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+6+1] offset:194

    s_waitcnt vmcnt(0)
    v_pack_b32_f16 v[v_tmp] v[v_gld_a+0] v[v_gld_a+1]
    v_lshrrev_b32 v[v_gld_a+0], 16, v[v_gld_a+0]
    v_lshrrev_b32 v[v_gld_a+1], 16, v[v_gld_a+1]
    v_pack_b32_f16 v[v_tmp+1] v[v_gld_a+0] v[v_gld_a+1]
    ds_write2_b32 v[v_sst_a_os], v[v_tmp], v[v_tmp+1], offset0:0, offset1:2

    .v_clear_acc_c a_c, 8
    ; make sure acc WAR harzard, at least 1 nop for src_c
    s_sub_i32 s[s_kitr], s[s_knum], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8_mfma_end

    ; move slice window by unroll-k along gemm-k
    v_add_u32 v[v_out_os], s[s_out_stride_k_k1], v[v_out_os]
    v_add_u32 v[v_wei_os], s[s_wei_stride_k_k1], v[v_wei_os]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
L_igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8_mfma_body:
    ; do fma accumulate with unroll 16
    ds_read2_b64 v[v_a+0:v_a+3], v[v_sld_a_os], offset0:0, offset1:8
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] 
    ds_read2_b64 v[v_a+4+0:v_a+4+3], v[v_sld_a_os], offset0:16, offset1:24
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:256
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+0:v_a+1], v[v_b+0:v_b+1], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    .v_clear_nc v_gld_b, 8
    v_cmp_eq_u32 vcc, 1, v[v_out_flag]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_load_short_d16 v[v_gld_b+0], v[v_out_os], s[s_p_out:s_p_out+3], 0 offen offset:0
    buffer_load_short_d16 v[v_gld_b+1], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_stride_n0] offen offset:0
    buffer_load_short_d16 v[v_gld_b+2], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+0] offen offset:0
    buffer_load_short_d16 v[v_gld_b+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+1] offen offset:0
    buffer_load_short_d16 v[v_gld_b+4], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_stride_k_save] offen offset:0
    buffer_load_short_d16 v[v_gld_b+5], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+2] offen offset:0
    buffer_load_short_d16 v[v_gld_b+6], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+3] offen offset:0
    buffer_load_short_d16 v[v_gld_b+7], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+4] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+2:v_a+3], v[v_b+0:v_b+1], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    .v_clear_nc v_gld_a, 2
    ds_read2_b64 v[v_a+0:v_a+3], v[v_sld_a_os], offset0:32, offset1:40
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] offset:512
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+4:v_a+5], v[v_b+2:v_b+3], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    buffer_load_dword v[v_gld_a+0], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    buffer_load_dword v[v_gld_a+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k_save] offen offset:0
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+6:v_a+7], v[v_b+2:v_b+3], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    v_add_u32 v[v_out_os], s[s_out_stride_k_k1], v[v_out_os]
    v_add_u32 v[v_wei_os], s[s_wei_stride_k_k1], v[v_wei_os]
    ds_read2_b64 v[v_a+4+0:v_a+4+3], v[v_sld_a_os], offset0:48, offset1:56
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:768
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_waitcnt vmcnt(2)
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+0] offset:0
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+0+1] offset:64
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+2] offset:128
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+0:v_a+1], v[v_b+0:v_b+1], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+2+1] offset:192
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+4] offset:2
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+4+1] offset:66
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+2:v_a+3], v[v_b+0:v_b+1], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+6] offset:130
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+6+1] offset:194
    s_waitcnt vmcnt(0)
    v_pack_b32_f16 v[v_tmp] v[v_gld_a+0] v[v_gld_a+1]
    v_lshrrev_b32 v[v_gld_a+0], 16, v[v_gld_a+0]
    v_lshrrev_b32 v[v_gld_a+1], 16, v[v_gld_a+1]
    v_pack_b32_f16 v[v_tmp+1] v[v_gld_a+0] v[v_gld_a+1]
    ds_write2_b32 v[v_sst_a_os], v[v_tmp], v[v_tmp+1], offset0:0, offset1:2
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+4:v_a+5], v[v_b+2:v_b+3], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+6:v_a+7], v[v_b+2:v_b+3], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    s_sub_i32 s[s_kitr], s[s_kitr], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8_mfma_finishing
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_branch L_igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8_mfma_body
L_igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8_mfma_finishing:
L_igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8_mfma_end:
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read2_b64 v[v_a+0:v_a+3], v[v_sld_a_os], offset0:0, offset1:8
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] 
    ds_read2_b64 v[v_a+4+0:v_a+4+3], v[v_sld_a_os], offset0:16, offset1:24
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:256
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+0:v_a+1], v[v_b+0:v_b+1], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+2:v_a+3], v[v_b+0:v_b+1], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    ds_read2_b64 v[v_a+0:v_a+3], v[v_sld_a_os], offset0:32, offset1:40
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] offset:512
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+4:v_a+5], v[v_b+2:v_b+3], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+6:v_a+7], v[v_b+2:v_b+3], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    ds_read2_b64 v[v_a+4+0:v_a+4+3], v[v_sld_a_os], offset0:48, offset1:56
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:768
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+0:v_a+1], v[v_b+0:v_b+1], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+2:v_a+3], v[v_b+0:v_b+1], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+4:v_a+5], v[v_b+2:v_b+3], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+6:v_a+7], v[v_b+2:v_b+3], a[a_c+4:a_c+7]     ; repeat:0x0, step:1x0, num_a_c:4
    s_nop 3
    ; coalescing store, mapping:mt_m:16, mt_n:32, wt_m:8, wt_n:32, ws:1, r_m:1, r_n:1, s_m:2, s_n:1 | 4x4x4, lanegroup_m_tcbw:4x1x1x1, lanegroup_n_tcbw:1x4x1x1
    ; coalescing_groups:1, num_dword_per_group:8
    ; init_co_sub_m_index xdlops, block_size:64, macro-tile:16x32 sub_m_index:[0, 4]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:2, l_mw:1, l_mb:1, l_mt:4 | n_mc:1, n_ml:2, n_mv:1
    ; nd_stride:[1, 2, 1, 1, 2, 1, 1]
    ; start group 0, i_g_mr:0, i_g_ms:0, i_g_mw:0, i_g_mb:0, i_g_mt:0, m index start from 0
    s_barrier
    v_accvgpr_read_b32 v[v_c], a[a_c]
    v_accvgpr_read_b32 v[v_c+1], a[a_c+1]
    v_accvgpr_read_b32 v[v_c+2], a[a_c+2]
    v_accvgpr_read_b32 v[v_c+3], a[a_c+3]
    v_cvt_f16_f32_e32 v[v_c], v[v_c]
    v_cvt_f16_f32_e32 v[v_c+1], v[v_c+1]
    v_cvt_f16_f32_e32 v[v_c+2], v[v_c+2]
    v_cvt_f16_f32_e32 v[v_c+3], v[v_c+3]
    v_pack_b32_f16 v[v_c], v[v_c], v[v_c+1]
    v_pack_b32_f16 v[v_c+1], v[v_c+2], v[v_c+3]
    ds_write_b64 v[v_co_sst], v[v_c:v_c+1]    ; idword:0(0,0),  0x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    v_accvgpr_read_b32 v[v_c+4], a[a_c+4]
    v_accvgpr_read_b32 v[v_c+5], a[a_c+5]
    v_accvgpr_read_b32 v[v_c+6], a[a_c+6]
    v_accvgpr_read_b32 v[v_c+7], a[a_c+7]
    v_cvt_f16_f32_e32 v[v_c+4], v[v_c+4]
    v_cvt_f16_f32_e32 v[v_c+5], v[v_c+5]
    v_cvt_f16_f32_e32 v[v_c+6], v[v_c+6]
    v_cvt_f16_f32_e32 v[v_c+7], v[v_c+7]
    v_pack_b32_f16 v[v_c+4], v[v_c+4], v[v_c+5]
    v_pack_b32_f16 v[v_c+5], v[v_c+6], v[v_c+7]
    ds_write_b64 v[v_co_sst], v[v_c+4:v_c+4+1] offset:512   ; idword:64(2,0),  2x0 | /4, i_mr:0, i_ms:1, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    s_waitcnt lgkmcnt(0)
    s_barrier
    ;   load from lds
    ds_read_b64 v[v_c:v_c+1], v[v_co_sld] 
    ds_read_b64 v[v_c+2:v_c+2+1], v[v_co_sld] offset:512
    v_cmpx_eq_u32 vcc, 1, v[v_in_flag]
    ;   store to global, m index start from 0, m0:0, m1:0
    s_mov_b32 s[s_tmp], 0   ; i_m:0(i_m0:0,i_m1:0)
    s_waitcnt lgkmcnt(1)
    buffer_store_short v[v_c], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mov_b32 s[s_tmp], s[s_in_stride_c]   ; i_m:1(i_m0:0,i_m1:1)
    buffer_store_short_d16_hi v[v_c], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mul_i32 s[s_tmp], 2, s[s_in_stride_c]   ; i_m:2(i_m0:0,i_m1:2)
    buffer_store_short v[v_c+1], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mul_i32 s[s_tmp], 3, s[s_in_stride_c]   ; i_m:3(i_m0:0,i_m1:3)
    buffer_store_short_d16_hi v[v_c+1], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mul_i32 s[s_tmp], 8, s[s_in_stride_c]   ; i_m:8(i_m0:0,i_m1:8)
    s_waitcnt lgkmcnt(0)
    buffer_store_short v[v_c+2], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mul_i32 s[s_tmp], 9, s[s_in_stride_c]   ; i_m:9(i_m0:0,i_m1:9)
    buffer_store_short_d16_hi v[v_c+2], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mul_i32 s[s_tmp], 10, s[s_in_stride_c]   ; i_m:10(i_m0:0,i_m1:10)
    buffer_store_short v[v_c+3], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mul_i32 s[s_tmp], 11, s[s_in_stride_c]   ; i_m:11(i_m0:0,i_m1:11)
    buffer_store_short_d16_hi v[v_c+3], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_mov_b64 exec, -1
L_igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8_out:
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 72
    .amdhsa_next_free_sgpr 68
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8
    .symbol: igemm_bwd_gtcx_nchw_fp16_bx1_ex1_bt16x32x16_wt8x32x4_ws2x1_wr1x1_ta1x2x1x2_1x8x1x8_tb1x2x4x1_1x8x1x8.kd
    .sgpr_count: 74
    .vgpr_count: 72
    .kernarg_segment_align: 8
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [64, 1, 1]
    .max_flat_workgroup_size: 64
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
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
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_iy  , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_ix  , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_dy  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_dx  , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_y   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_x   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_h   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: dtile_w   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: dslice_y  , .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: dslice_x  , .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: dslice_h  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
    - { .name: dslice_w  , .size: 4, .offset: 128, .value_kind: by_value, .value_type: i32}
    - { .name: dslice_h_left, .size: 4, .offset: 132, .value_kind: by_value, .value_type: i32}
    - { .name: dslice_w_left, .size: 4, .offset: 136, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset: 140, .value_kind: by_value, .value_type: i32}
    - { .name: magic_0   , .size: 4, .offset: 144, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset: 148, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset: 152, .value_kind: by_value, .value_type: i32}
    - { .name: magic_3   , .size: 4, .offset: 156, .value_kind: by_value, .value_type: i32}
    - { .name: magic_4   , .size: 4, .offset: 160, .value_kind: by_value, .value_type: i32}
    - { .name: magic_5   , .size: 4, .offset: 164, .value_kind: by_value, .value_type: i32}
    - { .name: magic_6   , .size: 4, .offset: 168, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 172, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_1, .size: 4, .offset: 176, .value_kind: by_value, .value_type: i32}
    - { .name: _pack_0   , .size: 4, .offset: 180, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
