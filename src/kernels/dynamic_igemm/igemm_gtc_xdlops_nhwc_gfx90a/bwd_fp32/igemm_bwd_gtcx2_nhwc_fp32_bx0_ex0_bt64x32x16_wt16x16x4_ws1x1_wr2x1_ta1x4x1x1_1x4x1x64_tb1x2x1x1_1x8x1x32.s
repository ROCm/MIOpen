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
; starting of kernel igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32
; tensor_layout              : 'nhwc'
; gemm_m_per_block           : 64
; gemm_n_per_block           : 32
; gemm_k_per_block           : 16
; wave_tile_m                : 16
; wave_step_m                : 1
; wave_repeat_m              : 2
; wave_tile_n                : 16
; wave_step_n                : 1
; wave_repeat_n              : 1
; wave_tile_k                : 4
; tensor_a_thread_lengths    : [1, 4, 1, 1]
; tensor_a_cluster_lengths   : [1, 4, 1, 64]
; tensor_b_thread_lengths    : [1, 2, 1, 1]
; tensor_b_cluster_lengths   : [1, 8, 1, 32]
; direction                  : 'bwd'
; precision                  : 'fp32'
; nxb                        : 0
; nxe                        : 0
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
.set k_shift_pack_0, 160
.set k__pack_0, 164
.set k_end, 168
.set k_gload_out_k_stride, 16
.set k_gload_wei_c_stride, 0

.set s_ka, 0
.set s_bx, 2
.set s_by, 3
.set s_p_in, 4
.set s_p_wei, 8
.set s_p_out, 12
.set s_hi, 16
.set s_wi, 17
.set s_n, 18
.set s_k, 19
.set s_c, 20
.set s_group, 21
.set s_magic_0, 6
.set s_magic_1, 7
.set s_magic_2, 22
.set s_magic_3, 23
.set s_shift_m2, 24
.set s_shift_m3, 25
.set s_out_stride_wo, 26
.set s_out_stride_n, 27
.set s_wei_stride_k, 28
.set s_in_stride_wi, 29
.set s_in_stride_n, 30
.set s_block_gtc_ig, 31
.set s_block_gtc_ic, 32
.set s_block_gtc_inb, 33
.set s_move_slice_out_stride_k, 34
.set s_move_slice_wei_stride_k, 35
.set s_knum, 3
.set s_gemm_k_num_k, 36
.set s_dim_br, 37
.set s_dim_mp, 38
.set s_dim_mr, 39
.set s_dim_np, 40
.set s_move_slice_k_ix, 41
.set s_flag_need_acc_yx, 42
.set s_shift_pack_0, 42
.set s_kitr, 1
.set s_out_offset, 43
.set s_tmp, 44
.set s_end, 50

.set v_c, 0  ; coalescing:8, needed:0, resuable:14
.set v_a, 0
.set v_b, 4
.set v_gld_a, 6
.set v_gld_b, 10
.set v_sst_a_os, 12
.set v_sld_a_os, 13
.set v_sst_b_os, 14
.set v_sld_b_os, 15
.set v_out_os, 16
.set v_out_iho_list, 17
.set v_out_iwo_list, 18
.set v_out_flag, 19
.set v_out_flag_n, 20
.set v_out_ik, 21
.set v_out_inb, 22
.set v_out_in, 23
.set v_wei_os, 24
.set v_wei_ic, 25
.set v_wei_ik, 26
.set v_in_os, 27
.set v_in_flag_c, 25
.set v_in_inb, 22
.set v_co_sst, 23
.set v_co_sld, 28
.set v_gemm_in, 29
.set v_gemm_im, 30
.set v_co_sub_m_index, 30
.set v_co_sub_n_index, 29
.set v_tmp, 32
.set v_wei_tmp_pack, 5
.set v_wei_flag, 32
.set v_end, 48

.set a_c, 40
.set a_end, 48

.text
.globl igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32
.p2align 8
.type igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32,@function
igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],       s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],      s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],      s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx4 s[s_hi+0:s_hi+3],        s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dword s[s_c], s[s_ka+0:s_ka+1],    0+k_c
    s_load_dword s[s_group], s[s_ka+0:s_ka+1],     0+k_group
    s_load_dwordx2 s[s_magic_0+0:s_magic_0+1],  s[s_ka+0:s_ka+1],  0+k_magic_0
    s_load_dwordx2 s[s_magic_2+0:s_magic_2+1],  s[s_ka+0:s_ka+1],  0+k_magic_2
    s_load_dword s[s_shift_pack_0], s[s_ka+0:s_ka+1],  0+k_shift_pack_0
    ; out(e, k, nb0, nb1) thread_lengths: 1x4x1x1, cluster_length: 1x4x1x64, k_pack:4
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_out_ik], 3, v[v_tmp]
    v_lshlrev_b32 v[v_out_ik], 2, v[v_out_ik]
    v_lshrrev_b32 v[v_tmp], 2, v[v_tmp]
    v_and_b32 v[v_out_inb], 63, v[v_tmp]
    ; wei(e, k, c0, c1) thread_length: 1x2x1x1, cluster_length: 1x8x1x32, k_pack:4
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_wei_ic], 31, v[v_tmp]
    v_lshrrev_b32 v[v_tmp], 5, v[v_tmp]
    v_and_b32 v[v_wei_ik], 7, v[v_tmp]
    v_lshlrev_b32 v[v_wei_ik], 1, v[v_wei_ik]

    s_waitcnt lgkmcnt(0)

    ; calculate index
    s_mul_i32 s[s_out_stride_wo], s[s_k], s[s_group]
    s_mul_i32 s[s_tmp+2], s[s_wi], s[s_out_stride_wo]
    s_mul_i32 s[s_out_stride_n], s[s_hi], s[s_tmp+2]
    s_mov_b32 s[s_wei_stride_k], s[s_c]
    s_mul_i32 s[s_in_stride_wi], s[s_c], s[s_group]
    s_mul_i32 s[s_tmp+1], s[s_wi], s[s_in_stride_wi]
    s_mul_i32 s[s_in_stride_n], s[s_hi], s[s_tmp+1]
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
    s_mul_i32 s[s_dim_br], s[s_hi], s[s_wi]
    s_mul_i32 s[s_dim_mr], s[s_n], s[s_dim_br]
    s_add_u32 s[s_tmp], 63, s[s_dim_mr]
    s_lshr_b32 s[s_tmp+1], s[s_tmp], 6
    s_lshl_b32 s[s_dim_mp], s[s_tmp+1], 6
    s_add_u32 s[s_tmp], 31, s[s_c]
    s_lshr_b32 s[s_tmp+1], s[s_tmp], 5
    s_lshl_b32 s[s_dim_np], s[s_tmp+1], 5

    ; gemm_m_per_block:64, gemm_n_per_block:32, source_access_order:0
    s_lshr_b32 s[s_tmp], s[s_dim_mp], 6
    s_lshr_b32 s[s_tmp+1], s[s_dim_np], 5
    s_mul_i32 s[0], s[s_tmp+1], s[s_tmp]
    s_mov_b32 s[s_knum], s[s_k]
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080008 ; offset:8, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_block_gtc_ig,s_bx,s_magic_1,s_tmp+3,0,s_tmp
    s_mov_b32 s[s_bx], s[s_tmp+4]
    s_lshr_b32 s[0], s[s_dim_np], 5
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080000 ; offset:0, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_tmp+5,s_bx,s_magic_0,s_tmp+3,0,s_tmp
    ; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im
    s_lshl_b32 s[s_block_gtc_ic], s[s_tmp+4], 5
    s_lshl_b32 s[s_block_gtc_inb], s[s_tmp+5], 6
    v_add_u32 v[v_tmp+5], s[s_block_gtc_inb], v[v_out_inb]
    s_bfe_u32 s[s_shift_m3], s[s_shift_pack_0], 0x00080018 ; offset:24, width:8
    .mdiv_u32_rem_vs v_tmp+4,v_out_in,v_tmp+5,s_magic_3,s_shift_m3,s_dim_br,v_tmp
    s_bfe_u32 s[s_shift_m2], s[s_shift_pack_0], 0x00080010 ; offset:16, width:8
    .mdiv_u32_rem_vs v_out_iwo_list,v_out_iho_list,v_tmp+4,s_magic_2,s_shift_m2,s_wi,v_tmp
    v_cmp_gt_u32 vcc, s[s_n], v[v_out_in]
    v_cndmask_b32 v[v_tmp], 0, 1, vcc
    v_lshlrev_b32 v[v_out_flag_n], 0, v[v_tmp]
    s_lshl_b32 s[s_block_gtc_ig], s[s_block_gtc_ig], 2
    ; calculate wei offset
    s_mul_i32 s[s_tmp+2], s[s_k], s[s_wei_stride_k]
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+2]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+2]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp]
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], s[s_tmp+1]
    v_add_u32 v[v_tmp+5], s[s_block_gtc_ic], v[v_wei_ic]
    v_mul_lo_u32 v[v_tmp+4], s[s_wei_stride_k], v[v_wei_ik]
    v_add_lshl_u32 v[v_wei_os], v[v_tmp+4], v[v_tmp+5], 2
    v_cmp_gt_u32 vcc, s[s_c], v[v_tmp+5]
    v_cndmask_b32 v[v_wei_flag], 0, 1, vcc
    v_mov_b32 v[v_wei_tmp_pack], v[v_wei_flag]

    s_lshl_b32 s[s_wei_stride_k], s[s_wei_stride_k], 2

    .v_clear_nc v_gld_b, 2
    s_mov_b32 s[s_p_wei+2], 0xffffffff
    s_mov_b32 s[s_p_wei+3], 0x27000
    v_cmpx_le_u32 vcc, 1, v[v_wei_flag]
    buffer_load_dword v[v_gld_b], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    buffer_load_dword v[v_gld_b+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k] offen offset:0
    s_mov_b64 exec, -1

    ; calculate output offset
    s_mov_b32 s[s_out_offset], 0
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_k]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_k]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]

    v_mul_lo_u32 v[v_tmp+1], s[s_out_stride_n], v[v_out_in]
    s_lshl_b32 s[s_out_stride_wo], s[s_out_stride_wo], 2
    v_add_lshl_u32 v[v_tmp+4], v[v_out_ik], v[v_tmp+1], 2
    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_out_iho_list]
    v_add_u32 v[v_tmp], v[v_out_iwo_list], v[v_tmp]
    v_mul_lo_u32 v[v_tmp], s[s_out_stride_wo], v[v_tmp]
    v_add_u32 v[v_out_os], v[v_tmp+4], v[v_tmp]
    v_bfe_u32 v[v_tmp+1], v[v_out_flag_n],  0, 1
    v_cmp_gt_u32 vcc, s[s_hi], v[v_out_iho_list]
    v_cndmask_b32 v[v_out_flag], 0, v[v_tmp+1], vcc
    v_cmp_gt_u32 vcc, s[s_wi], v[v_out_iwo_list]
    v_cndmask_b32 v[v_out_flag], 0, v[v_out_flag], vcc

    s_mov_b32 s[s_p_out+2], 0xffffffff
    s_mov_b32 s[s_p_out+3], 0x27000
    ; load output, nxe:0
    .v_clear_nc v_gld_a, 4
    v_cmpx_le_u32 vcc, 1, v[v_out_flag]
    buffer_load_dwordx4 v[v_gld_a:v_gld_a+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset] offen offset:0
    s_mov_b64 exec, -1

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get source matrix gemm index, k_pack:4, v_pack:1, k_pack_per_thread:4
    v_and_b32 v[v_gemm_in], 15, v[v_tmp+5]           ; block_n index 
    v_and_b32 v[v_gemm_im], 15, v[v_tmp+5]           ; block_m index 
    v_lshlrev_b32 v[v_gemm_in], 2, v[v_gemm_in]   ; shift left k_pack:4
    v_lshlrev_b32 v[v_gemm_im], 2, v[v_gemm_im]   ; shift left k_pack:4
    v_lshrrev_b32 v[v_tmp+5], 4, v[v_tmp+5]
    v_and_b32 v[v_tmp + 0], 3, v[v_tmp+5]          ; block_k_per_wave index
    v_lshl_or_b32 v[v_gemm_in],  v[v_tmp + 0], 0, v[v_gemm_in]  ; or lanegroup_k_per_thread:1
    v_lshl_or_b32 v[v_gemm_im],  v[v_tmp + 0], 0, v[v_gemm_im]  ; or lanegroup_k_per_thread:1
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_and_b32 v[v_tmp + 2], 1, v[v_tmp+5]  ; waves_per_n index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 2], 6, v[v_gemm_in]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_and_b32 v[v_tmp + 3], 1, v[v_tmp+5]  ; waves_per_m index
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp + 3], 6, v[v_gemm_im]

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

    ; LDS store, out: e,k,nb0,nb1: 1x4x1x1, 1x4x1x64, k_pack:4, k_pack_gld_a:4, fp32
    v_lshlrev_b32 v[v_tmp+2], 2,  v[v_out_inb]
    v_lshrrev_b32 v[v_tmp+1], 2,  v[v_out_ik]
    v_lshl_or_b32 v[v_tmp], v[v_tmp+1], 8, v[v_tmp+2]
    v_lshlrev_b32 v[v_sst_a_os], 2, v[v_tmp]

    v_lshlrev_b32 v[v_sld_a_os], 2, v[v_gemm_im] ; LDS load out
    ; LDS store, wei: e,k,c: 1x2x1x1, 1x8x1x32, k_pack:4, k_pack_gld_b:2, fp32
    v_lshlrev_b32 v[v_tmp+2], 2,  v[v_wei_ic]
    v_lshrrev_b32 v[v_tmp+1], 2,  v[v_wei_ik]
    v_lshl_or_b32 v[v_tmp], v[v_tmp+1], 7, v[v_tmp+2]
    v_and_b32 v[v_tmp+2], 3, v[v_wei_ik]
    v_or_b32 v[v_tmp], v[v_tmp], v[v_tmp+2]
    v_lshlrev_b32 v[v_sst_b_os], 2, v[v_tmp]
    v_add_u32 v[v_sst_b_os], 4096, v[v_sst_b_os]

    v_lshlrev_b32 v[v_sld_b_os], 2, v[v_gemm_in] ; LDS load wei
    v_add_u32 v[v_sld_b_os], 4096, v[v_sld_b_os]
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
    v_lshl_or_b32 v[v_co_sst], v[v_tmp], 7, v[v_tmp+1]
    v_lshlrev_b32 v[v_co_sst], 2, v[v_co_sst]
    v_lshlrev_b32 v[v_co_sld], 4, v[0]
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:64x32 sub_m_index:[0, 4, 8, 12, 16, 20, 24, 28]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:2, l_ms:1, l_mw:1, l_mb:1, l_mt:4 | n_mc:4, n_ml:1, n_mv:2
    ; nd_stride:[4, 4, 1, 1, 1, 1, 2, 1]
    v_lshrrev_b32 v[v_co_sub_m_index], 5, v[0]   ; get tid along m
    v_and_b32 v[v_tmp+0], 3, v[v_co_sub_m_index]                   ; => x_mc
    v_lshrrev_b32 v[v_co_sub_m_index], 2  ,v[v_co_sub_m_index]
    v_and_b32 v[v_tmp+1], 1, v[v_co_sub_m_index]                   ; => x_mv
    v_lshlrev_b32 v[v_co_sub_m_index], 2, v[v_tmp+0]      ; => accumulate x_mc
    v_lshl_or_b32 v[v_co_sub_m_index], v[v_tmp+1], 4, v[v_co_sub_m_index]      ; => accumulate x_mv
    ; init_co_sub_n_index xdlops
    v_and_b32 v[v_co_sub_n_index], 31, v[0]

    v_add_u32 v[v_tmp], s[s_block_gtc_ic], v[v_co_sub_n_index]
    v_cmp_gt_u32 vcc, s[s_c], v[v_tmp]
    v_cndmask_b32 v[v_in_flag_c], 0, 1, vcc
    ; input offset
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_c]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_c]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]

    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_ic], 2
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp+3]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], 0

    s_lshl_b32 s[s_in_stride_wi], s[s_in_stride_wi], 2
    v_add_u32 v[v_in_inb], s[s_block_gtc_inb], v[v_co_sub_m_index]   ; total n*h_dslice*w_dslice
    v_mul_lo_u32 v[v_in_os], s[s_in_stride_wi], v[v_in_inb]
    v_lshlrev_b32 v[v_co_sub_n_index], 2, v[v_co_sub_n_index]
    v_add_u32 v[v_in_os], v[v_in_os], v[v_co_sub_n_index]
    ; move slice stride
    s_lshl_b32 s[s_gemm_k_num_k], s[s_k], 2
    v_bfe_u32 v[v_wei_flag], v[v_wei_tmp_pack], 0, 1
    s_mov_b32 s[s_move_slice_out_stride_k], 64
    s_mul_i32 s[s_move_slice_wei_stride_k], 16, s[s_wei_stride_k]

    s_mov_b32 s[s_p_in+2], 0xffffffff
    s_mov_b32 s[s_p_in+3], 0x27000
    ; start MFMA loop, 16x16 wave tile with 2x1 repeat, 1x1 step, k_pack:4
    s_waitcnt vmcnt(1)
    ds_write_b64 v[v_sst_b_os], v[v_gld_b:v_gld_b+1] 

    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+0:v_gld_a+0+3] 

    .v_clear_nc a_c, 8
    ; make sure acc WAR harzard, at least 1 nop for src_c
    s_sub_i32 s[s_kitr], s[s_knum], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32_mfma_end

    s_add_u32 s[s_out_offset],  s[s_move_slice_out_stride_k], s[s_out_offset]
    v_add_u32 v[v_wei_os], s[s_move_slice_wei_stride_k], v[v_wei_os]

    
    s_waitcnt lgkmcnt(0)
    s_barrier
L_igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32_mfma_body:
    ; do fma accumulate with unroll 16
    ds_read_b32 v[v_b], v[v_sld_b_os] 
    ds_read_b32 v[v_a], v[v_sld_a_os] 
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:512
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_cmpx_le_u32 vcc, 1, v[v_wei_flag]
    buffer_load_dword v[v_gld_b], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    buffer_load_dword v[v_gld_b+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k] offen offset:0
    s_mov_b64 exec, -1
    .v_clear_nc v_gld_a, 4
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:512 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_a+2], v[v_sld_a_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_a+3], v[v_sld_a_os] offset:1536 ; load i_k:1 into local buffer 1, repeat 1
    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4
    v_cmpx_le_u32 vcc, 1, v[v_out_flag]
    buffer_load_dwordx4 v[v_gld_a:v_gld_a+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset] offen offset:0
    s_mov_b64 exec, -1
    ds_read_b32 v[v_b], v[v_sld_b_os] offset:1024 ; load i_k:2 into local buffer 0, repeat 0
    ds_read_b32 v[v_a], v[v_sld_a_os] offset:2048 ; load i_k:2 into local buffer 0, repeat 0
    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+2], v[v_b+1], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    s_add_u32 s[s_out_offset],  s[s_move_slice_out_stride_k], s[s_out_offset]
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:2560 ; load i_k:2 into local buffer 0, repeat 1
    ds_read_b32 v[v_a+2], v[v_sld_a_os] offset:3072 ; load i_k:3 into local buffer 1, repeat 0
    s_waitcnt lgkmcnt(4)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+3], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4
    v_add_u32 v[v_wei_os], s[s_move_slice_wei_stride_k], v[v_wei_os]
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:1536 ; load i_k:3 into local buffer 1, repeat 0
    ds_read_b32 v[v_a+3], v[v_sld_a_os] offset:3584 ; load i_k:3 into local buffer 1, repeat 1
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_waitcnt vmcnt(1)
    ds_write_b64 v[v_sst_b_os], v[v_gld_b:v_gld_b+1]
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+0:v_gld_a+0+3]
    s_barrier
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+2], v[v_b+1], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    s_sub_i32 s[s_kitr], s[s_kitr], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32_mfma_finishing
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+3], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_branch L_igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32_mfma_body
L_igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32_mfma_finishing:
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+3], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4

L_igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32_mfma_end:
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b32 v[v_b], v[v_sld_b_os] 
    ds_read_b32 v[v_a], v[v_sld_a_os] 
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:512
    ; k iteration : 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:512 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_a+2], v[v_sld_a_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b32 v[v_a+3], v[v_sld_a_os] offset:1536 ; load i_k:1 into local buffer 1, repeat 1

    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4
    ds_read_b32 v[v_b], v[v_sld_b_os] offset:1024 ; load i_k:2 into local buffer 0, repeat 0
    ds_read_b32 v[v_a], v[v_sld_a_os] offset:2048 ; load i_k:2 into local buffer 0, repeat 0

    ; k iteration : 4
    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+2], v[v_b+1], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    ds_read_b32 v[v_a+1], v[v_sld_a_os] offset:2560 ; load i_k:2 into local buffer 0, repeat 1
    ds_read_b32 v[v_a+2], v[v_sld_a_os] offset:3072 ; load i_k:3 into local buffer 1, repeat 0

    s_waitcnt lgkmcnt(4)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+3], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4
    ds_read_b32 v[v_b+1], v[v_sld_b_os] offset:1536 ; load i_k:3 into local buffer 1, repeat 0
    ds_read_b32 v[v_a+3], v[v_sld_a_os] offset:3584 ; load i_k:3 into local buffer 1, repeat 1

    ; k iteration : 8
    s_waitcnt lgkmcnt(4)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a], v[v_b], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4

    s_waitcnt lgkmcnt(3)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+1], v[v_b], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4

    ; k iteration : 12
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x4f32 v[a_c+0:a_c+3], v[v_a+2], v[v_b+1], v[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4

    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x4f32 v[a_c+4:a_c+7], v[v_a+3], v[v_b+1], v[a_c+4:a_c+7]     ; repeat:1x0, step:0x0, num_a_c:4

    s_nop 9
    ; coalescing store, mapping:mt_m:64, mt_n:32, wt_m:16, wt_n:16, ws:4, r_m:2, r_n:1, s_m:1, s_n:1 | 16x16x4, lanegroup_m_tcbw:4x4x1x1, lanegroup_n_tcbw:1x16x1x1
    ; coalescing_groups:1, num_dword_per_group:8
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:64x32 sub_m_index:[0, 4, 8, 12, 16, 20, 24, 28]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:2, l_ms:1, l_mw:1, l_mb:1, l_mt:4 | n_mc:4, n_ml:1, n_mv:2
    ; nd_stride:[4, 1, 1, 1, 1, 2, 1]
    ; start group 0, i_g_mr:0, i_g_ms:0, i_g_mw:0, i_g_mb:0, i_g_mt:0, m index start from 0
    s_barrier
    ds_write_b128 v[v_co_sst], v[a_c:a_c+3]    ; idword:0(0,0),  0x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+4:a_c+4+3] offset:4096   ; idword:256(8,0),  8x0 | /4, i_mr:1, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    s_mov_b32 s[s_tmp], 0   ; i_m:0(i_m0:0,i_m1:0)
    v_add_u32 v[v_in_inb], s[s_block_gtc_inb], v[v_co_sub_m_index]
    v_mov_b32 v[v_tmp], v[v_in_inb]
    s_waitcnt lgkmcnt(0)
    s_barrier
    ;   load from lds, i_ssgroup:0, num_sld_per_ssgroup:2
    ds_read_b128 v[v_c:v_c+3], v[v_co_sld] 
    ds_read_b128 v[v_c+4:v_c+4+3], v[v_co_sld] offset:4096
    v_cmpx_eq_u32 vcc, 1, v[v_in_flag_c]
    ;   store to global, m index start from 0, m0:0, m1:0
    s_waitcnt lgkmcnt(1)
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b32 s[s_tmp], s[s_in_stride_wi]   ; i_m:1(i_m0:0,i_m1:1)
    v_add_u32 v[v_tmp], 1, v[v_in_inb]
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c+1], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 2, s[s_in_stride_wi]   ; i_m:2(i_m0:0,i_m1:2)
    v_add_u32 v[v_tmp], 2, v[v_in_inb]
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c+2], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 3, s[s_in_stride_wi]   ; i_m:3(i_m0:0,i_m1:3)
    v_add_u32 v[v_tmp], 3, v[v_in_inb]
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c+3], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 32, s[s_in_stride_wi]   ; i_m:32(i_m0:0,i_m1:32)
    v_add_u32 v[v_tmp], 32, v[v_in_inb]
    s_waitcnt lgkmcnt(0)
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c+4], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 33, s[s_in_stride_wi]   ; i_m:33(i_m0:0,i_m1:33)
    v_add_u32 v[v_tmp], 33, v[v_in_inb]
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c+5], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 34, s[s_in_stride_wi]   ; i_m:34(i_m0:0,i_m1:34)
    v_add_u32 v[v_tmp], 34, v[v_in_inb]
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c+6], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 35, s[s_in_stride_wi]   ; i_m:35(i_m0:0,i_m1:35)
    v_add_u32 v[v_tmp], 35, v[v_in_inb]
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dword v[v_c+7], v[v_in_os], s[s_p_in:s_p_in+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b64 exec, -1
L_igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32_out:
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 48
    .amdhsa_next_free_sgpr 50
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_tg_split 0
    .amdhsa_accum_offset 40
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32
    .symbol: igemm_bwd_gtcx2_nhwc_fp32_bx0_ex0_bt64x32x16_wt16x16x4_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x2x1x1_1x8x1x32.kd
    .sgpr_count: 56
    .vgpr_count: 48
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
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
    - { .name: shift_pack_0, .size: 4, .offset: 160, .value_kind: by_value, .value_type: i32}
    - { .name: ks        , .size: 4, .offset: 164, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
