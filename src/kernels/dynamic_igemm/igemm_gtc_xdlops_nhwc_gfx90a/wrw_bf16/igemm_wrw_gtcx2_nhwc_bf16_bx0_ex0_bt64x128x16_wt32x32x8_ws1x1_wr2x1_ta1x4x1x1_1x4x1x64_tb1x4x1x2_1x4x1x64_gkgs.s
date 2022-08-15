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
; generated by igemm_codegen.py (df99f96d6cb25ae4786f5c06f4ab7ce6b887d384)
;
.include "igemm_wrw_gtcx2_nhwc_bf16_utils.inc"

;----------------------------------------------------------
; starting of kernel igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs
; tensor_layout              : 'nhwc'
; gemm_m_per_block           : 64
; gemm_n_per_block           : 128
; gemm_k_per_block           : 16
; wave_tile_m                : 32
; wave_step_m                : 1
; wave_repeat_m              : 2
; wave_tile_n                : 32
; wave_step_n                : 1
; wave_repeat_n              : 1
; wave_tile_k                : 8
; tensor_a_thread_lengths    : [1, 4, 1, 1]
; tensor_a_cluster_lengths   : [1, 4, 1, 64]
; tensor_b_thread_lengths    : [1, 4, 1, 2]
; tensor_b_cluster_lengths   : [1, 4, 1, 64]
; direction                  : 'wrw'
; precision                  : 'bf16'
; nxb                        : 0
; nxe                        : 0
; gemm_k_global_split        : 1
; 
; block_size                 : 256
; lds_total                  : 16384
; lds_buffer_num             : 2
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
.set s_out_offset, 63
.set s_sub_n, 65
.set s_in_stride_move_n, 66
.set s_out_stride_move_n, 67
.set s_k_padded, 68
.set s_c_padded, 69
.set s_out_move_step, 70
.set s_in_move_step, 71
.set s_tmp, 72
.set s_end, 78

.set v_c, 0  ; coalescing:16, needed:0, resuable:29
.set v_a, 0
.set v_b, 8
.set v_gld_a, 12
.set v_gld_b, 16
.set v_sst_a_os, 20
.set v_sst_b_os, 21
.set v_sld_a_os, 22
.set v_sld_b_os, 23
.set v_in_os, 24
.set v_in_os_base, 25
.set v_gtc_in, 26
.set v_out_os, 27
.set v_out_os_base, 28
.set v_co_sst, 29
.set v_co_sld, 30
.set v_wei_os, 31
.set v_wei_c_flag, 32
.set v_gtc_iec, 33
.set v_wei_ie, 34
.set v_gtc_ic, 35
.set v_gtc_ie, 36
.set v_gtc_ik, 37
.set v_gtc_inb_a, 38
.set v_gemm_in, 39
.set v_gemm_im, 40
.set v_wei_ic, 41
.set v_wei_iec, 42
.set v_co_sub_m_index, 43
.set v_co_sub_n_index, 44
.set v_cur_k, 45
.set v_tmp, 46
.set v_end, 88

.set a_c, 56
.set a_end, 88

.text
.globl igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs
.p2align 8
.type igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs,@function
igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],       s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],      s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],      s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx16 s[s_hi+0:s_hi+15],        s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dwordx2  s[s_group+0:s_group+1],      s[s_ka+0:s_ka+1],    0+k_group

    ; input, thread(1,nb,1,c): 1x4x1x2, cluster(1,nb,1,ec): 1x4x1x64
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_iec], 63, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_iec], 1, v[v_gtc_iec]
    v_lshrrev_b32 v[v_tmp], 6, v[v_tmp]
    v_and_b32 v[v_gtc_inb_a], 3, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_inb_a], 2, v[v_gtc_inb_a]

    ; output, thread(1,nb,1,k): 1x4x1x1, cluster(1,nb,1,k) 1x4x1x64
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_ik], 63, v[v_tmp]

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
    s_lshl_b32 s[s_tmp+4], s[s_tmp], 1
    s_lshl_b32 s[s_tmp+5], s[s_tmp+1], 1
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
    s_add_u32 s[s_tmp], 63, s[s_k]
    s_lshr_b32 s[s_tmp], s[s_tmp], 6
    s_lshl_b32 s[s_k_padded], s[s_tmp], 6
    ; pad c for 1x1 cases
    s_add_u32 s[s_tmp], 127, s[s_c]
    s_lshr_b32 s[s_tmp], s[s_tmp], 7
    s_lshl_b32 s[s_c_padded], s[s_tmp], 7

    ; add block i_n
    ; gemm_m_per_block:64, gemm_n_per_block:128
    s_lshr_b32 s[0], s[s_c_padded], 7
    s_lshr_b32 s[s_tmp], s[s_k_padded], 6
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
    s_lshl_b32 s[s_block_gtc_ik], s[s_tmp+5], 6
    s_lshl_b32 s[s_block_gtc_iec], s[s_tmp+4], 7

    ; config for output and input range
    s_mul_i32 s[s_p_out+2], s[s_n], s[s_out_stride_n]
    s_lshl_b32 s[s_p_out+2], s[s_p_out+2], 1
    s_mul_i32 s[s_p_in+2], s[s_n], s[s_in_stride_n]
    s_lshl_b32 s[s_p_in+2], s[s_p_in+2], 1
    v_add_u32 v[v_gtc_ic], s[s_block_gtc_iec], v[v_gtc_iec]
    ; calculate input offset
    s_lshl_b32 s[s_block_gtc_ig], s[s_block_gtc_ig], 1
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_c]
    s_sub_u32 s[s_p_in+2], s[s_p_in+2], s[s_tmp]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]

    v_mul_lo_u32 v[v_tmp], v[v_gtc_inb_a], s[s_in_stride_wi]
    v_add_lshl_u32 v[v_in_os], v[v_tmp], v[v_gtc_ic], 1

    s_lshl_b32 s[s_in_stride_wi], s[s_in_stride_wi], 1

    s_mul_i32 s[s_tmp], s[s_in_stride_wi], 2
    s_mov_b32 s[s_in_offset+0], s[s_tmp]
    s_mul_i32 s[s_tmp], s[s_in_stride_wi], 3
    s_mov_b32 s[s_in_offset+1], s[s_tmp]
    ; load input
    buffer_load_dword v[v_gld_b+0], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:0
    buffer_load_dword v[v_gld_b+1], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_stride_wi] offen offset:0
    buffer_load_dword v[v_gld_b+2], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_offset+0] offen offset:0
    buffer_load_dword v[v_gld_b+3], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_offset+1] offen offset:0

    ; calculate out offset
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_k]
    s_sub_u32 s[s_p_out+2], s[s_p_out+2], s[s_tmp]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]

    v_add_u32 v[v_cur_k], s[s_block_gtc_ik], v[v_gtc_ik]
    ;s_mul_i32 s[s_tmp], s[s_out_stride_n], s[s_block_gtc_in]
    ;v_add_lshl_u32 v[v_tmp+1], v[v_cur_k], s[s_tmp], 1
    v_mul_lo_u32 v[v_tmp], v[v_gtc_inb_a], s[s_out_stride_wo]
    v_add_lshl_u32 v[v_out_os], v[v_tmp], v[v_cur_k], 1

    s_lshl_b32 s[s_out_stride_wo], s[s_out_stride_wo], 1

    s_mul_i32 s[s_tmp], s[s_out_stride_wo], 2
    s_mov_b32 s[s_out_offset+0], s[s_tmp]
    s_mul_i32 s[s_tmp], s[s_out_stride_wo], 3
    s_mov_b32 s[s_out_offset+1], s[s_tmp]
    ; load output
    buffer_load_short_d16 v[v_gld_a+0], v[v_out_os], s[s_p_out:s_p_out+3], 0 offen offset:0
    buffer_load_short_d16 v[v_gld_a+1], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_stride_wo] offen offset:0
    buffer_load_short_d16 v[v_gld_a+2], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+0] offen offset:0
    buffer_load_short_d16 v[v_gld_a+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+1] offen offset:0

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get source matrix gemm index, k_pack:4, v_pack:1, k_pack_per_thread:1
    v_and_b32 v[v_gemm_in], 31, v[v_tmp+5]           ; block_n index 
    v_and_b32 v[v_gemm_im], 31, v[v_tmp+5]           ; block_m index 
    v_lshlrev_b32 v[v_gemm_in], 2, v[v_gemm_in]   ; shift left k_pack:4
    v_lshlrev_b32 v[v_gemm_im], 2, v[v_gemm_im]   ; shift left k_pack:4
    v_lshrrev_b32 v[v_tmp+5], 5, v[v_tmp+5]
    v_and_b32 v[v_tmp + 0], 1, v[v_tmp+5]          ; block_k_per_wave index
    v_and_b32 v[v_tmp + 1], 0, v[v_tmp + 0]   ; and k_pack_per_thread:1
    v_lshrrev_b32 v[v_tmp + 0], 0, v[v_tmp + 0] ; shift right k_pack_per_thread:1
    v_lshl_or_b32 v[v_gemm_in],  v[v_tmp + 1], 2, v[v_gemm_in]  ; or lanegroup_k_per_thread:4
    v_lshl_or_b32 v[v_gemm_im],  v[v_tmp + 1], 2, v[v_gemm_im]  ; or lanegroup_k_per_thread:4
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 0], 9, v[v_gemm_in]
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp + 0], 8, v[v_gemm_im]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_and_b32 v[v_tmp + 2], 3, v[v_tmp+5]  ; waves_per_n index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 2], 7, v[v_gemm_in]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get dst matrix gemm index
    v_and_b32 v[v_tmp+0], 31, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 5, v[v_tmp+5]
    v_and_b32 v[v_tmp+1], 1, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_mov_b32 v[v_co_sst], v[v_tmp+0]
    v_lshlrev_b32 v[v_co_sld], 2, v[v_tmp+1]
    v_and_b32 v[v_tmp+0], 3, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+0], 5, v[v_co_sst]

    ; LDS store, in: 1,nb,1,ec: 1x4x1x2, 1x4x1x64
    v_sub_i32 v[v_gtc_inb_a], v[v_gtc_inb_a], s[s_sub_n]
    v_lshlrev_b32 v[v_tmp+2], 2,  v[v_gtc_iec]
    v_lshrrev_b32 v[v_tmp+1], 2,  v[v_gtc_inb_a]
    v_lshl_add_u32 v[v_tmp], v[v_tmp+1], 9, v[v_tmp+2]
    v_and_b32 v[v_tmp+2], 0, v[v_gtc_inb_a]
    v_lshl_or_b32 v[v_tmp], v[v_tmp+2], 2, v[v_tmp]
    v_lshlrev_b32 v[v_sst_b_os], 1, v[v_tmp]
    v_add_u32 v[v_sst_b_os], 2048, v[v_sst_b_os]

    ; LDS store, out: 1,nb,1,k: 1x4x1x1, 1x4x1x64
    v_lshlrev_b32 v[v_tmp+2], 2,  v[v_gtc_ik]
    v_lshrrev_b32 v[v_tmp+1], 2,  v[v_gtc_inb_a]
    v_lshl_add_u32 v[v_tmp], v[v_tmp+1], 8, v[v_tmp+2]
    v_and_b32 v[v_tmp+2], 0, v[v_gtc_inb_a]
    v_lshl_or_b32 v[v_tmp], v[v_tmp+2], 2, v[v_tmp]
    v_lshlrev_b32 v[v_sst_a_os], 1, v[v_tmp]

    ; LDS load
    v_lshlrev_b32 v[v_sld_b_os], 1, v[v_gemm_in]
    v_lshlrev_b32 v[v_sld_a_os], 1, v[v_gemm_im]
    v_add_u32 v[v_sld_b_os], 2048, v[v_sld_b_os]

    v_mov_b32 v[v_gemm_in], v[v_co_sst]
    v_mov_b32 v[v_gemm_im], v[v_co_sld]
    ; init_co_lds_offset for xdlops
    v_lshrrev_b32 v[v_tmp], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp],  1 v[v_tmp]   ; thread id of lanegroup_m_per_cluster
    v_lshlrev_b32 v[v_co_sst], 2, v[v_tmp]
    v_lshrrev_b32 v[v_tmp], 2, v[v_co_sst]
    v_lshlrev_b32 v[v_tmp+1], 2, v[v_gemm_in]   ; implicit transpose with m granularity:4 while store
    v_lshl_or_b32 v[v_co_sst], v[v_tmp], 9, v[v_tmp+1]
    v_lshlrev_b32 v[v_co_sst], 2, v[v_co_sst]
    v_lshlrev_b32 v[v_co_sld], 4, v[0]
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:64x128 sub_m_index:[0, 4]
    ; g_mr:2, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:1, l_mw:1, l_mb:4, l_mt:4 | n_mc:2, n_ml:1, n_mv:1
    ; nd_stride:[4, 2, 1, 4, 1, 1, 1, 1]
    v_lshrrev_b32 v[v_co_sub_m_index], 7, v[0]   ; get tid along m
    v_and_b32 v[v_tmp+0], 1, v[v_co_sub_m_index]                   ; => x_mc
    v_lshlrev_b32 v[v_co_sub_m_index], 2, v[v_tmp+0]      ; => accumulate x_mc
    ; init_co_sub_n_index xdlops
    v_and_b32 v[v_co_sub_n_index], 127, v[0]

    ; weight offset
    s_mul_i32 s[s_block_gtc_ig], s[s_block_gtc_ig], 2
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

    ; compute v_co_sub_n_index along ec : 128
    v_and_b32 v[v_wei_iec], 127, v[v_co_sub_n_index]     ; => EC

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
    s_lshl_b32 s[s_out_move_step], s[s_tmp], 5
    s_lshl_b32 s[s_in_move_step], s[s_tmp+1], 5
    ; move slice stride
    s_lshl_b32 s[s_wei_stride_k], s[s_wei_stride_k], 2
    s_add_i32 s[s_knum], s[s_gemmk_per_wg], 15
    s_lshr_b32 s[s_knum], s[s_knum], 4
    s_lshl_b32 s[s_knum], s[s_knum], 4

    ; start MFMA loop, 32x32 wave tile with 2x1 repeat, 1x1 step, k_pack:4
    s_waitcnt vmcnt(4)
    v_pack_b32_f16 v[v_tmp], v[v_gld_b], v[v_gld_b+1]
    v_pack_b32_f16 v[v_tmp+1], v[v_gld_b+2], v[v_gld_b+3]
    v_pack_b32_f16 v[v_tmp+2], v[v_gld_b], v[v_gld_b+1] op_sel:[1, 1]
    v_pack_b32_f16 v[v_tmp+3], v[v_gld_b+2], v[v_gld_b+3] op_sel:[1, 1]
    ds_write_b128 v[v_sst_b_os], v[v_tmp:v_tmp+3] 

    s_waitcnt vmcnt(0)
    v_pack_b32_f16 v[v_tmp], v[v_gld_a], v[v_gld_a+1]
    v_pack_b32_f16 v[v_tmp+1], v[v_gld_a+2], v[v_gld_a+3]
    ds_write_b64 v[v_sst_a_os], v[v_tmp:v_tmp+1] 

    .v_clear_nc a_c, 32
    ; make sure acc WAR harzard, at least 1 nop for src_c
    s_sub_i32 s[s_kitr], s[s_knum], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs_mfma_end

    v_xor_b32 v[v_sst_b_os], 0x2000, v[v_sst_b_os] ; switch double buffer b store
    v_xor_b32 v[v_sst_a_os], 0x2000, v[v_sst_a_os] ; switch double buffer a store
    v_add_u32 v[v_in_os], v[v_in_os], s[s_in_move_step]
    v_add_u32 v[v_out_os], v[v_out_os], s[s_out_move_step]
L_igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs_mfma_body:
    ; load input
    buffer_load_dword v[v_gld_b+0], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:0
    buffer_load_dword v[v_gld_b+1], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_stride_wi] offen offset:0
    buffer_load_dword v[v_gld_b+2], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_offset+0] offen offset:0
    buffer_load_dword v[v_gld_b+3], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_offset+1] offen offset:0
    ; load output
    buffer_load_short_d16 v[v_gld_a+0], v[v_out_os], s[s_p_out:s_p_out+3], 0 offen offset:0
    buffer_load_short_d16 v[v_gld_a+1], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_stride_wo] offen offset:0
    buffer_load_short_d16 v[v_gld_a+2], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+0] offen offset:0
    buffer_load_short_d16 v[v_gld_a+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_out_offset+1] offen offset:0
    ; do fma accumulate with unroll 16
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] 
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] 
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:256
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_32x32x8bf16_1k v[a_c+0:a_c+15], v[v_a+0:v_a+1], v[v_b+0:v_b+1], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    v_add_u32 v[v_in_os], v[v_in_os], s[s_in_move_step]
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:2048 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+4:v_a+4+1], v[v_sld_a_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x8bf16_1k v[a_c+16:a_c+31], v[v_a+2:v_a+3], v[v_b+0:v_b+1], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    v_add_u32 v[v_out_os], v[v_out_os], s[s_out_move_step]
    ds_read_b64 v[v_a+6:v_a+6+1], v[v_sld_a_os] offset:1280 ; load i_k:1 into local buffer 1, repeat 1
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_32x32x8bf16_1k v[a_c+0:a_c+15], v[v_a+4:v_a+5], v[v_b+2:v_b+3], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    
    s_waitcnt lgkmcnt(0)
    s_waitcnt vmcnt(4)
    v_pack_b32_f16 v[v_tmp], v[v_gld_b], v[v_gld_b+1]
    v_pack_b32_f16 v[v_tmp+1], v[v_gld_b+2], v[v_gld_b+3]
    v_pack_b32_f16 v[v_tmp+2], v[v_gld_b], v[v_gld_b+1] op_sel:[1, 1]
    v_pack_b32_f16 v[v_tmp+3], v[v_gld_b+2], v[v_gld_b+3] op_sel:[1, 1]
    ds_write_b128 v[v_sst_b_os], v[v_tmp:v_tmp+3]
    s_waitcnt vmcnt(0)
    v_pack_b32_f16 v[v_tmp], v[v_gld_a], v[v_gld_a+1]
    v_pack_b32_f16 v[v_tmp+1], v[v_gld_a+2], v[v_gld_a+3]
    ds_write_b64 v[v_sst_a_os], v[v_tmp:v_tmp+1]
    v_mfma_f32_32x32x8bf16_1k v[a_c+16:a_c+31], v[v_a+6:v_a+7], v[v_b+2:v_b+3], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    v_xor_b32 v[v_sld_b_os], 8192, v[v_sld_b_os] ; switch double buffer b load
    v_xor_b32 v[v_sld_a_os], 8192, v[v_sld_a_os] ; switch double buffer a load
    s_sub_i32 s[s_kitr], s[s_kitr], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs_mfma_finishing
    v_xor_b32 v[v_sst_b_os], 8192, v[v_sst_b_os] ; switch double buffer b store
    v_xor_b32 v[v_sst_a_os], 8192, v[v_sst_a_os] ; switch double buffer a store
    s_branch L_igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs_mfma_body
L_igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs_mfma_finishing:

L_igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs_mfma_end:
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] 
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] 
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:256
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_32x32x8bf16_1k v[a_c+0:a_c+15], v[v_a+0:v_a+1], v[v_b+0:v_b+1], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:2048 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+4:v_a+4+1], v[v_sld_a_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0

    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x8bf16_1k v[a_c+16:a_c+31], v[v_a+2:v_a+3], v[v_b+0:v_b+1], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    ds_read_b64 v[v_a+6:v_a+6+1], v[v_sld_a_os] offset:1280 ; load i_k:1 into local buffer 1, repeat 1

    s_waitcnt lgkmcnt(1)
    v_mfma_f32_32x32x8bf16_1k v[a_c+0:a_c+15], v[v_a+4:v_a+5], v[v_b+2:v_b+3], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16

    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x8bf16_1k v[a_c+16:a_c+31], v[v_a+6:v_a+7], v[v_b+2:v_b+3], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16

    s_nop 15
    s_nop 2
    ; coalescing store, mapping:mt_m:64, mt_n:128, wt_m:32, wt_n:32, ws:4, r_m:2, r_n:1, s_m:1, s_n:1 | 32x32x8, lanegroup_m_tcbw:4x2x4x1, lanegroup_n_tcbw:1x32x1x1
    ; coalescing_groups:2, num_dword_per_group:16
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:64x128 sub_m_index:[0, 4]
    ; g_mr:2, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:1, l_mw:1, l_mb:4, l_mt:4 | n_mc:2, n_ml:1, n_mv:1
    ; nd_stride:[2, 1, 4, 1, 1, 1, 1]
    ; start group 0, i_g_mr:0, i_g_ms:0, i_g_mw:0, i_g_mb:0, i_g_mt:0, m index start from 0
    s_barrier
    ds_write_b128 v[v_co_sst], v[a_c:a_c+3]    ; idword:0(0,0),  0x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+4:a_c+4+3] offset:4096   ; idword:256(2,0),  2x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+8:a_c+8+3] offset:8192   ; idword:512(4,0),  4x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+12:a_c+12+3] offset:12288   ; idword:768(6,0),  6x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    s_mov_b32 s[s_tmp], 0   ; i_m:0(i_m0:0,i_m1:0)
    v_add_u32 v[v_cur_k], s[s_block_gtc_ik], v[v_co_sub_m_index]
    v_mov_b32 v[v_tmp], v[v_cur_k]
    s_waitcnt lgkmcnt(0)
    s_barrier
    ;   load from lds, i_ssgroup:0, num_sld_per_ssgroup:4
    ds_read_b128 v[v_c:v_c+3], v[v_co_sld] 
    ds_read_b128 v[v_c+4:v_c+4+3], v[v_co_sld] offset:4096
    ds_read_b128 v[v_c+8:v_c+8+3], v[v_co_sld] offset:8192
    ds_read_b128 v[v_c+12:v_c+12+3], v[v_co_sld] offset:12288
    v_cmpx_eq_u32 vcc, 1, v[v_wei_c_flag]
    ;   store to global, m index start from 0, m0:0, m1:0
    s_waitcnt lgkmcnt(3)
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
    s_mul_i32 s[s_tmp], 8, s[s_wei_stride_k]   ; i_m:8(i_m0:0,i_m1:8)
    v_add_u32 v[v_tmp], 8, v[v_cur_k]
    s_waitcnt lgkmcnt(2)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+4], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 9, s[s_wei_stride_k]   ; i_m:9(i_m0:0,i_m1:9)
    v_add_u32 v[v_tmp], 9, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+5], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 10, s[s_wei_stride_k]   ; i_m:10(i_m0:0,i_m1:10)
    v_add_u32 v[v_tmp], 10, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+6], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 11, s[s_wei_stride_k]   ; i_m:11(i_m0:0,i_m1:11)
    v_add_u32 v[v_tmp], 11, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+7], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 16, s[s_wei_stride_k]   ; i_m:16(i_m0:0,i_m1:16)
    v_add_u32 v[v_tmp], 16, v[v_cur_k]
    s_waitcnt lgkmcnt(1)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+8], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 17, s[s_wei_stride_k]   ; i_m:17(i_m0:0,i_m1:17)
    v_add_u32 v[v_tmp], 17, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+9], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 18, s[s_wei_stride_k]   ; i_m:18(i_m0:0,i_m1:18)
    v_add_u32 v[v_tmp], 18, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+10], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 19, s[s_wei_stride_k]   ; i_m:19(i_m0:0,i_m1:19)
    v_add_u32 v[v_tmp], 19, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+11], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 24, s[s_wei_stride_k]   ; i_m:24(i_m0:0,i_m1:24)
    v_add_u32 v[v_tmp], 24, v[v_cur_k]
    s_waitcnt lgkmcnt(0)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+12], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 25, s[s_wei_stride_k]   ; i_m:25(i_m0:0,i_m1:25)
    v_add_u32 v[v_tmp], 25, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+13], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 26, s[s_wei_stride_k]   ; i_m:26(i_m0:0,i_m1:26)
    v_add_u32 v[v_tmp], 26, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+14], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 27, s[s_wei_stride_k]   ; i_m:27(i_m0:0,i_m1:27)
    v_add_u32 v[v_tmp], 27, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+15], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b64 exec, -1
    ; start group 1, i_g_mr:1, i_g_ms:0, i_g_mw:0, i_g_mb:0, i_g_mt:0, m index start from 32
    s_barrier
    ds_write_b128 v[v_co_sst], v[a_c+16:a_c+16+3]    ; idword:0(0,0),  0x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+20:a_c+20+3] offset:4096   ; idword:256(2,0),  2x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+24:a_c+24+3] offset:8192   ; idword:512(4,0),  4x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b128 v[v_co_sst], v[a_c+28:a_c+28+3] offset:12288   ; idword:768(6,0),  6x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    s_mul_i32 s[s_tmp], 32, s[s_wei_stride_k]   ; i_m:32(i_m0:0,i_m1:32)
    v_add_u32 v[v_tmp], 32, v[v_cur_k]
    s_waitcnt lgkmcnt(0)
    s_barrier
    ;   load from lds, i_ssgroup:0, num_sld_per_ssgroup:4
    ds_read_b128 v[v_c:v_c+3], v[v_co_sld] 
    ds_read_b128 v[v_c+4:v_c+4+3], v[v_co_sld] offset:4096
    ds_read_b128 v[v_c+8:v_c+8+3], v[v_co_sld] offset:8192
    ds_read_b128 v[v_c+12:v_c+12+3], v[v_co_sld] offset:12288
    v_cmpx_eq_u32 vcc, 1, v[v_wei_c_flag]
    ;   store to global, m index start from 32, m0:0, m1:32
    s_waitcnt lgkmcnt(3)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 33, s[s_wei_stride_k]   ; i_m:33(i_m0:0,i_m1:33)
    v_add_u32 v[v_tmp], 33, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 34, s[s_wei_stride_k]   ; i_m:34(i_m0:0,i_m1:34)
    v_add_u32 v[v_tmp], 34, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+2], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 35, s[s_wei_stride_k]   ; i_m:35(i_m0:0,i_m1:35)
    v_add_u32 v[v_tmp], 35, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 40, s[s_wei_stride_k]   ; i_m:40(i_m0:0,i_m1:40)
    v_add_u32 v[v_tmp], 40, v[v_cur_k]
    s_waitcnt lgkmcnt(2)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+4], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 41, s[s_wei_stride_k]   ; i_m:41(i_m0:0,i_m1:41)
    v_add_u32 v[v_tmp], 41, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+5], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 42, s[s_wei_stride_k]   ; i_m:42(i_m0:0,i_m1:42)
    v_add_u32 v[v_tmp], 42, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+6], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 43, s[s_wei_stride_k]   ; i_m:43(i_m0:0,i_m1:43)
    v_add_u32 v[v_tmp], 43, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+7], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 48, s[s_wei_stride_k]   ; i_m:48(i_m0:0,i_m1:48)
    v_add_u32 v[v_tmp], 48, v[v_cur_k]
    s_waitcnt lgkmcnt(1)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+8], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 49, s[s_wei_stride_k]   ; i_m:49(i_m0:0,i_m1:49)
    v_add_u32 v[v_tmp], 49, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+9], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 50, s[s_wei_stride_k]   ; i_m:50(i_m0:0,i_m1:50)
    v_add_u32 v[v_tmp], 50, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+10], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 51, s[s_wei_stride_k]   ; i_m:51(i_m0:0,i_m1:51)
    v_add_u32 v[v_tmp], 51, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+11], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 56, s[s_wei_stride_k]   ; i_m:56(i_m0:0,i_m1:56)
    v_add_u32 v[v_tmp], 56, v[v_cur_k]
    s_waitcnt lgkmcnt(0)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+12], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 57, s[s_wei_stride_k]   ; i_m:57(i_m0:0,i_m1:57)
    v_add_u32 v[v_tmp], 57, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+13], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 58, s[s_wei_stride_k]   ; i_m:58(i_m0:0,i_m1:58)
    v_add_u32 v[v_tmp], 58, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+14], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 59, s[s_wei_stride_k]   ; i_m:59(i_m0:0,i_m1:59)
    v_add_u32 v[v_tmp], 59, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_atomic_add_f32 v[v_c+15], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b64 exec, -1

L_igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs_out:
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 88
    .amdhsa_next_free_sgpr 78
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_tg_split 0
    .amdhsa_accum_offset 56
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs
    .symbol: igemm_wrw_gtcx2_nhwc_bf16_bx0_ex0_bt64x128x16_wt32x32x8_ws1x1_wr2x1_ta1x4x1x1_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs.kd
    .sgpr_count: 84
    .vgpr_count: 88
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .group_segment_fixed_size: 16384
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
    - { .name: gemm_k_global_split, .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
