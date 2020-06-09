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

.macro .v_clear_nc vid, num
    _v = \vid
    .rept \num
        v_mov_b32 v[_v], 0
        _v = _v + 1
    .endr
.endm

.macro .v_fma_4x4_s8 c, a, b
    v_mac_f32 v[\c], v[\a], v[\b]
    v_mac_f32 v[\c+1], v[\a], v[\b+1]
    v_mac_f32 v[\c+2], v[\a], v[\b+2]
    v_mac_f32 v[\c+3], v[\a], v[\b+3]
    v_mac_f32 v[\c+8], v[\a+1], v[\b]
    v_mac_f32 v[\c+9], v[\a+1], v[\b+1]
    v_mac_f32 v[\c+10], v[\a+1], v[\b+2]
    v_mac_f32 v[\c+11], v[\a+1], v[\b+3]
    v_mac_f32 v[\c+16], v[\a+2], v[\b]
    v_mac_f32 v[\c+17], v[\a+2], v[\b+1]
    v_mac_f32 v[\c+18], v[\a+2], v[\b+2]
    v_mac_f32 v[\c+19], v[\a+2], v[\b+3]
    v_mac_f32 v[\c+24], v[\a+3], v[\b]
    v_mac_f32 v[\c+25], v[\a+3], v[\b+1]
    v_mac_f32 v[\c+26], v[\a+3], v[\b+2]
    v_mac_f32 v[\c+27], v[\a+3], v[\b+3]
.endm

; update v_out_flag for output
.macro .v_out_set_flag v_out_flag, v_out_iho, v_out_iwo, s_ho, s_wo, s_tmp2
    ;   flag: 0<= * <wo
    v_cmp_gt_u32 vcc, s[\s_ho], v[\v_out_iho]
    v_cndmask_b32 v[\v_out_flag], 0, 1, vcc
    ;   flag: 0<= * <wo
    v_cmp_gt_u32 vcc, s[\s_wo], v[\v_out_iwo]
    v_cndmask_b32 v[\v_out_flag], 0, v[\v_out_flag], vcc
.endm

; update v_in_flag
.macro .v_in_set_flag v_in_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp2
    ;   flag: 0<= * <wi
    v_cmp_gt_u32 vcc, s[\s_hi], v[\v_in_ihi]
    v_cndmask_b32 v[\v_in_flag], 0, 1, vcc
    ;   flag: 0<= * <wi
    v_cmp_gt_u32 vcc, s[\s_wi], v[\v_in_iwi]
    v_cndmask_b32 v[\v_in_flag], 0, v[\v_in_flag], vcc
.endm

.macro .v_out_calculate_os v_out_os, v_out_in, v_out_ik, v_out_iho, v_out_iwo, s_out_stride_n, s_out_stride_k, s_out_stride_ho, v_tmp2
    v_lshlrev_b32 v[\v_tmp2+1], 2, v[\v_out_iwo]
    v_mad_u32_u24 v[\v_out_os], s[\s_out_stride_ho], v[\v_out_iho], v[\v_tmp2+1]
    v_add3_u32 v[\v_out_os], v[\v_out_os], v[\v_out_in], v[\v_out_ik]
.endm

.macro .v_out_move_step_k_1 v_out_ik, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_out_stride_k, s_dslice_y, s_dslice_x, s_dtile_dy, s_dtile_dx, v_tmp2, s_tmp2
    v_add_i32 v[\v_out_dslice_ix], -1, v[\v_out_dslice_ix]

    v_cmpx_ge_i32 vcc, s[\s_dslice_x], v[\v_out_dslice_ix]
    v_mov_b32 v[\v_out_dslice_ix], 0
    v_add_i32 v[\v_out_dslice_iy], -1, v[\v_out_dslice_iy]
    s_mov_b64 exec, -1

    v_cmpx_ge_i32 vcc, s[\s_dslice_y], v[\v_out_dslice_iy]
    v_mov_b32 v[\v_out_dslice_iy], 0
    v_add_u32 v[\v_out_ik], s[\s_out_stride_k], v[\v_out_ik]    ; acc stride_k into ik
    s_mov_b64 exec, -1

    ; update ho, wo
    v_mad_i32_i24 v[\v_out_iho], s[\s_dtile_dy], v[\v_out_dslice_iy], v[\v_out_dslice_ih]
    v_mad_i32_i24 v[\v_out_iwo], s[\s_dtile_dx], v[\v_out_dslice_ix], v[\v_out_dslice_iw]
.endm

.macro .v_out_move_slice_window_v4r1_bwd v_out_ik, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_out_stride_k, s_move_slice_ik_out, s_move_slice_idslice_y, s_move_slice_idslice_x, s_dslice_y, s_dslice_x, s_dtile_dy, s_dtile_dx, v_tmp2, s_tmp2
    v_sub_i32 v[\v_out_dslice_ix], v[\v_out_dslice_ix], s[\s_move_slice_idslice_x]

    v_cmpx_ge_i32 vcc, s[\s_dslice_x], v[\v_out_dslice_ix]
    v_sub_i32 v[\v_out_dslice_ix], v[\v_out_dslice_ix], s[\s_dslice_x]
    v_add_i32 v[\v_out_dslice_iy], -1, v[\v_out_dslice_iy]
    s_mov_b64 exec, -1

    v_sub_i32 v[\v_out_dslice_iy], v[\v_out_dslice_iy],  s[\s_move_slice_idslice_y]
    v_cmpx_ge_i32 vcc, s[\s_dslice_y], v[\v_out_dslice_iy]
    
    v_sub_i32 v[\v_out_dslice_iy], v[\v_out_dslice_iy], s[\s_dslice_y]
    v_add_u32 v[\v_out_ik], s[\s_out_stride_k], v[\v_out_ik]
    s_mov_b64 exec, -1

    v_add_u32 v[\v_out_ik], s[\s_move_slice_ik_out], v[\v_out_ik]

    ; update ho, wo
    v_mad_i32_i24 v[\v_out_iho], s[\s_dtile_dy], v[\v_out_dslice_iy], v[\v_out_dslice_ih]
    v_mad_i32_i24 v[\v_out_iwo], s[\s_dtile_dx], v[\v_out_dslice_ix], v[\v_out_dslice_iw]
.endm

.macro .v_out_load_k_n_8_1 v_dst, s_p_out, v_out_os, v_out_flag, v_out_ik, v_out_in, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dslice_y, s_dslice_x, s_dtile_dy, s_dtile_dx, s_out_stride_n, s_out_stride_k, s_out_stride_ho, s_ho, s_wo, v_tmp2, s_tmp2
    .v_clear_nc \v_dst, 8
    .itr_out_k_idx = 0
    .rept 8
        v_cmpx_eq_u32 vcc, 1, v[\v_out_flag]
        buffer_load_dword v[\v_dst + .itr_out_k_idx], v[\v_out_os], s[\s_p_out:\s_p_out+3], 0 offen
        s_mov_b64 exec, -1
        .if .itr_out_k_idx != 7
            .v_out_move_step_k_1 \v_out_ik, \v_out_iho, \v_out_iwo, \v_out_dslice_ih, \v_out_dslice_iw, \v_out_dslice_iy, \v_out_dslice_ix, \s_out_stride_k, \s_dslice_y, \s_dslice_x, \s_dtile_dy, \s_dtile_dx, \v_tmp2, \s_tmp2
            .v_out_calculate_os \v_out_os, \v_out_in, \v_out_ik, \v_out_iho, \v_out_iwo, \s_out_stride_n, \s_out_stride_k, \s_out_stride_ho, \v_tmp2
            .v_out_set_flag \v_out_flag, \v_out_iho, \v_out_iwo, \s_ho, \s_wo, \s_tmp2
        .endif
        .itr_out_k_idx = .itr_out_k_idx + 1
    .endr
.endm

.macro .v_wei_calculate_os v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, s_wei_stride_k, s_wei_stride_c, s_wei_stride_y, v_tmp2
    v_lshlrev_b32 v[\v_tmp2+1], 2, v[\v_wei_ix]
    v_mad_u32_u24 v[\v_tmp2], v[\v_wei_iy], s[\s_wei_stride_y], v[\v_tmp2+1]
    v_add3_u32 v[\v_wei_os], v[\v_wei_ik], v[\v_tmp2], v[\v_wei_ic]
.endm

.macro .v_wei_move_step_k_1 v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_wei_stride_k, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, v_tmp2, s_tmp2
    v_add_u32 v[\v_wei_dslice_ix], v[\v_wei_dslice_ix], 1

    v_cmpx_le_u32 vcc, s[\s_dslice_x], v[\v_wei_dslice_ix]
    v_mov_b32 v[\v_wei_dslice_ix], 0
    v_add_u32 v[\v_wei_dslice_iy], v[\v_wei_dslice_iy], 1
    s_mov_b64 exec, -1

    v_cmpx_le_u32 vcc, s[\s_dslice_y], v[\v_wei_dslice_iy]
    v_mov_b32 v[\v_wei_dslice_iy], 0
    v_add_u32 v[\v_wei_ik], s[\s_wei_stride_k], v[\v_wei_ik]
    s_mov_b64 exec, -1

    v_mad_u32_u24 v[\v_wei_iy], v[\v_wei_dslice_iy], s[\s_dtile_y], v[\v_dtile_iy]
    v_mad_u32_u24 v[\v_wei_ix], v[\v_wei_dslice_ix], s[\s_dtile_x], v[\v_dtile_ix]
.endm


.macro .v_wei_move_slice_window_v4r1_bwd v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_wei_stride_k, s_move_slice_ik_wei, s_move_slice_idslice_y, s_move_slice_idslice_x, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, v_tmp2, s_tmp2
    v_add_u32 v[\v_wei_dslice_ix], s[\s_move_slice_idslice_x] , v[\v_wei_dslice_ix]

    v_cmpx_le_u32 vcc, s[\s_dslice_x], v[\v_wei_dslice_ix]
    v_subrev_u32 v[\v_wei_dslice_ix], s[\s_dslice_x], v[\v_wei_dslice_ix]
    v_add_u32 v[\v_wei_dslice_iy], 1 , v[\v_wei_dslice_iy]
    s_mov_b64 exec, -1

    v_add_u32 v[\v_wei_dslice_iy], s[\s_move_slice_idslice_y] , v[\v_wei_dslice_iy]
    v_cmpx_le_u32 vcc, s[\s_dslice_y], v[\v_wei_dslice_iy]
    v_subrev_u32 v[\v_wei_dslice_iy], s[\s_dslice_y] , v[\v_wei_dslice_iy]
    v_add_u32 v[\v_wei_ik], s[\s_wei_stride_k], v[\v_wei_ik]
    s_mov_b64 exec, -1

    v_add_u32 v[\v_wei_ik], s[\s_move_slice_ik_wei] , v[\v_wei_ik]

    v_mad_u32_u24 v[\v_wei_iy], v[\v_wei_dslice_iy], s[\s_dtile_y], v[\v_dtile_iy]
    v_mad_u32_u24 v[\v_wei_ix], v[\v_wei_dslice_ix], s[\s_dtile_x], v[\v_dtile_ix]
.endm

.macro .v_wei_load_k_n_8_1 v_dst, s_p_wei, v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, s_wei_stride_c, s_wei_stride_k, s_wei_stride_y, v_tmp2, s_tmp2
    .itr_wei_k_idx = 0
    .rept 8
        buffer_load_dword v[\v_dst + (.itr_wei_k_idx)], v[\v_wei_os], s[\s_p_wei:\s_p_wei+3], 0 offen
        .if .itr_wei_k_idx != 7
            .v_wei_move_step_k_1 \v_wei_ik, \v_wei_ic, \v_wei_iy, \v_wei_ix, \v_wei_dslice_iy, \v_wei_dslice_ix, \v_dtile_iy, \v_dtile_ix, \s_wei_stride_k, \s_dslice_y, \s_dslice_x, \s_dtile_y, \s_dtile_x, \v_tmp2, \s_tmp2
            .v_wei_calculate_os \v_wei_os, \v_wei_ik, \v_wei_ic, \v_wei_iy, \v_wei_ix, \s_wei_stride_k, \s_wei_stride_c, \s_wei_stride_y, \v_tmp2
        .endif
        .itr_wei_k_idx = .itr_wei_k_idx + 1
    .endr
.endm

.macro .v_in_calculate_os v_in_os, v_in_in, v_in_ic, v_in_ihi, v_in_iwi, s_in_stride_n, s_in_stride_c, s_in_stride_hi, v_tmp2
    v_lshlrev_b32 v[\v_tmp2+1], 2, v[\v_in_iwi]
    v_mad_u32_u24 v[\v_tmp2], v[\v_in_ihi], s[\s_in_stride_hi], v[\v_tmp2+1]
    v_add3_u32 v[\v_in_os], v[\v_tmp2], v[\v_in_ic], v[\v_in_in]
.endm

; n -> n*dslice_h*dslice_w, dslice_h,dslice_y -> hip,  dslice_w,dslicw_x -> wip
.macro .v_in_transform_gemm_n v_in_in, v_in_ihi, v_in_iwi, v_in_gemm_in, v_in_dslice_h, v_in_dslice_w, s_stride_dslice_hw, s_dslice_w, s_dtile_iy, s_dtile_ix, s_dslice_h_left, s_dslice_w_left, s_dilation_h, s_dilation_w, s_stride_h, s_stride_w, s_pad_h, s_pad_w, v_tmp4, s_tmp4
    ; n -> n*dslice_h*dslice_w
    .v_u32_div_vs \v_in_in, \v_in_gemm_in, \s_stride_dslice_hw, \v_tmp4, \s_tmp4
    v_mul_lo_u32 v[\v_tmp4+1], s[\s_stride_dslice_hw], v[\v_in_in]
    v_sub_u32 v[\v_in_gemm_in], v[\v_in_gemm_in], v[\v_tmp4+1]
    .v_u32_div_vs \v_in_dslice_h, \v_in_gemm_in, \s_dslice_w, \v_tmp4, \s_tmp4
    v_mul_lo_u32 v[\v_tmp4+1], s[\s_dslice_w], v[\v_in_dslice_h]
    v_sub_u32 v[\v_in_dslice_w], v[\v_in_gemm_in], v[\v_tmp4+1]

    ;
    v_add_u32 v[\v_in_dslice_h], s[\s_dslice_h_left], v[\v_in_dslice_h]
    v_add_u32 v[\v_in_dslice_w], s[\s_dslice_w_left], v[\v_in_dslice_w]

    ; dslice_h,dslice_y -> hip,  dslice_w,dslicw_x -> wip
    s_mul_i32 s[\s_tmp4], s[\s_dtile_iy], s[\s_dilation_h]
    v_mul_lo_u32 v[\v_tmp4], s[\s_stride_h], v[\v_in_dslice_h]
    v_add_u32 v[\v_tmp4], s[\s_tmp4], v[\v_tmp4]
    s_mul_i32 s[\s_tmp4+1], s[\s_dtile_ix], s[\s_dilation_w]
    v_mul_lo_u32 v[\v_tmp4+1], s[\s_stride_w], v[\v_in_dslice_w]
    v_add_u32 v[\v_tmp4+1], s[\s_tmp4+1], v[\v_tmp4+1]
    ; v_tmp4: hip, v_tmp4+1: wip

    ; hip->h, wip->w
    v_sub_i32 v[\v_in_ihi], v[\v_tmp4], s[\s_pad_h]
    v_sub_i32 v[\v_in_iwi], v[\v_tmp4+1], s[\s_pad_w]
.endm

.macro .v_in_move_step_n1 v_in_in, v_in_ihi, v_in_iwi, v_in_dslice_h, v_in_dslice_w, v_dtile_iy_x_dilation_h, v_dtile_ix_x_dilation_w, s_in_stride_n, s_dslice_h_left, s_dslice_w_left, s_dslice_h_shifted, s_dslice_w_shifted, s_dtile_iy, s_dtile_ix, s_dilation_h, s_dilation_w, s_stride_h, s_stride_w, s_pad_h, s_pad_w, v_tmp2, s_tmp2
    ; n -> n*dslice_h*dslice_w
    v_add_u32 v[\v_in_dslice_w], 1, v[\v_in_dslice_w]

    v_cmpx_le_u32 vcc, s[\s_dslice_w_shifted], v[\v_in_dslice_w]
    v_mov_b32 v[\v_in_dslice_w],  s[\s_dslice_w_left]
    v_add_u32 v[\v_in_dslice_h], 1, v[\v_in_dslice_h]
    s_mov_b64 exec, -1

    v_cmpx_le_u32 vcc, s[\s_dslice_h_shifted], v[\v_in_dslice_h]
    v_mov_b32 v[\v_in_dslice_h], s[\s_dslice_h_left]
    v_add_u32 v[\v_in_in], s[\s_in_stride_n], v[\v_in_in]
    s_mov_b64 exec, -1

    ; dslice_h,dslice_y -> hip,  dslice_w,dslicw_x -> wip
    v_mad_u32_u24 v[\v_tmp2], s[\s_stride_h], v[\v_in_dslice_h], v[\v_dtile_iy_x_dilation_h]
    v_mad_u32_u24 v[\v_tmp2+1], s[\s_stride_w], v[\v_in_dslice_w], v[\v_dtile_ix_x_dilation_w]
    ; v_tmp2: hip, v_tmp2+1: wip

    ; hip->h, wip->w
    v_sub_i32 v[\v_in_ihi], v[\v_tmp2],     s[\s_pad_h]
    v_sub_i32 v[\v_in_iwi], v[\v_tmp2+1],   s[\s_pad_w]
.endm

.macro .v_in_move_slice_window v_in_in, v_in_ihi, v_in_iwi, v_in_dslice_h, v_in_dslice_w, v_dtile_iy_x_dilation_h, v_dtile_ix_x_dilation_w, s_in_stride_n, s_move_slice_in_in, s_move_slice_in_dslice_h, s_move_slice_in_dslice_w, s_dslice_h_shifted, s_dslice_w_shifted, s_dslice_h, s_dslice_w, s_dtile_iy, s_dtile_ix, s_dilation_h, s_dilation_w, s_stride_h, s_stride_w, s_pad_h, s_pad_w, v_tmp2, s_tmp2
    ; n -> n*dslice_h*dslice_w
    v_add_u32 v[\v_in_dslice_w], s[\s_move_slice_in_dslice_w], v[\v_in_dslice_w]

    v_cmpx_le_u32 vcc, s[\s_dslice_w_shifted], v[\v_in_dslice_w]
    v_subrev_u32 v[\v_in_dslice_w], s[\s_dslice_w], v[\v_in_dslice_w]
    v_add_u32 v[\v_in_dslice_h], 1, v[\v_in_dslice_h]
    s_mov_b64 exec, -1

    v_add_u32 v[\v_in_dslice_h], s[\s_move_slice_in_dslice_h], v[\v_in_dslice_h]
    v_cmpx_le_u32 vcc, s[\s_dslice_h_shifted], v[\v_in_dslice_h]
    v_subrev_u32 v[\v_in_dslice_h], s[\s_dslice_h], v[\v_in_dslice_h]
    v_add_u32 v[\v_in_in], s[\s_in_stride_n], v[\v_in_in]
    s_mov_b64 exec, -1

    v_add_u32 v[\v_in_in], s[\s_move_slice_in_in], v[\v_in_in]

    ; dslice_h,dslice_y -> hip,  dslice_w,dslicw_x -> wip
    v_mad_u32_u24 v[\v_tmp2], s[\s_stride_h], v[\v_in_dslice_h], v[\v_dtile_iy_x_dilation_h]
    v_mad_u32_u24 v[\v_tmp2+1], s[\s_stride_w], v[\v_in_dslice_w], v[\v_dtile_ix_x_dilation_w]
    ; v_tmp2: hip, v_tmp2+1: wip

    ; hip->h, wip->w
    v_sub_i32 v[\v_in_ihi], v[\v_tmp2],     s[\s_pad_h]
    v_sub_i32 v[\v_in_iwi], v[\v_tmp2+1],   s[\s_pad_w]
.endm

.macro .v_in_write_m0_m1_n0_n1_step v_src, s_p_in, v_in_os, v_in_flag, v_in_ihi_itr, v_in_iwi_itr, v_in_in_itr, v_in_dslice_h_itr, v_in_dslice_w_itr, v_in_ic_itr, v_in_in, v_in_ic, v_in_ihi, v_in_iwi, v_in_dslice_h, v_in_dslice_w, v_dtile_iy_x_dilation_h, v_dtile_ix_x_dilation_w, s_move_slice_in_in, s_move_slice_in_dslice_h, s_move_slice_in_dslice_w, s_dslice_h_left, s_dslice_w_left, s_dslice_h_shifted, s_dslice_w_shifted, s_dslice_h, s_dslice_w, s_dtile_iy, s_dtile_ix, s_dilation_h, s_dilation_w, s_stride_h, s_stride_w, s_pad_h, s_pad_w, s_in_stride_n, s_in_stride_c, s_in_stride_c_m0, s_in_stride_hi, s_hi, s_wi, v_tmp2, s_tmp2, k_m0, k_m1, k_n0, k_n1
    v_mov_b32 v[\v_in_in_itr], v[\v_in_in]
    v_mov_b32 v[\v_in_dslice_h_itr], v[\v_in_dslice_h]
    v_mov_b32 v[\v_in_dslice_w_itr], v[\v_in_dslice_w]
    v_mov_b32 v[\v_in_ic_itr], v[\v_in_ic]
    v_mov_b32 v[\v_in_ihi_itr], v[\v_in_ihi]
    v_mov_b32 v[\v_in_iwi_itr], v[\v_in_iwi]
    .itr_m0 = 0
    .rept \k_m0

    .itr_m1 = 0
    .rept \k_m1

    .itr_n0 = 0
    .rept \k_n0

        .itr_n1 = 0
        .rept \k_n1
            v_cmpx_eq_u32 vcc, 1, v[\v_in_flag]
            buffer_store_dword v[\v_src + (.itr_m0 * 32 + .itr_m1 * 8 + .itr_n0 * 4 + .itr_n1) ], v[\v_in_os], s[\s_p_in:\s_p_in+3], 0 offen
            s_mov_b64 exec, -1

            .if .itr_n1 != \k_n1 - 1
                .v_in_move_step_n1 \v_in_in_itr, \v_in_ihi_itr, \v_in_iwi_itr, \v_in_dslice_h_itr, \v_in_dslice_w_itr, \v_dtile_iy_x_dilation_h, \v_dtile_ix_x_dilation_w, \s_in_stride_n, \s_dslice_h_left, \s_dslice_w_left, \s_dslice_h_shifted, \s_dslice_w_shifted, \s_dtile_iy, \s_dtile_ix, \s_dilation_h, \s_dilation_w, \s_stride_h, \s_stride_w, \s_pad_h, \s_pad_w, \v_tmp2, \s_tmp2
                .v_in_calculate_os \v_in_os, \v_in_in_itr, \v_in_ic_itr, \v_in_ihi_itr, \v_in_iwi_itr, \s_in_stride_n, \s_in_stride_c, \s_in_stride_hi, \v_tmp2
                .v_in_set_flag \v_in_flag, \v_in_ihi_itr, \v_in_iwi_itr, \s_hi, \s_wi, \s_tmp2
            .endif
            .itr_n1 = .itr_n1 + 1
        .endr   ; n1
        .if .itr_n0 != \k_n0 - 1
            v_mov_b32 v[\v_in_in_itr], v[\v_in_in]
            v_mov_b32 v[\v_in_dslice_h_itr], v[\v_in_dslice_h]
            v_mov_b32 v[\v_in_dslice_w_itr], v[\v_in_dslice_w]

            .v_in_move_slice_window \v_in_in_itr, \v_in_ihi_itr, \v_in_iwi_itr, \v_in_dslice_h_itr, \v_in_dslice_w_itr, \v_dtile_iy_x_dilation_h, \v_dtile_ix_x_dilation_w, \s_in_stride_n, \s_move_slice_in_in, \s_move_slice_in_dslice_h, \s_move_slice_in_dslice_w, \s_dslice_h_shifted, \s_dslice_w_shifted, \s_dslice_h, \s_dslice_w, \s_dtile_iy, \s_dtile_ix, \s_dilation_h, \s_dilation_w, \s_stride_h, \s_stride_w, \s_pad_h, \s_pad_w, \v_tmp2, \s_tmp2
            .v_in_calculate_os \v_in_os, \v_in_in_itr, \v_in_ic_itr, \v_in_ihi_itr, \v_in_iwi_itr, \s_in_stride_n, \s_in_stride_c, \s_in_stride_hi, \v_tmp2
            .v_in_set_flag \v_in_flag, \v_in_ihi_itr, \v_in_iwi_itr, \s_hi, \s_wi, \s_tmp2
        .endif
        .itr_n0 = .itr_n0 + 1
    .endr   ; n0

    .if .itr_m1 != \k_m1 - 1
        v_mov_b32 v[\v_in_in_itr], v[\v_in_in]
        v_mov_b32 v[\v_in_dslice_h_itr], v[\v_in_dslice_h]
        v_mov_b32 v[\v_in_dslice_w_itr], v[\v_in_dslice_w]
        v_mov_b32 v[\v_in_ihi_itr], v[\v_in_ihi]
        v_mov_b32 v[\v_in_iwi_itr], v[\v_in_iwi]
        v_add_u32 v[\v_in_ic_itr], s[\s_in_stride_c], v[\v_in_ic_itr]
        .v_in_calculate_os \v_in_os, \v_in_in_itr, \v_in_ic_itr, \v_in_ihi_itr, \v_in_iwi_itr, \s_in_stride_n, \s_in_stride_c, \s_in_stride_hi, \v_tmp2
        .v_in_set_flag \v_in_flag, \v_in_ihi_itr, \v_in_iwi_itr, \s_hi, \s_wi, \s_tmp2
    .endif
    .itr_m1 = .itr_m1 + 1
    .endr   ; m1

    .if .itr_m0 != \k_m0 - 1
        v_mov_b32 v[\v_in_in_itr], v[\v_in_in]
        v_mov_b32 v[\v_in_dslice_h_itr], v[\v_in_dslice_h]
        v_mov_b32 v[\v_in_dslice_w_itr], v[\v_in_dslice_w]
        v_mov_b32 v[\v_in_ihi_itr], v[\v_in_ihi]
        v_mov_b32 v[\v_in_iwi_itr], v[\v_in_iwi]
        v_add_u32 v[\v_in_ic_itr], s[\s_in_stride_c_m0], v[\v_in_ic]
        .v_in_calculate_os \v_in_os, \v_in_in_itr, \v_in_ic_itr, \v_in_ihi_itr, \v_in_iwi_itr, \s_in_stride_n, \s_in_stride_c, \s_in_stride_hi, \v_tmp2
        .v_in_set_flag \v_in_flag, \v_in_ihi_itr, \v_in_iwi_itr, \s_hi, \s_wi, \s_tmp2
    .endif
    .itr_m0 = .itr_m0 + 1
    .endr   ; m0
.endm


; store input to LDS. {k, n}:{8, 1}
.macro .v_out_sst_k_n_8_1 v_src, v_sst_os
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+0], v[\v_src+1]  offset0:0  offset1:2
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+2], v[\v_src+3]  offset0:4  offset1:6
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+4], v[\v_src+5]  offset0:8  offset1:10
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+6], v[\v_src+7]  offset0:12  offset1:14
.endm

; store weight to LDS. {k, n}:{8, 1}
.macro .v_wei_sst_k_n_8_1 v_src, v_sst_os
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+0], v[\v_src+1]  offset0:0  offset1:2
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+2], v[\v_src+3]  offset0:4  offset1:6
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+4], v[\v_src+5]  offset0:8  offset1:10
    ds_write2st64_b32 v[\v_sst_os], v[\v_src+6], v[\v_src+7]  offset0:12  offset1:14
.endm

;----------------------------------------------------------
; starting of kernel igemm_v4r1_bwd_dynamic
; block_size                       : 256
; thread_tile                      : 8x8
; 
; GemmMPerBlock                     : 128
; GemmNPerBlock                     : 128
; GemmKPerBlock                     : 16
; GemmMPerThread                    : 4 (x2) 
; GemmNPerThread                    : 4 (x2) 
; GemmKPerThread,                   : 1
; GemmMLevel0Cluster,               ; 4
; GemmNLevel0Cluster,               ; 4
; GemmMLevel1Cluster,               ; 4
; GemmNLevel1Cluster,               ; 4
; GemmThreadGemmDataPerReadM,       ; 4
; GemmThreadGemmDataPerReadN,       ; 4
; GemmABlockCopyThreadSliceLengths_GemmK_GemmM      : 8, 1
; GemmABlockCopyThreadClusterLengths_GemmK_GemmM    : 2, 128
; GemmABlockCopySrcDataPerRead_GemmM                : 1
; GemmABlockCopyDstDataPerWrite_GemmM               : 1
; GemmBBlockCopyThreadSliceLengths_GemmK_GemmN      : 8, 1
; GemmBBlockCopyThreadClusterLengths_GemmK_GemmN    : 2, 128
; GemmBBlockCopySrcDataPerRead_GemmN                : 1
; GemmBBlockCopyDstDataPerWrite_GemmN               : 1
; GemmCThreadCopyDstDataPerWrite_GemmN1             : 1
; kernarg offset
.set k_p_in,                    0
.set k_p_wei,                   8
.set k_p_out,                   16
.set k_hi,                      24
.set k_wi,                      28
.set k_n,                       32
.set k_k,                       36
.set k_c,                       40
.set k_ho,                      44
.set k_wo,                      48
.set k_stride_h,                52
.set k_stride_w,                56
.set k_dilation_h,              60
.set k_dilation_w,              64
.set k_pad_h,                   68
.set k_pad_w,                   72
.set k_y,                       76
.set k_x,                       80
.set k_dtile_iy,                84
.set k_dtile_ix,                88
.set k_dtile_dy,                92      ; ConvDilationH / GcdStrideDilationH
.set k_dtile_dx,                96      ; ConvDilationW / GcdStrideDilationW
.set k_dtile_y,                 100     ; ConvStrideH   / GcdStrideDilationH
.set k_dtile_x,                 104     ; ConvStrideW   / GcdStrideDilationW
.set k_dtile_h,                 108     ; Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
.set k_dtile_w,                 112     ; Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);
.set k_dslice_y,                116     ; YDotSlice
.set k_dslice_x,                120     ; XDotSlice
.set k_dslice_h,                124     ; HTildaSlice (iHTildaLeft->0)
.set k_dslice_w,                128     ; WTildaSlice (iWTildaLeft->0)
.set k_dslice_h_left,           132     ; HTildaLeft
.set k_dslice_w_left,           136     ; WTildaLeft
.set k_pack0,                   140
.set k_end,                     144

; sgpr
.set s_ka,                      0
.set s_bx,                      2
.set s_p_in,                    4
.set s_p_wei,                   8
.set s_p_out,                   12
.set s_hi,                      16
.set s_wi,                      17
.set s_n,                       18
.set s_k,                       19
.set s_c,                       20
.set s_ho,                      21
.set s_wo,                      22
.set s_stride_h,                23
.set s_stride_w,                24
.set s_dilation_h,              25
.set s_dilation_w,              26
.set s_pad_h,                   27
.set s_pad_w,                   28
.set s_y,                       29
.set s_x,                       30
.set s_dtile_iy,                31
.set s_dtile_ix,                32
.set s_dtile_dy,                33
.set s_dtile_dx,                34
.set s_dtile_y,                 35
.set s_dtile_x,                 36
.set s_dtile_h,                 37
.set s_dtile_w,                 38
.set s_dslice_y,                39
.set s_dslice_x,                40
.set s_dslice_y_neg,            2
.set s_dslice_x_neg,            3
.set s_dslice_h,                41
.set s_dslice_w,                42
.set s_dslice_h_left,           43
.set s_dslice_w_left,           44
.set s_dslice_h_shifted,        45
.set s_dslice_w_shifted,        46

.set s_out_stride_k,            47
.set s_out_stride_n,            48
.set s_out_stride_ho,           49
.set s_in_stride_c,             50
.set s_in_stride_n,             51
.set s_in_stride_hi,            52
.set s_wei_stride_c,            53
.set s_wei_stride_k,            54
.set s_wei_stride_y,            55
.set s_stride_dslice_hw,        56
.set s_stride_dslice_yx,        57
.set s_block_gemm_im,           58
.set s_block_gemm_in,           59
.set s_move_slice_ik,           60
.set s_move_slice_ik_wei,       61
.set s_move_slice_ik_out,       62
.set s_move_slice_idslice_y,    63
.set s_move_slice_idslice_x,    64
.set s_move_slice_in_in,        65
.set s_move_slice_in_dslice_h,  66
.set s_move_slice_in_dslice_w,  67
.set s_kitr,                    0
.set s_tmp,                     68
.set s_end,                     72

; vgpr
.set v_c,                       0
.set v_a,                       64
.set v_b,                       72
.set v_gld_a,                   80
.set v_gld_b,                   88
.set v_out_os,                  96
.set v_wei_os,                  97
.set v_sst_a_os,                98
.set v_sst_b_os,                99
.set v_sld_a_os,                100
.set v_sld_b_os,                101
.set v_out_flag,                102

.set v_out_gemm_ik,             63
.set v_out_gemm_in,             62
.set v_wei_gemm_ik,             61
.set v_wei_gemm_im,             60
.set v_gemm_in,                 59
.set v_gemm_im,                 58

.set v_out_ik,                  103
.set v_out_in,                  104     ; ! This is indeed the offset from in_n. only valid if each thread only load different K direction
.set v_out_iho,                 105
.set v_out_iwo,                 106
.set v_out_dslice_ih,           107
.set v_out_dslice_iw,           108
.set v_out_dslice_iy,           109
.set v_out_dslice_ix,           110

.set v_wei_ik,                  111
.set v_wei_ic,                  112
.set v_wei_iy,                  113
.set v_wei_ix,                  114
.set v_wei_dslice_iy,           115
.set v_wei_dslice_ix,           116

.set v_in_gemm_in0,             117
.set v_in_gemm_in1,             118
.set v_in_ic,                   119
.set v_in_in,                   v_gld_a+0
.set v_in_ihi,                  v_gld_a+1
.set v_in_iwi,                  v_gld_a+2
.set v_in_dslice_h,             v_gld_a+3
.set v_in_dslice_w,             v_gld_a+4
.set v_in_gemm_in,              v_gld_a+5

.set v_in_dslice_h_itr,         v_gld_a+6
.set v_in_dslice_w_itr,         v_gld_a+7
.set v_in_in_itr,               v_gld_a+8
.set v_in_ic_itr,               v_gld_a+9
.set v_in_os,                   v_gld_a+10
.set v_in_flag,                 v_gld_a+11
.set v_in_ihi_itr,              v_gld_a+12
.set v_in_iwi_itr,              v_gld_a+13
.set v_dtile_iy_x_dilation_h,   v_gld_a+14
.set v_dtile_ix_x_dilation_w,   v_gld_a+15
.set v_tmp,                     120
;.set v_tid,                     127
.set v_dtile_iy,                126
.set v_dtile_ix,                127
.set v_end,                     128
.set v_in_gemm_in0_itr,         v_in_dslice_h_itr
.set v_in_gemm_in1_itr,         v_in_dslice_w_itr

.text
.globl igemm_v4r1_bwd_dynamic
.p2align 8
.type igemm_v4r1_bwd_dynamic,@function
igemm_v4r1_bwd_dynamic:
    s_load_dwordx2  s[s_p_in:s_p_in+1],         s[s_ka:s_ka+1],     0+k_p_in
    s_load_dwordx2  s[s_p_wei:s_p_wei+1],       s[s_ka:s_ka+1],     0+k_p_wei
    s_load_dwordx2  s[s_p_out:s_p_out+1],       s[s_ka:s_ka+1],     0+k_p_out
    s_load_dwordx16 s[s_hi:s_hi+15],            s[s_ka:s_ka+1],     0+k_hi
    s_load_dwordx8  s[s_dtile_ix:s_dtile_ix+7], s[s_ka:s_ka+1],     0+k_dtile_ix
    s_load_dwordx4  s[s_dslice_x:s_dslice_x+3], s[s_ka:s_ka+1],     0+k_dslice_x
    s_load_dword    s[s_dslice_w_left],         s[s_ka:s_ka+1],     0+k_dslice_w_left

    ; GemmBBlockCopyThreadClusterLengths_GemmK_GemmN:{2,128}, slice:{8,1}
    v_and_b32 v[v_out_gemm_in], 127, v0
    v_lshrrev_b32 v[v_tmp], 7, v0
    v_lshlrev_b32 v[v_out_gemm_ik], 3, v[v_tmp]

    ; GemmABlockCopyThreadClusterLengths_GemmK_GemmM:{2,128}, slice:{8,1}
    v_and_b32 v[v_wei_gemm_im], 127, v0
    v_lshrrev_b32 v[v_tmp], 7, v0
    v_lshlrev_b32 v[v_wei_gemm_ik], 3, v[v_tmp]

    ; v_mov_b32 v[v_tid], v0

    s_mov_b32 s[s_p_in  + 2], 0xffffffff
    s_mov_b32 s[s_p_in  + 3], 0x27000
    s_mov_b32 s[s_p_wei + 2], 0xffffffff
    s_mov_b32 s[s_p_wei + 3], 0x27000
    s_mov_b32 s[s_p_out + 2], 0xffffffff
    s_mov_b32 s[s_p_out + 3], 0x27000
    s_waitcnt lgkmcnt(0)

    ; calculate index
    s_mul_i32 s[s_out_stride_k], s[s_ho], s[s_wo]
    s_mul_i32 s[s_out_stride_n], s[s_k], s[s_out_stride_k]
    s_mul_i32 s[s_in_stride_c], s[s_hi], s[s_wi]
    s_mul_i32 s[s_in_stride_n], s[s_c], s[s_in_stride_c]
    s_mul_i32 s[s_wei_stride_c], s[s_y], s[s_x]
    s_mul_i32 s[s_wei_stride_k], s[s_c], s[s_wei_stride_c]
    s_mul_i32 s[s_stride_dslice_hw], s[s_dslice_h], s[s_dslice_w]
    s_mul_i32 s[s_stride_dslice_yx], s[s_dslice_y], s[s_dslice_x]

    ; block gemm_m, gemm_n index on global
    s_mul_i32 s[s_tmp], s[s_stride_dslice_hw], s[s_n]
    s_lshr_b32 s[0], s[s_tmp], 7          ; gemm_n:128
    .v_u32_div_ss v_tmp+5, s_bx, 0, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_tmp], v[v_tmp+5]
    s_mul_i32 s[s_tmp+2], s[s_tmp], s[0]
    s_sub_i32 s[s_tmp+1], s[s_bx], s[s_tmp+2]
    s_lshl_b32 s[s_block_gemm_im], s[s_tmp], 7
    s_lshl_b32 s[s_block_gemm_in], s[s_tmp+1], 7

    ; calculate output transform
    ;   gemm_n -> n*dslice_h*dslice_w
    v_add_u32 v[v_tmp+4], s[s_block_gemm_in], v[v_out_gemm_in]
    .v_u32_div_vs v_out_in, v_tmp+4, s_stride_dslice_hw, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_stride_dslice_hw], v[v_out_in]
    v_sub_u32 v[v_tmp+4], v[v_tmp+4], v[v_tmp]
    .v_u32_div_vs v_out_dslice_ih, v_tmp+4, s_dslice_w, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_dslice_w], v[v_out_dslice_ih]
    v_sub_u32 v[v_out_dslice_iw], v[v_tmp+4], v[v_tmp]

    ; iHTildaLeft, iWTildaLeft
    v_add_u32 v[v_out_dslice_ih], s[s_dslice_h_left], v[v_out_dslice_ih]
    v_add_u32 v[v_out_dslice_iw], s[s_dslice_w_left], v[v_out_dslice_iw]

    ;   gemm_k -> k*dslice_y*dslice_x
    .v_u32_div_vs v_out_ik, v_out_gemm_ik, s_stride_dslice_yx, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_stride_dslice_yx], v[v_out_ik]
    v_sub_u32 v[v_tmp+4], v[v_out_gemm_ik], v[v_tmp]
    .v_u32_div_vs v_out_dslice_iy, v_tmp+4, s_dslice_x, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_dslice_x], v[v_out_dslice_iy]
    v_sub_u32 v[v_out_dslice_ix], v[v_tmp+4], v[v_tmp]

    ; dslice_y,dslice_h -> oh, dslice_x,dslice_w -> ow
    v_mul_lo_u32 v[v_tmp+1], s[s_dtile_dy], v[v_out_dslice_iy]
    v_sub_i32 v[v_out_iho], v[v_out_dslice_ih], v[v_tmp+1]
    v_mul_lo_u32 v[v_tmp+1], s[s_dtile_dx], v[v_out_dslice_ix]
    v_sub_i32 v[v_out_iwo], v[v_out_dslice_iw], v[v_tmp+1]

    s_mul_i32 s[s_dslice_x_neg], -1, s[s_dslice_x]
    s_mul_i32 s[s_dslice_y_neg], -1, s[s_dslice_y]

    v_mul_i32_i24 v[v_out_dslice_iy], -1, v[v_out_dslice_iy]
    v_mul_i32_i24 v[v_out_dslice_ix], -1, v[v_out_dslice_ix]

    ; update out flag
    .v_out_set_flag v_out_flag, v_out_iho, v_out_iwo, s_ho, s_wo, s_tmp

    s_lshl_b32 s[s_out_stride_n], s[s_out_stride_n], 2
    s_lshl_b32 s[s_out_stride_k], s[s_out_stride_k], 2
    s_lshl_b32 s[s_out_stride_ho], s[s_wo], 2
    v_mul_lo_u32 v[v_out_in], s[s_out_stride_n], v[v_out_in]
    v_mul_lo_u32 v[v_out_ik], s[s_out_stride_k], v[v_out_ik]
    .v_out_calculate_os v_out_os, v_out_in, v_out_ik, v_out_iho, v_out_iwo, s_out_stride_n, s_out_stride_k, s_out_stride_ho, v_tmp
    ; load output

    .v_out_load_k_n_8_1 v_gld_b, s_p_out, v_out_os, v_out_flag, v_out_ik, v_out_in, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dslice_y_neg, s_dslice_x_neg, s_dtile_dy, s_dtile_dx, s_out_stride_n, s_out_stride_k, s_out_stride_ho, s_ho, s_wo, v_tmp, s_tmp

    v_mov_b32 v[v_dtile_iy], s[s_dtile_iy]
    v_mov_b32 v[v_dtile_ix], s[s_dtile_ix]

    ; move slice window
    s_mov_b32 s[1], 9   ; unroll 16, but iterate 8 in each load
    .v_u32_div_ss v_tmp+4, 1, s_stride_dslice_yx, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_move_slice_ik], v[v_tmp+4]
    s_mul_i32 s[s_tmp], s[s_stride_dslice_yx], s[s_move_slice_ik]
    s_mul_i32 s[s_move_slice_ik_out], s[s_out_stride_k], s[s_move_slice_ik]
    s_sub_i32 s[1], s[1], s[s_tmp]
    .v_u32_div_ss v_tmp+4, 1, s_dslice_x, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_move_slice_idslice_y], v[v_tmp+4]
    s_mul_i32 s[s_tmp], s[s_dslice_x], s[s_move_slice_idslice_y]
    s_sub_i32 s[s_move_slice_idslice_x], s[1], s[s_tmp]

    .v_u32_div_vs v_wei_ik, v_wei_gemm_ik, s_stride_dslice_yx, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_stride_dslice_yx], v[v_wei_ik]
    v_sub_u32 v[v_tmp+4], v[v_wei_gemm_ik], v[v_tmp]
    .v_u32_div_vs v_wei_dslice_iy, v_tmp+4, s_dslice_x, v_tmp, s_tmp
    v_mul_lo_u32 v[v_tmp], s[s_dslice_x], v[v_wei_dslice_iy]
    v_sub_u32 v[v_wei_dslice_ix], v[v_tmp+4], v[v_tmp]

    ;  gemm_m-> wei_ic
    v_add_u32 v[v_wei_ic], s[s_block_gemm_im], v[v_wei_gemm_im]

    ; dslice_y -> y, dslice_x -> x
    v_mad_u32_u24 v[v_wei_iy], v[v_wei_dslice_iy], s[s_dtile_y], v[v_dtile_iy]
    v_mad_u32_u24 v[v_wei_ix], v[v_wei_dslice_ix], s[s_dtile_x], v[v_dtile_ix]

    ; calculate wei offset
    s_lshl_b32 s[s_wei_stride_c], s[s_wei_stride_c], 2
    s_lshl_b32 s[s_wei_stride_k], s[s_wei_stride_k], 2
    s_lshl_b32 s[s_wei_stride_y], s[s_x], 2
    s_mul_i32 s[s_move_slice_ik_wei], s[s_wei_stride_k], s[s_move_slice_ik]
    v_mul_lo_u32 v[v_wei_ic], s[s_wei_stride_c], v[v_wei_ic]
    v_mul_lo_u32 v[v_wei_ik], s[s_wei_stride_k], v[v_wei_ik]
    .v_wei_calculate_os v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, s_wei_stride_k, s_wei_stride_c, s_wei_stride_y, v_tmp
    .v_wei_load_k_n_8_1 v_gld_a, s_p_wei, v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, s_wei_stride_c, s_wei_stride_k, s_wei_stride_y, v_tmp, s_tmp

    ; c thread mapping
    v_and_b32 v[v_tmp+4], 15, v0
    v_and_b32 v[v_tmp], 3, v[v_tmp+4]
    v_lshrrev_b32 v[v_tmp+1], 2, v[v_tmp+4]

    v_lshrrev_b32 v[v_tmp+4], 4, v0
    v_and_b32 v[v_tmp+2], 3, v[v_tmp+4]
    v_lshrrev_b32 v[v_tmp+3], 2, v[v_tmp+4]

    v_lshl_or_b32 v[v_gemm_in], v[v_tmp+2], 2, v[v_tmp]               ; in
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp+3], 2, v[v_tmp+1]             ; im

    v_lshlrev_b32 v[v_sld_b_os], 4, v[v_gemm_in]
    v_lshlrev_b32 v[v_sld_a_os], 4, v[v_gemm_im]
    v_add_u32 v[v_sld_a_os], 8192, v[v_sld_a_os]

    ; calculate input index, m0, m1, n0, n1

    v_lshlrev_b32 v[v_tmp+1], 2, v[v_gemm_im]
    v_add_u32 v[v_in_ic], s[s_block_gemm_im], v[v_tmp+1]

    v_lshlrev_b32 v[v_tmp+1], 2, v[v_gemm_in]
    v_add_u32 v[v_tmp+5], s[s_block_gemm_in], v[v_tmp+1]
    v_lshrrev_b32 v[v_in_gemm_in0], 6, v[v_tmp+5]
    v_and_b32 v[v_in_gemm_in1], 63, v[v_tmp+5]

    s_lshl_b32 s[s_in_stride_hi], s[s_wi], 2
    s_lshl_b32 s[s_in_stride_c], s[s_in_stride_c], 2
    s_lshl_b32 s[s_in_stride_n], s[s_in_stride_n], 2

    s_add_u32 s[s_dslice_h_shifted], s[s_dslice_h], s[s_dslice_h_left]
    s_add_u32 s[s_dslice_w_shifted], s[s_dslice_w], s[s_dslice_w_left]

    ; input move n0
    s_mov_b32 s[0], 64                                          ; n0 -> 64
    .v_u32_div_ss v_tmp+4, 0, s_stride_dslice_hw, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_move_slice_in_in], v[v_tmp+4]
    s_mul_i32 s[s_tmp], s[s_stride_dslice_hw], s[s_move_slice_in_in]
    s_sub_i32 s[0], s[0], s[s_tmp]
    .v_u32_div_ss v_tmp+4, 0, s_dslice_w, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_move_slice_in_dslice_h], v[v_tmp+4]
    s_mul_i32 s[s_tmp], s[s_dslice_w], s[s_move_slice_in_dslice_h]
    s_sub_i32 s[s_move_slice_in_dslice_w], s[0], s[s_tmp]

    s_mul_i32 s[s_move_slice_in_in], s[s_in_stride_n], s[s_move_slice_in_in]

    ; out lds offset block k_n
    v_lshlrev_b32 v[v_tmp], 7, v[v_out_gemm_ik]
    v_or_b32 v[v_tmp+1], v[v_tmp], v[v_out_gemm_in]
    v_lshlrev_b32 v[v_sst_b_os], 2, v[v_tmp+1]

    ; wei lds offset block k_m
    v_lshlrev_b32 v[v_tmp], 7, v[v_wei_gemm_ik]
    v_or_b32 v[v_tmp+1], v[v_tmp], v[v_wei_gemm_im]
    v_lshlrev_b32 v[v_sst_a_os], 2, v[v_tmp+1]
    v_add_u32 v[v_sst_a_os], 8192, v[v_sst_a_os]

    .v_clear_nc v_c, 64

    ; start FMA loop, 8x8 thread tile with 4x4 sub-tile
    s_waitcnt vmcnt(8)
    .v_out_sst_k_n_8_1 v_gld_b, v_sst_b_os

    s_waitcnt vmcnt(0)
    .v_wei_sst_k_n_8_1 v_gld_a, v_sst_a_os


    ; gemm_k -> k*dslice_y*dslice_x
    s_mul_i32 s[s_tmp], s[s_stride_dslice_yx], s[s_k]
    s_sub_i32 s[s_kitr], s[s_tmp], 16
    s_cmp_gt_i32 s[s_kitr], 0


    s_cbranch_scc0 L_igemm_v4r1_bwd_dynamic_end
    .v_out_move_slice_window_v4r1_bwd v_out_ik, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_out_stride_k, s_move_slice_ik_out, s_move_slice_idslice_y, s_move_slice_idslice_x, s_dslice_y_neg, s_dslice_x_neg, s_dtile_dy, s_dtile_dx, v_tmp, s_tmp
    .v_out_calculate_os v_out_os, v_out_in, v_out_ik, v_out_iho, v_out_iwo, s_out_stride_n, s_out_stride_k, s_out_stride_ho, v_tmp
    .v_out_set_flag v_out_flag, v_out_iho, v_out_iwo, s_ho, s_wo, s_tmp
    .v_wei_move_slice_window_v4r1_bwd v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_wei_stride_k, s_move_slice_ik_wei, s_move_slice_idslice_y, s_move_slice_idslice_x, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, v_tmp, s_tmp
    .v_wei_calculate_os v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, s_wei_stride_k, s_wei_stride_c, s_wei_stride_y, v_tmp

    v_xor_b32 v[v_sst_b_os], 0x4000, v[v_sst_b_os] ; switch double buffer b store
    v_xor_b32 v[v_sst_a_os], 0x4000, v[v_sst_a_os] ; switch double buffer a store
    s_waitcnt lgkmcnt(0)
    s_barrier

    .v_out_load_k_n_8_1 v_gld_b, s_p_out, v_out_os, v_out_flag, v_out_ik, v_out_in, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dslice_y_neg, s_dslice_x_neg, s_dtile_dy, s_dtile_dx, s_out_stride_n, s_out_stride_k, s_out_stride_ho, s_ho, s_wo, v_tmp, s_tmp
    .v_wei_load_k_n_8_1 v_gld_a, s_p_wei, v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, s_wei_stride_c, s_wei_stride_k, s_wei_stride_y, v_tmp, s_tmp

L_igemm_v4r1_bwd_dynamic_fma_body:
    ; do fma accumulate with unroll 16
    ds_read_b128 v[v_a:v_a+3], v[v_sld_a_os]
    ds_read_b128 v[v_b:v_b+3], v[v_sld_b_os]
    ds_read_b128 v[v_b+4:v_b+4+3], v[v_sld_b_os] offset:256
    ds_read_b128 v[v_a+4:v_a+4+3], v[v_sld_a_os] offset:256
    .itr_k = 0
    .rept 15
        s_waitcnt lgkmcnt(2)
        .v_fma_4x4_s8 v_c,v_a,v_b

        s_waitcnt lgkmcnt(1)
        .v_fma_4x4_s8 v_c+4,v_a,v_b+4

        ds_read_b128 v[v_a:v_a+3], v[v_sld_a_os] offset:0+(.itr_k+1)*512
        s_waitcnt lgkmcnt(1)
        .v_fma_4x4_s8 v_c+32,v_a+4,v_b

        ds_read_b128 v[v_b:v_b+3], v[v_sld_b_os] offset:0+(.itr_k+1)*512
        .v_fma_4x4_s8 v_c+36,v_a+4,v_b+4

        ds_read_b128 v[v_b+4:v_b+4+3], v[v_sld_b_os] offset:0+(.itr_k+1)*512+256
        ds_read_b128 v[v_a+4:v_a+4+3], v[v_sld_a_os] offset:0+(.itr_k+1)*512+256
        .itr_k = .itr_k + 1
    .endr

    ; last unroll
    v_xor_b32 v[v_sld_b_os], 0x4000, v[v_sld_b_os] ; switch double buffer b load
    v_xor_b32 v[v_sld_a_os], 0x4000, v[v_sld_a_os] ; switch double buffer a load
    s_waitcnt lgkmcnt(2)
    .v_fma_4x4_s8 v_c,v_a,v_b

    s_waitcnt lgkmcnt(1)
    .v_fma_4x4_s8 v_c+4,v_a,v_b+4

    s_waitcnt vmcnt(8)
    .v_out_sst_k_n_8_1 v_gld_b, v_sst_b_os
    s_waitcnt vmcnt(0)
    .v_wei_sst_k_n_8_1 v_gld_a, v_sst_a_os

    s_sub_i32 s[s_kitr], s[s_kitr], 16
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_v4r1_bwd_dynamic_fma_finishing
    .v_out_move_slice_window_v4r1_bwd v_out_ik, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_out_stride_k, s_move_slice_ik_out, s_move_slice_idslice_y, s_move_slice_idslice_x, s_dslice_y_neg, s_dslice_x_neg, s_dtile_dy, s_dtile_dx, v_tmp, s_tmp
    .v_out_calculate_os v_out_os, v_out_in, v_out_ik, v_out_iho, v_out_iwo, s_out_stride_n, s_out_stride_k, s_out_stride_ho, v_tmp
    .v_out_set_flag v_out_flag, v_out_iho, v_out_iwo, s_ho, s_wo, s_tmp
    .v_wei_move_slice_window_v4r1_bwd v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_wei_stride_k, s_move_slice_ik_wei, s_move_slice_idslice_y, s_move_slice_idslice_x, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, v_tmp, s_tmp
    .v_wei_calculate_os v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, s_wei_stride_k, s_wei_stride_c, s_wei_stride_y, v_tmp

    s_waitcnt lgkmcnt(8)
    .v_fma_4x4_s8 v_c+32,v_a+4,v_b

    v_xor_b32 v[v_sst_b_os], 0x4000, v[v_sst_b_os] ; switch double buffer b store
    v_xor_b32 v[v_sst_a_os], 0x4000, v[v_sst_a_os] ; switch double buffer a store
    s_waitcnt lgkmcnt(0)
    s_barrier
    .v_out_load_k_n_8_1 v_gld_b, s_p_out, v_out_os, v_out_flag, v_out_ik, v_out_in, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dslice_y_neg, s_dslice_x_neg, s_dtile_dy, s_dtile_dx, s_out_stride_n, s_out_stride_k, s_out_stride_ho, s_ho, s_wo, v_tmp, s_tmp
    .v_wei_load_k_n_8_1 v_gld_a, s_p_wei, v_wei_os, v_wei_ik, v_wei_ic, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, v_dtile_iy, v_dtile_ix, s_dslice_y, s_dslice_x, s_dtile_y, s_dtile_x, s_wei_stride_c, s_wei_stride_k, s_wei_stride_y, v_tmp, s_tmp

    .v_fma_4x4_s8 v_c+36,v_a+4,v_b+4

    s_branch L_igemm_v4r1_bwd_dynamic_fma_body
L_igemm_v4r1_bwd_dynamic_fma_finishing:
    s_waitcnt lgkmcnt(8)
    .v_fma_4x4_s8 v_c+32,v_a+4,v_b
    .v_fma_4x4_s8 v_c+36,v_a+4,v_b+4
L_igemm_v4r1_bwd_dynamic_end:

    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b128 v[v_a:v_a+3], v[v_sld_a_os]
    ds_read_b128 v[v_b:v_b+3], v[v_sld_b_os]
    ds_read_b128 v[v_b+4:v_b+4+3], v[v_sld_b_os] offset:256
    ds_read_b128 v[v_a+4:v_a+4+3], v[v_sld_a_os] offset:256
    .itr_k = 0
    .rept 15
        s_waitcnt lgkmcnt(2)
        .v_fma_4x4_s8 v_c,v_a,v_b

        s_waitcnt lgkmcnt(1)
        .v_fma_4x4_s8 v_c+4,v_a,v_b+4

        ds_read_b128 v[v_a:v_a+3], v[v_sld_a_os] offset:0+(.itr_k+1)*512
        s_waitcnt lgkmcnt(1)
        .v_fma_4x4_s8 v_c+32,v_a+4,v_b

        ds_read_b128 v[v_b:v_b+3], v[v_sld_b_os] offset:0+(.itr_k+1)*512
        .v_fma_4x4_s8 v_c+36,v_a+4,v_b+4

        ds_read_b128 v[v_b+4:v_b+4+3], v[v_sld_b_os] offset:0+(.itr_k+1)*512+256
        ds_read_b128 v[v_a+4:v_a+4+3], v[v_sld_a_os] offset:0+(.itr_k+1)*512+256
        .itr_k = .itr_k + 1
    .endr

    ; last unroll
    s_waitcnt lgkmcnt(2)
    .v_fma_4x4_s8 v_c,v_a,v_b

    s_waitcnt lgkmcnt(1)
    .v_fma_4x4_s8 v_c+4,v_a,v_b+4

    s_waitcnt lgkmcnt(0)
    .v_fma_4x4_s8 v_c+32,v_a+4,v_b

    .v_fma_4x4_s8 v_c+36,v_a+4,v_b+4

    s_mul_i32 s[s_tmp], s[s_dtile_iy], s[s_dilation_h]
    v_mov_b32 v[v_dtile_iy_x_dilation_h], s[s_tmp]
    s_mul_i32 s[s_tmp+1], s[s_dtile_ix], s[s_dilation_w]
    v_mov_b32 v[v_dtile_ix_x_dilation_w], s[s_tmp+1]

    v_lshl_or_b32 v[v_in_gemm_in], v[v_in_gemm_in0], 6, v[v_in_gemm_in1]
    .v_in_transform_gemm_n v_in_in, v_in_ihi, v_in_iwi, v_in_gemm_in, v_in_dslice_h, v_in_dslice_w, s_stride_dslice_hw, s_dslice_w, s_dtile_iy, s_dtile_ix, s_dslice_h_left, s_dslice_w_left, s_dilation_h, s_dilation_w, s_stride_h, s_stride_w, s_pad_h, s_pad_w, v_tmp, s_tmp

    v_mul_lo_u32 v[v_in_in], s[s_in_stride_n], v[v_in_in]
    v_mul_lo_u32 v[v_in_ic], s[s_in_stride_c], v[v_in_ic]
    .v_in_calculate_os v_in_os, v_in_in, v_in_ic, v_in_ihi, v_in_iwi, s_in_stride_n, s_in_stride_c, s_in_stride_hi, v_tmp

    .v_in_set_flag v_in_flag, v_in_ihi, v_in_iwi, s_hi, s_wi, s_tmp

    s_lshl_b32 s[s_block_gemm_in], s[s_in_stride_c], 6
    .v_in_write_m0_m1_n0_n1_step v_c, s_p_in, v_in_os, v_in_flag, v_in_ihi_itr, v_in_iwi_itr, v_in_in_itr, v_in_dslice_h_itr, v_in_dslice_w_itr, v_in_ic_itr, v_in_in, v_in_ic, v_in_ihi, v_in_iwi, v_in_dslice_h, v_in_dslice_w, v_dtile_iy_x_dilation_h, v_dtile_ix_x_dilation_w, s_move_slice_in_in, s_move_slice_in_dslice_h, s_move_slice_in_dslice_w, s_dslice_h_left, s_dslice_w_left, s_dslice_h_shifted, s_dslice_w_shifted, s_dslice_h, s_dslice_w, s_dtile_iy, s_dtile_ix, s_dilation_h, s_dilation_w, s_stride_h, s_stride_w, s_pad_h, s_pad_w, s_in_stride_n, s_in_stride_c, s_block_gemm_in, s_in_stride_hi, s_hi, s_wi, v_tmp, s_tmp, 2, 4, 2, 4
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel igemm_v4r1_bwd_dynamic
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 128
    .amdhsa_next_free_sgpr 72
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_v4r1_bwd_dynamic
    .symbol: igemm_v4r1_bwd_dynamic.kd
    .sgpr_count: 72
    .vgpr_count: 128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .group_segment_fixed_size: 32768
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
    - { .name: dslice_h_left , .size: 4, .offset: 132, .value_kind: by_value, .value_type: i32}
    - { .name: dslice_w_left , .size: 4, .offset: 136, .value_kind: by_value, .value_type: i32}
    - { .name: __pack0   , .size: 4, .offset: 140, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata