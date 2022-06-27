	.text
	.protected	naive_conv_fwd_nchw_float ; -- Begin function naive_conv_fwd_nchw_float
	.globl	naive_conv_fwd_nchw_float
	.p2align	8
	.type	naive_conv_fwd_nchw_float,@function
naive_conv_fwd_nchw_float:               ; @naive_conv_fwd_nchw_float
naive_conv_fwd_nchw_float$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s14, s13
	v_cmp_gt_i32_e32 vcc, s7, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_15
; %bb.1:                                ; %.lr.ph172
	s_ashr_i32 s0, s11, 31
	s_add_i32 s1, s11, s0
	s_xor_b32 s3, s1, s0
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s24, 0x4f800000
	s_ashr_i32 s1, s10, 31
	s_add_i32 s2, s10, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s31, s2, s1
	v_cvt_f32_u32_e32 v4, s31
	s_ashr_i32 s25, s6, 31
	v_mul_f32_e32 v1, s24, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s1, s6, s25
	s_xor_b32 s30, s1, s25
	s_xor_b32 s2, s25, s0
	v_mul_lo_u32 v2, v1, s3
	v_mul_hi_u32 v3, v1, s3
	s_mul_i32 s34, s23, s11
	s_mul_i32 s23, s23, s12
	v_sub_u32_e32 v5, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v5, vcc
	v_mul_hi_u32 v2, v2, v1
	v_rcp_iflag_f32_e32 v3, v4
	s_load_dwordx2 s[26:27], s[4:5], 0x0
	s_load_dwordx2 s[28:29], s[4:5], 0x8
	s_load_dwordx2 s[4:5], s[4:5], 0x10
	s_sub_i32 s20, 0, s20
	v_add_u32_e32 v4, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s30
	v_mul_f32_e32 v2, s24, v3
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s3
	v_add_u32_e32 v4, 1, v1
	v_add_u32_e32 v5, -1, v1
	v_sub_u32_e32 v6, s30, v3
	v_cmp_ge_u32_e32 vcc, s30, v3
	v_cmp_le_u32_e64 s[0:1], s3, v6
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	s_mul_i32 s0, s11, s10
	v_mul_hi_u32 v4, v2, s31
	s_ashr_i32 s10, s0, 31
	v_mul_lo_u32 v3, v2, s31
	s_add_i32 s0, s0, s10
	s_xor_b32 s33, s0, s10
	v_cndmask_b32_e32 v1, v5, v1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s33
	v_sub_u32_e32 v6, 0, v3
	v_cndmask_b32_e32 v3, v3, v6, vcc
	v_mul_hi_u32 v3, v3, v2
	v_rcp_iflag_f32_e32 v4, v4
	v_xor_b32_e32 v1, s2, v1
	v_subrev_u32_e32 v5, s2, v1
	v_add_u32_e32 v7, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_mul_f32_e32 v3, s24, v4
	v_cvt_u32_f32_e32 v3, v3
	v_ashrrev_i32_e32 v1, 31, v5
	v_add_u32_e32 v6, v5, v1
	v_cndmask_b32_e32 v2, v2, v7, vcc
	v_mul_lo_u32 v4, v3, s33
	v_mul_hi_u32 v7, v3, s33
	v_xor_b32_e32 v6, v6, v1
	v_mul_hi_u32 v2, v2, v6
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v7
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_lo_u32 v2, v2, s31
	v_mul_hi_u32 v4, v4, v3
	v_sub_u32_e32 v8, v6, v2
	v_cmp_ge_u32_e64 s[2:3], v6, v2
	v_add_u32_e32 v6, v3, v4
	v_sub_u32_e32 v3, v3, v4
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	v_mul_hi_u32 v3, v3, s30
	v_cmp_le_u32_e32 vcc, s31, v8
	v_subrev_u32_e32 v2, s31, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_mul_lo_u32 v4, v3, s33
	v_add_u32_e32 v7, s31, v8
	v_cndmask_b32_e32 v2, v8, v2, vcc
	v_cndmask_b32_e64 v2, v7, v2, s[2:3]
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v7, v2, v1
	v_sub_u32_e32 v1, s30, v4
	v_cmp_le_u32_e32 vcc, s33, v1
	v_cmp_ge_u32_e64 s[0:1], s30, v4
	v_add_u32_e32 v2, 1, v3
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v1, -1, v3
	v_cndmask_b32_e32 v2, v3, v2, vcc
	v_cndmask_b32_e64 v3, v1, v2, s[0:1]
	v_mul_hi_i32 v2, v7, s23
	v_mul_lo_u32 v1, v7, s23
	s_xor_b32 s2, s25, s10
	v_xor_b32_e32 v3, s2, v3
	v_subrev_u32_e32 v6, s2, v3
	v_mad_i64_i32 v[1:2], s[0:1], v6, s12, v[1:2]
	v_mul_lo_u32 v3, v5, s11
	s_mul_i32 s2, s9, s8
	s_mul_hi_i32 s0, s9, s8
	v_mul_lo_u32 v2, s2, v2
	v_mul_hi_u32 v4, s2, v1
	v_mul_lo_u32 v5, s0, v1
	v_mul_lo_u32 v1, s2, v1
	v_sub_u32_e32 v3, s6, v3
	v_add_u32_e32 v2, v4, v2
	v_ashrrev_i32_e32 v4, 31, v3
	v_mad_i64_i32 v[3:4], s[0:1], v6, s11, v[3:4]
	v_add_u32_e32 v2, v2, v5
	s_mul_i32 s1, s21, s12
	s_ashr_i32 s3, s22, 31
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	s_mul_hi_i32 s0, s21, s12
	s_mul_i32 s3, s1, s3
	s_mul_hi_u32 s6, s1, s22
	s_mul_i32 s1, s1, s22
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s22
	s_add_i32 s3, s3, s0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, s27
	v_add_co_u32_e32 v1, vcc, s26, v1
	v_mul_lo_u32 v6, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_mul_lo_u32 v9, s3, v3
	v_addc_co_u32_e32 v2, vcc, v5, v2, vcc
	v_mul_lo_u32 v5, s1, v3
	v_mad_i64_i32 v[3:4], s[0:1], v7, s34, v[3:4]
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	s_mul_hi_i32 s0, s14, s13
	v_mul_lo_u32 v9, s7, v4
	v_mul_hi_u32 v10, s7, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_mul_lo_u32 v7, s7, v3
	v_add_co_u32_e32 v3, vcc, s28, v5
	v_mov_b32_e32 v8, s29
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 2, v[7:8]
	v_mov_b32_e32 v7, s5
	v_add_co_u32_e32 v5, vcc, s4, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[4:5], s12, 0
	v_cmp_gt_i32_e64 s[10:11], s21, 0
	v_cmp_gt_i32_e64 s[26:27], s22, 0
	s_mul_i32 s3, s22, s21
	s_sub_i32 s6, 0, s19
	s_mul_i32 s13, s17, s9
	s_mov_b64 s[28:29], 0
	s_branch BB0_3
BB0_2:                                  ; %._crit_edge168
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_mul_lo_u32 v7, v7, s14
	v_cvt_f32_f64_e32 v9, v[9:10]
	v_add_u32_e32 v0, 0x100, v0
	v_add_u32_e32 v7, v7, v8
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32_e32 v7, vcc, v5, v7
	v_addc_co_u32_e32 v8, vcc, v6, v8, vcc
	v_cmp_le_i32_e32 vcc, s7, v0
	s_or_b64 s[28:29], vcc, s[28:29]
	global_store_dword v[7:8], v9, off
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz BB0_15
BB0_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_6 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
                                        ;         Child Loop BB0_13 Depth 4
	s_ashr_i32 s0, s14, 31
	s_add_i32 s1, s14, s0
	s_xor_b32 s1, s1, s0
	v_cvt_f32_u32_e32 v7, s1
	v_rcp_iflag_f32_e32 v7, v7
	v_mul_f32_e32 v7, s24, v7
	v_cvt_u32_f32_e32 v7, v7
	v_mul_lo_u32 v8, v7, s1
	v_mul_hi_u32 v9, v7, s1
	v_sub_u32_e32 v10, 0, v8
	v_cmp_eq_u32_e32 vcc, 0, v9
	v_cndmask_b32_e32 v8, v8, v10, vcc
	v_mul_hi_u32 v8, v8, v7
	v_ashrrev_i32_e32 v9, 31, v0
	v_add_u32_e32 v10, v0, v9
	v_xor_b32_e32 v10, v10, v9
	v_add_u32_e32 v11, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_mul_hi_u32 v7, v7, v10
	v_xor_b32_e32 v9, s0, v9
	v_mul_lo_u32 v8, v7, s1
	v_add_u32_e32 v11, 1, v7
	v_add_u32_e32 v12, -1, v7
	v_sub_u32_e32 v13, v10, v8
	v_cmp_ge_u32_e32 vcc, v10, v8
	v_cmp_le_u32_e64 s[0:1], s1, v13
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v11, s[0:1]
	v_cndmask_b32_e32 v7, v12, v7, vcc
	v_xor_b32_e32 v7, v7, v9
	v_sub_u32_e32 v7, v7, v9
	v_mul_lo_u32 v8, v7, s14
	v_mov_b32_e32 v9, 0
	s_andn2_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v8, v0, v8
	s_cbranch_vccnz BB0_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_mul_lo_u32 v9, v7, s15
	v_mul_lo_u32 v10, s16, v8
	s_mov_b32 s23, 0
	s_mov_b32 s25, 0
	v_subrev_u32_e32 v11, s19, v9
	v_add_u32_e32 v9, s6, v9
	v_mul_lo_u32 v12, s9, v9
	v_add_u32_e32 v13, s20, v10
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	s_branch BB0_6
BB0_5:                                  ; %._crit_edge163
                                        ;   in Loop: Header=BB0_6 Depth=2
	s_add_i32 s25, s25, 1
	s_add_i32 s23, s23, s3
	s_cmp_eq_u32 s25, s12
	v_add_u32_e32 v12, s2, v12
	s_cbranch_scc1 BB0_2
BB0_6:                                  ; %.preheader
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_10 Depth 3
                                        ;         Child Loop BB0_13 Depth 4
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz BB0_5
; %bb.7:                                ; %.lr.ph162
                                        ;   in Loop: Header=BB0_6 Depth=2
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz BB0_5
; %bb.8:                                ; %.lr.ph162.split.us.preheader
                                        ;   in Loop: Header=BB0_6 Depth=2
	s_mov_b32 s30, 0
	v_mov_b32_e32 v14, v12
	s_mov_b32 s31, s23
	s_branch BB0_10
BB0_9:                                  ; %Flow59
                                        ;   in Loop: Header=BB0_10 Depth=3
	s_or_b64 exec, exec, s[34:35]
	s_add_i32 s30, s30, 1
	s_add_i32 s31, s31, s22
	s_cmp_eq_u32 s30, s21
	v_add_u32_e32 v14, s13, v14
	s_cbranch_scc1 BB0_5
BB0_10:                                 ; %.lr.ph162.split.us
                                        ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB0_13 Depth 4
	s_mul_i32 s0, s30, s17
	v_add_u32_e32 v15, s0, v11
	v_cmp_lt_i32_e32 vcc, -1, v15
	v_cmp_gt_i32_e64 s[0:1], s8, v15
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[34:35], s[0:1]
	s_cbranch_execz BB0_9
; %bb.11:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB0_10 Depth=3
	v_mov_b32_e32 v15, v13
	s_mov_b32 s36, s31
	s_mov_b32 s33, s22
	s_branch BB0_13
BB0_12:                                 ;   in Loop: Header=BB0_13 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s33, s33, -1
	s_add_i32 s36, s36, 1
	s_cmp_lg_u32 s33, 0
	v_add_u32_e32 v15, s18, v15
	s_cbranch_scc0 BB0_9
BB0_13:                                 ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ;       Parent Loop BB0_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v15
	v_cmp_gt_i32_e64 s[0:1], s9, v15
	s_and_b64 s[38:39], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[38:39]
	s_cbranch_execz BB0_12
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=4
	v_add_u32_e32 v16, v14, v15
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshlrev_b64 v[16:17], 2, v[16:17]
	s_ashr_i32 s37, s36, 31
	v_add_co_u32_e32 v16, vcc, v1, v16
	s_lshl_b64 s[38:39], s[36:37], 2
	v_addc_co_u32_e32 v17, vcc, v2, v17, vcc
	v_mov_b32_e32 v19, s39
	v_add_co_u32_e32 v18, vcc, s38, v3
	v_addc_co_u32_e32 v19, vcc, v4, v19, vcc
	global_load_dword v18, v[18:19], off
	global_load_dword v16, v[16:17], off
	s_waitcnt vmcnt(1)
	v_cvt_f64_f32_e32 v[18:19], v18
	s_waitcnt vmcnt(0)
	v_cvt_f64_f32_e32 v[16:17], v16
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[9:10], v[16:17], v[18:19], v[9:10]
	s_branch BB0_12
BB0_15:                                 ; %Flow65
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_fwd_nchw_float
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 40
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	naive_conv_fwd_nchw_float, .Lfunc_end0-naive_conv_fwd_nchw_float
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 1452
; NumSgprs: 42
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 42
; NumVGPRsForWavesPerEU: 20
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_bwd_nchw_float ; -- Begin function naive_conv_bwd_nchw_float
	.globl	naive_conv_bwd_nchw_float
	.p2align	8
	.type	naive_conv_bwd_nchw_float,@function
naive_conv_bwd_nchw_float:               ; @naive_conv_bwd_nchw_float
naive_conv_bwd_nchw_float$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s9, s8
	v_cmp_gt_i32_e32 vcc, s7, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_26
; %bb.1:                                ; %.lr.ph183
	s_ashr_i32 s25, s12, 31
	s_add_i32 s0, s12, s25
	s_xor_b32 s3, s0, s25
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s24, 0x4f800000
	s_ashr_i32 s0, s10, 31
	s_add_i32 s1, s10, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s28, s1, s0
	s_ashr_i32 s26, s6, 31
	v_cvt_f32_u32_e32 v2, s28
	v_mul_f32_e32 v1, s24, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s0, s6, s26
	s_xor_b32 s27, s0, s26
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s3
	v_mul_hi_u32 v4, v1, s3
	s_xor_b32 s2, s26, s25
	v_mul_f32_e32 v2, s24, v2
	v_sub_u32_e32 v5, 0, v3
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_mul_hi_u32 v3, v3, v1
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s30, s23, s12
	s_mul_hi_i32 s8, s9, s8
	v_add_u32_e32 v4, v1, v3
	v_sub_u32_e32 v1, v1, v3
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s27
	v_mul_hi_u32 v3, v2, s28
	v_mul_lo_u32 v5, v2, s28
	v_mul_lo_u32 v4, v1, s3
	v_add_u32_e32 v6, 1, v1
	v_add_u32_e32 v7, -1, v1
	v_sub_u32_e32 v8, s27, v4
	v_cmp_ge_u32_e32 vcc, s27, v4
	v_cmp_le_u32_e64 s[0:1], s3, v8
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v6, s[0:1]
	s_mul_i32 s0, s12, s10
	s_ashr_i32 s10, s0, 31
	s_add_i32 s0, s0, s10
	v_cndmask_b32_e32 v1, v7, v1, vcc
	v_sub_u32_e32 v7, 0, v5
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_xor_b32 s29, s0, s10
	v_cndmask_b32_e32 v3, v5, v7, vcc
	v_cvt_f32_u32_e32 v5, s29
	v_xor_b32_e32 v1, s2, v1
	v_mul_hi_u32 v3, v3, v2
	v_subrev_u32_e32 v1, s2, v1
	v_rcp_iflag_f32_e32 v5, v5
	v_ashrrev_i32_e32 v6, 31, v1
	v_mul_lo_u32 v4, v1, s12
	v_add_u32_e32 v1, v1, v6
	v_xor_b32_e32 v7, v1, v6
	v_add_u32_e32 v1, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_cndmask_b32_e32 v1, v2, v1, vcc
	v_mul_f32_e32 v2, s24, v5
	v_mul_hi_u32 v1, v1, v7
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s28
	v_sub_u32_e32 v1, s6, v4
	v_mul_lo_u32 v4, v2, s29
	v_mul_hi_u32 v5, v2, s29
	v_sub_u32_e32 v8, v7, v3
	v_cmp_ge_u32_e64 s[2:3], v7, v3
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v5
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_hi_u32 v4, v4, v2
	v_cmp_le_u32_e32 vcc, s28, v8
	v_subrev_u32_e32 v3, s28, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_add_u32_e32 v7, v2, v4
	v_sub_u32_e32 v2, v2, v4
	v_cndmask_b32_e64 v2, v2, v7, s[0:1]
	v_mul_hi_u32 v2, v2, s27
	v_add_u32_e32 v5, s28, v8
	v_cndmask_b32_e32 v3, v8, v3, vcc
	v_cndmask_b32_e64 v3, v5, v3, s[2:3]
	v_mul_lo_u32 v4, v2, s29
	v_xor_b32_e32 v3, v3, v6
	v_sub_u32_e32 v5, v3, v6
	s_xor_b32 s2, s26, s10
	v_sub_u32_e32 v3, s27, v4
	v_cmp_le_u32_e32 vcc, s29, v3
	v_cmp_ge_u32_e64 s[0:1], s27, v4
	v_add_u32_e32 v4, 1, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v3, -1, v2
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e64 v6, v3, v2, s[0:1]
	v_ashrrev_i32_e32 v2, 31, v1
	v_mad_i64_i32 v[3:4], s[0:1], v5, s30, v[1:2]
	v_xor_b32_e32 v6, s2, v6
	v_subrev_u32_e32 v7, s2, v6
	s_mul_i32 s6, s23, s11
	v_mad_i64_i32 v[3:4], s[0:1], v7, s12, v[3:4]
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[2:3], s[4:5], 0x8
	s_load_dwordx2 s[4:5], s[4:5], 0x10
	v_cmp_gt_i32_e64 s[26:27], s22, 0
	s_mov_b64 s[28:29], 0
	v_mul_lo_u32 v4, s7, v4
	v_mul_hi_u32 v6, s7, v3
	v_mul_lo_u32 v8, s8, v3
	v_mul_lo_u32 v3, s7, v3
	s_mul_i32 s8, s22, s21
	v_add_u32_e32 v4, v6, v4
	v_mul_lo_u32 v6, v7, s11
	v_add_u32_e32 v4, v4, v8
	v_mul_hi_i32 v8, v7, s11
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_mul_lo_u32 v9, v6, s25
	v_mul_hi_u32 v10, v6, s12
	v_mul_lo_u32 v6, v6, s12
	v_mul_lo_u32 v8, v8, s12
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v11, s1
	v_add_u32_e32 v9, v10, v9
	v_add_co_u32_e32 v6, vcc, v6, v1
	v_add_u32_e32 v8, v9, v8
	v_addc_co_u32_e32 v1, vcc, v8, v2, vcc
	v_mul_lo_u32 v8, s8, v1
	v_mul_hi_u32 v9, s8, v6
	v_add_co_u32_e32 v1, vcc, s0, v3
	s_mul_hi_i32 s0, s22, s21
	v_addc_co_u32_e32 v2, vcc, v11, v4, vcc
	v_add_u32_e32 v4, v9, v8
	v_mul_lo_u32 v8, s0, v6
	v_mul_lo_u32 v3, s8, v6
	v_mul_hi_i32 v6, v5, s6
	v_mul_lo_u32 v5, v5, s6
	v_add_u32_e32 v4, v4, v8
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	s_mul_i32 s6, s8, s12
	v_mad_i64_i32 v[5:6], s[0:1], v7, s11, v[5:6]
	s_mul_i32 s1, s14, s13
	s_mul_hi_i32 s0, s14, s13
	v_mov_b32_e32 v7, s3
	v_mul_lo_u32 v6, s1, v6
	v_mul_hi_u32 v8, s1, v5
	v_mul_lo_u32 v9, s0, v5
	v_mul_lo_u32 v5, s1, v5
	v_add_co_u32_e32 v3, vcc, s2, v3
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_addc_co_u32_e32 v4, vcc, v7, v4, vcc
	v_mov_b32_e32 v7, s5
	v_add_co_u32_e32 v5, vcc, s4, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[2:3], s11, 0
	v_cmp_gt_i32_e64 s[4:5], s21, 0
	s_branch BB1_3
BB1_2:                                  ; %Flow166
                                        ;   in Loop: Header=BB1_3 Depth=1
	v_mul_lo_u32 v7, v7, s9
	v_sub_u32_e32 v8, v0, v8
	v_cvt_f32_f64_e32 v9, v[9:10]
	v_add_u32_e32 v0, 0x100, v0
	v_add_u32_e32 v7, v7, v8
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32_e32 v7, vcc, v1, v7
	v_addc_co_u32_e32 v8, vcc, v2, v8, vcc
	v_cmp_le_i32_e32 vcc, s7, v0
	s_or_b64 s[28:29], vcc, s[28:29]
	global_store_dword v[7:8], v9, off
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz BB1_26
BB1_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_6 Depth 2
                                        ;       Child Loop BB1_10 Depth 3
                                        ;         Child Loop BB1_18 Depth 4
	s_ashr_i32 s0, s9, 31
	s_add_i32 s1, s9, s0
	s_xor_b32 s1, s1, s0
	v_cvt_f32_u32_e32 v7, s1
	v_rcp_iflag_f32_e32 v7, v7
	v_mul_f32_e32 v7, s24, v7
	v_cvt_u32_f32_e32 v7, v7
	v_mul_lo_u32 v8, v7, s1
	v_mul_hi_u32 v9, v7, s1
	v_sub_u32_e32 v10, 0, v8
	v_cmp_eq_u32_e32 vcc, 0, v9
	v_cndmask_b32_e32 v8, v8, v10, vcc
	v_mul_hi_u32 v8, v8, v7
	v_ashrrev_i32_e32 v9, 31, v0
	v_add_u32_e32 v10, v0, v9
	v_xor_b32_e32 v10, v10, v9
	v_add_u32_e32 v11, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_mul_hi_u32 v7, v7, v10
	v_xor_b32_e32 v9, s0, v9
	v_mul_lo_u32 v8, v7, s1
	v_add_u32_e32 v11, 1, v7
	v_add_u32_e32 v12, -1, v7
	v_sub_u32_e32 v13, v10, v8
	v_cmp_ge_u32_e32 vcc, v10, v8
	v_cmp_le_u32_e64 s[0:1], s1, v13
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v11, s[0:1]
	v_cndmask_b32_e32 v7, v12, v7, vcc
	v_xor_b32_e32 v7, v7, v9
	v_sub_u32_e32 v7, v7, v9
	v_mul_lo_u32 v8, v7, s9
	v_mov_b32_e32 v9, 0
	s_andn2_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v10, 0
	s_cbranch_vccnz BB1_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB1_3 Depth=1
	v_add_u32_e32 v9, s20, v0
	v_sub_u32_e32 v12, v9, v8
	v_mov_b32_e32 v9, 0
	v_add_u32_e32 v11, s19, v7
	s_mov_b32 s8, 0
	v_mov_b32_e32 v10, 0
	s_mov_b32 s10, 0
	s_branch BB1_6
BB1_5:                                  ; %._crit_edge174
                                        ;   in Loop: Header=BB1_6 Depth=2
	s_add_i32 s10, s10, 1
	s_add_i32 s8, s8, s6
	s_cmp_eq_u32 s10, s11
	s_cbranch_scc1 BB1_2
BB1_6:                                  ; %.preheader
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_10 Depth 3
                                        ;         Child Loop BB1_18 Depth 4
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz BB1_5
; %bb.7:                                ; %.lr.ph173
                                        ;   in Loop: Header=BB1_6 Depth=2
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz BB1_5
; %bb.8:                                ; %.lr.ph173.split.us.preheader
                                        ;   in Loop: Header=BB1_6 Depth=2
	s_mul_i32 s12, s10, s13
	s_mov_b32 s23, 0
	s_mov_b32 s25, s8
	s_branch BB1_10
BB1_9:                                  ; %._crit_edge.us
                                        ;   in Loop: Header=BB1_10 Depth=3
	s_add_i32 s23, s23, 1
	s_add_i32 s25, s25, s22
	s_cmp_eq_u32 s23, s21
	s_cbranch_scc1 BB1_5
BB1_10:                                 ; %.lr.ph173.split.us
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB1_18 Depth 4
	s_mul_i32 s0, s23, s17
	v_subrev_u32_e32 v25, s0, v11
	v_cmp_lt_i32_e32 vcc, -1, v25
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $sgpr30
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $sgpr31
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr21_vgpr22
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr23_vgpr24
	s_and_saveexec_b64 s[34:35], vcc
	s_xor_b64 s[34:35], exec, s[34:35]
	s_cbranch_execz BB1_12
; %bb.11:                               ;   in Loop: Header=BB1_10 Depth=3
	s_ashr_i32 s30, s15, 31
	s_add_i32 s0, s15, s30
	s_xor_b32 s31, s0, s30
	v_cvt_f32_u32_e32 v13, s31
	v_ashrrev_i32_e32 v17, 31, v25
	v_rcp_iflag_f32_e32 v13, v13
	v_mul_f32_e32 v13, s24, v13
	v_cvt_u32_f32_e32 v15, v13
	v_mul_lo_u32 v18, v15, s31
	v_mul_hi_u32 v20, v15, s31
	v_sub_u32_e32 v19, 0, v18
	v_cmp_eq_u32_e32 vcc, 0, v20
	v_cndmask_b32_e32 v13, v18, v19, vcc
	v_mul_hi_u32 v14, v13, v15
	v_add_u32_e32 v13, v25, v17
	v_xor_b32_e32 v13, v13, v17
	v_add_u32_e32 v16, v15, v14
	v_sub_u32_e32 v14, v15, v14
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_mul_hi_u32 v14, v14, v13
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v21, v15
	v_mul_lo_u32 v23, v14, s31
	v_mov_b32_e32 v14, v16
	v_sub_u32_e32 v16, v13, v23
	v_cmp_ge_u32_e64 s[0:1], v13, v23
	v_cmp_le_u32_e32 vcc, s31, v16
	v_subrev_u32_e32 v23, s31, v16
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v24, s31, v16
	v_cndmask_b32_e32 v16, v16, v23, vcc
	v_cndmask_b32_e64 v16, v24, v16, s[0:1]
	v_xor_b32_e32 v16, v16, v17
	v_sub_u32_e32 v16, v16, v17
	v_cmp_ne_u32_e32 vcc, 0, v16
	v_mov_b32_e32 v24, v14
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v23, v13
BB1_12:                                 ; %Flow161
                                        ;   in Loop: Header=BB1_10 Depth=3
	s_or_saveexec_b64 s[34:35], s[34:35]
	v_mov_b32_e32 v34, v22
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v16, s30
	v_mov_b32_e32 v28, s31
	v_mov_b32_e32 v14, v17
	v_mov_b32_e32 v29, v13
	v_mov_b32_e32 v31, v18
	v_mov_b32_e32 v32, v20
	v_mov_b32_e32 v35, v19
	v_mov_b32_e32 v26, v15
	v_mov_b32_e32 v33, v21
	v_mov_b32_e32 v36, v23
	s_xor_b64 exec, exec, s[34:35]
	s_cbranch_execz BB1_14
; %bb.13:                               ; %.lr.ph173.split.us._crit_edge
                                        ;   in Loop: Header=BB1_10 Depth=3
	s_ashr_i32 s33, s15, 31
	s_add_i32 s36, s15, s33
	s_xor_b32 s36, s36, s33
	v_cvt_f32_u32_e32 v16, s36
	v_ashrrev_i32_e32 v14, 31, v25
	v_mov_b32_e32 v27, 0
	v_add_u32_e32 v25, v25, v14
	v_rcp_iflag_f32_e32 v16, v16
	v_mov_b32_e32 v30, v27
	v_xor_b32_e32 v29, v25, v14
	v_mov_b32_e32 v37, v30
	v_mul_f32_e32 v16, s24, v16
	v_cvt_u32_f32_e32 v26, v16
	v_mov_b32_e32 v34, v27
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v16, s33
	v_mul_lo_u32 v31, v26, s36
	v_mul_hi_u32 v32, v26, s36
	v_mov_b32_e32 v28, s36
	v_mov_b32_e32 v33, v26
	v_sub_u32_e32 v35, 0, v31
	v_mov_b32_e32 v36, v29
BB1_14:                                 ; %Flow162
                                        ;   in Loop: Header=BB1_10 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_mov_b32_e32 v27, s31
	v_mov_b32_e32 v30, s30
	v_mov_b32_e32 v25, 1
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz BB1_16
; %bb.15:                               ;   in Loop: Header=BB1_10 Depth=3
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v21, v33
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, v37
	v_mov_b32_e32 v19, v35
	v_mov_b32_e32 v20, v32
	v_mov_b32_e32 v18, v31
	v_mov_b32_e32 v22, v34
	v_mov_b32_e32 v15, v26
	v_mov_b32_e32 v27, v28
	v_mov_b32_e32 v13, v29
	v_mov_b32_e32 v30, v16
	v_mov_b32_e32 v17, v14
BB1_16:                                 ; %.lr.ph.us
                                        ;   in Loop: Header=BB1_10 Depth=3
	s_or_b64 exec, exec, s[30:31]
	v_cmp_eq_u32_e32 vcc, 0, v20
	v_cndmask_b32_e32 v14, v18, v19, vcc
	v_mul_lo_u32 v16, v14, v22
	v_mul_hi_u32 v14, v14, v21
	s_mov_b32 s30, s25
	s_mov_b32 s33, s22
	v_add_u32_e32 v14, v14, v16
	v_add_u32_e32 v16, v15, v14
	v_sub_u32_e32 v14, v15, v14
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_mul_lo_u32 v15, v14, v24
	v_mul_hi_u32 v14, v14, v23
	v_xor_b32_e32 v16, v17, v30
	v_add_u32_e32 v14, v14, v15
	v_mul_lo_u32 v15, v14, v27
	v_add_u32_e32 v17, 1, v14
	v_add_u32_e32 v18, -1, v14
	v_sub_u32_e32 v19, v13, v15
	v_cmp_ge_u32_e32 vcc, v13, v15
	v_cmp_ge_u32_e64 s[0:1], v19, v27
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v13, v14, v17, s[0:1]
	v_cndmask_b32_e32 v13, v18, v13, vcc
	v_xor_b32_e32 v13, v13, v16
	v_sub_u32_e32 v14, v13, v16
	v_cmp_gt_i32_e32 vcc, s13, v14
	v_add_u32_e32 v14, s12, v14
	v_mul_lo_u32 v14, v14, s14
	v_cndmask_b32_e32 v13, 0, v25, vcc
	v_mov_b32_e32 v15, v12
	s_branch BB1_18
BB1_17:                                 ;   in Loop: Header=BB1_18 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s33, s33, -1
	s_add_i32 s30, s30, 1
	s_cmp_lg_u32 s33, 0
	v_subrev_u32_e32 v15, s18, v15
	s_cbranch_scc0 BB1_9
BB1_18:                                 ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
                                        ;       Parent Loop BB1_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v15
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $sgpr31
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $sgpr34
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr24_vgpr25
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr26_vgpr27
	s_and_saveexec_b64 s[36:37], vcc
	s_xor_b64 s[36:37], exec, s[36:37]
	s_cbranch_execz BB1_20
; %bb.19:                               ;   in Loop: Header=BB1_18 Depth=4
	s_ashr_i32 s31, s16, 31
	s_add_i32 s0, s16, s31
	s_xor_b32 s34, s0, s31
	v_cvt_f32_u32_e32 v16, s34
	v_ashrrev_i32_e32 v20, 31, v15
	v_rcp_iflag_f32_e32 v16, v16
	v_mul_f32_e32 v16, s24, v16
	v_cvt_u32_f32_e32 v18, v16
	v_mul_lo_u32 v21, v18, s34
	v_mul_hi_u32 v23, v18, s34
	v_sub_u32_e32 v22, 0, v21
	v_cmp_eq_u32_e32 vcc, 0, v23
	v_cndmask_b32_e32 v16, v21, v22, vcc
	v_mul_hi_u32 v17, v16, v18
	v_add_u32_e32 v16, v15, v20
	v_xor_b32_e32 v16, v16, v20
	v_add_u32_e32 v19, v18, v17
	v_sub_u32_e32 v17, v18, v17
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_mul_hi_u32 v17, v17, v16
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v25, v19
	v_mov_b32_e32 v24, v18
	v_mul_lo_u32 v26, v17, s34
	v_mov_b32_e32 v17, v19
	v_sub_u32_e32 v19, v16, v26
	v_cmp_ge_u32_e64 s[0:1], v16, v26
	v_cmp_le_u32_e32 vcc, s34, v19
	v_subrev_u32_e32 v26, s34, v19
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v27, s34, v19
	v_cndmask_b32_e32 v19, v19, v26, vcc
	v_cndmask_b32_e64 v19, v27, v19, s[0:1]
	v_xor_b32_e32 v19, v19, v20
	v_sub_u32_e32 v19, v19, v20
	v_cmp_ne_u32_e32 vcc, 0, v19
	v_mov_b32_e32 v27, v17
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v26, v16
BB1_20:                                 ; %Flow
                                        ;   in Loop: Header=BB1_18 Depth=4
	s_or_saveexec_b64 s[36:37], s[36:37]
	v_mov_b32_e32 v37, v25
	v_mov_b32_e32 v40, v27
	v_mov_b32_e32 v19, s31
	v_mov_b32_e32 v31, s34
	v_mov_b32_e32 v17, v20
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v34, v21
	v_mov_b32_e32 v35, v23
	v_mov_b32_e32 v38, v22
	v_mov_b32_e32 v29, v18
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v39, v26
	s_xor_b64 exec, exec, s[36:37]
	s_cbranch_execz BB1_22
; %bb.21:                               ; %._crit_edge
                                        ;   in Loop: Header=BB1_18 Depth=4
	s_ashr_i32 s35, s16, 31
	s_add_i32 s38, s16, s35
	s_xor_b32 s38, s38, s35
	v_cvt_f32_u32_e32 v19, s38
	v_ashrrev_i32_e32 v17, 31, v15
	v_mov_b32_e32 v30, 0
	v_add_u32_e32 v28, v15, v17
	v_rcp_iflag_f32_e32 v19, v19
	v_mov_b32_e32 v33, v30
	v_xor_b32_e32 v32, v28, v17
	v_mov_b32_e32 v40, v33
	v_mul_f32_e32 v19, s24, v19
	v_cvt_u32_f32_e32 v29, v19
	v_mov_b32_e32 v37, v30
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v19, s35
	v_mul_lo_u32 v34, v29, s38
	v_mul_hi_u32 v35, v29, s38
	v_mov_b32_e32 v31, s38
	v_mov_b32_e32 v36, v29
	v_sub_u32_e32 v38, 0, v34
	v_mov_b32_e32 v39, v32
BB1_22:                                 ; %Flow160
                                        ;   in Loop: Header=BB1_18 Depth=4
	s_or_b64 exec, exec, s[36:37]
	v_mov_b32_e32 v30, s34
	v_mov_b32_e32 v33, s31
	v_mov_b32_e32 v28, 1
	s_and_saveexec_b64 s[34:35], s[0:1]
	s_cbranch_execz BB1_24
; %bb.23:                               ;   in Loop: Header=BB1_18 Depth=4
	v_mov_b32_e32 v26, v39
	v_mov_b32_e32 v24, v36
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, v40
	v_mov_b32_e32 v22, v38
	v_mov_b32_e32 v23, v35
	v_mov_b32_e32 v21, v34
	v_mov_b32_e32 v25, v37
	v_mov_b32_e32 v18, v29
	v_mov_b32_e32 v30, v31
	v_mov_b32_e32 v16, v32
	v_mov_b32_e32 v33, v19
	v_mov_b32_e32 v20, v17
BB1_24:                                 ;   in Loop: Header=BB1_18 Depth=4
	s_or_b64 exec, exec, s[34:35]
	v_cmp_eq_u32_e32 vcc, 0, v23
	v_cndmask_b32_e32 v17, v21, v22, vcc
	v_mul_lo_u32 v19, v17, v25
	v_mul_hi_u32 v17, v17, v24
	v_add_u32_e32 v17, v17, v19
	v_add_u32_e32 v19, v18, v17
	v_sub_u32_e32 v17, v18, v17
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_mul_lo_u32 v18, v17, v27
	v_mul_hi_u32 v17, v17, v26
	v_xor_b32_e32 v19, v20, v33
	v_add_u32_e32 v17, v17, v18
	v_mul_lo_u32 v18, v17, v30
	v_add_u32_e32 v20, 1, v17
	v_add_u32_e32 v21, -1, v17
	v_sub_u32_e32 v22, v16, v18
	v_cmp_ge_u32_e32 vcc, v16, v18
	v_cmp_ge_u32_e64 s[0:1], v22, v30
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v16, v17, v20, s[0:1]
	v_cndmask_b32_e32 v16, v21, v16, vcc
	v_xor_b32_e32 v16, v16, v19
	v_sub_u32_e32 v16, v16, v19
	v_cmp_gt_i32_e32 vcc, s14, v16
	v_cndmask_b32_e32 v17, 0, v28, vcc
	v_and_b32_e32 v17, v17, v13
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_17
; %bb.25:                               ;   in Loop: Header=BB1_18 Depth=4
	v_add_u32_e32 v16, v16, v14
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshlrev_b64 v[16:17], 2, v[16:17]
	s_ashr_i32 s31, s30, 31
	v_add_co_u32_e32 v16, vcc, v5, v16
	s_lshl_b64 s[34:35], s[30:31], 2
	v_addc_co_u32_e32 v17, vcc, v6, v17, vcc
	v_mov_b32_e32 v19, s35
	v_add_co_u32_e32 v18, vcc, s34, v3
	v_addc_co_u32_e32 v19, vcc, v4, v19, vcc
	global_load_dword v18, v[18:19], off
	global_load_dword v16, v[16:17], off
	s_waitcnt vmcnt(1)
	v_cvt_f64_f32_e32 v[18:19], v18
	s_waitcnt vmcnt(0)
	v_cvt_f64_f32_e32 v[16:17], v16
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[9:10], v[16:17], v[18:19], v[9:10]
	s_branch BB1_17
BB1_26:                                 ; %Flow168
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_bwd_nchw_float
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 41
		.amdhsa_next_free_sgpr 39
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	naive_conv_bwd_nchw_float, .Lfunc_end1-naive_conv_bwd_nchw_float
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2556
; NumSgprs: 41
; NumVgprs: 41
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 41
; NumVGPRsForWavesPerEU: 41
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_wrw_nchw_float ; -- Begin function naive_conv_wrw_nchw_float
	.globl	naive_conv_wrw_nchw_float
	.p2align	8
	.type	naive_conv_wrw_nchw_float,@function
naive_conv_wrw_nchw_float:               ; @naive_conv_wrw_nchw_float
naive_conv_wrw_nchw_float$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s22, s21
	s_mul_i32 s24, s7, s12
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_15
; %bb.1:                                ; %.lr.ph173
	s_ashr_i32 s0, s11, 31
	s_add_i32 s1, s11, s0
	s_xor_b32 s35, s1, s0
	v_cvt_f32_u32_e32 v1, s35
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s1, s6, 31
	s_add_i32 s30, s6, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s30, s30, s1
	s_xor_b32 s33, s1, s0
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	s_load_dwordx2 s[26:27], s[4:5], 0x8
	s_load_dwordx2 s[28:29], s[4:5], 0x10
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mul_i32 s31, s9, s8
	s_ashr_i32 s4, s12, 31
	s_mul_i32 s4, s31, s4
	v_mul_lo_u32 v2, v1, s35
	v_mul_hi_u32 v3, v1, s35
	s_mul_hi_u32 s34, s31, s12
	s_mul_i32 s31, s31, s12
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_mul_hi_i32 s5, s9, s8
	s_mul_i32 s5, s5, s12
	s_sub_i32 s20, 0, s20
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_mul_hi_u32 v1, v1, s30
	v_mul_lo_u32 v2, v1, s35
	v_add_u32_e32 v3, 1, v1
	v_add_u32_e32 v4, -1, v1
	v_sub_u32_e32 v5, s30, v2
	v_cmp_ge_u32_e32 vcc, s30, v2
	v_cmp_le_u32_e64 s[0:1], s35, v5
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	v_xor_b32_e32 v1, s33, v1
	v_subrev_u32_e32 v5, s33, v1
	v_mul_lo_u32 v2, v5, s11
	v_ashrrev_i32_e32 v1, 31, v5
	v_mul_lo_u32 v4, s31, v1
	v_mul_hi_u32 v6, s31, v5
	s_add_i32 s0, s34, s4
	v_sub_u32_e32 v3, s6, v2
	s_add_i32 s0, s0, s5
	v_add_u32_e32 v2, v6, v4
	v_ashrrev_i32_e32 v4, 31, v3
	v_mul_lo_u32 v7, s0, v5
	v_mad_i64_i32 v[3:4], s[0:1], v5, s11, v[3:4]
	v_mul_lo_u32 v1, s31, v5
	s_mul_i32 s1, s21, s12
	s_ashr_i32 s4, s22, 31
	s_mul_hi_i32 s0, s21, s12
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s3
	s_mul_i32 s3, s1, s4
	s_mul_hi_u32 s5, s1, s22
	s_mul_i32 s1, s1, s22
	s_add_i32 s3, s5, s3
	s_mul_i32 s0, s0, s22
	v_add_u32_e32 v2, v2, v7
	s_add_i32 s3, s3, s0
	v_mul_lo_u32 v7, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	v_mul_lo_u32 v9, s3, v3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_mul_lo_u32 v5, s1, v3
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v8, v7
	s_mul_i32 s1, s14, s13
	v_add_u32_e32 v6, v6, v9
	s_mul_hi_i32 s0, s14, s13
	v_mul_lo_u32 v9, s1, v4
	v_mul_hi_u32 v10, s1, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_mul_lo_u32 v7, s1, v3
	v_add_co_u32_e32 v3, vcc, s26, v5
	v_mov_b32_e32 v8, s27
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 2, v[7:8]
	s_mul_i32 s0, s23, s14
	s_mul_i32 s5, s0, s13
	s_mul_i32 s0, s23, s12
	s_mul_i32 s5, s5, s11
	s_mul_i32 s11, s0, s9
	v_mov_b32_e32 v7, s29
	v_add_co_u32_e32 v5, vcc, s28, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s10, 0
	v_cmp_gt_i32_e64 s[28:29], s13, 0
	v_cmp_gt_i32_e64 s[30:31], s14, 0
	s_sub_i32 s6, 0, s19
	s_mul_i32 s11, s11, s8
	s_mul_i32 s12, s15, s9
	s_mov_b64 s[34:35], 0
	s_branch BB2_3
BB2_2:                                  ; %._crit_edge169
                                        ;   in Loop: Header=BB2_3 Depth=1
	v_mul_lo_u32 v9, v9, s21
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	s_or_b64 s[34:35], vcc, s[34:35]
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s22
	v_cvt_f32_f64_e32 v9, v[10:11]
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v3, v7
	v_addc_co_u32_e64 v8, s[0:1], v4, v8, s[0:1]
	global_store_dword v[7:8], v9, off
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz BB2_15
BB2_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_6 Depth 2
                                        ;       Child Loop BB2_10 Depth 3
                                        ;         Child Loop BB2_13 Depth 4
	s_add_i32 s0, s22, s4
	s_xor_b32 s0, s0, s4
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s21, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s21, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s23, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s23
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s33, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s4, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s23
	v_mul_hi_u32 v12, v9, s23
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s33
	s_xor_b32 s36, s0, s33
	v_cvt_f32_u32_e32 v11, s36
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s36
	v_mul_hi_u32 v13, v10, s36
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s23
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s22
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s23, v14
	v_subrev_u32_e32 v10, s23, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s36
	v_add_u32_e32 v13, s23, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s36, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s33, v9
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v7, v0, v7
	v_subrev_u32_e32 v9, s33, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB2_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB2_3 Depth=1
	v_mul_lo_u32 v12, v8, s17
	v_mul_lo_u32 v10, s8, v9
	v_mul_lo_u32 v11, s18, v7
	s_mov_b32 s2, 0
	s_mov_b32 s3, 0
	v_add3_u32 v10, s6, v12, v10
	v_mul_lo_u32 v13, s9, v10
	v_add_u32_e32 v14, s20, v11
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_branch BB2_6
BB2_5:                                  ; %._crit_edge164
                                        ;   in Loop: Header=BB2_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s5
	s_cmp_eq_u32 s3, s10
	v_add_u32_e32 v13, s11, v13
	s_cbranch_scc1 BB2_2
BB2_6:                                  ; %.preheader
                                        ;   Parent Loop BB2_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_10 Depth 3
                                        ;         Child Loop BB2_13 Depth 4
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz BB2_5
; %bb.7:                                ; %.lr.ph163
                                        ;   in Loop: Header=BB2_6 Depth=2
	s_andn2_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz BB2_5
; %bb.8:                                ; %.lr.ph163.split.us.preheader
                                        ;   in Loop: Header=BB2_6 Depth=2
	s_mov_b32 s23, 0
	v_mov_b32_e32 v15, v13
	s_mov_b32 s33, s2
	s_branch BB2_10
BB2_9:                                  ; %Flow59
                                        ;   in Loop: Header=BB2_10 Depth=3
	s_or_b64 exec, exec, s[36:37]
	s_add_i32 s23, s23, 1
	s_add_i32 s33, s33, s14
	s_cmp_eq_u32 s23, s13
	v_add_u32_e32 v15, s12, v15
	s_cbranch_scc1 BB2_5
BB2_10:                                 ; %.lr.ph163.split.us
                                        ;   Parent Loop BB2_3 Depth=1
                                        ;     Parent Loop BB2_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB2_13 Depth 4
	s_mul_i32 s0, s23, s15
	s_sub_i32 s0, s0, s19
	v_add_u32_e32 v16, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v16
	v_cmp_gt_i32_e64 s[0:1], s8, v16
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz BB2_9
; %bb.11:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB2_10 Depth=3
	v_mov_b32_e32 v16, v14
	s_mov_b32 s38, s33
	s_mov_b32 s40, s14
	s_branch BB2_13
BB2_12:                                 ;   in Loop: Header=BB2_13 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s40, s40, -1
	s_add_i32 s38, s38, 1
	s_cmp_lg_u32 s40, 0
	v_add_u32_e32 v16, s16, v16
	s_cbranch_scc0 BB2_9
BB2_13:                                 ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB2_3 Depth=1
                                        ;     Parent Loop BB2_6 Depth=2
                                        ;       Parent Loop BB2_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v16
	v_cmp_gt_i32_e64 s[0:1], s9, v16
	s_and_b64 s[42:43], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[42:43]
	s_cbranch_execz BB2_12
; %bb.14:                               ;   in Loop: Header=BB2_13 Depth=4
	v_add_u32_e32 v17, v15, v16
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 2, v[17:18]
	s_ashr_i32 s39, s38, 31
	v_add_co_u32_e32 v17, vcc, v1, v17
	s_lshl_b64 s[42:43], s[38:39], 2
	v_addc_co_u32_e32 v18, vcc, v2, v18, vcc
	v_mov_b32_e32 v20, s43
	v_add_co_u32_e32 v19, vcc, s42, v5
	v_addc_co_u32_e32 v20, vcc, v6, v20, vcc
	global_load_dword v19, v[19:20], off
	global_load_dword v17, v[17:18], off
	s_waitcnt vmcnt(1)
	v_cvt_f64_f32_e32 v[19:20], v19
	s_waitcnt vmcnt(0)
	v_cvt_f64_f32_e32 v[17:18], v17
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[17:18], v[19:20], v[10:11]
	s_branch BB2_12
BB2_15:                                 ; %Flow65
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_wrw_nchw_float
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 21
		.amdhsa_next_free_sgpr 44
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end2:
	.size	naive_conv_wrw_nchw_float, .Lfunc_end2-naive_conv_wrw_nchw_float
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 1484
; NumSgprs: 46
; NumVgprs: 21
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 21
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_fwd_ncdhw_float ; -- Begin function naive_conv_fwd_ncdhw_float
	.globl	naive_conv_fwd_ncdhw_float
	.p2align	8
	.type	naive_conv_fwd_ncdhw_float,@function
naive_conv_fwd_ncdhw_float:              ; @naive_conv_fwd_ncdhw_float
naive_conv_fwd_ncdhw_float$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s16, s15
	s_mul_i32 s24, s7, s14
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_18
; %bb.1:                                ; %.lr.ph227
	s_ashr_i32 s0, s12, 31
	s_add_i32 s1, s12, s0
	s_xor_b32 s39, s1, s0
	v_cvt_f32_u32_e32 v1, s39
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s1, s11, 31
	s_ashr_i32 s33, s6, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s3, s11, s1
	s_add_i32 s2, s6, s33
	s_xor_b32 s40, s3, s1
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s38, s2, s33
	s_xor_b32 s2, s33, s0
	s_load_dwordx2 s[36:37], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[26:27], s[4:5], 0x10
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_load_dwordx2 s[4:5], s[4:5], 0x68
	v_mul_lo_u32 v2, v1, s39
	v_mul_hi_u32 v3, v1, s39
	s_mov_b64 s[42:43], 0
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s41, s5, s12
	s_mul_i32 s5, s5, s13
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_cvt_f32_u32_e32 v2, s40
	v_mul_hi_u32 v1, v1, s38
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s39
	v_add_u32_e32 v4, -1, v1
	v_mul_f32_e32 v2, s25, v2
	v_sub_u32_e32 v5, s38, v3
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_ge_u32_e32 vcc, s38, v3
	v_cmp_le_u32_e64 s[0:1], s39, v5
	v_add_u32_e32 v3, 1, v1
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	s_mul_i32 s0, s12, s11
	v_mul_hi_u32 v4, v2, s40
	s_ashr_i32 s11, s0, 31
	v_mul_lo_u32 v3, v2, s40
	s_add_i32 s0, s0, s11
	s_xor_b32 s39, s0, s11
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s39
	v_sub_u32_e32 v6, 0, v3
	v_cndmask_b32_e32 v3, v3, v6, vcc
	v_mul_hi_u32 v3, v3, v2
	v_rcp_iflag_f32_e32 v4, v4
	v_xor_b32_e32 v1, s2, v1
	v_subrev_u32_e32 v1, s2, v1
	v_add_u32_e32 v7, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_mul_f32_e32 v3, s25, v4
	v_cvt_u32_f32_e32 v3, v3
	v_ashrrev_i32_e32 v5, 31, v1
	v_add_u32_e32 v6, v1, v5
	v_cndmask_b32_e32 v2, v2, v7, vcc
	v_mul_lo_u32 v4, v1, s12
	v_mul_lo_u32 v1, v3, s39
	v_mul_hi_u32 v7, v3, s39
	v_xor_b32_e32 v6, v6, v5
	v_mul_hi_u32 v2, v2, v6
	v_sub_u32_e32 v9, 0, v1
	v_cmp_eq_u32_e64 s[0:1], 0, v7
	v_cndmask_b32_e64 v1, v1, v9, s[0:1]
	v_mul_lo_u32 v2, v2, s40
	v_mul_hi_u32 v1, v1, v3
	v_sub_u32_e32 v8, v6, v2
	v_cmp_ge_u32_e64 s[2:3], v6, v2
	v_add_u32_e32 v6, v3, v1
	v_sub_u32_e32 v1, v3, v1
	v_cndmask_b32_e64 v1, v1, v6, s[0:1]
	v_mul_hi_u32 v1, v1, s38
	v_cmp_le_u32_e32 vcc, s40, v8
	v_subrev_u32_e32 v2, s40, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_mul_lo_u32 v3, v1, s39
	v_add_u32_e32 v7, s40, v8
	v_cndmask_b32_e32 v2, v8, v2, vcc
	v_cndmask_b32_e64 v2, v7, v2, s[2:3]
	v_xor_b32_e32 v2, v2, v5
	v_sub_u32_e32 v7, v2, v5
	v_sub_u32_e32 v2, s38, v3
	v_cmp_le_u32_e32 vcc, s39, v2
	v_cmp_ge_u32_e64 s[0:1], s38, v3
	v_add_u32_e32 v3, 1, v1
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v2, -1, v1
	v_cndmask_b32_e32 v1, v1, v3, vcc
	s_xor_b32 s2, s33, s11
	v_cndmask_b32_e64 v1, v2, v1, s[0:1]
	v_xor_b32_e32 v3, s2, v1
	v_mul_hi_i32 v2, v7, s5
	v_mul_lo_u32 v1, v7, s5
	v_subrev_u32_e32 v5, s2, v3
	s_ashr_i32 s2, s10, 31
	v_sub_u32_e32 v3, s6, v4
	v_mad_i64_i32 v[1:2], s[0:1], v5, s13, v[1:2]
	s_mul_i32 s1, s9, s8
	s_mul_hi_i32 s0, s9, s8
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s3, s1, s10
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s10
	s_mul_i32 s1, s1, s10
	s_add_i32 s2, s2, s0
	v_ashrrev_i32_e32 v4, 31, v3
	v_mul_lo_u32 v8, s2, v1
	v_mul_hi_u32 v6, s1, v1
	v_mul_lo_u32 v2, s1, v2
	v_mul_lo_u32 v1, s1, v1
	v_mad_i64_i32 v[3:4], s[0:1], v5, s12, v[3:4]
	s_mul_i32 s1, s30, s13
	s_ashr_i32 s2, s31, 31
	s_mul_hi_i32 s0, s30, s13
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s5, s1, s31
	s_add_i32 s2, s5, s2
	s_mul_i32 s0, s0, s31
	s_ashr_i32 s3, s4, 31
	s_mul_i32 s1, s1, s31
	s_add_i32 s0, s2, s0
	s_mul_i32 s2, s1, s3
	s_mul_hi_u32 s3, s1, s4
	v_add_u32_e32 v2, v6, v2
	s_mul_i32 s1, s1, s4
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s4
	v_add_u32_e32 v2, v2, v8
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v8, s1, v4
	v_mul_hi_u32 v9, s1, v3
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	v_mul_lo_u32 v10, s2, v3
	v_mul_lo_u32 v5, s1, v3
	v_mad_i64_i32 v[3:4], s[0:1], v7, s41, v[3:4]
	s_mul_i32 s1, s15, s14
	s_ashr_i32 s5, s16, 31
	s_mul_hi_i32 s0, s15, s14
	v_mov_b32_e32 v6, s37
	v_add_co_u32_e32 v1, vcc, s36, v1
	s_mul_i32 s2, s1, s5
	s_mul_hi_u32 s3, s1, s16
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v9, v8
	s_mul_i32 s1, s1, s16
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s16
	v_add_u32_e32 v6, v6, v10
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v9, s1, v4
	v_mul_hi_u32 v10, s1, v3
	v_mul_lo_u32 v11, s2, v3
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_mul_lo_u32 v7, s1, v3
	v_add_co_u32_e32 v3, vcc, s34, v5
	v_mov_b32_e32 v8, s35
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 2, v[7:8]
	s_mul_i32 s33, s10, s9
	s_mul_i32 s40, s20, s10
	s_mul_i32 s6, s4, s31
	v_mov_b32_e32 v7, s27
	v_add_co_u32_e32 v5, vcc, s26, v5
	s_sub_i32 s12, 0, s29
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s13, 0
	v_cmp_gt_i32_e64 s[34:35], s30, 0
	v_cmp_gt_i32_e64 s[36:37], s31, 0
	v_cmp_gt_i32_e64 s[38:39], s4, 0
	s_mul_i32 s11, s6, s30
	s_sub_i32 s14, 0, s28
	s_sub_i32 s29, 0, s23
	s_mul_i32 s33, s33, s8
	s_mul_i32 s40, s40, s9
	s_mul_i32 s41, s21, s10
	s_branch BB3_3
BB3_2:                                  ; %._crit_edge223
                                        ;   in Loop: Header=BB3_3 Depth=1
	v_mul_lo_u32 v9, v9, s15
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	s_or_b64 s[42:43], vcc, s[42:43]
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s16
	v_cvt_f32_f64_e32 v9, v[10:11]
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v5, v7
	v_addc_co_u32_e64 v8, s[0:1], v6, v8, s[0:1]
	global_store_dword v[7:8], v9, off
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz BB3_18
BB3_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_6 Depth 2
                                        ;       Child Loop BB3_9 Depth 3
                                        ;         Child Loop BB3_13 Depth 4
                                        ;           Child Loop BB3_16 Depth 5
	s_add_i32 s0, s16, s5
	s_xor_b32 s0, s0, s5
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s15, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s15, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s44, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s44
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s45, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s5, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s44
	v_mul_hi_u32 v12, v9, s44
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s45
	s_xor_b32 s46, s0, s45
	v_cvt_f32_u32_e32 v11, s46
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s46
	v_mul_hi_u32 v13, v10, s46
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s44
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s16
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s44, v14
	v_subrev_u32_e32 v10, s44, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s46
	v_add_u32_e32 v13, s44, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s46, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s45, v9
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v7, v0, v7
	v_subrev_u32_e32 v9, s45, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB3_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB3_3 Depth=1
	v_mul_lo_u32 v12, v9, s17
	v_mul_lo_u32 v13, v8, s18
	v_mul_lo_u32 v15, s19, v7
	s_mov_b32 s2, 0
	v_add_u32_e32 v10, s29, v12
	v_mul_lo_u32 v14, s9, v10
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_subrev_u32_e32 v12, s23, v12
	v_add3_u32 v14, s14, v13, v14
	v_mul_lo_u32 v16, s10, v14
	v_add_u32_e32 v14, s12, v15
	v_subrev_u32_e32 v13, s28, v13
	s_mov_b32 s3, 0
	v_add3_u32 v15, s12, v16, v15
	s_branch BB3_6
BB3_5:                                  ; %Flow85
                                        ;   in Loop: Header=BB3_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s11
	s_cmp_eq_u32 s3, s13
	v_add_u32_e32 v15, s33, v15
	s_cbranch_scc1 BB3_2
BB3_6:                                  ; %.preheader
                                        ;   Parent Loop BB3_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_9 Depth 3
                                        ;         Child Loop BB3_13 Depth 4
                                        ;           Child Loop BB3_16 Depth 5
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz BB3_5
; %bb.7:                                ; %.lr.ph217
                                        ;   in Loop: Header=BB3_6 Depth=2
	s_mov_b32 s44, 0
	v_mov_b32_e32 v16, v15
	s_mov_b32 s45, s2
	s_branch BB3_9
BB3_8:                                  ; %._crit_edge212
                                        ;   in Loop: Header=BB3_9 Depth=3
	s_add_i32 s44, s44, 1
	s_add_i32 s45, s45, s6
	s_cmp_eq_u32 s44, s30
	v_add_u32_e32 v16, s40, v16
	s_cbranch_scc1 BB3_5
BB3_9:                                  ;   Parent Loop BB3_3 Depth=1
                                        ;     Parent Loop BB3_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB3_13 Depth 4
                                        ;           Child Loop BB3_16 Depth 5
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB3_8
; %bb.10:                               ; %.lr.ph211
                                        ;   in Loop: Header=BB3_9 Depth=3
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB3_8
; %bb.11:                               ; %.lr.ph211.split.us.preheader
                                        ;   in Loop: Header=BB3_9 Depth=3
	s_mul_i32 s0, s44, s20
	v_add_u32_e32 v17, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v17
	s_and_b64 s[46:47], vcc, s[0:1]
	s_mov_b32 s48, 0
	v_mov_b32_e32 v17, v16
	s_mov_b32 s49, s45
	s_branch BB3_13
BB3_12:                                 ; %Flow81
                                        ;   in Loop: Header=BB3_13 Depth=4
	s_or_b64 exec, exec, s[50:51]
	s_add_i32 s48, s48, 1
	s_add_i32 s49, s49, s4
	s_cmp_lg_u32 s48, s31
	v_add_u32_e32 v17, s41, v17
	s_cbranch_scc0 BB3_8
BB3_13:                                 ; %.lr.ph211.split.us
                                        ;   Parent Loop BB3_3 Depth=1
                                        ;     Parent Loop BB3_6 Depth=2
                                        ;       Parent Loop BB3_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB3_16 Depth 5
	s_mul_i32 s0, s48, s21
	v_add_u32_e32 v18, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s9, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[50:51], s[0:1]
	s_cbranch_execz BB3_12
; %bb.14:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB3_13 Depth=4
	s_mov_b32 s52, 0
	s_mov_b32 s54, s49
	s_mov_b32 s53, s4
	s_branch BB3_16
BB3_15:                                 ;   in Loop: Header=BB3_16 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s53, s53, -1
	s_add_i32 s54, s54, 1
	s_add_i32 s52, s52, s22
	s_cmp_lg_u32 s53, 0
	s_cbranch_scc0 BB3_12
BB3_16:                                 ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB3_3 Depth=1
                                        ;     Parent Loop BB3_6 Depth=2
                                        ;       Parent Loop BB3_9 Depth=3
                                        ;         Parent Loop BB3_13 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_add_u32_e32 v18, s52, v14
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s10, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 s[56:57], s[46:47], s[0:1]
	s_and_saveexec_b64 s[0:1], s[56:57]
	s_cbranch_execz BB3_15
; %bb.17:                               ;   in Loop: Header=BB3_16 Depth=5
	v_add_u32_e32 v18, s52, v17
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshlrev_b64 v[18:19], 2, v[18:19]
	s_ashr_i32 s55, s54, 31
	v_add_co_u32_e32 v18, vcc, v1, v18
	s_lshl_b64 s[56:57], s[54:55], 2
	v_addc_co_u32_e32 v19, vcc, v2, v19, vcc
	v_mov_b32_e32 v21, s57
	v_add_co_u32_e32 v20, vcc, s56, v3
	v_addc_co_u32_e32 v21, vcc, v4, v21, vcc
	global_load_dword v20, v[20:21], off
	global_load_dword v18, v[18:19], off
	s_waitcnt vmcnt(1)
	v_cvt_f64_f32_e32 v[20:21], v20
	s_waitcnt vmcnt(0)
	v_cvt_f64_f32_e32 v[18:19], v18
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[18:19], v[20:21], v[10:11]
	s_branch BB3_15
BB3_18:                                 ; %Flow89
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_fwd_ncdhw_float
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 58
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	naive_conv_fwd_ncdhw_float, .Lfunc_end3-naive_conv_fwd_ncdhw_float
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2016
; NumSgprs: 60
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 60
; NumVGPRsForWavesPerEU: 22
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_bwd_ncdhw_float ; -- Begin function naive_conv_bwd_ncdhw_float
	.globl	naive_conv_bwd_ncdhw_float
	.p2align	8
	.type	naive_conv_bwd_ncdhw_float,@function
naive_conv_bwd_ncdhw_float:              ; @naive_conv_bwd_ncdhw_float
naive_conv_bwd_ncdhw_float$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s10, s9
	s_mul_i32 s24, s7, s8
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB4_35
; %bb.1:                                ; %.lr.ph244
	s_ashr_i32 s33, s13, 31
	s_add_i32 s0, s13, s33
	s_xor_b32 s3, s0, s33
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s38, s6, 31
	s_ashr_i32 s0, s11, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s1, s6, s38
	s_add_i32 s2, s11, s0
	s_xor_b32 s40, s2, s0
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s39, s1, s38
	s_xor_b32 s2, s38, s33
	s_load_dwordx2 s[36:37], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[26:27], s[4:5], 0x10
	v_mul_lo_u32 v2, v1, s3
	v_mul_hi_u32 v3, v1, s3
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_load_dwordx2 s[4:5], s[4:5], 0x68
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v11, s37
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	v_cvt_f32_u32_e32 v3, s40
	s_mul_i32 s42, s5, s13
	v_add_u32_e32 v4, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s39
	v_rcp_iflag_f32_e32 v3, v3
	v_mul_lo_u32 v2, v1, s3
	v_add_u32_e32 v4, 1, v1
	v_add_u32_e32 v5, -1, v1
	v_sub_u32_e32 v6, s39, v2
	v_cmp_ge_u32_e32 vcc, s39, v2
	v_mul_f32_e32 v2, s25, v3
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_le_u32_e64 s[0:1], s3, v6
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	s_mul_i32 s0, s13, s11
	v_mul_hi_u32 v4, v2, s40
	s_ashr_i32 s11, s0, 31
	v_mul_lo_u32 v3, v2, s40
	s_add_i32 s0, s0, s11
	s_xor_b32 s41, s0, s11
	v_cndmask_b32_e32 v1, v5, v1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s41
	v_sub_u32_e32 v7, 0, v3
	v_cndmask_b32_e32 v3, v3, v7, vcc
	v_xor_b32_e32 v1, s2, v1
	v_mul_hi_u32 v3, v3, v2
	v_subrev_u32_e32 v1, s2, v1
	v_rcp_iflag_f32_e32 v4, v4
	v_ashrrev_i32_e32 v6, 31, v1
	v_mul_lo_u32 v5, v1, s13
	v_add_u32_e32 v1, v1, v6
	v_xor_b32_e32 v7, v1, v6
	v_add_u32_e32 v1, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_cndmask_b32_e32 v1, v2, v1, vcc
	v_mul_f32_e32 v2, s25, v4
	v_mul_hi_u32 v1, v1, v7
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s40
	v_sub_u32_e32 v1, s6, v5
	v_mul_lo_u32 v4, v2, s41
	v_mul_hi_u32 v5, v2, s41
	v_sub_u32_e32 v8, v7, v3
	v_cmp_ge_u32_e64 s[2:3], v7, v3
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v5
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_hi_u32 v4, v4, v2
	v_cmp_le_u32_e32 vcc, s40, v8
	v_subrev_u32_e32 v3, s40, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_add_u32_e32 v7, v2, v4
	v_sub_u32_e32 v2, v2, v4
	v_cndmask_b32_e64 v2, v2, v7, s[0:1]
	v_mul_hi_u32 v2, v2, s39
	v_add_u32_e32 v5, s40, v8
	v_cndmask_b32_e32 v3, v8, v3, vcc
	v_cndmask_b32_e64 v3, v5, v3, s[2:3]
	v_mul_lo_u32 v4, v2, s41
	v_xor_b32_e32 v3, v3, v6
	v_sub_u32_e32 v7, v3, v6
	s_xor_b32 s2, s38, s11
	v_sub_u32_e32 v3, s39, v4
	v_cmp_le_u32_e32 vcc, s41, v3
	v_cmp_ge_u32_e64 s[0:1], s39, v4
	v_add_u32_e32 v4, 1, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v3, -1, v2
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e64 v2, v3, v2, s[0:1]
	v_xor_b32_e32 v5, s2, v2
	v_ashrrev_i32_e32 v2, 31, v1
	v_mad_i64_i32 v[3:4], s[0:1], v7, s42, v[1:2]
	v_subrev_u32_e32 v8, s2, v5
	s_ashr_i32 s6, s10, 31
	v_cmp_gt_i32_e64 s[38:39], s4, 0
	v_mad_i64_i32 v[3:4], s[0:1], v8, s13, v[3:4]
	s_mul_i32 s1, s9, s8
	s_mul_hi_i32 s0, s9, s8
	s_mul_i32 s2, s1, s6
	s_mul_hi_u32 s3, s1, s10
	s_mul_i32 s1, s1, s10
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s10
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v4, s1, v4
	v_mul_hi_u32 v5, s1, v3
	v_mul_lo_u32 v6, s2, v3
	v_mul_lo_u32 v3, s1, v3
	s_ashr_i32 s1, s4, 31
	v_add_u32_e32 v4, v5, v4
	v_mul_lo_u32 v5, v8, s12
	v_add_u32_e32 v4, v4, v6
	v_mul_hi_i32 v6, v8, s12
	s_mul_i32 s3, s31, s30
	v_mul_lo_u32 v9, v5, s33
	v_mul_hi_u32 v10, v5, s13
	v_mul_lo_u32 v5, v5, s13
	v_mul_lo_u32 v6, v6, s13
	s_mul_hi_i32 s2, s31, s30
	v_add_u32_e32 v9, v10, v9
	s_mul_i32 s0, s5, s12
	v_add_u32_e32 v6, v9, v6
	v_add_co_u32_e32 v1, vcc, v5, v1
	s_mul_i32 s1, s3, s1
	s_mul_hi_u32 s5, s3, s4
	s_mul_i32 s3, s3, s4
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	s_add_i32 s1, s5, s1
	s_mul_i32 s2, s2, s4
	s_add_i32 s1, s1, s2
	v_mul_lo_u32 v6, s3, v2
	v_mul_hi_u32 v9, s3, v1
	v_mul_lo_u32 v10, s1, v1
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_mul_lo_u32 v5, s3, v1
	v_add_co_u32_e32 v1, vcc, s36, v3
	v_add_u32_e32 v3, v9, v6
	v_add_u32_e32 v6, v3, v10
	v_addc_co_u32_e32 v2, vcc, v11, v4, vcc
	v_lshlrev_b64 v[3:4], 2, v[5:6]
	v_mul_hi_i32 v6, v7, s0
	v_mul_lo_u32 v5, v7, s0
	s_ashr_i32 s2, s16, 31
	v_mov_b32_e32 v7, s35
	v_add_co_u32_e32 v3, vcc, s34, v3
	v_mad_i64_i32 v[5:6], s[0:1], v8, s12, v[5:6]
	s_mul_i32 s1, s15, s14
	s_mul_hi_i32 s0, s15, s14
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s3, s1, s16
	s_mul_i32 s1, s1, s16
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s16
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v6, s1, v6
	v_mul_hi_u32 v8, s1, v5
	v_mul_lo_u32 v9, s2, v5
	v_mul_lo_u32 v5, s1, v5
	v_addc_co_u32_e32 v4, vcc, v7, v4, vcc
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_mul_i32 s5, s4, s31
	s_mul_i32 s8, s5, s30
	v_mov_b32_e32 v7, s27
	v_add_co_u32_e32 v5, vcc, s26, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s12, 0
	v_cmp_gt_i32_e64 s[34:35], s30, 0
	v_cmp_gt_i32_e64 s[36:37], s31, 0
	s_mul_i32 s8, s8, s13
	s_mov_b64 s[40:41], 0
	s_branch BB4_3
BB4_2:                                  ; %Flow245
                                        ;   in Loop: Header=BB4_3 Depth=1
	v_mul_lo_u32 v9, v9, s9
	v_sub_u32_e32 v7, v0, v7
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s10
	v_cvt_f32_f64_e32 v9, v[10:11]
	s_or_b64 s[40:41], vcc, s[40:41]
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v1, v7
	v_addc_co_u32_e64 v8, s[0:1], v2, v8, s[0:1]
	global_store_dword v[7:8], v9, off
	s_andn2_b64 exec, exec, s[40:41]
	s_cbranch_execz BB4_35
BB4_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_6 Depth 2
                                        ;       Child Loop BB4_9 Depth 3
                                        ;         Child Loop BB4_19 Depth 4
                                        ;           Child Loop BB4_27 Depth 5
	s_add_i32 s0, s10, s6
	s_xor_b32 s0, s0, s6
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s9, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s9, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s11, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s11
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s13, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s6, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s11
	v_mul_hi_u32 v12, v9, s11
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s13
	s_xor_b32 s33, s0, s13
	v_cvt_f32_u32_e32 v11, s33
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s33
	v_mul_hi_u32 v13, v10, s33
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s11
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s10
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s11, v14
	v_subrev_u32_e32 v10, s11, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s33
	v_add_u32_e32 v13, s11, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s33, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s13, v9
	v_mov_b32_e32 v10, 0
	v_subrev_u32_e32 v9, s13, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB4_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB4_3 Depth=1
	v_add_u32_e32 v10, s29, v0
	v_sub_u32_e32 v14, v10, v7
	v_mov_b32_e32 v10, 0
	v_add_u32_e32 v12, s23, v9
	v_add_u32_e32 v13, s28, v8
	s_mov_b32 s2, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s3, 0
	s_branch BB4_6
BB4_5:                                  ; %Flow243
                                        ;   in Loop: Header=BB4_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s8
	s_cmp_eq_u32 s3, s12
	s_cbranch_scc1 BB4_2
BB4_6:                                  ; %.preheader
                                        ;   Parent Loop BB4_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB4_9 Depth 3
                                        ;         Child Loop BB4_19 Depth 4
                                        ;           Child Loop BB4_27 Depth 5
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz BB4_5
; %bb.7:                                ; %.lr.ph234
                                        ;   in Loop: Header=BB4_6 Depth=2
	s_mul_i32 s11, s3, s14
	s_mov_b32 s13, 0
	s_mov_b32 s33, s2
	s_branch BB4_9
BB4_8:                                  ; %._crit_edge229
                                        ;   in Loop: Header=BB4_9 Depth=3
	s_add_i32 s13, s13, 1
	s_add_i32 s33, s33, s5
	s_cmp_eq_u32 s13, s30
	s_cbranch_scc1 BB4_5
BB4_9:                                  ;   Parent Loop BB4_3 Depth=1
                                        ;     Parent Loop BB4_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB4_19 Depth 4
                                        ;           Child Loop BB4_27 Depth 5
	s_mul_i32 s0, s13, s20
	v_subrev_u32_e32 v27, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v27
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $sgpr42
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $sgpr43
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr23_vgpr24
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr25_vgpr26
	s_and_saveexec_b64 s[44:45], vcc
	s_xor_b64 s[44:45], exec, s[44:45]
	s_cbranch_execz BB4_11
; %bb.10:                               ;   in Loop: Header=BB4_9 Depth=3
	s_ashr_i32 s42, s17, 31
	s_add_i32 s0, s17, s42
	s_xor_b32 s43, s0, s42
	v_cvt_f32_u32_e32 v15, s43
	v_ashrrev_i32_e32 v19, 31, v27
	v_rcp_iflag_f32_e32 v15, v15
	v_mul_f32_e32 v15, s25, v15
	v_cvt_u32_f32_e32 v17, v15
	v_mul_lo_u32 v20, v17, s43
	v_mul_hi_u32 v22, v17, s43
	v_sub_u32_e32 v21, 0, v20
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cndmask_b32_e32 v15, v20, v21, vcc
	v_mul_hi_u32 v16, v15, v17
	v_add_u32_e32 v15, v27, v19
	v_xor_b32_e32 v15, v15, v19
	v_add_u32_e32 v18, v17, v16
	v_sub_u32_e32 v16, v17, v16
	v_cndmask_b32_e32 v16, v16, v18, vcc
	v_mul_hi_u32 v16, v16, v15
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v24, v18
	v_mov_b32_e32 v23, v17
	v_mul_lo_u32 v25, v16, s43
	v_mov_b32_e32 v16, v18
	v_sub_u32_e32 v18, v15, v25
	v_cmp_ge_u32_e64 s[0:1], v15, v25
	v_cmp_le_u32_e32 vcc, s43, v18
	v_subrev_u32_e32 v25, s43, v18
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v26, s43, v18
	v_cndmask_b32_e32 v18, v18, v25, vcc
	v_cndmask_b32_e64 v18, v26, v18, s[0:1]
	v_xor_b32_e32 v18, v18, v19
	v_sub_u32_e32 v18, v18, v19
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_mov_b32_e32 v26, v16
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v25, v15
BB4_11:                                 ; %Flow240
                                        ;   in Loop: Header=BB4_9 Depth=3
	s_or_saveexec_b64 s[44:45], s[44:45]
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v40, v26
	v_mov_b32_e32 v28, s42
	v_mov_b32_e32 v31, s43
	v_mov_b32_e32 v16, v19
	v_mov_b32_e32 v32, v15
	v_mov_b32_e32 v34, v20
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v38, v21
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v36, v23
	v_mov_b32_e32 v39, v25
	s_xor_b64 exec, exec, s[44:45]
	s_cbranch_execz BB4_13
; %bb.12:                               ; %._crit_edge117
                                        ;   in Loop: Header=BB4_9 Depth=3
	s_ashr_i32 s46, s17, 31
	s_add_i32 s47, s17, s46
	s_xor_b32 s47, s47, s46
	v_cvt_f32_u32_e32 v18, s47
	v_ashrrev_i32_e32 v16, 31, v27
	v_mov_b32_e32 v30, 0
	v_add_u32_e32 v27, v27, v16
	v_rcp_iflag_f32_e32 v18, v18
	v_mov_b32_e32 v33, v30
	v_xor_b32_e32 v32, v27, v16
	v_mov_b32_e32 v40, v33
	v_mul_f32_e32 v18, s25, v18
	v_cvt_u32_f32_e32 v29, v18
	v_mov_b32_e32 v37, v30
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v28, s46
	v_mul_lo_u32 v34, v29, s47
	v_mul_hi_u32 v35, v29, s47
	v_mov_b32_e32 v31, s47
	v_mov_b32_e32 v36, v29
	v_sub_u32_e32 v38, 0, v34
	v_mov_b32_e32 v39, v32
BB4_13:                                 ; %Flow241
                                        ;   in Loop: Header=BB4_9 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_mov_b32_e32 v27, s43
	v_mov_b32_e32 v30, s42
	v_mov_b32_e32 v18, 1
	s_and_saveexec_b64 s[42:43], s[0:1]
	s_cbranch_execz BB4_15
; %bb.14:                               ;   in Loop: Header=BB4_9 Depth=3
	v_mov_b32_e32 v25, v39
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v26, v40
	v_mov_b32_e32 v21, v38
	v_mov_b32_e32 v22, v35
	v_mov_b32_e32 v20, v34
	v_mov_b32_e32 v24, v37
	v_mov_b32_e32 v17, v29
	v_mov_b32_e32 v27, v31
	v_mov_b32_e32 v15, v32
	v_mov_b32_e32 v30, v28
	v_mov_b32_e32 v19, v16
BB4_15:                                 ;   in Loop: Header=BB4_9 Depth=3
	s_or_b64 exec, exec, s[42:43]
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB4_8
; %bb.16:                               ; %.lr.ph228
                                        ;   in Loop: Header=BB4_9 Depth=3
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB4_8
; %bb.17:                               ; %.lr.ph228.split.us.preheader
                                        ;   in Loop: Header=BB4_9 Depth=3
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cndmask_b32_e32 v16, v20, v21, vcc
	v_mul_lo_u32 v20, v16, v24
	v_mul_hi_u32 v16, v16, v23
	v_xor_b32_e32 v19, v19, v30
	s_mov_b32 s42, 0
	s_mov_b32 s43, s33
	v_add_u32_e32 v16, v16, v20
	v_add_u32_e32 v20, v17, v16
	v_sub_u32_e32 v16, v17, v16
	v_cndmask_b32_e32 v16, v16, v20, vcc
	v_mul_lo_u32 v17, v16, v26
	v_mul_hi_u32 v16, v16, v25
	v_add_u32_e32 v16, v16, v17
	v_mul_lo_u32 v17, v16, v27
	v_add_u32_e32 v20, 1, v16
	v_add_u32_e32 v21, -1, v16
	v_sub_u32_e32 v22, v15, v17
	v_cmp_ge_u32_e32 vcc, v15, v17
	v_cmp_ge_u32_e64 s[0:1], v22, v27
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v15, v16, v20, s[0:1]
	v_cndmask_b32_e32 v15, v21, v15, vcc
	v_xor_b32_e32 v15, v15, v19
	v_sub_u32_e32 v16, v15, v19
	v_add_u32_e32 v15, s11, v16
	v_mul_lo_u32 v15, v15, s15
	v_cmp_gt_i32_e32 vcc, s14, v16
	v_cndmask_b32_e32 v16, 0, v18, vcc
	s_branch BB4_19
BB4_18:                                 ; %._crit_edge.us
                                        ;   in Loop: Header=BB4_19 Depth=4
	s_add_i32 s42, s42, 1
	s_add_i32 s43, s43, s4
	s_cmp_lg_u32 s42, s31
	s_cbranch_scc0 BB4_8
BB4_19:                                 ; %.lr.ph228.split.us
                                        ;   Parent Loop BB4_3 Depth=1
                                        ;     Parent Loop BB4_6 Depth=2
                                        ;       Parent Loop BB4_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB4_27 Depth 5
	s_mul_i32 s0, s42, s21
	v_subrev_u32_e32 v29, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v29
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $sgpr44
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $sgpr45
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr25_vgpr26
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr27_vgpr28
	s_and_saveexec_b64 s[46:47], vcc
	s_xor_b64 s[46:47], exec, s[46:47]
	s_cbranch_execz BB4_21
; %bb.20:                               ;   in Loop: Header=BB4_19 Depth=4
	s_ashr_i32 s44, s18, 31
	s_add_i32 s0, s18, s44
	s_xor_b32 s45, s0, s44
	v_cvt_f32_u32_e32 v17, s45
	v_ashrrev_i32_e32 v21, 31, v29
	v_rcp_iflag_f32_e32 v17, v17
	v_mul_f32_e32 v17, s25, v17
	v_cvt_u32_f32_e32 v19, v17
	v_mul_lo_u32 v22, v19, s45
	v_mul_hi_u32 v24, v19, s45
	v_sub_u32_e32 v23, 0, v22
	v_cmp_eq_u32_e32 vcc, 0, v24
	v_cndmask_b32_e32 v17, v22, v23, vcc
	v_mul_hi_u32 v18, v17, v19
	v_add_u32_e32 v17, v29, v21
	v_xor_b32_e32 v17, v17, v21
	v_add_u32_e32 v20, v19, v18
	v_sub_u32_e32 v18, v19, v18
	v_cndmask_b32_e32 v18, v18, v20, vcc
	v_mul_hi_u32 v18, v18, v17
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v26, v20
	v_mov_b32_e32 v25, v19
	v_mul_lo_u32 v27, v18, s45
	v_mov_b32_e32 v18, v20
	v_sub_u32_e32 v20, v17, v27
	v_cmp_ge_u32_e64 s[0:1], v17, v27
	v_cmp_le_u32_e32 vcc, s45, v20
	v_subrev_u32_e32 v27, s45, v20
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v28, s45, v20
	v_cndmask_b32_e32 v20, v20, v27, vcc
	v_cndmask_b32_e64 v20, v28, v20, s[0:1]
	v_xor_b32_e32 v20, v20, v21
	v_sub_u32_e32 v20, v20, v21
	v_cmp_ne_u32_e32 vcc, 0, v20
	v_mov_b32_e32 v28, v18
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v27, v17
BB4_21:                                 ; %Flow236
                                        ;   in Loop: Header=BB4_19 Depth=4
	s_or_saveexec_b64 s[46:47], s[46:47]
	v_mov_b32_e32 v38, v26
	v_mov_b32_e32 v41, v28
	v_mov_b32_e32 v20, s44
	v_mov_b32_e32 v32, s45
	v_mov_b32_e32 v18, v21
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v39, v23
	v_mov_b32_e32 v30, v19
	v_mov_b32_e32 v37, v25
	v_mov_b32_e32 v40, v27
	s_xor_b64 exec, exec, s[46:47]
	s_cbranch_execz BB4_23
; %bb.22:                               ; %.lr.ph228.split.us._crit_edge
                                        ;   in Loop: Header=BB4_19 Depth=4
	s_ashr_i32 s48, s18, 31
	s_add_i32 s49, s18, s48
	s_xor_b32 s49, s49, s48
	v_cvt_f32_u32_e32 v20, s49
	v_ashrrev_i32_e32 v18, 31, v29
	v_mov_b32_e32 v31, 0
	v_add_u32_e32 v29, v29, v18
	v_rcp_iflag_f32_e32 v20, v20
	v_mov_b32_e32 v34, v31
	v_xor_b32_e32 v33, v29, v18
	v_mov_b32_e32 v41, v34
	v_mul_f32_e32 v20, s25, v20
	v_cvt_u32_f32_e32 v30, v20
	v_mov_b32_e32 v38, v31
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v20, s48
	v_mul_lo_u32 v35, v30, s49
	v_mul_hi_u32 v36, v30, s49
	v_mov_b32_e32 v32, s49
	v_mov_b32_e32 v37, v30
	v_sub_u32_e32 v39, 0, v35
	v_mov_b32_e32 v40, v33
BB4_23:                                 ; %Flow237
                                        ;   in Loop: Header=BB4_19 Depth=4
	s_or_b64 exec, exec, s[46:47]
	v_mov_b32_e32 v31, s45
	v_mov_b32_e32 v34, s44
	v_mov_b32_e32 v29, 1
	s_and_saveexec_b64 s[44:45], s[0:1]
	s_cbranch_execz BB4_25
; %bb.24:                               ;   in Loop: Header=BB4_19 Depth=4
	v_mov_b32_e32 v27, v40
	v_mov_b32_e32 v25, v37
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, v41
	v_mov_b32_e32 v23, v39
	v_mov_b32_e32 v24, v36
	v_mov_b32_e32 v22, v35
	v_mov_b32_e32 v26, v38
	v_mov_b32_e32 v19, v30
	v_mov_b32_e32 v31, v32
	v_mov_b32_e32 v17, v33
	v_mov_b32_e32 v34, v20
	v_mov_b32_e32 v21, v18
BB4_25:                                 ; %.lr.ph.us
                                        ;   in Loop: Header=BB4_19 Depth=4
	s_or_b64 exec, exec, s[44:45]
	v_cmp_eq_u32_e32 vcc, 0, v24
	v_cndmask_b32_e32 v18, v22, v23, vcc
	v_mul_lo_u32 v20, v18, v26
	v_mul_hi_u32 v18, v18, v25
	s_mov_b32 s44, s43
	s_mov_b32 s46, s4
	v_add_u32_e32 v18, v18, v20
	v_add_u32_e32 v20, v19, v18
	v_sub_u32_e32 v18, v19, v18
	v_cndmask_b32_e32 v18, v18, v20, vcc
	v_mul_lo_u32 v19, v18, v28
	v_mul_hi_u32 v18, v18, v27
	v_xor_b32_e32 v20, v21, v34
	v_add_u32_e32 v18, v18, v19
	v_mul_lo_u32 v19, v18, v31
	v_add_u32_e32 v21, 1, v18
	v_add_u32_e32 v22, -1, v18
	v_sub_u32_e32 v23, v17, v19
	v_cmp_ge_u32_e32 vcc, v17, v19
	v_cmp_ge_u32_e64 s[0:1], v23, v31
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v17, v18, v21, s[0:1]
	v_cndmask_b32_e32 v17, v22, v17, vcc
	v_xor_b32_e32 v17, v17, v20
	v_sub_u32_e32 v18, v17, v20
	v_cmp_gt_i32_e32 vcc, s15, v18
	v_add_u32_e32 v18, v18, v15
	v_mul_lo_u32 v18, v18, s16
	v_cndmask_b32_e32 v17, 0, v29, vcc
	v_and_b32_e32 v17, v17, v16
	v_mov_b32_e32 v19, v14
	s_branch BB4_27
BB4_26:                                 ;   in Loop: Header=BB4_27 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s46, s46, -1
	s_add_i32 s44, s44, 1
	s_cmp_lg_u32 s46, 0
	v_subrev_u32_e32 v19, s22, v19
	s_cbranch_scc0 BB4_18
BB4_27:                                 ;   Parent Loop BB4_3 Depth=1
                                        ;     Parent Loop BB4_6 Depth=2
                                        ;       Parent Loop BB4_9 Depth=3
                                        ;         Parent Loop BB4_19 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_cmp_lt_i32_e32 vcc, -1, v19
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $sgpr45
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $sgpr47
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr28_vgpr29
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr27
                                        ; implicit-def: $vgpr26
                                        ; implicit-def: $vgpr30_vgpr31
	s_and_saveexec_b64 s[48:49], vcc
	s_xor_b64 s[48:49], exec, s[48:49]
	s_cbranch_execz BB4_29
; %bb.28:                               ;   in Loop: Header=BB4_27 Depth=5
	s_ashr_i32 s45, s19, 31
	s_add_i32 s0, s19, s45
	s_xor_b32 s47, s0, s45
	v_cvt_f32_u32_e32 v20, s47
	v_ashrrev_i32_e32 v24, 31, v19
	v_rcp_iflag_f32_e32 v20, v20
	v_mul_f32_e32 v20, s25, v20
	v_cvt_u32_f32_e32 v22, v20
	v_mul_lo_u32 v25, v22, s47
	v_mul_hi_u32 v27, v22, s47
	v_sub_u32_e32 v26, 0, v25
	v_cmp_eq_u32_e32 vcc, 0, v27
	v_cndmask_b32_e32 v20, v25, v26, vcc
	v_mul_hi_u32 v21, v20, v22
	v_add_u32_e32 v20, v19, v24
	v_xor_b32_e32 v20, v20, v24
	v_add_u32_e32 v23, v22, v21
	v_sub_u32_e32 v21, v22, v21
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_mul_hi_u32 v21, v21, v20
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v29, v23
	v_mov_b32_e32 v28, v22
	v_mul_lo_u32 v30, v21, s47
	v_mov_b32_e32 v21, v23
	v_sub_u32_e32 v23, v20, v30
	v_cmp_ge_u32_e64 s[0:1], v20, v30
	v_cmp_le_u32_e32 vcc, s47, v23
	v_subrev_u32_e32 v30, s47, v23
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v31, s47, v23
	v_cndmask_b32_e32 v23, v23, v30, vcc
	v_cndmask_b32_e64 v23, v31, v23, s[0:1]
	v_xor_b32_e32 v23, v23, v24
	v_sub_u32_e32 v23, v23, v24
	v_cmp_ne_u32_e32 vcc, 0, v23
	v_mov_b32_e32 v31, v21
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v30, v20
BB4_29:                                 ; %Flow
                                        ;   in Loop: Header=BB4_27 Depth=5
	s_or_saveexec_b64 s[48:49], s[48:49]
	v_mov_b32_e32 v41, v29
	v_mov_b32_e32 v44, v31
	v_mov_b32_e32 v23, s45
	v_mov_b32_e32 v35, s47
	v_mov_b32_e32 v21, v24
	v_mov_b32_e32 v36, v20
	v_mov_b32_e32 v38, v25
	v_mov_b32_e32 v39, v27
	v_mov_b32_e32 v42, v26
	v_mov_b32_e32 v33, v22
	v_mov_b32_e32 v40, v28
	v_mov_b32_e32 v43, v30
	s_xor_b64 exec, exec, s[48:49]
	s_cbranch_execz BB4_31
; %bb.30:                               ; %._crit_edge
                                        ;   in Loop: Header=BB4_27 Depth=5
	s_ashr_i32 s50, s19, 31
	s_add_i32 s51, s19, s50
	s_xor_b32 s51, s51, s50
	v_cvt_f32_u32_e32 v23, s51
	v_ashrrev_i32_e32 v21, 31, v19
	v_mov_b32_e32 v34, 0
	v_add_u32_e32 v32, v19, v21
	v_rcp_iflag_f32_e32 v23, v23
	v_mov_b32_e32 v37, v34
	v_xor_b32_e32 v36, v32, v21
	v_mov_b32_e32 v44, v37
	v_mul_f32_e32 v23, s25, v23
	v_cvt_u32_f32_e32 v33, v23
	v_mov_b32_e32 v41, v34
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v23, s50
	v_mul_lo_u32 v38, v33, s51
	v_mul_hi_u32 v39, v33, s51
	v_mov_b32_e32 v35, s51
	v_mov_b32_e32 v40, v33
	v_sub_u32_e32 v42, 0, v38
	v_mov_b32_e32 v43, v36
BB4_31:                                 ; %Flow235
                                        ;   in Loop: Header=BB4_27 Depth=5
	s_or_b64 exec, exec, s[48:49]
	v_mov_b32_e32 v34, s47
	v_mov_b32_e32 v37, s45
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[48:49], s[0:1]
	s_cbranch_execz BB4_33
; %bb.32:                               ;   in Loop: Header=BB4_27 Depth=5
	v_mov_b32_e32 v30, v43
	v_mov_b32_e32 v28, v40
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, v44
	v_mov_b32_e32 v26, v42
	v_mov_b32_e32 v27, v39
	v_mov_b32_e32 v25, v38
	v_mov_b32_e32 v29, v41
	v_mov_b32_e32 v22, v33
	v_mov_b32_e32 v34, v35
	v_mov_b32_e32 v20, v36
	v_mov_b32_e32 v37, v23
	v_mov_b32_e32 v24, v21
BB4_33:                                 ;   in Loop: Header=BB4_27 Depth=5
	s_or_b64 exec, exec, s[48:49]
	v_cmp_eq_u32_e32 vcc, 0, v27
	v_cndmask_b32_e32 v21, v25, v26, vcc
	v_mul_lo_u32 v23, v21, v29
	v_mul_hi_u32 v21, v21, v28
	v_add_u32_e32 v21, v21, v23
	v_add_u32_e32 v23, v22, v21
	v_sub_u32_e32 v21, v22, v21
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_mul_lo_u32 v22, v21, v31
	v_mul_hi_u32 v21, v21, v30
	v_xor_b32_e32 v23, v24, v37
	v_add_u32_e32 v21, v21, v22
	v_mul_lo_u32 v22, v21, v34
	v_add_u32_e32 v24, 1, v21
	v_add_u32_e32 v25, -1, v21
	v_sub_u32_e32 v26, v20, v22
	v_cmp_ge_u32_e32 vcc, v20, v22
	v_cmp_ge_u32_e64 s[0:1], v26, v34
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v20, v21, v24, s[0:1]
	v_cndmask_b32_e32 v20, v25, v20, vcc
	v_xor_b32_e32 v20, v20, v23
	v_sub_u32_e32 v20, v20, v23
	v_cmp_gt_i32_e32 vcc, s16, v20
	v_cndmask_b32_e32 v21, 0, v32, vcc
	v_and_b32_e32 v21, v17, v21
	v_cmp_ne_u32_e32 vcc, 0, v21
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB4_26
; %bb.34:                               ;   in Loop: Header=BB4_27 Depth=5
	v_add_u32_e32 v20, v20, v18
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshlrev_b64 v[20:21], 2, v[20:21]
	s_ashr_i32 s45, s44, 31
	v_add_co_u32_e32 v20, vcc, v5, v20
	s_lshl_b64 s[48:49], s[44:45], 2
	v_addc_co_u32_e32 v21, vcc, v6, v21, vcc
	v_mov_b32_e32 v23, s49
	v_add_co_u32_e32 v22, vcc, s48, v3
	v_addc_co_u32_e32 v23, vcc, v4, v23, vcc
	global_load_dword v22, v[22:23], off
	global_load_dword v20, v[20:21], off
	s_waitcnt vmcnt(1)
	v_cvt_f64_f32_e32 v[22:23], v22
	s_waitcnt vmcnt(0)
	v_cvt_f64_f32_e32 v[20:21], v20
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[20:21], v[22:23], v[10:11]
	s_branch BB4_26
BB4_35:                                 ; %Flow247
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_bwd_ncdhw_float
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 45
		.amdhsa_next_free_sgpr 52
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end4:
	.size	naive_conv_bwd_ncdhw_float, .Lfunc_end4-naive_conv_bwd_ncdhw_float
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 3624
; NumSgprs: 54
; NumVgprs: 45
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 45
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_wrw_ncdhw_float ; -- Begin function naive_conv_wrw_ncdhw_float
	.globl	naive_conv_wrw_ncdhw_float
	.p2align	8
	.type	naive_conv_wrw_ncdhw_float,@function
naive_conv_wrw_ncdhw_float:              ; @naive_conv_wrw_ncdhw_float
naive_conv_wrw_ncdhw_float$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_load_dwordx2 s[24:25], s[4:5], 0x68
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s24, s31
	s_mul_i32 s26, s7, s13
	s_mul_i32 s26, s26, s30
	v_cmp_gt_i32_e32 vcc, s26, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB5_18
; %bb.1:                                ; %.lr.ph229
	s_ashr_i32 s0, s12, 31
	s_add_i32 s1, s12, s0
	s_xor_b32 s41, s1, s0
	v_cvt_f32_u32_e32 v1, s41
	s_mov_b32 s27, 0x4f800000
	s_ashr_i32 s1, s6, 31
	s_add_i32 s33, s6, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s33, s33, s1
	s_xor_b32 s40, s1, s0
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[36:37], s[4:5], 0x10
	v_mul_f32_e32 v1, s27, v1
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s5, s10, 31
	s_mul_i32 s39, s9, s8
	s_ashr_i32 s4, s13, 31
	v_mul_lo_u32 v2, v1, s41
	v_mul_hi_u32 v3, v1, s41
	s_mul_i32 s5, s39, s5
	s_mul_hi_i32 s38, s9, s8
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_mul_i32 s38, s38, s10
	s_sub_i32 s29, 0, s29
	s_sub_i32 s42, 0, s28
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_mul_hi_u32 v1, v1, s33
	s_sub_i32 s43, 0, s23
	s_mul_i32 s44, s18, s10
	s_mov_b64 s[46:47], 0
	v_mul_lo_u32 v2, v1, s41
	v_add_u32_e32 v3, 1, v1
	v_add_u32_e32 v4, -1, v1
	v_sub_u32_e32 v5, s33, v2
	v_cmp_ge_u32_e32 vcc, s33, v2
	v_cmp_le_u32_e64 s[0:1], s41, v5
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	v_xor_b32_e32 v1, s40, v1
	v_subrev_u32_e32 v5, s40, v1
	s_mul_hi_u32 s0, s39, s10
	s_mul_i32 s1, s39, s10
	s_add_i32 s0, s0, s5
	v_mul_lo_u32 v2, v5, s12
	v_ashrrev_i32_e32 v1, 31, v5
	s_mul_i32 s4, s1, s4
	s_mul_hi_u32 s5, s1, s13
	s_mul_i32 s1, s1, s13
	v_mul_lo_u32 v4, s1, v1
	v_mul_hi_u32 v6, s1, v5
	s_add_i32 s0, s0, s38
	v_sub_u32_e32 v3, s6, v2
	s_add_i32 s4, s5, s4
	v_add_u32_e32 v2, v6, v4
	s_mul_i32 s0, s0, s13
	v_ashrrev_i32_e32 v4, 31, v3
	s_add_i32 s4, s4, s0
	v_mul_lo_u32 v1, s1, v5
	v_mad_i64_i32 v[3:4], s[0:1], v5, s12, v[3:4]
	v_mul_lo_u32 v7, s4, v5
	s_mul_i32 s1, s30, s13
	s_ashr_i32 s4, s31, 31
	s_mul_hi_i32 s0, s30, s13
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s3
	s_mul_i32 s3, s1, s4
	s_mul_hi_u32 s6, s1, s31
	s_ashr_i32 s5, s24, 31
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s31
	s_mul_i32 s1, s1, s31
	s_add_i32 s0, s3, s0
	s_mul_i32 s3, s1, s5
	s_mul_hi_u32 s6, s1, s24
	v_add_u32_e32 v2, v2, v7
	s_mul_i32 s1, s1, s24
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s24
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	s_add_i32 s3, s3, s0
	v_mul_lo_u32 v7, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_mul_lo_u32 v9, s3, v3
	v_add_co_u32_e32 v1, vcc, s2, v1
	s_mul_i32 s2, s15, s14
	s_ashr_i32 s0, s16, 31
	v_mul_lo_u32 v5, s1, v3
	s_mul_hi_i32 s1, s15, s14
	s_mul_i32 s0, s2, s0
	s_mul_hi_u32 s3, s2, s16
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v8, v7
	s_mul_i32 s2, s2, s16
	s_add_i32 s0, s3, s0
	s_mul_i32 s1, s1, s16
	v_add_u32_e32 v6, v6, v9
	s_add_i32 s0, s0, s1
	v_mul_lo_u32 v9, s2, v4
	v_mul_hi_u32 v10, s2, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_mul_lo_u32 v7, s2, v3
	s_mul_i32 s0, s25, s16
	s_mul_i32 s0, s0, s15
	v_add_co_u32_e32 v3, vcc, s34, v5
	v_mov_b32_e32 v8, s35
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	s_mul_i32 s0, s0, s14
	v_lshlrev_b64 v[5:6], 2, v[7:8]
	s_mul_i32 s12, s0, s12
	s_mul_i32 s0, s25, s13
	s_mul_i32 s0, s0, s10
	s_mul_i32 s13, s0, s9
	s_mul_i32 s25, s17, s10
	v_mov_b32_e32 v7, s37
	v_add_co_u32_e32 v5, vcc, s36, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	s_mul_i32 s6, s7, s30
	v_cmp_gt_i32_e64 s[34:35], s11, 0
	v_cmp_gt_i32_e64 s[36:37], s14, 0
	v_cmp_gt_i32_e64 s[38:39], s15, 0
	v_cmp_gt_i32_e64 s[40:41], s16, 0
	s_mul_i32 s33, s16, s15
	s_mul_i32 s13, s13, s8
	s_mul_i32 s25, s25, s9
	s_branch BB5_3
BB5_2:                                  ; %._crit_edge225
                                        ;   in Loop: Header=BB5_3 Depth=1
	v_mul_lo_u32 v10, v10, s30
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s26, v0
	s_or_b64 s[46:47], vcc, s[46:47]
	v_add_u32_e32 v9, v10, v9
	v_mul_lo_u32 v9, v9, s31
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s24
	v_cvt_f32_f64_e32 v9, v[11:12]
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v3, v7
	v_addc_co_u32_e64 v8, s[0:1], v4, v8, s[0:1]
	global_store_dword v[7:8], v9, off
	s_andn2_b64 exec, exec, s[46:47]
	s_cbranch_execz BB5_18
BB5_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_6 Depth 2
                                        ;       Child Loop BB5_9 Depth 3
                                        ;         Child Loop BB5_13 Depth 4
                                        ;           Child Loop BB5_16 Depth 5
	s_add_i32 s0, s24, s5
	s_xor_b32 s0, s0, s5
	v_cvt_f32_u32_e32 v7, s0
	v_ashrrev_i32_e32 v8, 31, v0
	v_add_u32_e32 v12, v0, v8
	s_add_i32 s1, s31, s4
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s2, s1, s4
	v_cvt_f32_u32_e32 v9, s2
	s_ashr_i32 s45, s7, 31
	v_mul_f32_e32 v7, s27, v7
	v_cvt_u32_f32_e32 v7, v7
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s27, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s5, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s2
	v_mul_hi_u32 v12, v9, s2
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s45
	s_xor_b32 s3, s0, s45
	v_cvt_f32_u32_e32 v11, s3
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_rcp_iflag_f32_e32 v11, v11
	v_mul_hi_u32 v10, v10, v9
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_mul_f32_e32 v11, s27, v11
	v_cvt_u32_f32_e32 v11, v11
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v10, v11, s3
	v_mul_hi_u32 v13, v11, s3
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_xor_b32_e32 v12, v12, v8
	v_sub_u32_e32 v14, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_mul_hi_u32 v9, v9, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v11
	v_mul_lo_u32 v7, v7, s24
	v_mul_lo_u32 v9, v9, s2
	v_add_u32_e32 v14, v11, v10
	v_sub_u32_e32 v10, v11, v10
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v0
	v_sub_u32_e32 v13, v12, v9
	v_cmp_ge_u32_e32 vcc, v12, v9
	v_cmp_le_u32_e64 s[0:1], s2, v13
	v_add_u32_e32 v11, s2, v13
	v_subrev_u32_e32 v9, s2, v13
	s_ashr_i32 s2, s30, 31
	s_add_i32 s48, s30, s2
	s_xor_b32 s48, s48, s2
	v_mul_lo_u32 v12, v10, s3
	v_cvt_f32_u32_e32 v14, s48
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v9, v13, v9, s[0:1]
	v_sub_u32_e32 v13, v0, v12
	v_cmp_le_u32_e64 s[0:1], s3, v13
	v_rcp_iflag_f32_e32 v13, v14
	v_cmp_ge_u32_e64 s[2:3], v0, v12
	v_add_u32_e32 v12, 1, v10
	s_and_b64 s[0:1], s[0:1], s[2:3]
	v_mul_f32_e32 v13, s27, v13
	v_cvt_u32_f32_e32 v13, v13
	v_add_u32_e32 v14, -1, v10
	v_cndmask_b32_e64 v10, v10, v12, s[0:1]
	v_cndmask_b32_e64 v10, v14, v10, s[2:3]
	v_xor_b32_e32 v10, s45, v10
	v_mul_hi_u32 v14, v13, s48
	v_subrev_u32_e32 v10, s45, v10
	v_mul_lo_u32 v12, v13, s48
	s_ashr_i32 s45, s6, 31
	s_add_i32 s2, s6, s45
	s_xor_b32 s49, s2, s45
	v_cmp_eq_u32_e64 s[0:1], 0, v14
	v_cvt_f32_u32_e32 v14, s49
	v_sub_u32_e32 v16, 0, v12
	v_cndmask_b32_e64 v12, v12, v16, s[0:1]
	v_mul_hi_u32 v12, v12, v13
	v_rcp_iflag_f32_e32 v14, v14
	v_ashrrev_i32_e32 v15, 31, v10
	v_add_u32_e32 v10, v10, v15
	v_add_u32_e32 v16, v13, v12
	v_sub_u32_e32 v12, v13, v12
	v_mul_f32_e32 v13, s27, v14
	v_xor_b32_e32 v10, v10, v15
	v_cndmask_b32_e64 v12, v12, v16, s[0:1]
	v_cvt_u32_f32_e32 v13, v13
	v_mul_hi_u32 v12, v12, v10
	v_cndmask_b32_e32 v9, v11, v9, vcc
	v_xor_b32_e32 v9, v9, v8
	v_mul_hi_u32 v14, v13, s49
	v_mul_lo_u32 v11, v12, s48
	v_mul_lo_u32 v12, v13, s49
	v_sub_u32_e32 v8, v9, v8
	v_cmp_eq_u32_e32 vcc, 0, v14
	v_sub_u32_e32 v9, v10, v11
	v_sub_u32_e32 v16, 0, v12
	v_cndmask_b32_e32 v12, v12, v16, vcc
	v_mul_hi_u32 v12, v12, v13
	v_cmp_ge_u32_e64 s[2:3], v10, v11
	v_cmp_le_u32_e64 s[0:1], s48, v9
	v_add_u32_e32 v14, s48, v9
	v_add_u32_e32 v10, v13, v12
	v_sub_u32_e32 v11, v13, v12
	v_cndmask_b32_e32 v10, v11, v10, vcc
	v_mul_hi_u32 v10, v10, v0
	v_subrev_u32_e32 v11, s48, v9
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_mul_lo_u32 v11, v10, s49
	v_cndmask_b32_e64 v9, v14, v9, s[2:3]
	v_xor_b32_e32 v9, v9, v15
	v_sub_u32_e32 v7, v0, v7
	v_sub_u32_e32 v12, v0, v11
	v_cmp_le_u32_e32 vcc, s49, v12
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v10
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v12, -1, v10
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_cndmask_b32_e64 v10, v12, v10, s[0:1]
	v_xor_b32_e32 v10, s45, v10
	v_mov_b32_e32 v11, 0
	v_sub_u32_e32 v9, v9, v15
	v_subrev_u32_e32 v10, s45, v10
	s_andn2_b64 vcc, exec, s[34:35]
	v_mov_b32_e32 v12, 0
	s_cbranch_vccnz BB5_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB5_3 Depth=1
	v_mul_lo_u32 v13, v9, s20
	v_mul_lo_u32 v11, s8, v10
	v_mul_lo_u32 v14, v8, s21
	v_mul_lo_u32 v16, s22, v7
	s_mov_b32 s2, 0
	v_add3_u32 v11, s43, v13, v11
	v_mul_lo_u32 v11, s9, v11
	v_add_u32_e32 v15, s29, v16
	s_mov_b32 s3, 0
	v_add3_u32 v11, s42, v11, v14
	v_mul_lo_u32 v17, s10, v11
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_add3_u32 v16, s29, v17, v16
	s_branch BB5_6
BB5_5:                                  ; %Flow85
                                        ;   in Loop: Header=BB5_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s12
	s_cmp_eq_u32 s3, s11
	v_add_u32_e32 v16, s13, v16
	s_cbranch_scc1 BB5_2
BB5_6:                                  ; %.preheader
                                        ;   Parent Loop BB5_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_9 Depth 3
                                        ;         Child Loop BB5_13 Depth 4
                                        ;           Child Loop BB5_16 Depth 5
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB5_5
; %bb.7:                                ; %.lr.ph219
                                        ;   in Loop: Header=BB5_6 Depth=2
	s_mov_b32 s45, 0
	v_mov_b32_e32 v17, v16
	s_mov_b32 s48, s2
	s_branch BB5_9
BB5_8:                                  ; %._crit_edge214
                                        ;   in Loop: Header=BB5_9 Depth=3
	s_add_i32 s45, s45, 1
	s_add_i32 s48, s48, s33
	s_cmp_eq_u32 s45, s14
	v_add_u32_e32 v17, s25, v17
	s_cbranch_scc1 BB5_5
BB5_9:                                  ;   Parent Loop BB5_3 Depth=1
                                        ;     Parent Loop BB5_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB5_13 Depth 4
                                        ;           Child Loop BB5_16 Depth 5
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB5_8
; %bb.10:                               ; %.lr.ph213
                                        ;   in Loop: Header=BB5_9 Depth=3
	s_andn2_b64 vcc, exec, s[40:41]
	s_cbranch_vccnz BB5_8
; %bb.11:                               ; %.lr.ph213.split.us.preheader
                                        ;   in Loop: Header=BB5_9 Depth=3
	s_mul_i32 s0, s45, s17
	s_sub_i32 s0, s0, s23
	v_add_u32_e32 v18, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s8, v18
	s_and_b64 s[50:51], vcc, s[0:1]
	s_mov_b32 s49, 0
	v_mov_b32_e32 v18, v17
	s_mov_b32 s52, s48
	s_branch BB5_13
BB5_12:                                 ; %Flow81
                                        ;   in Loop: Header=BB5_13 Depth=4
	s_or_b64 exec, exec, s[54:55]
	s_add_i32 s49, s49, 1
	s_add_i32 s52, s52, s16
	s_cmp_lg_u32 s49, s15
	v_add_u32_e32 v18, s44, v18
	s_cbranch_scc0 BB5_8
BB5_13:                                 ; %.lr.ph213.split.us
                                        ;   Parent Loop BB5_3 Depth=1
                                        ;     Parent Loop BB5_6 Depth=2
                                        ;       Parent Loop BB5_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB5_16 Depth 5
	s_mul_i32 s0, s49, s18
	s_sub_i32 s0, s0, s28
	v_add_u32_e32 v19, s0, v14
	v_cmp_lt_i32_e32 vcc, -1, v19
	v_cmp_gt_i32_e64 s[0:1], s9, v19
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 s[0:1], s[50:51], s[0:1]
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz BB5_12
; %bb.14:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB5_13 Depth=4
	s_mov_b32 s53, 0
	s_mov_b32 s56, s52
	s_mov_b32 s58, s16
	s_branch BB5_16
BB5_15:                                 ;   in Loop: Header=BB5_16 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s58, s58, -1
	s_add_i32 s56, s56, 1
	s_add_i32 s53, s53, s19
	s_cmp_lg_u32 s58, 0
	s_cbranch_scc0 BB5_12
BB5_16:                                 ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB5_3 Depth=1
                                        ;     Parent Loop BB5_6 Depth=2
                                        ;       Parent Loop BB5_9 Depth=3
                                        ;         Parent Loop BB5_13 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_add_u32_e32 v19, s53, v15
	v_cmp_lt_i32_e32 vcc, -1, v19
	v_cmp_gt_i32_e64 s[0:1], s10, v19
	s_and_b64 s[60:61], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[60:61]
	s_cbranch_execz BB5_15
; %bb.17:                               ;   in Loop: Header=BB5_16 Depth=5
	v_add_u32_e32 v19, s53, v18
	v_ashrrev_i32_e32 v20, 31, v19
	v_lshlrev_b64 v[19:20], 2, v[19:20]
	s_ashr_i32 s57, s56, 31
	v_add_co_u32_e32 v19, vcc, v1, v19
	s_lshl_b64 s[60:61], s[56:57], 2
	v_addc_co_u32_e32 v20, vcc, v2, v20, vcc
	v_mov_b32_e32 v22, s61
	v_add_co_u32_e32 v21, vcc, s60, v5
	v_addc_co_u32_e32 v22, vcc, v6, v22, vcc
	global_load_dword v21, v[21:22], off
	global_load_dword v19, v[19:20], off
	s_waitcnt vmcnt(1)
	v_cvt_f64_f32_e32 v[21:22], v21
	s_waitcnt vmcnt(0)
	v_cvt_f64_f32_e32 v[19:20], v19
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[11:12], v[19:20], v[21:22], v[11:12]
	s_branch BB5_15
BB5_18:                                 ; %Flow89
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_wrw_ncdhw_float
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 23
		.amdhsa_next_free_sgpr 62
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end5:
	.size	naive_conv_wrw_ncdhw_float, .Lfunc_end5-naive_conv_wrw_ncdhw_float
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2056
; NumSgprs: 64
; NumVgprs: 23
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 64
; NumVGPRsForWavesPerEU: 23
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_fwd_nchw_half ; -- Begin function naive_conv_fwd_nchw_half
	.globl	naive_conv_fwd_nchw_half
	.p2align	8
	.type	naive_conv_fwd_nchw_half,@function
naive_conv_fwd_nchw_half:               ; @naive_conv_fwd_nchw_half
naive_conv_fwd_nchw_half$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s14, s13
	v_cmp_gt_i32_e32 vcc, s7, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB6_15
; %bb.1:                                ; %.lr.ph184
	s_ashr_i32 s0, s11, 31
	s_add_i32 s1, s11, s0
	s_xor_b32 s3, s1, s0
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s24, 0x4f800000
	s_ashr_i32 s1, s10, 31
	s_add_i32 s2, s10, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s31, s2, s1
	v_cvt_f32_u32_e32 v4, s31
	s_ashr_i32 s25, s6, 31
	v_mul_f32_e32 v1, s24, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s1, s6, s25
	s_xor_b32 s30, s1, s25
	s_xor_b32 s2, s25, s0
	v_mul_lo_u32 v2, v1, s3
	v_mul_hi_u32 v3, v1, s3
	s_mul_i32 s34, s23, s11
	s_mul_i32 s23, s23, s12
	v_sub_u32_e32 v5, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v5, vcc
	v_mul_hi_u32 v2, v2, v1
	v_rcp_iflag_f32_e32 v3, v4
	s_load_dwordx2 s[26:27], s[4:5], 0x0
	s_load_dwordx2 s[28:29], s[4:5], 0x8
	s_load_dwordx2 s[4:5], s[4:5], 0x10
	s_sub_i32 s20, 0, s20
	v_add_u32_e32 v4, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s30
	v_mul_f32_e32 v2, s24, v3
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s3
	v_add_u32_e32 v4, 1, v1
	v_add_u32_e32 v5, -1, v1
	v_sub_u32_e32 v6, s30, v3
	v_cmp_ge_u32_e32 vcc, s30, v3
	v_cmp_le_u32_e64 s[0:1], s3, v6
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	s_mul_i32 s0, s11, s10
	v_mul_hi_u32 v4, v2, s31
	s_ashr_i32 s10, s0, 31
	v_mul_lo_u32 v3, v2, s31
	s_add_i32 s0, s0, s10
	s_xor_b32 s33, s0, s10
	v_cndmask_b32_e32 v1, v5, v1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s33
	v_sub_u32_e32 v6, 0, v3
	v_cndmask_b32_e32 v3, v3, v6, vcc
	v_mul_hi_u32 v3, v3, v2
	v_rcp_iflag_f32_e32 v4, v4
	v_xor_b32_e32 v1, s2, v1
	v_subrev_u32_e32 v5, s2, v1
	v_add_u32_e32 v7, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_mul_f32_e32 v3, s24, v4
	v_cvt_u32_f32_e32 v3, v3
	v_ashrrev_i32_e32 v1, 31, v5
	v_add_u32_e32 v6, v5, v1
	v_cndmask_b32_e32 v2, v2, v7, vcc
	v_mul_lo_u32 v4, v3, s33
	v_mul_hi_u32 v7, v3, s33
	v_xor_b32_e32 v6, v6, v1
	v_mul_hi_u32 v2, v2, v6
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v7
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_lo_u32 v2, v2, s31
	v_mul_hi_u32 v4, v4, v3
	v_sub_u32_e32 v8, v6, v2
	v_cmp_ge_u32_e64 s[2:3], v6, v2
	v_add_u32_e32 v6, v3, v4
	v_sub_u32_e32 v3, v3, v4
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	v_mul_hi_u32 v3, v3, s30
	v_cmp_le_u32_e32 vcc, s31, v8
	v_subrev_u32_e32 v2, s31, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_mul_lo_u32 v4, v3, s33
	v_add_u32_e32 v7, s31, v8
	v_cndmask_b32_e32 v2, v8, v2, vcc
	v_cndmask_b32_e64 v2, v7, v2, s[2:3]
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v7, v2, v1
	v_sub_u32_e32 v1, s30, v4
	v_cmp_le_u32_e32 vcc, s33, v1
	v_cmp_ge_u32_e64 s[0:1], s30, v4
	v_add_u32_e32 v2, 1, v3
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v1, -1, v3
	v_cndmask_b32_e32 v2, v3, v2, vcc
	v_cndmask_b32_e64 v3, v1, v2, s[0:1]
	v_mul_hi_i32 v2, v7, s23
	v_mul_lo_u32 v1, v7, s23
	s_xor_b32 s2, s25, s10
	v_xor_b32_e32 v3, s2, v3
	v_subrev_u32_e32 v6, s2, v3
	v_mad_i64_i32 v[1:2], s[0:1], v6, s12, v[1:2]
	v_mul_lo_u32 v3, v5, s11
	s_mul_i32 s2, s9, s8
	s_mul_hi_i32 s0, s9, s8
	v_mul_lo_u32 v2, s2, v2
	v_mul_hi_u32 v4, s2, v1
	v_mul_lo_u32 v5, s0, v1
	v_mul_lo_u32 v1, s2, v1
	v_sub_u32_e32 v3, s6, v3
	v_add_u32_e32 v2, v4, v2
	v_ashrrev_i32_e32 v4, 31, v3
	v_mad_i64_i32 v[3:4], s[0:1], v6, s11, v[3:4]
	v_add_u32_e32 v2, v2, v5
	s_mul_i32 s1, s21, s12
	s_ashr_i32 s3, s22, 31
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_mul_hi_i32 s0, s21, s12
	s_mul_i32 s3, s1, s3
	s_mul_hi_u32 s6, s1, s22
	s_mul_i32 s1, s1, s22
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s22
	s_add_i32 s3, s3, s0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, s27
	v_add_co_u32_e32 v1, vcc, s26, v1
	v_mul_lo_u32 v6, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_mul_lo_u32 v9, s3, v3
	v_addc_co_u32_e32 v2, vcc, v5, v2, vcc
	v_mul_lo_u32 v5, s1, v3
	v_mad_i64_i32 v[3:4], s[0:1], v7, s34, v[3:4]
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	s_mul_hi_i32 s0, s14, s13
	v_mul_lo_u32 v9, s7, v4
	v_mul_hi_u32 v10, s7, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s7, v3
	v_add_co_u32_e32 v3, vcc, s28, v5
	v_mov_b32_e32 v8, s29
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	v_mov_b32_e32 v7, s5
	v_add_co_u32_e32 v5, vcc, s4, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[4:5], s12, 0
	v_cmp_gt_i32_e64 s[10:11], s21, 0
	v_cmp_gt_i32_e64 s[26:27], s22, 0
	s_mul_i32 s3, s22, s21
	s_sub_i32 s6, 0, s19
	s_mul_i32 s13, s17, s9
	s_mov_b64 s[28:29], 0
	s_branch BB6_3
BB6_2:                                  ; %._crit_edge180
                                        ;   in Loop: Header=BB6_3 Depth=1
	v_mul_lo_u32 v7, v7, s14
	v_cvt_f32_f64_e32 v9, v[9:10]
	v_add_u32_e32 v0, 0x100, v0
	v_add_u32_e32 v7, v7, v8
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_cvt_f16_f32_e32 v9, v9
	v_add_co_u32_e32 v7, vcc, v5, v7
	v_addc_co_u32_e32 v8, vcc, v6, v8, vcc
	v_cmp_le_i32_e32 vcc, s7, v0
	s_or_b64 s[28:29], vcc, s[28:29]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz BB6_15
BB6_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_6 Depth 2
                                        ;       Child Loop BB6_10 Depth 3
                                        ;         Child Loop BB6_13 Depth 4
	s_ashr_i32 s0, s14, 31
	s_add_i32 s1, s14, s0
	s_xor_b32 s1, s1, s0
	v_cvt_f32_u32_e32 v7, s1
	v_rcp_iflag_f32_e32 v7, v7
	v_mul_f32_e32 v7, s24, v7
	v_cvt_u32_f32_e32 v7, v7
	v_mul_lo_u32 v8, v7, s1
	v_mul_hi_u32 v9, v7, s1
	v_sub_u32_e32 v10, 0, v8
	v_cmp_eq_u32_e32 vcc, 0, v9
	v_cndmask_b32_e32 v8, v8, v10, vcc
	v_mul_hi_u32 v8, v8, v7
	v_ashrrev_i32_e32 v9, 31, v0
	v_add_u32_e32 v10, v0, v9
	v_xor_b32_e32 v10, v10, v9
	v_add_u32_e32 v11, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_mul_hi_u32 v7, v7, v10
	v_xor_b32_e32 v9, s0, v9
	v_mul_lo_u32 v8, v7, s1
	v_add_u32_e32 v11, 1, v7
	v_add_u32_e32 v12, -1, v7
	v_sub_u32_e32 v13, v10, v8
	v_cmp_ge_u32_e32 vcc, v10, v8
	v_cmp_le_u32_e64 s[0:1], s1, v13
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v11, s[0:1]
	v_cndmask_b32_e32 v7, v12, v7, vcc
	v_xor_b32_e32 v7, v7, v9
	v_sub_u32_e32 v7, v7, v9
	v_mul_lo_u32 v8, v7, s14
	v_mov_b32_e32 v9, 0
	s_andn2_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v8, v0, v8
	s_cbranch_vccnz BB6_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB6_3 Depth=1
	v_mul_lo_u32 v9, v7, s15
	v_mul_lo_u32 v10, s16, v8
	s_mov_b32 s23, 0
	s_mov_b32 s25, 0
	v_subrev_u32_e32 v11, s19, v9
	v_add_u32_e32 v9, s6, v9
	v_mul_lo_u32 v12, s9, v9
	v_add_u32_e32 v13, s20, v10
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	s_branch BB6_6
BB6_5:                                  ; %._crit_edge175
                                        ;   in Loop: Header=BB6_6 Depth=2
	s_add_i32 s25, s25, 1
	s_add_i32 s23, s23, s3
	s_cmp_eq_u32 s25, s12
	v_add_u32_e32 v12, s2, v12
	s_cbranch_scc1 BB6_2
BB6_6:                                  ; %.preheader
                                        ;   Parent Loop BB6_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB6_10 Depth 3
                                        ;         Child Loop BB6_13 Depth 4
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz BB6_5
; %bb.7:                                ; %.lr.ph174
                                        ;   in Loop: Header=BB6_6 Depth=2
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz BB6_5
; %bb.8:                                ; %.lr.ph174.split.us.preheader
                                        ;   in Loop: Header=BB6_6 Depth=2
	s_mov_b32 s30, 0
	v_mov_b32_e32 v14, v12
	s_mov_b32 s31, s23
	s_branch BB6_10
BB6_9:                                  ; %Flow59
                                        ;   in Loop: Header=BB6_10 Depth=3
	s_or_b64 exec, exec, s[34:35]
	s_add_i32 s30, s30, 1
	s_add_i32 s31, s31, s22
	s_cmp_eq_u32 s30, s21
	v_add_u32_e32 v14, s13, v14
	s_cbranch_scc1 BB6_5
BB6_10:                                 ; %.lr.ph174.split.us
                                        ;   Parent Loop BB6_3 Depth=1
                                        ;     Parent Loop BB6_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB6_13 Depth 4
	s_mul_i32 s0, s30, s17
	v_add_u32_e32 v15, s0, v11
	v_cmp_lt_i32_e32 vcc, -1, v15
	v_cmp_gt_i32_e64 s[0:1], s8, v15
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[34:35], s[0:1]
	s_cbranch_execz BB6_9
; %bb.11:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB6_10 Depth=3
	v_mov_b32_e32 v15, v13
	s_mov_b32 s36, s31
	s_mov_b32 s33, s22
	s_branch BB6_13
BB6_12:                                 ;   in Loop: Header=BB6_13 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s33, s33, -1
	s_add_i32 s36, s36, 1
	s_cmp_lg_u32 s33, 0
	v_add_u32_e32 v15, s18, v15
	s_cbranch_scc0 BB6_9
BB6_13:                                 ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB6_3 Depth=1
                                        ;     Parent Loop BB6_6 Depth=2
                                        ;       Parent Loop BB6_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v15
	v_cmp_gt_i32_e64 s[0:1], s9, v15
	s_and_b64 s[38:39], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[38:39]
	s_cbranch_execz BB6_12
; %bb.14:                               ;   in Loop: Header=BB6_13 Depth=4
	v_add_u32_e32 v16, v14, v15
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshlrev_b64 v[16:17], 1, v[16:17]
	s_ashr_i32 s37, s36, 31
	v_add_co_u32_e32 v16, vcc, v1, v16
	s_lshl_b64 s[38:39], s[36:37], 1
	v_addc_co_u32_e32 v17, vcc, v2, v17, vcc
	v_mov_b32_e32 v19, s39
	v_add_co_u32_e32 v18, vcc, s38, v3
	v_addc_co_u32_e32 v19, vcc, v4, v19, vcc
	global_load_ushort v18, v[18:19], off
	global_load_ushort v16, v[16:17], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v18, v18
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v16, v16
	v_cvt_f64_f32_e32 v[18:19], v18
	v_cvt_f64_f32_e32 v[16:17], v16
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[9:10], v[16:17], v[18:19], v[9:10]
	s_branch BB6_12
BB6_15:                                 ; %Flow65
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_fwd_nchw_half
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 40
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end6:
	.size	naive_conv_fwd_nchw_half, .Lfunc_end6-naive_conv_fwd_nchw_half
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 1472
; NumSgprs: 42
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 42
; NumVGPRsForWavesPerEU: 20
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_bwd_nchw_half ; -- Begin function naive_conv_bwd_nchw_half
	.globl	naive_conv_bwd_nchw_half
	.p2align	8
	.type	naive_conv_bwd_nchw_half,@function
naive_conv_bwd_nchw_half:               ; @naive_conv_bwd_nchw_half
naive_conv_bwd_nchw_half$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s9, s8
	v_cmp_gt_i32_e32 vcc, s7, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB7_26
; %bb.1:                                ; %.lr.ph195
	s_ashr_i32 s25, s12, 31
	s_add_i32 s0, s12, s25
	s_xor_b32 s3, s0, s25
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s24, 0x4f800000
	s_ashr_i32 s0, s10, 31
	s_add_i32 s1, s10, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s28, s1, s0
	s_ashr_i32 s26, s6, 31
	v_cvt_f32_u32_e32 v2, s28
	v_mul_f32_e32 v1, s24, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s0, s6, s26
	s_xor_b32 s27, s0, s26
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s3
	v_mul_hi_u32 v4, v1, s3
	s_xor_b32 s2, s26, s25
	v_mul_f32_e32 v2, s24, v2
	v_sub_u32_e32 v5, 0, v3
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_mul_hi_u32 v3, v3, v1
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s30, s23, s12
	s_mul_hi_i32 s8, s9, s8
	v_add_u32_e32 v4, v1, v3
	v_sub_u32_e32 v1, v1, v3
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s27
	v_mul_hi_u32 v3, v2, s28
	v_mul_lo_u32 v5, v2, s28
	v_mul_lo_u32 v4, v1, s3
	v_add_u32_e32 v6, 1, v1
	v_add_u32_e32 v7, -1, v1
	v_sub_u32_e32 v8, s27, v4
	v_cmp_ge_u32_e32 vcc, s27, v4
	v_cmp_le_u32_e64 s[0:1], s3, v8
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v6, s[0:1]
	s_mul_i32 s0, s12, s10
	s_ashr_i32 s10, s0, 31
	s_add_i32 s0, s0, s10
	v_cndmask_b32_e32 v1, v7, v1, vcc
	v_sub_u32_e32 v7, 0, v5
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_xor_b32 s29, s0, s10
	v_cndmask_b32_e32 v3, v5, v7, vcc
	v_cvt_f32_u32_e32 v5, s29
	v_xor_b32_e32 v1, s2, v1
	v_mul_hi_u32 v3, v3, v2
	v_subrev_u32_e32 v1, s2, v1
	v_rcp_iflag_f32_e32 v5, v5
	v_ashrrev_i32_e32 v6, 31, v1
	v_mul_lo_u32 v4, v1, s12
	v_add_u32_e32 v1, v1, v6
	v_xor_b32_e32 v7, v1, v6
	v_add_u32_e32 v1, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_cndmask_b32_e32 v1, v2, v1, vcc
	v_mul_f32_e32 v2, s24, v5
	v_mul_hi_u32 v1, v1, v7
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s28
	v_sub_u32_e32 v1, s6, v4
	v_mul_lo_u32 v4, v2, s29
	v_mul_hi_u32 v5, v2, s29
	v_sub_u32_e32 v8, v7, v3
	v_cmp_ge_u32_e64 s[2:3], v7, v3
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v5
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_hi_u32 v4, v4, v2
	v_cmp_le_u32_e32 vcc, s28, v8
	v_subrev_u32_e32 v3, s28, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_add_u32_e32 v7, v2, v4
	v_sub_u32_e32 v2, v2, v4
	v_cndmask_b32_e64 v2, v2, v7, s[0:1]
	v_mul_hi_u32 v2, v2, s27
	v_add_u32_e32 v5, s28, v8
	v_cndmask_b32_e32 v3, v8, v3, vcc
	v_cndmask_b32_e64 v3, v5, v3, s[2:3]
	v_mul_lo_u32 v4, v2, s29
	v_xor_b32_e32 v3, v3, v6
	v_sub_u32_e32 v5, v3, v6
	s_xor_b32 s2, s26, s10
	v_sub_u32_e32 v3, s27, v4
	v_cmp_le_u32_e32 vcc, s29, v3
	v_cmp_ge_u32_e64 s[0:1], s27, v4
	v_add_u32_e32 v4, 1, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v3, -1, v2
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e64 v6, v3, v2, s[0:1]
	v_ashrrev_i32_e32 v2, 31, v1
	v_mad_i64_i32 v[3:4], s[0:1], v5, s30, v[1:2]
	v_xor_b32_e32 v6, s2, v6
	v_subrev_u32_e32 v7, s2, v6
	s_mul_i32 s6, s23, s11
	v_mad_i64_i32 v[3:4], s[0:1], v7, s12, v[3:4]
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[2:3], s[4:5], 0x8
	s_load_dwordx2 s[4:5], s[4:5], 0x10
	v_cmp_gt_i32_e64 s[26:27], s22, 0
	s_mov_b64 s[28:29], 0
	v_mul_lo_u32 v4, s7, v4
	v_mul_hi_u32 v6, s7, v3
	v_mul_lo_u32 v8, s8, v3
	v_mul_lo_u32 v3, s7, v3
	s_mul_i32 s8, s22, s21
	v_add_u32_e32 v4, v6, v4
	v_mul_lo_u32 v6, v7, s11
	v_add_u32_e32 v4, v4, v8
	v_mul_hi_i32 v8, v7, s11
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	v_mul_lo_u32 v9, v6, s25
	v_mul_hi_u32 v10, v6, s12
	v_mul_lo_u32 v6, v6, s12
	v_mul_lo_u32 v8, v8, s12
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v11, s1
	v_add_u32_e32 v9, v10, v9
	v_add_co_u32_e32 v6, vcc, v6, v1
	v_add_u32_e32 v8, v9, v8
	v_addc_co_u32_e32 v1, vcc, v8, v2, vcc
	v_mul_lo_u32 v8, s8, v1
	v_mul_hi_u32 v9, s8, v6
	v_add_co_u32_e32 v1, vcc, s0, v3
	s_mul_hi_i32 s0, s22, s21
	v_addc_co_u32_e32 v2, vcc, v11, v4, vcc
	v_add_u32_e32 v4, v9, v8
	v_mul_lo_u32 v8, s0, v6
	v_mul_lo_u32 v3, s8, v6
	v_mul_hi_i32 v6, v5, s6
	v_mul_lo_u32 v5, v5, s6
	v_add_u32_e32 v4, v4, v8
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	s_mul_i32 s6, s8, s12
	v_mad_i64_i32 v[5:6], s[0:1], v7, s11, v[5:6]
	s_mul_i32 s1, s14, s13
	s_mul_hi_i32 s0, s14, s13
	v_mov_b32_e32 v7, s3
	v_mul_lo_u32 v6, s1, v6
	v_mul_hi_u32 v8, s1, v5
	v_mul_lo_u32 v9, s0, v5
	v_mul_lo_u32 v5, s1, v5
	v_add_co_u32_e32 v3, vcc, s2, v3
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_addc_co_u32_e32 v4, vcc, v7, v4, vcc
	v_mov_b32_e32 v7, s5
	v_add_co_u32_e32 v5, vcc, s4, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[2:3], s11, 0
	v_cmp_gt_i32_e64 s[4:5], s21, 0
	s_branch BB7_3
BB7_2:                                  ; %Flow166
                                        ;   in Loop: Header=BB7_3 Depth=1
	v_mul_lo_u32 v7, v7, s9
	v_sub_u32_e32 v8, v0, v8
	v_cvt_f32_f64_e32 v9, v[9:10]
	v_add_u32_e32 v0, 0x100, v0
	v_add_u32_e32 v7, v7, v8
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_cvt_f16_f32_e32 v9, v9
	v_add_co_u32_e32 v7, vcc, v1, v7
	v_addc_co_u32_e32 v8, vcc, v2, v8, vcc
	v_cmp_le_i32_e32 vcc, s7, v0
	s_or_b64 s[28:29], vcc, s[28:29]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz BB7_26
BB7_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_6 Depth 2
                                        ;       Child Loop BB7_10 Depth 3
                                        ;         Child Loop BB7_18 Depth 4
	s_ashr_i32 s0, s9, 31
	s_add_i32 s1, s9, s0
	s_xor_b32 s1, s1, s0
	v_cvt_f32_u32_e32 v7, s1
	v_rcp_iflag_f32_e32 v7, v7
	v_mul_f32_e32 v7, s24, v7
	v_cvt_u32_f32_e32 v7, v7
	v_mul_lo_u32 v8, v7, s1
	v_mul_hi_u32 v9, v7, s1
	v_sub_u32_e32 v10, 0, v8
	v_cmp_eq_u32_e32 vcc, 0, v9
	v_cndmask_b32_e32 v8, v8, v10, vcc
	v_mul_hi_u32 v8, v8, v7
	v_ashrrev_i32_e32 v9, 31, v0
	v_add_u32_e32 v10, v0, v9
	v_xor_b32_e32 v10, v10, v9
	v_add_u32_e32 v11, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_mul_hi_u32 v7, v7, v10
	v_xor_b32_e32 v9, s0, v9
	v_mul_lo_u32 v8, v7, s1
	v_add_u32_e32 v11, 1, v7
	v_add_u32_e32 v12, -1, v7
	v_sub_u32_e32 v13, v10, v8
	v_cmp_ge_u32_e32 vcc, v10, v8
	v_cmp_le_u32_e64 s[0:1], s1, v13
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v11, s[0:1]
	v_cndmask_b32_e32 v7, v12, v7, vcc
	v_xor_b32_e32 v7, v7, v9
	v_sub_u32_e32 v7, v7, v9
	v_mul_lo_u32 v8, v7, s9
	v_mov_b32_e32 v9, 0
	s_andn2_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v10, 0
	s_cbranch_vccnz BB7_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB7_3 Depth=1
	v_add_u32_e32 v9, s20, v0
	v_sub_u32_e32 v12, v9, v8
	v_mov_b32_e32 v9, 0
	v_add_u32_e32 v11, s19, v7
	s_mov_b32 s8, 0
	v_mov_b32_e32 v10, 0
	s_mov_b32 s10, 0
	s_branch BB7_6
BB7_5:                                  ; %._crit_edge186
                                        ;   in Loop: Header=BB7_6 Depth=2
	s_add_i32 s10, s10, 1
	s_add_i32 s8, s8, s6
	s_cmp_eq_u32 s10, s11
	s_cbranch_scc1 BB7_2
BB7_6:                                  ; %.preheader
                                        ;   Parent Loop BB7_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB7_10 Depth 3
                                        ;         Child Loop BB7_18 Depth 4
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz BB7_5
; %bb.7:                                ; %.lr.ph185
                                        ;   in Loop: Header=BB7_6 Depth=2
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz BB7_5
; %bb.8:                                ; %.lr.ph185.split.us.preheader
                                        ;   in Loop: Header=BB7_6 Depth=2
	s_mul_i32 s12, s10, s13
	s_mov_b32 s23, 0
	s_mov_b32 s25, s8
	s_branch BB7_10
BB7_9:                                  ; %._crit_edge.us
                                        ;   in Loop: Header=BB7_10 Depth=3
	s_add_i32 s23, s23, 1
	s_add_i32 s25, s25, s22
	s_cmp_eq_u32 s23, s21
	s_cbranch_scc1 BB7_5
BB7_10:                                 ; %.lr.ph185.split.us
                                        ;   Parent Loop BB7_3 Depth=1
                                        ;     Parent Loop BB7_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB7_18 Depth 4
	s_mul_i32 s0, s23, s17
	v_subrev_u32_e32 v25, s0, v11
	v_cmp_lt_i32_e32 vcc, -1, v25
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $sgpr30
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $sgpr31
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr21_vgpr22
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr23_vgpr24
	s_and_saveexec_b64 s[34:35], vcc
	s_xor_b64 s[34:35], exec, s[34:35]
	s_cbranch_execz BB7_12
; %bb.11:                               ;   in Loop: Header=BB7_10 Depth=3
	s_ashr_i32 s30, s15, 31
	s_add_i32 s0, s15, s30
	s_xor_b32 s31, s0, s30
	v_cvt_f32_u32_e32 v13, s31
	v_ashrrev_i32_e32 v17, 31, v25
	v_rcp_iflag_f32_e32 v13, v13
	v_mul_f32_e32 v13, s24, v13
	v_cvt_u32_f32_e32 v15, v13
	v_mul_lo_u32 v18, v15, s31
	v_mul_hi_u32 v20, v15, s31
	v_sub_u32_e32 v19, 0, v18
	v_cmp_eq_u32_e32 vcc, 0, v20
	v_cndmask_b32_e32 v13, v18, v19, vcc
	v_mul_hi_u32 v14, v13, v15
	v_add_u32_e32 v13, v25, v17
	v_xor_b32_e32 v13, v13, v17
	v_add_u32_e32 v16, v15, v14
	v_sub_u32_e32 v14, v15, v14
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_mul_hi_u32 v14, v14, v13
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v21, v15
	v_mul_lo_u32 v23, v14, s31
	v_mov_b32_e32 v14, v16
	v_sub_u32_e32 v16, v13, v23
	v_cmp_ge_u32_e64 s[0:1], v13, v23
	v_cmp_le_u32_e32 vcc, s31, v16
	v_subrev_u32_e32 v23, s31, v16
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v24, s31, v16
	v_cndmask_b32_e32 v16, v16, v23, vcc
	v_cndmask_b32_e64 v16, v24, v16, s[0:1]
	v_xor_b32_e32 v16, v16, v17
	v_sub_u32_e32 v16, v16, v17
	v_cmp_ne_u32_e32 vcc, 0, v16
	v_mov_b32_e32 v24, v14
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v23, v13
BB7_12:                                 ; %Flow161
                                        ;   in Loop: Header=BB7_10 Depth=3
	s_or_saveexec_b64 s[34:35], s[34:35]
	v_mov_b32_e32 v34, v22
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v16, s30
	v_mov_b32_e32 v28, s31
	v_mov_b32_e32 v14, v17
	v_mov_b32_e32 v29, v13
	v_mov_b32_e32 v31, v18
	v_mov_b32_e32 v32, v20
	v_mov_b32_e32 v35, v19
	v_mov_b32_e32 v26, v15
	v_mov_b32_e32 v33, v21
	v_mov_b32_e32 v36, v23
	s_xor_b64 exec, exec, s[34:35]
	s_cbranch_execz BB7_14
; %bb.13:                               ; %.lr.ph185.split.us._crit_edge
                                        ;   in Loop: Header=BB7_10 Depth=3
	s_ashr_i32 s33, s15, 31
	s_add_i32 s36, s15, s33
	s_xor_b32 s36, s36, s33
	v_cvt_f32_u32_e32 v16, s36
	v_ashrrev_i32_e32 v14, 31, v25
	v_mov_b32_e32 v27, 0
	v_add_u32_e32 v25, v25, v14
	v_rcp_iflag_f32_e32 v16, v16
	v_mov_b32_e32 v30, v27
	v_xor_b32_e32 v29, v25, v14
	v_mov_b32_e32 v37, v30
	v_mul_f32_e32 v16, s24, v16
	v_cvt_u32_f32_e32 v26, v16
	v_mov_b32_e32 v34, v27
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v16, s33
	v_mul_lo_u32 v31, v26, s36
	v_mul_hi_u32 v32, v26, s36
	v_mov_b32_e32 v28, s36
	v_mov_b32_e32 v33, v26
	v_sub_u32_e32 v35, 0, v31
	v_mov_b32_e32 v36, v29
BB7_14:                                 ; %Flow162
                                        ;   in Loop: Header=BB7_10 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_mov_b32_e32 v27, s31
	v_mov_b32_e32 v30, s30
	v_mov_b32_e32 v25, 1
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz BB7_16
; %bb.15:                               ;   in Loop: Header=BB7_10 Depth=3
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v21, v33
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, v37
	v_mov_b32_e32 v19, v35
	v_mov_b32_e32 v20, v32
	v_mov_b32_e32 v18, v31
	v_mov_b32_e32 v22, v34
	v_mov_b32_e32 v15, v26
	v_mov_b32_e32 v27, v28
	v_mov_b32_e32 v13, v29
	v_mov_b32_e32 v30, v16
	v_mov_b32_e32 v17, v14
BB7_16:                                 ; %.lr.ph.us
                                        ;   in Loop: Header=BB7_10 Depth=3
	s_or_b64 exec, exec, s[30:31]
	v_cmp_eq_u32_e32 vcc, 0, v20
	v_cndmask_b32_e32 v14, v18, v19, vcc
	v_mul_lo_u32 v16, v14, v22
	v_mul_hi_u32 v14, v14, v21
	s_mov_b32 s30, s25
	s_mov_b32 s33, s22
	v_add_u32_e32 v14, v14, v16
	v_add_u32_e32 v16, v15, v14
	v_sub_u32_e32 v14, v15, v14
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_mul_lo_u32 v15, v14, v24
	v_mul_hi_u32 v14, v14, v23
	v_xor_b32_e32 v16, v17, v30
	v_add_u32_e32 v14, v14, v15
	v_mul_lo_u32 v15, v14, v27
	v_add_u32_e32 v17, 1, v14
	v_add_u32_e32 v18, -1, v14
	v_sub_u32_e32 v19, v13, v15
	v_cmp_ge_u32_e32 vcc, v13, v15
	v_cmp_ge_u32_e64 s[0:1], v19, v27
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v13, v14, v17, s[0:1]
	v_cndmask_b32_e32 v13, v18, v13, vcc
	v_xor_b32_e32 v13, v13, v16
	v_sub_u32_e32 v14, v13, v16
	v_cmp_gt_i32_e32 vcc, s13, v14
	v_add_u32_e32 v14, s12, v14
	v_mul_lo_u32 v14, v14, s14
	v_cndmask_b32_e32 v13, 0, v25, vcc
	v_mov_b32_e32 v15, v12
	s_branch BB7_18
BB7_17:                                 ;   in Loop: Header=BB7_18 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s33, s33, -1
	s_add_i32 s30, s30, 1
	s_cmp_lg_u32 s33, 0
	v_subrev_u32_e32 v15, s18, v15
	s_cbranch_scc0 BB7_9
BB7_18:                                 ;   Parent Loop BB7_3 Depth=1
                                        ;     Parent Loop BB7_6 Depth=2
                                        ;       Parent Loop BB7_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v15
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $sgpr31
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $sgpr34
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr24_vgpr25
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr26_vgpr27
	s_and_saveexec_b64 s[36:37], vcc
	s_xor_b64 s[36:37], exec, s[36:37]
	s_cbranch_execz BB7_20
; %bb.19:                               ;   in Loop: Header=BB7_18 Depth=4
	s_ashr_i32 s31, s16, 31
	s_add_i32 s0, s16, s31
	s_xor_b32 s34, s0, s31
	v_cvt_f32_u32_e32 v16, s34
	v_ashrrev_i32_e32 v20, 31, v15
	v_rcp_iflag_f32_e32 v16, v16
	v_mul_f32_e32 v16, s24, v16
	v_cvt_u32_f32_e32 v18, v16
	v_mul_lo_u32 v21, v18, s34
	v_mul_hi_u32 v23, v18, s34
	v_sub_u32_e32 v22, 0, v21
	v_cmp_eq_u32_e32 vcc, 0, v23
	v_cndmask_b32_e32 v16, v21, v22, vcc
	v_mul_hi_u32 v17, v16, v18
	v_add_u32_e32 v16, v15, v20
	v_xor_b32_e32 v16, v16, v20
	v_add_u32_e32 v19, v18, v17
	v_sub_u32_e32 v17, v18, v17
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_mul_hi_u32 v17, v17, v16
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v25, v19
	v_mov_b32_e32 v24, v18
	v_mul_lo_u32 v26, v17, s34
	v_mov_b32_e32 v17, v19
	v_sub_u32_e32 v19, v16, v26
	v_cmp_ge_u32_e64 s[0:1], v16, v26
	v_cmp_le_u32_e32 vcc, s34, v19
	v_subrev_u32_e32 v26, s34, v19
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v27, s34, v19
	v_cndmask_b32_e32 v19, v19, v26, vcc
	v_cndmask_b32_e64 v19, v27, v19, s[0:1]
	v_xor_b32_e32 v19, v19, v20
	v_sub_u32_e32 v19, v19, v20
	v_cmp_ne_u32_e32 vcc, 0, v19
	v_mov_b32_e32 v27, v17
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v26, v16
BB7_20:                                 ; %Flow
                                        ;   in Loop: Header=BB7_18 Depth=4
	s_or_saveexec_b64 s[36:37], s[36:37]
	v_mov_b32_e32 v37, v25
	v_mov_b32_e32 v40, v27
	v_mov_b32_e32 v19, s31
	v_mov_b32_e32 v31, s34
	v_mov_b32_e32 v17, v20
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v34, v21
	v_mov_b32_e32 v35, v23
	v_mov_b32_e32 v38, v22
	v_mov_b32_e32 v29, v18
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v39, v26
	s_xor_b64 exec, exec, s[36:37]
	s_cbranch_execz BB7_22
; %bb.21:                               ; %._crit_edge
                                        ;   in Loop: Header=BB7_18 Depth=4
	s_ashr_i32 s35, s16, 31
	s_add_i32 s38, s16, s35
	s_xor_b32 s38, s38, s35
	v_cvt_f32_u32_e32 v19, s38
	v_ashrrev_i32_e32 v17, 31, v15
	v_mov_b32_e32 v30, 0
	v_add_u32_e32 v28, v15, v17
	v_rcp_iflag_f32_e32 v19, v19
	v_mov_b32_e32 v33, v30
	v_xor_b32_e32 v32, v28, v17
	v_mov_b32_e32 v40, v33
	v_mul_f32_e32 v19, s24, v19
	v_cvt_u32_f32_e32 v29, v19
	v_mov_b32_e32 v37, v30
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v19, s35
	v_mul_lo_u32 v34, v29, s38
	v_mul_hi_u32 v35, v29, s38
	v_mov_b32_e32 v31, s38
	v_mov_b32_e32 v36, v29
	v_sub_u32_e32 v38, 0, v34
	v_mov_b32_e32 v39, v32
BB7_22:                                 ; %Flow160
                                        ;   in Loop: Header=BB7_18 Depth=4
	s_or_b64 exec, exec, s[36:37]
	v_mov_b32_e32 v30, s34
	v_mov_b32_e32 v33, s31
	v_mov_b32_e32 v28, 1
	s_and_saveexec_b64 s[34:35], s[0:1]
	s_cbranch_execz BB7_24
; %bb.23:                               ;   in Loop: Header=BB7_18 Depth=4
	v_mov_b32_e32 v26, v39
	v_mov_b32_e32 v24, v36
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, v40
	v_mov_b32_e32 v22, v38
	v_mov_b32_e32 v23, v35
	v_mov_b32_e32 v21, v34
	v_mov_b32_e32 v25, v37
	v_mov_b32_e32 v18, v29
	v_mov_b32_e32 v30, v31
	v_mov_b32_e32 v16, v32
	v_mov_b32_e32 v33, v19
	v_mov_b32_e32 v20, v17
BB7_24:                                 ;   in Loop: Header=BB7_18 Depth=4
	s_or_b64 exec, exec, s[34:35]
	v_cmp_eq_u32_e32 vcc, 0, v23
	v_cndmask_b32_e32 v17, v21, v22, vcc
	v_mul_lo_u32 v19, v17, v25
	v_mul_hi_u32 v17, v17, v24
	v_add_u32_e32 v17, v17, v19
	v_add_u32_e32 v19, v18, v17
	v_sub_u32_e32 v17, v18, v17
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_mul_lo_u32 v18, v17, v27
	v_mul_hi_u32 v17, v17, v26
	v_xor_b32_e32 v19, v20, v33
	v_add_u32_e32 v17, v17, v18
	v_mul_lo_u32 v18, v17, v30
	v_add_u32_e32 v20, 1, v17
	v_add_u32_e32 v21, -1, v17
	v_sub_u32_e32 v22, v16, v18
	v_cmp_ge_u32_e32 vcc, v16, v18
	v_cmp_ge_u32_e64 s[0:1], v22, v30
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v16, v17, v20, s[0:1]
	v_cndmask_b32_e32 v16, v21, v16, vcc
	v_xor_b32_e32 v16, v16, v19
	v_sub_u32_e32 v16, v16, v19
	v_cmp_gt_i32_e32 vcc, s14, v16
	v_cndmask_b32_e32 v17, 0, v28, vcc
	v_and_b32_e32 v17, v17, v13
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB7_17
; %bb.25:                               ;   in Loop: Header=BB7_18 Depth=4
	v_add_u32_e32 v16, v16, v14
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshlrev_b64 v[16:17], 1, v[16:17]
	s_ashr_i32 s31, s30, 31
	v_add_co_u32_e32 v16, vcc, v5, v16
	s_lshl_b64 s[34:35], s[30:31], 1
	v_addc_co_u32_e32 v17, vcc, v6, v17, vcc
	v_mov_b32_e32 v19, s35
	v_add_co_u32_e32 v18, vcc, s34, v3
	v_addc_co_u32_e32 v19, vcc, v4, v19, vcc
	global_load_ushort v18, v[18:19], off
	global_load_ushort v16, v[16:17], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v18, v18
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v16, v16
	v_cvt_f64_f32_e32 v[18:19], v18
	v_cvt_f64_f32_e32 v[16:17], v16
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[9:10], v[16:17], v[18:19], v[9:10]
	s_branch BB7_17
BB7_26:                                 ; %Flow168
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_bwd_nchw_half
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 41
		.amdhsa_next_free_sgpr 39
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end7:
	.size	naive_conv_bwd_nchw_half, .Lfunc_end7-naive_conv_bwd_nchw_half
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2576
; NumSgprs: 41
; NumVgprs: 41
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 41
; NumVGPRsForWavesPerEU: 41
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_wrw_nchw_half ; -- Begin function naive_conv_wrw_nchw_half
	.globl	naive_conv_wrw_nchw_half
	.p2align	8
	.type	naive_conv_wrw_nchw_half,@function
naive_conv_wrw_nchw_half:               ; @naive_conv_wrw_nchw_half
naive_conv_wrw_nchw_half$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s22, s21
	s_mul_i32 s24, s7, s12
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB8_15
; %bb.1:                                ; %.lr.ph185
	s_ashr_i32 s0, s11, 31
	s_add_i32 s1, s11, s0
	s_xor_b32 s35, s1, s0
	v_cvt_f32_u32_e32 v1, s35
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s1, s6, 31
	s_add_i32 s30, s6, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s30, s30, s1
	s_xor_b32 s33, s1, s0
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	s_load_dwordx2 s[26:27], s[4:5], 0x8
	s_load_dwordx2 s[28:29], s[4:5], 0x10
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mul_i32 s31, s9, s8
	s_ashr_i32 s4, s12, 31
	s_mul_i32 s4, s31, s4
	v_mul_lo_u32 v2, v1, s35
	v_mul_hi_u32 v3, v1, s35
	s_mul_hi_u32 s34, s31, s12
	s_mul_i32 s31, s31, s12
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_mul_hi_i32 s5, s9, s8
	s_mul_i32 s5, s5, s12
	s_sub_i32 s20, 0, s20
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_mul_hi_u32 v1, v1, s30
	v_mul_lo_u32 v2, v1, s35
	v_add_u32_e32 v3, 1, v1
	v_add_u32_e32 v4, -1, v1
	v_sub_u32_e32 v5, s30, v2
	v_cmp_ge_u32_e32 vcc, s30, v2
	v_cmp_le_u32_e64 s[0:1], s35, v5
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	v_xor_b32_e32 v1, s33, v1
	v_subrev_u32_e32 v5, s33, v1
	v_mul_lo_u32 v2, v5, s11
	v_ashrrev_i32_e32 v1, 31, v5
	v_mul_lo_u32 v4, s31, v1
	v_mul_hi_u32 v6, s31, v5
	s_add_i32 s0, s34, s4
	v_sub_u32_e32 v3, s6, v2
	s_add_i32 s0, s0, s5
	v_add_u32_e32 v2, v6, v4
	v_ashrrev_i32_e32 v4, 31, v3
	v_mul_lo_u32 v7, s0, v5
	v_mad_i64_i32 v[3:4], s[0:1], v5, s11, v[3:4]
	v_mul_lo_u32 v1, s31, v5
	s_mul_i32 s1, s21, s12
	s_ashr_i32 s4, s22, 31
	s_mul_hi_i32 s0, s21, s12
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s3
	s_mul_i32 s3, s1, s4
	s_mul_hi_u32 s5, s1, s22
	s_mul_i32 s1, s1, s22
	s_add_i32 s3, s5, s3
	s_mul_i32 s0, s0, s22
	v_add_u32_e32 v2, v2, v7
	s_add_i32 s3, s3, s0
	v_mul_lo_u32 v7, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	v_mul_lo_u32 v9, s3, v3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_mul_lo_u32 v5, s1, v3
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v8, v7
	s_mul_i32 s1, s14, s13
	v_add_u32_e32 v6, v6, v9
	s_mul_hi_i32 s0, s14, s13
	v_mul_lo_u32 v9, s1, v4
	v_mul_hi_u32 v10, s1, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s1, v3
	v_add_co_u32_e32 v3, vcc, s26, v5
	v_mov_b32_e32 v8, s27
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	s_mul_i32 s0, s23, s14
	s_mul_i32 s5, s0, s13
	s_mul_i32 s0, s23, s12
	s_mul_i32 s5, s5, s11
	s_mul_i32 s11, s0, s9
	v_mov_b32_e32 v7, s29
	v_add_co_u32_e32 v5, vcc, s28, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s10, 0
	v_cmp_gt_i32_e64 s[28:29], s13, 0
	v_cmp_gt_i32_e64 s[30:31], s14, 0
	s_sub_i32 s6, 0, s19
	s_mul_i32 s11, s11, s8
	s_mul_i32 s12, s15, s9
	s_mov_b64 s[34:35], 0
	s_branch BB8_3
BB8_2:                                  ; %._crit_edge181
                                        ;   in Loop: Header=BB8_3 Depth=1
	v_mul_lo_u32 v9, v9, s21
	v_cvt_f32_f64_e32 v10, v[10:11]
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s22
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_cvt_f16_f32_e32 v9, v10
	s_or_b64 s[34:35], vcc, s[34:35]
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v3, v7
	v_addc_co_u32_e64 v8, s[0:1], v4, v8, s[0:1]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz BB8_15
BB8_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB8_6 Depth 2
                                        ;       Child Loop BB8_10 Depth 3
                                        ;         Child Loop BB8_13 Depth 4
	s_add_i32 s0, s22, s4
	s_xor_b32 s0, s0, s4
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s21, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s21, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s23, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s23
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s33, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s4, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s23
	v_mul_hi_u32 v12, v9, s23
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s33
	s_xor_b32 s36, s0, s33
	v_cvt_f32_u32_e32 v11, s36
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s36
	v_mul_hi_u32 v13, v10, s36
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s23
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s22
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s23, v14
	v_subrev_u32_e32 v10, s23, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s36
	v_add_u32_e32 v13, s23, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s36, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s33, v9
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v7, v0, v7
	v_subrev_u32_e32 v9, s33, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB8_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB8_3 Depth=1
	v_mul_lo_u32 v12, v8, s17
	v_mul_lo_u32 v10, s8, v9
	v_mul_lo_u32 v11, s18, v7
	s_mov_b32 s2, 0
	s_mov_b32 s3, 0
	v_add3_u32 v10, s6, v12, v10
	v_mul_lo_u32 v13, s9, v10
	v_add_u32_e32 v14, s20, v11
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_branch BB8_6
BB8_5:                                  ; %._crit_edge176
                                        ;   in Loop: Header=BB8_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s5
	s_cmp_eq_u32 s3, s10
	v_add_u32_e32 v13, s11, v13
	s_cbranch_scc1 BB8_2
BB8_6:                                  ; %.preheader
                                        ;   Parent Loop BB8_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB8_10 Depth 3
                                        ;         Child Loop BB8_13 Depth 4
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz BB8_5
; %bb.7:                                ; %.lr.ph175
                                        ;   in Loop: Header=BB8_6 Depth=2
	s_andn2_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz BB8_5
; %bb.8:                                ; %.lr.ph175.split.us.preheader
                                        ;   in Loop: Header=BB8_6 Depth=2
	s_mov_b32 s23, 0
	v_mov_b32_e32 v15, v13
	s_mov_b32 s33, s2
	s_branch BB8_10
BB8_9:                                  ; %Flow59
                                        ;   in Loop: Header=BB8_10 Depth=3
	s_or_b64 exec, exec, s[36:37]
	s_add_i32 s23, s23, 1
	s_add_i32 s33, s33, s14
	s_cmp_eq_u32 s23, s13
	v_add_u32_e32 v15, s12, v15
	s_cbranch_scc1 BB8_5
BB8_10:                                 ; %.lr.ph175.split.us
                                        ;   Parent Loop BB8_3 Depth=1
                                        ;     Parent Loop BB8_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB8_13 Depth 4
	s_mul_i32 s0, s23, s15
	s_sub_i32 s0, s0, s19
	v_add_u32_e32 v16, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v16
	v_cmp_gt_i32_e64 s[0:1], s8, v16
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz BB8_9
; %bb.11:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB8_10 Depth=3
	v_mov_b32_e32 v16, v14
	s_mov_b32 s38, s33
	s_mov_b32 s40, s14
	s_branch BB8_13
BB8_12:                                 ;   in Loop: Header=BB8_13 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s40, s40, -1
	s_add_i32 s38, s38, 1
	s_cmp_lg_u32 s40, 0
	v_add_u32_e32 v16, s16, v16
	s_cbranch_scc0 BB8_9
BB8_13:                                 ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB8_3 Depth=1
                                        ;     Parent Loop BB8_6 Depth=2
                                        ;       Parent Loop BB8_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v16
	v_cmp_gt_i32_e64 s[0:1], s9, v16
	s_and_b64 s[42:43], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[42:43]
	s_cbranch_execz BB8_12
; %bb.14:                               ;   in Loop: Header=BB8_13 Depth=4
	v_add_u32_e32 v17, v15, v16
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_ashr_i32 s39, s38, 31
	v_add_co_u32_e32 v17, vcc, v1, v17
	s_lshl_b64 s[42:43], s[38:39], 1
	v_addc_co_u32_e32 v18, vcc, v2, v18, vcc
	v_mov_b32_e32 v20, s43
	v_add_co_u32_e32 v19, vcc, s42, v5
	v_addc_co_u32_e32 v20, vcc, v6, v20, vcc
	global_load_ushort v19, v[19:20], off
	global_load_ushort v17, v[17:18], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v17, v17
	v_cvt_f64_f32_e32 v[19:20], v19
	v_cvt_f64_f32_e32 v[17:18], v17
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[17:18], v[19:20], v[10:11]
	s_branch BB8_12
BB8_15:                                 ; %Flow65
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_wrw_nchw_half
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 21
		.amdhsa_next_free_sgpr 44
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end8:
	.size	naive_conv_wrw_nchw_half, .Lfunc_end8-naive_conv_wrw_nchw_half
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 1504
; NumSgprs: 46
; NumVgprs: 21
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 21
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_fwd_ncdhw_half ; -- Begin function naive_conv_fwd_ncdhw_half
	.globl	naive_conv_fwd_ncdhw_half
	.p2align	8
	.type	naive_conv_fwd_ncdhw_half,@function
naive_conv_fwd_ncdhw_half:              ; @naive_conv_fwd_ncdhw_half
naive_conv_fwd_ncdhw_half$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s16, s15
	s_mul_i32 s24, s7, s14
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB9_18
; %bb.1:                                ; %.lr.ph239
	s_ashr_i32 s0, s12, 31
	s_add_i32 s1, s12, s0
	s_xor_b32 s39, s1, s0
	v_cvt_f32_u32_e32 v1, s39
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s1, s11, 31
	s_ashr_i32 s33, s6, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s3, s11, s1
	s_add_i32 s2, s6, s33
	s_xor_b32 s40, s3, s1
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s38, s2, s33
	s_xor_b32 s2, s33, s0
	s_load_dwordx2 s[36:37], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[26:27], s[4:5], 0x10
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_load_dwordx2 s[4:5], s[4:5], 0x68
	v_mul_lo_u32 v2, v1, s39
	v_mul_hi_u32 v3, v1, s39
	s_mov_b64 s[42:43], 0
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s41, s5, s12
	s_mul_i32 s5, s5, s13
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_cvt_f32_u32_e32 v2, s40
	v_mul_hi_u32 v1, v1, s38
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s39
	v_add_u32_e32 v4, -1, v1
	v_mul_f32_e32 v2, s25, v2
	v_sub_u32_e32 v5, s38, v3
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_ge_u32_e32 vcc, s38, v3
	v_cmp_le_u32_e64 s[0:1], s39, v5
	v_add_u32_e32 v3, 1, v1
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	s_mul_i32 s0, s12, s11
	v_mul_hi_u32 v4, v2, s40
	s_ashr_i32 s11, s0, 31
	v_mul_lo_u32 v3, v2, s40
	s_add_i32 s0, s0, s11
	s_xor_b32 s39, s0, s11
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s39
	v_sub_u32_e32 v6, 0, v3
	v_cndmask_b32_e32 v3, v3, v6, vcc
	v_mul_hi_u32 v3, v3, v2
	v_rcp_iflag_f32_e32 v4, v4
	v_xor_b32_e32 v1, s2, v1
	v_subrev_u32_e32 v1, s2, v1
	v_add_u32_e32 v7, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_mul_f32_e32 v3, s25, v4
	v_cvt_u32_f32_e32 v3, v3
	v_ashrrev_i32_e32 v5, 31, v1
	v_add_u32_e32 v6, v1, v5
	v_cndmask_b32_e32 v2, v2, v7, vcc
	v_mul_lo_u32 v4, v1, s12
	v_mul_lo_u32 v1, v3, s39
	v_mul_hi_u32 v7, v3, s39
	v_xor_b32_e32 v6, v6, v5
	v_mul_hi_u32 v2, v2, v6
	v_sub_u32_e32 v9, 0, v1
	v_cmp_eq_u32_e64 s[0:1], 0, v7
	v_cndmask_b32_e64 v1, v1, v9, s[0:1]
	v_mul_lo_u32 v2, v2, s40
	v_mul_hi_u32 v1, v1, v3
	v_sub_u32_e32 v8, v6, v2
	v_cmp_ge_u32_e64 s[2:3], v6, v2
	v_add_u32_e32 v6, v3, v1
	v_sub_u32_e32 v1, v3, v1
	v_cndmask_b32_e64 v1, v1, v6, s[0:1]
	v_mul_hi_u32 v1, v1, s38
	v_cmp_le_u32_e32 vcc, s40, v8
	v_subrev_u32_e32 v2, s40, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_mul_lo_u32 v3, v1, s39
	v_add_u32_e32 v7, s40, v8
	v_cndmask_b32_e32 v2, v8, v2, vcc
	v_cndmask_b32_e64 v2, v7, v2, s[2:3]
	v_xor_b32_e32 v2, v2, v5
	v_sub_u32_e32 v7, v2, v5
	v_sub_u32_e32 v2, s38, v3
	v_cmp_le_u32_e32 vcc, s39, v2
	v_cmp_ge_u32_e64 s[0:1], s38, v3
	v_add_u32_e32 v3, 1, v1
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v2, -1, v1
	v_cndmask_b32_e32 v1, v1, v3, vcc
	s_xor_b32 s2, s33, s11
	v_cndmask_b32_e64 v1, v2, v1, s[0:1]
	v_xor_b32_e32 v3, s2, v1
	v_mul_hi_i32 v2, v7, s5
	v_mul_lo_u32 v1, v7, s5
	v_subrev_u32_e32 v5, s2, v3
	s_ashr_i32 s2, s10, 31
	v_sub_u32_e32 v3, s6, v4
	v_mad_i64_i32 v[1:2], s[0:1], v5, s13, v[1:2]
	s_mul_i32 s1, s9, s8
	s_mul_hi_i32 s0, s9, s8
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s3, s1, s10
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s10
	s_mul_i32 s1, s1, s10
	s_add_i32 s2, s2, s0
	v_ashrrev_i32_e32 v4, 31, v3
	v_mul_lo_u32 v8, s2, v1
	v_mul_hi_u32 v6, s1, v1
	v_mul_lo_u32 v2, s1, v2
	v_mul_lo_u32 v1, s1, v1
	v_mad_i64_i32 v[3:4], s[0:1], v5, s12, v[3:4]
	s_mul_i32 s1, s30, s13
	s_ashr_i32 s2, s31, 31
	s_mul_hi_i32 s0, s30, s13
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s5, s1, s31
	s_add_i32 s2, s5, s2
	s_mul_i32 s0, s0, s31
	s_ashr_i32 s3, s4, 31
	s_mul_i32 s1, s1, s31
	s_add_i32 s0, s2, s0
	s_mul_i32 s2, s1, s3
	s_mul_hi_u32 s3, s1, s4
	v_add_u32_e32 v2, v6, v2
	s_mul_i32 s1, s1, s4
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s4
	v_add_u32_e32 v2, v2, v8
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v8, s1, v4
	v_mul_hi_u32 v9, s1, v3
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	v_mul_lo_u32 v10, s2, v3
	v_mul_lo_u32 v5, s1, v3
	v_mad_i64_i32 v[3:4], s[0:1], v7, s41, v[3:4]
	s_mul_i32 s1, s15, s14
	s_ashr_i32 s5, s16, 31
	s_mul_hi_i32 s0, s15, s14
	v_mov_b32_e32 v6, s37
	v_add_co_u32_e32 v1, vcc, s36, v1
	s_mul_i32 s2, s1, s5
	s_mul_hi_u32 s3, s1, s16
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v9, v8
	s_mul_i32 s1, s1, s16
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s16
	v_add_u32_e32 v6, v6, v10
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v9, s1, v4
	v_mul_hi_u32 v10, s1, v3
	v_mul_lo_u32 v11, s2, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s1, v3
	v_add_co_u32_e32 v3, vcc, s34, v5
	v_mov_b32_e32 v8, s35
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	s_mul_i32 s33, s10, s9
	s_mul_i32 s40, s20, s10
	s_mul_i32 s6, s4, s31
	v_mov_b32_e32 v7, s27
	v_add_co_u32_e32 v5, vcc, s26, v5
	s_sub_i32 s12, 0, s29
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s13, 0
	v_cmp_gt_i32_e64 s[34:35], s30, 0
	v_cmp_gt_i32_e64 s[36:37], s31, 0
	v_cmp_gt_i32_e64 s[38:39], s4, 0
	s_mul_i32 s11, s6, s30
	s_sub_i32 s14, 0, s28
	s_sub_i32 s29, 0, s23
	s_mul_i32 s33, s33, s8
	s_mul_i32 s40, s40, s9
	s_mul_i32 s41, s21, s10
	s_branch BB9_3
BB9_2:                                  ; %._crit_edge235
                                        ;   in Loop: Header=BB9_3 Depth=1
	v_mul_lo_u32 v9, v9, s15
	v_cvt_f32_f64_e32 v10, v[10:11]
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s16
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_cvt_f16_f32_e32 v9, v10
	s_or_b64 s[42:43], vcc, s[42:43]
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v5, v7
	v_addc_co_u32_e64 v8, s[0:1], v6, v8, s[0:1]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz BB9_18
BB9_3:                                  ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB9_6 Depth 2
                                        ;       Child Loop BB9_9 Depth 3
                                        ;         Child Loop BB9_13 Depth 4
                                        ;           Child Loop BB9_16 Depth 5
	s_add_i32 s0, s16, s5
	s_xor_b32 s0, s0, s5
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s15, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s15, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s44, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s44
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s45, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s5, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s44
	v_mul_hi_u32 v12, v9, s44
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s45
	s_xor_b32 s46, s0, s45
	v_cvt_f32_u32_e32 v11, s46
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s46
	v_mul_hi_u32 v13, v10, s46
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s44
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s16
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s44, v14
	v_subrev_u32_e32 v10, s44, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s46
	v_add_u32_e32 v13, s44, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s46, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s45, v9
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v7, v0, v7
	v_subrev_u32_e32 v9, s45, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB9_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB9_3 Depth=1
	v_mul_lo_u32 v12, v9, s17
	v_mul_lo_u32 v13, v8, s18
	v_mul_lo_u32 v15, s19, v7
	s_mov_b32 s2, 0
	v_add_u32_e32 v10, s29, v12
	v_mul_lo_u32 v14, s9, v10
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_subrev_u32_e32 v12, s23, v12
	v_add3_u32 v14, s14, v13, v14
	v_mul_lo_u32 v16, s10, v14
	v_add_u32_e32 v14, s12, v15
	v_subrev_u32_e32 v13, s28, v13
	s_mov_b32 s3, 0
	v_add3_u32 v15, s12, v16, v15
	s_branch BB9_6
BB9_5:                                  ; %Flow85
                                        ;   in Loop: Header=BB9_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s11
	s_cmp_eq_u32 s3, s13
	v_add_u32_e32 v15, s33, v15
	s_cbranch_scc1 BB9_2
BB9_6:                                  ; %.preheader
                                        ;   Parent Loop BB9_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB9_9 Depth 3
                                        ;         Child Loop BB9_13 Depth 4
                                        ;           Child Loop BB9_16 Depth 5
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz BB9_5
; %bb.7:                                ; %.lr.ph229
                                        ;   in Loop: Header=BB9_6 Depth=2
	s_mov_b32 s44, 0
	v_mov_b32_e32 v16, v15
	s_mov_b32 s45, s2
	s_branch BB9_9
BB9_8:                                  ; %._crit_edge224
                                        ;   in Loop: Header=BB9_9 Depth=3
	s_add_i32 s44, s44, 1
	s_add_i32 s45, s45, s6
	s_cmp_eq_u32 s44, s30
	v_add_u32_e32 v16, s40, v16
	s_cbranch_scc1 BB9_5
BB9_9:                                  ;   Parent Loop BB9_3 Depth=1
                                        ;     Parent Loop BB9_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB9_13 Depth 4
                                        ;           Child Loop BB9_16 Depth 5
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB9_8
; %bb.10:                               ; %.lr.ph223
                                        ;   in Loop: Header=BB9_9 Depth=3
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB9_8
; %bb.11:                               ; %.lr.ph223.split.us.preheader
                                        ;   in Loop: Header=BB9_9 Depth=3
	s_mul_i32 s0, s44, s20
	v_add_u32_e32 v17, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v17
	s_and_b64 s[46:47], vcc, s[0:1]
	s_mov_b32 s48, 0
	v_mov_b32_e32 v17, v16
	s_mov_b32 s49, s45
	s_branch BB9_13
BB9_12:                                 ; %Flow81
                                        ;   in Loop: Header=BB9_13 Depth=4
	s_or_b64 exec, exec, s[50:51]
	s_add_i32 s48, s48, 1
	s_add_i32 s49, s49, s4
	s_cmp_lg_u32 s48, s31
	v_add_u32_e32 v17, s41, v17
	s_cbranch_scc0 BB9_8
BB9_13:                                 ; %.lr.ph223.split.us
                                        ;   Parent Loop BB9_3 Depth=1
                                        ;     Parent Loop BB9_6 Depth=2
                                        ;       Parent Loop BB9_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB9_16 Depth 5
	s_mul_i32 s0, s48, s21
	v_add_u32_e32 v18, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s9, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[50:51], s[0:1]
	s_cbranch_execz BB9_12
; %bb.14:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB9_13 Depth=4
	s_mov_b32 s52, 0
	s_mov_b32 s54, s49
	s_mov_b32 s53, s4
	s_branch BB9_16
BB9_15:                                 ;   in Loop: Header=BB9_16 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s53, s53, -1
	s_add_i32 s54, s54, 1
	s_add_i32 s52, s52, s22
	s_cmp_lg_u32 s53, 0
	s_cbranch_scc0 BB9_12
BB9_16:                                 ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB9_3 Depth=1
                                        ;     Parent Loop BB9_6 Depth=2
                                        ;       Parent Loop BB9_9 Depth=3
                                        ;         Parent Loop BB9_13 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_add_u32_e32 v18, s52, v14
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s10, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 s[56:57], s[46:47], s[0:1]
	s_and_saveexec_b64 s[0:1], s[56:57]
	s_cbranch_execz BB9_15
; %bb.17:                               ;   in Loop: Header=BB9_16 Depth=5
	v_add_u32_e32 v18, s52, v17
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshlrev_b64 v[18:19], 1, v[18:19]
	s_ashr_i32 s55, s54, 31
	v_add_co_u32_e32 v18, vcc, v1, v18
	s_lshl_b64 s[56:57], s[54:55], 1
	v_addc_co_u32_e32 v19, vcc, v2, v19, vcc
	v_mov_b32_e32 v21, s57
	v_add_co_u32_e32 v20, vcc, s56, v3
	v_addc_co_u32_e32 v21, vcc, v4, v21, vcc
	global_load_ushort v20, v[20:21], off
	global_load_ushort v18, v[18:19], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v20, v20
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v18, v18
	v_cvt_f64_f32_e32 v[20:21], v20
	v_cvt_f64_f32_e32 v[18:19], v18
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[18:19], v[20:21], v[10:11]
	s_branch BB9_15
BB9_18:                                 ; %Flow89
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_fwd_ncdhw_half
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 58
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end9:
	.size	naive_conv_fwd_ncdhw_half, .Lfunc_end9-naive_conv_fwd_ncdhw_half
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2036
; NumSgprs: 60
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 60
; NumVGPRsForWavesPerEU: 22
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_bwd_ncdhw_half ; -- Begin function naive_conv_bwd_ncdhw_half
	.globl	naive_conv_bwd_ncdhw_half
	.p2align	8
	.type	naive_conv_bwd_ncdhw_half,@function
naive_conv_bwd_ncdhw_half:              ; @naive_conv_bwd_ncdhw_half
naive_conv_bwd_ncdhw_half$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s10, s9
	s_mul_i32 s24, s7, s8
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB10_35
; %bb.1:                                ; %.lr.ph256
	s_ashr_i32 s33, s13, 31
	s_add_i32 s0, s13, s33
	s_xor_b32 s3, s0, s33
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s38, s6, 31
	s_ashr_i32 s0, s11, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s1, s6, s38
	s_add_i32 s2, s11, s0
	s_xor_b32 s40, s2, s0
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s39, s1, s38
	s_xor_b32 s2, s38, s33
	s_load_dwordx2 s[36:37], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[26:27], s[4:5], 0x10
	v_mul_lo_u32 v2, v1, s3
	v_mul_hi_u32 v3, v1, s3
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_load_dwordx2 s[4:5], s[4:5], 0x68
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v11, s37
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	v_cvt_f32_u32_e32 v3, s40
	s_mul_i32 s42, s5, s13
	v_add_u32_e32 v4, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s39
	v_rcp_iflag_f32_e32 v3, v3
	v_mul_lo_u32 v2, v1, s3
	v_add_u32_e32 v4, 1, v1
	v_add_u32_e32 v5, -1, v1
	v_sub_u32_e32 v6, s39, v2
	v_cmp_ge_u32_e32 vcc, s39, v2
	v_mul_f32_e32 v2, s25, v3
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_le_u32_e64 s[0:1], s3, v6
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	s_mul_i32 s0, s13, s11
	v_mul_hi_u32 v4, v2, s40
	s_ashr_i32 s11, s0, 31
	v_mul_lo_u32 v3, v2, s40
	s_add_i32 s0, s0, s11
	s_xor_b32 s41, s0, s11
	v_cndmask_b32_e32 v1, v5, v1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s41
	v_sub_u32_e32 v7, 0, v3
	v_cndmask_b32_e32 v3, v3, v7, vcc
	v_xor_b32_e32 v1, s2, v1
	v_mul_hi_u32 v3, v3, v2
	v_subrev_u32_e32 v1, s2, v1
	v_rcp_iflag_f32_e32 v4, v4
	v_ashrrev_i32_e32 v6, 31, v1
	v_mul_lo_u32 v5, v1, s13
	v_add_u32_e32 v1, v1, v6
	v_xor_b32_e32 v7, v1, v6
	v_add_u32_e32 v1, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_cndmask_b32_e32 v1, v2, v1, vcc
	v_mul_f32_e32 v2, s25, v4
	v_mul_hi_u32 v1, v1, v7
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s40
	v_sub_u32_e32 v1, s6, v5
	v_mul_lo_u32 v4, v2, s41
	v_mul_hi_u32 v5, v2, s41
	v_sub_u32_e32 v8, v7, v3
	v_cmp_ge_u32_e64 s[2:3], v7, v3
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v5
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_hi_u32 v4, v4, v2
	v_cmp_le_u32_e32 vcc, s40, v8
	v_subrev_u32_e32 v3, s40, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_add_u32_e32 v7, v2, v4
	v_sub_u32_e32 v2, v2, v4
	v_cndmask_b32_e64 v2, v2, v7, s[0:1]
	v_mul_hi_u32 v2, v2, s39
	v_add_u32_e32 v5, s40, v8
	v_cndmask_b32_e32 v3, v8, v3, vcc
	v_cndmask_b32_e64 v3, v5, v3, s[2:3]
	v_mul_lo_u32 v4, v2, s41
	v_xor_b32_e32 v3, v3, v6
	v_sub_u32_e32 v7, v3, v6
	s_xor_b32 s2, s38, s11
	v_sub_u32_e32 v3, s39, v4
	v_cmp_le_u32_e32 vcc, s41, v3
	v_cmp_ge_u32_e64 s[0:1], s39, v4
	v_add_u32_e32 v4, 1, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v3, -1, v2
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e64 v2, v3, v2, s[0:1]
	v_xor_b32_e32 v5, s2, v2
	v_ashrrev_i32_e32 v2, 31, v1
	v_mad_i64_i32 v[3:4], s[0:1], v7, s42, v[1:2]
	v_subrev_u32_e32 v8, s2, v5
	s_ashr_i32 s6, s10, 31
	v_cmp_gt_i32_e64 s[38:39], s4, 0
	v_mad_i64_i32 v[3:4], s[0:1], v8, s13, v[3:4]
	s_mul_i32 s1, s9, s8
	s_mul_hi_i32 s0, s9, s8
	s_mul_i32 s2, s1, s6
	s_mul_hi_u32 s3, s1, s10
	s_mul_i32 s1, s1, s10
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s10
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v4, s1, v4
	v_mul_hi_u32 v5, s1, v3
	v_mul_lo_u32 v6, s2, v3
	v_mul_lo_u32 v3, s1, v3
	s_ashr_i32 s1, s4, 31
	v_add_u32_e32 v4, v5, v4
	v_mul_lo_u32 v5, v8, s12
	v_add_u32_e32 v4, v4, v6
	v_mul_hi_i32 v6, v8, s12
	s_mul_i32 s3, s31, s30
	v_mul_lo_u32 v9, v5, s33
	v_mul_hi_u32 v10, v5, s13
	v_mul_lo_u32 v5, v5, s13
	v_mul_lo_u32 v6, v6, s13
	s_mul_hi_i32 s2, s31, s30
	v_add_u32_e32 v9, v10, v9
	s_mul_i32 s0, s5, s12
	v_add_u32_e32 v6, v9, v6
	v_add_co_u32_e32 v1, vcc, v5, v1
	s_mul_i32 s1, s3, s1
	s_mul_hi_u32 s5, s3, s4
	s_mul_i32 s3, s3, s4
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	s_add_i32 s1, s5, s1
	s_mul_i32 s2, s2, s4
	s_add_i32 s1, s1, s2
	v_mul_lo_u32 v6, s3, v2
	v_mul_hi_u32 v9, s3, v1
	v_mul_lo_u32 v10, s1, v1
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	v_mul_lo_u32 v5, s3, v1
	v_add_co_u32_e32 v1, vcc, s36, v3
	v_add_u32_e32 v3, v9, v6
	v_add_u32_e32 v6, v3, v10
	v_addc_co_u32_e32 v2, vcc, v11, v4, vcc
	v_lshlrev_b64 v[3:4], 1, v[5:6]
	v_mul_hi_i32 v6, v7, s0
	v_mul_lo_u32 v5, v7, s0
	s_ashr_i32 s2, s16, 31
	v_mov_b32_e32 v7, s35
	v_add_co_u32_e32 v3, vcc, s34, v3
	v_mad_i64_i32 v[5:6], s[0:1], v8, s12, v[5:6]
	s_mul_i32 s1, s15, s14
	s_mul_hi_i32 s0, s15, s14
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s3, s1, s16
	s_mul_i32 s1, s1, s16
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s16
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v6, s1, v6
	v_mul_hi_u32 v8, s1, v5
	v_mul_lo_u32 v9, s2, v5
	v_mul_lo_u32 v5, s1, v5
	v_addc_co_u32_e32 v4, vcc, v7, v4, vcc
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	s_mul_i32 s5, s4, s31
	s_mul_i32 s8, s5, s30
	v_mov_b32_e32 v7, s27
	v_add_co_u32_e32 v5, vcc, s26, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s12, 0
	v_cmp_gt_i32_e64 s[34:35], s30, 0
	v_cmp_gt_i32_e64 s[36:37], s31, 0
	s_mul_i32 s8, s8, s13
	s_mov_b64 s[40:41], 0
	s_branch BB10_3
BB10_2:                                 ; %Flow245
                                        ;   in Loop: Header=BB10_3 Depth=1
	v_mul_lo_u32 v9, v9, s9
	v_cvt_f32_f64_e32 v10, v[10:11]
	v_sub_u32_e32 v7, v0, v7
	v_add_u32_e32 v0, 0x100, v0
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s10
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_cvt_f16_f32_e32 v9, v10
	v_cmp_le_i32_e32 vcc, s24, v0
	s_or_b64 s[40:41], vcc, s[40:41]
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v1, v7
	v_addc_co_u32_e64 v8, s[0:1], v2, v8, s[0:1]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[40:41]
	s_cbranch_execz BB10_35
BB10_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB10_6 Depth 2
                                        ;       Child Loop BB10_9 Depth 3
                                        ;         Child Loop BB10_19 Depth 4
                                        ;           Child Loop BB10_27 Depth 5
	s_add_i32 s0, s10, s6
	s_xor_b32 s0, s0, s6
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s9, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s9, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s11, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s11
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s13, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s6, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s11
	v_mul_hi_u32 v12, v9, s11
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s13
	s_xor_b32 s33, s0, s13
	v_cvt_f32_u32_e32 v11, s33
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s33
	v_mul_hi_u32 v13, v10, s33
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s11
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s10
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s11, v14
	v_subrev_u32_e32 v10, s11, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s33
	v_add_u32_e32 v13, s11, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s33, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s13, v9
	v_mov_b32_e32 v10, 0
	v_subrev_u32_e32 v9, s13, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB10_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB10_3 Depth=1
	v_add_u32_e32 v10, s29, v0
	v_sub_u32_e32 v14, v10, v7
	v_mov_b32_e32 v10, 0
	v_add_u32_e32 v12, s23, v9
	v_add_u32_e32 v13, s28, v8
	s_mov_b32 s2, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s3, 0
	s_branch BB10_6
BB10_5:                                 ; %Flow243
                                        ;   in Loop: Header=BB10_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s8
	s_cmp_eq_u32 s3, s12
	s_cbranch_scc1 BB10_2
BB10_6:                                 ; %.preheader
                                        ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB10_9 Depth 3
                                        ;         Child Loop BB10_19 Depth 4
                                        ;           Child Loop BB10_27 Depth 5
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz BB10_5
; %bb.7:                                ; %.lr.ph246
                                        ;   in Loop: Header=BB10_6 Depth=2
	s_mul_i32 s11, s3, s14
	s_mov_b32 s13, 0
	s_mov_b32 s33, s2
	s_branch BB10_9
BB10_8:                                 ; %._crit_edge241
                                        ;   in Loop: Header=BB10_9 Depth=3
	s_add_i32 s13, s13, 1
	s_add_i32 s33, s33, s5
	s_cmp_eq_u32 s13, s30
	s_cbranch_scc1 BB10_5
BB10_9:                                 ;   Parent Loop BB10_3 Depth=1
                                        ;     Parent Loop BB10_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB10_19 Depth 4
                                        ;           Child Loop BB10_27 Depth 5
	s_mul_i32 s0, s13, s20
	v_subrev_u32_e32 v27, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v27
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $sgpr42
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $sgpr43
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr23_vgpr24
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr25_vgpr26
	s_and_saveexec_b64 s[44:45], vcc
	s_xor_b64 s[44:45], exec, s[44:45]
	s_cbranch_execz BB10_11
; %bb.10:                               ;   in Loop: Header=BB10_9 Depth=3
	s_ashr_i32 s42, s17, 31
	s_add_i32 s0, s17, s42
	s_xor_b32 s43, s0, s42
	v_cvt_f32_u32_e32 v15, s43
	v_ashrrev_i32_e32 v19, 31, v27
	v_rcp_iflag_f32_e32 v15, v15
	v_mul_f32_e32 v15, s25, v15
	v_cvt_u32_f32_e32 v17, v15
	v_mul_lo_u32 v20, v17, s43
	v_mul_hi_u32 v22, v17, s43
	v_sub_u32_e32 v21, 0, v20
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cndmask_b32_e32 v15, v20, v21, vcc
	v_mul_hi_u32 v16, v15, v17
	v_add_u32_e32 v15, v27, v19
	v_xor_b32_e32 v15, v15, v19
	v_add_u32_e32 v18, v17, v16
	v_sub_u32_e32 v16, v17, v16
	v_cndmask_b32_e32 v16, v16, v18, vcc
	v_mul_hi_u32 v16, v16, v15
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v24, v18
	v_mov_b32_e32 v23, v17
	v_mul_lo_u32 v25, v16, s43
	v_mov_b32_e32 v16, v18
	v_sub_u32_e32 v18, v15, v25
	v_cmp_ge_u32_e64 s[0:1], v15, v25
	v_cmp_le_u32_e32 vcc, s43, v18
	v_subrev_u32_e32 v25, s43, v18
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v26, s43, v18
	v_cndmask_b32_e32 v18, v18, v25, vcc
	v_cndmask_b32_e64 v18, v26, v18, s[0:1]
	v_xor_b32_e32 v18, v18, v19
	v_sub_u32_e32 v18, v18, v19
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_mov_b32_e32 v26, v16
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v25, v15
BB10_11:                                ; %Flow240
                                        ;   in Loop: Header=BB10_9 Depth=3
	s_or_saveexec_b64 s[44:45], s[44:45]
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v40, v26
	v_mov_b32_e32 v28, s42
	v_mov_b32_e32 v31, s43
	v_mov_b32_e32 v16, v19
	v_mov_b32_e32 v32, v15
	v_mov_b32_e32 v34, v20
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v38, v21
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v36, v23
	v_mov_b32_e32 v39, v25
	s_xor_b64 exec, exec, s[44:45]
	s_cbranch_execz BB10_13
; %bb.12:                               ; %._crit_edge117
                                        ;   in Loop: Header=BB10_9 Depth=3
	s_ashr_i32 s46, s17, 31
	s_add_i32 s47, s17, s46
	s_xor_b32 s47, s47, s46
	v_cvt_f32_u32_e32 v18, s47
	v_ashrrev_i32_e32 v16, 31, v27
	v_mov_b32_e32 v30, 0
	v_add_u32_e32 v27, v27, v16
	v_rcp_iflag_f32_e32 v18, v18
	v_mov_b32_e32 v33, v30
	v_xor_b32_e32 v32, v27, v16
	v_mov_b32_e32 v40, v33
	v_mul_f32_e32 v18, s25, v18
	v_cvt_u32_f32_e32 v29, v18
	v_mov_b32_e32 v37, v30
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v28, s46
	v_mul_lo_u32 v34, v29, s47
	v_mul_hi_u32 v35, v29, s47
	v_mov_b32_e32 v31, s47
	v_mov_b32_e32 v36, v29
	v_sub_u32_e32 v38, 0, v34
	v_mov_b32_e32 v39, v32
BB10_13:                                ; %Flow241
                                        ;   in Loop: Header=BB10_9 Depth=3
	s_or_b64 exec, exec, s[44:45]
	v_mov_b32_e32 v27, s43
	v_mov_b32_e32 v30, s42
	v_mov_b32_e32 v18, 1
	s_and_saveexec_b64 s[42:43], s[0:1]
	s_cbranch_execz BB10_15
; %bb.14:                               ;   in Loop: Header=BB10_9 Depth=3
	v_mov_b32_e32 v25, v39
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v26, v40
	v_mov_b32_e32 v21, v38
	v_mov_b32_e32 v22, v35
	v_mov_b32_e32 v20, v34
	v_mov_b32_e32 v24, v37
	v_mov_b32_e32 v17, v29
	v_mov_b32_e32 v27, v31
	v_mov_b32_e32 v15, v32
	v_mov_b32_e32 v30, v28
	v_mov_b32_e32 v19, v16
BB10_15:                                ;   in Loop: Header=BB10_9 Depth=3
	s_or_b64 exec, exec, s[42:43]
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB10_8
; %bb.16:                               ; %.lr.ph240
                                        ;   in Loop: Header=BB10_9 Depth=3
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB10_8
; %bb.17:                               ; %.lr.ph240.split.us.preheader
                                        ;   in Loop: Header=BB10_9 Depth=3
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cndmask_b32_e32 v16, v20, v21, vcc
	v_mul_lo_u32 v20, v16, v24
	v_mul_hi_u32 v16, v16, v23
	v_xor_b32_e32 v19, v19, v30
	s_mov_b32 s42, 0
	s_mov_b32 s43, s33
	v_add_u32_e32 v16, v16, v20
	v_add_u32_e32 v20, v17, v16
	v_sub_u32_e32 v16, v17, v16
	v_cndmask_b32_e32 v16, v16, v20, vcc
	v_mul_lo_u32 v17, v16, v26
	v_mul_hi_u32 v16, v16, v25
	v_add_u32_e32 v16, v16, v17
	v_mul_lo_u32 v17, v16, v27
	v_add_u32_e32 v20, 1, v16
	v_add_u32_e32 v21, -1, v16
	v_sub_u32_e32 v22, v15, v17
	v_cmp_ge_u32_e32 vcc, v15, v17
	v_cmp_ge_u32_e64 s[0:1], v22, v27
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v15, v16, v20, s[0:1]
	v_cndmask_b32_e32 v15, v21, v15, vcc
	v_xor_b32_e32 v15, v15, v19
	v_sub_u32_e32 v16, v15, v19
	v_add_u32_e32 v15, s11, v16
	v_mul_lo_u32 v15, v15, s15
	v_cmp_gt_i32_e32 vcc, s14, v16
	v_cndmask_b32_e32 v16, 0, v18, vcc
	s_branch BB10_19
BB10_18:                                ; %._crit_edge.us
                                        ;   in Loop: Header=BB10_19 Depth=4
	s_add_i32 s42, s42, 1
	s_add_i32 s43, s43, s4
	s_cmp_lg_u32 s42, s31
	s_cbranch_scc0 BB10_8
BB10_19:                                ; %.lr.ph240.split.us
                                        ;   Parent Loop BB10_3 Depth=1
                                        ;     Parent Loop BB10_6 Depth=2
                                        ;       Parent Loop BB10_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB10_27 Depth 5
	s_mul_i32 s0, s42, s21
	v_subrev_u32_e32 v29, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v29
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $sgpr44
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $sgpr45
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr25_vgpr26
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr27_vgpr28
	s_and_saveexec_b64 s[46:47], vcc
	s_xor_b64 s[46:47], exec, s[46:47]
	s_cbranch_execz BB10_21
; %bb.20:                               ;   in Loop: Header=BB10_19 Depth=4
	s_ashr_i32 s44, s18, 31
	s_add_i32 s0, s18, s44
	s_xor_b32 s45, s0, s44
	v_cvt_f32_u32_e32 v17, s45
	v_ashrrev_i32_e32 v21, 31, v29
	v_rcp_iflag_f32_e32 v17, v17
	v_mul_f32_e32 v17, s25, v17
	v_cvt_u32_f32_e32 v19, v17
	v_mul_lo_u32 v22, v19, s45
	v_mul_hi_u32 v24, v19, s45
	v_sub_u32_e32 v23, 0, v22
	v_cmp_eq_u32_e32 vcc, 0, v24
	v_cndmask_b32_e32 v17, v22, v23, vcc
	v_mul_hi_u32 v18, v17, v19
	v_add_u32_e32 v17, v29, v21
	v_xor_b32_e32 v17, v17, v21
	v_add_u32_e32 v20, v19, v18
	v_sub_u32_e32 v18, v19, v18
	v_cndmask_b32_e32 v18, v18, v20, vcc
	v_mul_hi_u32 v18, v18, v17
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v26, v20
	v_mov_b32_e32 v25, v19
	v_mul_lo_u32 v27, v18, s45
	v_mov_b32_e32 v18, v20
	v_sub_u32_e32 v20, v17, v27
	v_cmp_ge_u32_e64 s[0:1], v17, v27
	v_cmp_le_u32_e32 vcc, s45, v20
	v_subrev_u32_e32 v27, s45, v20
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v28, s45, v20
	v_cndmask_b32_e32 v20, v20, v27, vcc
	v_cndmask_b32_e64 v20, v28, v20, s[0:1]
	v_xor_b32_e32 v20, v20, v21
	v_sub_u32_e32 v20, v20, v21
	v_cmp_ne_u32_e32 vcc, 0, v20
	v_mov_b32_e32 v28, v18
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v27, v17
BB10_21:                                ; %Flow236
                                        ;   in Loop: Header=BB10_19 Depth=4
	s_or_saveexec_b64 s[46:47], s[46:47]
	v_mov_b32_e32 v38, v26
	v_mov_b32_e32 v41, v28
	v_mov_b32_e32 v20, s44
	v_mov_b32_e32 v32, s45
	v_mov_b32_e32 v18, v21
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v39, v23
	v_mov_b32_e32 v30, v19
	v_mov_b32_e32 v37, v25
	v_mov_b32_e32 v40, v27
	s_xor_b64 exec, exec, s[46:47]
	s_cbranch_execz BB10_23
; %bb.22:                               ; %.lr.ph240.split.us._crit_edge
                                        ;   in Loop: Header=BB10_19 Depth=4
	s_ashr_i32 s48, s18, 31
	s_add_i32 s49, s18, s48
	s_xor_b32 s49, s49, s48
	v_cvt_f32_u32_e32 v20, s49
	v_ashrrev_i32_e32 v18, 31, v29
	v_mov_b32_e32 v31, 0
	v_add_u32_e32 v29, v29, v18
	v_rcp_iflag_f32_e32 v20, v20
	v_mov_b32_e32 v34, v31
	v_xor_b32_e32 v33, v29, v18
	v_mov_b32_e32 v41, v34
	v_mul_f32_e32 v20, s25, v20
	v_cvt_u32_f32_e32 v30, v20
	v_mov_b32_e32 v38, v31
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v20, s48
	v_mul_lo_u32 v35, v30, s49
	v_mul_hi_u32 v36, v30, s49
	v_mov_b32_e32 v32, s49
	v_mov_b32_e32 v37, v30
	v_sub_u32_e32 v39, 0, v35
	v_mov_b32_e32 v40, v33
BB10_23:                                ; %Flow237
                                        ;   in Loop: Header=BB10_19 Depth=4
	s_or_b64 exec, exec, s[46:47]
	v_mov_b32_e32 v31, s45
	v_mov_b32_e32 v34, s44
	v_mov_b32_e32 v29, 1
	s_and_saveexec_b64 s[44:45], s[0:1]
	s_cbranch_execz BB10_25
; %bb.24:                               ;   in Loop: Header=BB10_19 Depth=4
	v_mov_b32_e32 v27, v40
	v_mov_b32_e32 v25, v37
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, v41
	v_mov_b32_e32 v23, v39
	v_mov_b32_e32 v24, v36
	v_mov_b32_e32 v22, v35
	v_mov_b32_e32 v26, v38
	v_mov_b32_e32 v19, v30
	v_mov_b32_e32 v31, v32
	v_mov_b32_e32 v17, v33
	v_mov_b32_e32 v34, v20
	v_mov_b32_e32 v21, v18
BB10_25:                                ; %.lr.ph.us
                                        ;   in Loop: Header=BB10_19 Depth=4
	s_or_b64 exec, exec, s[44:45]
	v_cmp_eq_u32_e32 vcc, 0, v24
	v_cndmask_b32_e32 v18, v22, v23, vcc
	v_mul_lo_u32 v20, v18, v26
	v_mul_hi_u32 v18, v18, v25
	s_mov_b32 s44, s43
	s_mov_b32 s46, s4
	v_add_u32_e32 v18, v18, v20
	v_add_u32_e32 v20, v19, v18
	v_sub_u32_e32 v18, v19, v18
	v_cndmask_b32_e32 v18, v18, v20, vcc
	v_mul_lo_u32 v19, v18, v28
	v_mul_hi_u32 v18, v18, v27
	v_xor_b32_e32 v20, v21, v34
	v_add_u32_e32 v18, v18, v19
	v_mul_lo_u32 v19, v18, v31
	v_add_u32_e32 v21, 1, v18
	v_add_u32_e32 v22, -1, v18
	v_sub_u32_e32 v23, v17, v19
	v_cmp_ge_u32_e32 vcc, v17, v19
	v_cmp_ge_u32_e64 s[0:1], v23, v31
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v17, v18, v21, s[0:1]
	v_cndmask_b32_e32 v17, v22, v17, vcc
	v_xor_b32_e32 v17, v17, v20
	v_sub_u32_e32 v18, v17, v20
	v_cmp_gt_i32_e32 vcc, s15, v18
	v_add_u32_e32 v18, v18, v15
	v_mul_lo_u32 v18, v18, s16
	v_cndmask_b32_e32 v17, 0, v29, vcc
	v_and_b32_e32 v17, v17, v16
	v_mov_b32_e32 v19, v14
	s_branch BB10_27
BB10_26:                                ;   in Loop: Header=BB10_27 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s46, s46, -1
	s_add_i32 s44, s44, 1
	s_cmp_lg_u32 s46, 0
	v_subrev_u32_e32 v19, s22, v19
	s_cbranch_scc0 BB10_18
BB10_27:                                ;   Parent Loop BB10_3 Depth=1
                                        ;     Parent Loop BB10_6 Depth=2
                                        ;       Parent Loop BB10_9 Depth=3
                                        ;         Parent Loop BB10_19 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_cmp_lt_i32_e32 vcc, -1, v19
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $sgpr45
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $sgpr47
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr28_vgpr29
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr27
                                        ; implicit-def: $vgpr26
                                        ; implicit-def: $vgpr30_vgpr31
	s_and_saveexec_b64 s[48:49], vcc
	s_xor_b64 s[48:49], exec, s[48:49]
	s_cbranch_execz BB10_29
; %bb.28:                               ;   in Loop: Header=BB10_27 Depth=5
	s_ashr_i32 s45, s19, 31
	s_add_i32 s0, s19, s45
	s_xor_b32 s47, s0, s45
	v_cvt_f32_u32_e32 v20, s47
	v_ashrrev_i32_e32 v24, 31, v19
	v_rcp_iflag_f32_e32 v20, v20
	v_mul_f32_e32 v20, s25, v20
	v_cvt_u32_f32_e32 v22, v20
	v_mul_lo_u32 v25, v22, s47
	v_mul_hi_u32 v27, v22, s47
	v_sub_u32_e32 v26, 0, v25
	v_cmp_eq_u32_e32 vcc, 0, v27
	v_cndmask_b32_e32 v20, v25, v26, vcc
	v_mul_hi_u32 v21, v20, v22
	v_add_u32_e32 v20, v19, v24
	v_xor_b32_e32 v20, v20, v24
	v_add_u32_e32 v23, v22, v21
	v_sub_u32_e32 v21, v22, v21
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_mul_hi_u32 v21, v21, v20
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v29, v23
	v_mov_b32_e32 v28, v22
	v_mul_lo_u32 v30, v21, s47
	v_mov_b32_e32 v21, v23
	v_sub_u32_e32 v23, v20, v30
	v_cmp_ge_u32_e64 s[0:1], v20, v30
	v_cmp_le_u32_e32 vcc, s47, v23
	v_subrev_u32_e32 v30, s47, v23
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v31, s47, v23
	v_cndmask_b32_e32 v23, v23, v30, vcc
	v_cndmask_b32_e64 v23, v31, v23, s[0:1]
	v_xor_b32_e32 v23, v23, v24
	v_sub_u32_e32 v23, v23, v24
	v_cmp_ne_u32_e32 vcc, 0, v23
	v_mov_b32_e32 v31, v21
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v30, v20
BB10_29:                                ; %Flow
                                        ;   in Loop: Header=BB10_27 Depth=5
	s_or_saveexec_b64 s[48:49], s[48:49]
	v_mov_b32_e32 v41, v29
	v_mov_b32_e32 v44, v31
	v_mov_b32_e32 v23, s45
	v_mov_b32_e32 v35, s47
	v_mov_b32_e32 v21, v24
	v_mov_b32_e32 v36, v20
	v_mov_b32_e32 v38, v25
	v_mov_b32_e32 v39, v27
	v_mov_b32_e32 v42, v26
	v_mov_b32_e32 v33, v22
	v_mov_b32_e32 v40, v28
	v_mov_b32_e32 v43, v30
	s_xor_b64 exec, exec, s[48:49]
	s_cbranch_execz BB10_31
; %bb.30:                               ; %._crit_edge
                                        ;   in Loop: Header=BB10_27 Depth=5
	s_ashr_i32 s50, s19, 31
	s_add_i32 s51, s19, s50
	s_xor_b32 s51, s51, s50
	v_cvt_f32_u32_e32 v23, s51
	v_ashrrev_i32_e32 v21, 31, v19
	v_mov_b32_e32 v34, 0
	v_add_u32_e32 v32, v19, v21
	v_rcp_iflag_f32_e32 v23, v23
	v_mov_b32_e32 v37, v34
	v_xor_b32_e32 v36, v32, v21
	v_mov_b32_e32 v44, v37
	v_mul_f32_e32 v23, s25, v23
	v_cvt_u32_f32_e32 v33, v23
	v_mov_b32_e32 v41, v34
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v23, s50
	v_mul_lo_u32 v38, v33, s51
	v_mul_hi_u32 v39, v33, s51
	v_mov_b32_e32 v35, s51
	v_mov_b32_e32 v40, v33
	v_sub_u32_e32 v42, 0, v38
	v_mov_b32_e32 v43, v36
BB10_31:                                ; %Flow235
                                        ;   in Loop: Header=BB10_27 Depth=5
	s_or_b64 exec, exec, s[48:49]
	v_mov_b32_e32 v34, s47
	v_mov_b32_e32 v37, s45
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[48:49], s[0:1]
	s_cbranch_execz BB10_33
; %bb.32:                               ;   in Loop: Header=BB10_27 Depth=5
	v_mov_b32_e32 v30, v43
	v_mov_b32_e32 v28, v40
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, v44
	v_mov_b32_e32 v26, v42
	v_mov_b32_e32 v27, v39
	v_mov_b32_e32 v25, v38
	v_mov_b32_e32 v29, v41
	v_mov_b32_e32 v22, v33
	v_mov_b32_e32 v34, v35
	v_mov_b32_e32 v20, v36
	v_mov_b32_e32 v37, v23
	v_mov_b32_e32 v24, v21
BB10_33:                                ;   in Loop: Header=BB10_27 Depth=5
	s_or_b64 exec, exec, s[48:49]
	v_cmp_eq_u32_e32 vcc, 0, v27
	v_cndmask_b32_e32 v21, v25, v26, vcc
	v_mul_lo_u32 v23, v21, v29
	v_mul_hi_u32 v21, v21, v28
	v_add_u32_e32 v21, v21, v23
	v_add_u32_e32 v23, v22, v21
	v_sub_u32_e32 v21, v22, v21
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_mul_lo_u32 v22, v21, v31
	v_mul_hi_u32 v21, v21, v30
	v_xor_b32_e32 v23, v24, v37
	v_add_u32_e32 v21, v21, v22
	v_mul_lo_u32 v22, v21, v34
	v_add_u32_e32 v24, 1, v21
	v_add_u32_e32 v25, -1, v21
	v_sub_u32_e32 v26, v20, v22
	v_cmp_ge_u32_e32 vcc, v20, v22
	v_cmp_ge_u32_e64 s[0:1], v26, v34
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v20, v21, v24, s[0:1]
	v_cndmask_b32_e32 v20, v25, v20, vcc
	v_xor_b32_e32 v20, v20, v23
	v_sub_u32_e32 v20, v20, v23
	v_cmp_gt_i32_e32 vcc, s16, v20
	v_cndmask_b32_e32 v21, 0, v32, vcc
	v_and_b32_e32 v21, v17, v21
	v_cmp_ne_u32_e32 vcc, 0, v21
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB10_26
; %bb.34:                               ;   in Loop: Header=BB10_27 Depth=5
	v_add_u32_e32 v20, v20, v18
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshlrev_b64 v[20:21], 1, v[20:21]
	s_ashr_i32 s45, s44, 31
	v_add_co_u32_e32 v20, vcc, v5, v20
	s_lshl_b64 s[48:49], s[44:45], 1
	v_addc_co_u32_e32 v21, vcc, v6, v21, vcc
	v_mov_b32_e32 v23, s49
	v_add_co_u32_e32 v22, vcc, s48, v3
	v_addc_co_u32_e32 v23, vcc, v4, v23, vcc
	global_load_ushort v22, v[22:23], off
	global_load_ushort v20, v[20:21], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v22, v22
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v20, v20
	v_cvt_f64_f32_e32 v[22:23], v22
	v_cvt_f64_f32_e32 v[20:21], v20
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[20:21], v[22:23], v[10:11]
	s_branch BB10_26
BB10_35:                                ; %Flow247
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_bwd_ncdhw_half
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 45
		.amdhsa_next_free_sgpr 52
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end10:
	.size	naive_conv_bwd_ncdhw_half, .Lfunc_end10-naive_conv_bwd_ncdhw_half
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 3644
; NumSgprs: 54
; NumVgprs: 45
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 45
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_wrw_ncdhw_half ; -- Begin function naive_conv_wrw_ncdhw_half
	.globl	naive_conv_wrw_ncdhw_half
	.p2align	8
	.type	naive_conv_wrw_ncdhw_half,@function
naive_conv_wrw_ncdhw_half:              ; @naive_conv_wrw_ncdhw_half
naive_conv_wrw_ncdhw_half$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_load_dwordx2 s[24:25], s[4:5], 0x68
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s24, s31
	s_mul_i32 s26, s7, s13
	s_mul_i32 s26, s26, s30
	v_cmp_gt_i32_e32 vcc, s26, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB11_18
; %bb.1:                                ; %.lr.ph241
	s_ashr_i32 s0, s12, 31
	s_add_i32 s1, s12, s0
	s_xor_b32 s41, s1, s0
	v_cvt_f32_u32_e32 v1, s41
	s_mov_b32 s27, 0x4f800000
	s_ashr_i32 s1, s6, 31
	s_add_i32 s33, s6, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s33, s33, s1
	s_xor_b32 s40, s1, s0
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[36:37], s[4:5], 0x10
	v_mul_f32_e32 v1, s27, v1
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s5, s10, 31
	s_mul_i32 s39, s9, s8
	s_ashr_i32 s4, s13, 31
	v_mul_lo_u32 v2, v1, s41
	v_mul_hi_u32 v3, v1, s41
	s_mul_i32 s5, s39, s5
	s_mul_hi_i32 s38, s9, s8
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_mul_i32 s38, s38, s10
	s_sub_i32 s29, 0, s29
	s_sub_i32 s42, 0, s28
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_mul_hi_u32 v1, v1, s33
	s_sub_i32 s43, 0, s23
	s_mul_i32 s44, s18, s10
	s_mov_b64 s[46:47], 0
	v_mul_lo_u32 v2, v1, s41
	v_add_u32_e32 v3, 1, v1
	v_add_u32_e32 v4, -1, v1
	v_sub_u32_e32 v5, s33, v2
	v_cmp_ge_u32_e32 vcc, s33, v2
	v_cmp_le_u32_e64 s[0:1], s41, v5
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	v_xor_b32_e32 v1, s40, v1
	v_subrev_u32_e32 v5, s40, v1
	s_mul_hi_u32 s0, s39, s10
	s_mul_i32 s1, s39, s10
	s_add_i32 s0, s0, s5
	v_mul_lo_u32 v2, v5, s12
	v_ashrrev_i32_e32 v1, 31, v5
	s_mul_i32 s4, s1, s4
	s_mul_hi_u32 s5, s1, s13
	s_mul_i32 s1, s1, s13
	v_mul_lo_u32 v4, s1, v1
	v_mul_hi_u32 v6, s1, v5
	s_add_i32 s0, s0, s38
	v_sub_u32_e32 v3, s6, v2
	s_add_i32 s4, s5, s4
	v_add_u32_e32 v2, v6, v4
	s_mul_i32 s0, s0, s13
	v_ashrrev_i32_e32 v4, 31, v3
	s_add_i32 s4, s4, s0
	v_mul_lo_u32 v1, s1, v5
	v_mad_i64_i32 v[3:4], s[0:1], v5, s12, v[3:4]
	v_mul_lo_u32 v7, s4, v5
	s_mul_i32 s1, s30, s13
	s_ashr_i32 s4, s31, 31
	s_mul_hi_i32 s0, s30, s13
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s3
	s_mul_i32 s3, s1, s4
	s_mul_hi_u32 s6, s1, s31
	s_ashr_i32 s5, s24, 31
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s31
	s_mul_i32 s1, s1, s31
	s_add_i32 s0, s3, s0
	s_mul_i32 s3, s1, s5
	s_mul_hi_u32 s6, s1, s24
	v_add_u32_e32 v2, v2, v7
	s_mul_i32 s1, s1, s24
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s24
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_add_i32 s3, s3, s0
	v_mul_lo_u32 v7, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_mul_lo_u32 v9, s3, v3
	v_add_co_u32_e32 v1, vcc, s2, v1
	s_mul_i32 s2, s15, s14
	s_ashr_i32 s0, s16, 31
	v_mul_lo_u32 v5, s1, v3
	s_mul_hi_i32 s1, s15, s14
	s_mul_i32 s0, s2, s0
	s_mul_hi_u32 s3, s2, s16
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v8, v7
	s_mul_i32 s2, s2, s16
	s_add_i32 s0, s3, s0
	s_mul_i32 s1, s1, s16
	v_add_u32_e32 v6, v6, v9
	s_add_i32 s0, s0, s1
	v_mul_lo_u32 v9, s2, v4
	v_mul_hi_u32 v10, s2, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s2, v3
	s_mul_i32 s0, s25, s16
	s_mul_i32 s0, s0, s15
	v_add_co_u32_e32 v3, vcc, s34, v5
	v_mov_b32_e32 v8, s35
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	s_mul_i32 s0, s0, s14
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	s_mul_i32 s12, s0, s12
	s_mul_i32 s0, s25, s13
	s_mul_i32 s0, s0, s10
	s_mul_i32 s13, s0, s9
	s_mul_i32 s25, s17, s10
	v_mov_b32_e32 v7, s37
	v_add_co_u32_e32 v5, vcc, s36, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	s_mul_i32 s6, s7, s30
	v_cmp_gt_i32_e64 s[34:35], s11, 0
	v_cmp_gt_i32_e64 s[36:37], s14, 0
	v_cmp_gt_i32_e64 s[38:39], s15, 0
	v_cmp_gt_i32_e64 s[40:41], s16, 0
	s_mul_i32 s33, s16, s15
	s_mul_i32 s13, s13, s8
	s_mul_i32 s25, s25, s9
	s_branch BB11_3
BB11_2:                                 ; %._crit_edge237
                                        ;   in Loop: Header=BB11_3 Depth=1
	v_mul_lo_u32 v10, v10, s30
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s26, v0
	s_or_b64 s[46:47], vcc, s[46:47]
	v_add_u32_e32 v9, v10, v9
	v_mul_lo_u32 v9, v9, s31
	v_cvt_f32_f64_e32 v10, v[11:12]
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s24
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_cvt_f16_f32_e32 v9, v10
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v3, v7
	v_addc_co_u32_e64 v8, s[0:1], v4, v8, s[0:1]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[46:47]
	s_cbranch_execz BB11_18
BB11_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB11_6 Depth 2
                                        ;       Child Loop BB11_9 Depth 3
                                        ;         Child Loop BB11_13 Depth 4
                                        ;           Child Loop BB11_16 Depth 5
	s_add_i32 s0, s24, s5
	s_xor_b32 s0, s0, s5
	v_cvt_f32_u32_e32 v7, s0
	v_ashrrev_i32_e32 v8, 31, v0
	v_add_u32_e32 v12, v0, v8
	s_add_i32 s1, s31, s4
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s2, s1, s4
	v_cvt_f32_u32_e32 v9, s2
	s_ashr_i32 s45, s7, 31
	v_mul_f32_e32 v7, s27, v7
	v_cvt_u32_f32_e32 v7, v7
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s27, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s5, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s2
	v_mul_hi_u32 v12, v9, s2
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s45
	s_xor_b32 s3, s0, s45
	v_cvt_f32_u32_e32 v11, s3
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_rcp_iflag_f32_e32 v11, v11
	v_mul_hi_u32 v10, v10, v9
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_mul_f32_e32 v11, s27, v11
	v_cvt_u32_f32_e32 v11, v11
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v10, v11, s3
	v_mul_hi_u32 v13, v11, s3
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_xor_b32_e32 v12, v12, v8
	v_sub_u32_e32 v14, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_mul_hi_u32 v9, v9, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v11
	v_mul_lo_u32 v7, v7, s24
	v_mul_lo_u32 v9, v9, s2
	v_add_u32_e32 v14, v11, v10
	v_sub_u32_e32 v10, v11, v10
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v0
	v_sub_u32_e32 v13, v12, v9
	v_cmp_ge_u32_e32 vcc, v12, v9
	v_cmp_le_u32_e64 s[0:1], s2, v13
	v_add_u32_e32 v11, s2, v13
	v_subrev_u32_e32 v9, s2, v13
	s_ashr_i32 s2, s30, 31
	s_add_i32 s48, s30, s2
	s_xor_b32 s48, s48, s2
	v_mul_lo_u32 v12, v10, s3
	v_cvt_f32_u32_e32 v14, s48
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v9, v13, v9, s[0:1]
	v_sub_u32_e32 v13, v0, v12
	v_cmp_le_u32_e64 s[0:1], s3, v13
	v_rcp_iflag_f32_e32 v13, v14
	v_cmp_ge_u32_e64 s[2:3], v0, v12
	v_add_u32_e32 v12, 1, v10
	s_and_b64 s[0:1], s[0:1], s[2:3]
	v_mul_f32_e32 v13, s27, v13
	v_cvt_u32_f32_e32 v13, v13
	v_add_u32_e32 v14, -1, v10
	v_cndmask_b32_e64 v10, v10, v12, s[0:1]
	v_cndmask_b32_e64 v10, v14, v10, s[2:3]
	v_xor_b32_e32 v10, s45, v10
	v_mul_hi_u32 v14, v13, s48
	v_subrev_u32_e32 v10, s45, v10
	v_mul_lo_u32 v12, v13, s48
	s_ashr_i32 s45, s6, 31
	s_add_i32 s2, s6, s45
	s_xor_b32 s49, s2, s45
	v_cmp_eq_u32_e64 s[0:1], 0, v14
	v_cvt_f32_u32_e32 v14, s49
	v_sub_u32_e32 v16, 0, v12
	v_cndmask_b32_e64 v12, v12, v16, s[0:1]
	v_mul_hi_u32 v12, v12, v13
	v_rcp_iflag_f32_e32 v14, v14
	v_ashrrev_i32_e32 v15, 31, v10
	v_add_u32_e32 v10, v10, v15
	v_add_u32_e32 v16, v13, v12
	v_sub_u32_e32 v12, v13, v12
	v_mul_f32_e32 v13, s27, v14
	v_xor_b32_e32 v10, v10, v15
	v_cndmask_b32_e64 v12, v12, v16, s[0:1]
	v_cvt_u32_f32_e32 v13, v13
	v_mul_hi_u32 v12, v12, v10
	v_cndmask_b32_e32 v9, v11, v9, vcc
	v_xor_b32_e32 v9, v9, v8
	v_mul_hi_u32 v14, v13, s49
	v_mul_lo_u32 v11, v12, s48
	v_mul_lo_u32 v12, v13, s49
	v_sub_u32_e32 v8, v9, v8
	v_cmp_eq_u32_e32 vcc, 0, v14
	v_sub_u32_e32 v9, v10, v11
	v_sub_u32_e32 v16, 0, v12
	v_cndmask_b32_e32 v12, v12, v16, vcc
	v_mul_hi_u32 v12, v12, v13
	v_cmp_ge_u32_e64 s[2:3], v10, v11
	v_cmp_le_u32_e64 s[0:1], s48, v9
	v_add_u32_e32 v14, s48, v9
	v_add_u32_e32 v10, v13, v12
	v_sub_u32_e32 v11, v13, v12
	v_cndmask_b32_e32 v10, v11, v10, vcc
	v_mul_hi_u32 v10, v10, v0
	v_subrev_u32_e32 v11, s48, v9
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_mul_lo_u32 v11, v10, s49
	v_cndmask_b32_e64 v9, v14, v9, s[2:3]
	v_xor_b32_e32 v9, v9, v15
	v_sub_u32_e32 v7, v0, v7
	v_sub_u32_e32 v12, v0, v11
	v_cmp_le_u32_e32 vcc, s49, v12
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v10
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v12, -1, v10
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_cndmask_b32_e64 v10, v12, v10, s[0:1]
	v_xor_b32_e32 v10, s45, v10
	v_mov_b32_e32 v11, 0
	v_sub_u32_e32 v9, v9, v15
	v_subrev_u32_e32 v10, s45, v10
	s_andn2_b64 vcc, exec, s[34:35]
	v_mov_b32_e32 v12, 0
	s_cbranch_vccnz BB11_2
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB11_3 Depth=1
	v_mul_lo_u32 v13, v9, s20
	v_mul_lo_u32 v11, s8, v10
	v_mul_lo_u32 v14, v8, s21
	v_mul_lo_u32 v16, s22, v7
	s_mov_b32 s2, 0
	v_add3_u32 v11, s43, v13, v11
	v_mul_lo_u32 v11, s9, v11
	v_add_u32_e32 v15, s29, v16
	s_mov_b32 s3, 0
	v_add3_u32 v11, s42, v11, v14
	v_mul_lo_u32 v17, s10, v11
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_add3_u32 v16, s29, v17, v16
	s_branch BB11_6
BB11_5:                                 ; %Flow85
                                        ;   in Loop: Header=BB11_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s12
	s_cmp_eq_u32 s3, s11
	v_add_u32_e32 v16, s13, v16
	s_cbranch_scc1 BB11_2
BB11_6:                                 ; %.preheader
                                        ;   Parent Loop BB11_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB11_9 Depth 3
                                        ;         Child Loop BB11_13 Depth 4
                                        ;           Child Loop BB11_16 Depth 5
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB11_5
; %bb.7:                                ; %.lr.ph231
                                        ;   in Loop: Header=BB11_6 Depth=2
	s_mov_b32 s45, 0
	v_mov_b32_e32 v17, v16
	s_mov_b32 s48, s2
	s_branch BB11_9
BB11_8:                                 ; %._crit_edge226
                                        ;   in Loop: Header=BB11_9 Depth=3
	s_add_i32 s45, s45, 1
	s_add_i32 s48, s48, s33
	s_cmp_eq_u32 s45, s14
	v_add_u32_e32 v17, s25, v17
	s_cbranch_scc1 BB11_5
BB11_9:                                 ;   Parent Loop BB11_3 Depth=1
                                        ;     Parent Loop BB11_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB11_13 Depth 4
                                        ;           Child Loop BB11_16 Depth 5
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB11_8
; %bb.10:                               ; %.lr.ph225
                                        ;   in Loop: Header=BB11_9 Depth=3
	s_andn2_b64 vcc, exec, s[40:41]
	s_cbranch_vccnz BB11_8
; %bb.11:                               ; %.lr.ph225.split.us.preheader
                                        ;   in Loop: Header=BB11_9 Depth=3
	s_mul_i32 s0, s45, s17
	s_sub_i32 s0, s0, s23
	v_add_u32_e32 v18, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s8, v18
	s_and_b64 s[50:51], vcc, s[0:1]
	s_mov_b32 s49, 0
	v_mov_b32_e32 v18, v17
	s_mov_b32 s52, s48
	s_branch BB11_13
BB11_12:                                ; %Flow81
                                        ;   in Loop: Header=BB11_13 Depth=4
	s_or_b64 exec, exec, s[54:55]
	s_add_i32 s49, s49, 1
	s_add_i32 s52, s52, s16
	s_cmp_lg_u32 s49, s15
	v_add_u32_e32 v18, s44, v18
	s_cbranch_scc0 BB11_8
BB11_13:                                ; %.lr.ph225.split.us
                                        ;   Parent Loop BB11_3 Depth=1
                                        ;     Parent Loop BB11_6 Depth=2
                                        ;       Parent Loop BB11_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB11_16 Depth 5
	s_mul_i32 s0, s49, s18
	s_sub_i32 s0, s0, s28
	v_add_u32_e32 v19, s0, v14
	v_cmp_lt_i32_e32 vcc, -1, v19
	v_cmp_gt_i32_e64 s[0:1], s9, v19
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 s[0:1], s[50:51], s[0:1]
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz BB11_12
; %bb.14:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB11_13 Depth=4
	s_mov_b32 s53, 0
	s_mov_b32 s56, s52
	s_mov_b32 s58, s16
	s_branch BB11_16
BB11_15:                                ;   in Loop: Header=BB11_16 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s58, s58, -1
	s_add_i32 s56, s56, 1
	s_add_i32 s53, s53, s19
	s_cmp_lg_u32 s58, 0
	s_cbranch_scc0 BB11_12
BB11_16:                                ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB11_3 Depth=1
                                        ;     Parent Loop BB11_6 Depth=2
                                        ;       Parent Loop BB11_9 Depth=3
                                        ;         Parent Loop BB11_13 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_add_u32_e32 v19, s53, v15
	v_cmp_lt_i32_e32 vcc, -1, v19
	v_cmp_gt_i32_e64 s[0:1], s10, v19
	s_and_b64 s[60:61], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[60:61]
	s_cbranch_execz BB11_15
; %bb.17:                               ;   in Loop: Header=BB11_16 Depth=5
	v_add_u32_e32 v19, s53, v18
	v_ashrrev_i32_e32 v20, 31, v19
	v_lshlrev_b64 v[19:20], 1, v[19:20]
	s_ashr_i32 s57, s56, 31
	v_add_co_u32_e32 v19, vcc, v1, v19
	s_lshl_b64 s[60:61], s[56:57], 1
	v_addc_co_u32_e32 v20, vcc, v2, v20, vcc
	v_mov_b32_e32 v22, s61
	v_add_co_u32_e32 v21, vcc, s60, v5
	v_addc_co_u32_e32 v22, vcc, v6, v22, vcc
	global_load_ushort v21, v[21:22], off
	global_load_ushort v19, v[19:20], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v21, v21
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v19, v19
	v_cvt_f64_f32_e32 v[21:22], v21
	v_cvt_f64_f32_e32 v[19:20], v19
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[11:12], v[19:20], v[21:22], v[11:12]
	s_branch BB11_15
BB11_18:                                ; %Flow89
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_wrw_ncdhw_half
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 23
		.amdhsa_next_free_sgpr 62
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end11:
	.size	naive_conv_wrw_ncdhw_half, .Lfunc_end11-naive_conv_wrw_ncdhw_half
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2076
; NumSgprs: 64
; NumVgprs: 23
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 64
; NumVGPRsForWavesPerEU: 23
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_fwd_nchw_ushort ; -- Begin function naive_conv_fwd_nchw_ushort
	.globl	naive_conv_fwd_nchw_ushort
	.p2align	8
	.type	naive_conv_fwd_nchw_ushort,@function
naive_conv_fwd_nchw_ushort:               ; @naive_conv_fwd_nchw_ushort
naive_conv_fwd_nchw_ushort$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s14, s13
	v_cmp_gt_i32_e32 vcc, s7, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB12_19
; %bb.1:                                ; %.lr.ph183
	s_ashr_i32 s0, s11, 31
	s_add_i32 s1, s11, s0
	s_xor_b32 s3, s1, s0
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s24, 0x4f800000
	s_ashr_i32 s1, s10, 31
	s_add_i32 s2, s10, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s31, s2, s1
	v_cvt_f32_u32_e32 v4, s31
	s_ashr_i32 s25, s6, 31
	v_mul_f32_e32 v1, s24, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s1, s6, s25
	s_xor_b32 s30, s1, s25
	s_xor_b32 s2, s25, s0
	v_mul_lo_u32 v2, v1, s3
	v_mul_hi_u32 v3, v1, s3
	s_mul_i32 s34, s23, s11
	s_mul_i32 s23, s23, s12
	v_sub_u32_e32 v5, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v5, vcc
	v_mul_hi_u32 v2, v2, v1
	v_rcp_iflag_f32_e32 v3, v4
	s_load_dwordx2 s[26:27], s[4:5], 0x0
	s_load_dwordx2 s[28:29], s[4:5], 0x8
	s_load_dwordx2 s[4:5], s[4:5], 0x10
	s_sub_i32 s20, 0, s20
	v_add_u32_e32 v4, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s30
	v_mul_f32_e32 v2, s24, v3
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s3
	v_add_u32_e32 v4, 1, v1
	v_add_u32_e32 v5, -1, v1
	v_sub_u32_e32 v6, s30, v3
	v_cmp_ge_u32_e32 vcc, s30, v3
	v_cmp_le_u32_e64 s[0:1], s3, v6
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	s_mul_i32 s0, s11, s10
	v_mul_hi_u32 v4, v2, s31
	s_ashr_i32 s10, s0, 31
	v_mul_lo_u32 v3, v2, s31
	s_add_i32 s0, s0, s10
	s_xor_b32 s33, s0, s10
	v_cndmask_b32_e32 v1, v5, v1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s33
	v_sub_u32_e32 v6, 0, v3
	v_cndmask_b32_e32 v3, v3, v6, vcc
	v_mul_hi_u32 v3, v3, v2
	v_rcp_iflag_f32_e32 v4, v4
	v_xor_b32_e32 v1, s2, v1
	v_subrev_u32_e32 v5, s2, v1
	v_add_u32_e32 v7, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_mul_f32_e32 v3, s24, v4
	v_cvt_u32_f32_e32 v3, v3
	v_ashrrev_i32_e32 v1, 31, v5
	v_add_u32_e32 v6, v5, v1
	v_cndmask_b32_e32 v2, v2, v7, vcc
	v_mul_lo_u32 v4, v3, s33
	v_mul_hi_u32 v7, v3, s33
	v_xor_b32_e32 v6, v6, v1
	v_mul_hi_u32 v2, v2, v6
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v7
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_lo_u32 v2, v2, s31
	v_mul_hi_u32 v4, v4, v3
	v_sub_u32_e32 v8, v6, v2
	v_cmp_ge_u32_e64 s[2:3], v6, v2
	v_add_u32_e32 v6, v3, v4
	v_sub_u32_e32 v3, v3, v4
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	v_mul_hi_u32 v3, v3, s30
	v_cmp_le_u32_e32 vcc, s31, v8
	v_subrev_u32_e32 v2, s31, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_mul_lo_u32 v4, v3, s33
	v_add_u32_e32 v7, s31, v8
	v_cndmask_b32_e32 v2, v8, v2, vcc
	v_cndmask_b32_e64 v2, v7, v2, s[2:3]
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v7, v2, v1
	v_sub_u32_e32 v1, s30, v4
	v_cmp_le_u32_e32 vcc, s33, v1
	v_cmp_ge_u32_e64 s[0:1], s30, v4
	v_add_u32_e32 v2, 1, v3
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v1, -1, v3
	v_cndmask_b32_e32 v2, v3, v2, vcc
	v_cndmask_b32_e64 v3, v1, v2, s[0:1]
	v_mul_hi_i32 v2, v7, s23
	v_mul_lo_u32 v1, v7, s23
	s_xor_b32 s2, s25, s10
	v_xor_b32_e32 v3, s2, v3
	v_subrev_u32_e32 v6, s2, v3
	v_mad_i64_i32 v[1:2], s[0:1], v6, s12, v[1:2]
	v_mul_lo_u32 v3, v5, s11
	s_mul_i32 s2, s9, s8
	s_mul_hi_i32 s0, s9, s8
	v_mul_lo_u32 v2, s2, v2
	v_mul_hi_u32 v4, s2, v1
	v_mul_lo_u32 v5, s0, v1
	v_mul_lo_u32 v1, s2, v1
	v_sub_u32_e32 v3, s6, v3
	v_add_u32_e32 v2, v4, v2
	v_ashrrev_i32_e32 v4, 31, v3
	v_mad_i64_i32 v[3:4], s[0:1], v6, s11, v[3:4]
	v_add_u32_e32 v2, v2, v5
	s_mul_i32 s1, s21, s12
	s_ashr_i32 s3, s22, 31
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_mul_hi_i32 s0, s21, s12
	s_mul_i32 s3, s1, s3
	s_mul_hi_u32 s6, s1, s22
	s_mul_i32 s1, s1, s22
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s22
	s_add_i32 s3, s3, s0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, s27
	v_add_co_u32_e32 v1, vcc, s26, v1
	v_mul_lo_u32 v6, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_mul_lo_u32 v9, s3, v3
	v_addc_co_u32_e32 v2, vcc, v5, v2, vcc
	v_mul_lo_u32 v5, s1, v3
	v_mad_i64_i32 v[3:4], s[0:1], v7, s34, v[3:4]
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	s_mul_hi_i32 s0, s14, s13
	v_mul_lo_u32 v9, s7, v4
	v_mul_hi_u32 v10, s7, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s7, v3
	v_add_co_u32_e32 v3, vcc, s28, v5
	v_mov_b32_e32 v8, s29
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	v_mov_b32_e32 v7, s5
	v_add_co_u32_e32 v5, vcc, s4, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[4:5], s12, 0
	v_cmp_gt_i32_e64 s[10:11], s21, 0
	v_cmp_gt_i32_e64 s[26:27], s22, 0
	s_mul_i32 s3, s22, s21
	s_sub_i32 s6, 0, s19
	s_mul_i32 s13, s17, s9
	s_mov_b64 s[28:29], 0
	s_mov_b32 s23, 0x7f800000
	s_movk_i32 s25, 0x7fff
	s_branch BB12_3
BB12_2:                                 ; %_Z19__float_to_bfloat16f.exit
                                        ;   in Loop: Header=BB12_3 Depth=1
	s_or_b64 exec, exec, s[30:31]
	v_mul_lo_u32 v7, v7, s14
	v_add_u32_e32 v0, 0x100, v0
	v_lshrrev_b32_e32 v9, 16, v10
	v_add_u32_e32 v7, v7, v8
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e32 v7, vcc, v5, v7
	v_addc_co_u32_e32 v8, vcc, v6, v8, vcc
	v_cmp_le_i32_e32 vcc, s7, v0
	s_or_b64 s[28:29], vcc, s[28:29]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz BB12_19
BB12_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB12_6 Depth 2
                                        ;       Child Loop BB12_10 Depth 3
                                        ;         Child Loop BB12_13 Depth 4
	s_ashr_i32 s0, s14, 31
	s_add_i32 s1, s14, s0
	s_xor_b32 s1, s1, s0
	v_cvt_f32_u32_e32 v7, s1
	v_rcp_iflag_f32_e32 v7, v7
	v_mul_f32_e32 v7, s24, v7
	v_cvt_u32_f32_e32 v7, v7
	v_mul_lo_u32 v8, v7, s1
	v_mul_hi_u32 v9, v7, s1
	v_sub_u32_e32 v10, 0, v8
	v_cmp_eq_u32_e32 vcc, 0, v9
	v_cndmask_b32_e32 v8, v8, v10, vcc
	v_mul_hi_u32 v8, v8, v7
	v_ashrrev_i32_e32 v9, 31, v0
	v_add_u32_e32 v10, v0, v9
	v_xor_b32_e32 v10, v10, v9
	v_add_u32_e32 v11, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_mul_hi_u32 v7, v7, v10
	v_xor_b32_e32 v9, s0, v9
	v_mul_lo_u32 v8, v7, s1
	v_add_u32_e32 v11, 1, v7
	v_add_u32_e32 v12, -1, v7
	v_sub_u32_e32 v13, v10, v8
	v_cmp_ge_u32_e32 vcc, v10, v8
	v_cmp_le_u32_e64 s[0:1], s1, v13
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v11, s[0:1]
	v_cndmask_b32_e32 v7, v12, v7, vcc
	v_xor_b32_e32 v7, v7, v9
	v_sub_u32_e32 v7, v7, v9
	v_mul_lo_u32 v8, v7, s14
	v_mov_b32_e32 v9, 0
	s_andn2_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v8, v0, v8
	s_cbranch_vccnz BB12_15
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB12_3 Depth=1
	v_mul_lo_u32 v9, v7, s15
	v_mul_lo_u32 v10, s16, v8
	s_mov_b32 s30, 0
	s_mov_b32 s31, 0
	v_subrev_u32_e32 v11, s19, v9
	v_add_u32_e32 v9, s6, v9
	v_mul_lo_u32 v12, s9, v9
	v_add_u32_e32 v13, s20, v10
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	s_branch BB12_6
BB12_5:                                 ; %._crit_edge174
                                        ;   in Loop: Header=BB12_6 Depth=2
	s_add_i32 s31, s31, 1
	s_add_i32 s30, s30, s3
	s_cmp_eq_u32 s31, s12
	v_add_u32_e32 v12, s2, v12
	s_cbranch_scc1 BB12_15
BB12_6:                                 ; %.preheader
                                        ;   Parent Loop BB12_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB12_10 Depth 3
                                        ;         Child Loop BB12_13 Depth 4
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz BB12_5
; %bb.7:                                ; %.lr.ph173
                                        ;   in Loop: Header=BB12_6 Depth=2
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz BB12_5
; %bb.8:                                ; %.lr.ph173.split.us.preheader
                                        ;   in Loop: Header=BB12_6 Depth=2
	s_mov_b32 s33, 0
	v_mov_b32_e32 v14, v12
	s_mov_b32 s34, s30
	s_branch BB12_10
BB12_9:                                 ; %Flow60
                                        ;   in Loop: Header=BB12_10 Depth=3
	s_or_b64 exec, exec, s[36:37]
	s_add_i32 s33, s33, 1
	s_add_i32 s34, s34, s22
	s_cmp_eq_u32 s33, s21
	v_add_u32_e32 v14, s13, v14
	s_cbranch_scc1 BB12_5
BB12_10:                                ; %.lr.ph173.split.us
                                        ;   Parent Loop BB12_3 Depth=1
                                        ;     Parent Loop BB12_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB12_13 Depth 4
	s_mul_i32 s0, s33, s17
	v_add_u32_e32 v15, s0, v11
	v_cmp_lt_i32_e32 vcc, -1, v15
	v_cmp_gt_i32_e64 s[0:1], s8, v15
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz BB12_9
; %bb.11:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB12_10 Depth=3
	v_mov_b32_e32 v15, v13
	s_mov_b32 s38, s34
	s_mov_b32 s35, s22
	s_branch BB12_13
BB12_12:                                ;   in Loop: Header=BB12_13 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s35, s35, -1
	s_add_i32 s38, s38, 1
	s_cmp_lg_u32 s35, 0
	v_add_u32_e32 v15, s18, v15
	s_cbranch_scc0 BB12_9
BB12_13:                                ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB12_3 Depth=1
                                        ;     Parent Loop BB12_6 Depth=2
                                        ;       Parent Loop BB12_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v15
	v_cmp_gt_i32_e64 s[0:1], s9, v15
	s_and_b64 s[40:41], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[40:41]
	s_cbranch_execz BB12_12
; %bb.14:                               ;   in Loop: Header=BB12_13 Depth=4
	v_add_u32_e32 v16, v14, v15
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshlrev_b64 v[16:17], 1, v[16:17]
	s_ashr_i32 s39, s38, 31
	v_add_co_u32_e32 v16, vcc, v1, v16
	s_lshl_b64 s[40:41], s[38:39], 1
	v_addc_co_u32_e32 v17, vcc, v2, v17, vcc
	v_mov_b32_e32 v19, s41
	v_add_co_u32_e32 v18, vcc, s40, v3
	v_addc_co_u32_e32 v19, vcc, v4, v19, vcc
	global_load_ushort v18, v[18:19], off
	global_load_ushort v16, v[16:17], off
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v18, 16, v18
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	v_cvt_f64_f32_e32 v[16:17], v16
	v_cvt_f64_f32_e32 v[18:19], v18
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[9:10], v[16:17], v[18:19], v[9:10]
	s_branch BB12_12
BB12_15:                                ; %._crit_edge179
                                        ;   in Loop: Header=BB12_3 Depth=1
	v_cvt_f32_f64_e32 v9, v[9:10]
	v_and_b32_e32 v10, s23, v9
	v_cmp_ne_u32_e32 vcc, s23, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.16:                               ;   in Loop: Header=BB12_3 Depth=1
	v_bfe_u32 v10, v9, 16, 1
	v_add3_u32 v10, v9, v10, s25
; %bb.17:                               ; %Flow
                                        ;   in Loop: Header=BB12_3 Depth=1
	s_or_saveexec_b64 s[30:31], s[0:1]
	s_xor_b64 exec, exec, s[30:31]
	s_cbranch_execz BB12_2
; %bb.18:                               ;   in Loop: Header=BB12_3 Depth=1
	v_mov_b32_e32 v10, 0
	v_or_b32_e32 v11, 0x10000, v9
	v_cmp_eq_u32_sdwa s[0:1], v9, v10 src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v10, v11, v9, s[0:1]
	s_branch BB12_2
BB12_19:                                ; %Flow66
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_fwd_nchw_ushort
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 42
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end12:
	.size	naive_conv_fwd_nchw_ushort, .Lfunc_end12-naive_conv_fwd_nchw_ushort
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 1556
; NumSgprs: 44
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 20
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_bwd_nchw_ushort ; -- Begin function naive_conv_bwd_nchw_ushort
	.globl	naive_conv_bwd_nchw_ushort
	.p2align	8
	.type	naive_conv_bwd_nchw_ushort,@function
naive_conv_bwd_nchw_ushort:               ; @naive_conv_bwd_nchw_ushort
naive_conv_bwd_nchw_ushort$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s9, s8
	v_cmp_gt_i32_e32 vcc, s7, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB13_30
; %bb.1:                                ; %.lr.ph194
	s_ashr_i32 s25, s12, 31
	s_add_i32 s0, s12, s25
	s_xor_b32 s3, s0, s25
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s24, 0x4f800000
	s_ashr_i32 s0, s10, 31
	s_add_i32 s1, s10, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s28, s1, s0
	s_ashr_i32 s26, s6, 31
	v_cvt_f32_u32_e32 v2, s28
	v_mul_f32_e32 v1, s24, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s0, s6, s26
	s_xor_b32 s27, s0, s26
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s3
	v_mul_hi_u32 v4, v1, s3
	s_xor_b32 s2, s26, s25
	v_mul_f32_e32 v2, s24, v2
	v_sub_u32_e32 v5, 0, v3
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_mul_hi_u32 v3, v3, v1
	v_cvt_u32_f32_e32 v2, v2
	s_mul_i32 s30, s23, s12
	s_mul_hi_i32 s8, s9, s8
	v_add_u32_e32 v4, v1, v3
	v_sub_u32_e32 v1, v1, v3
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s27
	v_mul_hi_u32 v3, v2, s28
	v_mul_lo_u32 v5, v2, s28
	v_mul_lo_u32 v4, v1, s3
	v_add_u32_e32 v6, 1, v1
	v_add_u32_e32 v7, -1, v1
	v_sub_u32_e32 v8, s27, v4
	v_cmp_ge_u32_e32 vcc, s27, v4
	v_cmp_le_u32_e64 s[0:1], s3, v8
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v6, s[0:1]
	s_mul_i32 s0, s12, s10
	s_ashr_i32 s10, s0, 31
	s_add_i32 s0, s0, s10
	v_cndmask_b32_e32 v1, v7, v1, vcc
	v_sub_u32_e32 v7, 0, v5
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_xor_b32 s29, s0, s10
	v_cndmask_b32_e32 v3, v5, v7, vcc
	v_cvt_f32_u32_e32 v5, s29
	v_xor_b32_e32 v1, s2, v1
	v_mul_hi_u32 v3, v3, v2
	v_subrev_u32_e32 v1, s2, v1
	v_rcp_iflag_f32_e32 v5, v5
	v_ashrrev_i32_e32 v6, 31, v1
	v_mul_lo_u32 v4, v1, s12
	v_add_u32_e32 v1, v1, v6
	v_xor_b32_e32 v7, v1, v6
	v_add_u32_e32 v1, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_cndmask_b32_e32 v1, v2, v1, vcc
	v_mul_f32_e32 v2, s24, v5
	v_mul_hi_u32 v1, v1, v7
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s28
	v_sub_u32_e32 v1, s6, v4
	v_mul_lo_u32 v4, v2, s29
	v_mul_hi_u32 v5, v2, s29
	v_sub_u32_e32 v8, v7, v3
	v_cmp_ge_u32_e64 s[2:3], v7, v3
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v5
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_hi_u32 v4, v4, v2
	v_cmp_le_u32_e32 vcc, s28, v8
	v_subrev_u32_e32 v3, s28, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_add_u32_e32 v7, v2, v4
	v_sub_u32_e32 v2, v2, v4
	v_cndmask_b32_e64 v2, v2, v7, s[0:1]
	v_mul_hi_u32 v2, v2, s27
	v_add_u32_e32 v5, s28, v8
	v_cndmask_b32_e32 v3, v8, v3, vcc
	v_cndmask_b32_e64 v3, v5, v3, s[2:3]
	v_mul_lo_u32 v4, v2, s29
	v_xor_b32_e32 v3, v3, v6
	v_sub_u32_e32 v5, v3, v6
	s_xor_b32 s2, s26, s10
	v_sub_u32_e32 v3, s27, v4
	v_cmp_le_u32_e32 vcc, s29, v3
	v_cmp_ge_u32_e64 s[0:1], s27, v4
	v_add_u32_e32 v4, 1, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v3, -1, v2
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e64 v6, v3, v2, s[0:1]
	v_ashrrev_i32_e32 v2, 31, v1
	v_mad_i64_i32 v[3:4], s[0:1], v5, s30, v[1:2]
	v_xor_b32_e32 v6, s2, v6
	v_subrev_u32_e32 v7, s2, v6
	s_mul_i32 s6, s23, s11
	v_mad_i64_i32 v[3:4], s[0:1], v7, s12, v[3:4]
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[2:3], s[4:5], 0x8
	s_load_dwordx2 s[4:5], s[4:5], 0x10
	v_cmp_gt_i32_e64 s[26:27], s22, 0
	s_mov_b64 s[28:29], 0
	v_mul_lo_u32 v4, s7, v4
	v_mul_hi_u32 v6, s7, v3
	v_mul_lo_u32 v8, s8, v3
	v_mul_lo_u32 v3, s7, v3
	s_mul_i32 s8, s22, s21
	v_add_u32_e32 v4, v6, v4
	v_mul_lo_u32 v6, v7, s11
	v_add_u32_e32 v4, v4, v8
	v_mul_hi_i32 v8, v7, s11
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	v_mul_lo_u32 v9, v6, s25
	v_mul_hi_u32 v10, v6, s12
	v_mul_lo_u32 v6, v6, s12
	v_mul_lo_u32 v8, v8, s12
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v11, s1
	v_add_u32_e32 v9, v10, v9
	v_add_co_u32_e32 v6, vcc, v6, v1
	v_add_u32_e32 v8, v9, v8
	v_addc_co_u32_e32 v1, vcc, v8, v2, vcc
	v_mul_lo_u32 v8, s8, v1
	v_mul_hi_u32 v9, s8, v6
	v_add_co_u32_e32 v1, vcc, s0, v3
	s_mul_hi_i32 s0, s22, s21
	v_addc_co_u32_e32 v2, vcc, v11, v4, vcc
	v_add_u32_e32 v4, v9, v8
	v_mul_lo_u32 v8, s0, v6
	v_mul_lo_u32 v3, s8, v6
	v_mul_hi_i32 v6, v5, s6
	v_mul_lo_u32 v5, v5, s6
	v_add_u32_e32 v4, v4, v8
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	s_mul_i32 s6, s8, s12
	v_mad_i64_i32 v[5:6], s[0:1], v7, s11, v[5:6]
	s_mul_i32 s1, s14, s13
	s_mul_hi_i32 s0, s14, s13
	v_mov_b32_e32 v7, s3
	v_mul_lo_u32 v6, s1, v6
	v_mul_hi_u32 v8, s1, v5
	v_mul_lo_u32 v9, s0, v5
	v_mul_lo_u32 v5, s1, v5
	v_add_co_u32_e32 v3, vcc, s2, v3
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_addc_co_u32_e32 v4, vcc, v7, v4, vcc
	v_mov_b32_e32 v7, s5
	v_add_co_u32_e32 v5, vcc, s4, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[2:3], s11, 0
	v_cmp_gt_i32_e64 s[4:5], s21, 0
	s_mov_b32 s8, 0x7f800000
	s_movk_i32 s10, 0x7fff
	s_branch BB13_3
BB13_2:                                 ; %_Z19__float_to_bfloat16f.exit
                                        ;   in Loop: Header=BB13_3 Depth=1
	s_or_b64 exec, exec, s[30:31]
	v_mul_lo_u32 v7, v7, s9
	v_add_u32_e32 v0, 0x100, v0
	v_lshrrev_b32_e32 v9, 16, v10
	v_add_u32_e32 v7, v7, v8
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e32 v7, vcc, v1, v7
	v_addc_co_u32_e32 v8, vcc, v2, v8, vcc
	v_cmp_le_i32_e32 vcc, s7, v0
	s_or_b64 s[28:29], vcc, s[28:29]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz BB13_30
BB13_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB13_6 Depth 2
                                        ;       Child Loop BB13_10 Depth 3
                                        ;         Child Loop BB13_18 Depth 4
	s_ashr_i32 s0, s9, 31
	s_add_i32 s1, s9, s0
	s_xor_b32 s1, s1, s0
	v_cvt_f32_u32_e32 v7, s1
	v_rcp_iflag_f32_e32 v7, v7
	v_mul_f32_e32 v7, s24, v7
	v_cvt_u32_f32_e32 v7, v7
	v_mul_lo_u32 v8, v7, s1
	v_mul_hi_u32 v9, v7, s1
	v_sub_u32_e32 v10, 0, v8
	v_cmp_eq_u32_e32 vcc, 0, v9
	v_cndmask_b32_e32 v8, v8, v10, vcc
	v_mul_hi_u32 v8, v8, v7
	v_ashrrev_i32_e32 v9, 31, v0
	v_add_u32_e32 v10, v0, v9
	v_xor_b32_e32 v10, v10, v9
	v_add_u32_e32 v11, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_mul_hi_u32 v7, v7, v10
	v_xor_b32_e32 v9, s0, v9
	v_mul_lo_u32 v8, v7, s1
	v_add_u32_e32 v11, 1, v7
	v_add_u32_e32 v12, -1, v7
	v_sub_u32_e32 v13, v10, v8
	v_cmp_ge_u32_e32 vcc, v10, v8
	v_cmp_le_u32_e64 s[0:1], s1, v13
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v11, s[0:1]
	v_cndmask_b32_e32 v7, v12, v7, vcc
	v_xor_b32_e32 v7, v7, v9
	v_sub_u32_e32 v7, v7, v9
	v_mul_lo_u32 v8, v7, s9
	v_mov_b32_e32 v9, 0
	s_andn2_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v10, 0
	s_cbranch_vccnz BB13_26
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB13_3 Depth=1
	v_add_u32_e32 v9, s20, v0
	v_sub_u32_e32 v12, v9, v8
	v_mov_b32_e32 v9, 0
	v_add_u32_e32 v11, s19, v7
	s_mov_b32 s12, 0
	v_mov_b32_e32 v10, 0
	s_mov_b32 s23, 0
	s_branch BB13_6
BB13_5:                                 ; %._crit_edge185
                                        ;   in Loop: Header=BB13_6 Depth=2
	s_add_i32 s23, s23, 1
	s_add_i32 s12, s12, s6
	s_cmp_eq_u32 s23, s11
	s_cbranch_scc1 BB13_26
BB13_6:                                 ; %.preheader
                                        ;   Parent Loop BB13_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB13_10 Depth 3
                                        ;         Child Loop BB13_18 Depth 4
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz BB13_5
; %bb.7:                                ; %.lr.ph184
                                        ;   in Loop: Header=BB13_6 Depth=2
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz BB13_5
; %bb.8:                                ; %.lr.ph184.split.us.preheader
                                        ;   in Loop: Header=BB13_6 Depth=2
	s_mul_i32 s25, s23, s13
	s_mov_b32 s30, 0
	s_mov_b32 s31, s12
	s_branch BB13_10
BB13_9:                                 ; %._crit_edge.us
                                        ;   in Loop: Header=BB13_10 Depth=3
	s_add_i32 s30, s30, 1
	s_add_i32 s31, s31, s22
	s_cmp_eq_u32 s30, s21
	s_cbranch_scc1 BB13_5
BB13_10:                                ; %.lr.ph184.split.us
                                        ;   Parent Loop BB13_3 Depth=1
                                        ;     Parent Loop BB13_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB13_18 Depth 4
	s_mul_i32 s0, s30, s17
	v_subrev_u32_e32 v25, s0, v11
	v_cmp_lt_i32_e32 vcc, -1, v25
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $sgpr33
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $sgpr34
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr21_vgpr22
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr23_vgpr24
	s_and_saveexec_b64 s[36:37], vcc
	s_xor_b64 s[36:37], exec, s[36:37]
	s_cbranch_execz BB13_12
; %bb.11:                               ;   in Loop: Header=BB13_10 Depth=3
	s_ashr_i32 s33, s15, 31
	s_add_i32 s0, s15, s33
	s_xor_b32 s34, s0, s33
	v_cvt_f32_u32_e32 v13, s34
	v_ashrrev_i32_e32 v17, 31, v25
	v_rcp_iflag_f32_e32 v13, v13
	v_mul_f32_e32 v13, s24, v13
	v_cvt_u32_f32_e32 v15, v13
	v_mul_lo_u32 v18, v15, s34
	v_mul_hi_u32 v20, v15, s34
	v_sub_u32_e32 v19, 0, v18
	v_cmp_eq_u32_e32 vcc, 0, v20
	v_cndmask_b32_e32 v13, v18, v19, vcc
	v_mul_hi_u32 v14, v13, v15
	v_add_u32_e32 v13, v25, v17
	v_xor_b32_e32 v13, v13, v17
	v_add_u32_e32 v16, v15, v14
	v_sub_u32_e32 v14, v15, v14
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_mul_hi_u32 v14, v14, v13
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v21, v15
	v_mul_lo_u32 v23, v14, s34
	v_mov_b32_e32 v14, v16
	v_sub_u32_e32 v16, v13, v23
	v_cmp_ge_u32_e64 s[0:1], v13, v23
	v_cmp_le_u32_e32 vcc, s34, v16
	v_subrev_u32_e32 v23, s34, v16
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v24, s34, v16
	v_cndmask_b32_e32 v16, v16, v23, vcc
	v_cndmask_b32_e64 v16, v24, v16, s[0:1]
	v_xor_b32_e32 v16, v16, v17
	v_sub_u32_e32 v16, v16, v17
	v_cmp_ne_u32_e32 vcc, 0, v16
	v_mov_b32_e32 v24, v14
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v23, v13
BB13_12:                                ; %Flow162
                                        ;   in Loop: Header=BB13_10 Depth=3
	s_or_saveexec_b64 s[36:37], s[36:37]
	v_mov_b32_e32 v34, v22
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v16, s33
	v_mov_b32_e32 v28, s34
	v_mov_b32_e32 v14, v17
	v_mov_b32_e32 v29, v13
	v_mov_b32_e32 v31, v18
	v_mov_b32_e32 v32, v20
	v_mov_b32_e32 v35, v19
	v_mov_b32_e32 v26, v15
	v_mov_b32_e32 v33, v21
	v_mov_b32_e32 v36, v23
	s_xor_b64 exec, exec, s[36:37]
	s_cbranch_execz BB13_14
; %bb.13:                               ; %.lr.ph184.split.us._crit_edge
                                        ;   in Loop: Header=BB13_10 Depth=3
	s_ashr_i32 s35, s15, 31
	s_add_i32 s38, s15, s35
	s_xor_b32 s38, s38, s35
	v_cvt_f32_u32_e32 v16, s38
	v_ashrrev_i32_e32 v14, 31, v25
	v_mov_b32_e32 v27, 0
	v_add_u32_e32 v25, v25, v14
	v_rcp_iflag_f32_e32 v16, v16
	v_mov_b32_e32 v30, v27
	v_xor_b32_e32 v29, v25, v14
	v_mov_b32_e32 v37, v30
	v_mul_f32_e32 v16, s24, v16
	v_cvt_u32_f32_e32 v26, v16
	v_mov_b32_e32 v34, v27
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v16, s35
	v_mul_lo_u32 v31, v26, s38
	v_mul_hi_u32 v32, v26, s38
	v_mov_b32_e32 v28, s38
	v_mov_b32_e32 v33, v26
	v_sub_u32_e32 v35, 0, v31
	v_mov_b32_e32 v36, v29
BB13_14:                                ; %Flow163
                                        ;   in Loop: Header=BB13_10 Depth=3
	s_or_b64 exec, exec, s[36:37]
	v_mov_b32_e32 v27, s34
	v_mov_b32_e32 v30, s33
	v_mov_b32_e32 v25, 1
	s_and_saveexec_b64 s[34:35], s[0:1]
	s_cbranch_execz BB13_16
; %bb.15:                               ;   in Loop: Header=BB13_10 Depth=3
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v21, v33
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, v37
	v_mov_b32_e32 v19, v35
	v_mov_b32_e32 v20, v32
	v_mov_b32_e32 v18, v31
	v_mov_b32_e32 v22, v34
	v_mov_b32_e32 v15, v26
	v_mov_b32_e32 v27, v28
	v_mov_b32_e32 v13, v29
	v_mov_b32_e32 v30, v16
	v_mov_b32_e32 v17, v14
BB13_16:                                ; %.lr.ph.us
                                        ;   in Loop: Header=BB13_10 Depth=3
	s_or_b64 exec, exec, s[34:35]
	v_cmp_eq_u32_e32 vcc, 0, v20
	v_cndmask_b32_e32 v14, v18, v19, vcc
	v_mul_lo_u32 v16, v14, v22
	v_mul_hi_u32 v14, v14, v21
	s_mov_b32 s34, s31
	s_mov_b32 s33, s22
	v_add_u32_e32 v14, v14, v16
	v_add_u32_e32 v16, v15, v14
	v_sub_u32_e32 v14, v15, v14
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_mul_lo_u32 v15, v14, v24
	v_mul_hi_u32 v14, v14, v23
	v_xor_b32_e32 v16, v17, v30
	v_add_u32_e32 v14, v14, v15
	v_mul_lo_u32 v15, v14, v27
	v_add_u32_e32 v17, 1, v14
	v_add_u32_e32 v18, -1, v14
	v_sub_u32_e32 v19, v13, v15
	v_cmp_ge_u32_e32 vcc, v13, v15
	v_cmp_ge_u32_e64 s[0:1], v19, v27
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v13, v14, v17, s[0:1]
	v_cndmask_b32_e32 v13, v18, v13, vcc
	v_xor_b32_e32 v13, v13, v16
	v_sub_u32_e32 v14, v13, v16
	v_cmp_gt_i32_e32 vcc, s13, v14
	v_add_u32_e32 v14, s25, v14
	v_mul_lo_u32 v14, v14, s14
	v_cndmask_b32_e32 v13, 0, v25, vcc
	v_mov_b32_e32 v15, v12
	s_branch BB13_18
BB13_17:                                ;   in Loop: Header=BB13_18 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s33, s33, -1
	s_add_i32 s34, s34, 1
	s_cmp_lg_u32 s33, 0
	v_subrev_u32_e32 v15, s18, v15
	s_cbranch_scc0 BB13_9
BB13_18:                                ;   Parent Loop BB13_3 Depth=1
                                        ;     Parent Loop BB13_6 Depth=2
                                        ;       Parent Loop BB13_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v15
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $sgpr35
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $sgpr36
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr24_vgpr25
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr26_vgpr27
	s_and_saveexec_b64 s[38:39], vcc
	s_xor_b64 s[38:39], exec, s[38:39]
	s_cbranch_execz BB13_20
; %bb.19:                               ;   in Loop: Header=BB13_18 Depth=4
	s_ashr_i32 s35, s16, 31
	s_add_i32 s0, s16, s35
	s_xor_b32 s36, s0, s35
	v_cvt_f32_u32_e32 v16, s36
	v_ashrrev_i32_e32 v20, 31, v15
	v_rcp_iflag_f32_e32 v16, v16
	v_mul_f32_e32 v16, s24, v16
	v_cvt_u32_f32_e32 v18, v16
	v_mul_lo_u32 v21, v18, s36
	v_mul_hi_u32 v23, v18, s36
	v_sub_u32_e32 v22, 0, v21
	v_cmp_eq_u32_e32 vcc, 0, v23
	v_cndmask_b32_e32 v16, v21, v22, vcc
	v_mul_hi_u32 v17, v16, v18
	v_add_u32_e32 v16, v15, v20
	v_xor_b32_e32 v16, v16, v20
	v_add_u32_e32 v19, v18, v17
	v_sub_u32_e32 v17, v18, v17
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_mul_hi_u32 v17, v17, v16
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v25, v19
	v_mov_b32_e32 v24, v18
	v_mul_lo_u32 v26, v17, s36
	v_mov_b32_e32 v17, v19
	v_sub_u32_e32 v19, v16, v26
	v_cmp_ge_u32_e64 s[0:1], v16, v26
	v_cmp_le_u32_e32 vcc, s36, v19
	v_subrev_u32_e32 v26, s36, v19
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v27, s36, v19
	v_cndmask_b32_e32 v19, v19, v26, vcc
	v_cndmask_b32_e64 v19, v27, v19, s[0:1]
	v_xor_b32_e32 v19, v19, v20
	v_sub_u32_e32 v19, v19, v20
	v_cmp_ne_u32_e32 vcc, 0, v19
	v_mov_b32_e32 v27, v17
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v26, v16
BB13_20:                                ; %Flow160
                                        ;   in Loop: Header=BB13_18 Depth=4
	s_or_saveexec_b64 s[38:39], s[38:39]
	v_mov_b32_e32 v37, v25
	v_mov_b32_e32 v40, v27
	v_mov_b32_e32 v19, s35
	v_mov_b32_e32 v31, s36
	v_mov_b32_e32 v17, v20
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v34, v21
	v_mov_b32_e32 v35, v23
	v_mov_b32_e32 v38, v22
	v_mov_b32_e32 v29, v18
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v39, v26
	s_xor_b64 exec, exec, s[38:39]
	s_cbranch_execz BB13_22
; %bb.21:                               ; %._crit_edge
                                        ;   in Loop: Header=BB13_18 Depth=4
	s_ashr_i32 s37, s16, 31
	s_add_i32 s40, s16, s37
	s_xor_b32 s40, s40, s37
	v_cvt_f32_u32_e32 v19, s40
	v_ashrrev_i32_e32 v17, 31, v15
	v_mov_b32_e32 v30, 0
	v_add_u32_e32 v28, v15, v17
	v_rcp_iflag_f32_e32 v19, v19
	v_mov_b32_e32 v33, v30
	v_xor_b32_e32 v32, v28, v17
	v_mov_b32_e32 v40, v33
	v_mul_f32_e32 v19, s24, v19
	v_cvt_u32_f32_e32 v29, v19
	v_mov_b32_e32 v37, v30
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v19, s37
	v_mul_lo_u32 v34, v29, s40
	v_mul_hi_u32 v35, v29, s40
	v_mov_b32_e32 v31, s40
	v_mov_b32_e32 v36, v29
	v_sub_u32_e32 v38, 0, v34
	v_mov_b32_e32 v39, v32
BB13_22:                                ; %Flow161
                                        ;   in Loop: Header=BB13_18 Depth=4
	s_or_b64 exec, exec, s[38:39]
	v_mov_b32_e32 v30, s36
	v_mov_b32_e32 v33, s35
	v_mov_b32_e32 v28, 1
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz BB13_24
; %bb.23:                               ;   in Loop: Header=BB13_18 Depth=4
	v_mov_b32_e32 v26, v39
	v_mov_b32_e32 v24, v36
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, v40
	v_mov_b32_e32 v22, v38
	v_mov_b32_e32 v23, v35
	v_mov_b32_e32 v21, v34
	v_mov_b32_e32 v25, v37
	v_mov_b32_e32 v18, v29
	v_mov_b32_e32 v30, v31
	v_mov_b32_e32 v16, v32
	v_mov_b32_e32 v33, v19
	v_mov_b32_e32 v20, v17
BB13_24:                                ;   in Loop: Header=BB13_18 Depth=4
	s_or_b64 exec, exec, s[36:37]
	v_cmp_eq_u32_e32 vcc, 0, v23
	v_cndmask_b32_e32 v17, v21, v22, vcc
	v_mul_lo_u32 v19, v17, v25
	v_mul_hi_u32 v17, v17, v24
	v_add_u32_e32 v17, v17, v19
	v_add_u32_e32 v19, v18, v17
	v_sub_u32_e32 v17, v18, v17
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_mul_lo_u32 v18, v17, v27
	v_mul_hi_u32 v17, v17, v26
	v_xor_b32_e32 v19, v20, v33
	v_add_u32_e32 v17, v17, v18
	v_mul_lo_u32 v18, v17, v30
	v_add_u32_e32 v20, 1, v17
	v_add_u32_e32 v21, -1, v17
	v_sub_u32_e32 v22, v16, v18
	v_cmp_ge_u32_e32 vcc, v16, v18
	v_cmp_ge_u32_e64 s[0:1], v22, v30
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v16, v17, v20, s[0:1]
	v_cndmask_b32_e32 v16, v21, v16, vcc
	v_xor_b32_e32 v16, v16, v19
	v_sub_u32_e32 v16, v16, v19
	v_cmp_gt_i32_e32 vcc, s14, v16
	v_cndmask_b32_e32 v17, 0, v28, vcc
	v_and_b32_e32 v17, v17, v13
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB13_17
; %bb.25:                               ;   in Loop: Header=BB13_18 Depth=4
	v_add_u32_e32 v16, v16, v14
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshlrev_b64 v[16:17], 1, v[16:17]
	s_ashr_i32 s35, s34, 31
	v_add_co_u32_e32 v16, vcc, v5, v16
	s_lshl_b64 s[36:37], s[34:35], 1
	v_addc_co_u32_e32 v17, vcc, v6, v17, vcc
	v_mov_b32_e32 v19, s37
	v_add_co_u32_e32 v18, vcc, s36, v3
	v_addc_co_u32_e32 v19, vcc, v4, v19, vcc
	global_load_ushort v18, v[18:19], off
	global_load_ushort v16, v[16:17], off
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v18, 16, v18
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	v_cvt_f64_f32_e32 v[16:17], v16
	v_cvt_f64_f32_e32 v[18:19], v18
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[9:10], v[16:17], v[18:19], v[9:10]
	s_branch BB13_17
BB13_26:                                ; %Flow167
                                        ;   in Loop: Header=BB13_3 Depth=1
	v_cvt_f32_f64_e32 v9, v[9:10]
	v_sub_u32_e32 v8, v0, v8
	v_and_b32_e32 v10, s8, v9
	v_cmp_ne_u32_e32 vcc, s8, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.27:                               ;   in Loop: Header=BB13_3 Depth=1
	v_bfe_u32 v10, v9, 16, 1
	v_add3_u32 v10, v9, v10, s10
; %bb.28:                               ; %Flow
                                        ;   in Loop: Header=BB13_3 Depth=1
	s_or_saveexec_b64 s[30:31], s[0:1]
	s_xor_b64 exec, exec, s[30:31]
	s_cbranch_execz BB13_2
; %bb.29:                               ;   in Loop: Header=BB13_3 Depth=1
	v_mov_b32_e32 v10, 0
	v_or_b32_e32 v11, 0x10000, v9
	v_cmp_eq_u32_sdwa s[0:1], v9, v10 src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v10, v11, v9, s[0:1]
	s_branch BB13_2
BB13_30:                                ; %Flow169
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_bwd_nchw_ushort
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 41
		.amdhsa_next_free_sgpr 41
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end13:
	.size	naive_conv_bwd_nchw_ushort, .Lfunc_end13-naive_conv_bwd_nchw_ushort
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2660
; NumSgprs: 43
; NumVgprs: 41
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 43
; NumVGPRsForWavesPerEU: 41
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_wrw_nchw_ushort ; -- Begin function naive_conv_wrw_nchw_ushort
	.globl	naive_conv_wrw_nchw_ushort
	.p2align	8
	.type	naive_conv_wrw_nchw_ushort,@function
naive_conv_wrw_nchw_ushort:               ; @naive_conv_wrw_nchw_ushort
naive_conv_wrw_nchw_ushort$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s22, s21
	s_mul_i32 s24, s7, s12
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB14_19
; %bb.1:                                ; %.lr.ph184
	s_ashr_i32 s0, s11, 31
	s_add_i32 s1, s11, s0
	s_xor_b32 s35, s1, s0
	v_cvt_f32_u32_e32 v1, s35
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s1, s6, 31
	s_add_i32 s30, s6, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s30, s30, s1
	s_xor_b32 s33, s1, s0
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	s_load_dwordx2 s[26:27], s[4:5], 0x8
	s_load_dwordx2 s[28:29], s[4:5], 0x10
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mul_i32 s31, s9, s8
	s_ashr_i32 s4, s12, 31
	s_mul_i32 s4, s31, s4
	v_mul_lo_u32 v2, v1, s35
	v_mul_hi_u32 v3, v1, s35
	s_mul_hi_u32 s34, s31, s12
	s_mul_i32 s31, s31, s12
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_mul_hi_i32 s5, s9, s8
	s_mul_i32 s5, s5, s12
	s_sub_i32 s20, 0, s20
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_mul_hi_u32 v1, v1, s30
	v_mul_lo_u32 v2, v1, s35
	v_add_u32_e32 v3, 1, v1
	v_add_u32_e32 v4, -1, v1
	v_sub_u32_e32 v5, s30, v2
	v_cmp_ge_u32_e32 vcc, s30, v2
	v_cmp_le_u32_e64 s[0:1], s35, v5
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	v_xor_b32_e32 v1, s33, v1
	v_subrev_u32_e32 v5, s33, v1
	v_mul_lo_u32 v2, v5, s11
	v_ashrrev_i32_e32 v1, 31, v5
	v_mul_lo_u32 v4, s31, v1
	v_mul_hi_u32 v6, s31, v5
	s_add_i32 s0, s34, s4
	v_sub_u32_e32 v3, s6, v2
	s_add_i32 s0, s0, s5
	v_add_u32_e32 v2, v6, v4
	v_ashrrev_i32_e32 v4, 31, v3
	v_mul_lo_u32 v7, s0, v5
	v_mad_i64_i32 v[3:4], s[0:1], v5, s11, v[3:4]
	v_mul_lo_u32 v1, s31, v5
	s_mul_i32 s1, s21, s12
	s_ashr_i32 s4, s22, 31
	s_mul_hi_i32 s0, s21, s12
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s3
	s_mul_i32 s3, s1, s4
	s_mul_hi_u32 s5, s1, s22
	s_mul_i32 s1, s1, s22
	s_add_i32 s3, s5, s3
	s_mul_i32 s0, s0, s22
	v_add_u32_e32 v2, v2, v7
	s_add_i32 s3, s3, s0
	v_mul_lo_u32 v7, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	v_mul_lo_u32 v9, s3, v3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_mul_lo_u32 v5, s1, v3
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v8, v7
	s_mul_i32 s1, s14, s13
	v_add_u32_e32 v6, v6, v9
	s_mul_hi_i32 s0, s14, s13
	v_mul_lo_u32 v9, s1, v4
	v_mul_hi_u32 v10, s1, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s1, v3
	v_add_co_u32_e32 v3, vcc, s26, v5
	v_mov_b32_e32 v8, s27
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	s_mul_i32 s0, s23, s14
	s_mul_i32 s5, s0, s13
	s_mul_i32 s0, s23, s12
	s_mul_i32 s5, s5, s11
	s_mul_i32 s11, s0, s9
	v_mov_b32_e32 v7, s29
	v_add_co_u32_e32 v5, vcc, s28, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s10, 0
	v_cmp_gt_i32_e64 s[28:29], s13, 0
	v_cmp_gt_i32_e64 s[30:31], s14, 0
	s_sub_i32 s6, 0, s19
	s_mul_i32 s11, s11, s8
	s_mul_i32 s12, s15, s9
	s_mov_b64 s[34:35], 0
	s_mov_b32 s23, 0x7f800000
	s_movk_i32 s33, 0x7fff
	s_branch BB14_3
BB14_2:                                 ; %_Z19__float_to_bfloat16f.exit
                                        ;   in Loop: Header=BB14_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_mul_lo_u32 v9, v9, s21
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	s_or_b64 s[34:35], vcc, s[34:35]
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s22
	v_lshrrev_b32_e32 v9, 16, v10
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v3, v7
	v_addc_co_u32_e64 v8, s[0:1], v4, v8, s[0:1]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz BB14_19
BB14_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB14_6 Depth 2
                                        ;       Child Loop BB14_10 Depth 3
                                        ;         Child Loop BB14_13 Depth 4
	s_add_i32 s0, s22, s4
	s_xor_b32 s0, s0, s4
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s21, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s21, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s36, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s36
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s37, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s4, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s36
	v_mul_hi_u32 v12, v9, s36
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s37
	s_xor_b32 s38, s0, s37
	v_cvt_f32_u32_e32 v11, s38
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s38
	v_mul_hi_u32 v13, v10, s38
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s36
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s22
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s36, v14
	v_subrev_u32_e32 v10, s36, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s38
	v_add_u32_e32 v13, s36, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s38, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s37, v9
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v7, v0, v7
	v_subrev_u32_e32 v9, s37, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB14_15
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB14_3 Depth=1
	v_mul_lo_u32 v12, v8, s17
	v_mul_lo_u32 v10, s8, v9
	v_mul_lo_u32 v11, s18, v7
	s_mov_b32 s2, 0
	s_mov_b32 s3, 0
	v_add3_u32 v10, s6, v12, v10
	v_mul_lo_u32 v13, s9, v10
	v_add_u32_e32 v14, s20, v11
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_branch BB14_6
BB14_5:                                 ; %._crit_edge175
                                        ;   in Loop: Header=BB14_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s5
	s_cmp_eq_u32 s3, s10
	v_add_u32_e32 v13, s11, v13
	s_cbranch_scc1 BB14_15
BB14_6:                                 ; %.preheader
                                        ;   Parent Loop BB14_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB14_10 Depth 3
                                        ;         Child Loop BB14_13 Depth 4
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz BB14_5
; %bb.7:                                ; %.lr.ph174
                                        ;   in Loop: Header=BB14_6 Depth=2
	s_andn2_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz BB14_5
; %bb.8:                                ; %.lr.ph174.split.us.preheader
                                        ;   in Loop: Header=BB14_6 Depth=2
	s_mov_b32 s36, 0
	v_mov_b32_e32 v15, v13
	s_mov_b32 s37, s2
	s_branch BB14_10
BB14_9:                                 ; %Flow60
                                        ;   in Loop: Header=BB14_10 Depth=3
	s_or_b64 exec, exec, s[38:39]
	s_add_i32 s36, s36, 1
	s_add_i32 s37, s37, s14
	s_cmp_eq_u32 s36, s13
	v_add_u32_e32 v15, s12, v15
	s_cbranch_scc1 BB14_5
BB14_10:                                ; %.lr.ph174.split.us
                                        ;   Parent Loop BB14_3 Depth=1
                                        ;     Parent Loop BB14_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB14_13 Depth 4
	s_mul_i32 s0, s36, s15
	s_sub_i32 s0, s0, s19
	v_add_u32_e32 v16, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v16
	v_cmp_gt_i32_e64 s[0:1], s8, v16
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[38:39], s[0:1]
	s_cbranch_execz BB14_9
; %bb.11:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB14_10 Depth=3
	v_mov_b32_e32 v16, v14
	s_mov_b32 s40, s37
	s_mov_b32 s42, s14
	s_branch BB14_13
BB14_12:                                ;   in Loop: Header=BB14_13 Depth=4
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s42, s42, -1
	s_add_i32 s40, s40, 1
	s_cmp_lg_u32 s42, 0
	v_add_u32_e32 v16, s16, v16
	s_cbranch_scc0 BB14_9
BB14_13:                                ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB14_3 Depth=1
                                        ;     Parent Loop BB14_6 Depth=2
                                        ;       Parent Loop BB14_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	v_cmp_lt_i32_e32 vcc, -1, v16
	v_cmp_gt_i32_e64 s[0:1], s9, v16
	s_and_b64 s[44:45], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[44:45]
	s_cbranch_execz BB14_12
; %bb.14:                               ;   in Loop: Header=BB14_13 Depth=4
	v_add_u32_e32 v17, v15, v16
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_ashr_i32 s41, s40, 31
	v_add_co_u32_e32 v17, vcc, v1, v17
	s_lshl_b64 s[44:45], s[40:41], 1
	v_addc_co_u32_e32 v18, vcc, v2, v18, vcc
	v_mov_b32_e32 v20, s45
	v_add_co_u32_e32 v19, vcc, s44, v5
	v_addc_co_u32_e32 v20, vcc, v6, v20, vcc
	global_load_ushort v19, v[19:20], off
	global_load_ushort v17, v[17:18], off
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v19, 16, v19
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v17, 16, v17
	v_cvt_f64_f32_e32 v[17:18], v17
	v_cvt_f64_f32_e32 v[19:20], v19
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[17:18], v[19:20], v[10:11]
	s_branch BB14_12
BB14_15:                                ; %._crit_edge180
                                        ;   in Loop: Header=BB14_3 Depth=1
	v_cvt_f32_f64_e32 v11, v[10:11]
	v_and_b32_e32 v10, s23, v11
	v_cmp_ne_u32_e32 vcc, s23, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.16:                               ;   in Loop: Header=BB14_3 Depth=1
	v_bfe_u32 v10, v11, 16, 1
	v_add3_u32 v10, v11, v10, s33
; %bb.17:                               ; %Flow
                                        ;   in Loop: Header=BB14_3 Depth=1
	s_or_saveexec_b64 s[2:3], s[0:1]
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execz BB14_2
; %bb.18:                               ;   in Loop: Header=BB14_3 Depth=1
	v_mov_b32_e32 v10, 0
	v_or_b32_e32 v12, 0x10000, v11
	v_cmp_eq_u32_sdwa s[0:1], v11, v10 src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v10, v12, v11, s[0:1]
	s_branch BB14_2
BB14_19:                                ; %Flow66
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_wrw_nchw_ushort
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 21
		.amdhsa_next_free_sgpr 46
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end14:
	.size	naive_conv_wrw_nchw_ushort, .Lfunc_end14-naive_conv_wrw_nchw_ushort
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 1588
; NumSgprs: 48
; NumVgprs: 21
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 48
; NumVGPRsForWavesPerEU: 21
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_fwd_ncdhw_ushort ; -- Begin function naive_conv_fwd_ncdhw_ushort
	.globl	naive_conv_fwd_ncdhw_ushort
	.p2align	8
	.type	naive_conv_fwd_ncdhw_ushort,@function
naive_conv_fwd_ncdhw_ushort:              ; @naive_conv_fwd_ncdhw_ushort
naive_conv_fwd_ncdhw_ushort$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s16, s15
	s_mul_i32 s24, s7, s14
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB15_22
; %bb.1:                                ; %.lr.ph238
	s_ashr_i32 s0, s12, 31
	s_add_i32 s1, s12, s0
	s_xor_b32 s39, s1, s0
	v_cvt_f32_u32_e32 v1, s39
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s1, s11, 31
	s_ashr_i32 s33, s6, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s3, s11, s1
	s_add_i32 s2, s6, s33
	s_xor_b32 s40, s3, s1
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s38, s2, s33
	s_xor_b32 s2, s33, s0
	s_load_dwordx2 s[36:37], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[26:27], s[4:5], 0x10
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_load_dwordx2 s[4:5], s[4:5], 0x68
	v_mul_lo_u32 v2, v1, s39
	v_mul_hi_u32 v3, v1, s39
	s_mov_b64 s[42:43], 0
	s_mov_b32 s44, 0x7f800000
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s41, s5, s12
	s_mul_i32 s5, s5, s13
	s_movk_i32 s45, 0x7fff
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_cvt_f32_u32_e32 v2, s40
	v_mul_hi_u32 v1, v1, s38
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s39
	v_add_u32_e32 v4, -1, v1
	v_mul_f32_e32 v2, s25, v2
	v_sub_u32_e32 v5, s38, v3
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_ge_u32_e32 vcc, s38, v3
	v_cmp_le_u32_e64 s[0:1], s39, v5
	v_add_u32_e32 v3, 1, v1
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	s_mul_i32 s0, s12, s11
	v_mul_hi_u32 v4, v2, s40
	s_ashr_i32 s11, s0, 31
	v_mul_lo_u32 v3, v2, s40
	s_add_i32 s0, s0, s11
	s_xor_b32 s39, s0, s11
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s39
	v_sub_u32_e32 v6, 0, v3
	v_cndmask_b32_e32 v3, v3, v6, vcc
	v_mul_hi_u32 v3, v3, v2
	v_rcp_iflag_f32_e32 v4, v4
	v_xor_b32_e32 v1, s2, v1
	v_subrev_u32_e32 v1, s2, v1
	v_add_u32_e32 v7, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_mul_f32_e32 v3, s25, v4
	v_cvt_u32_f32_e32 v3, v3
	v_ashrrev_i32_e32 v5, 31, v1
	v_add_u32_e32 v6, v1, v5
	v_cndmask_b32_e32 v2, v2, v7, vcc
	v_mul_lo_u32 v4, v1, s12
	v_mul_lo_u32 v1, v3, s39
	v_mul_hi_u32 v7, v3, s39
	v_xor_b32_e32 v6, v6, v5
	v_mul_hi_u32 v2, v2, v6
	v_sub_u32_e32 v9, 0, v1
	v_cmp_eq_u32_e64 s[0:1], 0, v7
	v_cndmask_b32_e64 v1, v1, v9, s[0:1]
	v_mul_lo_u32 v2, v2, s40
	v_mul_hi_u32 v1, v1, v3
	v_sub_u32_e32 v8, v6, v2
	v_cmp_ge_u32_e64 s[2:3], v6, v2
	v_add_u32_e32 v6, v3, v1
	v_sub_u32_e32 v1, v3, v1
	v_cndmask_b32_e64 v1, v1, v6, s[0:1]
	v_mul_hi_u32 v1, v1, s38
	v_cmp_le_u32_e32 vcc, s40, v8
	v_subrev_u32_e32 v2, s40, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_mul_lo_u32 v3, v1, s39
	v_add_u32_e32 v7, s40, v8
	v_cndmask_b32_e32 v2, v8, v2, vcc
	v_cndmask_b32_e64 v2, v7, v2, s[2:3]
	v_xor_b32_e32 v2, v2, v5
	v_sub_u32_e32 v7, v2, v5
	v_sub_u32_e32 v2, s38, v3
	v_cmp_le_u32_e32 vcc, s39, v2
	v_cmp_ge_u32_e64 s[0:1], s38, v3
	v_add_u32_e32 v3, 1, v1
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v2, -1, v1
	v_cndmask_b32_e32 v1, v1, v3, vcc
	s_xor_b32 s2, s33, s11
	v_cndmask_b32_e64 v1, v2, v1, s[0:1]
	v_xor_b32_e32 v3, s2, v1
	v_mul_hi_i32 v2, v7, s5
	v_mul_lo_u32 v1, v7, s5
	v_subrev_u32_e32 v5, s2, v3
	s_ashr_i32 s2, s10, 31
	v_sub_u32_e32 v3, s6, v4
	v_mad_i64_i32 v[1:2], s[0:1], v5, s13, v[1:2]
	s_mul_i32 s1, s9, s8
	s_mul_hi_i32 s0, s9, s8
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s3, s1, s10
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s10
	s_mul_i32 s1, s1, s10
	s_add_i32 s2, s2, s0
	v_ashrrev_i32_e32 v4, 31, v3
	v_mul_lo_u32 v8, s2, v1
	v_mul_hi_u32 v6, s1, v1
	v_mul_lo_u32 v2, s1, v2
	v_mul_lo_u32 v1, s1, v1
	v_mad_i64_i32 v[3:4], s[0:1], v5, s12, v[3:4]
	s_mul_i32 s1, s30, s13
	s_ashr_i32 s2, s31, 31
	s_mul_hi_i32 s0, s30, s13
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s5, s1, s31
	s_add_i32 s2, s5, s2
	s_mul_i32 s0, s0, s31
	s_ashr_i32 s3, s4, 31
	s_mul_i32 s1, s1, s31
	s_add_i32 s0, s2, s0
	s_mul_i32 s2, s1, s3
	s_mul_hi_u32 s3, s1, s4
	v_add_u32_e32 v2, v6, v2
	s_mul_i32 s1, s1, s4
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s4
	v_add_u32_e32 v2, v2, v8
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v8, s1, v4
	v_mul_hi_u32 v9, s1, v3
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	v_mul_lo_u32 v10, s2, v3
	v_mul_lo_u32 v5, s1, v3
	v_mad_i64_i32 v[3:4], s[0:1], v7, s41, v[3:4]
	s_mul_i32 s1, s15, s14
	s_ashr_i32 s5, s16, 31
	s_mul_hi_i32 s0, s15, s14
	v_mov_b32_e32 v6, s37
	v_add_co_u32_e32 v1, vcc, s36, v1
	s_mul_i32 s2, s1, s5
	s_mul_hi_u32 s3, s1, s16
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v9, v8
	s_mul_i32 s1, s1, s16
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s16
	v_add_u32_e32 v6, v6, v10
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v9, s1, v4
	v_mul_hi_u32 v10, s1, v3
	v_mul_lo_u32 v11, s2, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s1, v3
	v_add_co_u32_e32 v3, vcc, s34, v5
	v_mov_b32_e32 v8, s35
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	s_mul_i32 s33, s10, s9
	s_mul_i32 s40, s20, s10
	s_mul_i32 s6, s4, s31
	v_mov_b32_e32 v7, s27
	v_add_co_u32_e32 v5, vcc, s26, v5
	s_sub_i32 s12, 0, s29
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s13, 0
	v_cmp_gt_i32_e64 s[34:35], s30, 0
	v_cmp_gt_i32_e64 s[36:37], s31, 0
	v_cmp_gt_i32_e64 s[38:39], s4, 0
	s_mul_i32 s11, s6, s30
	s_sub_i32 s14, 0, s28
	s_sub_i32 s29, 0, s23
	s_mul_i32 s33, s33, s8
	s_mul_i32 s40, s40, s9
	s_mul_i32 s41, s21, s10
	s_branch BB15_3
BB15_2:                                 ; %_Z19__float_to_bfloat16f.exit
                                        ;   in Loop: Header=BB15_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_mul_lo_u32 v9, v9, s15
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	s_or_b64 s[42:43], vcc, s[42:43]
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s16
	v_lshrrev_b32_e32 v9, 16, v10
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v5, v7
	v_addc_co_u32_e64 v8, s[0:1], v6, v8, s[0:1]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz BB15_22
BB15_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB15_6 Depth 2
                                        ;       Child Loop BB15_9 Depth 3
                                        ;         Child Loop BB15_13 Depth 4
                                        ;           Child Loop BB15_16 Depth 5
	s_add_i32 s0, s16, s5
	s_xor_b32 s0, s0, s5
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s15, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s15, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s46, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s46
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s47, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s5, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s46
	v_mul_hi_u32 v12, v9, s46
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s47
	s_xor_b32 s48, s0, s47
	v_cvt_f32_u32_e32 v11, s48
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v11, v10, s48
	v_mul_hi_u32 v13, v10, s48
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v11
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_mul_lo_u32 v9, v9, s46
	v_mul_hi_u32 v11, v11, v10
	v_mul_lo_u32 v7, v7, s16
	v_sub_u32_e32 v14, v12, v9
	v_cmp_ge_u32_e64 s[2:3], v12, v9
	v_add_u32_e32 v9, v10, v11
	v_sub_u32_e32 v10, v10, v11
	v_cndmask_b32_e32 v9, v10, v9, vcc
	v_mul_hi_u32 v9, v9, v0
	v_cmp_le_u32_e64 s[0:1], s46, v14
	v_subrev_u32_e32 v10, s46, v14
	s_and_b64 vcc, s[0:1], s[2:3]
	v_mul_lo_u32 v11, v9, s48
	v_add_u32_e32 v13, s46, v14
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cndmask_b32_e64 v10, v13, v10, s[2:3]
	v_xor_b32_e32 v10, v10, v8
	v_sub_u32_e32 v8, v10, v8
	v_sub_u32_e32 v10, v0, v11
	v_cmp_le_u32_e32 vcc, s48, v10
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v9
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v10, -1, v9
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e64 v9, v10, v9, s[0:1]
	v_xor_b32_e32 v9, s47, v9
	v_mov_b32_e32 v10, 0
	v_sub_u32_e32 v7, v0, v7
	v_subrev_u32_e32 v9, s47, v9
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB15_18
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB15_3 Depth=1
	v_mul_lo_u32 v12, v9, s17
	v_mul_lo_u32 v13, v8, s18
	v_mul_lo_u32 v15, s19, v7
	s_mov_b32 s2, 0
	v_add_u32_e32 v10, s29, v12
	v_mul_lo_u32 v14, s9, v10
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_subrev_u32_e32 v12, s23, v12
	v_add3_u32 v14, s14, v13, v14
	v_mul_lo_u32 v16, s10, v14
	v_add_u32_e32 v14, s12, v15
	v_subrev_u32_e32 v13, s28, v13
	s_mov_b32 s3, 0
	v_add3_u32 v15, s12, v16, v15
	s_branch BB15_6
BB15_5:                                 ; %Flow86
                                        ;   in Loop: Header=BB15_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s11
	s_cmp_eq_u32 s3, s13
	v_add_u32_e32 v15, s33, v15
	s_cbranch_scc1 BB15_18
BB15_6:                                 ; %.preheader
                                        ;   Parent Loop BB15_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB15_9 Depth 3
                                        ;         Child Loop BB15_13 Depth 4
                                        ;           Child Loop BB15_16 Depth 5
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz BB15_5
; %bb.7:                                ; %.lr.ph228
                                        ;   in Loop: Header=BB15_6 Depth=2
	s_mov_b32 s46, 0
	v_mov_b32_e32 v16, v15
	s_mov_b32 s47, s2
	s_branch BB15_9
BB15_8:                                 ; %._crit_edge223
                                        ;   in Loop: Header=BB15_9 Depth=3
	s_add_i32 s46, s46, 1
	s_add_i32 s47, s47, s6
	s_cmp_eq_u32 s46, s30
	v_add_u32_e32 v16, s40, v16
	s_cbranch_scc1 BB15_5
BB15_9:                                 ;   Parent Loop BB15_3 Depth=1
                                        ;     Parent Loop BB15_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB15_13 Depth 4
                                        ;           Child Loop BB15_16 Depth 5
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB15_8
; %bb.10:                               ; %.lr.ph222
                                        ;   in Loop: Header=BB15_9 Depth=3
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB15_8
; %bb.11:                               ; %.lr.ph222.split.us.preheader
                                        ;   in Loop: Header=BB15_9 Depth=3
	s_mul_i32 s0, s46, s20
	v_add_u32_e32 v17, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v17
	s_and_b64 s[48:49], vcc, s[0:1]
	s_mov_b32 s50, 0
	v_mov_b32_e32 v17, v16
	s_mov_b32 s51, s47
	s_branch BB15_13
BB15_12:                                ; %Flow82
                                        ;   in Loop: Header=BB15_13 Depth=4
	s_or_b64 exec, exec, s[52:53]
	s_add_i32 s50, s50, 1
	s_add_i32 s51, s51, s4
	s_cmp_lg_u32 s50, s31
	v_add_u32_e32 v17, s41, v17
	s_cbranch_scc0 BB15_8
BB15_13:                                ; %.lr.ph222.split.us
                                        ;   Parent Loop BB15_3 Depth=1
                                        ;     Parent Loop BB15_6 Depth=2
                                        ;       Parent Loop BB15_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB15_16 Depth 5
	s_mul_i32 s0, s50, s21
	v_add_u32_e32 v18, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s9, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[52:53], s[0:1]
	s_cbranch_execz BB15_12
; %bb.14:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB15_13 Depth=4
	s_mov_b32 s54, 0
	s_mov_b32 s56, s51
	s_mov_b32 s55, s4
	s_branch BB15_16
BB15_15:                                ;   in Loop: Header=BB15_16 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s55, s55, -1
	s_add_i32 s56, s56, 1
	s_add_i32 s54, s54, s22
	s_cmp_lg_u32 s55, 0
	s_cbranch_scc0 BB15_12
BB15_16:                                ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB15_3 Depth=1
                                        ;     Parent Loop BB15_6 Depth=2
                                        ;       Parent Loop BB15_9 Depth=3
                                        ;         Parent Loop BB15_13 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_add_u32_e32 v18, s54, v14
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s10, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 s[58:59], s[48:49], s[0:1]
	s_and_saveexec_b64 s[0:1], s[58:59]
	s_cbranch_execz BB15_15
; %bb.17:                               ;   in Loop: Header=BB15_16 Depth=5
	v_add_u32_e32 v18, s54, v17
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshlrev_b64 v[18:19], 1, v[18:19]
	s_ashr_i32 s57, s56, 31
	v_add_co_u32_e32 v18, vcc, v1, v18
	s_lshl_b64 s[58:59], s[56:57], 1
	v_addc_co_u32_e32 v19, vcc, v2, v19, vcc
	v_mov_b32_e32 v21, s59
	v_add_co_u32_e32 v20, vcc, s58, v3
	v_addc_co_u32_e32 v21, vcc, v4, v21, vcc
	global_load_ushort v20, v[20:21], off
	global_load_ushort v18, v[18:19], off
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v20, 16, v20
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v18, 16, v18
	v_cvt_f64_f32_e32 v[18:19], v18
	v_cvt_f64_f32_e32 v[20:21], v20
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[18:19], v[20:21], v[10:11]
	s_branch BB15_15
BB15_18:                                ; %._crit_edge234
                                        ;   in Loop: Header=BB15_3 Depth=1
	v_cvt_f32_f64_e32 v11, v[10:11]
	v_and_b32_e32 v10, s44, v11
	v_cmp_ne_u32_e32 vcc, s44, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.19:                               ;   in Loop: Header=BB15_3 Depth=1
	v_bfe_u32 v10, v11, 16, 1
	v_add3_u32 v10, v11, v10, s45
; %bb.20:                               ; %Flow
                                        ;   in Loop: Header=BB15_3 Depth=1
	s_or_saveexec_b64 s[2:3], s[0:1]
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execz BB15_2
; %bb.21:                               ;   in Loop: Header=BB15_3 Depth=1
	v_mov_b32_e32 v10, 0
	v_or_b32_e32 v12, 0x10000, v11
	v_cmp_eq_u32_sdwa s[0:1], v11, v10 src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v10, v12, v11, s[0:1]
	s_branch BB15_2
BB15_22:                                ; %Flow90
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_fwd_ncdhw_ushort
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 60
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end15:
	.size	naive_conv_fwd_ncdhw_ushort, .Lfunc_end15-naive_conv_fwd_ncdhw_ushort
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2120
; NumSgprs: 62
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 22
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_bwd_ncdhw_ushort ; -- Begin function naive_conv_bwd_ncdhw_ushort
	.globl	naive_conv_bwd_ncdhw_ushort
	.p2align	8
	.type	naive_conv_bwd_ncdhw_ushort,@function
naive_conv_bwd_ncdhw_ushort:              ; @naive_conv_bwd_ncdhw_ushort
naive_conv_bwd_ncdhw_ushort$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s10, s9
	s_mul_i32 s24, s7, s8
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB16_39
; %bb.1:                                ; %.lr.ph255
	s_ashr_i32 s33, s13, 31
	s_add_i32 s0, s13, s33
	s_xor_b32 s3, s0, s33
	v_cvt_f32_u32_e32 v1, s3
	s_mov_b32 s25, 0x4f800000
	s_ashr_i32 s38, s6, 31
	s_ashr_i32 s0, s11, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s1, s6, s38
	s_add_i32 s2, s11, s0
	s_xor_b32 s40, s2, s0
	v_mul_f32_e32 v1, s25, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s39, s1, s38
	s_xor_b32 s2, s38, s33
	s_load_dwordx2 s[36:37], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[26:27], s[4:5], 0x10
	v_mul_lo_u32 v2, v1, s3
	v_mul_hi_u32 v3, v1, s3
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_load_dwordx2 s[4:5], s[4:5], 0x68
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v11, s37
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	v_cvt_f32_u32_e32 v3, s40
	s_mul_i32 s42, s5, s13
	v_add_u32_e32 v4, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_mul_hi_u32 v1, v1, s39
	v_rcp_iflag_f32_e32 v3, v3
	v_mul_lo_u32 v2, v1, s3
	v_add_u32_e32 v4, 1, v1
	v_add_u32_e32 v5, -1, v1
	v_sub_u32_e32 v6, s39, v2
	v_cmp_ge_u32_e32 vcc, s39, v2
	v_mul_f32_e32 v2, s25, v3
	v_cvt_u32_f32_e32 v2, v2
	v_cmp_le_u32_e64 s[0:1], s3, v6
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	s_mul_i32 s0, s13, s11
	v_mul_hi_u32 v4, v2, s40
	s_ashr_i32 s11, s0, 31
	v_mul_lo_u32 v3, v2, s40
	s_add_i32 s0, s0, s11
	s_xor_b32 s41, s0, s11
	v_cndmask_b32_e32 v1, v5, v1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cvt_f32_u32_e32 v4, s41
	v_sub_u32_e32 v7, 0, v3
	v_cndmask_b32_e32 v3, v3, v7, vcc
	v_xor_b32_e32 v1, s2, v1
	v_mul_hi_u32 v3, v3, v2
	v_subrev_u32_e32 v1, s2, v1
	v_rcp_iflag_f32_e32 v4, v4
	v_ashrrev_i32_e32 v6, 31, v1
	v_mul_lo_u32 v5, v1, s13
	v_add_u32_e32 v1, v1, v6
	v_xor_b32_e32 v7, v1, v6
	v_add_u32_e32 v1, v2, v3
	v_sub_u32_e32 v2, v2, v3
	v_cndmask_b32_e32 v1, v2, v1, vcc
	v_mul_f32_e32 v2, s25, v4
	v_mul_hi_u32 v1, v1, v7
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, v1, s40
	v_sub_u32_e32 v1, s6, v5
	v_mul_lo_u32 v4, v2, s41
	v_mul_hi_u32 v5, v2, s41
	v_sub_u32_e32 v8, v7, v3
	v_cmp_ge_u32_e64 s[2:3], v7, v3
	v_sub_u32_e32 v9, 0, v4
	v_cmp_eq_u32_e64 s[0:1], 0, v5
	v_cndmask_b32_e64 v4, v4, v9, s[0:1]
	v_mul_hi_u32 v4, v4, v2
	v_cmp_le_u32_e32 vcc, s40, v8
	v_subrev_u32_e32 v3, s40, v8
	s_and_b64 vcc, vcc, s[2:3]
	v_add_u32_e32 v7, v2, v4
	v_sub_u32_e32 v2, v2, v4
	v_cndmask_b32_e64 v2, v2, v7, s[0:1]
	v_mul_hi_u32 v2, v2, s39
	v_add_u32_e32 v5, s40, v8
	v_cndmask_b32_e32 v3, v8, v3, vcc
	v_cndmask_b32_e64 v3, v5, v3, s[2:3]
	v_mul_lo_u32 v4, v2, s41
	v_xor_b32_e32 v3, v3, v6
	v_sub_u32_e32 v7, v3, v6
	s_xor_b32 s2, s38, s11
	v_sub_u32_e32 v3, s39, v4
	v_cmp_le_u32_e32 vcc, s41, v3
	v_cmp_ge_u32_e64 s[0:1], s39, v4
	v_add_u32_e32 v4, 1, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v3, -1, v2
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e64 v2, v3, v2, s[0:1]
	v_xor_b32_e32 v5, s2, v2
	v_ashrrev_i32_e32 v2, 31, v1
	v_mad_i64_i32 v[3:4], s[0:1], v7, s42, v[1:2]
	v_subrev_u32_e32 v8, s2, v5
	s_ashr_i32 s6, s10, 31
	v_cmp_gt_i32_e64 s[38:39], s4, 0
	v_mad_i64_i32 v[3:4], s[0:1], v8, s13, v[3:4]
	s_mul_i32 s1, s9, s8
	s_mul_hi_i32 s0, s9, s8
	s_mul_i32 s2, s1, s6
	s_mul_hi_u32 s3, s1, s10
	s_mul_i32 s1, s1, s10
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s10
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v4, s1, v4
	v_mul_hi_u32 v5, s1, v3
	v_mul_lo_u32 v6, s2, v3
	v_mul_lo_u32 v3, s1, v3
	s_ashr_i32 s1, s4, 31
	v_add_u32_e32 v4, v5, v4
	v_mul_lo_u32 v5, v8, s12
	v_add_u32_e32 v4, v4, v6
	v_mul_hi_i32 v6, v8, s12
	s_mul_i32 s3, s31, s30
	v_mul_lo_u32 v9, v5, s33
	v_mul_hi_u32 v10, v5, s13
	v_mul_lo_u32 v5, v5, s13
	v_mul_lo_u32 v6, v6, s13
	s_mul_hi_i32 s2, s31, s30
	v_add_u32_e32 v9, v10, v9
	s_mul_i32 s0, s5, s12
	v_add_u32_e32 v6, v9, v6
	v_add_co_u32_e32 v1, vcc, v5, v1
	s_mul_i32 s1, s3, s1
	s_mul_hi_u32 s5, s3, s4
	s_mul_i32 s3, s3, s4
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	s_add_i32 s1, s5, s1
	s_mul_i32 s2, s2, s4
	s_add_i32 s1, s1, s2
	v_mul_lo_u32 v6, s3, v2
	v_mul_hi_u32 v9, s3, v1
	v_mul_lo_u32 v10, s1, v1
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	v_mul_lo_u32 v5, s3, v1
	v_add_co_u32_e32 v1, vcc, s36, v3
	v_add_u32_e32 v3, v9, v6
	v_add_u32_e32 v6, v3, v10
	v_addc_co_u32_e32 v2, vcc, v11, v4, vcc
	v_lshlrev_b64 v[3:4], 1, v[5:6]
	v_mul_hi_i32 v6, v7, s0
	v_mul_lo_u32 v5, v7, s0
	s_ashr_i32 s2, s16, 31
	v_mov_b32_e32 v7, s35
	v_add_co_u32_e32 v3, vcc, s34, v3
	v_mad_i64_i32 v[5:6], s[0:1], v8, s12, v[5:6]
	s_mul_i32 s1, s15, s14
	s_mul_hi_i32 s0, s15, s14
	s_mul_i32 s2, s1, s2
	s_mul_hi_u32 s3, s1, s16
	s_mul_i32 s1, s1, s16
	s_add_i32 s2, s3, s2
	s_mul_i32 s0, s0, s16
	s_add_i32 s2, s2, s0
	v_mul_lo_u32 v6, s1, v6
	v_mul_hi_u32 v8, s1, v5
	v_mul_lo_u32 v9, s2, v5
	v_mul_lo_u32 v5, s1, v5
	v_addc_co_u32_e32 v4, vcc, v7, v4, vcc
	v_add_u32_e32 v6, v8, v6
	v_add_u32_e32 v6, v6, v9
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	s_mul_i32 s5, s4, s31
	s_mul_i32 s8, s5, s30
	v_mov_b32_e32 v7, s27
	v_add_co_u32_e32 v5, vcc, s26, v5
	s_mul_i32 s8, s8, s13
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	v_cmp_gt_i32_e64 s[26:27], s12, 0
	v_cmp_gt_i32_e64 s[34:35], s30, 0
	v_cmp_gt_i32_e64 s[36:37], s31, 0
	s_mov_b64 s[40:41], 0
	s_mov_b32 s11, 0x7f800000
	s_movk_i32 s13, 0x7fff
	s_branch BB16_3
BB16_2:                                 ; %_Z19__float_to_bfloat16f.exit
                                        ;   in Loop: Header=BB16_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_mul_lo_u32 v8, v8, s9
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s24, v0
	s_or_b64 s[40:41], vcc, s[40:41]
	v_add_u32_e32 v7, v8, v7
	v_mul_lo_u32 v7, v7, s10
	v_lshrrev_b32_e32 v10, 16, v10
	v_add_u32_e32 v7, v7, v9
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v1, v7
	v_addc_co_u32_e64 v8, s[0:1], v2, v8, s[0:1]
	global_store_short v[7:8], v10, off
	s_andn2_b64 exec, exec, s[40:41]
	s_cbranch_execz BB16_39
BB16_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB16_6 Depth 2
                                        ;       Child Loop BB16_9 Depth 3
                                        ;         Child Loop BB16_19 Depth 4
                                        ;           Child Loop BB16_27 Depth 5
	s_add_i32 s0, s10, s6
	s_xor_b32 s0, s0, s6
	v_cvt_f32_u32_e32 v7, s0
	s_ashr_i32 s1, s9, 31
	v_ashrrev_i32_e32 v8, 31, v0
	s_add_i32 s2, s9, s1
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s33, s2, s1
	v_add_u32_e32 v12, v0, v8
	v_cvt_f32_u32_e32 v9, s33
	v_mul_f32_e32 v7, s25, v7
	v_cvt_u32_f32_e32 v7, v7
	s_ashr_i32 s42, s7, 31
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s25, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s6, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s33
	v_mul_hi_u32 v12, v9, s33
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s42
	s_xor_b32 s43, s0, s42
	v_cvt_f32_u32_e32 v11, s43
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v9
	v_rcp_iflag_f32_e32 v11, v11
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_mul_f32_e32 v10, s25, v11
	v_cvt_u32_f32_e32 v10, v10
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v13, v10, s43
	v_mul_hi_u32 v14, v10, s43
	v_xor_b32_e32 v12, v12, v8
	v_mul_hi_u32 v9, v9, v12
	v_sub_u32_e32 v15, 0, v13
	v_cmp_eq_u32_e32 vcc, 0, v14
	v_cndmask_b32_e32 v13, v13, v15, vcc
	v_mul_lo_u32 v11, v9, s33
	v_mul_hi_u32 v13, v13, v10
	v_mul_lo_u32 v9, v7, s10
	v_sub_u32_e32 v7, v12, v11
	v_cmp_ge_u32_e64 s[2:3], v12, v11
	v_add_u32_e32 v11, v10, v13
	v_sub_u32_e32 v10, v10, v13
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_mul_hi_u32 v10, v10, v0
	v_cmp_le_u32_e64 s[0:1], s33, v7
	v_subrev_u32_e32 v11, s33, v7
	s_and_b64 vcc, s[0:1], s[2:3]
	v_add_u32_e32 v14, s33, v7
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_mul_lo_u32 v11, v10, s43
	v_cndmask_b32_e64 v7, v14, v7, s[2:3]
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_sub_u32_e32 v8, v0, v11
	v_cmp_le_u32_e32 vcc, s43, v8
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v10
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v8, -1, v10
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_cndmask_b32_e64 v8, v8, v10, s[0:1]
	v_xor_b32_e32 v8, s42, v8
	v_mov_b32_e32 v10, 0
	v_subrev_u32_e32 v8, s42, v8
	s_andn2_b64 vcc, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	s_cbranch_vccnz BB16_35
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB16_3 Depth=1
	v_add_u32_e32 v10, s29, v0
	v_sub_u32_e32 v14, v10, v9
	v_mov_b32_e32 v10, 0
	v_add_u32_e32 v12, s23, v8
	v_add_u32_e32 v13, s28, v7
	s_mov_b32 s2, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s3, 0
	s_branch BB16_6
BB16_5:                                 ; %Flow244
                                        ;   in Loop: Header=BB16_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s8
	s_cmp_eq_u32 s3, s12
	s_cbranch_scc1 BB16_35
BB16_6:                                 ; %.preheader
                                        ;   Parent Loop BB16_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB16_9 Depth 3
                                        ;         Child Loop BB16_19 Depth 4
                                        ;           Child Loop BB16_27 Depth 5
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz BB16_5
; %bb.7:                                ; %.lr.ph245
                                        ;   in Loop: Header=BB16_6 Depth=2
	s_mul_i32 s33, s3, s14
	s_mov_b32 s42, 0
	s_mov_b32 s43, s2
	s_branch BB16_9
BB16_8:                                 ; %._crit_edge240
                                        ;   in Loop: Header=BB16_9 Depth=3
	s_add_i32 s42, s42, 1
	s_add_i32 s43, s43, s5
	s_cmp_eq_u32 s42, s30
	s_cbranch_scc1 BB16_5
BB16_9:                                 ;   Parent Loop BB16_3 Depth=1
                                        ;     Parent Loop BB16_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB16_19 Depth 4
                                        ;           Child Loop BB16_27 Depth 5
	s_mul_i32 s0, s42, s20
	v_subrev_u32_e32 v27, s0, v12
	v_cmp_lt_i32_e32 vcc, -1, v27
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $sgpr44
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $sgpr45
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr23_vgpr24
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr25_vgpr26
	s_and_saveexec_b64 s[46:47], vcc
	s_xor_b64 s[46:47], exec, s[46:47]
	s_cbranch_execz BB16_11
; %bb.10:                               ;   in Loop: Header=BB16_9 Depth=3
	s_ashr_i32 s44, s17, 31
	s_add_i32 s0, s17, s44
	s_xor_b32 s45, s0, s44
	v_cvt_f32_u32_e32 v15, s45
	v_ashrrev_i32_e32 v19, 31, v27
	v_rcp_iflag_f32_e32 v15, v15
	v_mul_f32_e32 v15, s25, v15
	v_cvt_u32_f32_e32 v17, v15
	v_mul_lo_u32 v20, v17, s45
	v_mul_hi_u32 v22, v17, s45
	v_sub_u32_e32 v21, 0, v20
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cndmask_b32_e32 v15, v20, v21, vcc
	v_mul_hi_u32 v16, v15, v17
	v_add_u32_e32 v15, v27, v19
	v_xor_b32_e32 v15, v15, v19
	v_add_u32_e32 v18, v17, v16
	v_sub_u32_e32 v16, v17, v16
	v_cndmask_b32_e32 v16, v16, v18, vcc
	v_mul_hi_u32 v16, v16, v15
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v24, v18
	v_mov_b32_e32 v23, v17
	v_mul_lo_u32 v25, v16, s45
	v_mov_b32_e32 v16, v18
	v_sub_u32_e32 v18, v15, v25
	v_cmp_ge_u32_e64 s[0:1], v15, v25
	v_cmp_le_u32_e32 vcc, s45, v18
	v_subrev_u32_e32 v25, s45, v18
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v26, s45, v18
	v_cndmask_b32_e32 v18, v18, v25, vcc
	v_cndmask_b32_e64 v18, v26, v18, s[0:1]
	v_xor_b32_e32 v18, v18, v19
	v_sub_u32_e32 v18, v18, v19
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_mov_b32_e32 v26, v16
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v25, v15
BB16_11:                                ; %Flow241
                                        ;   in Loop: Header=BB16_9 Depth=3
	s_or_saveexec_b64 s[46:47], s[46:47]
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v40, v26
	v_mov_b32_e32 v28, s44
	v_mov_b32_e32 v31, s45
	v_mov_b32_e32 v16, v19
	v_mov_b32_e32 v32, v15
	v_mov_b32_e32 v34, v20
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v38, v21
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v36, v23
	v_mov_b32_e32 v39, v25
	s_xor_b64 exec, exec, s[46:47]
	s_cbranch_execz BB16_13
; %bb.12:                               ; %._crit_edge117
                                        ;   in Loop: Header=BB16_9 Depth=3
	s_ashr_i32 s48, s17, 31
	s_add_i32 s49, s17, s48
	s_xor_b32 s49, s49, s48
	v_cvt_f32_u32_e32 v18, s49
	v_ashrrev_i32_e32 v16, 31, v27
	v_mov_b32_e32 v30, 0
	v_add_u32_e32 v27, v27, v16
	v_rcp_iflag_f32_e32 v18, v18
	v_mov_b32_e32 v33, v30
	v_xor_b32_e32 v32, v27, v16
	v_mov_b32_e32 v40, v33
	v_mul_f32_e32 v18, s25, v18
	v_cvt_u32_f32_e32 v29, v18
	v_mov_b32_e32 v37, v30
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v28, s48
	v_mul_lo_u32 v34, v29, s49
	v_mul_hi_u32 v35, v29, s49
	v_mov_b32_e32 v31, s49
	v_mov_b32_e32 v36, v29
	v_sub_u32_e32 v38, 0, v34
	v_mov_b32_e32 v39, v32
BB16_13:                                ; %Flow242
                                        ;   in Loop: Header=BB16_9 Depth=3
	s_or_b64 exec, exec, s[46:47]
	v_mov_b32_e32 v27, s45
	v_mov_b32_e32 v30, s44
	v_mov_b32_e32 v18, 1
	s_and_saveexec_b64 s[44:45], s[0:1]
	s_cbranch_execz BB16_15
; %bb.14:                               ;   in Loop: Header=BB16_9 Depth=3
	v_mov_b32_e32 v25, v39
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v26, v40
	v_mov_b32_e32 v21, v38
	v_mov_b32_e32 v22, v35
	v_mov_b32_e32 v20, v34
	v_mov_b32_e32 v24, v37
	v_mov_b32_e32 v17, v29
	v_mov_b32_e32 v27, v31
	v_mov_b32_e32 v15, v32
	v_mov_b32_e32 v30, v28
	v_mov_b32_e32 v19, v16
BB16_15:                                ;   in Loop: Header=BB16_9 Depth=3
	s_or_b64 exec, exec, s[44:45]
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB16_8
; %bb.16:                               ; %.lr.ph239
                                        ;   in Loop: Header=BB16_9 Depth=3
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB16_8
; %bb.17:                               ; %.lr.ph239.split.us.preheader
                                        ;   in Loop: Header=BB16_9 Depth=3
	v_cmp_eq_u32_e32 vcc, 0, v22
	v_cndmask_b32_e32 v16, v20, v21, vcc
	v_mul_lo_u32 v20, v16, v24
	v_mul_hi_u32 v16, v16, v23
	v_xor_b32_e32 v19, v19, v30
	s_mov_b32 s44, 0
	s_mov_b32 s45, s43
	v_add_u32_e32 v16, v16, v20
	v_add_u32_e32 v20, v17, v16
	v_sub_u32_e32 v16, v17, v16
	v_cndmask_b32_e32 v16, v16, v20, vcc
	v_mul_lo_u32 v17, v16, v26
	v_mul_hi_u32 v16, v16, v25
	v_add_u32_e32 v16, v16, v17
	v_mul_lo_u32 v17, v16, v27
	v_add_u32_e32 v20, 1, v16
	v_add_u32_e32 v21, -1, v16
	v_sub_u32_e32 v22, v15, v17
	v_cmp_ge_u32_e32 vcc, v15, v17
	v_cmp_ge_u32_e64 s[0:1], v22, v27
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v15, v16, v20, s[0:1]
	v_cndmask_b32_e32 v15, v21, v15, vcc
	v_xor_b32_e32 v15, v15, v19
	v_sub_u32_e32 v16, v15, v19
	v_add_u32_e32 v15, s33, v16
	v_mul_lo_u32 v15, v15, s15
	v_cmp_gt_i32_e32 vcc, s14, v16
	v_cndmask_b32_e32 v16, 0, v18, vcc
	s_branch BB16_19
BB16_18:                                ; %._crit_edge.us
                                        ;   in Loop: Header=BB16_19 Depth=4
	s_add_i32 s44, s44, 1
	s_add_i32 s45, s45, s4
	s_cmp_lg_u32 s44, s31
	s_cbranch_scc0 BB16_8
BB16_19:                                ; %.lr.ph239.split.us
                                        ;   Parent Loop BB16_3 Depth=1
                                        ;     Parent Loop BB16_6 Depth=2
                                        ;       Parent Loop BB16_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB16_27 Depth 5
	s_mul_i32 s0, s44, s21
	v_subrev_u32_e32 v29, s0, v13
	v_cmp_lt_i32_e32 vcc, -1, v29
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $sgpr46
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $sgpr47
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr25_vgpr26
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr27_vgpr28
	s_and_saveexec_b64 s[48:49], vcc
	s_xor_b64 s[48:49], exec, s[48:49]
	s_cbranch_execz BB16_21
; %bb.20:                               ;   in Loop: Header=BB16_19 Depth=4
	s_ashr_i32 s46, s18, 31
	s_add_i32 s0, s18, s46
	s_xor_b32 s47, s0, s46
	v_cvt_f32_u32_e32 v17, s47
	v_ashrrev_i32_e32 v21, 31, v29
	v_rcp_iflag_f32_e32 v17, v17
	v_mul_f32_e32 v17, s25, v17
	v_cvt_u32_f32_e32 v19, v17
	v_mul_lo_u32 v22, v19, s47
	v_mul_hi_u32 v24, v19, s47
	v_sub_u32_e32 v23, 0, v22
	v_cmp_eq_u32_e32 vcc, 0, v24
	v_cndmask_b32_e32 v17, v22, v23, vcc
	v_mul_hi_u32 v18, v17, v19
	v_add_u32_e32 v17, v29, v21
	v_xor_b32_e32 v17, v17, v21
	v_add_u32_e32 v20, v19, v18
	v_sub_u32_e32 v18, v19, v18
	v_cndmask_b32_e32 v18, v18, v20, vcc
	v_mul_hi_u32 v18, v18, v17
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v26, v20
	v_mov_b32_e32 v25, v19
	v_mul_lo_u32 v27, v18, s47
	v_mov_b32_e32 v18, v20
	v_sub_u32_e32 v20, v17, v27
	v_cmp_ge_u32_e64 s[0:1], v17, v27
	v_cmp_le_u32_e32 vcc, s47, v20
	v_subrev_u32_e32 v27, s47, v20
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v28, s47, v20
	v_cndmask_b32_e32 v20, v20, v27, vcc
	v_cndmask_b32_e64 v20, v28, v20, s[0:1]
	v_xor_b32_e32 v20, v20, v21
	v_sub_u32_e32 v20, v20, v21
	v_cmp_ne_u32_e32 vcc, 0, v20
	v_mov_b32_e32 v28, v18
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v27, v17
BB16_21:                                ; %Flow237
                                        ;   in Loop: Header=BB16_19 Depth=4
	s_or_saveexec_b64 s[48:49], s[48:49]
	v_mov_b32_e32 v38, v26
	v_mov_b32_e32 v41, v28
	v_mov_b32_e32 v20, s46
	v_mov_b32_e32 v32, s47
	v_mov_b32_e32 v18, v21
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v39, v23
	v_mov_b32_e32 v30, v19
	v_mov_b32_e32 v37, v25
	v_mov_b32_e32 v40, v27
	s_xor_b64 exec, exec, s[48:49]
	s_cbranch_execz BB16_23
; %bb.22:                               ; %.lr.ph239.split.us._crit_edge
                                        ;   in Loop: Header=BB16_19 Depth=4
	s_ashr_i32 s50, s18, 31
	s_add_i32 s51, s18, s50
	s_xor_b32 s51, s51, s50
	v_cvt_f32_u32_e32 v20, s51
	v_ashrrev_i32_e32 v18, 31, v29
	v_mov_b32_e32 v31, 0
	v_add_u32_e32 v29, v29, v18
	v_rcp_iflag_f32_e32 v20, v20
	v_mov_b32_e32 v34, v31
	v_xor_b32_e32 v33, v29, v18
	v_mov_b32_e32 v41, v34
	v_mul_f32_e32 v20, s25, v20
	v_cvt_u32_f32_e32 v30, v20
	v_mov_b32_e32 v38, v31
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v20, s50
	v_mul_lo_u32 v35, v30, s51
	v_mul_hi_u32 v36, v30, s51
	v_mov_b32_e32 v32, s51
	v_mov_b32_e32 v37, v30
	v_sub_u32_e32 v39, 0, v35
	v_mov_b32_e32 v40, v33
BB16_23:                                ; %Flow238
                                        ;   in Loop: Header=BB16_19 Depth=4
	s_or_b64 exec, exec, s[48:49]
	v_mov_b32_e32 v31, s47
	v_mov_b32_e32 v34, s46
	v_mov_b32_e32 v29, 1
	s_and_saveexec_b64 s[46:47], s[0:1]
	s_cbranch_execz BB16_25
; %bb.24:                               ;   in Loop: Header=BB16_19 Depth=4
	v_mov_b32_e32 v27, v40
	v_mov_b32_e32 v25, v37
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, v41
	v_mov_b32_e32 v23, v39
	v_mov_b32_e32 v24, v36
	v_mov_b32_e32 v22, v35
	v_mov_b32_e32 v26, v38
	v_mov_b32_e32 v19, v30
	v_mov_b32_e32 v31, v32
	v_mov_b32_e32 v17, v33
	v_mov_b32_e32 v34, v20
	v_mov_b32_e32 v21, v18
BB16_25:                                ; %.lr.ph.us
                                        ;   in Loop: Header=BB16_19 Depth=4
	s_or_b64 exec, exec, s[46:47]
	v_cmp_eq_u32_e32 vcc, 0, v24
	v_cndmask_b32_e32 v18, v22, v23, vcc
	v_mul_lo_u32 v20, v18, v26
	v_mul_hi_u32 v18, v18, v25
	s_mov_b32 s46, s45
	s_mov_b32 s48, s4
	v_add_u32_e32 v18, v18, v20
	v_add_u32_e32 v20, v19, v18
	v_sub_u32_e32 v18, v19, v18
	v_cndmask_b32_e32 v18, v18, v20, vcc
	v_mul_lo_u32 v19, v18, v28
	v_mul_hi_u32 v18, v18, v27
	v_xor_b32_e32 v20, v21, v34
	v_add_u32_e32 v18, v18, v19
	v_mul_lo_u32 v19, v18, v31
	v_add_u32_e32 v21, 1, v18
	v_add_u32_e32 v22, -1, v18
	v_sub_u32_e32 v23, v17, v19
	v_cmp_ge_u32_e32 vcc, v17, v19
	v_cmp_ge_u32_e64 s[0:1], v23, v31
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v17, v18, v21, s[0:1]
	v_cndmask_b32_e32 v17, v22, v17, vcc
	v_xor_b32_e32 v17, v17, v20
	v_sub_u32_e32 v18, v17, v20
	v_cmp_gt_i32_e32 vcc, s15, v18
	v_add_u32_e32 v18, v18, v15
	v_mul_lo_u32 v18, v18, s16
	v_cndmask_b32_e32 v17, 0, v29, vcc
	v_and_b32_e32 v17, v17, v16
	v_mov_b32_e32 v19, v14
	s_branch BB16_27
BB16_26:                                ;   in Loop: Header=BB16_27 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s48, s48, -1
	s_add_i32 s46, s46, 1
	s_cmp_lg_u32 s48, 0
	v_subrev_u32_e32 v19, s22, v19
	s_cbranch_scc0 BB16_18
BB16_27:                                ;   Parent Loop BB16_3 Depth=1
                                        ;     Parent Loop BB16_6 Depth=2
                                        ;       Parent Loop BB16_9 Depth=3
                                        ;         Parent Loop BB16_19 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_cmp_lt_i32_e32 vcc, -1, v19
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $sgpr47
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $sgpr49
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr28_vgpr29
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr27
                                        ; implicit-def: $vgpr26
                                        ; implicit-def: $vgpr30_vgpr31
	s_and_saveexec_b64 s[50:51], vcc
	s_xor_b64 s[50:51], exec, s[50:51]
	s_cbranch_execz BB16_29
; %bb.28:                               ;   in Loop: Header=BB16_27 Depth=5
	s_ashr_i32 s47, s19, 31
	s_add_i32 s0, s19, s47
	s_xor_b32 s49, s0, s47
	v_cvt_f32_u32_e32 v20, s49
	v_ashrrev_i32_e32 v24, 31, v19
	v_rcp_iflag_f32_e32 v20, v20
	v_mul_f32_e32 v20, s25, v20
	v_cvt_u32_f32_e32 v22, v20
	v_mul_lo_u32 v25, v22, s49
	v_mul_hi_u32 v27, v22, s49
	v_sub_u32_e32 v26, 0, v25
	v_cmp_eq_u32_e32 vcc, 0, v27
	v_cndmask_b32_e32 v20, v25, v26, vcc
	v_mul_hi_u32 v21, v20, v22
	v_add_u32_e32 v20, v19, v24
	v_xor_b32_e32 v20, v20, v24
	v_add_u32_e32 v23, v22, v21
	v_sub_u32_e32 v21, v22, v21
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_mul_hi_u32 v21, v21, v20
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v29, v23
	v_mov_b32_e32 v28, v22
	v_mul_lo_u32 v30, v21, s49
	v_mov_b32_e32 v21, v23
	v_sub_u32_e32 v23, v20, v30
	v_cmp_ge_u32_e64 s[0:1], v20, v30
	v_cmp_le_u32_e32 vcc, s49, v23
	v_subrev_u32_e32 v30, s49, v23
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v31, s49, v23
	v_cndmask_b32_e32 v23, v23, v30, vcc
	v_cndmask_b32_e64 v23, v31, v23, s[0:1]
	v_xor_b32_e32 v23, v23, v24
	v_sub_u32_e32 v23, v23, v24
	v_cmp_ne_u32_e32 vcc, 0, v23
	v_mov_b32_e32 v31, v21
	s_and_b64 s[0:1], vcc, exec
	v_mov_b32_e32 v30, v20
BB16_29:                                ; %Flow235
                                        ;   in Loop: Header=BB16_27 Depth=5
	s_or_saveexec_b64 s[50:51], s[50:51]
	v_mov_b32_e32 v41, v29
	v_mov_b32_e32 v44, v31
	v_mov_b32_e32 v23, s47
	v_mov_b32_e32 v35, s49
	v_mov_b32_e32 v21, v24
	v_mov_b32_e32 v36, v20
	v_mov_b32_e32 v38, v25
	v_mov_b32_e32 v39, v27
	v_mov_b32_e32 v42, v26
	v_mov_b32_e32 v33, v22
	v_mov_b32_e32 v40, v28
	v_mov_b32_e32 v43, v30
	s_xor_b64 exec, exec, s[50:51]
	s_cbranch_execz BB16_31
; %bb.30:                               ; %._crit_edge
                                        ;   in Loop: Header=BB16_27 Depth=5
	s_ashr_i32 s52, s19, 31
	s_add_i32 s53, s19, s52
	s_xor_b32 s53, s53, s52
	v_cvt_f32_u32_e32 v23, s53
	v_ashrrev_i32_e32 v21, 31, v19
	v_mov_b32_e32 v34, 0
	v_add_u32_e32 v32, v19, v21
	v_rcp_iflag_f32_e32 v23, v23
	v_mov_b32_e32 v37, v34
	v_xor_b32_e32 v36, v32, v21
	v_mov_b32_e32 v44, v37
	v_mul_f32_e32 v23, s25, v23
	v_cvt_u32_f32_e32 v33, v23
	v_mov_b32_e32 v41, v34
	s_or_b64 s[0:1], s[0:1], exec
	v_mov_b32_e32 v23, s52
	v_mul_lo_u32 v38, v33, s53
	v_mul_hi_u32 v39, v33, s53
	v_mov_b32_e32 v35, s53
	v_mov_b32_e32 v40, v33
	v_sub_u32_e32 v42, 0, v38
	v_mov_b32_e32 v43, v36
BB16_31:                                ; %Flow236
                                        ;   in Loop: Header=BB16_27 Depth=5
	s_or_b64 exec, exec, s[50:51]
	v_mov_b32_e32 v34, s49
	v_mov_b32_e32 v37, s47
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[50:51], s[0:1]
	s_cbranch_execz BB16_33
; %bb.32:                               ;   in Loop: Header=BB16_27 Depth=5
	v_mov_b32_e32 v30, v43
	v_mov_b32_e32 v28, v40
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, v44
	v_mov_b32_e32 v26, v42
	v_mov_b32_e32 v27, v39
	v_mov_b32_e32 v25, v38
	v_mov_b32_e32 v29, v41
	v_mov_b32_e32 v22, v33
	v_mov_b32_e32 v34, v35
	v_mov_b32_e32 v20, v36
	v_mov_b32_e32 v37, v23
	v_mov_b32_e32 v24, v21
BB16_33:                                ;   in Loop: Header=BB16_27 Depth=5
	s_or_b64 exec, exec, s[50:51]
	v_cmp_eq_u32_e32 vcc, 0, v27
	v_cndmask_b32_e32 v21, v25, v26, vcc
	v_mul_lo_u32 v23, v21, v29
	v_mul_hi_u32 v21, v21, v28
	v_add_u32_e32 v21, v21, v23
	v_add_u32_e32 v23, v22, v21
	v_sub_u32_e32 v21, v22, v21
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_mul_lo_u32 v22, v21, v31
	v_mul_hi_u32 v21, v21, v30
	v_xor_b32_e32 v23, v24, v37
	v_add_u32_e32 v21, v21, v22
	v_mul_lo_u32 v22, v21, v34
	v_add_u32_e32 v24, 1, v21
	v_add_u32_e32 v25, -1, v21
	v_sub_u32_e32 v26, v20, v22
	v_cmp_ge_u32_e32 vcc, v20, v22
	v_cmp_ge_u32_e64 s[0:1], v26, v34
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v20, v21, v24, s[0:1]
	v_cndmask_b32_e32 v20, v25, v20, vcc
	v_xor_b32_e32 v20, v20, v23
	v_sub_u32_e32 v20, v20, v23
	v_cmp_gt_i32_e32 vcc, s16, v20
	v_cndmask_b32_e32 v21, 0, v32, vcc
	v_and_b32_e32 v21, v17, v21
	v_cmp_ne_u32_e32 vcc, 0, v21
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB16_26
; %bb.34:                               ;   in Loop: Header=BB16_27 Depth=5
	v_add_u32_e32 v20, v20, v18
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshlrev_b64 v[20:21], 1, v[20:21]
	s_ashr_i32 s47, s46, 31
	v_add_co_u32_e32 v20, vcc, v5, v20
	s_lshl_b64 s[50:51], s[46:47], 1
	v_addc_co_u32_e32 v21, vcc, v6, v21, vcc
	v_mov_b32_e32 v23, s51
	v_add_co_u32_e32 v22, vcc, s50, v3
	v_addc_co_u32_e32 v23, vcc, v4, v23, vcc
	global_load_ushort v22, v[22:23], off
	global_load_ushort v20, v[20:21], off
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v22, 16, v22
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v20, 16, v20
	v_cvt_f64_f32_e32 v[20:21], v20
	v_cvt_f64_f32_e32 v[22:23], v22
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[10:11], v[20:21], v[22:23], v[10:11]
	s_branch BB16_26
BB16_35:                                ; %Flow246
                                        ;   in Loop: Header=BB16_3 Depth=1
	v_cvt_f32_f64_e32 v11, v[10:11]
	v_sub_u32_e32 v9, v0, v9
	v_and_b32_e32 v10, s11, v11
	v_cmp_ne_u32_e32 vcc, s11, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.36:                               ;   in Loop: Header=BB16_3 Depth=1
	v_bfe_u32 v10, v11, 16, 1
	v_add3_u32 v10, v11, v10, s13
; %bb.37:                               ; %Flow
                                        ;   in Loop: Header=BB16_3 Depth=1
	s_or_saveexec_b64 s[2:3], s[0:1]
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execz BB16_2
; %bb.38:                               ;   in Loop: Header=BB16_3 Depth=1
	v_mov_b32_e32 v10, 0
	v_or_b32_e32 v12, 0x10000, v11
	v_cmp_eq_u32_sdwa s[0:1], v11, v10 src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v10, v12, v11, s[0:1]
	s_branch BB16_2
BB16_39:                                ; %Flow248
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_bwd_ncdhw_ushort
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 45
		.amdhsa_next_free_sgpr 54
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end16:
	.size	naive_conv_bwd_ncdhw_ushort, .Lfunc_end16-naive_conv_bwd_ncdhw_ushort
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 3728
; NumSgprs: 56
; NumVgprs: 45
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 56
; NumVGPRsForWavesPerEU: 45
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	naive_conv_wrw_ncdhw_ushort ; -- Begin function naive_conv_wrw_ncdhw_ushort
	.globl	naive_conv_wrw_ncdhw_ushort
	.p2align	8
	.type	naive_conv_wrw_ncdhw_ushort,@function
naive_conv_wrw_ncdhw_ushort:              ; @naive_conv_wrw_ncdhw_ushort
naive_conv_wrw_ncdhw_ushort$local:
; %bb.0:
	s_load_dwordx16 s[8:23], s[4:5], 0x18
	s_load_dwordx2 s[24:25], s[4:5], 0x68
	s_load_dwordx4 s[28:31], s[4:5], 0x58
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s24, s31
	s_mul_i32 s26, s7, s13
	s_mul_i32 s26, s26, s30
	v_cmp_gt_i32_e32 vcc, s26, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB17_22
; %bb.1:                                ; %.lr.ph240
	s_ashr_i32 s0, s12, 31
	s_add_i32 s1, s12, s0
	s_xor_b32 s41, s1, s0
	v_cvt_f32_u32_e32 v1, s41
	s_mov_b32 s27, 0x4f800000
	s_ashr_i32 s1, s6, 31
	s_add_i32 s33, s6, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s33, s33, s1
	s_xor_b32 s40, s1, s0
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x8
	s_load_dwordx2 s[36:37], s[4:5], 0x10
	v_mul_f32_e32 v1, s27, v1
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s5, s10, 31
	s_mul_i32 s39, s9, s8
	s_ashr_i32 s4, s13, 31
	v_mul_lo_u32 v2, v1, s41
	v_mul_hi_u32 v3, v1, s41
	s_mul_i32 s5, s39, s5
	s_mul_hi_i32 s38, s9, s8
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	s_mul_i32 s38, s38, s10
	s_sub_i32 s29, 0, s29
	s_sub_i32 s42, 0, s28
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_mul_hi_u32 v1, v1, s33
	s_sub_i32 s43, 0, s23
	s_mul_i32 s44, s18, s10
	s_mov_b64 s[46:47], 0
	v_mul_lo_u32 v2, v1, s41
	v_add_u32_e32 v3, 1, v1
	v_add_u32_e32 v4, -1, v1
	s_mov_b32 s45, 0x7f800000
	v_sub_u32_e32 v5, s33, v2
	v_cmp_ge_u32_e32 vcc, s33, v2
	v_cmp_le_u32_e64 s[0:1], s41, v5
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	v_xor_b32_e32 v1, s40, v1
	v_subrev_u32_e32 v5, s40, v1
	s_mul_hi_u32 s0, s39, s10
	s_mul_i32 s1, s39, s10
	s_add_i32 s0, s0, s5
	v_mul_lo_u32 v2, v5, s12
	v_ashrrev_i32_e32 v1, 31, v5
	s_mul_i32 s4, s1, s4
	s_mul_hi_u32 s5, s1, s13
	s_mul_i32 s1, s1, s13
	v_mul_lo_u32 v4, s1, v1
	v_mul_hi_u32 v6, s1, v5
	s_add_i32 s0, s0, s38
	v_sub_u32_e32 v3, s6, v2
	s_add_i32 s4, s5, s4
	v_add_u32_e32 v2, v6, v4
	s_mul_i32 s0, s0, s13
	v_ashrrev_i32_e32 v4, 31, v3
	s_add_i32 s4, s4, s0
	v_mul_lo_u32 v1, s1, v5
	v_mad_i64_i32 v[3:4], s[0:1], v5, s12, v[3:4]
	v_mul_lo_u32 v7, s4, v5
	s_mul_i32 s1, s30, s13
	s_ashr_i32 s4, s31, 31
	s_mul_hi_i32 s0, s30, s13
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s3
	s_mul_i32 s3, s1, s4
	s_mul_hi_u32 s6, s1, s31
	s_ashr_i32 s5, s24, 31
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s31
	s_mul_i32 s1, s1, s31
	s_add_i32 s0, s3, s0
	s_mul_i32 s3, s1, s5
	s_mul_hi_u32 s6, s1, s24
	v_add_u32_e32 v2, v2, v7
	s_mul_i32 s1, s1, s24
	s_add_i32 s3, s6, s3
	s_mul_i32 s0, s0, s24
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_add_i32 s3, s3, s0
	v_mul_lo_u32 v7, s1, v4
	v_mul_hi_u32 v8, s1, v3
	v_mul_lo_u32 v9, s3, v3
	v_add_co_u32_e32 v1, vcc, s2, v1
	s_mul_i32 s2, s15, s14
	s_ashr_i32 s0, s16, 31
	v_mul_lo_u32 v5, s1, v3
	s_mul_hi_i32 s1, s15, s14
	s_mul_i32 s0, s2, s0
	s_mul_hi_u32 s3, s2, s16
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_u32_e32 v6, v8, v7
	s_mul_i32 s2, s2, s16
	s_add_i32 s0, s3, s0
	s_mul_i32 s1, s1, s16
	v_add_u32_e32 v6, v6, v9
	s_add_i32 s0, s0, s1
	v_mul_lo_u32 v9, s2, v4
	v_mul_hi_u32 v10, s2, v3
	v_mul_lo_u32 v11, s0, v3
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_mul_lo_u32 v7, s2, v3
	s_mul_i32 s0, s25, s16
	s_mul_i32 s0, s0, s15
	v_add_co_u32_e32 v3, vcc, s34, v5
	v_mov_b32_e32 v8, s35
	v_add_u32_e32 v5, v10, v9
	v_addc_co_u32_e32 v4, vcc, v8, v6, vcc
	v_add_u32_e32 v8, v5, v11
	s_mul_i32 s0, s0, s14
	v_lshlrev_b64 v[5:6], 1, v[7:8]
	s_mul_i32 s12, s0, s12
	s_mul_i32 s0, s25, s13
	s_mul_i32 s0, s0, s10
	s_mul_i32 s13, s0, s9
	s_mul_i32 s25, s17, s10
	v_mov_b32_e32 v7, s37
	v_add_co_u32_e32 v5, vcc, s36, v5
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc
	s_mul_i32 s6, s7, s30
	v_cmp_gt_i32_e64 s[34:35], s11, 0
	v_cmp_gt_i32_e64 s[36:37], s14, 0
	v_cmp_gt_i32_e64 s[38:39], s15, 0
	v_cmp_gt_i32_e64 s[40:41], s16, 0
	s_mul_i32 s33, s16, s15
	s_mul_i32 s13, s13, s8
	s_mul_i32 s25, s25, s9
	s_movk_i32 s48, 0x7fff
	s_branch BB17_3
BB17_2:                                 ; %_Z19__float_to_bfloat16f.exit
                                        ;   in Loop: Header=BB17_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_mul_lo_u32 v10, v10, s30
	v_add_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc, s26, v0
	s_or_b64 s[46:47], vcc, s[46:47]
	v_add_u32_e32 v9, v10, v9
	v_mul_lo_u32 v9, v9, s31
	v_add_u32_e32 v8, v9, v8
	v_mul_lo_u32 v8, v8, s24
	v_lshrrev_b32_e32 v9, 16, v11
	v_add_u32_e32 v7, v8, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[7:8], 1, v[7:8]
	v_add_co_u32_e64 v7, s[0:1], v3, v7
	v_addc_co_u32_e64 v8, s[0:1], v4, v8, s[0:1]
	global_store_short v[7:8], v9, off
	s_andn2_b64 exec, exec, s[46:47]
	s_cbranch_execz BB17_22
BB17_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB17_6 Depth 2
                                        ;       Child Loop BB17_9 Depth 3
                                        ;         Child Loop BB17_13 Depth 4
                                        ;           Child Loop BB17_16 Depth 5
	s_add_i32 s0, s24, s5
	s_xor_b32 s0, s0, s5
	v_cvt_f32_u32_e32 v7, s0
	v_ashrrev_i32_e32 v8, 31, v0
	v_add_u32_e32 v12, v0, v8
	s_add_i32 s1, s31, s4
	v_rcp_iflag_f32_e32 v7, v7
	s_xor_b32 s2, s1, s4
	v_cvt_f32_u32_e32 v9, s2
	s_ashr_i32 s49, s7, 31
	v_mul_f32_e32 v7, s27, v7
	v_cvt_u32_f32_e32 v7, v7
	v_rcp_iflag_f32_e32 v9, v9
	v_mul_lo_u32 v10, v7, s0
	v_mul_hi_u32 v11, v7, s0
	v_mul_f32_e32 v9, s27, v9
	v_cvt_u32_f32_e32 v9, v9
	v_sub_u32_e32 v13, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v11
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_mul_hi_u32 v10, v10, v7
	v_xor_b32_e32 v11, v12, v8
	v_xor_b32_e32 v8, s5, v8
	v_add_u32_e32 v12, v7, v10
	v_sub_u32_e32 v7, v7, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc
	v_mul_hi_u32 v7, v7, v11
	v_mul_lo_u32 v10, v9, s2
	v_mul_hi_u32 v12, v9, s2
	v_mul_lo_u32 v13, v7, s0
	v_add_u32_e32 v15, 1, v7
	v_add_u32_e32 v16, -1, v7
	v_sub_u32_e32 v14, 0, v10
	v_sub_u32_e32 v17, v11, v13
	v_cmp_ge_u32_e32 vcc, v11, v13
	v_cmp_le_u32_e64 s[0:1], s0, v17
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v7, v7, v15, s[0:1]
	s_add_i32 s0, s7, s49
	s_xor_b32 s3, s0, s49
	v_cvt_f32_u32_e32 v11, s3
	v_cndmask_b32_e32 v7, v16, v7, vcc
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_rcp_iflag_f32_e32 v11, v11
	v_mul_hi_u32 v10, v10, v9
	v_xor_b32_e32 v7, v7, v8
	v_sub_u32_e32 v7, v7, v8
	v_mul_f32_e32 v11, s27, v11
	v_cvt_u32_f32_e32 v11, v11
	v_add_u32_e32 v13, v9, v10
	v_sub_u32_e32 v9, v9, v10
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_mul_lo_u32 v10, v11, s3
	v_mul_hi_u32 v13, v11, s3
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_u32_e32 v12, v7, v8
	v_xor_b32_e32 v12, v12, v8
	v_sub_u32_e32 v14, 0, v10
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_mul_hi_u32 v9, v9, v12
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v11
	v_mul_lo_u32 v7, v7, s24
	v_mul_lo_u32 v9, v9, s2
	v_add_u32_e32 v14, v11, v10
	v_sub_u32_e32 v10, v11, v10
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_mul_hi_u32 v10, v10, v0
	v_sub_u32_e32 v13, v12, v9
	v_cmp_ge_u32_e32 vcc, v12, v9
	v_cmp_le_u32_e64 s[0:1], s2, v13
	v_add_u32_e32 v11, s2, v13
	v_subrev_u32_e32 v9, s2, v13
	s_ashr_i32 s2, s30, 31
	s_add_i32 s50, s30, s2
	s_xor_b32 s50, s50, s2
	v_mul_lo_u32 v12, v10, s3
	v_cvt_f32_u32_e32 v14, s50
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v9, v13, v9, s[0:1]
	v_sub_u32_e32 v13, v0, v12
	v_cmp_le_u32_e64 s[0:1], s3, v13
	v_rcp_iflag_f32_e32 v13, v14
	v_cmp_ge_u32_e64 s[2:3], v0, v12
	v_add_u32_e32 v12, 1, v10
	s_and_b64 s[0:1], s[0:1], s[2:3]
	v_mul_f32_e32 v13, s27, v13
	v_cvt_u32_f32_e32 v13, v13
	v_add_u32_e32 v14, -1, v10
	v_cndmask_b32_e64 v10, v10, v12, s[0:1]
	v_cndmask_b32_e64 v10, v14, v10, s[2:3]
	v_xor_b32_e32 v10, s49, v10
	v_mul_hi_u32 v14, v13, s50
	v_subrev_u32_e32 v10, s49, v10
	v_mul_lo_u32 v12, v13, s50
	s_ashr_i32 s49, s6, 31
	s_add_i32 s2, s6, s49
	s_xor_b32 s51, s2, s49
	v_cmp_eq_u32_e64 s[0:1], 0, v14
	v_cvt_f32_u32_e32 v14, s51
	v_sub_u32_e32 v16, 0, v12
	v_cndmask_b32_e64 v12, v12, v16, s[0:1]
	v_mul_hi_u32 v12, v12, v13
	v_rcp_iflag_f32_e32 v14, v14
	v_ashrrev_i32_e32 v15, 31, v10
	v_add_u32_e32 v10, v10, v15
	v_add_u32_e32 v16, v13, v12
	v_sub_u32_e32 v12, v13, v12
	v_mul_f32_e32 v13, s27, v14
	v_xor_b32_e32 v10, v10, v15
	v_cndmask_b32_e64 v12, v12, v16, s[0:1]
	v_cvt_u32_f32_e32 v13, v13
	v_mul_hi_u32 v12, v12, v10
	v_cndmask_b32_e32 v9, v11, v9, vcc
	v_xor_b32_e32 v9, v9, v8
	v_mul_hi_u32 v14, v13, s51
	v_mul_lo_u32 v11, v12, s50
	v_mul_lo_u32 v12, v13, s51
	v_sub_u32_e32 v8, v9, v8
	v_cmp_eq_u32_e32 vcc, 0, v14
	v_sub_u32_e32 v9, v10, v11
	v_sub_u32_e32 v16, 0, v12
	v_cndmask_b32_e32 v12, v12, v16, vcc
	v_mul_hi_u32 v12, v12, v13
	v_cmp_ge_u32_e64 s[2:3], v10, v11
	v_cmp_le_u32_e64 s[0:1], s50, v9
	v_add_u32_e32 v14, s50, v9
	v_add_u32_e32 v10, v13, v12
	v_sub_u32_e32 v11, v13, v12
	v_cndmask_b32_e32 v10, v11, v10, vcc
	v_mul_hi_u32 v10, v10, v0
	v_subrev_u32_e32 v11, s50, v9
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_mul_lo_u32 v11, v10, s51
	v_cndmask_b32_e64 v9, v14, v9, s[2:3]
	v_xor_b32_e32 v9, v9, v15
	v_sub_u32_e32 v7, v0, v7
	v_sub_u32_e32 v12, v0, v11
	v_cmp_le_u32_e32 vcc, s51, v12
	v_cmp_ge_u32_e64 s[0:1], v0, v11
	v_add_u32_e32 v11, 1, v10
	s_and_b64 vcc, vcc, s[0:1]
	v_add_u32_e32 v12, -1, v10
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_cndmask_b32_e64 v10, v12, v10, s[0:1]
	v_xor_b32_e32 v10, s49, v10
	v_mov_b32_e32 v12, 0
	v_sub_u32_e32 v9, v9, v15
	v_subrev_u32_e32 v10, s49, v10
	s_andn2_b64 vcc, exec, s[34:35]
	v_mov_b32_e32 v13, 0
	s_cbranch_vccnz BB17_18
; %bb.4:                                ; %.preheader.lr.ph
                                        ;   in Loop: Header=BB17_3 Depth=1
	v_mul_lo_u32 v11, v9, s20
	v_mul_lo_u32 v12, s8, v10
	v_mul_lo_u32 v14, v8, s21
	v_mul_lo_u32 v16, s22, v7
	s_mov_b32 s2, 0
	v_add3_u32 v12, s43, v11, v12
	v_mul_lo_u32 v12, s9, v12
	v_add_u32_e32 v15, s29, v16
	s_mov_b32 s3, 0
	v_add3_u32 v12, s42, v12, v14
	v_mul_lo_u32 v17, s10, v12
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_add3_u32 v16, s29, v17, v16
	s_branch BB17_6
BB17_5:                                 ; %Flow86
                                        ;   in Loop: Header=BB17_6 Depth=2
	s_add_i32 s3, s3, 1
	s_add_i32 s2, s2, s12
	s_cmp_eq_u32 s3, s11
	v_add_u32_e32 v16, s13, v16
	s_cbranch_scc1 BB17_18
BB17_6:                                 ; %.preheader
                                        ;   Parent Loop BB17_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB17_9 Depth 3
                                        ;         Child Loop BB17_13 Depth 4
                                        ;           Child Loop BB17_16 Depth 5
	s_andn2_b64 vcc, exec, s[36:37]
	s_cbranch_vccnz BB17_5
; %bb.7:                                ; %.lr.ph230
                                        ;   in Loop: Header=BB17_6 Depth=2
	s_mov_b32 s49, 0
	v_mov_b32_e32 v17, v16
	s_mov_b32 s50, s2
	s_branch BB17_9
BB17_8:                                 ; %._crit_edge225
                                        ;   in Loop: Header=BB17_9 Depth=3
	s_add_i32 s49, s49, 1
	s_add_i32 s50, s50, s33
	s_cmp_eq_u32 s49, s14
	v_add_u32_e32 v17, s25, v17
	s_cbranch_scc1 BB17_5
BB17_9:                                 ;   Parent Loop BB17_3 Depth=1
                                        ;     Parent Loop BB17_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB17_13 Depth 4
                                        ;           Child Loop BB17_16 Depth 5
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccnz BB17_8
; %bb.10:                               ; %.lr.ph224
                                        ;   in Loop: Header=BB17_9 Depth=3
	s_andn2_b64 vcc, exec, s[40:41]
	s_cbranch_vccnz BB17_8
; %bb.11:                               ; %.lr.ph224.split.us.preheader
                                        ;   in Loop: Header=BB17_9 Depth=3
	s_mul_i32 s0, s49, s17
	s_sub_i32 s0, s0, s23
	v_add_u32_e32 v18, s0, v11
	v_cmp_lt_i32_e32 vcc, -1, v18
	v_cmp_gt_i32_e64 s[0:1], s8, v18
	s_and_b64 s[52:53], vcc, s[0:1]
	s_mov_b32 s51, 0
	v_mov_b32_e32 v18, v17
	s_mov_b32 s54, s50
	s_branch BB17_13
BB17_12:                                ; %Flow82
                                        ;   in Loop: Header=BB17_13 Depth=4
	s_or_b64 exec, exec, s[56:57]
	s_add_i32 s51, s51, 1
	s_add_i32 s54, s54, s16
	s_cmp_lg_u32 s51, s15
	v_add_u32_e32 v18, s44, v18
	s_cbranch_scc0 BB17_8
BB17_13:                                ; %.lr.ph224.split.us
                                        ;   Parent Loop BB17_3 Depth=1
                                        ;     Parent Loop BB17_6 Depth=2
                                        ;       Parent Loop BB17_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB17_16 Depth 5
	s_mul_i32 s0, s51, s18
	s_sub_i32 s0, s0, s28
	v_add_u32_e32 v19, s0, v14
	v_cmp_lt_i32_e32 vcc, -1, v19
	v_cmp_gt_i32_e64 s[0:1], s9, v19
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 s[0:1], s[52:53], s[0:1]
	s_and_saveexec_b64 s[56:57], s[0:1]
	s_cbranch_execz BB17_12
; %bb.14:                               ; %.lr.ph.us.split.us.preheader
                                        ;   in Loop: Header=BB17_13 Depth=4
	s_mov_b32 s55, 0
	s_mov_b32 s58, s54
	s_mov_b32 s60, s16
	s_branch BB17_16
BB17_15:                                ;   in Loop: Header=BB17_16 Depth=5
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s60, s60, -1
	s_add_i32 s58, s58, 1
	s_add_i32 s55, s55, s19
	s_cmp_lg_u32 s60, 0
	s_cbranch_scc0 BB17_12
BB17_16:                                ; %.lr.ph.us.split.us
                                        ;   Parent Loop BB17_3 Depth=1
                                        ;     Parent Loop BB17_6 Depth=2
                                        ;       Parent Loop BB17_9 Depth=3
                                        ;         Parent Loop BB17_13 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	v_add_u32_e32 v19, s55, v15
	v_cmp_lt_i32_e32 vcc, -1, v19
	v_cmp_gt_i32_e64 s[0:1], s10, v19
	s_and_b64 s[62:63], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[62:63]
	s_cbranch_execz BB17_15
; %bb.17:                               ;   in Loop: Header=BB17_16 Depth=5
	v_add_u32_e32 v19, s55, v18
	v_ashrrev_i32_e32 v20, 31, v19
	v_lshlrev_b64 v[19:20], 1, v[19:20]
	s_ashr_i32 s59, s58, 31
	v_add_co_u32_e32 v19, vcc, v1, v19
	s_lshl_b64 s[62:63], s[58:59], 1
	v_addc_co_u32_e32 v20, vcc, v2, v20, vcc
	v_mov_b32_e32 v22, s63
	v_add_co_u32_e32 v21, vcc, s62, v5
	v_addc_co_u32_e32 v22, vcc, v6, v22, vcc
	global_load_ushort v21, v[21:22], off
	global_load_ushort v19, v[19:20], off
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v21, 16, v21
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v19, 16, v19
	v_cvt_f64_f32_e32 v[19:20], v19
	v_cvt_f64_f32_e32 v[21:22], v21
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
	v_fma_f64 v[12:13], v[19:20], v[21:22], v[12:13]
	s_branch BB17_15
BB17_18:                                ; %._crit_edge236
                                        ;   in Loop: Header=BB17_3 Depth=1
	v_cvt_f32_f64_e32 v12, v[12:13]
	v_and_b32_e32 v11, s45, v12
	v_cmp_ne_u32_e32 vcc, s45, v11
                                        ; implicit-def: $vgpr11
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.19:                               ;   in Loop: Header=BB17_3 Depth=1
	v_bfe_u32 v11, v12, 16, 1
	v_add3_u32 v11, v12, v11, s48
; %bb.20:                               ; %Flow
                                        ;   in Loop: Header=BB17_3 Depth=1
	s_or_saveexec_b64 s[2:3], s[0:1]
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execz BB17_2
; %bb.21:                               ;   in Loop: Header=BB17_3 Depth=1
	v_mov_b32_e32 v11, 0
	v_or_b32_e32 v13, 0x10000, v12
	v_cmp_eq_u32_sdwa s[0:1], v12, v11 src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e64 v11, v13, v12, s[0:1]
	s_branch BB17_2
BB17_22:                                ; %Flow90
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel naive_conv_wrw_ncdhw_ushort
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 23
		.amdhsa_next_free_sgpr 64
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end17:
	.size	naive_conv_wrw_ncdhw_ushort, .Lfunc_end17-naive_conv_wrw_ncdhw_ushort
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2160
; NumSgprs: 66
; NumVgprs: 23
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 23
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.ident	"clang version 11.0.0 (/src/external/llvm-project/clang 6c08b900599eee52e12bce1e76b20dc413ce30e7)"
	.section	".note.GNU-stack"
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_fwd_nchw_float
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         naive_conv_fwd_nchw_float.kd
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_bwd_nchw_float
    .private_segment_fixed_size: 0
    .sgpr_count:     41
    .sgpr_spill_count: 0
    .symbol:         naive_conv_bwd_nchw_float.kd
    .vgpr_count:     41
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_wrw_nchw_float
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         naive_conv_wrw_nchw_float.kd
    .vgpr_count:     21
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_fwd_ncdhw_float
    .private_segment_fixed_size: 0
    .sgpr_count:     60
    .sgpr_spill_count: 0
    .symbol:         naive_conv_fwd_ncdhw_float.kd
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_bwd_ncdhw_float
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         naive_conv_bwd_ncdhw_float.kd
    .vgpr_count:     45
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_wrw_ncdhw_float
    .private_segment_fixed_size: 0
    .sgpr_count:     64
    .sgpr_spill_count: 0
    .symbol:         naive_conv_wrw_ncdhw_float.kd
    .vgpr_count:     23
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_fwd_nchw_half
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         naive_conv_fwd_nchw_half.kd
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_bwd_nchw_half
    .private_segment_fixed_size: 0
    .sgpr_count:     41
    .sgpr_spill_count: 0
    .symbol:         naive_conv_bwd_nchw_half.kd
    .vgpr_count:     41
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_wrw_nchw_half
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         naive_conv_wrw_nchw_half.kd
    .vgpr_count:     21
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_fwd_ncdhw_half
    .private_segment_fixed_size: 0
    .sgpr_count:     60
    .sgpr_spill_count: 0
    .symbol:         naive_conv_fwd_ncdhw_half.kd
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_bwd_ncdhw_half
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         naive_conv_bwd_ncdhw_half.kd
    .vgpr_count:     45
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_wrw_ncdhw_half
    .private_segment_fixed_size: 0
    .sgpr_count:     64
    .sgpr_spill_count: 0
    .symbol:         naive_conv_wrw_ncdhw_half.kd
    .vgpr_count:     23
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_fwd_nchw_ushort
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         naive_conv_fwd_nchw_ushort.kd
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_bwd_nchw_ushort
    .private_segment_fixed_size: 0
    .sgpr_count:     43
    .sgpr_spill_count: 0
    .symbol:         naive_conv_bwd_nchw_ushort.kd
    .vgpr_count:     41
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_wrw_nchw_ushort
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         naive_conv_wrw_nchw_ushort.kd
    .vgpr_count:     21
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_fwd_ncdhw_ushort
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         naive_conv_fwd_ncdhw_ushort.kd
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_bwd_ncdhw_ushort
    .private_segment_fixed_size: 0
    .sgpr_count:     56
    .sgpr_spill_count: 0
    .symbol:         naive_conv_bwd_ncdhw_ushort.kd
    .vgpr_count:     45
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .access:         read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .access:         read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i16
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         36
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         44
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 168
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           naive_conv_wrw_ncdhw_ushort
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         naive_conv_wrw_ncdhw_ushort.kd
    .vgpr_count:     23
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
