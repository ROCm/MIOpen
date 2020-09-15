.include "rocm_version.inc".include "inst_wrappers.inc".include "utilities.inc".include
    "gpr_alloc.inc"

    // kernarg layout:
    kernarg = 4 in_desc = 0.set in_ptr_off,
    0x0.set out_ptr_off, 0x8.set scale_ptr_off, 0x10.set bias_ptr_off, 0x18.set inhw_off,
    0x20.if(MIO_SAVE_MEAN_VARIANCE == 1) && (MIO_RUNNING_RESULT == 1).set expAvgFactor_off,
    0x28.set resultRunningMean_off, 0x30.set resultRunningVariance_off, 0x38.set epsilon_off,
    0x40.set resultSaveMean_off, 0x48.set resultSaveInvVariance_off,
    0x50.elseif(MIO_SAVE_MEAN_VARIANCE == 0) && (MIO_RUNNING_RESULT == 1).set expAvgFactor_off,
    0x28.set resultRunningMean_off, 0x30.set resultRunningVariance_off, 0x38.set epsilon_off,
    0x40.set resultSaveMean_off, 0x0.set resultSaveInvVariance_off,
    0x0.elseif(MIO_SAVE_MEAN_VARIANCE == 1) && (MIO_RUNNING_RESULT == 0).set expAvgFactor_off,
    0x0.set resultRunningMean_off, 0x0.set resultRunningVariance_off, 0x0.set epsilon_off,
    0x28.set resultSaveMean_off, 0x30.set resultSaveInvVariance_off,
    0x38.elseif(MIO_SAVE_MEAN_VARIANCE == 0) && (MIO_RUNNING_RESULT == 0).set expAvgFactor_off,
    0x0.set resultRunningMean_off, 0x0.set resultRunningVariance_off, 0x0.set epsilon_off,
    0x28.set resultSaveMean_off, 0x0.set resultSaveInvVariance_off,
    0x0.endif

    madmix_instructions_available = 0 fmamix_instructions_available =
        0.if(.option.machine_version_major == 9)
            .if(.option.machine_version_stepping > 2) fmamix_instructions_available =
            1.else madmix_instructions_available = 1.endif
                                                       .endif

                                                       .GPR_ALLOC_BEGIN
                                                       //.SGPR_ALLOC_FROM 4
                                                       .SGPR_ALLOC_FROM 0.SGPR_ALLOC stmp,
    30.SGPR_RESERVE_XNACK
        .SGPR_RESERVE_VCC

        .VGPR_ALLOC_FROM 0.VGPR_ALLOC tid.VGPR_ALLOC vtmp,
    12

        .LDS_ALLOC_FROM 0.LDS_ALLOC UNUSED_accums_lds,
    136.GPR_ALLOC_END

        .text.globl miopenGcnAsmBNFwdTrainSpatial.p2align 8.type miopenGcnAsmBNFwdTrainSpatial,
    @function miopenGcnAsmBNFwdTrainSpatial : s_add_u32 flat_scratch_lo,
    s[stmp + 6],
    s[stmp + 9] s_mov_b32 s[stmp + 14],
    s[stmp + 8] s_mov_b32 s[stmp + 15],
    0 v_cmp_eq_u32 s [stmp + 10:stmp + 11],
    0,
    v[tid] s_addc_u32 flat_scratch_hi,
    s[stmp + 7],
    0 s_and_saveexec_b64 s [stmp + 6:stmp + 7],
    s [stmp + 10:stmp + 11] s_cbranch_execz skip_store_bias_scale;
If loacl ID is 0, then store bias and scale...
    // read scale ptr
    s_load_dwordx2 s [stmp + 12:stmp + 13],
s [kernarg:kernarg + 1], 0x0 + scale_ptr_off s_lshl_b64 s [stmp + 18:stmp + 19],
s [stmp + 14:stmp + 15], 2
    // read bias ptr
    s_load_dwordx2 s [stmp + 16:stmp + 17],
s [kernarg:kernarg + 1], 0x0 + bias_ptr_off v_mov_b32 v[vtmp],
    0 s_waitcnt lgkmcnt(0) s_add_u32 s[stmp + 12], s[stmp + 12],
    s[stmp + 18] s_addc_u32 s[stmp + 13], s[stmp + 13], s[stmp + 19] s_load_dword s[stmp + 8],
s [stmp + 12:stmp + 13], 0x0 s_add_u32 s[stmp + 16], s[stmp + 16],
    s[stmp + 18] s_addc_u32 s[stmp + 17], s[stmp + 17],
    s[stmp + 19] s_waitcnt lgkmcnt(0) v_mov_b32 v[vtmp + 1], s[stmp + 8] s_load_dword s[stmp + 8],
s [stmp + 16:stmp + 17], 0x0 s_waitcnt lgkmcnt(0) v_mov_b32 v[vtmp + 2],
    s[stmp + 8] ds_write2_b32 v[vtmp], v[vtmp + 2],
    v[vtmp + 1] offset1 : 1 skip_store_bias_scale : s_or_b64 exec,
                                                    exec,
                                                    s [stmp + 6:stmp + 7] v_cvt_f32_u32 v[vtmp],
                                                    v[tid] s_mov_b32 s[stmp + 6],
                                                    0 + MIO_BN_HW;
Read input arguments... s_load_dwordx2 s [stmp + 12:stmp + 13], s [kernarg:kernarg + 1],
    0x0 + resultSaveInvVariance_off s_load_dwordx2 s [stmp + 16:stmp + 17], s [kernarg:kernarg + 1],
    0x0 + resultSaveMean_off s_load_dwordx2 s [stmp + 18:stmp + 19], s [kernarg:kernarg + 1],
    0x0 + resultRunningVariance_off v_mul_f32 v[vtmp], 0 + MIO_BN_IHW_DIV,
    v[vtmp] v_cvt_u32_f32 v[vtmp], v[vtmp] s_load_dwordx2 s [stmp + 22:stmp + 23],
s [kernarg:kernarg + 1], 0x0 + out_ptr_off s_load_dword s[stmp + 8], s [kernarg:kernarg + 1],
    0x0 + inhw_off s_load_dwordx2 s [stmp + 20:stmp + 21], s [kernarg:kernarg + 1],
    0x0 + resultRunningMean_off v_sub_u32 v[vtmp + 1], s[stmp + 14],
    v[vtmp] v_mul_lo_u32 v[vtmp + 1], v[vtmp + 1], s[stmp + 6] s_mov_b32 s[stmp + 6],
    0 + MIO_BN_SEGMENT v_mov_b32 v[vtmp + 4], 0 v_cmp_gt_u32 s [stmp + 6:stmp + 7], s[stmp + 6],
    v[tid] v_add_u32 v[vtmp + 1], v[vtmp + 1], v[tid] v_mov_b32 v[vtmp + 2],
    v[vtmp + 4] s_waitcnt lgkmcnt(0) s_barrier s_waitcnt lgkmcnt(0) s_and_saveexec_b64 s
    [stmp + 24:stmp + 25],
s [stmp + 6:stmp + 7] s_cbranch_execz skip_mean_variance_calc;
If local ID > MIO_BN_SEGMENT then skip mean / variance calculation... s_mov_b32 s[stmp + 28], 0
    // s_movk_i32 s[stmp+26], 0+MIO_BN_CHW-MIO_BN_HW
    s_mov_b32 s[stmp + 26],
    0 + MIO_BN_CHW - MIO_BN_HW v_mul_lo_u32 v[vtmp + 2], v[vtmp], s[stmp + 26]
    // load input ptr
    s_load_dwordx2 s [stmp + 26:stmp + 27],
s [kernarg:kernarg + 1], 0x0 + in_ptr_off s_mul_i32 s[stmp + 29], s[stmp + 14],
    0 + MIO_BN_HW s_cmpk_eq_i32 s[stmp + 28], 0 + MIO_BN_NLOOPM v_mov_b32 v[vtmp + 4],
    0 v_add3_u32 v[vtmp + 2], v[tid], v[vtmp + 2], s[stmp + 29] v_mov_b32 v[vtmp + 5],
    v[vtmp + 4] s_cbranch_scc1 skip_loop_1 mean_variance_calc_loop
    : v_mov_b32 v[vtmp + 3],
      0 v_lshlrev_b64 v [vtmp + 6:vtmp + 7],
      2,
      v [vtmp + 2:vtmp + 3] s_waitcnt lgkmcnt(0) v_mov_b32 v[vtmp + 8],
      s[stmp + 27] v_add_co_u32 v[vtmp + 6],
      vcc,
      s[stmp + 26],
      v[vtmp + 6] v_addc_co_u32 v[vtmp + 7],
      vcc,
      v[vtmp + 8],
      v[vtmp + 7],
      vcc global_load_dword v[vtmp + 6],
      v [vtmp + 6:vtmp + 7],
      off v_mov_b32 v[vtmp + 9],
      4 v_add_u32 v[vtmp + 9],
      s[stmp + 28],
      v[vtmp + 9] s_add_i32 s[stmp + 28],
      s[stmp + 28],
      4 v_add_u32 v[vtmp + 2],
      0 + MIO_BN_SEG_OFFSET,
      v[vtmp + 2] s_cmpk_eq_i32 s[stmp + 28],
      0 + MIO_BN_NLOOPM * 4 s_waitcnt vmcnt(0) v_add_f32 v[vtmp + 5],
      v[vtmp + 5],
      v[vtmp + 6] v_fmac_f32 v[vtmp + 4],
      v[vtmp + 6],
      v[vtmp + 6] buffer_store_dword v[vtmp + 6],
      v[vtmp + 9],
      s [stmp:stmp + 3],
      s[stmp + 9] offen s_cbranch_scc0 mean_variance_calc_loop skip_loop_1:;Mean/varaiance calculation for last iteration outside for loop
	s_mov_b32 s[stmp+28], 0+MIO_BN_CHW
	v_mul_lo_u32 v[vtmp+2], v[vtmp], s[stmp+28]
	s_mov_b32 s[stmp+28], 0+MIO_BN_CHW_SNHW
	v_add3_u32 v[vtmp+2], v[vtmp+1], v[vtmp+2], s[stmp+28]
	s_mov_b32 s[stmp+28], 0+MIO_BN_NCHW
	v_cmp_gt_u32 vcc, s[stmp+28], v[vtmp+2]
	s_and_saveexec_b64 s[stmp+28:stmp+29], vcc
	s_cbranch_execz skip_loop_2
	v_mov_b32 v[vtmp+3], 0
	v_lshlrev_b64 v[vtmp+2:vtmp+3], 2, v[vtmp+2:vtmp+3]
	v_mov_b32 v[vtmp+6], s[stmp+27]
	v_add_co_u32 v[vtmp+2], vcc, s[stmp+26], v[vtmp+2]
	v_addc_co_u32 v[vtmp+3], vcc, v[vtmp+6], v[vtmp+3], vcc
	global_load_dword v[vtmp+3], v[vtmp+2:vtmp+3], off
skip_loop_2:
	s_or_b64 exec, exec, s[stmp+28:stmp+29]
	s_waitcnt vmcnt(0)
	v_add_f32 v[vtmp+2], v[vtmp+5], v[vtmp+3]
	v_fmac_f32 v[vtmp+4], v[vtmp+3], v[vtmp+3]
	buffer_store_dword v[vtmp+3], off, s[stmp:stmp+3], s[stmp+9] offset:208
skip_mean_variance_calc:
	s_or_b64 exec, exec, s[stmp+24:stmp+25]
        s_load_dwordx2 s[stmp+24:stmp+25], s[kernarg:kernarg+1], 0x0 + expAvgFactor_off
        s_load_dwordx2 s[stmp+4:stmp+5], s[kernarg:kernarg+1], 0x0 + epsilon_off
	v_and_b32 v[vtmp+5], 63, v[tid]
	v_cmp_eq_u32 vcc, 63, v[vtmp+5]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
;
DPP reduction s_nop 4 v_add_f32 v[vtmp + 2] v[vtmp + 2] v[vtmp +
                                                          2] row_shr : 1 bound_ctrl : 0 v_add_f32
    v[vtmp + 4] v[vtmp + 4] v[vtmp + 4] row_shr : 1 bound_ctrl : 0 s_nop 0 v_add_f32
        v[vtmp + 2] v[vtmp + 2] v[vtmp + 2] row_shr : 2 bound_ctrl : 0 v_add_f32
            v[vtmp + 4] v[vtmp + 4] v[vtmp + 4] row_shr : 2 bound_ctrl : 0 s_nop 0 v_add_f32
                v[vtmp + 2] v[vtmp + 2] v[vtmp + 2] row_shr : 4 bank_mask : 0xe v_add_f32
                    v[vtmp + 4] v[vtmp + 4] v[vtmp +
                                              4] row_shr : 4 bank_mask : 0xe s_nop 0 v_add_f32
                        v[vtmp + 2] v[vtmp + 2] v[vtmp + 2] row_shr : 8 bank_mask : 0xc v_add_f32
                            v[vtmp + 4] v[vtmp +
                                          4] v[vtmp +
                                               4] row_shr : 8 bank_mask : 0xc s_nop 0 v_add_f32
                                v[vtmp + 2] v[vtmp + 2] v[vtmp +
                                                          2] row_bcast : 15 row_mask : 0xa v_add_f32
                                    v[vtmp + 4] v[vtmp + 4] v
                                        [vtmp + 4] row_bcast : 15 row_mask : 0xa s_nop 0 v_add_f32
                                            v[vtmp + 2] v[vtmp + 2] v
                                                [vtmp + 2] row_bcast : 31 row_mask : 0xc v_add_f32
                                                    v[vtmp + 4] v[vtmp + 4] v
                                                        [vtmp +
                                                         4] row_bcast : 31 row_mask : 0xc s_nop 0;
DPP reduction end s_and_saveexec_b64 s [stmp + 26:stmp + 27], vcc v_lshrrev_b32 v[vtmp + 5], 4,
    v[tid] v_and_b32 v[vtmp + 5], 60, v[vtmp + 5] ds_write2_b32 v[vtmp + 5], v[vtmp + 4],
    v[vtmp + 2] offset0 : 2 offset1 : 0 + MIO_BN_LDSGCN_SIZE + 2 s_or_b64 exec, exec,
s [stmp + 26:stmp + 27] v_mov_b32 v[vtmp + 4], 0 v_mov_b32 v[vtmp + 2], 0 v_mov_b32 v[vtmp + 5],
    v[vtmp + 4] s_waitcnt lgkmcnt(0) s_barrier s_waitcnt lgkmcnt(0);
Mean / Variance reduce loop reduce_mean_variance_loop : ds_read2_b32 v [vtmp + 6:vtmp + 7],
    v[vtmp + 2] offset0 : 0 + MIO_BN_LDSGCN_SIZE + 2 offset1 : 0 + MIO_BN_LDSGCN_SIZE +
        3 ds_read2_b32 v [vtmp + 8:vtmp + 9],
    v[vtmp + 2] offset0 : 2 offset1 : 3 v_add_u32 v[vtmp + 2], 8, v[vtmp + 2] v_cmp_ne_u32 vcc,
    0 + MIO_BN_LDSGCN_SIZE * 4, v[vtmp + 2] s_and_b64 vcc, exec,
    vcc s_waitcnt lgkmcnt(1) v_add_f32 v[vtmp + 5], v[vtmp + 5],
    v[vtmp + 6] s_waitcnt lgkmcnt(0) v_add_f32 v[vtmp + 4], v[vtmp + 4],
    v[vtmp + 8] v_add_f32 v[vtmp + 5], v[vtmp + 5], v[vtmp + 7] v_add_f32 v[vtmp + 4], v[vtmp + 4],
    v[vtmp + 9] s_cbranch_vccnz reduce_mean_variance_loop

    v_cvt_f32_f64 v[vtmp + 6],
s [stmp + 4:stmp + 5] v_mul_f32 v[vtmp + 2], s[stmp + 8], v[vtmp + 5] v_mul_f32 v[vtmp + 4],
    s[stmp + 8], v[vtmp + 4] v_fmac_f32 v[vtmp + 4], -v[vtmp + 2], v[vtmp + 2] v_cmp_ngt_f32 vcc, 0,
    v[vtmp + 4] v_cndmask_b32 v[vtmp + 5], 0, v[vtmp + 4], vcc v_add_f32 v[vtmp + 4], v[vtmp + 5],
    v[vtmp + 6] s_mov_b32 s[stmp + 4], 0xd800000 v_cmp_gt_f32 vcc, s[stmp + 4],
    v[vtmp + 4] v_mov_b32 v[vtmp + 7], 0x71800000 v_cndmask_b32 v[vtmp + 7], 1.0, v[vtmp + 7],
    vcc v_mul_f32 v[vtmp + 4], v[vtmp + 4], v[vtmp + 7] v_rsq_f32 v[vtmp + 4],
    v[vtmp + 4] v_mov_b32 v[vtmp + 6], 0x58800000 v_cndmask_b32 v[vtmp + 6], 1.0, v[vtmp + 6],
    vcc s_mov_b32 s[stmp + 8], 0 v_mul_f32 v[vtmp + 4], v[vtmp + 6],
    v[vtmp + 4] s_and_saveexec_b64 s [stmp + 4:stmp + 5],
s [stmp + 6:stmp + 7] s_cbranch_execnz skip_store_vars s_or_b64 exec, exec,
s [stmp + 4:stmp + 5] s_and_saveexec_b64 s [stmp + 4:stmp + 5],
s [stmp + 10:stmp + 11] s_cbranch_execnz store_running_saved_vars BN_no_update
    : s_endpgm skip_store_vars
      : v_mov_b32 v[vtmp + 6],
        0 s_mov_b32 s[stmp + 6],
        0 + MIO_BN_CHW - MIO_BN_HW v_mul_lo_u32 v[vtmp + 8],
        v[vtmp],
        s[stmp + 6] s_mul_i32 s[stmp + 6],
        s[stmp + 14],
        0 + MIO_BN_HW s_cmpk_eq_i32 s[stmp + 8],
        0 + MIO_BN_NLOOPM ds_read2_b32 v [vtmp + 6:vtmp + 7],
        v[vtmp + 6] offset1 : 1 v_add3_u32 v[vtmp + 8],
        v[tid],
        v[vtmp + 8],
        s[stmp + 6] s_cbranch_scc1 skip_loop_3 BN_update_loop
        : v_mov_b32 v[vtmp - 1],
          4 v_add_u32 v[vtmp - 1],
          s[stmp + 8],
          v[vtmp - 1] buffer_load_dword v[vtmp - 1],
          v[vtmp - 1],
          s [stmp:stmp + 3],
          s[stmp + 9] offen v_mov_b32 v[vtmp + 9],
          0 v_lshlrev_b64 v [vtmp + 9:vtmp + 10],
          2,
          v [vtmp + 8:vtmp + 9] s_add_i32 s[stmp + 8],
          s[stmp + 8],
          4 v_mov_b32 v[vtmp + 11],
          s[stmp + 23] v_add_co_u32 v[vtmp + 9],
          vcc,
          s[stmp + 22],
          v[vtmp + 9] v_add_u32 v[vtmp + 8],
          0 + MIO_BN_SEG_OFFSET,
          v[vtmp + 8] s_cmpk_eq_i32 s[stmp + 8],
          0 + MIO_BN_NLOOPM * 4 v_addc_co_u32 v[vtmp + 10],
          vcc,
          v[vtmp + 11],
          v[vtmp + 10],
          vcc s_waitcnt vmcnt(0) v_sub_f32 v[vtmp - 1],
          v[vtmp - 1],
          v[vtmp + 2] v_mul_f32 v[vtmp - 1],
          v[vtmp + 4],
          v[vtmp - 1] s_waitcnt lgkmcnt(0) v_fma_f32 v[vtmp - 1],
          v[vtmp + 7],
          v[vtmp - 1],
          v[vtmp + 6] global_store_dword v [vtmp + 9:vtmp + 10],
          v[vtmp - 1],
          off s_cbranch_scc0 BN_update_loop skip_loop_3
          : s_mov_b32 s[stmp + 6],
            0 + MIO_BN_CHW v_mul_lo_u32 v[vtmp - 1],
            v[vtmp],
            s[stmp + 6] s_mov_b32 s[stmp + 6],
            0 + MIO_BN_CHW_SNHW v_add3_u32 v[vtmp - 1],
            v[vtmp + 1],
            v[vtmp - 1],
            s[stmp + 6] s_mov_b32 s[stmp + 6],
            0 + MIO_BN_NCHW v_cmp_gt_u32 vcc,
            s[stmp + 6],
            v[vtmp - 1] s_and_saveexec_b64 s [stmp + 6:stmp + 7],
            vcc s_cbranch_execz BN_skip_store v_mov_b32 v[vtmp],
            0 v_lshlrev_b64 v [vtmp - 1:vtmp],
            2,
            v [vtmp - 1:vtmp] v_mov_b32 v[vtmp + 1],
            s[stmp + 23] v_add_co_u32 v[vtmp - 1],
            vcc,
            s[stmp + 22],
            v[vtmp - 1] v_addc_co_u32 v[vtmp],
            vcc,
            v[vtmp + 1],
            v[vtmp],
            vcc v_sub_f32 v[vtmp + 1],
            v[vtmp + 3],
            v[vtmp + 2] v_mul_f32 v[vtmp + 1],
            v[vtmp + 1],
            v[vtmp + 4] v_fmac_f32 v[vtmp + 6],
            v[vtmp + 7],
            v[vtmp + 1] global_store_dword v [vtmp - 1:vtmp],
            v[vtmp + 6],
            off BN_skip_store
            : s_or_b64 exec,
              exec,
              s [stmp + 6:stmp + 7] s_or_b64 exec,
              exec,
              s [stmp + 4:stmp + 5] s_and_saveexec_b64 s
              [stmp + 4:stmp + 5],
              s [stmp + 10:stmp + 11] s_cbranch_execz BN_no_update store_running_saved_vars
              : s_lshl_b64 s [stmp + 4:stmp + 5],
                s [stmp + 14:stmp + 15],
                2 s_add_u32 s[stmp + 6],
                s[stmp + 20],
                s[stmp + 4] s_addc_u32 s[stmp + 7],
                s[stmp + 21],
                s[stmp + 5] v_mov_b32 v[vtmp - 1],
                s[stmp + 6] v_mov_b32 v[vtmp],
                s[stmp + 7] s_add_u32 s[stmp + 6],
                s[stmp + 18],
                s[stmp + 4] v_mul_f32 v[vtmp + 7],
                0 + MIO_BN_NHW_DIV,
                v[vtmp + 5] s_addc_u32 s[stmp + 7],
                s[stmp + 19],
                s[stmp + 5] v_mov_b32 v[vtmp + 5],
                s[stmp + 6] v_mov_b32 v[vtmp + 6],
                s[stmp + 7] global_load_dword v[vtmp + 3],
                v [vtmp - 1:vtmp],
                off global_load_dword v[vtmp + 11],
                v [vtmp + 5:vtmp + 6],
                off v_cvt_f32_f64 v[vtmp + 1],
                s [stmp + 24:stmp + 25] s_add_u32 s[stmp + 6],
                s[stmp + 16],
                s[stmp + 4] s_addc_u32 s[stmp + 7],
                s[stmp + 17],
                s[stmp + 5] s_add_u32 s[stmp + 4],
                s[stmp + 12],
                s[stmp + 4] v_mul_f32 v[vtmp + 10],
                v[vtmp + 7],
                v[vtmp + 1] v_mov_b32 v[vtmp + 8],
                s[stmp + 7] v_mov_b32 v[vtmp + 7],
                s[stmp + 6] s_addc_u32 s[stmp + 5],
                s[stmp + 13],
                s[stmp + 5] v_sub_f32 v[vtmp + 9],
                1.0,
                v[vtmp + 1] global_store_dword v [vtmp + 7:vtmp + 8],
                v[vtmp + 2],
                off v_mov_b32 v[vtmp + 8],
                s[stmp + 5] v_mov_b32 v[vtmp + 7],
                s[stmp + 4] s_waitcnt vmcnt(2) v_fmac_f32 v[vtmp + 3],
                -v[vtmp + 1],
                v[vtmp + 3] v_fmac_f32 v[vtmp + 3],
                v[vtmp + 2],
                v[vtmp + 1] s_waitcnt vmcnt(1) v_fmac_f32 v[vtmp + 10],
                v[vtmp + 9],
                v[vtmp + 11] global_store_dword v [vtmp + 5:vtmp + 6],
                v[vtmp + 10],
                off global_store_dword v [vtmp - 1:vtmp],
                v[vtmp + 3],
                off global_store_dword v [vtmp + 7:vtmp + 8],
                v[vtmp + 4],
                off s_endpgm

                    //.section	.rodata,#alloc
                    .rodata.p2align
                6.amdhsa_kernel miopenGcnAsmBNFwdTrainSpatial
                    //.amdhsa_group_segment_fixed_size 136
                    .amdhsa_group_segment_fixed_size.AUTO_LDS_BYTE_SIZE
                    .amdhsa_private_segment_fixed_size 212.amdhsa_user_sgpr_private_segment_buffer
                1.amdhsa_user_sgpr_dispatch_ptr 0.amdhsa_user_sgpr_queue_ptr
                0.amdhsa_user_sgpr_kernarg_segment_ptr 1.amdhsa_user_sgpr_dispatch_id
                0.amdhsa_user_sgpr_flat_scratch_init 1.amdhsa_user_sgpr_private_segment_size
                0.amdhsa_system_sgpr_private_segment_wavefront_offset
                1.amdhsa_system_sgpr_workgroup_id_x 1.amdhsa_system_sgpr_workgroup_info
                0.amdhsa_system_vgpr_workitem_id 0;
.amdhsa_next_free_vgpr 13;
.amdhsa_next_free_sgpr 30.amdhsa_next_free_vgpr.AUTO_VGPR_COUNT
    .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr.amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0.amdhsa_float_denorm_mode_32 3.amdhsa_float_denorm_mode_16_64 3
    .amdhsa_dx10_clamp 1.amdhsa_ieee_mode 1.amdhsa_fp16_overflow 0.end_amdhsa_kernel.text
    .Lfunc_end0 :.size miopenGcnAsmBNFwdTrainSpatial,
    .Lfunc_end0 -
        miopenGcnAsmBNFwdTrainSpatial

            .amdgpu_metadata-- -
        amdhsa.kernels
        : -.args : -.access : read_only.address_space
                              : global.is_const
                                : true.is_restrict
                                  : true.name : in.offset : 0.size : 8.type_name
                                                : 'float*'.value_kind : global_buffer.value_type
                                                                        : f32 -
            .address_space : global.is_restrict
                             : true.name : out.offset : 8.size : 8.type_name
                                           : 'float*'.value_kind : global_buffer.value_type : f32 -
            .access : read_only.address_space
                      : constant.is_const
                        : true.is_restrict
                          : true.name : scale.offset : 16.size : 8.type_name
                                        : 'float*'.value_kind : global_buffer.value_type : f32 -
            .access : read_only.address_space
                      : constant.is_const
                        : true.is_restrict
                          : true.name : bias.offset : 24.size : 8.type_name
                                        : 'float*'.value_kind : global_buffer.value_type : f32 -
            .name : INHW.offset : 32.size : 4.type_name : float.value_kind : by_value.value_type
                                                                             : f32 -
            .name : expAvgFactor.offset : 40.size : 8.type_name : double.value_kind
                                                                  : by_value.value_type : f64 -
            .address_space : global.is_restrict
                             : true.name
                               : resultRunningMean.offset : 48.size : 8.type_name
                                 : 'float*'.value_kind : global_buffer.value_type : f32 -
            .address_space : global.is_restrict
                             : true.name
                               : resultRunningVariance.offset : 56.size : 8.type_name
                                 : 'float*'.value_kind : global_buffer.value_type : f32 -
            .name : epsilon.offset : 64.size : 8.type_name : double.value_kind : by_value.value_type
                                                                                 : f64 -
            .address_space : global.is_restrict
                             : true.name : resultSaveMean.offset : 72.size : 8.type_name
                                           : 'float*'.value_kind : global_buffer.value_type : f32 -
            .address_space : global.is_restrict
                             : true.name
                               : resultSaveInvVariance.offset : 80.size : 8.type_name
                                 : 'float*'.value_kind : global_buffer.value_type : f32 -
            .offset : 88.size : 8.value_kind : hidden_global_offset_x.value_type : i64 -
            .offset : 96.size : 8.value_kind : hidden_global_offset_y.value_type : i64 -
            .offset : 104.size : 8.value_kind : hidden_global_offset_z.value_type : i64 -
            .address_space : global.offset : 112.size : 8.value_kind : hidden_none.value_type : i8 -
            .address_space : global.offset : 120.size : 8.value_kind : hidden_none.value_type : i8 -
            .address_space : global.offset : 128.size : 8.value_kind : hidden_none.value_type : i8 -
            .address_space : global.offset : 136.size : 8.value_kind
                             : hidden_multigrid_sync_arg.value_type
                               : i8.group_segment_fixed_size : 136.kernarg_segment_align : 8
            .kernarg_segment_size : 144.language : OpenCL C.language_version : -1 -
        2.max_flat_workgroup_size : 1024.name
        : miopenGcnAsmBNFwdTrainSpatial.private_segment_fixed_size : 212.reqd_workgroup_size
          : -1024 -
        1 -
        1.sgpr_count : 36.sgpr_spill_count : 0.symbol : miopenGcnAsmBNFwdTrainSpatial.kd
            .vgpr_count : 13.vgpr_spill_count : 0.wavefront_size : 64 amdhsa.version : -1 -
        0 ...

            .end_amdgpu_metadata
