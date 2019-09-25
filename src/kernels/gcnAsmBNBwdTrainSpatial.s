/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
	.hsa_code_object_version 2,1

	.hsa_code_object_isa 

	.text
	.amdgpu_hsa_kernel gcnAsmBNBwdTrainSpatial

.include "gpr_alloc.inc"
.include "common.inc"
.include "inst_wrappers.inc"

// kernarg layout:
kernarg = 4
in_desc = 0
.set x_in_ptr_off, 0x0
.set dy_in_ptr_off, 0x8
.set dx_out_ptr_off, 0x10
.set bnScale_ptr_off, 0x18
.set dscale_ptr_off, 0x20
.set dbias_ptr_off, 0x28
.if (MIO_BN_USESAVED == 0)
    .set epsilon_off, 0x30
    .set inhw_off, 0x38
.elseif (MIO_BN_USESAVED == 1) 
    .set SavedMean_off, 0x30
    .set SavedInvVariance_off, 0x38
    .set inhw_off, 0x40
.endif

.set bn_bwd_lds_mask, 0x1C

madmix_instructions_available = 0
fmamix_instructions_available = 0
.if (.option.machine_version_major == 9)
    .if(.option.machine_version_stepping > 2)
        fmamix_instructions_available = 1
    .else
        madmix_instructions_available = 1
    .endif
.endif

.GPR_ALLOC_BEGIN

    //.SGPR_ALLOC_FROM 4
    .SGPR_ALLOC_FROM 0
    .SGPR_ALLOC stmp,8 
    .SGPR_ALLOC soffset_in, 2  //8
    .SGPR_ALLOC soffset_dy_in, 2 //10
    .SGPR_ALLOC soffset_dscale, 2 //12 
    .SGPR_ALLOC soffset_dbias, 2  //14
    .SGPR_ALLOC soffset_dx_out, 2 //16
    .SGPR_ALLOC soffset_inhw //18
    .SGPR_ALLOC stmp8 //19 
    .SGPR_ALLOC stmp9 //20
    .SGPR_ALLOC stmp10 //21
    .SGPR_RESERVE_XNACK

    .VGPR_ALLOC_FROM 0
    .VGPR_ALLOC tid
    .VGPR_ALLOC vtmp1, 2
    .VGPR_ALLOC v_db //v3 
    .VGPR_ALLOC v_ds //v4
    .VGPR_ALLOC qtmp2, 4 //v5-v8
    .VGPR_ALLOC qtmp3, 4 //v9-v12
    .VGPR_ALLOC qtmp4, 4 //13-v16
    .VGPR_ALLOC qtmp5, 4 //v17-v20

    //.LDS_ALLOC_FROM 0
    //.LDS_ALLOC accums_lds, 10

.GPR_ALLOC_END

gcnAsmBNBwdTrainSpatial:

	.amd_kernel_code_t
		kernel_code_entry_byte_offset = 256
		granulated_workitem_vgpr_count = .AUTO_VGPR_GRANULATED_COUNT
		granulated_wavefront_sgpr_count = .AUTO_SGPR_GRANULATED_COUNT 
		//float_mode = 240
		float_mode = 192
		user_sgpr_count = 6
		enable_sgpr_workgroup_id_x = 1
		enable_sgpr_private_segment_buffer = 1
		enable_sgpr_kernarg_segment_ptr = 1
		private_element_size = 1
		is_ptr64 = 1
		workgroup_group_segment_byte_size = 44
		kernarg_segment_byte_size = 120
		wavefront_sgpr_count = .AUTO_SGPR_COUNT 
		workitem_vgpr_count = .AUTO_VGPR_COUNT
		kernarg_segment_alignment = 4
		group_segment_alignment = 4
		private_segment_alignment = 4
	.end_amd_kernel_code_t

  // s[kernarg:kernarg+1] - kernel arg base address...
  // V0 - work item id...
  // s8: group ID 
	s_load_dwordx2 s[soffset_in:soffset_in+1], s[kernarg:kernarg+1], 0x0 + x_in_ptr_off                  
	s_load_dwordx2 s[soffset_dy_in:soffset_dy_in+1], s[kernarg:kernarg+1], 0x0 + dy_in_ptr_off
	s_mov_b32 s[stmp+7], 0                                     
  // set an equal to zero executive mask for first thread (==0)
	v_cmp_eq_u32 s[stmp:stmp+1], 0, v[tid]                      
  // save current exec mask in s[stmp+2:stmp+3] which is not equal to zero      
	s_and_saveexec_b64 s[stmp+2:stmp+3], s[stmp:stmp+1]                   
	s_cbranch_execz skip_bnScale_update                               

    //.SGPR_ALLOC soffset_dscale, 2 //12 
    //.SGPR_ALLOC soffset_dbias, 2  //14
    // .SGPR_ALLOC soffset_dx_out, 2 //16
    //.SGPR_ALLOC soffset_inhw //18

        stemp1 = soffset_dx_out //16
        stemp2 = soffset_dscale //12
        stemp3 = soffset_inhw //18
	s_load_dwordx2 s[stemp1:stemp1+1], s[kernarg:kernarg+1], 0x0 + bnScale_ptr_off
	s_load_dwordx4 s[stemp2:stemp2+3], s[kernarg:kernarg+1], 0x0 + SavedMean_off
  // shift grpid by 2 for adding to memory offset 
	s_lshl_b64 s[stemp3:stemp3+1], s[stmp+6:stmp+7], 2                      
	v_mov_b32 v[vtmp1], 0                                 
	s_waitcnt lgkmcnt(0)                                
  // shift grpid by 2 for adding to byte memory offset 
  // (bnScale + grpid)
	s_add_u32 s[stemp1], s[stemp1], s[stemp3]                             
	s_addc_u32 s[stemp1+1], s[stemp1+1], s[stemp3+1]                            
	s_add_u32 s[stemp2], s[stemp2], s[stemp3]                             
	s_addc_u32 s[stemp2+1], s[stemp2+1], s[stemp3+1]                            
	s_add_u32 s[stemp2+2], s[stemp2+2], s[stemp3]                             
	s_addc_u32 s[stemp2+3], s[stemp2+3], s[stemp3+1]                            
  // *(bnScale + grpid)
  // *(savedMean + grpid)
  // *(savedInvVariance + grpid)
	s_load_dword s[stemp1], s[stemp1:stemp1+1], 0x0                     
	s_load_dword s[stemp2], s[stemp2:stemp2+1], 0x0                     
	s_load_dword s[stemp2+1], s[stemp2+2:stemp2+3], 0x0                     
	s_waitcnt lgkmcnt(0)                                
	v_mov_b32 v[vtmp1+1], s[stemp1]                               
	v_mov_b32 v[v_db], s[stemp2]                               
  // LDS memory: lcl_scale = *(bnScale + grpid);
	ds_write2_b32 v[vtmp1], v[v_db], v[vtmp1+1] offset0:1 offset1:2        
	v_mov_b32 v[vtmp1+1], s[stemp2+1]                       
	ds_write_b32 v[vtmp1], v[vtmp1+1]                                 
skip_bnScale_update:
	s_or_b64 exec, exec, s[stmp+2:stmp+3]
	s_waitcnt lgkmcnt(0)                                
	s_barrier                                           
	v_mov_b32 v[v_db], 0                                 
	s_load_dwordx2 s[soffset_dx_out:soffset_dx_out+1], s[kernarg:kernarg+1], 0x0 + dx_out_ptr_off
	s_load_dwordx2 s[soffset_dscale:soffset_dscale+1], s[kernarg:kernarg+1], 0x0 + dscale_ptr_off
	s_load_dwordx2 s[soffset_dbias:soffset_dbias+1], s[kernarg:kernarg+1], 0x0 + dbias_ptr_off
	s_load_dword s[soffset_inhw], s[kernarg:kernarg+1], 0x0 + inhw_off
	ds_read2_b32 v[vtmp1:vtmp1+1], v[v_db] offset1:1                   
 // compute channel id(cidx) = grpid * MIO_BN_HW 
	s_movk_i32 s[stmp+2], 0+MIO_BN_HW                                 
	s_mul_i32 s[stemp3+1], s[stmp+6], s[stmp+2]                               
	v_cmp_gt_u32 s[stmp+2:stmp+3], s[stmp+2], v[tid]                     
	v_mov_b32 v[v_ds], 0                                 
	s_and_saveexec_b64 s[stmp9:stmp10], s[stmp+2:stmp+3]                 
	s_cbranch_execz skip_delta_update                               

	_v_add_nc_u32 v[qtmp2], s[stmp8], v[tid]                           
	v_mov_b32 v[v_db], 0                                 
	v_mov_b32 v[qtmp2+1], 0                                 
	v_mov_b32 v[v_ds], 0                                 
compute_delta_values:
	_v_add_nc_u32 v[qtmp2+2], v[qtmp2], v[qtmp2+1]                            
	v_mov_b32 v[qtmp2+3], 0                                 
	v_lshlrev_b64 v[qtmp3:qtmp3+1], 1, v[qtmp2+2:qtmp2+3]                    
	v_mov_b32 v[qtmp3+3], s[soffset_dy_in+1]                              
	_v_add_co_u32 v[qtmp3+2], s[stmp+4:stmp+5], s[soffset_dy_in], v[qtmp3]               
	_v_add_nc_u32 v[qtmp2+2], 0+MIO_BN_CHW, v[qtmp2+2]                       
	_v_add_co_ci_u32 v[qtmp3+3], s[stmp+4:stmp+5], v[qtmp3+3], v[qtmp3+1], s[stmp+4:stmp+5]     
	v_mov_b32 v[qtmp4], s[soffset_in+1]                               
	_v_add_co_u32 v[qtmp3], s[stmp+4:stmp+5], s[soffset_in], v[qtmp3]                 
	v_lshlrev_b64 v[qtmp2+2:qtmp2+3], 1, v[qtmp2+2:qtmp2+3]                     
	_v_add_co_ci_u32 v[qtmp3+1], s[stmp+4:stmp+5], v[qtmp4], v[qtmp3+1], s[stmp+4:stmp+5]     
	v_mov_b32 v[qtmp4+1], s[soffset_dy_in+1]                              
	_v_add_co_u32 v[qtmp4], s[stmp+4:stmp+5], s[soffset_dy_in], v[qtmp2+2]               
	_v_add_co_ci_u32 v[qtmp4+1], s[stmp+4:stmp+5], v[qtmp4+1], v[qtmp2+3], s[stmp+4:stmp+5]      
	v_mov_b32 v[qtmp4+2], s[soffset_in+1]                               
	_v_add_co_u32 v[qtmp2+2], s[stmp+4:stmp+5], s[soffset_in], v[qtmp2+2]                 
	flat_load_ushort v[qtmp3+2], v[qtmp3+2:qtmp3+3]               
	_v_add_co_ci_u32 v[qtmp2+3], s[stmp+4:stmp+5], v[qtmp4+2], v[qtmp2+3], s[stmp+4:stmp+5]       
	flat_load_ushort v[qtmp3], v[qtmp3:qtmp3+1]               
	flat_load_ushort v[qtmp3+1], v[qtmp4:qtmp4+1]               
	flat_load_ushort v[qtmp2+2], v[qtmp2+2:qtmp2+3]                 
	_v_add_nc_u32 v[qtmp2+1], 0+MIO_BN_CHW*2, v[qtmp2+1]                       
	v_cmp_eq_u32 vcc, 0+MIO_BN_NCHW, v[qtmp2+1]                 
	s_and_b64 vcc, exec, vcc                            
	s_waitcnt vmcnt(3)                                  
	v_cvt_f32_f16 v[qtmp2+3], v[qtmp3+2]                           
	s_waitcnt vmcnt(2)                                  
	v_cvt_f32_f16 v[qtmp3], v[qtmp3]                            
	s_waitcnt vmcnt(1)                                  
	v_cvt_f32_f16 v[qtmp3+3], v[qtmp3+1]                          
	s_waitcnt vmcnt(0)                                  
	v_cvt_f32_f16 v[qtmp2+2], v[qtmp2+2]                            
	v_add_f32 v[v_db], v[v_db], v[qtmp2+3]                            
	s_waitcnt lgkmcnt(0)                                
	v_sub_f32 v[qtmp2+3], v[qtmp3], v[vtmp1+1]                            
	v_mul_f32 v[qtmp2+3], v[vtmp1], v[qtmp2+3]                            
	v_sub_f32 v[qtmp2+2], v[qtmp2+2], v[vtmp1+1]                            
        .if(fmamix_instructions_available)
	    v_fma_mix_f32 v[v_ds], v[qtmp2+3], v[qtmp3+2], v[v_ds] op_sel_hi:[0,1,0]     
        .else
            v_cvt_f32_f16 v[qtmp3+2], v[qtmp3+2]   
	    v_fma_f32 v[v_ds], v[qtmp2+3], v[qtmp3+2], v[v_ds] 
        .endif
	v_mul_f32 v[qtmp2+2], v[vtmp1], v[qtmp2+2]                            
	v_add_f32 v[v_db], v[v_db], v[qtmp3+3]                           
        .if(fmamix_instructions_available)
	    v_fma_mix_f32 v[v_ds], v[qtmp2+2], v[qtmp3+1], v[v_ds] op_sel_hi:[0,1,0]     
        .else
            v_cvt_f32_f16 v[qtmp3+1], v[qtmp3+1]   
	    v_fma_f32 v[v_ds], v[qtmp2+2], v[qtmp3+1], v[v_ds] 
        .endif
	s_cbranch_vccz compute_delta_values                                
skip_delta_update:
	s_or_b64 exec, exec, s[stmp9:stmp10]
	s_waitcnt lgkmcnt(0)                                
	s_barrier                                           
 // DPP interleaved reduction...          
	v_and_b32 v[qtmp2], 63, v[tid]                            
	v_cmp_eq_u32 vcc, 63, v[qtmp2]                        
	s_nop 4                                             
	v_add_f32_dpp v[v_ds], v[v_ds], v[v_ds]  row_shr:1 bound_ctrl:0    
	v_add_f32_dpp v[v_db], v[v_db], v[v_db]  row_shr:1 bound_ctrl:0    
	s_nop 0                                             
	v_add_f32_dpp v[v_ds], v[v_ds], v[v_ds]  row_shr:2 bound_ctrl:0    
	v_add_f32_dpp v[v_db], v[v_db], v[v_db]  row_shr:2 bound_ctrl:0    
	s_nop 0                                             
	v_add_f32_dpp v[v_ds], v[v_ds], v[v_ds]  row_shr:4 bank_mask:0xe   
	v_add_f32_dpp v[v_db], v[v_db], v[v_db]  row_shr:4 bank_mask:0xe   
	s_nop 0                                             
	v_add_f32_dpp v[v_ds], v[v_ds], v[v_ds]  row_shr:8 bank_mask:0xc   
	v_add_f32_dpp v[v_db], v[v_db], v[v_db]  row_shr:8 bank_mask:0xc   
	s_nop 0                                             
	v_add_f32_dpp v[v_ds], v[v_ds], v[v_ds]  row_bcast:15 row_mask:0xa 
	v_add_f32_dpp v[v_db], v[v_db], v[v_db]  row_bcast:15 row_mask:0xa 
	s_nop 0                                             
	v_add_f32_dpp v[v_ds], v[v_ds], v[v_ds]  row_bcast:31 row_mask:0xc 
	v_add_f32_dpp v[v_db], v[v_db], v[v_db]  row_bcast:31 row_mask:0xc 
	s_nop 1                                             
	s_and_saveexec_b64 s[stmp+4:stmp+5], vcc                      
	v_lshrrev_b32 v[qtmp2], 4, v[tid]                         
	v_and_b32 v[qtmp2], 0x0+bn_bwd_lds_mask, v[qtmp2]                            
	ds_write2_b32 v[qtmp2], v[v_db], v[v_ds] offset0:3 offset1:0+MIO_BN_LDSGCN_SIZE+3        
	s_or_b64 exec, exec, s[stmp+4:stmp+5]                         
	s_waitcnt lgkmcnt(0)                                
	s_barrier                                           
	v_mov_b32 v[v_db], 0                                 
	v_mov_b32 v[qtmp2], 0                                 
	v_mov_b32 v[v_ds], 0                                 
bn_ldsgcn_size_loop:
	ds_read2_b32 v[qtmp2+1:qtmp2+2], v[qtmp2] offset0:3 offset1:0+MIO_BN_LDSGCN_SIZE+3         
	_v_add_nc_u32 v[qtmp2], 4, v[qtmp2]                             
	v_cmp_eq_u32 vcc, 0+MIO_BN_LDSGCN_SIZE*4, v[qtmp2]                        
	s_and_b64 vcc, exec, vcc                            
	s_waitcnt lgkmcnt(0)                                
	v_add_f32 v[v_ds], v[v_ds], v[qtmp2+2]                            
	v_add_f32 v[v_db], v[v_db], v[qtmp2+1]                            
	s_cbranch_vccz bn_ldsgcn_size_loop                                
	s_barrier                                           
	s_and_saveexec_b64 s[stmp+4:stmp+5], s[stmp+2:stmp+3]                   
	s_cbranch_execz skip_normalization                              

        .GPR_REUSE tid, vtmp6

	v_mov_b32 v[qtmp2], 0                                 
	ds_read_b32 v[qtmp2+2], v[qtmp2] offset:8                         
	_v_add_nc_u32 v[vtmp6], s[stmp8], v[vtmp6]                           
	v_xor_b32 v[qtmp2+1], 0x80000000, v[v_db]                    
	s_waitcnt lgkmcnt(0)                                
	v_mul_f32 v[qtmp2+2], v[vtmp1], v[qtmp2+2]                            
	v_mul_f32 v[qtmp2+2], s[soffset_inhw], v[qtmp2+2]                           
	s_mov_b32 s[soffset_inhw], 0+MIO_BN_NHW_FLOAT                           
apply_normalization:
	_v_add_nc_u32 v[qtmp2+3], v[vtmp6], v[qtmp2]                            
	v_mov_b32 v[qtmp3], 0                                 
	v_lshlrev_b64 v[qtmp3+1:qtmp3+2], 1, v[qtmp2+3:qtmp3]                   
	v_mov_b32 v[qtmp4], s[soffset_dy_in+1]                              
	_v_add_co_u32 v[qtmp3+3], vcc, s[soffset_dy_in], v[qtmp3+1]                 
	_v_add_nc_u32 v[qtmp2+3], 0+MIO_BN_CHW, v[qtmp2+3]                       
	_v_add_co_ci_u32 v[qtmp4], vcc, v[qtmp4], v[qtmp3+2], vcc           
	v_lshlrev_b64 v[qtmp2+3:qtmp3], 1, v[qtmp2+3:qtmp3]                     
	v_mov_b32 v[qtmp4+2], s[soffset_in+1]                               
	_v_add_co_u32 v[qtmp4+1], vcc, s[soffset_in], v[qtmp3+1]                  
	_v_add_co_ci_u32 v[qtmp4+2], vcc, v[qtmp4+2], v[qtmp3+2], vcc           
	v_mov_b32 v[qtmp5], s[soffset_dy_in+1]                              
	_v_add_co_u32 v[qtmp4+3], vcc, s[soffset_dy_in], v[qtmp2+3]                  
	_v_add_co_ci_u32 v[qtmp5], vcc, v[qtmp5], v[qtmp3], vcc            
	_v_add_co_u32 v[qtmp5+1], vcc, s[soffset_in], v[qtmp2+3]                   
	v_mov_b32 v[qtmp5+2], s[soffset_in+1]                               
	flat_load_ushort v[qtmp3+3], v[qtmp3+3:qtmp4]               
	_v_add_co_ci_u32 v[qtmp5+2], vcc, v[qtmp5+2], v[qtmp3], vcc            
	flat_load_ushort v[qtmp4], v[qtmp4+1:qtmp4+2]               
	flat_load_ushort v[qtmp4+1], v[qtmp4+3:qtmp5]               
	flat_load_ushort v[qtmp4+2], v[qtmp5+1:qtmp5+2]               
	v_mov_b32 v[qtmp5], s[soffset_dx_out+1]                              
	_v_add_co_u32 v[qtmp2+3], s[stmp+2:stmp+3], s[soffset_dx_out], v[qtmp2+3]                
	_v_add_co_ci_u32 v[qtmp3], s[stmp+2:stmp+3], v[qtmp5], v[qtmp3], s[stmp+2:stmp+3]       
	_v_add_nc_u32 v[qtmp2], 0+MIO_BN_CHW*2, v[qtmp2]                       
	v_cmp_eq_u32 vcc, 0+MIO_BN_NCHW, v[qtmp2]                 
	v_mov_b32 v[qtmp4+3], s[soffset_dx_out+1]                              
	_v_add_co_u32 v[qtmp3+1], s[stmp+2:stmp+3], s[soffset_dx_out], v[qtmp3+1]              
	s_and_b64 vcc, exec, vcc                            
	_v_add_co_ci_u32 v[qtmp3+2], s[stmp+2:stmp+3], v[qtmp4+3], v[qtmp3+2], s[stmp+2:stmp+3]     
	s_waitcnt vmcnt(3)                                  
        .if(fmamix_instructions_available)
	    v_fma_mix_f32 v[qtmp3+3], v[qtmp3+3], s[soffset_inhw], v[qtmp2+1] op_sel_hi:[1,0,0]   
        .else
            v_cvt_f32_f16 v[qtmp3+3], v[qtmp3+3]      
	    v_fma_f32 v[qtmp3+3], v[qtmp3+3], s[soffset_inhw], v[qtmp2+1]
        .endif
	s_waitcnt vmcnt(2)                                  
	v_cvt_f32_f16 v[qtmp4], v[qtmp4]                          
	s_waitcnt vmcnt(1)                                  
        .if(fmamix_instructions_available)
	    v_fma_mix_f32 v[qtmp4+1], v[qtmp4+1], s[soffset_inhw], v[qtmp2+1] op_sel_hi:[1,0,0]   
        .else
            v_cvt_f32_f16 v[qtmp4+1], v[qtmp4+1]      
	    v_fma_f32 v[qtmp4+1], v[qtmp4+1], s[soffset_inhw], v[qtmp2+1]
        .endif
	s_waitcnt vmcnt(0)                                  
	v_cvt_f32_f16 v[qtmp4+2], v[qtmp4+2]                          
	v_sub_f32 v[qtmp4], v[qtmp4], v[vtmp1+1]                          
	v_mul_f32 v[qtmp4], v[vtmp1], v[qtmp4]                          
	v_sub_f32 v[qtmp4+2], v[qtmp4+2], v[vtmp1+1]                          
	v_mul_f32 v[qtmp4], v[v_ds], v[qtmp4]                          
	v_mul_f32 v[qtmp4+2], v[vtmp1], v[qtmp4+2]                          
	v_sub_f32 v[qtmp3+3], v[qtmp3+3], v[qtmp4]                         
	v_mul_f32 v[qtmp4], v[v_ds], v[qtmp4+2]                          
	v_sub_f32 v[qtmp4], v[qtmp4+1], v[qtmp4]                         
	v_mul_f32 v[qtmp3+3], v[qtmp2+2], v[qtmp3+3]                          
	v_mul_f32 v[qtmp4], v[qtmp2+2], v[qtmp4]                          
	v_cvt_f16_f32 v[qtmp3+3], v[qtmp3+3]                          
	v_cvt_f16_f32 v[qtmp4], v[qtmp4]                          
	flat_store_short v[qtmp3+1:qtmp3+2], v[qtmp3+3]               
	flat_store_short v[qtmp2+3:qtmp3], v[qtmp4]                 
	s_cbranch_vccz apply_normalization                               
skip_normalization:
	s_or_b64 exec, exec, s[stmp+4:stmp+5]                         
	s_and_saveexec_b64 s[stmp+2:stmp+3], s[stmp:stmp+1]                   

	s_lshl_b64 s[stmp:stmp+1], s[stmp+6:stmp+7], 2                        
	s_add_u32 s[stmp+2], s[soffset_dbias], s[stmp]                               
	s_addc_u32 s[stmp+3], s[soffset_dbias+1], s[stmp+1]                              
	s_add_u32 s[stmp], s[soffset_dscale], s[stmp]                               
	s_addc_u32 s[stmp+1], s[soffset_dscale+1], s[stmp+1]                              
	v_mov_b32 v[qtmp2], s[stmp]                                
	v_mov_b32 v[qtmp2+1], s[stmp+1]                                
	v_mov_b32 v[qtmp3], s[stmp+2]                                
	v_mov_b32 v[qtmp3+1], s[stmp+3]                                
	flat_store_dword v[qtmp2:qtmp2+1], v[v_ds]                  
	flat_store_dword v[qtmp3:qtmp3+1], v[v_db]                  

	s_endpgm                                            



.Lfunc_end0:
    .size gcnAsmBNBwdTrainSpatial, .Lfunc_end0 - gcnAsmBNBwdTrainSpatial

ROCM_METADATA_VERSION = 4

.macro metadata wg_x, use_save_flag
  .if ROCM_METADATA_VERSION == 4
    .if (\use_save_flag == 0) 
      .amd_amdgpu_hsa_metadata
      { Version: [ 1, 0 ],
           Kernels:
           -  { Name: gcnAsmBNBwdTrainSpatial, SymbolName: 'gcnAsmBNBwdTrainSpatial@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
               Attrs:
                 { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                 CodeProps:
                 { KernargSegmentSize: 112, GroupSegmentFixedSize: 212, PrivateSegmentFixedSize: 132, KernargSegmentAlign: 8, WavefrontSize: 64, NumSGPRs: 32, NumVGPRs: 20, MaxFlatWorkGroupSize: 832}
                 Args:
                 - { Name: x_in    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: dy_in   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: dx_out   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: bnScale   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true}
                 - { Name: dscale    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true }
                 - { Name: dbias    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true }
                 - { Name: epsilon, Size: 8, Align: 8, ValueKind: ByValue, ValueType: F64, TypeName: 'double', AccQual: Default }
                 - { Name: INHW    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default }
                 //- { Name: HiddenGlobalOffsetX, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetY, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetZ, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
               }
      }
      .end_amd_amdgpu_hsa_metadata
    .elseif (\use_save_flag == 1) 
      .amd_amdgpu_hsa_metadata
      { Version: [ 1, 0 ],
           Kernels:
           -  { Name: gcnAsmBNBwdTrainSpatial, SymbolName: 'gcnAsmBNBwdTrainSpatial@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
               Attrs:
                 { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                 CodeProps:
                 { KernargSegmentSize: 112, GroupSegmentFixedSize: 212, PrivateSegmentFixedSize: 132, KernargSegmentAlign: 8, WavefrontSize: 64, NumSGPRs: 32, NumVGPRs: 20, MaxFlatWorkGroupSize: 832}
                 Args:
                 - { Name: x_in    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: dy_in   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: dx_out   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: bnScale   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true}
                 - { Name: dscale    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true }
                 - { Name: dbias    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true }
                 - { Name: savedMean    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
                 - { Name: savedInvVariance    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
                 - { Name: INHW    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default }
                 //- { Name: HiddenGlobalOffsetX, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetY, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetZ, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
               }
      }
      .end_amd_amdgpu_hsa_metadata
  .endif
  .endif
.endm


.altmacro
.macro metadata_wrapper x, y 
    metadata %\x, %\y
.endm
 
metadata_wrapper MIO_BN_GRP0, MIO_BN_USESAVED


