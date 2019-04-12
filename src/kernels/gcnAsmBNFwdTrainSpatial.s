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
  .amdgpu_hsa_kernel gcnAsmBNFwdTrainSpatial

// kernarg layout:
kernarg = 4
in_desc = 0
.set in_ptr_off, 0x0
.set out_ptr_off, 0x8
.set scale_ptr_off, 0x10
.set bias_ptr_off, 0x18
.set inhw_off, 0x20
.if (MIO_SAVE_MEAN_VARIANCE == 1) && (MIO_RUNNING_RESULT == 1) 
    .set expAvgFactor_off, 0x28
    .set resultRunningMean_off, 0x30
    .set resultRunningVariance_off, 0x38
    .set epsilon_off, 0x40
    .set resultSaveMean_off, 0x48
    .set resultSaveInvVariance_off, 0x50
.elseif (MIO_SAVE_MEAN_VARIANCE == 0) && (MIO_RUNNING_RESULT == 1) 
    .set expAvgFactor_off, 0x28
    .set resultRunningMean_off, 0x30
    .set resultRunningVariance_off, 0x38
    .set epsilon_off, 0x40
    .set resultSaveMean_off, 0x0
    .set resultSaveInvVariance_off, 0x0
.elseif (MIO_SAVE_MEAN_VARIANCE == 1) && (MIO_RUNNING_RESULT == 0) 
    .set expAvgFactor_off, 0x0
    .set resultRunningMean_off, 0x0
    .set resultRunningVariance_off, 0x0
    .set epsilon_off, 0x28
    .set resultSaveMean_off, 0x30
    .set resultSaveInvVariance_off, 0x38
.elseif (MIO_SAVE_MEAN_VARIANCE == 0) && (MIO_RUNNING_RESULT == 0) 
    .set expAvgFactor_off, 0x0
    .set resultRunningMean_off, 0x0
    .set resultRunningVariance_off, 0x0
    .set epsilon_off, 0x28
    .set resultSaveMean_off, 0x0
    .set resultSaveInvVariance_off, 0x0
.endif

madmix_instructions_available = 0
fmamix_instructions_available = 0
.if (.option.machine_version_major == 9)
    .if(.option.machine_version_stepping > 2)
        fmamix_instructions_available = 1
    .else
        madmix_instructions_available = 1
    .endif
.endif

gcnAsmBNFwdTrainSpatial:
  .amd_kernel_code_t
    kernel_code_entry_byte_offset = 256
    kernel_code_prefetch_byte_size = 0
    granulated_workitem_vgpr_count = 2
    granulated_wavefront_sgpr_count = 4
    priority = 0
    float_mode = 240
    priv = 0
    enable_dx10_clamp = 1
    debug_mode = 0
    enable_ieee_mode = 1
    enable_sgpr_private_segment_wave_byte_offset = 1
    user_sgpr_count = 8
    enable_trap_handler = 0
    enable_sgpr_workgroup_id_x = 1
    enable_sgpr_workgroup_id_y = 0
    enable_sgpr_workgroup_id_z = 0
    enable_sgpr_workgroup_info = 0
    enable_vgpr_workitem_id = 0
    enable_exception_msb = 0
    granulated_lds_size = 0
    enable_exception = 0
    enable_sgpr_private_segment_buffer = 1
    enable_sgpr_dispatch_ptr = 0
    enable_sgpr_queue_ptr = 0
    enable_sgpr_kernarg_segment_ptr = 1
    enable_sgpr_dispatch_id = 0
    enable_sgpr_private_segment_size = 0
    enable_sgpr_grid_workgroup_count_x = 0
    enable_sgpr_grid_workgroup_count_y = 0
    enable_sgpr_grid_workgroup_count_z = 0
    enable_ordered_append_gds = 0
    private_element_size = 1
    is_ptr64 = 1
    is_dynamic_callstack = 0
    is_debug_enabled = 0
    is_xnack_enabled = 0
    workitem_private_segment_byte_size = 132
    workgroup_group_segment_byte_size = 136
    gds_segment_byte_size = 0
    kernarg_segment_byte_size = 136
    workgroup_fbarrier_count = 0
    wavefront_sgpr_count = 40
    workitem_vgpr_count = 12
    reserved_vgpr_first = 0
    reserved_vgpr_count = 0
    reserved_sgpr_first = 0
    reserved_sgpr_count = 0
    debug_wavefront_private_segment_offset_sgpr = 0
    debug_private_segment_buffer_sgpr = 0
    kernarg_segment_alignment = 4
    group_segment_alignment = 4
    private_segment_alignment = 4
    wavefront_size = 6
    call_convention = -1
    runtime_loader_kernel_symbol = 0
  .end_amd_kernel_code_t

  // s[kernarg:kernarg+1] - kernel arg base address...
  // V0 - work item id...
  // s8: group ID 

  s_mov_b32 s12, s8                  
  s_mov_b32 s13, 0                  
  v_cmp_eq_u32 s[10:11], 0, v0   
  // set an executive mask for first thread
  // save current exec mask in s[6:7]      
  s_and_saveexec_b64 s[6:7], s[10:11]  
  // Branch if all lanes fail... 
  s_cbranch_execz skip_local_store
  // read scale ptr 
  s_load_dwordx2 s[14:15], s[kernarg:kernarg+1], 0x0 + scale_ptr_off
  // read bias ptr 
  s_load_dwordx2 s[16:17], s[kernarg:kernarg+1], 0x0 + bias_ptr_off
  // s[18:19] = grpid*4 for byte offset as registers are dwords..
  s_lshl_b64 s[18:19], s[12:13], 2  
  v_mov_b32 v1, 0              
  // wait for memory reads to finish...
  s_waitcnt lgkmcnt(0)     
  // Add group ID offset... 
  // s[14:15] = &scale[grpid]
  s_add_u32 s14, s14, s18   
  s_addc_u32 s15, s15, s19  
  // s[16:17] = &bias[grpid]
  s_add_u32 s16, s16, s18  
  s_addc_u32 s17, s17, s19  
  // read scale data
  // s8 = scale[grpid]
  s_load_dword s8, s[14:15], 0x0   
  // read bias data
  // s14 = bias[grpid]
  s_load_dword s14, s[16:17], 0x0   
  // wait for memory reads to finish...
  s_waitcnt lgkmcnt(0)     
  // store it in LDS memory..
  v_mov_b32 v2, s8     
  v_mov_b32 v3, s14   
  // lcl_scale = scale[grpid]
  // lcl_bias = bias[grpid]
  ds_write2_b32 v1, v3, v2 offset1:1  
skip_local_store:
  // restore an executive mask 
  s_or_b64 exec, exec, s[6:7]   
  s_waitcnt lgkmcnt(0)      
 // As multiple wavefronts are in a work-group (work group size is: 1024), 
 // the S_BARRIER instruction is used here to force each wavefront to wait until all other wavefronts reach this point...
  s_barrier    
 // load input arguments...
        .if(MIO_SAVE_MEAN_VARIANCE)
     s_load_dwordx2 s[22:23], s[kernarg:kernarg+1], 0x0 + resultSaveMean_off
     s_load_dwordx2 s[20:21], s[kernarg:kernarg+1], 0x0 + resultSaveInvVariance_off 
        .endif
        .if(MIO_RUNNING_RESULT)
     s_load_dwordx2 s[16:17], s[kernarg:kernarg+1], 0x0 + expAvgFactor_off
     s_load_dwordx2 s[18:19], s[kernarg:kernarg+1], 0x0 + resultRunningMean_off
     s_load_dwordx2 s[14:15], s[kernarg:kernarg+1], 0x0 + resultRunningVariance_off 
        .endif
  s_load_dwordx2 s[24:25], s[kernarg:kernarg+1], 0x0 + out_ptr_off
  s_load_dword s26, s[kernarg:kernarg+1], 0x0 + inhw_off
  s_load_dwordx2 s[28:29], s[kernarg:kernarg+1], 0x0 + epsilon_off 
        v_mov_b32 v1, 0+MIO_BN_HW //MIO_BN_HW 
  //v_mov_b32 v1, 0x100  
        // local id < MIO_BN_HW
  v_cmp_lt_u32 s[6:7], v0, v1   
 // compute channel id(cidx) = grpid * MIO_BN_HW (256)
        s_mul_i32 s8, s12, 0+MIO_BN_HW       
  v_mov_b32 v1, 0    
  v_mov_b32 v2, 0   
  s_and_saveexec_b64 s[30:31], s[6:7]   
 // Branch if all lanes fail... 
  s_cbranch_execz skip_mini_batch_compute
 // load input ptr 
  s_load_dwordx2 s[32:33], s[kernarg:kernarg+1], 0x0   
 // compute cidx + lid ....
  v_add_u32 v3, s8, v0    
 // set variance value to 0 (v1 = 0) 
  v_mov_b32 v1, 0     
 // set loop count register (v5) to 0.
  v_mov_b32 v5, 0    
 // set mean value to 0 (v2 = 0) 
  v_mov_b32 v2, 0   
mini_batch_compute:
  v_mov_b32 v4, 0    
 // v[3:4]: cidx + lid ....
  v_lshlrev_b64 v[8:9], 1, v[3:4]   
 // Add MIO_BN_CHW (0x20000:131072)
 // index = n * MIO_BN_CHW + cidx + lid 
 //   v_add_u32 v6, 0x8000, v3  
    v_add_u32 v6, 0+MIO_BN_CHW, v3  
  v_mov_b32 v7, v4       
  s_waitcnt lgkmcnt(0)  
 // move i/p ptr (s33)...
  v_mov_b32 v4, s33    
 // Adjust i/p ptr for first element...
  v_add_co_u32 v8, s[4:5], s32, v8   
  v_lshlrev_b64 v[6:7], 1, v[6:7]  
  v_addc_co_u32 v9, s[4:5], v4, v9, s[4:5]   
 // move i/p ptr (s33)...
  v_mov_b32 v10, s33        
 // Adjust i/p ptr for next element..
  v_add_co_u32 v6, s[4:5], s32, v6   
  v_addc_co_u32 v7, s[4:5], v10, v7, s[4:5]   
 // Read input values ...
  global_load_ushort v4, v[6:7], off     
  global_load_ushort v6, v[8:9], off   
 // increment loop count..
 // v11 = loop count + 4 .
  v_add_u32 v11, 4, v5         
 // s27 = loop count + 8 .
  v_add_u32 v5, 4, v5         
        v_cmp_eq_u32 vcc, 0+MIO_BN_N*2, v5  
//  v_cmp_eq_u32 vcc, 0x80, v5  
 // Add MIO_BN_CHW (0x10000)
 // index = n * MIO_BN_CHW + cidx + lid 
        v_add_u32 v3, 0+MIO_BN_CHW*2, v3       
  s_and_b64 vcc, exec, vcc    
 // v10 = 1 in IEEE-754 Floating Point 
        v_mov_b32 v10, 0x3f800000
  s_waitcnt vmcnt(0)       

        .if(fmamix_instructions_available)
 // compute variance += (*(in + index) * *(in + index))
      v_fma_mix_f32 v1, v6, v6, v1 op_sel_hi:[1,1,0]      

 // compute variance += (*(in + index) * *(in + index))
      v_fma_mix_f32 v1, v4, v4, v1 op_sel_hi:[1,1,0]    
 // compute mean += *(in + index)
            v_fma_mix_f32 v2, v6, v10, v2 op_sel_hi:[1,0,0]
 // v6: next 32 bit i/p and v4: 16 bit i/p...
 // v11 = offset.    
 // s9 = scalar index  
      buffer_store_short v6, v11, s[0:3], s9 offen      
 // compute mean += *(in + index)
            v_fma_mix_f32 v2, v4, v10, v2 op_sel_hi:[1,0,0]
 // store minibatch[n] = *(in + index +1) 
 // v6 = *(in + index + 1) 
      buffer_store_short v4, v11, s[0:3], s9 offen offset:2   
        .else
            v_cvt_f32_f16_e32 v7, v6    
 // v6: next 32 bit i/p and v4: 16 bit i/p...
 // v11 = offset.    
 // s9 = scalar index  
            buffer_store_short v6, v11, s[0:3], s9 offen  
            v_cvt_f32_f16_e32 v6, v4      
 // store minibatch[n] = *(in + index +1) 
 // v6 = *(in + index + 1) 
            buffer_store_short v4, v11, s[0:3], s9 offen offset:2  
 // compute mean += *(in + index)
            v_add_f32_e32 v2, v2, v7    
 // compute variance += (*(in + index) * *(in + index))
            v_fma_f32 v1, v7, v7, v1   
 // compute mean += *(in + index)
            v_add_f32_e32 v2, v2, v6  
 // compute variance += (*(in + index) * *(in + index))
            v_fma_f32 v1, v6, v6, v1
        .endif

  s_cbranch_vccz mini_batch_compute
skip_mini_batch_compute:
 // restore exec mask..                     
  s_or_b64 exec, exec, s[30:31]    
  s_waitcnt vmcnt(0) lgkmcnt(0)   
 // As multiple wavefronts are in a work-group (work group size is: 1024), 
 // the S_BARRIER instruction is used here to force each wavefront to wait until all other wavefronts reach this point...
  s_barrier         
 // (lid % 64)
  v_and_b32 v3, 63, v0   
  s_nop 4      
 // DPP interleaved reduction...          
 // v2 : mean and v1: variance...         
  v_add_f32_dpp v2, v2, v2  row_shr:1 bound_ctrl:0
  v_add_f32_dpp v1, v1, v1  row_shr:1 bound_ctrl:0
  s_nop 0 
  v_add_f32_dpp v2, v2, v2  row_shr:2 bound_ctrl:0
  v_add_f32_dpp v1, v1, v1  row_shr:2 bound_ctrl:0
  s_nop 0  
  v_add_f32_dpp v2, v2, v2  row_shr:4 bank_mask:0xe
  v_add_f32_dpp v1, v1, v1  row_shr:4 bank_mask:0xe
  s_nop 0    
  v_add_f32_dpp v2, v2, v2  row_shr:8 bank_mask:0xc
  v_add_f32_dpp v1, v1, v1  row_shr:8 bank_mask:0xc
  s_nop 0   
  v_add_f32_dpp v2, v2, v2  row_bcast:15 row_mask:0xa 
  v_add_f32_dpp v1, v1, v1  row_bcast:15 row_mask:0xa 
  s_nop 0   
  v_add_f32_dpp v2, v2, v2  row_bcast:31 row_mask:0xc 
  v_add_f32_dpp v1, v1, v1  row_bcast:31 row_mask:0xc 
  s_nop 0   
 // write mean and variance to LDS...
 // v2 : mean and v1: variance...         
  v_cmp_eq_u32 vcc, 63, v3   
  s_and_saveexec_b64 s[4:5], vcc     
  v_lshrrev_b32 v3, 4, v0     
  v_and_b32 v3, 0xffffffc, v3  
 // lcl_mean[ldsidx] = mean
 // lcl_variance[ldsidx] = variance
 // v2 : mean and v1: variance...         
 // v3 : LDS address 
  ds_write2_b32 v3, v1, v2 offset0:2 offset1:18   
  s_or_b64 exec, exec, s[4:5]  
 // Wait for count of outstanding LDS writes to complete..
  s_waitcnt lgkmcnt(0)   
 // Forcing each wavefront to wait here until all other wavefronts reach this point...
  s_barrier      
 // set mean to 0 
  v_mov_b32 v2, 0       
 // set LDS start address 
  v_mov_b32 v3, 0    
 // set variance to 0 
  v_mov_b32 v1, 0   
 // Add mean and variance values in for loop (MIO_BN_LDSGCN_SIZE)
bn_ldsgcn_size_loop:
 // Load mean and variance values to v[4:5] from LDS 
  ds_read2_b32 v[4:5], v3 offset0:2 offset1:18  
 // increment loop count...
  v_add_u32 v3, 4, v3   
 // compare loop count with MIO_BN_LDSGCN_SIZE...
        v_cmp_eq_u32 vcc, 0+MIO_BN_LDSGCN_SIZE*4, v3      
//  v_cmp_eq_u32 vcc, 64, v3      
  s_and_b64 vcc, exec, vcc     
  s_waitcnt lgkmcnt(0)    
 // variance += lcl_variance[i] 
  v_add_f32 v1, v1, v5  
 // mean += lcl_mean[i] 
  v_add_f32 v2, v2, v4  
 // if(VCC == 0) then continue in loop... 
  s_cbranch_vccz bn_ldsgcn_size_loop
 // Force each wavefront to wait here until all other wavefronts reach this point...
  s_barrier   
 // Convert from double precision float to single precision float...
 //     s[28:29]: epsilon
 //     v3: epsilon 
  v_cvt_f32_f64 v3, s[28:29] 
 // v1: variance *= INHW
 //     s26 = INHW
  v_mul_f32 v1, s26, v1     
 // v2: mean *= INHW
  v_mul_f32 v2, s26, v2    
 // variance = mad(-mean,mean,variance)
  //v_fmac_f32 v2, -v1, v1  
        v_fma_f32 v2, -v1, v1, v2
 // variance = variance + epsilon..
  v_add_f32 v5, v2, v3   
 // variance = Convert from single precision float to double precision float..
 // v[3:4] = variance in 64 bit 
  v_cvt_f64_f32 v[3:4], v5  
 // v[5:6] = 1.0/sqrt(variance) 
        v_rsq_f64 v[5:6], v[3:4]            
 // Convert from double precision float to single precision float...
 // v[5:6] = (1.0/sqrt(variance)... 
 // v[3] = v[5:6]... 
        v_cvt_f32_f64 v3, v[5:6]             
 // s6-s7 : thread/local IDs
 // s[26:27] = EXEC, EXEC = S[6:7] & EXEC...
  s_and_saveexec_b64 s[26:27], s[6:7]     
  s_cbranch_execz skip_bn_values_update
  v_mov_b32 v4, 0   
  ds_read2_b32 v[5:6], v4 offset1:1   
 // s8 = compute channel id(cidx) (cidx = grpid * MIO_BN_HW (256))
 // compute cidx + lid ....
  v_add_u32 v7, s8, v0   
bn_values_update:
  v_add_u32 v0, 4, v4   
 // buffer_load_dword v[reg], off, s[in_desc:in_desc+3], s[\s_offset] offset:0+imm_off
 // v0 = in[n]/minibatch[n]  ....
  //buffer_load_dword v0, v0, s[0:3], s9 offen   
  buffer_load_ushort v0, v0, s[0:3], s9 offen   
 // v[7:8] = 2 * (cidx + lid) as it uses byte offset...
  v_mov_b32 v8, 0     
  v_lshlrev_b64 v[8:9], 1, v[7:8]    
 // increment loop count 
  v_add_u32 v4, 2, v4    
 // compare with loop count ...
  //s_cmpk_eq_i32 s24, 0x80             
    v_cmp_eq_u32 vcc, 0+MIO_BN_N*2, v4     
 // s[24:25] - output ptr...
 // v[8:9] : Adjust output ptr with index offset 
  v_mov_b32 v10, s25    
  v_add_co_u32 v8, s[4:5], s24, v8    
 // index (v7) = 1*MIO_BN_CHW (131072) + cidx + lid ...
  v_add_u32 v7, 0+MIO_BN_CHW, v7  
  //v_add_u32 v7, 0x8000, v7  
  s_and_b64 vcc, exec, vcc  
 // v[8:9] : Adjust output ptr with index offset 
  v_addc_co_u32 v9, s[4:5], v10, v9, s[4:5]   
  s_waitcnt vmcnt(0)     
 // v0 : convert to 32 bit
  v_cvt_f32_f16 v0, v0                          
 // v1 : mean
 // v0 = (minibatch[n] - mean)
  v_sub_f32 v0, v0, v1   
 // v3 : invVariance
 // v0 = inhat = (minibatch[n] - mean)*invVariance
  v_mul_f32 v0, v3, v0  
  s_waitcnt lgkmcnt(0)  
 // pvscale = lcl_scale and pvbias = lcl_bias ...
 // v0 = mad (pvscale, inhat, pvbias)   
        .if(fmamix_instructions_available)
      v_fma_mixlo_f16 v0, v6, v0, v5   
        .else
            v_fma_f32 v0, v6, v0, v5 
            v_cvt_f16_f32_e32 v0, v0
        .endif
 // store in out[index] 
  global_store_short v[8:9], v0, off   
  s_cbranch_vccz bn_values_update
skip_bn_values_update:
 // reset exec mask     
  s_or_b64 exec, exec, s[26:27]    
  s_and_saveexec_b64 s[4:5], s[10:11]       
 // if(lid != 0) go to end of program...
 // Branch if all lanes fail... 
  s_cbranch_execz end_of_program
 // s[4:5] = grpid * 4
  s_lshl_b64 s[4:5], s[12:13], 2   
 // (resultSaveMean + grpid)
  s_add_u32 s6, s22, s4        
  s_addc_u32 s7, s23, s5      
  v_mov_b32 v4, s6           
  v_mov_b32 v5, s7          
 // (resultSaveInvVariance + grpid)
  s_add_u32 s6, s20, s4    
  s_addc_u32 s7, s21, s5  
  v_mov_b32 v6, s6   
  v_mov_b32 v7, s7  
 // (resultRunningMean + grpid)
  s_add_u32 s6, s18, s4      
  s_addc_u32 s7, s19, s5    
 // (resultRunningVariance + grpid)
  s_add_u32 s4, s14, s4    
        .if(MIO_SAVE_MEAN_VARIANCE)
 // v1 : mean
 // *(resultSaveMean + grpid) = mean;
     global_store_dword v[4:5], v1, off  
 // v3 : invVariance
 // *(resultSaveInvVariance + grpid) = invVariance;
     global_store_dword v[6:7], v3, off  
        .endif
  s_addc_u32 s5, s15, s5             
 // v[3:4] = resultRunningMean[grpid] 
 // v[5:6] = resultRunningVariance[grpid] 
  v_mov_b32 v3, s6            
  v_mov_b32 v6, s5           
  v_mov_b32 v4, s7       
  v_mov_b32 v5, s4      

        .if(MIO_RUNNING_RESULT)
 // v8 = pvt_runMean = *(resultRunningMean + grpid)
     global_load_dword v8, v[3:4], off    
 // v9 = *(resultRunningVariance + grpid)
     global_load_dword v9, v[5:6], off   
 // convert "expAvgFactor" double precision float to single precision float...
     v_cvt_f32_f64 v0, s[16:17]         
 // v2 :variance 
 // v2 :variance *= (MIO_BN_NHW/(MIO_BN_NHW-1.0))..
           v_mul_f32 v2, 0+MIO_BN_NHW_DIV, v2           
  //v_mul_f32 v2, 0x3f800400, v2     
 // v0: expAvgFactor
 // v7 = (1 - (_FLOAT)expAvgFactor)
     v_sub_f32 v7, 1.0, v0      
 // v2 = adjust * expAvgFactor 
     v_mul_f32 v2, v2, v0      
     s_waitcnt vmcnt(1)       
 // v8= pvt_newRunMean = mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean)
  //v_fmac_f32 v8, -v0, v8   
           v_fma_f32 v8, -v0, v8, v8
 // v0: expAvgFactor
 // v1: mean 
 // v8: pvt_newRunMean 
 // v8= mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean)
  //v_fmac_f32 v8, v1, v0   
           v_fma_f32 v8, v1, v0, v8
     s_waitcnt vmcnt(0)     
 // v7 : (1 - (_FLOAT)expAvgFactor)
 // v9 : *(resultRunningVariance + grpid)
 // v2 : adjust * expAvgFactor 
 // v2 = (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + grpid) + (_FLOAT)expAvgFactor * adjust;
  //v_fmac_f32 v2, v7, v9      
           v_fma_f32 v2, v7, v9, v2
 // resultRunningMean[grpid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean)
     global_store_dword v[3:4], v8, off      
 // resultRunningVariance[grpid] = (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + grpid) + (_FLOAT)expAvgFactor * adjust;
     global_store_dword v[5:6], v2, off     
        .endif
end_of_program:
  s_endpgm   

.Lfunc_end0:
    .size gcnAsmBNFwdTrainSpatial, .Lfunc_end0 - gcnAsmBNFwdTrainSpatial

ROCM_METADATA_VERSION = 4

.macro metadata wg_x, save_flag, result_running_flag
  .if ROCM_METADATA_VERSION == 4
    .if (\save_flag == 1) && (\result_running_flag == 1) 
      .amd_amdgpu_hsa_metadata
      { Version: [ 1, 0 ],
           Kernels:
           -  { Name: gcnAsmBNFwdTrainSpatial, SymbolName: 'gcnAsmBNFwdTrainSpatial@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
               Attrs:
                 { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                 CodeProps:
                 { KernargSegmentSize: 136, GroupSegmentFixedSize: 136, PrivateSegmentFixedSize: 132, KernargSegmentAlign: 8, WavefrontSize: 64, NumSGPRs: 40, NumVGPRs: 12, MaxFlatWorkGroupSize: 1024}
                 Args:
                 - { Name: in      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: out     , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: scale   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: bias    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: INHW    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default }
                 - { Name: expAvgFactor    , Size: 8, Align: 8, ValueKind: ByValue, ValueType: F64, TypeName: 'double', AccQual: Default }
                 - { Name: resultRunningMean,  Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: resultRunningVariance, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: epsilon, Size: 8, Align: 8, ValueKind: ByValue, ValueType: F64, TypeName: 'double', AccQual: Default }
                 - { Name: resultSaveMean, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: resultSaveInvVariance, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 //- { Name: HiddenGlobalOffsetX, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetY, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetZ, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
               }
      }
      .end_amd_amdgpu_hsa_metadata
    .elseif (\save_flag == 0) && (\result_running_flag == 1) 
      .amd_amdgpu_hsa_metadata
      { Version: [ 1, 0 ],
           Kernels:
           -  { Name: gcnAsmBNFwdTrainSpatial, SymbolName: 'gcnAsmBNFwdTrainSpatial@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
               Attrs:
                 { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                 CodeProps:
                 { KernargSegmentSize: 136, GroupSegmentFixedSize: 136, PrivateSegmentFixedSize: 132, KernargSegmentAlign: 8, WavefrontSize: 64, NumSGPRs: 40, NumVGPRs: 12, MaxFlatWorkGroupSize: 1024}
                 Args:
                 - { Name: in      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: out     , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: scale   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: bias    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: INHW    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default }
                 - { Name: expAvgFactor    , Size: 8, Align: 8, ValueKind: ByValue, ValueType: F64, TypeName: 'double', AccQual: Default }
                 - { Name: resultRunningMean,  Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: resultRunningVariance, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: epsilon, Size: 8, Align: 8, ValueKind: ByValue, ValueType: F64, TypeName: 'double', AccQual: Default }
                 //- { Name: HiddenGlobalOffsetX, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetY, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetZ, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
               }
      }
      .end_amd_amdgpu_hsa_metadata
    .elseif (\save_flag == 1) && (\result_running_flag == 0) 
      .amd_amdgpu_hsa_metadata
      { Version: [ 1, 0 ],
           Kernels:
           -  { Name: gcnAsmBNFwdTrainSpatial, SymbolName: 'gcnAsmBNFwdTrainSpatial@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
               Attrs:
                 { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                 CodeProps:
                 { KernargSegmentSize: 136, GroupSegmentFixedSize: 136, PrivateSegmentFixedSize: 132, KernargSegmentAlign: 8, WavefrontSize: 64, NumSGPRs: 40, NumVGPRs: 12, MaxFlatWorkGroupSize: 1024}
                 Args:
                 - { Name: in      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: out     , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: scale   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: bias    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: INHW    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default }
                 - { Name: epsilon, Size: 8, Align: 8, ValueKind: ByValue, ValueType: F64, TypeName: 'double', AccQual: Default }
                 - { Name: resultSaveMean, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: resultSaveInvVariance, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 //- { Name: HiddenGlobalOffsetX, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetY, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetZ, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
               }
      }
      .end_amd_amdgpu_hsa_metadata
    .elseif (\save_flag == 0) && (\result_running_flag == 0) 
      .amd_amdgpu_hsa_metadata
      { Version: [ 1, 0 ],
           Kernels:
           -  { Name: gcnAsmBNFwdTrainSpatial, SymbolName: 'gcnAsmBNFwdTrainSpatial@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
               Attrs:
                 { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                 CodeProps:
                 { KernargSegmentSize: 136, GroupSegmentFixedSize: 136, PrivateSegmentFixedSize: 132, KernargSegmentAlign: 8, WavefrontSize: 64, NumSGPRs: 40, NumVGPRs: 12, MaxFlatWorkGroupSize: 1024}
                 Args:
                 - { Name: in      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: out     , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F16, TypeName: 'half*', AddrSpaceQual: Global, AccQual: Default, IsRestrict: true}
                 - { Name: scale   , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: bias    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Constant, AccQual: ReadOnly, IsConst: true, IsRestrict: true}
                 - { Name: INHW    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default }
                 - { Name: epsilon, Size: 8, Align: 8, ValueKind: ByValue, ValueType: F64, TypeName: 'double', AccQual: Default }
                 //- { Name: HiddenGlobalOffsetX, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetY, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
                 //- { Name: HiddenGlobalOffsetZ, Size: 8, Align: 8, ValueKind: ByValue, ValueType: I64 }
               }
      }
      .end_amd_amdgpu_hsa_metadata
  .endif
  .endif
.endm

//.if MIO_BN_GRP0 == 832
//    metadata 832
//.else
//    metadata 1024
//.endif

.altmacro
.macro metadata_wrapper x, y, z
    metadata %\x, %\y, %\z
.endm
 
metadata_wrapper MIO_BN_GRP0, MIO_SAVE_MEAN_VARIANCE, MIO_RUNNING_RESULT

//metadata 1024
//metadata 832 
//metadata 0+MIO_BN_GRP0 

