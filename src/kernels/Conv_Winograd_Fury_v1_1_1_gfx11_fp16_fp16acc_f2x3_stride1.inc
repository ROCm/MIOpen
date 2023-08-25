/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

.macro _v_mad_u64_u32_gfx11 vdst:req, vsrc0:req, vsrc1:req, vsrc2:req
    .long  0xD6FE6A00 + (\vdst << 0)
    .long  0x00000000 + ((\vsrc0 + (1 << 8)) << 0) + ((\vsrc1 + (1 << 8)) << 9) + ((\vsrc2 + (1 << 8)) << 18)
.endm

.macro _s_mov_b32_4_gfx11 vdst:req
    .long  0xBE8000FF + (\vdst << 16)
    .long  0x00000004
.endm

.macro _v_cmp_f16_1_gfx11 vsrc:req
    .long  0xD404006A
    .long  0x0001FE00 + (\vsrc << 0)
    .long  0x00003C00
.endm

.macro _v_cmp_u16_vs_gfx11 vsrc:req, src:req, vhl, shl
    .long  0xD439006A + (\vhl << 12) + (\shl << 8)
    .long  0x00000100 + (\vsrc << 0) + (\src << 9)
.endm

.macro _v_pk_op_x16_gfx1x op:req vdst:req, vs0:req, src0:req, vs1:req, src1:req, neghi:req, neglo:req
    .long  0xCC000000 | (\op << 16) | (\vdst << 0) | (\neghi << 8)
    .long  0x18000000 | (\vs0 << (0+8)) | (\src0 << 0) | (\vs1 << (9+8)) | (\src1 << 9) | (\neglo << 29)
.endm

.macro _v_pk_op_x16_v2_gfx1x op:req vdst:req, vsrc0:req, vsrc1:req
    .long  0xCC000000 + (\op << 16) + (\vdst << 0)
    .long  0x10020000 + ((\vsrc0 + 128) << 0) + (\vsrc1 << 9)
.endm

.macro _v_pk_op_x16_v3_gfx1x op:req vdst:req, vsrc0:req, vsrc1:req
    .long  0xCC000800 + (\op << 16) + (\vdst << 0)
    .long  0x18000100 + (\vsrc0 << 0) + (\vsrc1 << 9)
.endm

.macro _v_pk_op_x16_v4_gfx1x op:req vdst:req, imm0:req, vsrc1:req
    .long  0xCC000000 + (\op << 16) + (\vdst << 0)
    .long  0x180200FF + (\vsrc1 << 9)
    .long  \imm0
.endm

.macro _v_pk_add_f16_0_gfx1x vdst:req, vsrc0:req
    .long  0xCC0F0000 + (\vdst << 0)
    .long  0x00010100 + (\vsrc0 << 0)
.endm

.macro _v_pk_mul_f16_05_gfx1x vdst:req, vsrc0:req
    .long  0xCC100000 + (\vdst << 0)
    .long  0x0801E100 + (\vsrc0 << 0)
.endm

s_version 0x2006                                           
s_set_inst_prefetch_distance 0x3                           
s_mov_b32 s0, 0                                            
s_load_b512 s[8:23], s[2:3], null                          
s_load_b512 s[24:39], s[2:3], 0x40                         
s_load_b256 s[40:47], s[2:3], 0x80                         
s_load_b128 s[48:51], s[2:3], 0xa0                         
s_load_b64 s[52:53], s[2:3], 0xb0                          
v_and_b32_e32 v9, 0xff, v0                                 
v_lshrrev_b32_e32 v10, 1, v9                               
v_and_b32_e32 v11, 1, v0                                   
v_add_nc_u32_e64 v6, v10, 32                               
v_bfi_b32 v2, 31, v9, v10                                  
v_bfe_u32 v4, v9, 5, 1                                     
v_bfi_b32 v2, 0xbf, v2, v6                                 
v_and_b32_e32 v8, 0x60, v10                                
v_lshlrev_b32_e32 v2, 4, v2                                
v_add_lshl_u32 v3, v10, v8, 4                              
v_mad_u32_u24 v2, v4, 0x800, v2                            
v_mad_u32_u16 v3, 0x1040, v11, v3                          
v_xor_b32_e32 v1, 0x600, v2                                
v_cmp_lt_u32_e64 vcc, v0, 0x100                            
s_cbranch_vccz 1624                                        
v_bfe_u32 v67, v0, 6, 1                                    
v_and_b32_e32 v65, 63, v0                                  
v_cmp_eq_u32_e64 vcc, v67, 1                               
v_cndmask_b32_e64 v67, 0, 0x400, vcc                       
v_cndmask_b32_e64 v84, 0, 0x100, vcc                       
v_lshl_add_u32 v66, v65, 2, v84                            
v_lshl_add_u32 v65, v65, 4, v67                            
s_mov_b32 s6, 4                                            
s_mov_b32 s7, 0                                            
v_mov_b32_e32 v48, -1                                      
s_mov_b32 s55, 0xbc00c000                                  
v_readfirstlane_b32 s69, v0                                
s_and_b32 null, 64, s69                                    
s_cmov_b32 s55, 0x3c00c000                                 
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
s_bitcmp1_b32 s40, 6                                       
s_cbranch_scc0 14                                          
s_load_b64 s[20:21], s[20:21], null                        
s_load_b64 s[22:23], s[22:23], null                        
s_load_b64 s[24:25], s[24:25], null                        
s_cmp_eq_u64 0, s[36:37]                                   
s_cbranch_scc1 2                                           
s_load_b64 s[36:37], s[36:37], null                        
s_cmp_eq_u64 0, s[38:39]                                   
s_cbranch_scc1 2                                           
s_load_b64 s[38:39], s[38:39], null                        
s_lshl_b32 s68, s14, 1                                     
s_lshl_b32 s70, s68, 3                                     
s_and_b32 null, 0x80, s69                                  
s_cselect_b32 s70, s70, 0                                  
s_cselect_b32 s5, 8, 0                                     
s_sub_u32 s5, s28, s5                                      
s_cmov_b32 s5, 0                                           
s_min_u32 s5, 8, s5                                        
s_mov_b32 s54, 0x19014000                                  
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
s_bitcmp1_b32 s40, 13                                      
s_cbranch_scc0 14                                          
s_add_u32 s20, s20, s44                                    
s_addc_u32 s21, s21, s45                                   
s_add_u32 s22, s22, s46                                    
s_addc_u32 s23, s23, s47                                   
s_add_u32 s24, s24, s48                                    
s_addc_u32 s25, s25, s49                                   
s_cmp_eq_u64 0, s[36:37]                                   
s_cselect_b64 s[50:51], 0, s[50:51]                        
s_add_u32 s36, s36, s50                                    
s_addc_u32 s37, s37, s51                                   
s_cmp_eq_u64 0, s[38:39]                                   
s_cselect_b64 s[52:53], 0, s[52:53]                        
s_add_u32 s38, s38, s52                                    
s_addc_u32 s39, s39, s53                                   
s_and_b32 s21, s21, 0xffff                                 
s_add_u32 s21, s21, 0x20000                                
s_add_u32 s20, s20, s70                                    
s_addc_u32 s21, s21, 0                                     
s_mov_b64 s[56:57], s[20:21]                               
s_mov_b32 s58, 0x80000000                                  
s_mov_b32 s59, 0                                           
s_getpc_b64 s[66:67]                                       
v_cmp_lt_u32_e64 vcc, v0, 0x80                             
s_cmp_gt_u32 vcc_lo, 0                                     
s_mov_b32 s69, 0x1334                                      
s_cmov_b32 s69, 0xe48                                      
s_mov_b32 s70, 0x158c                                      
s_cmov_b32 s70, 0x10a0                                     
s_add_u32 s64, s66, s69                                    
s_addc_u32 s65, s67, 0                                     
s_add_u32 s66, s66, s70                                    
s_addc_u32 s67, s67, 0                                     
v_mov_b32_e32 v177, 0x38003800                             
v_readfirstlane_b32 s69, v0                                
v_bfe_u32 v176, v0, 2, 3                                   
v_xor_b32_dpp v175, v0, v0 quad_perm:[0,0,3,1] row_mask:0xf bank_mask:0xf
v_lshlrev_b32_e32 v176, 1, v176                            
v_mad_i32_i24 v67, v175, s34, s35                          
v_mad_u32_u24 v67, s32, v176, v67                          
v_lshl_add_u32 v84, s33, 0, v67                            
v_lshl_add_u32 v101, s33, 1, v67                           
v_cmp_lt_u32_e64 vcc, v176, s28                            
v_cndmask_b32_e32 v67, 0x80000000, v67, vcc                
v_cndmask_b32_e32 v84, 0x80000000, v84, vcc                
v_cndmask_b32_e32 v101, 0x80000000, v101, vcc              
v_cmp_lt_u32_e64 vcc, v175, s30                            
v_cndmask_b32_e32 v67, 0x80000000, v67, vcc                
v_cndmask_b32_e32 v84, 0x80000000, v84, vcc                
v_cndmask_b32_e32 v101, 0x80000000, v101, vcc              
v_add_nc_u16 v176, v176, 1                                 
v_cmp_lt_u32_e64 vcc, 1, s29                               
v_cndmask_b32_e32 v84, 0x80000000, v84, vcc                
v_cmp_lt_u32_e64 vcc, 2, s29                               
v_cndmask_b32_e32 v101, 0x80000000, v101, vcc              
v_add_nc_u32_e64 v102, v67, s32                            
v_add_nc_u32_e64 v103, v84, s32                            
v_add_nc_u32_e64 v136, v101, s32                           
v_cmp_lt_u32_e64 vcc, v176, s28                            
v_cndmask_b32_e32 v102, 0x80000000, v102, vcc              
v_cndmask_b32_e32 v103, 0x80000000, v103, vcc              
v_cndmask_b32_e32 v136, 0x80000000, v136, vcc              
v_xor_b32_dpp v180, v0, v0 quad_perm:[0,0,3,1] row_mask:0xf bank_mask:0xf
v_bfe_u32 v181, v0, 2, 2                                   
v_mul_u32_u24_e64 v153, v180, 16                           
v_bfe_u32 v180, v0, 4, 2                                   
v_mad_u32_u16 v153, v181, 4, v153                          
v_bfe_u32 v175, v0, 1, 3                                   
v_bfe_u32 v181, v0, 6, 2                                   
v_mul_u32_u24_e64 v170, v175, 0x60                         
v_mad_u32_u16 v153, v180, 48, v153                         
v_bfe_u32 v175, v0, 6, 2                                   
v_mad_u32_u16 v153, v181, 0xc0, v153                       
v_mad_u32_u16 v170, v175, 0x300, v170                      
s_mov_b64 s[72:73], s[24:25]                               
s_and_b32 s73, s73, 0xffff                                 
s_add_u32 s73, s73, 0x20000                                
s_mov_b32 s74, 0x80000000                                  
s_mov_b32 s75, 0x11014000                                  
s_lshl_b32 s76, s31, 1                                     
s_lshr_b32 s69, s69, 5                                     
s_mul_i32 s77, s76, s69                                    
s_mul_i32 s70, s76, 7                                      
s_mul_i32 s71, s76, 1                                      
s_sub_u32 s69, s27, s69                                    
s_sub_u32 s69, s69, 1                                      
s_cselect_b32 s75, 0, s75                                  
s_add_u32 s72, s72, s77                                    
s_addc_u32 s73, s73, 0                                     
s_mov_b32 exec_hi, 0                                       
buffer_load_d16_b16 v49, v67, s[72:75], 0 idxen            
buffer_load_d16_b16 v50, v84, s[72:75], 0 idxen            
buffer_load_d16_b16 v52, v101, s[72:75], 0 idxen           
buffer_load_d16_hi_b16 v49, v102, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v50, v103, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v52, v136, s[72:75], 0 idxen        
s_sub_u32 s69, s69, 1                                      
s_cselect_b32 s75, 0, s75                                  
s_add_u32 s72, s72, s71                                    
s_addc_u32 s73, s73, 0                                     
buffer_load_d16_b16 v57, v67, s[72:75], 0 idxen            
buffer_load_d16_b16 v58, v84, s[72:75], 0 idxen            
buffer_load_d16_b16 v60, v101, s[72:75], 0 idxen           
buffer_load_d16_hi_b16 v57, v102, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v58, v103, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v60, v136, s[72:75], 0 idxen        
s_sub_u32 s69, s69, 7                                      
s_cselect_b32 s75, 0, s75                                  
s_add_u32 s72, s72, s70                                    
s_addc_u32 s73, s73, 0                                     
s_mov_b32 exec_hi, -1                                      
s_mov_b32 exec_lo, 0                                       
buffer_load_d16_b16 v49, v67, s[72:75], 0 idxen            
buffer_load_d16_b16 v50, v84, s[72:75], 0 idxen            
buffer_load_d16_b16 v52, v101, s[72:75], 0 idxen           
buffer_load_d16_hi_b16 v49, v102, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v50, v103, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v52, v136, s[72:75], 0 idxen        
s_sub_u32 s69, s69, 1                                      
s_cselect_b32 s75, 0, s75                                  
s_add_u32 s72, s72, s71                                    
s_addc_u32 s73, s73, 0                                     
buffer_load_d16_b16 v57, v67, s[72:75], 0 idxen            
buffer_load_d16_b16 v58, v84, s[72:75], 0 idxen            
buffer_load_d16_b16 v60, v101, s[72:75], 0 idxen           
buffer_load_d16_hi_b16 v57, v102, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v58, v103, s[72:75], 0 idxen        
buffer_load_d16_hi_b16 v60, v136, s[72:75], 0 idxen        
s_sub_u32 s69, s69, 7                                      
s_cselect_b32 s75, 0, s75                                  
s_add_u32 s72, s72, s70                                    
s_addc_u32 s73, s73, 0                                     
s_mov_b32 exec_lo, -1                                      
s_waitcnt vmcnt(0)                                         
_v_pk_op_x16_gfx1x 16, 51, 1, 52, 1, 177, 0, 0 // op16_mul_f
v_pk_fma_f16 v51, v50, v177, v51 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fmac_f16 v51, v49, v177                               
_v_pk_op_x16_gfx1x 15, 50, 1, 50, 1, 51, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 59, 1, 60, 1, 177, 0, 0 // op16_mul_f
v_pk_fma_f16 v59, v58, v177, v59 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fmac_f16 v59, v57, v177                               
_v_pk_op_x16_gfx1x 15, 58, 1, 58, 1, 59, 0, 0 // op16_add_f
ds_store_b32 v153, v49 offset:43264                        
ds_store_b32 v153, v50 offset:44032                        
ds_store_b32 v153, v52 offset:44800                        
ds_store_b32 v153, v51 offset:45568                        
ds_store_b32 v153, v57 offset:46336                        
ds_store_b32 v153, v58 offset:47104                        
ds_store_b32 v153, v60 offset:47872                        
ds_store_b32 v153, v59 offset:48640                        
s_waitcnt lgkmcnt(0)                                       
s_barrier                                                  
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[49:52], v170 offset:43264                   
ds_load_b128 v[53:56], v170 offset:43312                   
ds_load_b128 v[68:71], v170 offset:43280                   
ds_load_b128 v[72:75], v170 offset:43328                   
ds_load_b128 v[104:107], v170 offset:43296                 
ds_load_b128 v[108:111], v170 offset:43344                 
ds_load_b128 v[57:60], v170 offset:46336                   
ds_load_b128 v[61:64], v170 offset:46384                   
ds_load_b128 v[76:79], v170 offset:46352                   
ds_load_b128 v[80:83], v170 offset:46400                   
ds_load_b128 v[112:115], v170 offset:46368                 
ds_load_b128 v[116:119], v170 offset:46416                 
s_mov_b32 exec_hi, -1                                      
s_waitcnt lgkmcnt(0)                                       
_v_pk_op_x16_gfx1x 16, 85, 1, 104, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 86, 1, 105, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 87, 1, 106, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 88, 1, 107, 1, 177, 0, 0 // op16_mul_f
v_pk_fma_f16 v85, v68, v177, v85 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v86, v69, v177, v86 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v87, v70, v177, v87 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v88, v71, v177, v88 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fmac_f16 v85, v49, v177                               
v_pk_fmac_f16 v86, v50, v177                               
v_pk_fmac_f16 v87, v51, v177                               
v_pk_fmac_f16 v88, v52, v177                               
_v_pk_op_x16_gfx1x 15, 68, 1, 68, 1, 85, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 69, 1, 69, 1, 86, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 70, 1, 70, 1, 87, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 71, 1, 71, 1, 88, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 89, 1, 108, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 90, 1, 109, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 91, 1, 110, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 92, 1, 111, 1, 177, 0, 0 // op16_mul_f
v_pk_fma_f16 v89, v72, v177, v89 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v90, v73, v177, v90 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v91, v74, v177, v91 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v92, v75, v177, v92 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fmac_f16 v89, v53, v177                               
v_pk_fmac_f16 v90, v54, v177                               
v_pk_fmac_f16 v91, v55, v177                               
v_pk_fmac_f16 v92, v56, v177                               
_v_pk_op_x16_gfx1x 15, 72, 1, 72, 1, 89, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 73, 1, 73, 1, 90, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 74, 1, 74, 1, 91, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 75, 1, 75, 1, 92, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 93, 1, 112, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 94, 1, 113, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 95, 1, 114, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 96, 1, 115, 1, 177, 0, 0 // op16_mul_f
v_pk_fma_f16 v93, v76, v177, v93 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v94, v77, v177, v94 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v95, v78, v177, v95 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v96, v79, v177, v96 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fmac_f16 v93, v57, v177                               
v_pk_fmac_f16 v94, v58, v177                               
v_pk_fmac_f16 v95, v59, v177                               
v_pk_fmac_f16 v96, v60, v177                               
_v_pk_op_x16_gfx1x 15, 76, 1, 76, 1, 93, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 77, 1, 77, 1, 94, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 78, 1, 78, 1, 95, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 79, 1, 79, 1, 96, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 97, 1, 116, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 98, 1, 117, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 99, 1, 118, 1, 177, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 100, 1, 119, 1, 177, 0, 0 // op16_mul_f
v_pk_fma_f16 v97, v80, v177, v97 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v98, v81, v177, v98 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v99, v82, v177, v99 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fma_f16 v100, v83, v177, v100 neg_lo:[0,1,0] neg_hi:[0,1,0]
v_pk_fmac_f16 v97, v61, v177                               
v_pk_fmac_f16 v98, v62, v177                               
v_pk_fmac_f16 v99, v63, v177                               
v_pk_fmac_f16 v100, v64, v177                              
_v_pk_op_x16_gfx1x 15, 80, 1, 80, 1, 97, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 81, 1, 81, 1, 98, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 82, 1, 82, 1, 99, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 83, 1, 83, 1, 100, 0, 0 // op16_add_f
_s_mov_b32_4_gfx11 60
s_setprio 2                                                
s_waitcnt vmcnt(0)                                         
_v_pk_op_x16_gfx1x 15, 36, 1, 4, 1, 5, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 37, 1, 8, 1, 9, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 38, 1, 12, 1, 13, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 39, 1, 16, 1, 17, 1, 1 // op16_add_f
v_pk_fma_f16 v40, v4, s55, v6 op_sel:[0,1,0]               
v_pk_fma_f16 v41, v8, s55, v10 op_sel:[0,1,0]              
v_pk_fma_f16 v42, v12, s55, v14 op_sel:[0,1,0]             
v_pk_fma_f16 v43, v16, s55, v18 op_sel:[0,1,0]             
buffer_load_d16_b16 v5, v171, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v4, v173, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v13, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v12, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v5, v171, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v4, v173, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v13, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v12, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v9, v171, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v8, v173, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v17, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v16, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v9, v171, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v8, v173, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v17, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v16, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[64:65]                            
s_branch 3701                                              
_s_mov_b32_4_gfx11 60
s_setprio 2                                                
v_pk_fma_f16 v36, v6, s55, v40 op_sel_hi:[1,0,1]           
v_pk_fma_f16 v37, v10, s55, v41 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v38, v14, s55, v42 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v39, v18, s55, v43 op_sel_hi:[1,0,1]          
_v_pk_op_x16_gfx1x 15, 40, 1, 6, 1, 7, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 41, 1, 10, 1, 11, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 42, 1, 14, 1, 15, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 43, 1, 18, 1, 19, 1, 1 // op16_add_f
buffer_load_d16_b16 v6, v171, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v7, v173, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v14, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v15, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v6, v171, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v7, v173, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v14, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v15, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v10, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v11, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v18, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v19, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v10, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v11, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v18, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v19, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[66:67]                            
s_branch 3632                                              
_s_mov_b32_4_gfx11 60
s_setprio 2                                                
s_waitcnt vmcnt(0)                                         
_v_pk_op_x16_gfx1x 15, 36, 1, 23, 1, 22, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 37, 1, 27, 1, 26, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 38, 1, 31, 1, 30, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 39, 1, 35, 1, 34, 1, 1 // op16_add_f
v_pk_fma_f16 v40, v23, s55, v20 op_sel:[0,1,0]             
v_pk_fma_f16 v41, v27, s55, v24 op_sel:[0,1,0]             
v_pk_fma_f16 v42, v31, s55, v28 op_sel:[0,1,0]             
v_pk_fma_f16 v43, v35, s55, v32 op_sel:[0,1,0]             
buffer_load_d16_b16 v22, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v23, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v30, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v31, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v22, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v23, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v30, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v31, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v26, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v27, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v34, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v35, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v26, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v27, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v34, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v35, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[64:65]                            
s_branch 3562                                              
_s_mov_b32_4_gfx11 60
s_setprio 2                                                
v_pk_fma_f16 v36, v20, s55, v40 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v37, v24, s55, v41 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v38, v28, s55, v42 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v39, v32, s55, v43 op_sel_hi:[1,0,1]          
_v_pk_op_x16_gfx1x 15, 40, 1, 20, 1, 21, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 41, 1, 24, 1, 25, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 42, 1, 28, 1, 29, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 43, 1, 32, 1, 33, 1, 1 // op16_add_f
buffer_load_d16_b16 v20, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v21, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v28, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v29, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v20, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v21, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v28, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v29, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v24, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v25, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v32, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v33, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v24, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v25, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v32, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v33, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[66:67]                            
s_branch 3493                                              
_s_mov_b32_4_gfx11 60
s_setprio 2                                                
s_waitcnt vmcnt(0)                                         
_v_pk_op_x16_gfx1x 15, 36, 1, 4, 1, 5, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 37, 1, 8, 1, 9, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 38, 1, 7, 1, 6, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 39, 1, 11, 1, 10, 1, 1 // op16_add_f
v_pk_fma_f16 v40, v4, s55, v13 op_sel:[0,1,0]              
v_pk_fma_f16 v41, v8, s55, v17 op_sel:[0,1,0]              
v_pk_fma_f16 v42, v7, s55, v14 op_sel:[0,1,0]              
v_pk_fma_f16 v43, v11, s55, v18 op_sel:[0,1,0]             
buffer_load_d16_b16 v5, v171, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v4, v173, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v6, v172, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v7, v174, s[56:59], 0 idxen glc dlc    
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v5, v171, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v4, v173, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v6, v172, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v7, v174, s[56:59], 0 idxen glc dlc 
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v9, v171, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v8, v173, s[56:59], 0 idxen glc dlc    
buffer_load_d16_b16 v10, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v11, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v9, v171, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v8, v173, s[56:59], 0 idxen glc dlc 
buffer_load_d16_hi_b16 v10, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v11, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[64:65]                            
s_branch 3423                                              
_s_mov_b32_4_gfx11 60
s_setprio 2                                                
v_pk_fma_f16 v36, v13, s55, v40 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v37, v17, s55, v41 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v38, v14, s55, v42 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v39, v18, s55, v43 op_sel_hi:[1,0,1]          
_v_pk_op_x16_gfx1x 15, 40, 1, 13, 1, 12, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 41, 1, 17, 1, 16, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 42, 1, 14, 1, 15, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 43, 1, 18, 1, 19, 1, 1 // op16_add_f
buffer_load_d16_b16 v13, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v12, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v14, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v15, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v13, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v12, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v14, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v15, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v17, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v16, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v18, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v19, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v17, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v16, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v18, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v19, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[66:67]                            
s_branch 3354                                              
_s_mov_b32_4_gfx11 60
s_setprio 2                                                
s_waitcnt vmcnt(0)                                         
_v_pk_op_x16_gfx1x 15, 36, 1, 23, 1, 22, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 37, 1, 27, 1, 26, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 38, 1, 21, 1, 20, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 39, 1, 25, 1, 24, 1, 1 // op16_add_f
v_pk_fma_f16 v40, v23, s55, v30 op_sel:[0,1,0]             
v_pk_fma_f16 v41, v27, s55, v34 op_sel:[0,1,0]             
v_pk_fma_f16 v42, v21, s55, v28 op_sel:[0,1,0]             
v_pk_fma_f16 v43, v25, s55, v32 op_sel:[0,1,0]             
buffer_load_d16_b16 v22, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v23, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v20, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v21, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v22, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v23, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v20, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v21, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v26, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v27, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v24, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v25, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v26, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v27, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v24, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v25, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[64:65]                            
s_branch 3284                                              
s_mov_b32 s60, 0xfffff754                                  
s_setprio 2                                                
v_pk_fma_f16 v36, v30, s55, v40 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v37, v34, s55, v41 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v38, v28, s55, v42 op_sel_hi:[1,0,1]          
v_pk_fma_f16 v39, v32, s55, v43 op_sel_hi:[1,0,1]          
_v_pk_op_x16_gfx1x 15, 40, 1, 30, 1, 31, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 41, 1, 34, 1, 35, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 42, 1, 28, 1, 29, 1, 1 // op16_add_f
_v_pk_op_x16_gfx1x 15, 43, 1, 32, 1, 33, 1, 1 // op16_add_f
buffer_load_d16_b16 v30, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v31, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v28, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v29, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v30, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v31, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v28, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v29, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_b16 v34, v171, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v35, v173, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v32, v172, s[56:59], 0 idxen glc dlc   
buffer_load_d16_b16 v33, v174, s[56:59], 0 idxen glc dlc   
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
buffer_load_d16_hi_b16 v34, v171, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v35, v173, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v32, v172, s[56:59], 0 idxen glc dlc
buffer_load_d16_hi_b16 v33, v174, s[56:59], 0 idxen glc dlc
s_sub_u32 s1, s1, 1                                        
s_cselect_b32 s59, 0, s59                                  
s_add_u32 s56, s56, s68                                    
s_addc_u32 s57, s57, 0                                     
s_swappc_b64 s[62:63], s[66:67]                            
s_branch 3215                                              
ds_store_b128 v1, v[154:157] offset:4160                   
ds_store_b128 v1, v[166:169] offset:16                     
s_setprio 1                                                
s_ashr_i32 s61, s60, 31                                    
s_add_u32 s62, s62, s60                                    
s_addc_u32 s63, s63, s61                                   
s_mov_b32 exec_hi, 0                                       
s_waitcnt lgkmcnt(0)                                       
s_barrier                                                  
_v_pk_op_x16_gfx1x 15, 120, 1, 120, 1, 137, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 121, 1, 138, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 122, 1, 139, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 123, 1, 140, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 124, 1, 141, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 125, 1, 142, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 126, 1, 143, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 127, 1, 144, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 145, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 146, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 147, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 148, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 149, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 150, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 151, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 152, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[154:157], v[49:56], v[120:127], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[154:157], v[57:64], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[166:169], v[104:111], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[166:169], v[112:119], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 120, 1, 128, 1, 137, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 129, 1, 138, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 130, 1, 139, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 131, 1, 140, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 132, 1, 141, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 133, 1, 142, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 134, 1, 143, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 135, 1, 144, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 137, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 138, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 139, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 140, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 141, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 142, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 143, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 144, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[158:161], v[68:75], v[120:127], 0 op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1                                      
v_wmma_f16_16x16x16_f16 v[158:161], v[76:83], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[162:165], v[85:92], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[162:165], v[93:100], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 158, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 159, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 160, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 161, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 158, 1, 166, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 159, 1, 167, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 160, 1, 168, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 161, 1, 169, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 162, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 163, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 164, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 165, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 166, 1, 162, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 167, 1, 163, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 168, 1, 164, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 169, 1, 165, 2, 2 // op16_add_f
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[120:123], v1 offset:24768                   
ds_load_b128 v[124:127], v1 offset:26816                   
ds_load_b128 v[128:131], v1 offset:28928                   
ds_load_b128 v[132:135], v1 offset:30976                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[36:39] offset:16512                    
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[137:140], v1 offset:24784                   
ds_load_b128 v[141:144], v1 offset:26832                   
ds_load_b128 v[145:148], v1 offset:28944                   
ds_load_b128 v[149:152], v1 offset:30992                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[40:43] offset:17024                    
s_waitcnt lgkmcnt(10)                                      
s_swappc_b64 s[62:63], s[62:63]                            
ds_store_b128 v2, v[154:157] offset:12416                  
ds_store_b128 v2, v[166:169] offset:8272                   
s_setprio 1                                                
s_ashr_i32 s61, s60, 31                                    
s_sub_u32 s6, s6, s7                                       
s_cselect_b64 s[60:61], 0, s[60:61]                        
s_add_u32 s62, s62, s60                                    
s_addc_u32 s63, s63, s61                                   
s_mov_b32 exec_hi, 0                                       
s_waitcnt lgkmcnt(0)                                       
s_barrier                                                  
v_add_co_u32 v48, vcc, v48, s26                            
_v_pk_op_x16_gfx1x 15, 120, 1, 120, 1, 137, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 121, 1, 138, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 122, 1, 139, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 123, 1, 140, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 124, 1, 141, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 125, 1, 142, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 126, 1, 143, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 127, 1, 144, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 145, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 146, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 147, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 148, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 149, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 150, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 151, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 152, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[154:157], v[49:56], v[120:127], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[154:157], v[57:64], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[166:169], v[104:111], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[166:169], v[112:119], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 120, 1, 128, 1, 137, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 129, 1, 138, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 130, 1, 139, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 131, 1, 140, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 132, 1, 141, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 133, 1, 142, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 134, 1, 143, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 135, 1, 144, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 137, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 138, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 139, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 140, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 141, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 142, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 143, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 144, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[158:161], v[68:75], v[120:127], 0 op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1                                      
v_wmma_f16_16x16x16_f16 v[158:161], v[76:83], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[162:165], v[85:92], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[162:165], v[93:100], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 158, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 159, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 160, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 161, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 158, 1, 166, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 159, 1, 167, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 160, 1, 168, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 161, 1, 169, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 162, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 163, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 164, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 165, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 166, 1, 162, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 167, 1, 163, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 168, 1, 164, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 169, 1, 165, 2, 2 // op16_add_f
s_bfe_u32 null, vcc_lo, 0x10000                            
s_cselect_b32 s54, s54, 0                                  
s_cselect_b32 s7, s7, 1                                    
ds_load_b128 v[171:174], v65 offset:33024                  
ds_load_b32 v48, v66 offset:35072                          
s_sub_u32 s1, s5, 1                                        
s_cselect_b32 s59, 0, s54                                  
s_mov_b64 s[56:57], s[20:21]                               
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[120:123], v2 offset:16512                   
ds_load_b128 v[124:127], v2 offset:18560                   
ds_load_b128 v[128:131], v2 offset:20672                   
ds_load_b128 v[132:135], v2 offset:22720                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[36:39] offset:24768                    
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[137:140], v2 offset:16528                   
ds_load_b128 v[141:144], v2 offset:18576                   
ds_load_b128 v[145:148], v2 offset:20688                   
ds_load_b128 v[149:152], v2 offset:22736                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[40:43] offset:25280                    
s_waitcnt lgkmcnt(10)                                      
s_swappc_b64 s[62:63], s[62:63]                            
ds_store_b128 v1, v[154:157] offset:4160                   
ds_store_b128 v1, v[166:169] offset:16                     
s_setprio 1                                                
s_ashr_i32 s61, s60, 31                                    
s_add_u32 s62, s62, s60                                    
s_addc_u32 s63, s63, s61                                   
s_mov_b32 exec_hi, 0                                       
s_waitcnt lgkmcnt(0)                                       
_v_pk_op_x16_gfx1x 15, 120, 1, 120, 1, 137, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 121, 1, 138, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 122, 1, 139, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 123, 1, 140, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 124, 1, 141, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 125, 1, 142, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 126, 1, 143, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 127, 1, 144, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 145, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 146, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 147, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 148, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 149, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 150, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 151, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 152, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[154:157], v[49:56], v[120:127], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[154:157], v[57:64], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[166:169], v[104:111], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[166:169], v[112:119], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 120, 1, 128, 1, 137, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 129, 1, 138, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 130, 1, 139, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 131, 1, 140, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 132, 1, 141, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 133, 1, 142, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 134, 1, 143, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 135, 1, 144, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 137, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 138, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 139, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 140, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 141, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 142, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 143, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 144, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[158:161], v[68:75], v[120:127], 0 op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1                                      
v_wmma_f16_16x16x16_f16 v[158:161], v[76:83], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[162:165], v[85:92], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[162:165], v[93:100], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 158, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 159, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 160, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 161, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 158, 1, 166, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 159, 1, 167, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 160, 1, 168, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 161, 1, 169, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 162, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 163, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 164, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 165, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 166, 1, 162, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 167, 1, 163, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 168, 1, 164, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 169, 1, 165, 2, 2 // op16_add_f
s_barrier                                                  
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[120:123], v1 offset:24768                   
ds_load_b128 v[124:127], v1 offset:26816                   
ds_load_b128 v[128:131], v1 offset:28928                   
ds_load_b128 v[132:135], v1 offset:30976                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[36:39] offset:16512                    
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[137:140], v1 offset:24784                   
ds_load_b128 v[141:144], v1 offset:26832                   
ds_load_b128 v[145:148], v1 offset:28944                   
ds_load_b128 v[149:152], v1 offset:30992                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[40:43] offset:17024                    
s_waitcnt lgkmcnt(10)                                      
s_swappc_b64 s[62:63], s[62:63]                            
ds_store_b128 v2, v[154:157] offset:12416                  
ds_store_b128 v2, v[166:169] offset:8272                   
s_setprio 1                                                
s_ashr_i32 s61, s60, 31                                    
s_sub_u32 s6, s6, s7                                       
s_cselect_b64 s[60:61], 0, s[60:61]                        
s_add_u32 s62, s62, s60                                    
s_addc_u32 s63, s63, s61                                   
s_mov_b32 exec_hi, 0                                       
s_waitcnt lgkmcnt(0)                                       
v_add_co_u32 v48, vcc, v48, s26                            
_v_pk_op_x16_gfx1x 15, 120, 1, 120, 1, 137, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 121, 1, 138, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 122, 1, 139, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 123, 1, 140, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 124, 1, 141, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 125, 1, 142, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 126, 1, 143, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 127, 1, 144, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 145, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 146, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 147, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 148, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 149, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 150, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 151, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 152, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[154:157], v[49:56], v[120:127], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[154:157], v[57:64], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[166:169], v[104:111], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[166:169], v[112:119], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 120, 1, 128, 1, 137, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 121, 1, 129, 1, 138, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 122, 1, 130, 1, 139, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 123, 1, 131, 1, 140, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 124, 1, 132, 1, 141, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 125, 1, 133, 1, 142, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 126, 1, 134, 1, 143, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 127, 1, 135, 1, 144, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 145, 1, 137, 1, 128, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 146, 1, 138, 1, 129, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 147, 1, 139, 1, 130, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 148, 1, 140, 1, 131, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 149, 1, 141, 1, 132, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 150, 1, 142, 1, 133, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 151, 1, 143, 1, 134, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 152, 1, 144, 1, 135, 2, 2 // op16_add_f
v_wmma_f16_16x16x16_f16 v[158:161], v[68:75], v[120:127], 0 op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1                                      
v_wmma_f16_16x16x16_f16 v[158:161], v[76:83], v[120:127], 0 op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16 v[162:165], v[85:92], v[145:152], 0 op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[162:165], v[93:100], v[145:152], 0 op_sel:[0,0,1]
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 158, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 159, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 160, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 161, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 158, 1, 166, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 159, 1, 167, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 160, 1, 168, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 161, 1, 169, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 154, 1, 154, 1, 162, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 155, 1, 155, 1, 163, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 156, 1, 156, 1, 164, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 157, 1, 157, 1, 165, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 166, 1, 166, 1, 162, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 167, 1, 167, 1, 163, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 168, 1, 168, 1, 164, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 169, 1, 169, 1, 165, 2, 2 // op16_add_f
s_barrier                                                  
s_bfe_u32 null, vcc_lo, 0x10000                            
s_cselect_b32 s54, s54, 0                                  
s_cselect_b32 s7, s7, 1                                    
ds_load_b128 v[171:174], v65 offset:33024                  
ds_load_b32 v48, v66 offset:35072                          
s_sub_u32 s1, s5, 1                                        
s_cselect_b32 s59, 0, s54                                  
s_mov_b64 s[56:57], s[20:21]                               
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[120:123], v2 offset:16512                   
ds_load_b128 v[124:127], v2 offset:18560                   
ds_load_b128 v[128:131], v2 offset:20672                   
ds_load_b128 v[132:135], v2 offset:22720                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[36:39] offset:24768                    
s_mov_b32 exec_hi, 0                                       
ds_load_b128 v[137:140], v2 offset:16528                   
ds_load_b128 v[141:144], v2 offset:18576                   
ds_load_b128 v[145:148], v2 offset:20688                   
ds_load_b128 v[149:152], v2 offset:22736                   
s_mov_b32 exec_hi, -1                                      
ds_store_b128 v3, v[40:43] offset:25280                    
s_waitcnt lgkmcnt(10)                                      
s_swappc_b64 s[62:63], s[62:63]                            
v_bfe_u32 v79, v0, 6, 1                                    
v_and_b32_e32 v18, 63, v0                                  
v_cmp_eq_u32_e64 vcc, v79, 1                               
v_cndmask_b32_e64 v81, 0, 0xa00, vcc                       
v_cndmask_b32_e64 v82, 0, 0x500, vcc                       
v_cndmask_b32_e64 v79, 0, 0x400, vcc                       
v_cndmask_b32_e64 v80, 0, 0x100, vcc                       
v_lshl_add_u32 v15, v18, 2, v82                            
v_lshl_add_u32 v14, v18, 3, v81                            
v_lshl_add_u32 v19, v18, 2, v80                            
v_lshl_add_u32 v18, v18, 4, v79                            
s_barrier                                                  
s_mov_b32 s70, 0xbc00c000                                  
s_mov_b32 s54, 0x10000                                     
s_mov_b32 s55, 0x30002                                     
s_mov_b32 s56, 0x10000                                     
v_readfirstlane_b32 s79, v0                                
s_and_b32 null, 64, s79                                    
s_cmov_b32 s70, 0x3c00c000                                 
s_cmov_b32 s54, 0x20003                                    
s_cmov_b32 s55, 1                                          
s_cmov_b32 s56, 1                                          
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
s_bitcmp1_b32 s40, 6                                       
s_cbranch_scc0 14                                          
s_load_b64 s[20:21], s[20:21], null                        
s_load_b64 s[22:23], s[22:23], null                        
s_load_b64 s[24:25], s[24:25], null                        
s_cmp_eq_u64 0, s[36:37]                                   
s_cbranch_scc1 2                                           
s_load_b64 s[36:37], s[36:37], null                        
s_cmp_eq_u64 0, s[38:39]                                   
s_cbranch_scc1 2                                           
s_load_b64 s[38:39], s[38:39], null                        
s_cmp_eq_u32 s8, 0x60                                      
s_cbranch_scc0 16                                          
s_mul_i32 s79, s4, 0xab                                    
s_lshr_b32 s79, s79, 10                                    
s_mul_i32 s85, s79, 6                                      
s_sub_u32 s85, s4, s85                                     
s_bfe_u32 s83, s79, 0x20000                                
s_bfe_u32 s84, s79, 0x10002                                
s_bfe_u32 s82, s79, 0x10003                                
s_mov_b32 s86, s85                                         
s_lshl1_add_u32 s86, s86, s84                              
s_lshl2_add_u32 s86, s86, s83                              
s_lshl1_add_u32 s86, s86, s82                              
s_mov_b32 s4, s86                                          
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
s_bitcmp1_b32 s40, 13                                      
s_cbranch_scc0 14                                          
s_add_u32 s20, s20, s44                                    
s_addc_u32 s21, s21, s45                                   
s_add_u32 s22, s22, s46                                    
s_addc_u32 s23, s23, s47                                   
s_add_u32 s24, s24, s48                                    
s_addc_u32 s25, s25, s49                                   
s_cmp_eq_u64 0, s[36:37]                                   
s_cselect_b64 s[50:51], 0, s[50:51]                        
s_add_u32 s36, s36, s50                                    
s_addc_u32 s37, s37, s51                                   
s_cmp_eq_u64 0, s[38:39]                                   
s_cselect_b64 s[52:53], 0, s[52:53]                        
s_add_u32 s38, s38, s52                                    
s_addc_u32 s39, s39, s53                                   
s_cmp_eq_u64 s[38:39], 0                                   
s_cselect_b32 s47, 0, 0x11014000                           
s_and_b32 s39, s39, 0xffff                                 
s_add_u32 s39, s39, 0x20000                                
s_mov_b64 s[44:45], s[38:39]                               
s_mov_b32 s46, 0x80000000                                  
v_and_b32_e64 v79, v0, 63                                  
v_lshlrev_b32_e32 v79, 1, v79                              
v_cmp_lt_u32_e64 vcc, v79, s27                             
v_add_nc_u32_e64 v80, v79, 1                               
v_cndmask_b32_e32 v79, 0x80000000, v79, vcc                
v_cmp_lt_u32_e64 vcc, v80, s27                             
v_cndmask_b32_e32 v80, 0x80000000, v80, vcc                
buffer_load_d16_b16 v81, v79, s[44:47], 0 idxen            
buffer_load_d16_hi_b16 v81, v80, s[44:47], 0 idxen         
s_waitcnt vmcnt(0)                                         
v_readlane_b32 s71, v81, 0                                 
v_readlane_b32 s72, v81, 1                                 
v_readlane_b32 s73, v81, 2                                 
v_readlane_b32 s74, v81, 3                                 
v_readlane_b32 s75, v81, 4                                 
v_readlane_b32 s76, v81, 5                                 
v_readlane_b32 s77, v81, 6                                 
v_readlane_b32 s78, v81, 7                                 
s_getpc_b64 s[44:45]                                       
s_and_b32 s41, s41, 0xff                                   
s_cmp_eq_u32 s41, 2                                        
s_cbranch_scc1 26                                          
s_cmp_eq_u32 s41, 0                                        
s_cselect_b32 s42, 1.0, s42                                
v_cvt_f16_f32_e32 v79, s42                                 
v_readfirstlane_b32 s42, v79                               
v_cvt_f16_f32_e32 v79, s43                                 
v_readfirstlane_b32 s43, v79                               
_v_cmp_f16_1_gfx11 42
s_pack_ll_b32_b16 s42, s42, s42                            
s_pack_ll_b32_b16 s43, s43, s43                            
s_cmp_eq_u32 s41, 3                                        
s_cbranch_scc1 9                                           
s_cbranch_vccnz 4                                          
s_add_u32 s80, s44, 0x3c4                                  
s_addc_u32 s81, s45, 0                                     
s_branch 11                                                
s_add_u32 s80, s44, 0xac4                                  
s_addc_u32 s81, s45, 0                                     
s_branch 7                                                 
s_add_u32 s80, s44, 0x1ac4                                 
s_addc_u32 s81, s45, 0                                     
s_branch 3                                                 
s_add_u32 s80, s44, 0x11c4                                 
s_addc_u32 s81, s45, 0                                     
s_and_b32 s23, s23, 0xffff                                 
s_add_u32 s23, s23, 0x20000                                
s_lshl_b32 s63, s19, 1                                     
s_mov_b64 s[64:65], s[22:23]                               
s_mov_b32 s66, 0x80000000                                  
s_mov_b32 s67, 0                                           
s_mov_b32 s68, 0                                           
s_mov_b32 s69, 0                                           
v_lshrrev_b32_e64 v82, 16, s9                              
v_bfi_b32 v83, 0xffff, s9, 0                               
v_and_b32_e32 v85, 1, v0                                   
v_bfe_u32 v91, v0, 6, 1                                    
v_and_b32_e32 v80, 63, v0                                  
v_mad_u32_u16 v86, 0x7c, s8, 0                             
v_mad_u32_u16 v91, 2, s4, v91                              
v_mad_u32_u16 v84, v82, v83, 0                             
v_cmp_eq_u32_e32 vcc, 0, v85                               
v_cndmask_b32_e32 v92, v84, v83, vcc                       
v_mad_u32_u16 v81, 62, v91, v80                            
v_cndmask_b32_e32 v81, v86, v81, vcc                       
v_clz_i32_u32_e32 v98, v92                                 
v_lshlrev_b32_e32 v99, v98, v92                            
v_and_b32_e32 v97, 0xffffff00, v99                         
v_cmp_eq_u32_e32 vcc, 0x80000000, v99                      
v_cvt_f32_u32_e32 v97, v97                                 
v_rcp_f32_e32 v93, v97                                     
v_sub_co_ci_u32_e32 v94, vcc, 32, v98, vcc                 
v_cvt_f32_ubyte0_e32 v98, v99                              
v_fma_f32 v97, v97, v93, -1.0                              
v_fma_f32 v97, v98, v93, v97                               
v_fmaak_f32 v97, v97, v93, 0x9f000000                      
v_mul_f32_e32 v97, 0x5f800000, v97                         
v_mov_b32_e32 v98, 0                                       
v_cvt_floor_i32_f32_e64 v97, -v97                          
v_lshl_add_u32 v93, v93, 9, v97                            
_v_mad_u64_u32_gfx11 98, 99, 93, 98
v_sub_co_ci_u32_e64 v93, vcc, v93, -1, vcc                 
v_mov_b32_dpp v96, v94 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v94, v94 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v95, v93 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v93, v93 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mul_hi_u32 v97, v81, v93                                 
v_add_co_u32 v79, vcc, v97, v81                            
v_add_co_ci_u32_e64 v97, vcc, 0, 0, vcc                    
v_cmp_eq_u32_e32 vcc, 32, v94                              
v_cndmask_b32_e32 v79, v79, v97, vcc                       
v_alignbit_b32 v79, v97, v79, v94                          
v_mul_hi_u32 v97, v81, v95                                 
v_add_co_u32 v4, vcc, v97, v81                             
v_add_co_ci_u32_e64 v97, vcc, 0, 0, vcc                    
v_cmp_eq_u32_e32 vcc, 32, v96                              
v_cndmask_b32_e32 v4, v4, v97, vcc                         
v_alignbit_b32 v4, v97, v4, v96                            
v_mad_u32_u16 v90, v79, v83, 0                             
v_mad_u32_u16 v89, v4, v82, 0                              
v_sub_nc_u32_e32 v90, v81, v90                             
v_sub_nc_u32_e32 v89, v79, v89                             
v_readlane_b32 s44, v90, 1                                 
v_sub_nc_u32_e32 v90, v90, v83                             
v_readlane_b32 s5, v89, 1                                  
v_sub_nc_u32_e32 v89, v89, v82                             
v_readlane_b32 s1, v4, 1                                   
v_sub_nc_u32_e64 v4, v4, s26                               
s_lshl_b32 s5, s5, 16                                      
s_and_b32 s44, s44, 0xffff                                 
s_add_u32 s5, s5, s44                                      
v_mov_b32_dpp v90, v90 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v89, v89 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v4, v4 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32 v90, vcc, v90, v85                            
v_cndmask_b32_e32 v88, 0, v83, vcc                         
v_add_co_ci_u32_e64 v89, vcc, v89, 0, vcc                  
v_cndmask_b32_e32 v87, 0, v82, vcc                         
v_add_co_ci_u32_e64 v4, vcc, v4, 0, vcc                    
v_min_u32_e64 v85, v80, 63                                 
v_sub_nc_u32_e32 v90, v90, v88                             
v_sub_nc_u32_e32 v89, v89, v87                             
v_cmp_eq_u32_e32 vcc, v80, v85                             
v_lshlrev_b32_e32 v5, 16, v89                              
v_bfi_b32 v5, 0xffff, v90, v5                              
v_add_nc_u32_e64 v100, v4, s26                             
v_med3_u32 v85, v80, 1, 62                                 
v_mul_lo_u32 v6, v100, s12                                 
v_mul_lo_u32 v11, v100, s17                                
s_mul_i32 s6, s1, s12                                      
s_mul_i32 s7, s1, s17                                      
v_cndmask_b32_e32 v6, 0x80000000, v6, vcc                  
v_cmp_eq_u32_e32 vcc, v80, v85                             
v_cndmask_b32_e32 v11, 0x80000000, v11, vcc                
v_cmp_ge_u32_e64 vcc, v100, s26                            
v_cndmask_b32_e64 v6, v6, 0x80000000, vcc                  
v_cndmask_b32_e64 v11, v11, 0x80000000, vcc                
s_mov_b32 s57, 4                                           
s_lshl_b32 s60, s57, 9                                     
s_lshl_b32 s61, s57, 8                                     
v_add_nc_u32_e32 v16, s60, v14                             
v_add_nc_u32_e32 v17, s61, v15                             
s_setpc_b64 s[80:81]                                       
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v23, v27, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 59, 1, 31, 1, 35, 0, 0 // op16_add_f
v_pk_fma_f16 v24, v28, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 60, 1, 32, 1, 36, 0, 0 // op16_add_f
v_pk_fma_f16 v25, v29, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 61, 1, 33, 1, 37, 0, 0 // op16_add_f
v_pk_fma_f16 v26, v30, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 62, 1, 34, 1, 38, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 59, 1, 59, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 60, 1, 60, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 61, 1, 61, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 62, 1, 62, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
v_add_co_u32 v85, vcc, v5, s5                              
v_pk_mad_u16 v87, v5, 0x20001, s11                         
v_pk_mad_u16 v88, v5, 0x20001, s16                         
_v_pk_op_x16_gfx1x 13, 90, 1, 85, 1, 5, 0, 0 // op16_min_u
v_cndmask_b32_e64 v79, 0, s12, vcc                         
v_cndmask_b32_e64 v80, 0, s17, vcc                         
v_mad_u32_u16 v7, v87, 1, v6                               
v_mad_u32_u16 v12, v88, 1, v11                             
v_add3_u32 v6, v6, s6, v79                                 
v_add3_u32 v11, v11, s7, v80                               
_v_pk_op_x16_gfx1x 11, 82, 1, 5, 1, 90, 0, 0 // op16_sub_u
v_add_co_ci_u32_e64 v4, s[58:59], v4, s1, vcc              
v_cndmask_b32_e64 v6, v6, 0x80000000, s[58:59]             
v_cndmask_b32_e64 v11, v11, 0x80000000, s[58:59]           
_v_cmp_u16_vs_gfx11 87, 10, 0, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
_v_cmp_u16_vs_gfx11 88, 15, 0, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_v2_gfx1x 6, 82, 15, 82 // op16_ashrrev_i
_v_pk_op_x16_v3_gfx1x 10, 81, 87, 55 // op16_add_u
_v_pk_op_x16_v3_gfx1x 10, 84, 87, 54 // op16_add_u
v_mad_u32_u16 v10, v81, s13, v7 op_sel:[1,0,0,0]           
v_mad_u32_u16 v8, v84, s13, v7 op_sel:[1,0,0,0]            
_v_pk_op_x16_v3_gfx1x 10, 86, 88, 56 // op16_add_u
_v_cmp_u16_vs_gfx11 81, 10, 1, 8
v_cndmask_b32_e32 v10, 0x80000000, v10, vcc                
_v_cmp_u16_vs_gfx11 84, 10, 1, 8
v_cndmask_b32_e32 v8, 0x80000000, v8, vcc                  
v_mad_u32_u16 v13, v86, s18, v12 op_sel:[1,0,0,0]          
v_mad_u32_u16 v9, v81, s13, v7                             
v_mad_u32_u16 v7, v84, s13, v7                             
_v_cmp_u16_vs_gfx11 86, 15, 1, 8
v_cndmask_b32_e32 v13, 0x80000000, v13, vcc                
_v_cmp_u16_vs_gfx11 81, 10, 1, 0
v_cndmask_b32_e32 v9, 0x80000000, v9, vcc                  
_v_cmp_u16_vs_gfx11 84, 10, 1, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
v_mad_u32_u16 v12, v86, s18, v12                           
v_pk_mad_u16 v5, v82, s9, v85                              
_v_cmp_u16_vs_gfx11 86, 15, 1, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 59, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 60, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 61, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 62, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 18, 59, 1, 59, 1, 71, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 60, 1, 60, 1, 73, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 61, 1, 61, 1, 75, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 62, 1, 62, 1, 77, 0, 0 // op16_max_f
s_setprio 0                                                
ds_load_b128 v[31:34], v3                                  
ds_store_b128 v18, v[7:10] offset:33024                    
ds_load_b128 v[35:38], v3 offset:512                       
ds_store_b32 v19, v4 offset:35072                          
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_add_co_u32 v22, vcc, v22, s26                            
v_add_nc_u32_e32 v16, s60, v14                             
v_add_nc_u32_e32 v17, s61, v15                             
v_pk_fma_f16 v39, v43, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 67, 1, 47, 1, 51, 0, 0 // op16_add_f
v_pk_fma_f16 v40, v44, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 68, 1, 48, 1, 52, 0, 0 // op16_add_f
v_pk_fma_f16 v41, v45, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 69, 1, 49, 1, 53, 0, 0 // op16_add_f
v_pk_fma_f16 v42, v46, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 70, 1, 50, 1, 54, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 67, 1, 67, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 68, 1, 68, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 69, 1, 69, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 70, 1, 70, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 67, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 68, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 69, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 70, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 18, 67, 1, 67, 1, 71, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 68, 1, 68, 1, 73, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 69, 1, 69, 1, 75, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 70, 1, 70, 1, 77, 0, 0 // op16_max_f
buffer_store_b16 v55, v20, s[64:67], 0 idxen               
buffer_store_b16 v59, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v55, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v59, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v56, v20, s[64:67], 0 idxen               
buffer_store_b16 v60, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v56, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v60, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v57, v20, s[64:67], 0 idxen               
buffer_store_b16 v61, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v57, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v61, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v58, v20, s[64:67], 0 idxen               
buffer_store_b16 v62, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v58, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v62, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v63, v20, s[64:67], 0 idxen               
buffer_store_b16 v67, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v63, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v67, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v64, v20, s[64:67], 0 idxen               
buffer_store_b16 v68, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v64, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v68, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v65, v20, s[64:67], 0 idxen               
buffer_store_b16 v69, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v65, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v69, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v66, v20, s[64:67], 0 idxen               
buffer_store_b16 v70, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v66, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v70, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
s_wait_idle                                                
s_setprio 0                                                
ds_load_b64 v[20:21], v16 offset:35584                     
ds_store_b64 v16, v[12:13] offset:35584                    
ds_load_b32 v22, v17 offset:40704                          
ds_store_b32 v17, v4 offset:40704                          
ds_load_b128 v[47:50], v3 offset:2048                      
ds_load_b128 v[51:54], v3 offset:2560                      
s_setprio 3                                                
s_cmp_eq_u32 s68, 0                                        
s_cselect_b32 vcc_lo, -1, vcc_lo                           
s_bfe_u32 null, vcc_lo, 0x10000                            
s_cbranch_scc0 1873                                        
s_sub_u32 s57, s57, 1                                      
s_cselect_b32 s57, 4, s57                                  
s_mov_b32 s68, s69                                         
s_cselect_b32 s69, 0x11014000, s69                         
s_lshl_b32 s60, s57, 9                                     
s_lshl_b32 s61, s57, 8                                     
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v55, v31, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 31, 1, 27, 1, 31, 2, 2 // op16_add_f
v_pk_fma_f16 v56, v32, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 32, 1, 28, 1, 32, 2, 2 // op16_add_f
v_pk_fma_f16 v57, v33, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 33, 1, 29, 1, 33, 2, 2 // op16_add_f
v_pk_fma_f16 v58, v34, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 34, 1, 30, 1, 34, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 55, 1, 55, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 56, 1, 56, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 57, 1, 57, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 58, 1, 58, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 55, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 56, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 57, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 58, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 18, 55, 1, 55, 1, 71, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 56, 1, 56, 1, 73, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 57, 1, 57, 1, 75, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 58, 1, 58, 1, 77, 0, 0 // op16_max_f
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[23:26], v3 offset:8256                      
ds_load_b128 v[27:30], v3 offset:8768                      
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v63, v47, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 47, 1, 43, 1, 47, 2, 2 // op16_add_f
v_pk_fma_f16 v64, v48, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 48, 1, 44, 1, 48, 2, 2 // op16_add_f
v_pk_fma_f16 v65, v49, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 49, 1, 45, 1, 49, 2, 2 // op16_add_f
v_pk_fma_f16 v66, v50, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 50, 1, 46, 1, 50, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 63, 1, 63, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 64, 1, 64, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 65, 1, 65, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 66, 1, 66, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 63, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 64, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 65, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 66, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 18, 63, 1, 63, 1, 71, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 64, 1, 64, 1, 73, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 65, 1, 65, 1, 75, 0, 0 // op16_max_f
_v_pk_op_x16_gfx1x 18, 66, 1, 66, 1, 77, 0, 0 // op16_max_f
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[39:42], v3 offset:10304                     
ds_load_b128 v[43:46], v3 offset:10816                     
s_setprio 3                                                
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_branch 65092                                             
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v23, v27, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 59, 1, 31, 1, 35, 0, 0 // op16_add_f
v_pk_fma_f16 v24, v28, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 60, 1, 32, 1, 36, 0, 0 // op16_add_f
v_pk_fma_f16 v25, v29, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 61, 1, 33, 1, 37, 0, 0 // op16_add_f
v_pk_fma_f16 v26, v30, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 62, 1, 34, 1, 38, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 59, 1, 59, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 60, 1, 60, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 61, 1, 61, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 62, 1, 62, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
v_add_co_u32 v85, vcc, v5, s5                              
v_pk_mad_u16 v87, v5, 0x20001, s11                         
v_pk_mad_u16 v88, v5, 0x20001, s16                         
_v_pk_op_x16_gfx1x 13, 90, 1, 85, 1, 5, 0, 0 // op16_min_u
v_cndmask_b32_e64 v79, 0, s12, vcc                         
v_cndmask_b32_e64 v80, 0, s17, vcc                         
v_mad_u32_u16 v7, v87, 1, v6                               
v_mad_u32_u16 v12, v88, 1, v11                             
v_add3_u32 v6, v6, s6, v79                                 
v_add3_u32 v11, v11, s7, v80                               
_v_pk_op_x16_gfx1x 11, 82, 1, 5, 1, 90, 0, 0 // op16_sub_u
v_add_co_ci_u32_e64 v4, s[58:59], v4, s1, vcc              
v_cndmask_b32_e64 v6, v6, 0x80000000, s[58:59]             
v_cndmask_b32_e64 v11, v11, 0x80000000, s[58:59]           
_v_cmp_u16_vs_gfx11 87, 10, 0, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
_v_cmp_u16_vs_gfx11 88, 15, 0, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_v2_gfx1x 6, 82, 15, 82 // op16_ashrrev_i
_v_pk_op_x16_v3_gfx1x 10, 81, 87, 55 // op16_add_u
_v_pk_op_x16_v3_gfx1x 10, 84, 87, 54 // op16_add_u
v_mad_u32_u16 v10, v81, s13, v7 op_sel:[1,0,0,0]           
v_mad_u32_u16 v8, v84, s13, v7 op_sel:[1,0,0,0]            
_v_pk_op_x16_v3_gfx1x 10, 86, 88, 56 // op16_add_u
_v_cmp_u16_vs_gfx11 81, 10, 1, 8
v_cndmask_b32_e32 v10, 0x80000000, v10, vcc                
_v_cmp_u16_vs_gfx11 84, 10, 1, 8
v_cndmask_b32_e32 v8, 0x80000000, v8, vcc                  
v_mad_u32_u16 v13, v86, s18, v12 op_sel:[1,0,0,0]          
v_mad_u32_u16 v9, v81, s13, v7                             
v_mad_u32_u16 v7, v84, s13, v7                             
_v_cmp_u16_vs_gfx11 86, 15, 1, 8
v_cndmask_b32_e32 v13, 0x80000000, v13, vcc                
_v_cmp_u16_vs_gfx11 81, 10, 1, 0
v_cndmask_b32_e32 v9, 0x80000000, v9, vcc                  
_v_cmp_u16_vs_gfx11 84, 10, 1, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
v_mad_u32_u16 v12, v86, s18, v12                           
v_pk_mad_u16 v5, v82, s9, v85                              
_v_cmp_u16_vs_gfx11 86, 15, 1, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 59, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 60, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 61, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 62, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 17, 59, 1, 59, 1, 71, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 60, 1, 60, 1, 73, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 61, 1, 61, 1, 75, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 62, 1, 62, 1, 77, 0, 0 // op16_min_f
s_setprio 0                                                
ds_load_b128 v[31:34], v3                                  
ds_store_b128 v18, v[7:10] offset:33024                    
ds_load_b128 v[35:38], v3 offset:512                       
ds_store_b32 v19, v4 offset:35072                          
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_add_co_u32 v22, vcc, v22, s26                            
v_add_nc_u32_e32 v16, s60, v14                             
v_add_nc_u32_e32 v17, s61, v15                             
v_pk_fma_f16 v39, v43, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 67, 1, 47, 1, 51, 0, 0 // op16_add_f
v_pk_fma_f16 v40, v44, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 68, 1, 48, 1, 52, 0, 0 // op16_add_f
v_pk_fma_f16 v41, v45, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 69, 1, 49, 1, 53, 0, 0 // op16_add_f
v_pk_fma_f16 v42, v46, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 70, 1, 50, 1, 54, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 67, 1, 67, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 68, 1, 68, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 69, 1, 69, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 70, 1, 70, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 67, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 68, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 69, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 70, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 17, 67, 1, 67, 1, 71, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 68, 1, 68, 1, 73, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 69, 1, 69, 1, 75, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 70, 1, 70, 1, 77, 0, 0 // op16_min_f
buffer_store_b16 v55, v20, s[64:67], 0 idxen               
buffer_store_b16 v59, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v55, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v59, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v56, v20, s[64:67], 0 idxen               
buffer_store_b16 v60, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v56, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v60, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v57, v20, s[64:67], 0 idxen               
buffer_store_b16 v61, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v57, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v61, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v58, v20, s[64:67], 0 idxen               
buffer_store_b16 v62, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v58, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v62, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v63, v20, s[64:67], 0 idxen               
buffer_store_b16 v67, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v63, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v67, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v64, v20, s[64:67], 0 idxen               
buffer_store_b16 v68, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v64, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v68, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v65, v20, s[64:67], 0 idxen               
buffer_store_b16 v69, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v65, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v69, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v66, v20, s[64:67], 0 idxen               
buffer_store_b16 v70, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v66, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v70, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
s_wait_idle                                                
s_setprio 0                                                
ds_load_b64 v[20:21], v16 offset:35584                     
ds_store_b64 v16, v[12:13] offset:35584                    
ds_load_b32 v22, v17 offset:40704                          
ds_store_b32 v17, v4 offset:40704                          
ds_load_b128 v[47:50], v3 offset:2048                      
ds_load_b128 v[51:54], v3 offset:2560                      
s_setprio 3                                                
s_cmp_eq_u32 s68, 0                                        
s_cselect_b32 vcc_lo, -1, vcc_lo                           
s_bfe_u32 null, vcc_lo, 0x10000                            
s_cbranch_scc0 1425                                        
s_sub_u32 s57, s57, 1                                      
s_cselect_b32 s57, 4, s57                                  
s_mov_b32 s68, s69                                         
s_cselect_b32 s69, 0x11014000, s69                         
s_lshl_b32 s60, s57, 9                                     
s_lshl_b32 s61, s57, 8                                     
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v55, v31, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 31, 1, 27, 1, 31, 2, 2 // op16_add_f
v_pk_fma_f16 v56, v32, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 32, 1, 28, 1, 32, 2, 2 // op16_add_f
v_pk_fma_f16 v57, v33, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 33, 1, 29, 1, 33, 2, 2 // op16_add_f
v_pk_fma_f16 v58, v34, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 34, 1, 30, 1, 34, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 55, 1, 55, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 56, 1, 56, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 57, 1, 57, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 58, 1, 58, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 55, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 56, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 57, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 58, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 17, 55, 1, 55, 1, 71, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 56, 1, 56, 1, 73, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 57, 1, 57, 1, 75, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 58, 1, 58, 1, 77, 0, 0 // op16_min_f
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[23:26], v3 offset:8256                      
ds_load_b128 v[27:30], v3 offset:8768                      
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v63, v47, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 47, 1, 43, 1, 47, 2, 2 // op16_add_f
v_pk_fma_f16 v64, v48, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 48, 1, 44, 1, 48, 2, 2 // op16_add_f
v_pk_fma_f16 v65, v49, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 49, 1, 45, 1, 49, 2, 2 // op16_add_f
v_pk_fma_f16 v66, v50, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 50, 1, 46, 1, 50, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 63, 1, 63, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 64, 1, 64, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 65, 1, 65, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 66, 1, 66, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 71, 0, 42, 1, 63, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 73, 0, 42, 1, 64, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 75, 0, 42, 1, 65, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 77, 0, 42, 1, 66, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 17, 63, 1, 63, 1, 71, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 64, 1, 64, 1, 73, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 65, 1, 65, 1, 75, 0, 0 // op16_min_f
_v_pk_op_x16_gfx1x 17, 66, 1, 66, 1, 77, 0, 0 // op16_min_f
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[39:42], v3 offset:10304                     
ds_load_b128 v[43:46], v3 offset:10816                     
s_setprio 3                                                
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_branch 65092                                             
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v23, v27, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 59, 1, 31, 1, 35, 0, 0 // op16_add_f
v_pk_fma_f16 v24, v28, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 60, 1, 32, 1, 36, 0, 0 // op16_add_f
v_pk_fma_f16 v25, v29, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 61, 1, 33, 1, 37, 0, 0 // op16_add_f
v_pk_fma_f16 v26, v30, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 62, 1, 34, 1, 38, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 59, 1, 59, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 60, 1, 60, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 61, 1, 61, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 62, 1, 62, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
v_add_co_u32 v85, vcc, v5, s5                              
v_pk_mad_u16 v87, v5, 0x20001, s11                         
v_pk_mad_u16 v88, v5, 0x20001, s16                         
_v_pk_op_x16_gfx1x 13, 90, 1, 85, 1, 5, 0, 0 // op16_min_u
v_cndmask_b32_e64 v79, 0, s12, vcc                         
v_cndmask_b32_e64 v80, 0, s17, vcc                         
v_mad_u32_u16 v7, v87, 1, v6                               
v_mad_u32_u16 v12, v88, 1, v11                             
v_add3_u32 v6, v6, s6, v79                                 
v_add3_u32 v11, v11, s7, v80                               
_v_pk_op_x16_gfx1x 11, 82, 1, 5, 1, 90, 0, 0 // op16_sub_u
v_add_co_ci_u32_e64 v4, s[58:59], v4, s1, vcc              
v_cndmask_b32_e64 v6, v6, 0x80000000, s[58:59]             
v_cndmask_b32_e64 v11, v11, 0x80000000, s[58:59]           
_v_cmp_u16_vs_gfx11 87, 10, 0, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
_v_cmp_u16_vs_gfx11 88, 15, 0, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_v2_gfx1x 6, 82, 15, 82 // op16_ashrrev_i
_v_pk_op_x16_v3_gfx1x 10, 81, 87, 55 // op16_add_u
_v_pk_op_x16_v3_gfx1x 10, 84, 87, 54 // op16_add_u
v_mad_u32_u16 v10, v81, s13, v7 op_sel:[1,0,0,0]           
v_mad_u32_u16 v8, v84, s13, v7 op_sel:[1,0,0,0]            
_v_pk_op_x16_v3_gfx1x 10, 86, 88, 56 // op16_add_u
_v_cmp_u16_vs_gfx11 81, 10, 1, 8
v_cndmask_b32_e32 v10, 0x80000000, v10, vcc                
_v_cmp_u16_vs_gfx11 84, 10, 1, 8
v_cndmask_b32_e32 v8, 0x80000000, v8, vcc                  
v_mad_u32_u16 v13, v86, s18, v12 op_sel:[1,0,0,0]          
v_mad_u32_u16 v9, v81, s13, v7                             
v_mad_u32_u16 v7, v84, s13, v7                             
_v_cmp_u16_vs_gfx11 86, 15, 1, 8
v_cndmask_b32_e32 v13, 0x80000000, v13, vcc                
_v_cmp_u16_vs_gfx11 81, 10, 1, 0
v_cndmask_b32_e32 v9, 0x80000000, v9, vcc                  
_v_cmp_u16_vs_gfx11 84, 10, 1, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
v_mad_u32_u16 v12, v86, s18, v12                           
v_pk_mad_u16 v5, v82, s9, v85                              
_v_cmp_u16_vs_gfx11 86, 15, 1, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_v4_gfx1x 16, 59, 0xbdc5bdc5, 59 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 60, 0xbdc5bdc5, 60 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 61, 0xbdc5bdc5, 61 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 62, 0xbdc5bdc5, 62 // op16_mul_f
v_exp_f16_e32 v59, v59                                     
v_exp_f16_e32 v60, v60                                     
v_exp_f16_e32 v61, v61                                     
v_exp_f16_e32 v62, v62                                     
.long 0x7f76b1bb                                           
.long 0x7f78b1bc                                           
.long 0x7f7ab1bd                                           
.long 0x7f7cb1be                                           
_v_pk_op_x16_v4_gfx1x 15, 59, 0x3c003c00, 59 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 60, 0x3c003c00, 60 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 61, 0x3c003c00, 61 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 62, 0x3c003c00, 62 // op16_add_f
v_rcp_f16_e32 v59, v59                                     
v_rcp_f16_e32 v60, v60                                     
v_rcp_f16_e32 v61, v61                                     
v_rcp_f16_e32 v62, v62                                     
.long 0x7f76a9bb                                           
.long 0x7f78a9bc                                           
.long 0x7f7aa9bd                                           
.long 0x7f7ca9be                                           
s_setprio 0                                                
ds_load_b128 v[31:34], v3                                  
ds_store_b128 v18, v[7:10] offset:33024                    
ds_load_b128 v[35:38], v3 offset:512                       
ds_store_b32 v19, v4 offset:35072                          
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_add_co_u32 v22, vcc, v22, s26                            
v_add_nc_u32_e32 v16, s60, v14                             
v_add_nc_u32_e32 v17, s61, v15                             
v_pk_fma_f16 v39, v43, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 67, 1, 47, 1, 51, 0, 0 // op16_add_f
v_pk_fma_f16 v40, v44, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 68, 1, 48, 1, 52, 0, 0 // op16_add_f
v_pk_fma_f16 v41, v45, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 69, 1, 49, 1, 53, 0, 0 // op16_add_f
v_pk_fma_f16 v42, v46, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 70, 1, 50, 1, 54, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 67, 1, 67, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 68, 1, 68, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 69, 1, 69, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 70, 1, 70, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_v4_gfx1x 16, 67, 0xbdc5bdc5, 67 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 68, 0xbdc5bdc5, 68 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 69, 0xbdc5bdc5, 69 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 70, 0xbdc5bdc5, 70 // op16_mul_f
v_exp_f16_e32 v67, v67                                     
v_exp_f16_e32 v68, v68                                     
v_exp_f16_e32 v69, v69                                     
v_exp_f16_e32 v70, v70                                     
.long 0x7f86b1c3                                           
.long 0x7f88b1c4                                           
.long 0x7f8ab1c5                                           
.long 0x7f8cb1c6                                           
_v_pk_op_x16_v4_gfx1x 15, 67, 0x3c003c00, 67 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 68, 0x3c003c00, 68 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 69, 0x3c003c00, 69 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 70, 0x3c003c00, 70 // op16_add_f
v_rcp_f16_e32 v67, v67                                     
v_rcp_f16_e32 v68, v68                                     
v_rcp_f16_e32 v69, v69                                     
v_rcp_f16_e32 v70, v70                                     
.long 0x7f86a9c3                                           
.long 0x7f88a9c4                                           
.long 0x7f8aa9c5                                           
.long 0x7f8ca9c6                                           
buffer_store_b16 v55, v20, s[64:67], 0 idxen               
buffer_store_b16 v59, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v55, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v59, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v56, v20, s[64:67], 0 idxen               
buffer_store_b16 v60, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v56, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v60, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v57, v20, s[64:67], 0 idxen               
buffer_store_b16 v61, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v57, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v61, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v58, v20, s[64:67], 0 idxen               
buffer_store_b16 v62, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v58, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v62, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v63, v20, s[64:67], 0 idxen               
buffer_store_b16 v67, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v63, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v67, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v64, v20, s[64:67], 0 idxen               
buffer_store_b16 v68, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v64, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v68, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v65, v20, s[64:67], 0 idxen               
buffer_store_b16 v69, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v65, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v69, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v66, v20, s[64:67], 0 idxen               
buffer_store_b16 v70, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v66, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v70, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
s_wait_idle                                                
s_setprio 0                                                
ds_load_b64 v[20:21], v16 offset:35584                     
ds_store_b64 v16, v[12:13] offset:35584                    
ds_load_b32 v22, v17 offset:40704                          
ds_store_b32 v17, v4 offset:40704                          
ds_load_b128 v[47:50], v3 offset:2048                      
ds_load_b128 v[51:54], v3 offset:2560                      
s_setprio 3                                                
s_cmp_eq_u32 s68, 0                                        
s_cselect_b32 vcc_lo, -1, vcc_lo                           
s_bfe_u32 null, vcc_lo, 0x10000                            
s_cbranch_scc0 929                                         
s_sub_u32 s57, s57, 1                                      
s_cselect_b32 s57, 4, s57                                  
s_mov_b32 s68, s69                                         
s_cselect_b32 s69, 0x11014000, s69                         
s_lshl_b32 s60, s57, 9                                     
s_lshl_b32 s61, s57, 8                                     
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v55, v31, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 31, 1, 27, 1, 31, 2, 2 // op16_add_f
v_pk_fma_f16 v56, v32, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 32, 1, 28, 1, 32, 2, 2 // op16_add_f
v_pk_fma_f16 v57, v33, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 33, 1, 29, 1, 33, 2, 2 // op16_add_f
v_pk_fma_f16 v58, v34, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 34, 1, 30, 1, 34, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 55, 1, 55, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 56, 1, 56, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 57, 1, 57, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 58, 1, 58, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
_v_pk_op_x16_v4_gfx1x 16, 55, 0xbdc5bdc5, 55 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 56, 0xbdc5bdc5, 56 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 57, 0xbdc5bdc5, 57 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 58, 0xbdc5bdc5, 58 // op16_mul_f
v_exp_f16_e32 v55, v55                                     
v_exp_f16_e32 v56, v56                                     
v_exp_f16_e32 v57, v57                                     
v_exp_f16_e32 v58, v58                                     
.long 0x7f6eb1b7                                           
.long 0x7f70b1b8                                           
.long 0x7f72b1b9                                           
.long 0x7f74b1ba                                           
_v_pk_op_x16_v4_gfx1x 15, 55, 0x3c003c00, 55 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 56, 0x3c003c00, 56 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 57, 0x3c003c00, 57 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 58, 0x3c003c00, 58 // op16_add_f
v_rcp_f16_e32 v55, v55                                     
v_rcp_f16_e32 v56, v56                                     
v_rcp_f16_e32 v57, v57                                     
v_rcp_f16_e32 v58, v58                                     
.long 0x7f6ea9b7                                           
.long 0x7f70a9b8                                           
.long 0x7f72a9b9                                           
.long 0x7f74a9ba                                           
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[23:26], v3 offset:8256                      
ds_load_b128 v[27:30], v3 offset:8768                      
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v63, v47, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 47, 1, 43, 1, 47, 2, 2 // op16_add_f
v_pk_fma_f16 v64, v48, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 48, 1, 44, 1, 48, 2, 2 // op16_add_f
v_pk_fma_f16 v65, v49, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 49, 1, 45, 1, 49, 2, 2 // op16_add_f
v_pk_fma_f16 v66, v50, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 50, 1, 46, 1, 50, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 63, 1, 63, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 64, 1, 64, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 65, 1, 65, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 66, 1, 66, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_v4_gfx1x 16, 63, 0xbdc5bdc5, 63 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 64, 0xbdc5bdc5, 64 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 65, 0xbdc5bdc5, 65 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 66, 0xbdc5bdc5, 66 // op16_mul_f
v_exp_f16_e32 v63, v63                                     
v_exp_f16_e32 v64, v64                                     
v_exp_f16_e32 v65, v65                                     
v_exp_f16_e32 v66, v66                                     
.long 0x7f7eb1bf                                           
.long 0x7f80b1c0                                           
.long 0x7f82b1c1                                           
.long 0x7f84b1c2                                           
_v_pk_op_x16_v4_gfx1x 15, 63, 0x3c003c00, 63 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 64, 0x3c003c00, 64 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 65, 0x3c003c00, 65 // op16_add_f
_v_pk_op_x16_v4_gfx1x 15, 66, 0x3c003c00, 66 // op16_add_f
v_rcp_f16_e32 v63, v63                                     
v_rcp_f16_e32 v64, v64                                     
v_rcp_f16_e32 v65, v65                                     
v_rcp_f16_e32 v66, v66                                     
.long 0x7f7ea9bf                                           
.long 0x7f80a9c0                                           
.long 0x7f82a9c1                                           
.long 0x7f84a9c2                                           
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[39:42], v3 offset:10304                     
ds_load_b128 v[43:46], v3 offset:10816                     
s_setprio 3                                                
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_branch 64996                                             
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_nop 0                                                    
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v23, v27, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 59, 1, 31, 1, 35, 0, 0 // op16_add_f
v_pk_fma_f16 v24, v28, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 60, 1, 32, 1, 36, 0, 0 // op16_add_f
v_pk_fma_f16 v25, v29, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 61, 1, 33, 1, 37, 0, 0 // op16_add_f
v_pk_fma_f16 v26, v30, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 62, 1, 34, 1, 38, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 59, 1, 59, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 60, 1, 60, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 61, 1, 61, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 62, 1, 62, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
v_add_co_u32 v85, vcc, v5, s5                              
v_pk_mad_u16 v87, v5, 0x20001, s11                         
v_pk_mad_u16 v88, v5, 0x20001, s16                         
_v_pk_op_x16_gfx1x 13, 90, 1, 85, 1, 5, 0, 0 // op16_min_u
v_cndmask_b32_e64 v79, 0, s12, vcc                         
v_cndmask_b32_e64 v80, 0, s17, vcc                         
v_mad_u32_u16 v7, v87, 1, v6                               
v_mad_u32_u16 v12, v88, 1, v11                             
v_add3_u32 v6, v6, s6, v79                                 
v_add3_u32 v11, v11, s7, v80                               
_v_pk_op_x16_gfx1x 11, 82, 1, 5, 1, 90, 0, 0 // op16_sub_u
v_add_co_ci_u32_e64 v4, s[58:59], v4, s1, vcc              
v_cndmask_b32_e64 v6, v6, 0x80000000, s[58:59]             
v_cndmask_b32_e64 v11, v11, 0x80000000, s[58:59]           
_v_cmp_u16_vs_gfx11 87, 10, 0, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
_v_cmp_u16_vs_gfx11 88, 15, 0, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_v2_gfx1x 6, 82, 15, 82 // op16_ashrrev_i
_v_pk_op_x16_v3_gfx1x 10, 81, 87, 55 // op16_add_u
_v_pk_op_x16_v3_gfx1x 10, 84, 87, 54 // op16_add_u
v_mad_u32_u16 v10, v81, s13, v7 op_sel:[1,0,0,0]           
v_mad_u32_u16 v8, v84, s13, v7 op_sel:[1,0,0,0]            
_v_pk_op_x16_v3_gfx1x 10, 86, 88, 56 // op16_add_u
_v_cmp_u16_vs_gfx11 81, 10, 1, 8
v_cndmask_b32_e32 v10, 0x80000000, v10, vcc                
_v_cmp_u16_vs_gfx11 84, 10, 1, 8
v_cndmask_b32_e32 v8, 0x80000000, v8, vcc                  
v_mad_u32_u16 v13, v86, s18, v12 op_sel:[1,0,0,0]          
v_mad_u32_u16 v9, v81, s13, v7                             
v_mad_u32_u16 v7, v84, s13, v7                             
_v_cmp_u16_vs_gfx11 86, 15, 1, 8
v_cndmask_b32_e32 v13, 0x80000000, v13, vcc                
_v_cmp_u16_vs_gfx11 81, 10, 1, 0
v_cndmask_b32_e32 v9, 0x80000000, v9, vcc                  
_v_cmp_u16_vs_gfx11 84, 10, 1, 0
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc                  
v_mad_u32_u16 v12, v86, s18, v12                           
v_pk_mad_u16 v5, v82, s9, v85                              
_v_cmp_u16_vs_gfx11 86, 15, 1, 0
v_cndmask_b32_e32 v12, 0x80000000, v12, vcc                
_v_pk_op_x16_gfx1x 16, 59, 1, 59, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 60, 1, 60, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 61, 1, 61, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 62, 1, 62, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 71, 0x3dc53dc5, 59 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 73, 0x3dc53dc5, 60 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 75, 0x3dc53dc5, 61 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 77, 0x3dc53dc5, 62 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 72, 0xbdc5bdc5, 59 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 74, 0xbdc5bdc5, 60 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 76, 0xbdc5bdc5, 61 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 78, 0xbdc5bdc5, 62 // op16_mul_f
v_exp_f16_e32 v71, v71                                     
v_exp_f16_e32 v73, v73                                     
v_exp_f16_e32 v75, v75                                     
v_exp_f16_e32 v77, v77                                     
.long 0x7f8eb1c7                                           
.long 0x7f92b1c9                                           
.long 0x7f96b1cb                                           
.long 0x7f9ab1cd                                           
v_exp_f16_e32 v72, v72                                     
v_exp_f16_e32 v74, v74                                     
v_exp_f16_e32 v76, v76                                     
v_exp_f16_e32 v78, v78                                     
.long 0x7f90b1c8                                           
.long 0x7f94b1ca                                           
.long 0x7f98b1cc                                           
.long 0x7f9cb1ce                                           
_v_pk_op_x16_gfx1x 15, 59, 1, 71, 1, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 60, 1, 73, 1, 74, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 61, 1, 75, 1, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 62, 1, 77, 1, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 71, 1, 71, 1, 72, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 73, 1, 73, 1, 74, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 75, 1, 75, 1, 76, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 77, 1, 77, 1, 78, 2, 2 // op16_add_f
v_rcp_f16_e32 v59, v59                                     
v_rcp_f16_e32 v60, v60                                     
v_rcp_f16_e32 v61, v61                                     
v_rcp_f16_e32 v62, v62                                     
.long 0x7f76a9bb                                           
.long 0x7f78a9bc                                           
.long 0x7f7aa9bd                                           
.long 0x7f7ca9be                                           
_v_pk_op_x16_gfx1x 16, 59, 1, 59, 1, 71, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 60, 1, 60, 1, 73, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 61, 1, 61, 1, 75, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 62, 1, 62, 1, 77, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 59, 1, 59, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 60, 1, 60, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 61, 1, 61, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 62, 1, 62, 0, 42, 0, 0 // op16_mul_f
s_setprio 0                                                
ds_load_b128 v[31:34], v3                                  
ds_store_b128 v18, v[7:10] offset:33024                    
ds_load_b128 v[35:38], v3 offset:512                       
ds_store_b32 v19, v4 offset:35072                          
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_add_co_u32 v22, vcc, v22, s26                            
v_add_nc_u32_e32 v16, s60, v14                             
v_add_nc_u32_e32 v17, s61, v15                             
v_pk_fma_f16 v39, v43, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 67, 1, 47, 1, 51, 0, 0 // op16_add_f
v_pk_fma_f16 v40, v44, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 68, 1, 48, 1, 52, 0, 0 // op16_add_f
v_pk_fma_f16 v41, v45, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 69, 1, 49, 1, 53, 0, 0 // op16_add_f
v_pk_fma_f16 v42, v46, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 70, 1, 50, 1, 54, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 67, 1, 67, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 68, 1, 68, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 69, 1, 69, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 70, 1, 70, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 67, 1, 67, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 68, 1, 68, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 69, 1, 69, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 70, 1, 70, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 71, 0x3dc53dc5, 67 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 73, 0x3dc53dc5, 68 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 75, 0x3dc53dc5, 69 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 77, 0x3dc53dc5, 70 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 72, 0xbdc5bdc5, 67 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 74, 0xbdc5bdc5, 68 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 76, 0xbdc5bdc5, 69 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 78, 0xbdc5bdc5, 70 // op16_mul_f
v_exp_f16_e32 v71, v71                                     
v_exp_f16_e32 v73, v73                                     
v_exp_f16_e32 v75, v75                                     
v_exp_f16_e32 v77, v77                                     
.long 0x7f8eb1c7                                           
.long 0x7f92b1c9                                           
.long 0x7f96b1cb                                           
.long 0x7f9ab1cd                                           
v_exp_f16_e32 v72, v72                                     
v_exp_f16_e32 v74, v74                                     
v_exp_f16_e32 v76, v76                                     
v_exp_f16_e32 v78, v78                                     
.long 0x7f90b1c8                                           
.long 0x7f94b1ca                                           
.long 0x7f98b1cc                                           
.long 0x7f9cb1ce                                           
_v_pk_op_x16_gfx1x 15, 67, 1, 71, 1, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 68, 1, 73, 1, 74, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 69, 1, 75, 1, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 70, 1, 77, 1, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 71, 1, 71, 1, 72, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 73, 1, 73, 1, 74, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 75, 1, 75, 1, 76, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 77, 1, 77, 1, 78, 2, 2 // op16_add_f
v_rcp_f16_e32 v67, v67                                     
v_rcp_f16_e32 v68, v68                                     
v_rcp_f16_e32 v69, v69                                     
v_rcp_f16_e32 v70, v70                                     
.long 0x7f86a9c3                                           
.long 0x7f88a9c4                                           
.long 0x7f8aa9c5                                           
.long 0x7f8ca9c6                                           
_v_pk_op_x16_gfx1x 16, 67, 1, 67, 1, 71, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 68, 1, 68, 1, 73, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 69, 1, 69, 1, 75, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 70, 1, 70, 1, 77, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 67, 1, 67, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 68, 1, 68, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 69, 1, 69, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 70, 1, 70, 0, 42, 0, 0 // op16_mul_f
buffer_store_b16 v55, v20, s[64:67], 0 idxen               
buffer_store_b16 v59, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v55, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v59, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v56, v20, s[64:67], 0 idxen               
buffer_store_b16 v60, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v56, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v60, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v57, v20, s[64:67], 0 idxen               
buffer_store_b16 v61, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v57, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v61, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v58, v20, s[64:67], 0 idxen               
buffer_store_b16 v62, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v58, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v62, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v63, v20, s[64:67], 0 idxen               
buffer_store_b16 v67, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v63, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v67, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v64, v20, s[64:67], 0 idxen               
buffer_store_b16 v68, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v64, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v68, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v65, v20, s[64:67], 0 idxen               
buffer_store_b16 v69, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v65, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v69, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_b16 v66, v20, s[64:67], 0 idxen               
buffer_store_b16 v70, v21, s[64:67], 0 idxen               
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
buffer_store_d16_hi_b16 v66, v20, s[64:67], 0 idxen        
buffer_store_d16_hi_b16 v70, v21, s[64:67], 0 idxen        
s_sub_u32 s62, s62, 1                                      
s_cselect_b32 s67, 0, s67                                  
s_add_u32 s64, s64, s63                                    
s_addc_u32 s65, s65, 0                                     
s_wait_idle                                                
s_setprio 0                                                
ds_load_b64 v[20:21], v16 offset:35584                     
ds_store_b64 v16, v[12:13] offset:35584                    
ds_load_b32 v22, v17 offset:40704                          
ds_store_b32 v17, v4 offset:40704                          
ds_load_b128 v[47:50], v3 offset:2048                      
ds_load_b128 v[51:54], v3 offset:2560                      
s_setprio 3                                                
s_cmp_eq_u32 s68, 0                                        
s_cselect_b32 vcc_lo, -1, vcc_lo                           
s_bfe_u32 null, vcc_lo, 0x10000                            
s_cbranch_scc0 257                                         
s_sub_u32 s57, s57, 1                                      
s_cselect_b32 s57, 4, s57                                  
s_mov_b32 s68, s69                                         
s_cselect_b32 s69, 0x11014000, s69                         
s_lshl_b32 s60, s57, 9                                     
s_lshl_b32 s61, s57, 8                                     
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v55, v31, s70, v23 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 31, 1, 27, 1, 31, 2, 2 // op16_add_f
v_pk_fma_f16 v56, v32, s70, v24 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 32, 1, 28, 1, 32, 2, 2 // op16_add_f
v_pk_fma_f16 v57, v33, s70, v25 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 33, 1, 29, 1, 33, 2, 2 // op16_add_f
v_pk_fma_f16 v58, v34, s70, v26 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 34, 1, 30, 1, 34, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 55, 1, 55, 0, 71, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 56, 1, 56, 0, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 57, 1, 57, 0, 73, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 58, 1, 58, 0, 74, 0, 0 // op16_add_f
s_barrier                                                  
_v_pk_op_x16_gfx1x 16, 55, 1, 55, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 56, 1, 56, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 57, 1, 57, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 58, 1, 58, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 71, 0x3dc53dc5, 55 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 73, 0x3dc53dc5, 56 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 75, 0x3dc53dc5, 57 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 77, 0x3dc53dc5, 58 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 72, 0xbdc5bdc5, 55 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 74, 0xbdc5bdc5, 56 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 76, 0xbdc5bdc5, 57 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 78, 0xbdc5bdc5, 58 // op16_mul_f
v_exp_f16_e32 v71, v71                                     
v_exp_f16_e32 v73, v73                                     
v_exp_f16_e32 v75, v75                                     
v_exp_f16_e32 v77, v77                                     
.long 0x7f8eb1c7                                           
.long 0x7f92b1c9                                           
.long 0x7f96b1cb                                           
.long 0x7f9ab1cd                                           
v_exp_f16_e32 v72, v72                                     
v_exp_f16_e32 v74, v74                                     
v_exp_f16_e32 v76, v76                                     
v_exp_f16_e32 v78, v78                                     
.long 0x7f90b1c8                                           
.long 0x7f94b1ca                                           
.long 0x7f98b1cc                                           
.long 0x7f9cb1ce                                           
_v_pk_op_x16_gfx1x 15, 55, 1, 71, 1, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 56, 1, 73, 1, 74, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 57, 1, 75, 1, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 58, 1, 77, 1, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 71, 1, 71, 1, 72, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 73, 1, 73, 1, 74, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 75, 1, 75, 1, 76, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 77, 1, 77, 1, 78, 2, 2 // op16_add_f
v_rcp_f16_e32 v55, v55                                     
v_rcp_f16_e32 v56, v56                                     
v_rcp_f16_e32 v57, v57                                     
v_rcp_f16_e32 v58, v58                                     
.long 0x7f6ea9b7                                           
.long 0x7f70a9b8                                           
.long 0x7f72a9b9                                           
.long 0x7f74a9ba                                           
_v_pk_op_x16_gfx1x 16, 55, 1, 55, 1, 71, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 56, 1, 56, 1, 73, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 57, 1, 57, 1, 75, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 58, 1, 58, 1, 77, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 55, 1, 55, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 56, 1, 56, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 57, 1, 57, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 58, 1, 58, 0, 42, 0, 0 // op16_mul_f
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[23:26], v3 offset:8256                      
ds_load_b128 v[27:30], v3 offset:8768                      
s_setprio 3                                                
s_waitcnt lgkmcnt(0)                                       
v_pk_fma_f16 v63, v47, s70, v39 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 47, 1, 43, 1, 47, 2, 2 // op16_add_f
v_pk_fma_f16 v64, v48, s70, v40 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 48, 1, 44, 1, 48, 2, 2 // op16_add_f
v_pk_fma_f16 v65, v49, s70, v41 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 49, 1, 45, 1, 49, 2, 2 // op16_add_f
v_pk_fma_f16 v66, v50, s70, v42 op_sel:[0,1,0]             
_v_pk_op_x16_gfx1x 15, 50, 1, 46, 1, 50, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 63, 1, 63, 0, 75, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 64, 1, 64, 0, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 65, 1, 65, 0, 77, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 66, 1, 66, 0, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 16, 63, 1, 63, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 64, 1, 64, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 65, 1, 65, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 66, 1, 66, 0, 43, 0, 0 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 71, 0x3dc53dc5, 63 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 73, 0x3dc53dc5, 64 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 75, 0x3dc53dc5, 65 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 77, 0x3dc53dc5, 66 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 72, 0xbdc5bdc5, 63 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 74, 0xbdc5bdc5, 64 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 76, 0xbdc5bdc5, 65 // op16_mul_f
_v_pk_op_x16_v4_gfx1x 16, 78, 0xbdc5bdc5, 66 // op16_mul_f
v_exp_f16_e32 v71, v71                                     
v_exp_f16_e32 v73, v73                                     
v_exp_f16_e32 v75, v75                                     
v_exp_f16_e32 v77, v77                                     
.long 0x7f8eb1c7                                           
.long 0x7f92b1c9                                           
.long 0x7f96b1cb                                           
.long 0x7f9ab1cd                                           
v_exp_f16_e32 v72, v72                                     
v_exp_f16_e32 v74, v74                                     
v_exp_f16_e32 v76, v76                                     
v_exp_f16_e32 v78, v78                                     
.long 0x7f90b1c8                                           
.long 0x7f94b1ca                                           
.long 0x7f98b1cc                                           
.long 0x7f9cb1ce                                           
_v_pk_op_x16_gfx1x 15, 63, 1, 71, 1, 72, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 64, 1, 73, 1, 74, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 65, 1, 75, 1, 76, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 66, 1, 77, 1, 78, 0, 0 // op16_add_f
_v_pk_op_x16_gfx1x 15, 71, 1, 71, 1, 72, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 73, 1, 73, 1, 74, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 75, 1, 75, 1, 76, 2, 2 // op16_add_f
_v_pk_op_x16_gfx1x 15, 77, 1, 77, 1, 78, 2, 2 // op16_add_f
v_rcp_f16_e32 v63, v63                                     
v_rcp_f16_e32 v64, v64                                     
v_rcp_f16_e32 v65, v65                                     
v_rcp_f16_e32 v66, v66                                     
.long 0x7f7ea9bf                                           
.long 0x7f80a9c0                                           
.long 0x7f82a9c1                                           
.long 0x7f84a9c2                                           
_v_pk_op_x16_gfx1x 16, 63, 1, 63, 1, 71, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 64, 1, 64, 1, 73, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 65, 1, 65, 1, 75, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 66, 1, 66, 1, 77, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 63, 1, 63, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 64, 1, 64, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 65, 1, 65, 0, 42, 0, 0 // op16_mul_f
_v_pk_op_x16_gfx1x 16, 66, 1, 66, 0, 42, 0, 0 // op16_mul_f
s_setprio 0                                                
s_nop 15                                                   
s_nop 15                                                   
ds_load_b128 v[39:42], v3 offset:10304                     
ds_load_b128 v[43:46], v3 offset:10816                     
s_setprio 3                                                
s_sub_u32 s62, s27, 1                                      
s_cselect_b32 s67, 0, s68                                  
s_mov_b64 s[64:65], s[22:23]                               
s_branch 64804                                             
s_endpgm                                                   
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end                                                 
s_code_end
