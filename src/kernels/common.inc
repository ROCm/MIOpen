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

.ifdef EXPERIMENTAL_COv3
    .ifdef .option.machine_version_major
        .error ".option.machine_version_major already defined"
        .end
    .endif
    .set .option.machine_version_major, .amdgcn.gfx_generation_number
.endif

LAYOUT_DATA_NCHW = 0
LAYOUT_DATA_CNHW = 1
LAYOUT_DATA_NHWC = 2
LAYOUT_DATA_CHWN = 3

// acc_type and buf_type: 0 - FP64, 1 - FP32, 2 - FP16, 5 - int32, 6 - int16, 7 - int8
TYPE_FP64  = 0
TYPE_FP32  = 1
TYPE_FP16  = 2
TYPE_BFP16 = 3
TYPE_INT64 = 4
TYPE_INT32 = 5
TYPE_INT16 = 6
TYPE_INT8  = 7
TYPE_INT4  = 8

.macro log2 lg2, num, max_bits=8
    \lg2 = 0
    lg_i = \num
    .rept \max_bits
        lg_i = lg_i / 2
        .if lg_i > 0
            \lg2 = \lg2 + 1
        .endif
    .endr
.endm

.macro default symbol, value
    .ifnotdef \symbol
        \symbol = \value
    .endif
.endm

.macro static_assert fufufu
    .if !\fufufu
        .error "\fufufu is false"
        .end
    .endif
.endm

.macro swap a, b
    __tmp = \a
    \a = \b
    \b = __tmp
.endm

.macro m_bpermute vgpr, cnt, addr
    v = \vgpr
    .rept \cnt
        ds_bpermute_b32 v[v], v[\addr], v[v]
        v = v + 1
    .endr
.endm

.macro m_swizzle vgpr, cnt, pattern
    v = \vgpr
    .rept \cnt
        ds_swizzle_b32 v[v], v[v] offset:\pattern
        v = v + 1
    .endr
.endm

.if (.option.machine_version_major == 8)
    .set max_hw_vctn, 15
.elseif (.option.machine_version_major == 9)
    .set max_hw_vctn, 63
.endif
max_hw_lcnt = 15
.macro s_wait vmcnt=max_hw_vctn, lgkmcnt=max_hw_lcnt
    vm_cnt = \vmcnt
    lgkm_cnt = \lgkmcnt
    .if vm_cnt > max_hw_vctn
        vm_cnt = max_hw_vctn
    .elseif vm_cnt < 0
        vm_cnt = 0
    .endif
    .if lgkm_cnt > max_hw_lcnt
        lgkm_cnt = max_hw_lcnt
    .elseif lgkm_cnt < 0
        lgkm_cnt = 0
    .endif
    s_waitcnt vmcnt(0 + vm_cnt) & lgkmcnt(0 + lgkm_cnt)
.endm


maxU24 = 1 << 24

wave_size = 64
log2 wave_size_log2, wave_size

.macro m_buffer_load_dwordx size, dst, off, desc, soff, ioff=0
    .if \size == 1
        buffer_load_dword v[\dst], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .elseif \size == 2
        buffer_load_dwordx2 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .elseif \size == 3
        buffer_load_dwordx3 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .elseif \size == 4
        buffer_load_dwordx4 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .endif
.endm

.macro m_buffer_load_ushort size, dst, off, desc, soff, ioff=0
    .if \size == 1
        buffer_load_ushort v[\dst], v[\off],  s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .endif
.endm

.macro m_buffer_store_short size, src, off, desc, soff, ioff=0
    .if \size == 1
        buffer_store_short v[\src], v[\off],  s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .endif
.endm

.macro u32_div numer, denom, uquo, v_tmp, s_tmp
    u_div   v[\numer], v[\denom] v[\uquo], \v_tmp, \s_tmp, \s_tmp + 2 
.endm


// Unsigned division function from the SC implementation
// 4 s_tmps, 4 v_tmps
//
.macro u_div numer, denom, uquo, vtmp, stmp1, stmp2
    v_cvt_f32_u32     v[\vtmp],     \denom
    v_rcp_f32         v[\vtmp],     v[\vtmp]
    v_mul_f32         v[\vtmp],     0x4f800000,   v[\vtmp]
    v_cvt_u32_f32     v[\vtmp],     v[\vtmp]

    v_mul_lo_u32      v[\vtmp+1],   \denom,       v[\vtmp]
    v_mul_hi_u32      v[\vtmp+2],   \denom,       v[\vtmp]
   _v_sub_co_u32      v[\vtmp+3],   vcc,          0,           v[\vtmp+1]
    v_cmp_ne_i32      s[\stmp1:\stmp1+1], 0,      v[\vtmp+2]
    v_cndmask_b32     v[\vtmp+1],   v[\vtmp+3],   v[\vtmp+1],  s[\stmp1:\stmp1+1]
    v_mul_hi_u32      v[\vtmp+1],   v[\vtmp+1],   v[\vtmp]
   _v_sub_co_u32      v[\vtmp+2],   vcc,          v[\vtmp],    v[\vtmp+1]
   _v_add_co_u32      v[\vtmp],     vcc,          v[\vtmp],    v[\vtmp+1]
    v_cndmask_b32     v[\vtmp],     v[\vtmp],     v[\vtmp+2],  s[\stmp1:\stmp1+1]
    v_mul_hi_u32      v[\vtmp],     v[\vtmp],     \numer
    v_mul_lo_u32      v[\vtmp+1],   v[\vtmp],     \denom
   _v_sub_co_u32      v[\vtmp+2],   vcc,          \numer,      v[\vtmp+1]
    v_cmp_ge_u32      s[\stmp1:\stmp1+1],         \numer,      v[\vtmp+1]
    v_cmp_ge_u32      s[\stmp2:\stmp2+1],         v[\vtmp+2],  \denom
   _v_add_co_u32      v[\vtmp+2],   vcc,          1,           v[\vtmp]
    s_and_b64         s[\stmp2:\stmp2+1], s[\stmp1:\stmp1+1],  s[\stmp2:\stmp2+1]
   _v_add_co_u32      v[\vtmp+1],   vcc, -1,      v[\vtmp]
    v_cndmask_b32     v[\vtmp+2],   v[\vtmp],     v[\vtmp+2],  s[\stmp2:\stmp2+1]
    v_cndmask_b32     v[\vtmp+2],   v[\vtmp+1],   v[\vtmp+2],  s[\stmp1:\stmp1+1]
    v_cmp_ne_i32      vcc,          0,            \denom
    v_cndmask_b32     \uquo,        -1,           v[\vtmp+2],  vcc
.endm

.altmacro
.macro ceil_2_32_div_u16 m, denom, vtmp, stmp
    v_cvt_f32_u32     v[\vtmp],     \denom
    v_rcp_f32         v[\vtmp],     v[\vtmp]
    v_mul_f32         v[\vtmp],     0x4f800000,   v[\vtmp]
    v_cvt_u32_f32     v[\vtmp],     v[\vtmp]

    v_mul_lo_u32      v[\vtmp+1],   \denom,       v[\vtmp]
    v_mul_hi_u32      v[\vtmp+2],   \denom,       v[\vtmp]
   _v_sub_nc_u32      v[\vtmp+3],   0,            v[\vtmp+1]
    v_cmp_ne_i32      s[\stmp:\stmp+1], 0,        v[\vtmp+2]
    v_cndmask_b32     v[\vtmp+1],   v[\vtmp+3],   v[\vtmp+1],  s[\stmp:\stmp+1]
    v_mul_hi_u32      v[\vtmp+1],   v[\vtmp+1],   v[\vtmp]
   _v_sub_nc_u32      v[\vtmp+2],   v[\vtmp],     v[\vtmp+1]
    v_add_co_u32      v[\vtmp],     vcc,          v[\vtmp],    v[\vtmp+1]
    v_cndmask_b32     v[\vtmp],     v[\vtmp],     v[\vtmp+2],  s[\stmp:\stmp+1]
    v_mul_hi_u32      v[\vtmp],     -1,           v[\vtmp]
    v_mul_lo_u32      v[\vtmp+1],   v[\vtmp],     \denom
   _v_sub_nc_u32      v[\vtmp+2],   -1,           v[\vtmp+1]
    v_cmp_ge_u32      s[\stmp:\stmp+1],           v[\vtmp+2],  \denom
    v_add_u32         v[\vtmp+2],   1,            v[\vtmp]
    v_add_co_u32      v[\vtmp+1],   vcc, -1,      v[\vtmp]
    v_cndmask_b32     v[\vtmp+2],   v[\vtmp],     v[\vtmp+2],  s[\stmp:\stmp+1]
    v_add_u32         v[\vtmp+2],   1,            v[\vtmp+2]
    v_cmp_ne_i32      vcc,          0,            \denom
    v_cndmask_b32     \m,        -1,           v[\vtmp+2],  vcc
.endm

.macro disable_srd srd
    s_mov_b32 s[\srd+3], 0
.endm
.macro enable_srd srd
    s_mov_b32 s[\srd+3], 0x00020000            // DATA_FORMAT, need to just be non-zero;
.endm

.macro label l, n
    \l\n:
.endm
.macro _s_cbranch cond, l, n
    s_cbranch_\cond \l\n
.endm
.macro _s_branch l, n
    s_branch \l\n
.endm

.macro v_reg_data_type_convert v_dst_reg, dst_type, v_src_reg, src_type, v_tmp_reg = 0x7fffffff, s2_tmp_b64 = 0x7fffffff
    .if(\src_type == TYPE_FP16 && \dst_type == TYPE_FP32)
        v_cvt_f32_f16 \v_dst_reg, \v_src_reg
    .elseif(\src_type == TYPE_FP32 && \dst_type == TYPE_FP16)
        v_cvt_f16_f32 \v_dst_reg, \v_src_reg
    .elseif(\src_type == TYPE_FP32 && \dst_type == TYPE_BFP16)
        v_mov_b32 \v_tmp_reg, 0x1
        v_cmp_class_f32 \s2_tmp_b64, \v_src_reg, \v_tmp_reg //Signaling NaN  
        s_and_saveexec_b64 \s2_tmp_b64, \s2_tmp_b64
        
        v_or_b32 \v_src_reg,  0x10000, \v_src_reg
        s_mov_b64 exec, \s2_tmp_b64        
        .ifdef MIOPEN_USE_RNE_BFLOAT16
            .if(MIOPEN_USE_RNE_BFLOAT16 == 1)
                v_mov_b32 \v_tmp_reg, 0x207
                v_cmp_class_f32 exec, \v_src_reg, \v_tmp_reg //check NANs and INFs
                
                s_xor_b64 exec, exec, \s2_tmp_b64
                v_bfe_u32 \v_tmp_reg, \v_src_reg, 16, 1
                v_add_u32 \v_tmp_reg, 0x7fff, \v_tmp_reg
                v_add_u32 \v_dst_reg, \v_src_reg, \v_tmp_reg

                s_mov_b64 exec, \s2_tmp_b64
            .endif
        .endif        
        
        v_lshrrev_b32 \v_dst_reg, 16, \v_src_reg
    .elseif(\src_type == TYPE_BFP16 && \dst_type == TYPE_FP32)
        v_lshlrev_b32 \v_dst_reg, 16, \v_src_reg
    .elseif (\dst_type != \src_type)
        .error "wrong convertion type" 
    .endif
.endm

div_const_1_2=0x80000000
div_const_1_3=0x55555556
div_const_1_4=0x40000000
div_const_1_5=0x33333334
div_const_1_6=0x2aaaaaab
div_const_1_7=0x24924925
div_const_1_8=0x20000000
div_const_1_9=0x1c71c71d
div_const_1_10=0x1999999a
div_const_1_11=0x1745d175
div_const_1_12=0x15555556
div_const_1_13=0x13b13b14
div_const_1_14=0x12492493
div_const_1_15=0x11111112
div_const_1_16=0x10000000
div_const_1_17=0x0f0f0f10
div_const_1_18=0x0e38e38f
div_const_1_19=0x0d79435f
div_const_1_20=0x0ccccccd
div_const_1_21=0x0c30c30d
div_const_1_22=0x0ba2e8bb
div_const_1_23=0x0b21642d
div_const_1_24=0x0aaaaaab
div_const_1_25=0x0a3d70a4
div_const_1_26=0x09d89d8a
div_const_1_27=0x097b425f
div_const_1_28=0x0924924a
div_const_1_29=0x08d3dcb1
div_const_1_30=0x08888889
div_const_1_31=0x08421085
div_const_1_32=0x08000000
div_const_1_33=0x07c1f07d
div_const_1_34=0x07878788
div_const_1_35=0x07507508
div_const_1_36=0x071c71c8
div_const_1_37=0x06eb3e46
div_const_1_38=0x06bca1b0
div_const_1_39=0x06906907
div_const_1_40=0x06666667
div_const_1_41=0x063e7064
div_const_1_42=0x06186187
div_const_1_43=0x05f417d1
div_const_1_44=0x05d1745e
div_const_1_45=0x05b05b06
div_const_1_46=0x0590b217
div_const_1_47=0x0572620b
div_const_1_48=0x05555556
div_const_1_49=0x0539782a
div_const_1_50=0x051eb852
div_const_1_51=0x05050506
div_const_1_52=0x04ec4ec5
div_const_1_53=0x04d4873f
div_const_1_54=0x04bda130
div_const_1_55=0x04a7904b
div_const_1_56=0x04924925
div_const_1_57=0x047dc120
div_const_1_58=0x0469ee59
div_const_1_59=0x0456c798
div_const_1_60=0x04444445
div_const_1_61=0x04325c54
div_const_1_62=0x04210843
div_const_1_63=0x04104105
div_const_1_64=0x04000000

.macro _s_div_const_u32_u16 dst, src, denum
    .if \denum == 1
        s_mov_b32 \dst, \src
    .elseif \denum >=2 && \denum <= 64
        s_mul_hi_u32 \dst, div_const_1_\denum, \src
    .else
        static_assert(0)
    .endif
.endm

.macro _v_div_const_u32_u16 dst, src, denum, tmp
    .if \denum == 1
        v_mov_b32 \dst, \src
    .elseif \denum >=2 && \denum <= 64
        s_mov_b32 \tmp, div_const_1_\denum
        v_mul_hi_u32 \dst, \tmp, \src
    .else
        static_assert(0)
    .endif
.endm

.macro _s_ceil_u32 dst, src, denum
    s_add_u32 \dst, \denum - 1, \src
   _s_div_const_u32_u16 \dst, \dst, \denum
.endm
