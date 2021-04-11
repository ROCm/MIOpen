/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

.if ROCM_METADATA_VERSION == 4
.hsa_code_object_version 2,1
.hsa_code_object_isa
.endif

.text
.globl miopenGcnAsmConv1x1U_stride2
.p2align 8
.type miopenGcnAsmConv1x1U_stride2,@function

.if ROCM_METADATA_VERSION == 4
.amdgpu_hsa_kernel miopenGcnAsmConv1x1U_stride2
.endif

.include "rocm_version.inc"
.include "gpr_alloc.inc"
.include "utilities.inc"
.include "conv_common.inc"
.include "inst_wrappers.inc"

// initial state:
// s[0:1] - kernarg address
// s2 - wg x (HW)
// s3 - wg y (K)
// s4 - wg z (N)
kernarg = 0
gid_x = 2
gid_y = 3
gid_z = 4
gid_hw = gid_x
gid_k = gid_y
gid_n = gid_z

// kernarg layout:
// dwords 0:4 - n, c, H, W, k
// dwords 5:7 - not used
// dwords 8:9 - input buffer pointer
// dwords 10:11 - weights pointer
// dwords 12:13 - output buffer pointer

.set in_ptr_off, 0x20
.set wei_ptr_off, 0x28
.set out_ptr_off, 0x30
.set unused_dbg_ptr_off, 0x38
.set KERNEL_ARGUMENTS_SIZE, unused_dbg_ptr_off + 8

maxU24 = 1 << 24
maxU31 = 1 << 31
invalid_addr_lit = 0x7FFFFFFF

static_assert (input_buffer_size < maxU31)
static_assert (filter_buffer_size < maxU31)
static_assert (output_buffer_size < maxU31)

static_assert (pad_h == 0 && pad_w == 0)
static_assert ((stride_h == 2 || stride_h == 1) && stride_w == stride_h)
static_assert (wei_h == 1 && wei_w == 1)
strided_in_w = (img_w + stride_w - 1) / stride_w
strided_in_h = (img_h + stride_h - 1) / stride_h
gid_hw_size = ( (strided_in_w + w_per_wave - 1) / w_per_wave) * ( (strided_in_h + h_per_wave - 1) / h_per_wave)

static_assert ( (acc_type == TYPE_FP32 && buf_type == TYPE_FP32) )


// perf params
default chunk_size, 16 // 1..2^n
default dwords_per_ld, 2 // 1, 2, 3, 4
default k_mult, 1 // 1..32
default c_mult, 1 // 1..32
default n_mult, 1 // 1..32
default h_mult, 1 // 1..32
default w_mult, 1 // 1..32
default h_per_chunk, 1
default lds_limit, .MAX_LDS / 8
default data_prefetch 2
default waves_k_in_group 1
default waves_c_in_group 1

// chunk parameters
static_assert (h_per_chunk & (h_per_chunk - 1) == 0)
static_assert (chunk_size & (chunk_size - 1) == 0)
static_assert (h_per_chunk <= chunk_size)
static_assert (waves_k_in_group * waves_c_in_group <= 16)

max_scalar_load_X = 16
log2 max_scalar_load_X_log2, max_scalar_load_X

strided_in_img_w = (img_w + stride_w - 1) / stride_w
strided_in_img_h = (img_h + stride_h - 1) / stride_h

dwords_per_element = 1
elements_in_dword = 1

.if(buf_type != TYPE_FP32)
    static_assert(0)
.endif

n_per_gpr = wave_size / chunk_size
w_per_chunk = (chunk_size / h_per_chunk)

elements_per_ld = (dwords_per_ld * elements_in_dword)
active_elements= (elements_per_ld + stride_w - 1) / stride_w
w_active_per_ld = active_elements

h_chunk_step_mod = 0
.if(h_chunk_step_mod == 1)
    in_out_h_step =  1
    h_per_chunk_element_stride = h_mult
.else
    in_out_h_step = h_per_chunk //or 1
    h_per_chunk_element_stride = 1 // or h_mult
.endif

w_per_item = w_active_per_ld * w_mult

w_per_wave = w_per_chunk * w_per_item
h_per_wave = h_per_chunk * h_mult
n_per_wave = n_mult * n_per_gpr
k_per_wave = k_mult * 1   //feature

c_per_wave_tmp = ( (input_channels / c_mult ) / waves_c_in_group)
.if( ( (input_channels / c_mult)  - (c_per_wave_tmp * (waves_c_in_group - 1))) > (waves_c_in_group - 1)  &&  (c_per_wave_tmp * waves_c_in_group) != (input_channels / c_mult) )
    c_per_wave_tmp = c_per_wave_tmp + 1
.endif
c_per_wave = c_per_wave_tmp * c_mult
c_per_last_wave = input_channels - (c_per_wave * (waves_c_in_group - 1 ))

waves_per_xDim = (strided_in_img_w + w_per_wave - 1) / w_per_wave

n_per_group = n_per_wave * 1    // n_waves_cnt
k_per_group = k_per_wave * waves_k_in_group    // k_waves_cnt

in_gprs_chunk_off = dwords_per_ld
in_gprs_h_offset = w_mult * in_gprs_chunk_off
in_gprs_channel_offset =  h_mult * in_gprs_h_offset
in_gprs_batch_offset = c_mult * in_gprs_channel_offset
in_gprs_image_offset = n_mult * in_gprs_batch_offset
in_gprs = in_gprs_image_offset * data_prefetch

acc_w_offset = 1
acc_h_offset = acc_w_offset * w_mult * active_elements
acc_k_offset = h_mult * acc_h_offset
acc_n_offset = acc_k_offset * k_mult
accums_cnt = acc_n_offset * n_mult
filter_sgprs = c_mult * k_mult * data_prefetch

lds_per_group = 0
prefetch_LDS = 1
.if waves_c_in_group > 1
    lds_per_wave = lds_limit / (waves_c_in_group - 1) / waves_k_in_group
    lds_gprs_per_wave = lds_per_wave / (4 * .WAVE_SIZE)
    static_assert(lds_gprs_per_wave > 0)
    .if lds_gprs_per_wave >= accums_cnt
        sync_loops = 1
        lds_gprs_per_loop = accums_cnt
        lds_per_group = lds_gprs_per_loop * 4 * .WAVE_SIZE * (waves_c_in_group - 1) * waves_k_in_group
    .else
        lds_gprs_per_loop = lds_gprs_per_wave / 2
        sync_loops = (accums_cnt + lds_gprs_per_loop - 1) / lds_gprs_per_loop
        lds_per_group = 2 * lds_gprs_per_loop * 4 * .WAVE_SIZE * (waves_c_in_group - 1) * waves_k_in_group
        prefetch_LDS = 2
    .endif
    static_assert(lds_gprs_per_loop > 0)
    lds_reg_offset = 4 * .WAVE_SIZE
    lds_c_offset   = lds_reg_offset * lds_gprs_per_loop
    lds_pref_offset = lds_c_offset * (waves_c_in_group - 1)
    lds_k_offset   =  lds_pref_offset * prefetch_LDS
.endif

active_mask = -1
active_mask_lo = active_mask & 0xFFFFFFFF
active_mask_hi = (active_mask >> 32) & 0xFFFFFFFF


static_assert (input_n_stride * (batch_size + n_per_wave) < maxU31)
static_assert (output_n_stride * (batch_size + n_per_wave) < maxU31)

.GPR_ALLOC_BEGIN
    .SGPR_ALLOC_FROM 5
    .SGPR_ALLOC soffset_in
    .SGPR_ALLOC soffset_out
    .SGPR_ALLOC soffset_wei
    .SGPR_ALLOC desc_in, 4 // input buffer descriptor
    .SGPR_ALLOC desc_out, 4 // weights buffer descriptor
    .SGPR_ALLOC desc_wei, 4 // output buffer descriptor
    .SGPR_ALLOC filter_storage, filter_sgprs
    .SGPR_ALLOC wave_c_id // wave_c_id in group
    .SGPR_ALLOC wave_k_id // wave_k_id in group
    .SGPR_ALLOC loop_cnt
    .SGPR_ALLOC stmp_offset
    .SGPR_ALLOC stmp
    //xnack disabled by default
    //.SGPR_RESERVE_XNACK
    .SGPR_RESERVE_VCC

    .VGPR_ALLOC_FROM 0
    .VGPR_ALLOC tid
    .VGPR_ALLOC vtid_h
    .VGPR_ALLOC voffset_in
    .VGPR_ALLOC voffset_out
    .VGPR_ALLOC input_storage, in_gprs
    .if(idilation_h > 1 )
        store_buffer_size = 4
        .if(dwords_per_ld == 1)
            store_buffer_size = 2
        .endif
        .if((in_gprs - 1) < store_buffer_size)
            .VGPR_ALLOC input_storage_ex, store_buffer_size - (in_gprs - 1)
        .endif
    .elseif (gid_hw_size >= 0x10000 && in_gprs < 2)
        .VGPR_ALLOC input_storage_ex
    .endif
    .VGPR_ALLOC accums, accums_cnt
    .VGPR_ALLOC vtmp
    .if(gid_hw_size >= 0x10000)
        .VGPR_ALLOC vtmp_ex
    .endif

    .LDS_ALLOC_FROM 0
    .LDS_ALLOC accums_lds, lds_per_group

.GPR_ALLOC_END

max_waves_per_CU = (256 / .AUTO_VGPR_COUNT) * 4
static_assert( max_waves_per_CU >= waves_c_in_group * waves_k_in_group)


miopenGcnAsmConv1x1U_stride2:
.if ROCM_METADATA_VERSION == 4
    .amd_kernel_code_t
     enable_sgpr_kernarg_segment_ptr = 1
     enable_sgpr_workgroup_id_x = 1
     enable_sgpr_workgroup_id_y = 1
     enable_sgpr_workgroup_id_z = 1
     is_ptr64 = 1
     granulated_workitem_vgpr_count = .AUTO_VGPR_GRANULATED_COUNT
     granulated_wavefront_sgpr_count = .AUTO_SGPR_GRANULATED_COUNT
     enable_vgpr_workitem_id = 1
     user_sgpr_count = 2
     kernarg_segment_byte_size = 64
     wavefront_sgpr_count = .AUTO_SGPR_COUNT
     workitem_vgpr_count = .AUTO_VGPR_COUNT
     float_mode = 192
     workgroup_group_segment_byte_size = .AUTO_LDS_BYTE_SIZE
    .end_amd_kernel_code_t
.endif

    s_load_dwordx2 s[desc_in:desc_in+1], s[kernarg:kernarg+1], 0x0 + in_ptr_off
    s_load_dwordx2 s[desc_wei:desc_wei+1], s[kernarg:kernarg+1], 0x0 + wei_ptr_off
    s_load_dwordx2 s[desc_out:desc_out+1], s[kernarg:kernarg+1], 0x0 + out_ptr_off

    // mask off unused lanes
    s_mov_b32 exec_lo, active_mask_lo
    s_mov_b32 exec_hi, active_mask_hi

    // fill format and size fields of buffer descriptors
    static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
    s_mov_b32 s[desc_in+2], input_buffer_size
    s_mov_b32 s[desc_in+3], 0x00027000
    s_mov_b32 s[desc_wei+2], filter_buffer_size
    s_mov_b32 s[desc_wei+3], 0x00027000
    s_mov_b32 s[desc_out+2], output_buffer_size
    s_mov_b32 s[desc_out+3], 0x00027000

    .macro store_waveId_in_sgpr tid_v, tmp_v, res_s
        v_lshrrev_b32 v[\tmp_v], 6, v[\tid_v]
        v_readfirstlane_b32 s[\res_s], v[\tmp_v]
    .endm

    //up to 2^18 vals
    .macro div_vgpr_const_fp64 a_src, v2_a_dest, a_modval, v2_tmp
        v_cvt_f64_u32   v[\v2_a_dest:\v2_a_dest+1], s[\a_src]
        v_cvt_f64_u32   v[\v2_tmp:\v2_tmp+1],  0 + \a_modval
        v_add_f64       v[\v2_a_dest:\v2_a_dest+1], 0.5, v[\v2_a_dest:\v2_a_dest+1]
        v_rcp_f64       v[\v2_tmp:\v2_tmp+1],  v[\v2_tmp:\v2_tmp+1]
        v_mul_f64       v[\v2_a_dest:\v2_a_dest+1], v[\v2_a_dest:\v2_a_dest+1], v[\v2_tmp:\v2_tmp+1]
        v_trunc_F64     v[\v2_a_dest:\v2_a_dest+1], v[\v2_a_dest:\v2_a_dest+1]
        v_cvt_u32_f64   v[\v2_a_dest], v[\v2_a_dest:\v2_a_dest+1]
    .endm

    .macro div_vgpr_const_fp32 a_src, a_dest, a_modval, v_tmp
        v_cvt_f32_u32   v[\a_dest], s[\a_src]
        v_cvt_f32_u32   v[\v_tmp],  0 + \a_modval
        v_add_f32       v[\a_dest], 0x3f000000, v[\a_dest]
        v_rcp_f32       v[\v_tmp],  v[\v_tmp]
        v_mul_f32       v[\a_dest], v[\a_dest], v[\v_tmp]
        v_trunc_F32     v[\a_dest], v[\a_dest]
        v_cvt_u32_f32   v[\a_dest], v[\a_dest]
    .endm


    .macro div_vgpr_const a_src, a_dest, a_modval, v_tmp, check_a_src_size = 0
        .if(\check_a_src_size >= 0x10000 || \a_modval >= 0x10000)
            div_vgpr_const_fp64 \a_src, \a_dest, \a_modval, \v_tmp
        .else
            div_vgpr_const_fp32 \a_src, \a_dest, \a_modval, \v_tmp
        .endif
    .endm

    store_waveId_in_sgpr tid, vtmp, wave_c_id

    div_vgpr_const wave_c_id, vtmp, waves_c_in_group, accums

    v_readfirstlane_b32 s[wave_k_id], v[vtmp]
    s_mul_i32 s[stmp], s[wave_k_id], 0 + waves_c_in_group
    s_sub_u32 s[wave_c_id], s[wave_c_id], s[stmp]

    v_and_b32 v[tid], 0x3f, v[tid]

    log2 chunk_size_log2, chunk_size
    // calculate input/output offsets
    v_lshrrev_b32 v[vtmp],        0 + chunk_size_log2, v[tid] // vtmp = wave part id
    .if( (input_n_stride < maxU24) & (output_n_stride < maxU24))
        v_mul_u32_u24 v[voffset_in] , 0 + input_n_stride,  v[vtmp]
        v_mul_u32_u24 v[voffset_out], 0 + output_n_stride, v[vtmp]
    .else
        v_mov_b32 v[voffset_in] , 0 + input_n_stride
        v_mul_lo_u32 v[voffset_in] , v[voffset_in],  v[vtmp]
        v_mov_b32 v[voffset_out] , 0 + output_n_stride
        v_mul_lo_u32 v[voffset_out], v[voffset_out], v[vtmp]
    .endif

    log2 w_per_chunk_log2, w_per_chunk
    log2 h_per_chunk_log2, h_per_chunk

    .GPR_REUSE loop_cnt, s_val_in_tmp_stride
    .GPR_REUSE stmp_offset, s_val_out_tmp_stride

    s_mov_b32 s[s_val_out_tmp_stride], 0 + output_h_stride * idilation_h
    s_mov_b32 s[s_val_in_tmp_stride] , 0 + input_h_stride * stride_h


    .macro get_global_Htid v_dest, v_group_id, v_tmp_reg, v_tid
        // v_group_id = group_id_h
        v_bfe_u32 v[\v_dest], v[\v_tid], 0 + w_per_chunk_log2, 0 + h_per_chunk_log2
        // v_dest = Item.h_img_id in wave part

        .if(h_per_chunk_element_stride != 1)
            v_mul_u32_u24   v[\v_dest],  0 + h_per_chunk_element_stride,  v[\v_dest]
        .endif

        v_mul_u32_u24 v[\v_tmp_reg], 0 + h_per_wave, v[\v_group_id]
        // v_tmp_reg = Group.img_h in chanel

        _v_add_nc_u32 v[\v_dest], v[\v_dest], v[\v_tmp_reg]
        // v_dest = Item.h_img_id in chanel
    .endm

    .macro get_global_Wtid v_dest, v_group_id, v_tmp_reg, v_tid
        v_bfe_u32 v[\v_dest], v[\v_tid], 0, 0 + w_per_chunk_log2
        // v_dest = Item.w_img_id in wave part

        v_mul_u32_u24 v[\v_tmp_reg], 0 + waves_per_xDim, v[\v_group_id]
        _v_sub_nc_u32 v[\v_tmp_reg], s[gid_hw], v[\v_tmp_reg]
        //v_tmp_reg = group_id_w

        v_mul_u32_u24 v[\v_tmp_reg], 0 + w_per_wave, v[\v_tmp_reg]
        // v_tmp_reg = group_id_w * w_per_wave = group_w
        v_mul_u32_u24 v[\v_dest], 0 + w_active_per_ld, v[\v_dest]
        _v_add_nc_u32 v[\v_dest], v[\v_dest], v[\v_tmp_reg]

        // v_dest = Item.w_img_id in chanel
    .endm

    .GPR_REUSE tid, vtid_w
    .macro get_gloabl_H_W_tids h_dest, w_dest, v_temps, v_tid
        //gid_hw = group_id_w +  group_id_h * group_id_x_cnt
        div_vgpr_const gid_hw, vtmp, waves_per_xDim, \v_temps +1 , gid_hw_size
        //vtmp = group_id_h

        get_global_Htid \h_dest, vtmp, \v_temps, \v_tid
        get_global_Wtid \w_dest, vtmp, \v_temps,  \v_tid
    .endm

    get_gloabl_H_W_tids vtid_h, vtid_w, input_storage, vtid_w


    // add h_offset
    V_MAD_U32_U24 v[voffset_out], s[s_val_out_tmp_stride], v[vtid_h], v[voffset_out]
    V_MAD_U32_U24 v[voffset_in] , s[s_val_in_tmp_stride],  v[vtid_h], v[voffset_in]

    // add w_offset
    static_assert(input_w_stride * stride_w <= 64)
    V_MAD_U32_U24 v[voffset_in], 0 + input_w_stride * stride_w, v[vtid_w], v[voffset_in]
    V_MAD_U32_U24 v[voffset_out], 0 + output_w_stride * idilation_w, v[vtid_w], v[voffset_out]

    // add n_offset
    s_mul_i32 s[soffset_in],  s[gid_n], 0 + input_n_stride * n_per_wave
    s_mul_i32 s[soffset_out], s[gid_n], 0 + output_n_stride * n_per_wave

    // add k_offset
    s_mul_i32 s[gid_k], s[gid_k], 0 + k_per_group
    s_mul_i32 s[stmp], s[wave_k_id], 0 + k_per_wave
    s_add_i32 s[gid_k], s[gid_k], s[stmp]
    s_mul_i32 s[stmp], s[gid_k], 0 + output_k_stride
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]

    s_mul_i32 s[soffset_wei], s[gid_k], 0 + filter_k_stride

    //add c_offset
    s_mul_i32 s[stmp], s[wave_c_id], 0 + c_per_wave * filter_c_stride
    s_add_u32 s[soffset_wei], s[soffset_wei], s[stmp]

    s_mul_i32 s[stmp], s[wave_c_id], 0 + c_per_wave * input_c_stride
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]

    s_waitcnt 0

    .GPR_REUSE s_val_in_tmp_stride, loop_cnt
    .GPR_REUSE s_val_out_tmp_stride, stmp_offset

    .altmacro
    .macro chunk_symb_set_val_wrapper symb_size, set_val
        x\symb_size\()_chunks = \set_val
    .endm

    .macro chunk_symb_get_val_wrapper symb_size, ret
        \ret = x\symb_size\()_chunks
    .endm

    .macro chunk_symb_set_val symb_size, set_val
        chunk_symb_set_val_wrapper %\symb_size, %\set_val
    .endm

    .macro chunk_symb_get_val symb_size, ret
        xxol = 0
        chunk_symb_get_val_wrapper %\symb_size, xxol
        \ret = xxol
    .endm

    .macro init_ld_meta_data
        .if(weights_layout == 0)
            filter_c_gpr_stride = 1
            filter_k_gpr_stride = (c_mult + elements_in_dword - 1) / elements_in_dword
            filter_ld_dim0_size = (c_mult + elements_in_dword - 1) / elements_in_dword
            static_assert(input_channels % filter_ld_dim0_size == 0)
            filter_ld_dim1_size = k_mult
            sequential_read_stride = filter_k_stride
        .else
            filter_c_gpr_stride = (k_mult + elements_in_dword - 1) / elements_in_dword
            filter_k_gpr_stride = 1
            filter_ld_dim0_size = (k_mult + elements_in_dword - 1) / elements_in_dword
            static_assert(output_channels % filter_ld_dim0_size == 0)
            filter_ld_dim1_size = c_mult
            sequential_read_stride = filter_c_stride
        .endif


        size_mod\@ = max_scalar_load_X
        rest\@ = filter_ld_dim0_size
        cur_sizeMod_cnt\@ = 0
        .rept  (max_scalar_load_X_log2 + 1)
            cur_sizeMod_cnt\@ = rest\@ / size_mod\@
            chunk_symb_set_val size_mod\@, cur_sizeMod_cnt\@
            rest\@ = rest\@ - cur_sizeMod_cnt\@ * size_mod\@
            size_mod\@ = size_mod\@ / 2
        .endr

        w_mult_ld_stride_byte = active_elements * stride_w * input_w_stride * w_per_chunk

        mbufs_cnt = c_mult * n_mult * w_mult * h_mult
    .endm

    init_ld_meta_data

    .macro get_filter_sgpr_regId_2dimP arg_dim0_id, arg_dim1_id, arg_prefetch_id, ret_regId

        static_assert(\arg_prefetch_id < data_prefetch)

        size_mod\@ = max_scalar_load_X
        id_dim1_size\@ = filter_ld_dim1_size
        temp_id\@ = \arg_dim0_id
        size_mod_cnt\@ = 0
        tmp_reg_id\@ = 0
        .rept (max_scalar_load_X_log2 + 1)
            .if(temp_id\@ != -1)
                chunk_symb_get_val size_mod\@, size_mod_cnt\@
                bank_size\@ = (size_mod_cnt\@) * (size_mod\@)
                .if(temp_id\@ >= (bank_size\@))
                    temp_id\@ = temp_id\@ - (bank_size\@)
                    tmp_reg_id\@ = tmp_reg_id\@ + (bank_size\@) * data_prefetch * id_dim1_size\@
                .else
                    tmp_reg_id\@ = tmp_reg_id\@ + (bank_size\@ * (id_dim1_size\@ * \arg_prefetch_id + \arg_dim1_id)) + temp_id\@
                    temp_id\@ = -1
                .endif
                size_mod\@ = size_mod\@ / 2
            .endif
        .endr
        \ret_regId = tmp_reg_id\@ + filter_storage
    .endm

    .macro get_filter_sgpr_regId_CKP arg_c_id, arg_k_id, arg_prefetch_id, ret_regId
        static_assert(\arg_c_id < c_mult)
        static_assert(\arg_k_id < k_mult)
        static_assert(\arg_prefetch_id < data_prefetch)
        .if(weights_layout == 0)
            get_filter_sgpr_regId_2dimP \arg_c_id, \arg_k_id, \arg_prefetch_id, \ret_regId
        .else
            get_filter_sgpr_regId_2dimP \arg_k_id, \arg_c_id, \arg_prefetch_id, \ret_regId
        .endif
    .endm

    .macro xsload base, xx, cnt
        .rept \xx
            .if \cnt == 1
                s_buffer_load_dword s[\base], s[desc_wei:desc_wei+3], s[soffset_wei]
            .else
                s_buffer_load_dwordx\cnt s[\base:\base+\cnt-1], s[desc_wei:desc_wei+3], s[soffset_wei]
            .endif
            \base = \base + \cnt
            s_add_u32 s[soffset_wei], s[soffset_wei], 0+4*\cnt
        .endr
    .endm

    .macro load_filters arg_prefetch_id
        seq_dimX\@   = filter_ld_dim0_size
        seq_dimY\@   = filter_ld_dim1_size
        seq_stride\@ = sequential_read_stride
        fbase   = 0
        dimY_it = 0
        .rept seq_dimY\@
            imm_off = 0
            dimX_it = 0
            size_it = max_scalar_load_X
            size_it_cnt = 0
            .rept (max_scalar_load_X_log2 + 1)
                chunk_symb_get_val size_it, size_it_cnt
                get_filter_sgpr_regId_2dimP dimX_it, dimY_it, \arg_prefetch_id, fbase

                xsload fbase, size_it_cnt, %size_it

                dimX_it = dimX_it + (size_it_cnt * size_it)
                size_it = size_it / 2
            .endr

            dimY_it = dimY_it + 1
            .if(weights_layout == 0 && dimY_it == seq_dimY\@)
                s_add_i32 s[soffset_wei], s[soffset_wei], 0 - seq_stride\@ * (seq_dimY\@ - 1)
            .else
                s_add_i32 s[soffset_wei], s[soffset_wei], 0 + seq_stride\@ - 4 * seq_dimX\@
            .endif
        .endr
    .endm

    .macro get_input_vgpr_regId_all seq_W, seq_H, seq_C, seq_N, prefetch_id, ret_regId
        static_assert(\seq_W < (w_mult * dwords_per_ld))
        static_assert(\seq_H < (h_mult))
        static_assert(\seq_C < (c_mult))
        static_assert(\seq_N < (n_mult))
        static_assert(\prefetch_id < data_prefetch)
        \ret_regId = input_storage + (\prefetch_id * in_gprs_image_offset) + (\seq_N * in_gprs_batch_offset) + (\seq_C * in_gprs_channel_offset) + (in_gprs_h_offset * \seq_H) + \seq_W
    .endm

    .macro get_input_vgpr_regId_active seq_W, seq_H, seq_C, seq_N, prefetch_id, ret_regId
        static_assert(\seq_W < (w_per_item))
        static_assert(\seq_H < (h_mult))
        static_assert(\seq_C < (c_mult))
        static_assert(\seq_N < (n_mult))
        static_assert(\prefetch_id < data_prefetch)
        w_mult_id\@ = \seq_W / active_elements
        w_act_id\@ = \seq_W % active_elements
        w_all_id\@ = (w_act_id\@ * stride_w) + (w_mult_id\@ * in_gprs_chunk_off)
        get_input_vgpr_regId_all w_all_id\@, \seq_H \seq_C, \seq_N, \prefetch_id, \ret_regId
    .endm

    .macro ioffset_as_soffset_conv_2pow12 i_val, s_ptr, surpl
        s_add_u32 s[\s_ptr], s[\s_ptr], 0 + \i_val
        \surpl = \surpl + \i_val
    .endm

    .macro load_input prefetch_id
        ibase\@ = 0
        nb\@ = 0
        .rept n_mult
            c_it\@ = 0
            s_mov_b32 s[stmp_offset], s[soffset_in]
            .rept c_mult
                s_cmpk_le_i32 s[loop_cnt], 0 + c_it\@
                s_cmov_b32 s[stmp_offset], 0 + invalid_addr_lit
                ld_h_it\@ = 0
                .rept h_mult
                    ld_w_it\@ = 0
                    imm_off\@ = 0
                    s_offset_surplus\@ = 0

                    .rept w_mult
                        get_input_vgpr_regId_all ld_w_it\@, ld_h_it\@, c_it\@, nb\@, \prefetch_id, ibase\@
                        .if( (imm_off\@ - s_offset_surplus\@) >= 4096)
                            ioffset_as_soffset_conv_2pow12 (imm_off\@ - s_offset_surplus\@), stmp_offset, s_offset_surplus\@
                        .endif
                        m_buffer_load_dwordx dwords_per_ld, ibase\@, voffset_in, desc_in, stmp_offset, (imm_off\@ - s_offset_surplus\@)
                        imm_off\@ = imm_off\@ + w_mult_ld_stride_byte
                        ld_w_it\@ = ld_w_it\@ + dwords_per_ld
                    .endr
                    ld_h_it\@ = ld_h_it\@ + 1
                    .if(ld_h_it\@ < h_mult)
                        s_add_u32 s[stmp_offset], s[stmp_offset], 0 + input_h_stride * stride_h * in_out_h_step - s_offset_surplus\@
                    .else
                        .if(h_mult > 1)
                            s_sub_i32 s[stmp_offset], s[stmp_offset], 0 + input_h_stride * stride_h * in_out_h_step * (h_mult - 1) + s_offset_surplus\@
                        .elseif (s_offset_surplus\@ > 0)
                            s_sub_i32 s[stmp_offset], s[stmp_offset], 0 + s_offset_surplus\@
                        .endif
                    .endif
                .endr

                c_it\@ = c_it\@ + 1
                s_add_u32 s[stmp_offset], s[stmp_offset], 0 +input_c_stride
            .endr
            nb\@ = nb\@ + 1

            .if nb\@ == n_mult
                s_add_i32 s[soffset_in], s[soffset_in], 0 + (input_c_stride * c_mult) - input_n_stride * n_per_gpr * (n_mult - 1)
            .else
                s_add_i32 s[soffset_in], s[soffset_in], 0 + input_n_stride * n_per_gpr
            .endif
        .endr

        s_addk_i32 s[loop_cnt], 0 - (1 * c_mult)
        .if (1)
            s_cmpk_le_i32 s[loop_cnt], 0
            s_cmov_b32 s[desc_in+2], 0
        .endif
    .endm

    .macro get_acc_idx acc, k, n, h, chunk
        static_assert(\chunk < (w_mult * active_elements))
        static_assert(\k < (k_mult))
        static_assert(\h < (h_mult))
        static_assert(\n < (n_mult))
        \acc = accums + acc_k_offset * \k + \n * acc_n_offset + \h * acc_h_offset + \chunk
    .endm

    .macro conv prefetch_id
        c\@ = 0
        .rept c_mult
            k\@ = 0
            .rept k_mult
                nb\@ = 0
                .rept n_mult
                    h_it\@ = 0
                    .rept h_mult
                        chunk = 0
                        .rept w_per_item
                            inp_gprc\@ = 0
                            f_gpr\@   = 0
                            acc\@     = 0
                            get_acc_idx acc\@, k\@, nb\@, h_it\@, chunk
                            get_filter_sgpr_regId_CKP c\@, k\@, \prefetch_id, f_gpr\@
                            get_input_vgpr_regId_active chunk, h_it\@, c\@, nb\@, \prefetch_id, inp_gpr\@

                            .if acc_type == TYPE_FP32 && buf_type == TYPE_FP32 && vec_c_in == 1
                                v_mac_f32 v[acc\@], s[f_gpr\@], v[inp_gpr\@]
                            .endif

                            chunk = chunk + 1
                        .endr
                        h_it\@ = h_it\@ + 1
                    .endr
                    nb\@ = nb\@+ 1
                .endr
                k\@ = k\@ + 1
            .endr
            c\@ = c\@ + 1
        .endr
    .endm

    s_mov_b32 s[loop_cnt], 0 + c_per_wave
    s_cmpk_eq_u32 s[wave_c_id], 0 + waves_c_in_group - 1
    s_cmov_b32 s[loop_cnt], 0 + c_per_last_wave

    load_input 0

    load_filters  0

    // zeroing accums
    i = 0
    .rept accums_cnt
        v_mov_b32 v[accums + i], 0
        i = i + 1
    .endr
    wave_sync_allow = 0
    .macro wave_sync_mainLoop step
        .if (\step == 1 && waves_k_in_group * waves_c_in_group > 1 && wave_sync_allow == 1)
            .if(waves_k_in_group > 1)
                S_BITCMP1_B32 s[wave_k_id], 0x0
            .else
                S_BITCMP1_B32 s[wave_c_id], 0x0
            .endif
                s_cbranch_scc1 barrier1_skip
            s_barrier
            barrier1_skip:
        .elseif (\step == 2 && waves_k_in_group * waves_c_in_group > 1 && wave_sync_allow == 1)
            s_barrier
        .elseif (\step == 3 && waves_k_in_group * waves_c_in_group > 1 && wave_sync_allow == 1)
            .if(waves_k_in_group > 1)
                S_BITCMP0_B32 s[wave_k_id], 0x0
            .else
                S_BITCMP0_B32 s[wave_c_id], 0x0
            .endif
            s_cbranch_scc1 barrier3_skip
            s_barrier
            barrier3_skip:
            .if(c_per_last_wave != c_per_wave && waves_c_in_group > 1)
                .if(c_per_wave > c_per_last_wave )
                    s_cmpk_eq_u32 s[wave_c_id], 0 + waves_c_in_group - 1
                    s_cmov_b32 s[loop_cnt], 0 + (c_per_wave - c_per_last_wave + c_mult - 1) / c_mult
                .elseif (c_per_last_wave > c_per_wave )
                    s_cmpk_lg_u32 s[wave_c_id], 0 + waves_c_in_group - 1
                    s_cmov_b32 s[loop_cnt], 0 + (c_per_last_wave - c_per_wave + c_mult - 1) / c_mult
                .endif
                s_cbranch_scc0 barrier4_skip

                    barrier4_loop:
                    s_sub_u32 s[loop_cnt], s[loop_cnt], 1
                    s_barrier
                    s_cmpk_gt_i32 s[loop_cnt], 1
                    s_cbranch_scc1 barrier4_loop
                barrier4_skip:
            .endif
        .endif
    .endm

    wave_sync_mainLoop 1

loop_begin:
    load_input 1
    wave_sync_mainLoop 2
    s_wait mbufs_cnt, 0
    load_filters  1

    conv 0

    load_input 0
    wave_sync_mainLoop 2
    s_wait mbufs_cnt, 0
    load_filters  0

    conv 1

loop_end:
    s_cmpk_gt_i32 s[loop_cnt], 1 * c_mult
    s_cbranch_scc1 loop_begin

    load_input 1
    wave_sync_mainLoop 2
    s_wait mbufs_cnt, 0
    load_filters  1

    conv 0
    wave_sync_mainLoop 2
    s_waitcnt 0

    conv 1
    wave_sync_mainLoop 3

    .macro lds_c_reduction
        .if waves_c_in_group > 1
            //reuse
            lds_off = voffset_in

            //restore lane_id
            v_mbcnt_lo_u32_b32 v[lds_off], -1, 0
            v_mbcnt_hi_u32_b32 v[lds_off], -1, v[lds_off]
            s_mul_i32 s[stmp_offset], s[wave_k_id], 0 + lds_k_offset
            s_mul_i32 s[stmp], s[wave_c_id], 0 + lds_c_offset

            v_lshlrev_b32 v[lds_off], 2, v[lds_off]
            _v_add_nc_u32 v[lds_off], s[stmp_offset], v[lds_off]

            s_mov_b32 m0, -1
            s_cmpk_eq_u32 s[wave_c_id], 0 + waves_c_in_group - 1
            s_cbranch_scc1 last_wave

            _v_add_nc_u32 v[lds_off], s[stmp], v[lds_off]

            acc_id = 0
            sync_loop = 0

            .rept sync_loops
                imm_off = (sync_loop % 2) * lds_pref_offset
                .rept lds_gprs_per_loop
                    .if acc_id < accums_cnt
                        ds_write_b32 v[lds_off], v[accums + acc_id], offset:0+imm_off
                    .endif
                    acc_id = acc_id + 1
                    imm_off = imm_off + 4 * .WAVE_SIZE
                .endr
                s_waitcnt 0
                s_barrier
                sync_loop = sync_loop + 1
            .endr

        s_endpgm
        last_wave:
            acc_id = 0
            sync_loop = 0
            .rept sync_loops
                s_barrier
                gpr = 0
                .rept lds_gprs_per_loop
                    wave_c = 0
                    .rept waves_c_in_group-1
                        imm_off = gpr * lds_reg_offset + wave_c * lds_c_offset + (sync_loop % 2) * lds_pref_offset
                        .if acc_id < accums_cnt
                            ds_read_b32 v[vtmp], v[lds_off] offset:0+imm_off
                            s_waitcnt 0
                            .if acc_type == TYPE_FP32
                                v_add_f32 v[accums + acc_id], v[vtmp], v[accums + acc_id]
                            .elseif acc_type == TYPE_INT32
                                _v_add_nc_u32 v[accums + acc_id], v[vtmp], v[accums + acc_id]
                            .endif
                        .endif
                        wave_c = wave_c + 1
                    .endr
                    acc_id = acc_id + 1
                    gpr = gpr + 1
                .endr
                sync_loop = sync_loop + 1
            .endr
        .GPR_INVALIDATE lds_off
        .endif
    .endm

    lds_c_reduction

    .macro fill_dil_buffer elements_cnt, ptr
        it_fill_dil\@ = 0
        .rept \elements_cnt
             v_mov_b32 v[acc_dil_buff + it_fill_dil\@ * idilation_w ], v[\ptr + it_fill_dil\@]
             it_fill_dil\@ = it_fill_dil\@ + 1
        .endr
    .endm

    .macro reset_dil_buffer cnt
        it_reset\@ = 0
        .rept \cnt
            v_mov_b32 v[acc_dil_buff + it_reset\@], 0
            it_reset\@ = it_reset\@ + 1
        .endr
    .endm

     .macro m_buffer_store_dwordx size, src, off, desc, soff, ioff=0
        .if \size == 1
            buffer_store_dword v[\src], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
        .elseif \size == 2
            buffer_store_dwordx2 v[\src:\src+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
        .elseif \size == 3
            buffer_store_dwordx3 v[\src:\src+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
        .elseif \size == 4
            buffer_store_dwordx4 v[\src:\src+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
        .elseif \size == 0

        .else
            .error "m_buffer_store_dwordx unknown size"
        .endif
    .endm

    .macro buffer_stor_elements cnt, acc_ptr, v_offset, s_desc, s_offset, val_offset, s_offset_surplus=0
        .if(buf_type == TYPE_FP32)
            acc_ptr_\@ = \acc_ptr
            acc_cnt_\@ = \cnt
            it_acc\@ = 0
            reps = (\cnt + 3) / 4
            .rept reps
                .if((\cnt - it_acc\@) > 4)
                    acc_cnt_\@ = 4
                .else
                    acc_cnt_\@ = (\cnt - it_acc\@)
                .endif
                acc_ptr_\@ = \acc_ptr + it_acc\@

                .if(idilation_w > 1 && \acc_ptr != -1)
                    nonzeros_cnt\@ = (acc_cnt_\@ + idilation_w - 1) / idilation_w
                    acc_off\@ = \acc_ptr + (it_acc\@ + idilation_w - 1) / idilation_w
                    fill_dil_buffer nonzeros_cnt\@, acc_off\@
                    acc_ptr_\@ = acc_dil_buff
                .elseif (idilation_w > 1 && \acc_ptr == -1)
                    acc_ptr_\@ = acc_dil_buff
                .endif

                i_off\@ = \val_offset + it_acc\@ * 4 - \s_offset_surplus
                .if(i_off\@ >= 4096)
                    ioffset_as_soffset_conv_2pow12 (\val_offset - \s_offset_surplus), \s_offset, \s_offset_surplus
                    i_off\@ = \val_offset + it_acc\@ * 4 - \s_offset_surplus
                .endif
                m_buffer_store_dwordx acc_cnt_\@, acc_ptr_\@, \v_offset, \s_desc, \s_offset, i_off\@
                it_acc\@ = it_acc\@ + acc_cnt_\@
            .endr
        .endif
    .endm

    .macro get_store_acc_idx acc, k, nb, h, chunk
        .if(\h % idilation_h == 0)
            get_acc_idx \acc, \k, \nb, (\h / idilation_h), \chunk
        .else
            \acc = -1
        .endif
    .endm

    .GPR_REUSE voffset_in, v_offset_part
    .if(idilation_w > 1)
        acc_dil_buff = input_storage + 1
        reset_dil_buffer store_buffer_size
    .endif
    .GPR_REUSE input_storage, v_offset_single

    s_vcc_st_sing = desc_in + 2
    .GPR_REUSE desc_in, s_vcc_st_part

    s_save_exec = desc_wei + 2
    s_val_rem_out_range_part = desc_wei + 1
    .GPR_REUSE desc_wei, s_val_rem_out_range_sing

    .macro store_result is_full_w
        h_mult_dilated = h_mult * idilation_h
        .if(h_mult_dilated == 1)
            v_cmpx_gt_i32 vcc, 0 + out_h , v[vtid_h]
        .endif

        s_mov_b32 s[s_save_exec], exec_lo
        s_mov_b32 s[s_save_exec + 1], exec_hi
        active_elements_dilated = active_elements * idilation_w
        .if(\is_full_w == 0)
            rem_w_out = out_w - (waves_per_xDim - 1) * w_per_wave * idilation_w
            full_w_mult\@ = (rem_w_out) / (active_elements_dilated * w_per_chunk)
            rem_w_out_part = rem_w_out - full_w_mult\@ * active_elements_dilated * w_per_chunk
            single_out_size = rem_w_out_part % (active_elements_dilated)

            s_mov_b32 s[s_val_rem_out_range_part], 0 + (rem_w_out_part - single_out_size) + (waves_per_xDim - 1) * w_per_wave * idilation_w
            s_mov_b32 s[s_val_rem_out_range_sing], 0 + (rem_w_out_part ) + (waves_per_xDim - 1) * w_per_wave * idilation_w

            v_cmp_lt_i32 s[s_vcc_st_part:s_vcc_st_part+1], v[vtid_w],  s[s_val_rem_out_range_part]
            v_cmp_lt_i32 s[s_vcc_st_sing:s_vcc_st_sing+1], v[vtid_w],  s[s_val_rem_out_range_sing]
            s_xor_b64 s[s_vcc_st_sing:s_vcc_st_sing+1], s[s_vcc_st_sing:s_vcc_st_sing+1], s[s_vcc_st_part:s_vcc_st_part+1]

            //calculate v_offset_part, v_offset_single
            v_mov_b32 v[vtmp], 0 + invalid_addr_lit

            V_CNDMASK_B32 v[v_offset_part], v[vtmp], v[voffset_out], s[s_vcc_st_part:s_vcc_st_part+1]
            V_CNDMASK_B32 v[v_offset_single], v[vtmp], v[voffset_out], s[s_vcc_st_sing:s_vcc_st_sing+1]
        .else
            full_w_mult\@ = w_mult
        .endif

        k\@ = 0
        .rept k_mult
            nb\@ = 0
            s_cmpk_ge_i32 s[gid_k], 0 + output_channels - k\@
            s_cmov_b32 s[desc_out+2], 0
            .rept n_mult
                h_it\@ = 0
                .rept h_mult_dilated
                    .if((idilation_h > 1) && (h_it\@ % idilation_h == 1))
                        reset_dil_buffer store_buffer_size
                    .endif
                    .if(h_mult_dilated > 1)
                        h_dil_hi\@ = h_it\@ / idilation_h
                        h_dil_lo\@  = h_it\@ % idilation_h
                        v_cmpx_gt_i32 vcc, 0 + out_h - (h_dil_hi\@ * in_out_h_step * idilation_h + h_dil_lo\@), v[vtid_h]
                    .endif
                    chunk = 0
                    soffset_surplus = 0
                    .rept full_w_mult\@
                        get_store_acc_idx acc, k\@, nb\@,  h_it\@, chunk * active_elements
                        ioffset = 0 + 4 * chunk * active_elements_dilated * w_per_chunk
                        buffer_stor_elements active_elements_dilated, acc, voffset_out, desc_out, soffset_out, ioffset, soffset_surplus
                        chunk = chunk + 1
                    .endr
                    .if(\is_full_w == 0)
                        .if(rem_w_out_part != 0)
                            get_store_acc_idx acc, k\@, nb\@,  h_it\@, chunk * active_elements
                            ioffset = 0+4 * chunk * active_elements_dilated * w_per_chunk
                            buffer_stor_elements active_elements_dilated, acc , v_offset_part, desc_out, soffset_out, ioffset, soffset_surplus

                            buffer_stor_elements single_out_size, acc , v_offset_single, desc_out, soffset_out, ioffset, soffset_surplus
                        .endif
                    .endif
                    h_it\@ = h_it\@ + 1
                    .if(h_mult_dilated > 1)
                        zero_h_offset = output_h_stride
                        normal_h_offset = output_h_stride * in_out_h_step * idilation_h
                        .if(h_it\@ < h_mult_dilated)
                            .if( (h_it\@ % idilation_h) != 0)
                                s_add_u32 s[soffset_out], s[soffset_out], 0 + zero_h_offset - soffset_surplus
                            .else
                                s_add_u32 s[soffset_out], s[soffset_out], 0 + normal_h_offset - zero_h_offset * (idilation_h - 1) - soffset_surplus
                            .endif
                        .else
                            s_sub_u32 s[soffset_out], s[soffset_out], 0 + normal_h_offset * (h_mult - 1) + zero_h_offset * (idilation_h - 1) + soffset_surplus
                        .endif
                    .elseif (soffset_surplus != 0)
                        s_sub_u32 s[soffset_out], s[soffset_out], 0 + soffset_surplus
                    .endif
                .endr
                .if(h_mult_dilated > 1)
                    s_mov_b32 exec_lo, s[s_save_exec]
                    s_mov_b32 exec_hi, s[s_save_exec + 1]
                .endif
                nb\@ = nb\@ + 1
                .if nb\@ == n_mult
                    s_add_u32 s[soffset_out], s[soffset_out], 0 + output_k_stride - n_per_gpr * output_n_stride * (n_mult - 1)
                .else
                    s_add_u32 s[soffset_out], s[soffset_out], 0 + n_per_gpr * output_n_stride
                .endif
            .endr
            k\@ = k\@ + 1
        .endr
    .endm

    .macro store_results_untyped
        .if(waves_per_xDim > 1 || waves_per_xDim * w_per_wave * idilation_w == out_w)
            .if( waves_per_xDim * w_per_wave * idilation_w != out_w)
                v_cmp_le_i32 vcc, 0 + (waves_per_xDim - 1) * w_per_wave * idilation_w , v[vtid_w]

                s_cbranch_vccnz not_full_w
            .endif
            full_w:
            store_result 1

            s_endpgm
        .endif

        .if(waves_per_xDim * w_per_wave * idilation_w != out_w)
            not_full_w:
            store_result 0
        .endif
    .endm

    v_mul_u32_u24 v[vtid_w], 0 + idilation_w, v[vtid_w]
    v_mul_u32_u24 v[vtid_h], 0 + idilation_w, v[vtid_h]

    store_results_untyped

s_endpgm

.Lfunc_end0:
    .size miopenGcnAsmConv1x1U_stride2, .Lfunc_end0 - miopenGcnAsmConv1x1U_stride2

waves_in_group = waves_c_in_group * waves_k_in_group
workgroup_size_x = waves_in_group * 64

.if ROCM_METADATA_VERSION == 5
.rodata
.p2align 6
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_stepping == 10)
.amdhsa_kernel miopenGcnAsmConv1x1U_stride2
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 1
        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
        .amdhsa_accum_offset ((.AUTO_VGPR_COUNT + 4 - 1) / 4) * 4
.end_amdhsa_kernel
.else
.amdhsa_kernel miopenGcnAsmConv1x1U_stride2
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 1
        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
.end_amdhsa_kernel
.endif

.altmacro
.macro METADATA sc,vc,wg_x,lds_sz,kernarg_size
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: miopenGcnAsmConv1x1U_stride2
    .symbol: miopenGcnAsmConv1x1U_stride2.kd
    .sgpr_count: \sc
    .vgpr_count: \vc
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: \kernarg_size
    .kernarg_segment_align: 8
    .group_segment_fixed_size: \lds_sz
    .private_segment_fixed_size: 0
    .reqd_workgroup_size: [ \wg_x, 1, 1 ]
    .max_flat_workgroup_size: \wg_x
    .wavefront_size: 64
    .args:
    - { .size: 4, .offset:  0, .value_kind: by_value, .value_type: i32, .name: N }
    - { .size: 4, .offset:  4, .value_kind: by_value, .value_type: i32, .name: C }
    - { .size: 4, .offset:  8, .value_kind: by_value, .value_type: i32, .name: H }
    - { .size: 4, .offset: 12, .value_kind: by_value, .value_type: i32, .name: W }
    - { .size: 4, .offset: 16, .value_kind: by_value, .value_type: i32, .name: K }
    - { .size: 4, .offset: 20, .value_kind: by_value, .value_type: i32, .name: n_groups }
    - { .size: 4, .offset: 24, .value_kind: by_value, .value_type: i32, .name: unused_0 }
    - { .size: 4, .offset: 28, .value_kind: by_value, .value_type: i32, .name: unused_1 }
    - { .size: 8, .offset: 32, .value_kind: global_buffer, .value_type: f32, .name: x,  .address_space: global, .is_const: true }
    - { .size: 8, .offset: 40, .value_kind: global_buffer, .value_type: f32, .name: w,  .address_space: global, .is_const: true }
    - { .size: 8, .offset: 48, .value_kind: global_buffer, .value_type: f32, .name: y,  .address_space: global, .is_const: false }
    - { .size: 8, .offset: 56, .value_kind: global_buffer, .value_type: i32, .name: unused_dbg_ptr, .address_space: global, .is_const: false }
...
.end_amdgpu_metadata
.endm // METADATA

METADATA %.AUTO_SGPR_COUNT, %.AUTO_VGPR_COUNT, %workgroup_size_x, %.AUTO_LDS_BYTE_SIZE, %KERNEL_ARGUMENTS_SIZE

.elseif ROCM_METADATA_VERSION == 4
.macro METADATA wg_x, lds_size
    .amd_amdgpu_hsa_metadata
    { Version: [ 1, 0 ],
        Kernels:
        - { Name: miopenGcnAsmConv1x1U_stride2, SymbolName: 'miopenGcnAsmConv1x1U_stride2@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
            Attrs:
              { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
            CodeProps:
              { KernargSegmentSize: 64, GroupSegmentFixedSize: \lds_size, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: \wg_x }
            Args:
            - { Name: N       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: C       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: H       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: W       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: K       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: n_groups, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: unused_0, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: unused_1, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: x       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Name: w       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Name: y       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default }
            - { Name: ret_addr, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: I32, TypeName: 'int*'  , AddrSpaceQual: Global, AccQual: Default }
          }
    }
    .end_amd_amdgpu_hsa_metadata
.endm

.altmacro
.macro METADATA_WRAPPER wg_x, lds_size
    METADATA %\wg_x, %\lds_size
.endm

METADATA_WRAPPER workgroup_size_x, .AUTO_LDS_BYTE_SIZE
.endif
