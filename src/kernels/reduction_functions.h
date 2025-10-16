// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_REDUCTION_FUNCTIONS_H
#define GUARD_REDUCTION_FUNCTIONS_H

#if !MIOPEN_USE_AMDGCN
static inline void lds_reduce2(_FLOAT_ACCUM* x,
                               _FLOAT_ACCUM* y,
                               _FLOAT_ACCUM scale,
                               local _FLOAT_ACCUM* lcl_data_x,
                               local _FLOAT_ACCUM* lcl_data_y,
                               uint lid)
{
    lcl_data_x[lid] = (_FLOAT_ACCUM)*x;
    lcl_data_y[lid] = (_FLOAT_ACCUM)*y;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_LDS_SIZE >> 1); red > 0; red >>= 1)
    {
        if(lid < red)
        {
            lcl_data_x[lid] += lcl_data_x[lid + red];
            lcl_data_y[lid] += lcl_data_y[lid + red];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    *x = (_FLOAT_ACCUM)(lcl_data_x[0] * scale);
    *y = (_FLOAT_ACCUM)(lcl_data_y[0] * scale);
}

static inline void ReduceKernel(local _FLOAT_ACCUM* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{
    _FLOAT_ACCUM sum        = (_FLOAT_ACCUM)0.;
    unsigned int lcl_offset = unit_id * unit_len;

    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

static inline void
regLDSreduce(_FLOAT_ACCUM* value, local _FLOAT_ACCUM* data, uint localID, _FLOAT_ACCUM scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}

#endif

static inline void lds_reduce2_2d(_FLOAT_ACCUM_C* x,
                                  _FLOAT_ACCUM_C* y,
                                  _FLOAT_ACCUM scale,
                                  local _FLOAT_ACCUM_C* lcl_data,
                                  unsigned int xstride,
                                  unsigned int xlid,
                                  unsigned int ylid,
                                  unsigned int size)
{
    unsigned int offset1 = 2 * (xlid + ylid * xstride);
    // store the values by pairs (so the compiler will generate
    // one instruction to read/write them)
    lcl_data[offset1 + 0] = (_FLOAT_ACCUM_C)*x;
    lcl_data[offset1 + 1] = (_FLOAT_ACCUM_C)*y;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (size >> 1); red > 0; red >>= 1)
    {
        if(ylid < red)
        {
            unsigned int offset2 = offset1 + red * xstride * 2;
            // make sure there is one read and one write
            *x += lcl_data[offset2 + 0];
            *y += lcl_data[offset2 + 1];
            lcl_data[offset1 + 0] = *x;
            lcl_data[offset1 + 1] = *y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    *x = (_FLOAT_ACCUM_C)(lcl_data[xlid * 2 + 0] * scale);
    *y = (_FLOAT_ACCUM_C)(lcl_data[xlid * 2 + 1] * scale);
}

#if MIOPEN_USE_AMDGCN
static inline void dpp_reduction(_FLOAT_ACCUM* temp_sum)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "s_nop 1\n"
                     : "=v"(*temp_sum)
                     : "0"(*temp_sum));
}

static inline void dpp_interleaved_reduction(_FLOAT_ACCUM* temp_sum1, _FLOAT_ACCUM* temp_sum2)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                     "s_nop 0"
                     : "=v"(*temp_sum1), "=v"(*temp_sum2)
                     : "0"(*temp_sum1), "1"(*temp_sum2));
}

static inline void
dpp_triple_reduction(_FLOAT_ACCUM* temp_sum1, _FLOAT_ACCUM* temp_sum2, _FLOAT_ACCUM* temp_sum3)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %2 %2 %2 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %2 %2 %2 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %2 %2 %2row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %2 %2 %2 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %2 %2 %2 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                     "v_add_f32 %2 %2 %2 row_bcast:31 row_mask:0xc\n"
                     : "=v"(*temp_sum1), "=v"(*temp_sum2), "=v"(*temp_sum3)
                     : "0"(*temp_sum1), "1"(*temp_sum2), "2"(*temp_sum3));
}

static inline void gcn_reduce2(_FLOAT_ACCUM* x,
                               _FLOAT_ACCUM* y,
                               _FLOAT_ACCUM scale,
                               local _FLOAT_ACCUM* lcl_data_x,
                               local _FLOAT_ACCUM* lcl_data_y,
                               uint lid)
{
    unsigned int ldsidx = lid >> 6;
    dpp_interleaved_reduction(x, y);
    if((lid % 64) == 63)
    {
        lcl_data_x[ldsidx] = *x;
        lcl_data_y[ldsidx] = *y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    *x = *y = (_FLOAT_ACCUM)0.;
    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        *x += lcl_data_x[i];
        *y += lcl_data_y[i];
    }
    *x *= scale;
    *y *= scale;
}

#endif // MIOPEN_USE_AMDGCN

#endif // GUARD_REDUCTION_FUNCTIONS_H
