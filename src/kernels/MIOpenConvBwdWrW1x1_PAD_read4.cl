
#if 0 // nef ML_OPEN_RUNNING
// W 7 x H 7 x C 2048 x K 2048
//#define MLO_GRP_SZ
#define MLO_GRP_SZ0 64
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_FILTER_SIZE0 1
#define MLO_FILTER_SIZE1 1
#define MLO_FILTER_PAD0 3
#define MLO_FILTER_PAD1 3
#define MLO_FILTER_STRIDE0 2
#define MLO_FILTER_STRIDE1 2
#define STRIDE_W 1
#define STRIDE_H 1
#define MLO_N_OUTPUTS 2048
#define MLO_N_INPUTS 1024
#define MLO_BATCH_SZ 16
//MLO_N_BATCH_LOOPS
#define MLO_IN_WIDTH 7
#define MLO_IN_HEIGHT 7
#define MLO_OUT_WIDTH 7
#define MLO_OUT_HEIGHT 7
#endif

#if 0 // nef ML_OPEN_RUNNING
// W 7 x H 7 x C 2048 x K 2048
//#define MLO_GRP_SZ
#define MLO_GRP_SZ0 256
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_FILTER_SIZE0 1
#define MLO_FILTER_SIZE1 1
#define MLO_FILTER_PAD0 0
#define MLO_FILTER_PAD1 0
#define MLO_FILTER_STRIDE0 2
#define MLO_FILTER_STRIDE1 2
#define STRIDE_W 1
#define STRIDE_H 1
#define MLO_N_OUTPUTS 2048
#define MLO_N_INPUTS 1024
#define MLO_BATCH_SZ 16
//MLO_N_BATCH_LOOPS
#define MLO_IN_WIDTH 14
#define MLO_IN_HEIGHT 14
#define MLO_OUT_WIDTH 7
#define MLO_OUT_HEIGHT 7
#endif

#if 0 // nef ML_OPEN_RUNNING
// W 14 x H 14 x C 1024 x K 512
//#define MLO_GRP_SZ
#define MLO_GRP_SZ0 256
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_FILTER_SIZE0 1
#define MLO_FILTER_SIZE1 1
#define MLO_FILTER_PAD0 0
#define MLO_FILTER_PAD1 0
#define MLO_FILTER_STRIDE0 2
#define MLO_FILTER_STRIDE1 2
#define STRIDE_W 1
#define STRIDE_H 1
#define MLO_N_OUTPUTS 512
#define MLO_N_INPUTS 1024
#define MLO_BATCH_SZ 16
//MLO_N_BATCH_LOOPS
#define MLO_IN_WIDTH 14
#define MLO_IN_HEIGHT 14
#define MLO_OUT_WIDTH 7
#define MLO_OUT_HEIGHT 7

#define MLO_LDS_REDUCTOIN 1
#define MLO_GLOBAL_ATOMIC 0
#define MLO_N_LDS_REDUCTION_ONCE 8
#define MLO_N_LDS_SIZE_PER_THREAD 8

#endif

#if 0 // nef ML_OPEN_RUNNING
// W 28 x H 28 x C 192 x K 64 X N 16
//#define MLO_GRP_SZ
#define MLO_GRP_SZ0 64
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_FILTER_SIZE0 1
#define MLO_FILTER_SIZE1 1
#define MLO_FILTER_PAD0 0
#define MLO_FILTER_PAD1 0
#define MLO_FILTER_STRIDE0 1
#define MLO_FILTER_STRIDE1 1
#define STRIDE_W 1
#define STRIDE_H 1
#define MLO_N_OUTPUTS 64
#define MLO_N_INPUTS 192
#define MLO_BATCH_SZ 16
#define MLO_IN_WIDTH 28
#define MLO_IN_HEIGHT 28
#define MLO_OUT_WIDTH 28
#define MLO_OUT_HEIGHT 28
#define MLO_LDS_REDUCTOIN 1
#define MLO_GLOBAL_ATOMIC 0

#define MLO_N_LOAD_DWORDS_PER_MAP_ONCE 64
#define MLO_N_LCL_IN_MAPS 8
#define MLO_N_LCL_OUT_MAPS 8

#define MLO_N_LCL_IN_MAPS_ONCE 8
#define MLO_N_LCL_OUT_MAPS_ONCE 8

#define MLO_READ_UNIT 2   

//READ_UNIT == 1 for STRIDE and PAD mode

#define MLO_OUT_BATCH_STRIDE (MLO_OUT_WIDTH * MLO_OUT_HEIGHT * MLO_N_OUTPUTS)
#define MLO_OUT_CHANNEL_STRIDE (MLO_OUT_WIDTH * MLO_OUT_WIDTH)

#define MLO_IN_BATCH_STRIDE (MLO_IN_WIDTH * MLO_IN_HEIGHT * MLO_N_INPUTS)
#define MLO_IN_CHANNEL_STRIDE (MLO_IN_WIDTH * MLO_IN_HEIGHT)
#define MLO_WEI_BATCH_STRIDE (MLO_N_INPUTS * MLO_N_OUTPUTS)
#define MLO_WEI_CHANNEL_STRIDE (1 * 1 * MLO_N_INPUTS)
#define MLO_MAX_LOADS ((MLO_OUT_CHANNEL_STRIDE / MLO_READ_UNIT) * MLO_BATCH_SZ)

#define MLO_ACCUM_SZ (MLO_N_LCL_IN_MAPS * MLO_N_LCL_OUT_MAPS)
#define MLO_OUT_READ_SZ (N_LCL_OUT_MAPS * MLO_READ_UNIT)
#define MLO_IN_READ_SZ (MLO_N_LCL_IN_MAPS * MLO_READ_UNIT)

#define MLO_OUT_CHANNEL_READ_SZ (MLO_OUT_CHANNEL_STRIDE / MLO_READ_UNIT)

#define MLO_N_IN_TILE_BLOCK 4
#endif

#if 0
#define MLO_READ_UNIT 4

#define MLO_OUT_BATCH_STRIDE (MLO_OUT_WIDTH * MLO_OUT_HEIGHT * MLO_N_OUTPUTS)
#define MLO_OUT_CHANNEL_STRIDE (MLO_OUT_WIDTH * MLO_OUT_WIDTH)
#define MLO_OUT_STRIDE (1)
#define MLO_IN_BATCH_STRIDE (MLO_IN_WIDTH * MLO_IN_HEIGHT * MLO_N_INPUTS)
#define MLO_IN_CHANNEL_STRIDE (MLO_IN_WIDTH * MLO_IN_HEIGHT)
#define MLO_IN_STRIDE (2)
#define MLO_WEI_BATCH_STRIDE (MLO_N_INPUTS * MLO_N_OUTPUTS)
#define MLO_WEI_CHANNEL_STRIDE (1 * 1 * MLO_N_INPUTS)

//#define MLO_N_LCL_IN_MAPS       8
//#define MLO_N_LCL_OUT_MAPS      8
#define MLO_CACHELINE_DWORD_SZ 64

#endif

#if 0
#if(MLO_FILTER_PAD0 > 0 || MLO_FILTER_PAD1 > 0)
#define MLO_IN_PAD_MIN_X0 (MLO_FILTER_STRIDE0 - (MLO_FILTER_PAD0 % MLO_FILTER_STRIDE0))
#define MLO_IN_PAD_MIN_Y0 (MLO_FILTER_STRIDE1 - (MLO_FILTER_PAD1 % MLO_FILTER_STRIDE1))

#define MLO_IN_PAD_MIN_X (MLO_IN_PAD_MIN_X0 % MLO_FILTER_STRIDE0)
#define MLO_IN_PAD_MIN_Y (MLO_IN_PAD_MIN_Y0 % MLO_FILTER_STRIDE1)

#define MLO_OUT_PAD_MIN_X ((MLO_FILTER_PAD0 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0)
#define MLO_OUT_PAD_MIN_Y ((MLO_FILTER_PAD1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)

#define MLO_OUT_PAD_WIDTH \
    (((MLO_IN_WIDTH - MLO_IN_PAD_MIN_X + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0))
#define MLO_OUT_PAD_HEIGHT \
    (((MLO_IN_HEIGHT - MLO_IN_PAD_MIN_Y + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE1))

#else
#define MLO_IN_PAD_MIN_X 0
#define MLO_IN_PAD_MIN_Y 0

#define MLO_OUT_PAD_MIN_X 0
#define MLO_OUT_PAD_MIN_Y 0

#define MLO_OUT_PAD_WIDTH MLO_OUT_WIDTH
#define MLO_OUT_PAD_HEIGHT MLO_OUT_HEIGHT
#endif
#endif

#if 0

#define MLO_N_BATCH_PER_WAVE ((MLO_BATCH_SZ + (MLO_GRP_SZ0 / 64) - 1) / (MLO_GRP_SZ0 / 64))
#define MLO_N_LOOPS_PER_MAP                                                          \
    ((MLO_OUT_PAD_WIDTH * MLO_OUT_PAD_HEIGHT + MLO_N_LOAD_DWORDS_PER_MAP_ONCE - 1) / \
     (MLO_N_LOAD_DWORDS_PER_MAP_ONCE))
#define MLO_N_DWORDS_LAST_LOAD_PER_MAP \
    ((MLO_OUT_PAD_WIDTH * MLO_OUT_PAD_HEIGHT) % (MLO_N_LOAD_DWORDS_PER_MAP_ONCE))

#define MLO_IN_NON_ALIGN (MLO_N_INPUTS & 0x7)
#define MLO_OUT_NON_ALIGN (MLO_N_OUTPUTS & 0x7)
#define MLO_IN_LAST_GROUP (MLO_N_INPUTS & (~0x7))
#define MLO_OUT_LAST_GROUP (MLO_N_OUTPUTS & (~0x7))
#define MLO_MAX_LOAD_DWORD ((MLO_OUT_CHANNEL_STRIDE / MLO_READ_UNIT) * MLO_BATCH_SZ)
#endif

// FLAT_LOAD_DWORDx4 is max
// MIN(MLO_OUT_PAD_WIDTH , 8)

#if 0

//64x64 only works MLO_OUT_PAD_WIDTH<=8
//following are trick to speed-up shader compiler for unused MIOpenCvBwdWrW_64x64
#if MLO_OUT_PAD_WIDTH > 8
#define MLO_N_LOAD_DWORD_ONCE_PER_THREAD 1
#else
#define MLO_N_DWORD_PER_LOOP (MLO_OUT_PAD_WIDTH)
#endif

#endif

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#define UNUSED __attribute__((__unused__))
#define DBG_OUT_OF_RNGE 0

__attribute__((always_inline)) uint iDiv(uint v, uint d)
{
    uint r = (uint)((float)v * (1.f / (float)d) + 0.0000000001f);
    return (r);
}

__attribute__((always_inline)) uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24(u, d);
    return (r);
}

//#define MLO_WRITE_UNIT 4
#define MLO_OUT_CHANNEL_STRIDE_ALIGNED (MLO_OUT_CHANNEL_STRIDE / MLO_WRITE_UNIT)
#define MLO_OUT_STRIDE_ALIGNED (MLO_OUT_STRIDE / MLO_WRITE_UNIT)

__attribute__((reqd_work_group_size(MLO_GRP0_SZ0, MLO_GRP0_SZ1, MLO_GRP0_SZ2))) __kernel void
MIOpenSubsample(const __global _FLOAT* __restrict in, __global _FLOAT* __restrict out)
{
    uint stack_pos = get_global_id(0);
    uint batch_id  = get_global_id(1);
    uint map_id    = iDiv(stack_pos, MLO_OUT_CHANNEL_STRIDE_ALIGNED);
    uint pix_pos   = iMod(stack_pos, map_id, MLO_OUT_CHANNEL_STRIDE_ALIGNED);
    uint out_y     = iDiv(pix_pos, MLO_OUT_STRIDE_ALIGNED);
    uint out_x     = iMod(pix_pos, out_y, MLO_OUT_STRIDE_ALIGNED) * MLO_WRITE_UNIT;

    uint out_off = batch_id * MLO_IN_BATCH_STRIDE + stack_pos * MLO_WRITE_UNIT;
    uint in_y    = out_y * MLO_FILTER0_STRIDE1;
    uint in_x    = out_x * MLO_FILTER0_STRIDE0;
    uint in_off  = batch_id * MLO_IN0_BATCH_STRIDE + map_id * MLO_IN0_CHANNEL_STRIDE +
                  in_y * MLO_IN0_STRIDE + in_x;

    const __global _FLOAT* in_ptr = &in[in_off];
    __global _FLOAT* out_ptr      = &out[out_off];

    for(uint i = 0; i < MLO_WRITE_UNIT; ++i, in_ptr += MLO_FILTER0_STRIDE0, out_ptr++)
    {
        *out_ptr = *in_ptr;
    }
}

// top_df        ==> out        in [Batch][output][out_H][out_W]
// bot           ==> gard_input in [Batch][inputs][IN_H][IN_W]
// weights_df    ==> weights    in [output][input][filter][filter]

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvBwdWrW_8x8map(const __global _FLOAT* __restrict top_df,
                      const __global _FLOAT* __restrict bot,
                      __global _FLOAT* __restrict weights_df,
                      UNUSED _FLOAT padding_val)
{
    __local _FLOAT sdata[MLO_GRP_SZ0 * 8];
    // 8x8 MUL_ADD per thread
    // Every thread 1DWORDs per MAP * 8 Maps
    // every 64 threads Load continous 64 DWORDs per MAP * 8 Maps
    // LDS reduction of every 64 threads for 8x8 results
    // NO LDS data exchange

    // Global_group_Id0:  [TILE_BLOCK]
    // Global_Id0:  [C/8] * 64 or 256
    // Global_group_Id1:  [K/8]
    // Global_group_Id2:  [C/8/ (MLO_N_IN_TILE_BLOCK= 8 or 4) ]

    // uint group_id0 = get_group_id(0);
    // uint group_id1 = get_group_id(1);

    uint local_Id0 = get_local_id(0);

// traverse small batch size to have better performance
#if MLO_IN_BATCH_STRIDE < MLO_OUT_BATCH_STRIDE
    uint C_OFFSET = get_group_id(0) * MLO_N_LCL_IN_MAPS;
    uint K_OFFSET = get_group_id(1) * MLO_N_LCL_OUT_MAPS;

#else
    uint K_OFFSET          = get_group_id(0) * MLO_N_LCL_OUT_MAPS;
    uint C_OFFSET          = get_group_id(1) * MLO_N_LCL_IN_MAPS;

#endif

    uint glb_out_off0 = K_OFFSET * MLO_OUT_CHANNEL_STRIDE;
    uint glb_in_off0  = C_OFFSET * MLO_IN_CHANNEL_STRIDE;

    // NO preload
    __private _FLOAT load_buf_top[MLO_N_LCL_OUT_MAPS * MLO_READ_UNIT];
    __private _FLOAT load_buf_bot[MLO_N_LCL_IN_MAPS * MLO_READ_UNIT];

    __private _FLOAT accum[MLO_ACCUM_SZ];

    // CNHW will be continous address to utlize X4 load;
    // NCHW will be hard mode till now

    for(uint i = 0; i < MLO_ACCUM_SZ; i++)
    {
        accum[i] = 0;
    }

    for(uint i = 0; i < MLO_N_LCL_IN_MAPS; i++)
    {
        sdata[local_Id0 + i * MLO_GRP_SZ0] = 0;
    }

    for(uint faked_off = local_Id0; faked_off < MLO_MAX_LOADS; faked_off += MLO_GRP_SZ0)
    {
#if MLO_FILTER_PAD0 > 0 || MLO_FILTER_PAD1 > 0 || \
    (!TWO_PASSES && (MLO_FILTER_STRIDE0 > 1 || MLO_FILTER_STRIDE1 > 1))

        uint batch_id = iDiv(faked_off, ((MLO_OUT_PAD_WIDTH / MLO_READ_UNIT) * MLO_OUT_PAD_HEIGHT));
        uint faked_off2 =
            iMod(faked_off, batch_id, ((MLO_OUT_PAD_WIDTH / MLO_READ_UNIT) * MLO_OUT_PAD_HEIGHT));

        uint out_y_off = iDiv(faked_off2, (MLO_OUT_PAD_WIDTH / MLO_READ_UNIT));
        uint out_x_off =
            iMod(faked_off2, out_y_off, (MLO_OUT_PAD_WIDTH / MLO_READ_UNIT)) * MLO_READ_UNIT;

        uint out_image_off =
            (out_y_off + MLO_OUT_PAD_MIN_Y) * MLO_OUT_WIDTH + (out_x_off + MLO_OUT_PAD_MIN_X);

        uint in_x_off = out_x_off * MLO_FILTER_STRIDE0 + MLO_IN_PAD_MIN_X;
        uint in_y_off = out_y_off * MLO_FILTER_STRIDE1 + MLO_IN_PAD_MIN_Y;

        uint in_image_off = in_y_off * MLO_IN_STRIDE + in_x_off;

#else
        uint batch_id      = iDiv(faked_off, (MLO_OUT_CHANNEL_READ_SZ));           // batch
        uint image_off     = iMod(faked_off, batch_id, (MLO_OUT_CHANNEL_READ_SZ)); // pixel offset
        uint in_image_off  = image_off * MLO_READ_UNIT;
        uint out_image_off = image_off * MLO_READ_UNIT;
#endif
        uint glb_in_off = glb_in_off0 + batch_id * MLO_IN_BATCH_STRIDE + in_image_off;

        // *(p+index) Pointer Mode will use OFfset mode in ASSEMBLY
        //  P[Index] will not use OFfset mode in ASSEMBLY

        const __global _FLOAT* bot1 = bot + glb_in_off;
        for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                load_buf_bot[c * MLO_READ_UNIT + i] = *(bot1 + i * MLO_FILTER_STRIDE0);
            }
            bot1 += MLO_IN_CHANNEL_STRIDE;
        }

        uint glb_out_off = glb_out_off0 + batch_id * MLO_OUT_BATCH_STRIDE + out_image_off;

        const __global _FLOAT* top1 = top_df + glb_out_off;
        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {

            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                load_buf_top[k * MLO_READ_UNIT + i] = *(top1 + i);
            }
            top1 += MLO_OUT_CHANNEL_STRIDE;
        }

        // processing
        // outside loop-i save 1 VGPR than loop-i inside
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {
                    {
                        accum[k * MLO_N_LCL_IN_MAPS + c] += load_buf_bot[c * MLO_READ_UNIT + i] *
                                                            load_buf_top[k * MLO_READ_UNIT + i];
                    }
                }
            }
        }
    }

#define LAST_PIXELS (MLO_OUT_CHANNEL_STRIDE % MLO_READ_UNIT)

// PAD/STRIDE never goes to following since MLO_READ_UNIT == 1
#if LAST_PIXELS > 0 && MLO_FILTER_PAD0 == 0 && MLO_FILTER_PAD1 == 0 && MLO_FILTER_STRIDE0 == 1 && \
    MLO_FILTER_STRIDE1 == 1
#define MLO_MAX_LOADS2 (MLO_BATCH_SZ * LAST_PIXELS)
#define MLO_LAST_PIXEL_OFFSET (MLO_OUT_CHANNEL_STRIDE - LAST_PIXELS)

    for(uint faked_off = local_Id0; faked_off < MLO_MAX_LOADS2; faked_off += MLO_GRP_SZ0)
    {

        uint batch_id = iDiv(faked_off, (LAST_PIXELS)); // batch
        uint image_off =
            iMod(faked_off, batch_id, (LAST_PIXELS)) + MLO_LAST_PIXEL_OFFSET; // pixel offset
        uint in_image_off  = image_off * 1;
        uint out_image_off = image_off * 1;

        uint glb_in_off = glb_in_off0 + batch_id * MLO_IN_BATCH_STRIDE + in_image_off;

        // *(p+index) Pointer Mode will use OFfset mode in ASSEMBLY
        //  P[Index] will not use OFfset mode in ASSEMBLY

        const __global _FLOAT* bot1 = bot + glb_in_off;

        for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
        {
            for(uint i = 0; i < 1; ++i)
            {
                load_buf_bot[c * MLO_READ_UNIT + i] = *(bot1 + i);
            }
            bot1 += MLO_IN_CHANNEL_STRIDE;
        }

        uint glb_out_off = glb_out_off0 + batch_id * MLO_OUT_BATCH_STRIDE + out_image_off;

        const __global _FLOAT* top1 = top_df + glb_out_off;
        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {
            for(uint i = 0; i < 1; ++i)
            {
                load_buf_top[k * MLO_READ_UNIT + i] = *(top1 + i);
            }
            top1 += MLO_OUT_CHANNEL_STRIDE;
        }

        // processing
        // outside loop-i save 1 VGPR than loop-i inside
        for(uint i = 0; i < 1; ++i)
        {
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {
                    {
                        accum[k * MLO_N_LCL_IN_MAPS + c] += load_buf_bot[c * MLO_READ_UNIT + i] *
                                                            load_buf_top[k * MLO_READ_UNIT + i];
                    }
                }
            }
        }
    }
#endif

    __private _FLOAT accum_to_store = 0;

    for(uint K = 0; K < MLO_N_LCL_OUT_MAPS; K++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint C = 0; C < MLO_N_LCL_IN_MAPS; C++)
        {
            sdata[local_Id0 + MLO_GRP_SZ0 * C] = accum[K * MLO_N_LCL_IN_MAPS + C];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduction every MLO_GRP_SZ0* MLO_GRP_SZ1 to trhead 0
        for(uint s = ((MLO_GRP_SZ0) >> 2); s > 0; s = (s >> 2))
        {
            if(local_Id0 < s)
            {
                for(uint C = 0; C < MLO_N_LCL_IN_MAPS; C++)
                {
                    sdata[local_Id0 + MLO_GRP_SZ0 * C] +=
                        sdata[MLO_GRP_SZ0 * C + local_Id0 + s] +
                        sdata[MLO_GRP_SZ0 * C + local_Id0 + 2 * s] +
                        sdata[MLO_GRP_SZ0 * C + local_Id0 + 3 * s];
                }
            }
            // NO need inside 1 wave: barrier(CLK_LOCAL_MEM_FENCE);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // MLO_N_LCL_IN_MAPS store
        if((local_Id0 & ~0x7) == (K * MLO_N_LCL_IN_MAPS))
        {
            accum_to_store = sdata[0 + (local_Id0 & 0x7) * MLO_GRP_SZ0];
        }

        // only 1st wave need to barrier: it will remove all scratch registers
    }

    if(local_Id0 < (MLO_ACCUM_SZ))
    {

        // Store to Memory
        __global _FLOAT* __restrict weights_ptr =
            weights_df + (K_OFFSET + (local_Id0 / MLO_N_LCL_IN_MAPS)) * MLO_WEI_CHANNEL_STRIDE +
            C_OFFSET + (local_Id0 % MLO_N_LCL_IN_MAPS);

        {
            __global _FLOAT* weights_ptr2 = weights_ptr;
            *weights_ptr2                 = accum_to_store;
        }
    }
}

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvBwdWrW_16x16map(const __global _FLOAT* __restrict top_df,
                        const __global _FLOAT* __restrict bot,
                        __global _FLOAT* __restrict weights_df,
                        UNUSED _FLOAT padding_val)
{
    __local _FLOAT sdata[MLO_GRP_SZ0 * 8];
    // 64 threds split into 4 grpoups: every 16 threads accumulate 8x8
    //
    // 256 threads split inot 4 groups : every 64 threads accumulate 8x8
    //

    // Global_group_Id0:  [TILE_BLOCK]
    // Global_Id0:  [C/16] * 64 or 256
    // Global_group_Id1:  [K/16]
    // Global_group_Id2:  [C/16/ (MLO_N_IN_TILE_BLOCK= 8 or 4) ]

    // uint group_id0 = get_group_id(0);
    // uint group_id1 = get_group_id(1);

    uint local_Id0 = get_local_id(0);

// traverse small batch size to have better performance
#if MLO_IN_BATCH_STRIDE < MLO_OUT_BATCH_STRIDE
    uint C_OFFSET = get_group_id(0) * MLO_N_LCL_IN_MAPS;
    uint K_OFFSET = get_group_id(1) * MLO_N_LCL_OUT_MAPS;

#else
    uint K_OFFSET          = get_group_id(0) * MLO_N_LCL_OUT_MAPS;
    uint C_OFFSET          = get_group_id(1) * MLO_N_LCL_IN_MAPS;

#endif

    // Split into 4 groups for C[0,1,0,1], K [0,0,1,1]
    uint k_offset2 = ((local_Id0 / (MLO_GRP_SZ0 / 4)) / 2) * MLO_N_LCL_OUT_MAPS_ONCE;
    uint c_offset2 = ((local_Id0 / (MLO_GRP_SZ0 / 4)) % 2) * MLO_N_LCL_IN_MAPS_ONCE;

    uint glb_out_off0 = (K_OFFSET + k_offset2) * MLO_OUT_CHANNEL_STRIDE;
    uint glb_in_off0  = (C_OFFSET + c_offset2) * MLO_IN_CHANNEL_STRIDE;

    // NO preload
    __private _FLOAT load_buf_top[MLO_N_LCL_OUT_MAPS_ONCE * MLO_READ_UNIT];
    __private _FLOAT load_buf_bot[MLO_N_LCL_IN_MAPS_ONCE * MLO_READ_UNIT];

    __private _FLOAT accum[MLO_ACCUM_SZ];

    // CNHW will be continous address to utlize X4 load;
    // NCHW will be hard mode till now

    for(uint i = 0; i < MLO_ACCUM_SZ; i++)
    {
        accum[i] = 0;
    }

    for(uint i = 0; i < MLO_N_LCL_OUT_MAPS_ONCE; i++)
    {
        sdata[local_Id0 + i * MLO_GRP_SZ0] = 0;
    }

    for(uint faked_off = (local_Id0 % (MLO_GRP_SZ0 / 4)); faked_off < MLO_MAX_LOADS;
        faked_off += (MLO_GRP_SZ0 / 4))
    {
#if MLO_FILTER_PAD0 > 0 || MLO_FILTER_PAD1 > 0 || \
    (!TWO_PASSES && (MLO_FILTER_STRIDE0 > 1 || MLO_FILTER_STRIDE1 > 1))

#if 1 // MLO_READ_UNIT == 1
        uint batch_id = iDiv(faked_off, ((MLO_OUT_PAD_WIDTH / MLO_READ_UNIT) * MLO_OUT_PAD_HEIGHT));
        uint faked_off2 =
            iMod(faked_off, batch_id, ((MLO_OUT_PAD_WIDTH / MLO_READ_UNIT) * MLO_OUT_PAD_HEIGHT));

        uint out_y_off = iDiv(faked_off2, (MLO_OUT_PAD_WIDTH / MLO_READ_UNIT));
        uint out_x_off =
            iMod(faked_off2, out_y_off, (MLO_OUT_PAD_WIDTH / MLO_READ_UNIT)) * MLO_READ_UNIT;

        uint out_image_off =
            (out_y_off + MLO_OUT_PAD_MIN_Y) * MLO_OUT_WIDTH + (out_x_off + MLO_OUT_PAD_MIN_X);

        uint in_x_off = out_x_off * MLO_FILTER_STRIDE0 + MLO_IN_PAD_MIN_X;
        uint in_y_off = out_y_off * MLO_FILTER_STRIDE1 + MLO_IN_PAD_MIN_Y;

        uint in_image_off = in_y_off * MLO_IN_STRIDE + in_x_off;
#endif
#if 0 // PER_ROW which will be enabled after SGPR offset is enabled.
        uint batch_id   = iDiv( faked_off,  (MLO_OUT_PAD_WIDTH )); 
        uint faked_off2 = iMod( faked_off,  batch_id, (MLO_OUT_PAD_WIDTH )); 

        uint out_x_off = 0;
        uint out_y_off = faked_off2;

        uint out_image_off = (out_y_off + MLO_OUT_PAD_MIN_Y) * MLO_OUT_WIDTH + (out_x_off +MLO_OUT_PAD_MIN_X);
        
        uint in_x_off = out_x_off * MLO_FILTER_STRIDE0 + MLO_IN_PAD_MIN_X;
        uint in_y_off = out_y_off * MLO_FILTER_STRIDE1 + MLO_IN_PAD_MIN_Y;
        
        uint in_image_off  = in_y_off * MLO_IN_WIDTH + in_x_off;

        //uint glb_in_off  = glb_in_off0  + batch_id * MLO_IN_BATCH_STRIDE   + in_image_off ;
        //uint glb_out_off = glb_out_off0 + batch_id * MLO_OUT_BATCH_STRIDE  + out_image_off;

#endif

#else

        uint batch_id      = iDiv(faked_off, (MLO_OUT_CHANNEL_READ_SZ));           // batch
        uint image_off     = iMod(faked_off, batch_id, (MLO_OUT_CHANNEL_READ_SZ)); // pixel offset
        uint in_image_off  = image_off * MLO_READ_UNIT;
        uint out_image_off = image_off * MLO_READ_UNIT;
#endif
        uint glb_in_off = glb_in_off0 + batch_id * MLO_IN_BATCH_STRIDE + in_image_off;

        // *(p+index) Pointer Mode will use OFfset mode in ASSEMBLY
        //  P[Index] will not use OFfset mode in ASSEMBLY

        const __global _FLOAT* bot1 = bot + glb_in_off;

        for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                load_buf_bot[c * MLO_READ_UNIT + i] = *(bot1 + i * MLO_FILTER_STRIDE0);
            }
            bot1 += MLO_IN_CHANNEL_STRIDE;
        }

        uint glb_out_off = glb_out_off0 + batch_id * MLO_OUT_BATCH_STRIDE + out_image_off;

        const __global _FLOAT* top1 = top_df + glb_out_off;

        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS_ONCE; ++k)
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                load_buf_top[k * MLO_READ_UNIT + i] = *(top1 + i);
            }
            top1 += MLO_OUT_CHANNEL_STRIDE;
        }

        // processing
        // outside loop-i save 1 VGPR than loop-i inside
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS_ONCE; ++k)
            {
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
                {
                    // for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        accum[k * MLO_N_LCL_IN_MAPS_ONCE + c] +=
                            load_buf_bot[c * MLO_READ_UNIT + i] *
                            load_buf_top[k * MLO_READ_UNIT + i];
                    }
                }
            }
        }
    }

#undef LAST_PIXELS
#undef MLO_MAX_LOADS2
#undef MLO_LAST_PIXEL_OFFSET

#define LAST_PIXELS (MLO_OUT_CHANNEL_STRIDE % MLO_READ_UNIT)

// PAD/STRIDE never goes to LAST_PIXELS
#if LAST_PIXELS > 0 && MLO_FILTER_PAD0 == 0 && MLO_FILTER_PAD1 == 0 && MLO_FILTER_STRIDE0 == 1 && \
    MLO_FILTER_STRIDE1 == 1
#define MLO_MAX_LOADS2 (MLO_BATCH_SZ * LAST_PIXELS)
#define MLO_LAST_PIXEL_OFFSET (MLO_OUT_CHANNEL_STRIDE - LAST_PIXELS)

    for(uint faked_off = (local_Id0 % (MLO_GRP_SZ0 / 4)); faked_off < MLO_MAX_LOADS2;
        faked_off += (MLO_GRP_SZ0 / 4))
    {
        uint batch_id  = iDiv(faked_off, (LAST_PIXELS));           // batch
        uint image_off = iMod(faked_off, batch_id, (LAST_PIXELS)); // pixel offset
        image_off += MLO_LAST_PIXEL_OFFSET;

        uint glb_in_off = glb_in_off0 + batch_id * MLO_IN_BATCH_STRIDE + image_off * 1;

        // *(p+index) Pointer Mode will use OFfset mode in ASSEMBLY
        //  P[Index] will not use OFfset mode in ASSEMBLY

        const __global _FLOAT* bot1 = bot + glb_in_off;
        for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
        {
            for(uint i = 0; i < 1; ++i)
            {
                load_buf_bot[c * MLO_READ_UNIT + i] = *(bot1 + i);
            }
            bot1 += MLO_IN_CHANNEL_STRIDE;
        }

        uint glb_out_off = glb_out_off0 + batch_id * MLO_OUT_BATCH_STRIDE + image_off * 1;

        const __global _FLOAT* top1 = top_df + glb_out_off;

        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS_ONCE; ++k)
        {
            for(uint i = 0; i < 1; ++i)
            {
                load_buf_top[k * MLO_READ_UNIT + i] = *(top1 + i);
            }
            top1 += MLO_OUT_CHANNEL_STRIDE;
        }

        // processing
        // outside loop-i save 1 VGPR than loop-i inside
        for(uint i = 0; i < 1; ++i)
        {
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS_ONCE; ++k)
            {
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
                {
                    {
                        accum[k * MLO_N_LCL_IN_MAPS_ONCE + c] +=
                            load_buf_bot[c * MLO_READ_UNIT + i] *
                            load_buf_top[k * MLO_READ_UNIT + i];
                    }
                }
            }
        }
    }
#endif

    __private _FLOAT accum_to_store[4];
    accum_to_store[0] = 0;
    accum_to_store[1] = 0;
    accum_to_store[2] = 0;
    accum_to_store[3] = 0;

    // 1 loop reduction 8xC per thread, 32 result per workgroup

    for(uint K = 0; K < MLO_N_LCL_OUT_MAPS_ONCE; K++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint C = 0; C < MLO_N_LCL_IN_MAPS_ONCE; C++)
        {
            sdata[local_Id0 + MLO_GRP_SZ0 * C] = accum[K * MLO_N_LCL_IN_MAPS_ONCE + C];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduction every MLO_GRP_SZ0* MLO_GRP_SZ1 to trhead 0 inside 1 wave
        for(uint s = (MLO_GRP_SZ0 >> 2); s >= 4; s = (s >> 2))
        {
            // Every time reduce to 1/4
            // Final: offset 0, 1, 2, 3 has the accumualte value
            if(local_Id0 < s)
            {
                for(uint C = 0; C < MLO_N_LCL_IN_MAPS_ONCE; C++)
                {
                    // reduce to final 4x
                    sdata[local_Id0 + MLO_GRP_SZ0 * C] =
                        sdata[local_Id0 * 4 + 0 + MLO_GRP_SZ0 * C] +
                        sdata[local_Id0 * 4 + 1 + MLO_GRP_SZ0 * C] +
                        sdata[local_Id0 * 4 + 2 + MLO_GRP_SZ0 * C] +
                        sdata[local_Id0 * 4 + 3 + MLO_GRP_SZ0 * C];
                }
            }
            // NO need inside 1 wave: barrier(CLK_LOCAL_MEM_FENCE);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // MLO_N_LCL_IN_MAPS store 32 DWORD once
        if((local_Id0 & ~0x7) == (K * (MLO_N_LCL_IN_MAPS_ONCE)))
        {
            accum_to_store[0] = sdata[0 + (local_Id0 & 0x7) * MLO_GRP_SZ0];
            accum_to_store[1] = sdata[1 + (local_Id0 & 0x7) * MLO_GRP_SZ0];
            accum_to_store[2] = sdata[2 + (local_Id0 & 0x7) * MLO_GRP_SZ0];
            accum_to_store[3] = sdata[3 + (local_Id0 & 0x7) * MLO_GRP_SZ0];
        }

        // only 1st wave need to barrier: it will remove all scratch registers
    }

    if(local_Id0 < (MLO_ACCUM_SZ))
    {

        // Store to Memory
        __global _FLOAT* __restrict weights_ptr =
            weights_df +
            (K_OFFSET + (local_Id0 / MLO_N_LCL_IN_MAPS_ONCE)) * MLO_WEI_CHANNEL_STRIDE + C_OFFSET +
            (local_Id0 % MLO_N_LCL_IN_MAPS_ONCE);

        {
            __global _FLOAT* weights_ptr1 = weights_ptr;
            for(uint k = 0; k < 2; k++)
                for(uint c = 0; c < 2; c++)
                {
                    __global _FLOAT* weights_ptr2 =
                        weights_ptr1 + k * MLO_N_LCL_IN_MAPS_ONCE * MLO_WEI_CHANNEL_STRIDE +
                        c * MLO_N_LCL_IN_MAPS_ONCE;
                    *weights_ptr2 = accum_to_store[k * 2 + c];
                }
        }
    }
}
