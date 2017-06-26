#define _FLOAT float

static inline int iDiv(int v, int d)
{
	int r = (int)((float)v / d + 0.00001f);
	return(r);
}

static inline int iMod(int v, int u, int d)
{
	int r = v - mul24((int)u, (int)d);
	return(r);
}

static inline void ReduceKernel(__local _FLOAT * lcl_mem, int sum_stride, int unit_id, int unit_len)
{
    _FLOAT sum = 0;
    int lcl_offset = unit_id * unit_len;
    for(int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

__attribute__((reqd_work_group_size(MLO_CONVBWD_GROUP_SZ0,MLO_CONVBWD_GROUP_SZ1,1)))
__kernel void MIOpenConvBwdB(
       const __global _FLOAT * top_df,
       __global _FLOAT * bias_df
	   )
{
    int lid = (int) get_local_id(0);
    int output_map = get_group_id(1);
 
    __local _FLOAT lcl_sum[MLO_CONVBWDB_LCL_MEMSZ];
    _FLOAT sum = 0;

    for (int j = lid; j < MLO_WK_SIZE * MLO_OUT_BATCH_SZ; j += MLO_CONVBWD_GROUP_SZ0)
    {
        int map_id = iDiv(j, MLO_WK_SIZE);
        int read_id = iMod(j, map_id, MLO_WK_SIZE);
        int glb_top_df_offset = output_map * MLO_OUT_CHANNEL_STRIDE + (map_id * MLO_OUT_BATCH_STRIDE) + (read_id * MLO_CONVBWDB_UNITSIZE);
#if MLO_N_PIX_OFF
        if (read_id == MLO_WK_SIZE - 1)
        {
        for (int k = 0; k < MLO_N_PIX_OFF; k++)
            sum += top_df[glb_top_df_offset + k];
        }
#else
        for (int k = 0; k < MLO_CONVBWDB_UNITSIZE; k++)
            sum += top_df[glb_top_df_offset + k];
#endif
    }
    lcl_sum[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(lid < (MLO_CONVBWD_GROUP_SZ0 >> 2))
    {
        ReduceKernel(lcl_sum, 1, lid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < (MLO_CONVBWD_GROUP_SZ0 >> 4))
    {
        ReduceKernel(lcl_sum, 4, lid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0)
    {
        ReduceKernel(lcl_sum, 16, lid, 256);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bias_df[output_map] = lcl_sum[0];
}
