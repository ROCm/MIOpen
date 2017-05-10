

#define _FLOAT					float
#define _FLOAT2					float2
#define _FLOAT4					float4
#define _FLOAT8					float8

#ifndef FLT_MAX
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 1
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif


#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif 

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif


#ifndef MIO_BN_GRP0
#define MIO_BN_GRP0 1
#endif

#ifndef MIO_BN_GRP1
#define MIO_BN_GRP1 1
#endif

#ifndef MIO_BN_GRP2
#define MIO_BN_GRP2 1
#endif

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 0
#endif

/*
static inline void ReduceKernel(__local _FLOAT * lcl_mem, int sum_stride, int unit_id, int unit_len){
    _FLOAT sum = 0;
    int lcl_offset = unit_id * unit_len;
    
    #pragma unroll
    for(int i = 0; i < unit_len; i += sum_stride){
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}
*/

static inline void ReduceKernel(__local _FLOAT * lcl_mem, unsigned int sum_stride, unsigned  int unit_id, unsigned int unit_len){
    _FLOAT sum = 0;
    unsigned int lcl_offset = unit_id * unit_len;
    
    #pragma unroll
    for(unsigned int i = 0; i < unit_len; i += sum_stride){
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

#if(MIO_BN_VARIANT==0)



__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedSingleDX(
                                    const __global _FLOAT   * __restrict x_in, 
                                    const __global _FLOAT   * __restrict dy_in,
                                    __global _FLOAT         * __restrict dx_out,
                                    const __global _FLOAT 	*bnScale, 
                                    __global _FLOAT   * __restrict dscale,
                                    __global _FLOAT   * __restrict dbias,
                                    const __global _FLOAT	*savedMean, 
                                    const __global _FLOAT	*savedInvVariance,
                                    float INHW
){

    //SPATIAL
    __private _FLOAT mean       = 0.;
    __private _FLOAT invVar     = 0.;
    __private _FLOAT xhat       = 0.;
    __private _FLOAT pvt_scale  = 0.;
    __private _FLOAT pvt_dscale = 0.;
    __private _FLOAT pvt_dbias  = 0.;
    _FLOAT elemStd;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
     
//TODO: double and quad up channels in a single workgroup
    unsigned int cidx = xgid*MIO_BN_HW;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    mean = savedMean[xgid];
    invVar = savedInvVariance[xgid];

    //DONE WITH variance
    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            pvt_dbias += dy_in[index];
        }  
    }
    lcl_data[ylid] = pvt_dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dbias = lcl_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    pvt_dscale = 0.;
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            elemStd  = x_in[index] - mean;// (x_i - mean)
            xhat 	 = elemStd*invVar;
            pvt_dscale   = mad(xhat, dy_in[index], pvt_dscale);
        }//end for n
    }
    lcl_data[ylid] = pvt_dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = lcl_data[0]*INHW;
   
    pvt_scale   = bnScale[xgid];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Group level reduction
    //Need to reduce over all elements in NxHxW
    //move across the sections of an image in the mini_batch stack
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            elemStd 	= x_in[index] - mean;// (x_i - mean)
            xhat 	= elemStd*invVar; //recalculating this again...
            tmp1 	= mad(NHW,dy_in[index],-pvt_dbias);
            tmp2 	= -xhat*pvt_dscale;
            tmp3 	= (pvt_scale*invVar)*INHW;
            dx_out[index] = tmp3*(tmp2+tmp1);//DEBUG
        }
    }
    if(ygid == 0){
        dbias[xgid]  = pvt_dbias;
        dscale[xgid] = pvt_dscale;
    }
    
}//end spatial



//Recalc everything
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSingleDX(
                        const __global _FLOAT 	* __restrict x_in, 
                        const __global _FLOAT   * __restrict dy_in,
                        __global _FLOAT         * __restrict dx_out,
			const __global _FLOAT 	*bnScale, 
                        __global _FLOAT   * __restrict dscale,
                        __global _FLOAT   * __restrict dbias,
                        double                  epsilon,
                        float              INHW
){

    //SPATIAL
    __private _FLOAT mean       = 0.;
    __private _FLOAT variance   = 0.;
    __private _FLOAT invVar= 0.;
    __private _FLOAT xhat      = 0.;
    __private _FLOAT pvt_scale  = 0.;
    __private _FLOAT pvt_dscale   = 0.;
    __private _FLOAT pvt_dbias   = 0.;
	_FLOAT elemStd = 0.;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
       _FLOAT tmp1, tmp2, tmp3;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            mean += x_in[index];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0]*INHW;///NHW;
    
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            elemStd = (x_in[index] - mean);
            variance += elemStd*elemStd;
        }  
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0]*INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVar = rsqrt(variance + epsilon);
    
    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //DONE WITH variance
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            pvt_dbias += dy_in[index];
        }  
    }
    lcl_data[ylid] = pvt_dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dbias = lcl_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);
     
    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = 0.;
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            elemStd 	 = x_in[index] - mean;// (x_i - mean)
            xhat 	 = elemStd*invVar;
            pvt_dscale   = mad(xhat, dy_in[index], pvt_dscale);
        }//end for n
    }
    lcl_data[ylid] = pvt_dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = lcl_data[0]*INHW;
   
    pvt_scale   = bnScale[xgid];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Group level reduction
    //Need to reduce over all elements in NxHxW
    //move across the sections of an image in the mini_batch stack
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            elemStd 	= x_in[index] - mean;// (x_i - mean)
            xhat 	= elemStd*invVar; //recalculating this again...
            tmp1 	= mad(NHW,dy_in[index],-pvt_dbias);
            tmp2 	= -xhat*pvt_dscale;
            tmp3 	= (pvt_scale*invVar)*INHW;
            dx_out[index] = tmp3*(tmp2+tmp1);
        }
    }
    if(ygid == 0){
        dbias[xgid]  = pvt_dbias;
        dscale[xgid] = pvt_dscale;
    }

}//end spatial



#elif(MIO_BN_VARIANT==1)

//=============== SINGLE WORKGROUP PER CHANNEL

//Recalc everything
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSingleLDSDX(
                        const __global _FLOAT 	* __restrict x_in, 
                        const __global _FLOAT   * __restrict dy_in,
                        __global _FLOAT         * __restrict dx_out,
			const __global _FLOAT 	*bnScale, 
                        __global _FLOAT   * __restrict dscale,
                        __global _FLOAT   * __restrict dbias,
                        double                  epsilon,
                        float              INHW
){

    //SPATIAL
    __private _FLOAT mean       = 0.;
    __private _FLOAT variance   = 0.;
    __private _FLOAT invVar= 0.;
    __private _FLOAT xhat      = 0.;
    __private _FLOAT pvscale  = 0.;
    __private _FLOAT pvt_dscale   = 0.;
    __private _FLOAT pvt_dbias   = 0.;
    _FLOAT elemStd = 0.;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    _FLOAT tmp1, tmp2, tmp3;
    
    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_scale;
    
    if(ylid==0) lcl_scale = bnScale[xgid];
    
    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            lcl_indata[n][ylid] = x_in[index];
            mean += lcl_indata[n][ylid];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0]*INHW;///NHW;
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd = (lcl_indata[n][ylid] - mean);
            variance = mad(elemStd,elemStd,variance);
        }  
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0]*INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVar = rsqrt(variance + epsilon);
    
    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //DONE WITH variance
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            pvt_dbias += dy_in[index];
        }  
    }
    lcl_data[ylid] = pvt_dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dbias = lcl_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);
     
    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = 0.;
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd 	 = lcl_indata[n][ylid] - mean;// (x_i - mean)
            xhat 	 = elemStd*invVar;
            pvt_dscale   = mad(xhat, dy_in[index], pvt_dscale);
        }//end for n
    }
    lcl_data[ylid] = pvt_dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = lcl_data[0]*INHW;
   
    barrier(CLK_LOCAL_MEM_FENCE);

    // Group level reduction
    //Need to reduce over all elements in NxHxW
    //move across the sections of an image in the mini_batch stack
    if(ygid<MIO_BN_HW){
        pvscale = lcl_scale;
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd 	= lcl_indata[n][ylid] - mean;// (x_i - mean)
            xhat 	= elemStd*invVar; //recalculating this again...
            tmp1 	= mad(NHW,dy_in[index],-pvt_dbias);
            tmp2 	= -xhat*pvt_dscale;
            tmp3 	= (pvscale*invVar)*INHW;
            dx_out[index] = tmp3*(tmp2+tmp1);
        }
    }
    if(ygid == 0){
        dbias[xgid]  = pvt_dbias;
        dscale[xgid] = pvt_dscale;
    }

}//end spatial

#elif(MIO_BN_VARIANT==2)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedSingleLDSDX(
                                    const __global _FLOAT   * __restrict x_in, 
                                    const __global _FLOAT   * __restrict dy_in,
                                    __global _FLOAT         * __restrict dx_out,
                                    const __global _FLOAT 	*bnScale, 
                                    __global _FLOAT   * __restrict dscale,
                                    __global _FLOAT   * __restrict dbias,
                                    const __global _FLOAT	*savedMean, 
                                    const __global _FLOAT	*savedInvVariance,
                                    float INHW
){

    //SPATIAL
    __private _FLOAT mean       = 0.;
    __private _FLOAT invVar= 0.;
    __private _FLOAT xhat      = 0.;
    __private _FLOAT pvscale  = 0.;
    __private _FLOAT pvt_dscale  = 0.;
    __private _FLOAT pvt_dbias   = 0.;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_data0[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_data1[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_scale, lcl_mean, lcl_ivar;
    
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int idx = cidx+ygid;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    if(ylid==0){
        lcl_mean = savedMean[xgid];
        lcl_ivar = savedInvVariance[xgid];
        lcl_scale = bnScale[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    invVar = lcl_ivar;
    mean = lcl_mean;
  
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            pvt_dbias += dy_in[index];
            lcl_indata[n][ylid] = x_in[index] - mean;
            xhat 	 = lcl_indata[n][ylid]*invVar;
            pvt_dscale   = mad(xhat, dy_in[index], pvt_dscale);
        }  
    }
    lcl_data0[ylid] = pvt_dbias;
    lcl_data1[ylid] = pvt_dscale;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 2)){
        ReduceKernel(lcl_data0, 1, ylid, 4);
        ReduceKernel(lcl_data1, 1, ylid, 4);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)){
        ReduceKernel(lcl_data0, 4, ylid, 16);
        ReduceKernel(lcl_data1, 4, ylid, 16);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0){
        ReduceKernel(lcl_data0, 16, ylid, MIO_BN_LDS_SIZE);
        ReduceKernel(lcl_data1, 16, ylid, MIO_BN_LDS_SIZE);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dbias  = lcl_data0[0];
    pvt_dscale = lcl_data1[0]*INHW;

    // Group level reduction
    //Need to reduce over all elements in NxHxW
    //move across the sections of an image in the mini_batch stack
    if(ylid<MIO_BN_HW){
        pvscale   = lcl_scale;
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index         = n*MIO_BN_CHW + idx;
            xhat 	  = lcl_indata[n][ylid]*invVar; //recalculating this again...
            tmp1 	  = mad(NHW,dy_in[index],-pvt_dbias);
            tmp2 	  = -xhat*pvt_dscale;
            tmp3 	  = (pvscale*invVar)*INHW;
            dx_out[index] = tmp3*(tmp2+tmp1);//DEBUG
        }
    }
    if(ygid == 0){
        dbias[xgid]  = pvt_dbias;
        dscale[xgid] = pvt_dscale;
    }
}//end spatial






#elif(MIO_BN_VARIANT==3)



//Recalc everything
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSingleDX(
                        const __global _FLOAT 	* __restrict x_in, 
                        const __global _FLOAT   * __restrict dy_in,
                        __global _FLOAT         * __restrict dx_out,
			const __global _FLOAT 	*bnScale, 
                        __global _FLOAT         * __restrict dscale,
                        __global _FLOAT         * __restrict dbias,
                        double                  epsilon,
                        float                   INHW
){

    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT variance   = 0.;
    _FLOAT invVar     = 0.;
    _FLOAT xhat       = 0.;
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_dscale = 0.;
    _FLOAT pvt_dbias  = 0.;
    _FLOAT elemStd    = 0.;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            mean += x_in[index];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0]*INHW;///NHW;
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd = (x_in[index] - mean);
            variance += elemStd*elemStd;
        }  
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0]*INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVar = rsqrt(variance + epsilon);
    
    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //DONE WITH variance
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            pvt_dbias += dy_in[index];
        }  
    }
    lcl_data[ylid] = pvt_dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dbias = lcl_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);
     
    lcl_data[ylid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = 0.;
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd 	 = x_in[index] - mean;// (x_i - mean)
            xhat 	 = elemStd*invVar;
            pvt_dscale   = mad(xhat, dy_in[index], pvt_dscale);
        }//end for n
    }
    lcl_data[ylid] = pvt_dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = lcl_data[0]*INHW;
   
    pvt_scale   = bnScale[xgid];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Group level reduction
    //Need to reduce over all elements in NxHxW
    //move across the sections of an image in the mini_batch stack
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd 	= x_in[index] - mean;// (x_i - mean)
            xhat 	= elemStd*invVar; //recalculating this again...
            tmp1 	= mad(NHW,dy_in[index],-pvt_dbias);
            tmp2 	= -xhat*pvt_dscale;
            tmp3 	= (pvt_scale*invVar)*INHW;
            dx_out[index] = tmp3*(tmp2+tmp1);
        }
    }
    if(ygid == 0){
        dbias[xgid]  = pvt_dbias;
        dscale[xgid] = pvt_dscale;
    }

}//end spatial

#elif(MIO_BN_VARIANT==4)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedSingleDX(
                                    const __global _FLOAT   * __restrict x_in, 
                                    const __global _FLOAT   * __restrict dy_in,
                                    __global _FLOAT         * __restrict dx_out,
                                    const __global _FLOAT 	*bnScale, 
                                    __global _FLOAT   * __restrict dscale,
                                    __global _FLOAT   * __restrict dbias,
                                    const __global _FLOAT	*savedMean, 
                                    const __global _FLOAT	*savedInvVariance,
                                    float INHW
){

    //SPATIAL
    _FLOAT mean      = 0.;
    _FLOAT invVar    = 0.;
    _FLOAT xhat      = 0.;
    _FLOAT pvt_scale = 0.;
    _FLOAT pvt_dscale= 0.;
    _FLOAT pvt_dbias = 0.;
    _FLOAT elemStd   = 0.;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_mean, lcl_ivar;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    if(ylid==0){
        lcl_mean = savedMean[xgid];
        lcl_ivar = savedInvVariance[xgid];
    }
    
    
    //DONE WITH variance
    lcl_data[ylid] = 0.;
    mean  = lcl_mean;
    invVar= lcl_ivar; 
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            pvt_dbias += dy_in[index];
        }  
    }
    lcl_data[ylid] = pvt_dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dbias = lcl_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    pvt_dscale = 0.;
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd  = x_in[index] - mean;// (x_i - mean)
            xhat 	 = elemStd*invVar;
            pvt_dscale   = mad(xhat, dy_in[index], pvt_dscale);
        }//end for n
    }
    lcl_data[ylid] = pvt_dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    pvt_dscale = lcl_data[0]*INHW;
   
    pvt_scale   = bnScale[xgid];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Group level reduction
    //Need to reduce over all elements in NxHxW
    //move across the sections of an image in the mini_batch stack
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd 	= x_in[index] - mean;// (x_i - mean)
            xhat 	= elemStd*invVar; //recalculating this again...
            tmp1 	= mad(NHW,dy_in[index],-pvt_dbias);
            tmp2 	= -xhat*pvt_dscale;
            tmp3 	= (pvt_scale*invVar)*INHW;
            dx_out[index] = tmp3*(tmp2+tmp1);//DEBUG
        }
    }
    if(ygid == 0){
        dbias[xgid]  = pvt_dbias;
        dscale[xgid] = pvt_dscale;
    }
    
}//end spatial






#elif(MIO_BN_VARIANT==5)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialMean(
                                        const __global _FLOAT    * __restrict in,  
                                        __global _FLOAT          * __restrict meanbuff){

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid       = get_local_id(1); 
    unsigned int ygrp_id    = get_group_id(1);
    unsigned int xgid       = get_global_id(0);
    unsigned int ygid       = get_global_id(1);
    unsigned int ygrp_sz    = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    _FLOAT mean = 0.;
    
    //move across the sections of the image mini_batch stack 
    lcl_data[ylid] = 0.0;//zero out local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            mean += in[index];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid==0){
        unsigned int meanindex = cidx+ygrp_sz*ygrp_id;//making assumption of n=0 here
        meanbuff[meanindex] = lcl_data[0];//pre-stage for group reduction
    }
}//end spatial mean kernel


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialFinalMean(
                                        __global _FLOAT		* __restrict meanvarbuff){

    __private _FLOAT mean = 0.;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1); 
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx = xgid*MIO_BN_HW;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;
    lcl_data[ylid] = 0.0;//zero out local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int gn = 0; gn<yngrps; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int meanindex   = cidx + ygrp_sz*offset; 
        
        if(offset < yngrps){//modify to span larger number of groups
            lcl_data[ylid] += meanvarbuff[meanindex];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    
    #pragma unroll
    for(int i = 0; i<MIO_BN_LDS_SIZE;i++){
       mean += lcl_data[i];
    }
    mean /= NHW;
    //DONE WITH MEAN

    if(ylid==0){	
        unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
        meanvarbuff[meanstashindex] = mean;//stash mean
    }
}

//This kernel is independent of others
//Partial summation of dBias = sum(dY)
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialDBias(
                                        const __global _FLOAT    * __restrict dy_in,  
                                        __global _FLOAT          * __restrict dbiasbuff){

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
	
    unsigned int xgid       = get_global_id(0);
    unsigned int ylid       = get_local_id(1); 
    unsigned int ygrp_id    = get_group_id(1);
    unsigned int ygid       = get_global_id(1);
    unsigned int ygrp_sz    = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    _FLOAT dbias = 0.;
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            dbias += dy_in[index];
        }  
    }
    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid==0){
        unsigned int biasstashindex=cidx+ygrp_sz*ygrp_id+6;
        dbiasbuff[biasstashindex] = lcl_data[0];//dbias;
    }
}//end spatial mean kernel




__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialVariance(
					const __global _FLOAT   * __restrict in, 
					__global _FLOAT		* __restrict meanvarbuff){   
    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT elemStd    = 0.;
    _FLOAT variance   = 0.;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;

    unsigned int meanstashindex = cidx + ygrp_sz*ygrp_id + 1;
    mean = meanvarbuff[meanstashindex];//load stashed mean
      
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd = (in[index] - mean);
            variance += elemStd*elemStd;
        }  
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);
   
    if(ylid==0){
        unsigned int varindex = cidx + ygrp_sz*ygrp_id + 2;
        meanvarbuff[varindex] = lcl_data[0];//pre-stage for group reduction
    }
}//end spatial variance


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialFinalVariance(
			__global _FLOAT		* __restrict varbuff, 
			double			epsilon){

    //SPATIAL
    __private _FLOAT variance   = 0.;
    __private _FLOAT invVariance= 0.;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    
    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;
    
    #pragma unroll
    for(int gn = 0; gn<MIO_BN_NGRPS; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        if(offset < yngrps){//modify to span larger number of groups
            unsigned int varindex   = cidx + ygrp_sz*offset + 2;
            variance += varbuff[varindex];//load per group variance
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int fn = (MIO_BN_LDS_SIZE>>1); fn > 0; fn=fn>>1){//final LDS block reduction
        if(ylid<fn){
            lcl_data[ylid] += lcl_data[ylid+fn];//every wi in a wg should have this value
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    variance = lcl_data[0]/NHW;
    
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(fabs(variance + epsilon));
    //DONE WITH variance
    //stash inverse variance

    if(ylid==0){
        unsigned int varstashindex=cidx+ygrp_sz*ygrp_id+3;
        varbuff[varstashindex] = invVariance;//stash
    }
}//end spatial final variance


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialDScale(
					const __global _FLOAT 	*x_in, 
					const __global _FLOAT 	*dy_in, 
					__global _FLOAT		*buff){

	__local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

	int ylid = get_local_id(1); //accumilate / reduction 
	int ygrp_id = get_group_id(1);
	int xgid = get_global_id(0);
	int ygid = get_global_id(1);
	int ygrp_sz = get_local_size(1);
	int cidx = MIO_BN_HW*xgid;
	unsigned int index, ncIdx;
	
	_FLOAT mean = 0.;
        _FLOAT invVar = 0.;
	_FLOAT elemStd = 0.;
        _FLOAT xhat = 0.;
        _FLOAT dscale = 0.;

	unsigned int meanstashindex = cidx + ygrp_sz*ygrp_id + 1;
	unsigned int varstashindex  = cidx + ygrp_sz*ygrp_id + 3;
        
        mean   = buff[meanstashindex];//load stashed mean
        invVar = buff[varstashindex];
	
	//Need to reduce over all elements in NxHxW
	//move across the sections of an image in the mini_batch stack
	
        if(ygid<MIO_BN_HW){
            #pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
                ncIdx = n*MIO_BN_CHW + cidx;
                index = ncIdx + ygid;
                elemStd 	 = x_in[index] - mean;// (x_i - mean)
                xhat 		 = elemStd*invVar;
                dscale= mad(xhat, dy_in[index], dscale);
            }//end for
	}//end if
        lcl_data[ylid] = dscale;
	barrier(CLK_LOCAL_MEM_FENCE);
        
        if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
        barrier(CLK_LOCAL_MEM_FENCE);
        if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
        barrier(CLK_LOCAL_MEM_FENCE);
        if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
        barrier(CLK_LOCAL_MEM_FENCE);
        
	if(ylid==0){
		unsigned int gammaindex = cidx + ygrp_sz*ygrp_id + 4;
		buff[gammaindex] = lcl_data[0];//pre-stage for group reduction
	}
}


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialFinalDScale(
					__global _FLOAT			*buff,
					__global _FLOAT			*delta_scale){

   __private _FLOAT dscale = 0.;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1); 
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1); 
    unsigned int yngrps  = get_num_groups(1);
    int cidx = MIO_BN_HW*xgid;
 
    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    #pragma unroll
    for(int gn = 0; gn<MIO_BN_NGRPS; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        if(offset < yngrps){//modify to span larger number of groups
            unsigned int gammaindex   = cidx + ygrp_sz*offset+4;
            dscale += buff[gammaindex];
        }
    }
    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int fn = (MIO_BN_LDS_SIZE>>1); fn > 0; fn=fn>>1){//final LDS block reduction
        if(ylid<fn){
            lcl_data[ylid] += lcl_data[ylid+fn];//every wi in a wg should have this value
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dscale = lcl_data[0]/NHW;

    if(ygid==0) delta_scale[xgid] = dscale;
}


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialFinalDBias(
					__global _FLOAT			*buff,
					__global _FLOAT			*delta_bias){

  // __private _FLOAT dbias = 0.;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1); 
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx = MIO_BN_HW*xgid;
    /*
    #pragma unroll
    for(int gn = 0; gn<MIO_BN_NGRPS; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        if(offset < yngrps){//modify to span larger number of groups
            unsigned int betaindex   = cidx + ygrp_sz*offset + 6;
            dbias += buff[betaindex];      
        }
    }
    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int fn = (MIO_BN_LDS_SIZE>>1); fn > 0; fn=fn>>1){//final LDS block reduction
        if(ylid<fn){
            lcl_data[ylid] += lcl_data[ylid+fn];//every wi in a wg should have this value
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    */
    lcl_data[ylid] = 0.0;//zero out local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int gn = 0; gn<yngrps; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int betaindex   = cidx + ygrp_sz*offset + 1;
        if(offset < yngrps){//modify to span larger number of groups
            lcl_data[ylid] += buff[betaindex];      
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);

    //DONE WITH DSCALE REDUCTION
    if(ygid==0) delta_bias[xgid] = lcl_data[0];//dbias;
}


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialDX(
					const __global _FLOAT 	*x_in, 
					const __global _FLOAT 	*dy_in, 
					__global _FLOAT 	*dx_out,
					const __global _FLOAT 	*bnScale, 
					__global _FLOAT	 	*delta_scale,
					__global _FLOAT	 	*delta_bias){


    int ygrp_id = get_group_id(1);
    int xgid = get_global_id(0);
    int ygid = get_global_id(1);
    int ygrp_sz = get_local_size(1);
    int cidx = MIO_BN_HW*xgid;
    unsigned int ncIdx,index;
    _FLOAT mean, invVar;
    _FLOAT elemStd, xhat;
    _FLOAT scale, dscale, dbias;
    _FLOAT tmp1, tmp2, tmp3;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    unsigned int meanstashindex = cidx+ygrp_sz*ygrp_id+1;
    mean        = dx_out[meanstashindex];//load stashed mean
    
    unsigned int varstashindex = cidx+ygrp_sz*ygrp_id+3;
    invVar = dx_out[varstashindex];	
    
    scale  = bnScale[xgid];
    dscale = delta_scale[xgid];
    dbias  = delta_bias[xgid];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
//________________________________________________
// Group level reduction
	//Need to reduce over all elements in NxHxW
	//move across the sections of an image in the mini_batch stack
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            ncIdx = n*MIO_BN_CHW + cidx;
            index = ncIdx + ygid;
            elemStd = x_in[index] - mean;// (x_i - mean)
            xhat 	= elemStd*invVar; //recalculating this again...
            tmp1 	= mad(NHW,dy_in[index],-dbias);
            tmp2 	= -xhat*dscale;
            tmp3 	= (scale*invVar)/NHW;
            dx_out[index] = tmp3*(tmp2+tmp1);//DEBUG
	}
    }
}

//============================================================



#else




//===================== SPATIAL SAVED ========================




__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedDScale(
					const __global _FLOAT 	*x_in, 
					const __global _FLOAT 	*dy_in, 
					const __global _FLOAT	*savedMean, 
					const __global _FLOAT	*savedInvVariance,
					__global _FLOAT		*dscalebuff){

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    int ylid = get_local_id(1); //accumilate / reduction 
    int ygrp_id = get_group_id(1);
    int xgid = get_global_id(0);
    int ygid = get_global_id(1);
    int ygrp_sz = get_local_size(1);
    int cidx = MIO_BN_HW*xgid;
    unsigned int index;

    _FLOAT mean, invVar;
    _FLOAT elemStd, xhat;
    _FLOAT dscale = 0.;
    
    mean = savedMean[xgid];
    invVar = savedInvVariance[xgid];

    //Need to reduce over all elements in NxHxW
    //move across the sections of an image in the mini_batch stack    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd 	 = x_in[index] - mean;// (x_i - mean)
            xhat         = elemStd*invVar;
            dscale= mad(xhat, dy_in[index], dscale);
        }//end for
    }//end if
    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid==0){
        unsigned int gammaindex = cidx + ygrp_sz*ygrp_id;
        dscalebuff[gammaindex] = lcl_data[0];//pre-stage for group reduction
    }
}


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedFinalDScale(
					__global _FLOAT			*buff,
					__global _FLOAT			*delta_scale){

   __private _FLOAT dscale = 0.;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1); 
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx = MIO_BN_HW*xgid;
 
    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    #pragma unroll
    for(int gn = 0; gn<MIO_BN_NGRPS; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        if(offset < yngrps){//modify to span larger number of groups
            unsigned int gammaindex   = cidx + ygrp_sz*offset;
            dscale += buff[gammaindex];
        }
    }
    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int fn = (MIO_BN_LDS_SIZE>>1); fn > 0; fn=fn>>1){//final LDS block reduction
        if(ylid<fn){
            lcl_data[ylid] += lcl_data[ylid+fn];//every wi in a wg should have this value
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dscale = lcl_data[0]/NHW;
    
    if(ygid==0) delta_scale[xgid] = dscale;
}


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedDBias(
                                        const __global _FLOAT    * __restrict dy_in,  
                                        __global _FLOAT          * __restrict dbiasbuff){

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
	
    unsigned int ylid       = get_local_id(1); 
    unsigned int ygrp_id    = get_group_id(1);
    unsigned int xgid       = get_global_id(0);
    unsigned int ygid       = get_global_id(1);
    unsigned int ygrp_sz    = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    _FLOAT dbias = 0.;
    
    //move across the sections of the image mini_batch stack 
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;    
            dbias += dy_in[index];

        }  
    }
    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid==0){
        unsigned int biasstashindex=cidx+ygrp_sz*ygrp_id+1;
        dbiasbuff[biasstashindex] = lcl_data[0];// dbias; 
    }
}



__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedFinalDBias(
					__global _FLOAT			*buff,
					__global _FLOAT			*delta_bias){

    ///__private _FLOAT dbias = 0.;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1); 
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx = MIO_BN_HW*xgid;
/*
    #pragma unroll
    for(int gn = 0; gn<MIO_BN_NGRPS; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        if(offset < yngrps){//modify to span larger number of groups
            unsigned int betaindex   = cidx + ygrp_sz*offset + 1;
            dbias += buff[betaindex];      
        }
    }
    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int fn = (MIO_BN_LDS_SIZE>>1); fn > 0; fn=fn>>1){//final LDS block reduction
        if(ylid<fn){
            lcl_data[ylid] += lcl_data[ylid+fn];//every wi in a wg should have this value
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    */
    lcl_data[ylid] = 0.0;//zero out local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int gn = 0; gn<yngrps; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int betaindex   = cidx + ygrp_sz*offset + 1;
        if(offset < yngrps){//modify to span larger number of groups
            lcl_data[ylid] += buff[betaindex];      
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
/*    #pragma unroll
    for(int i = 0; i<MIO_BN_LDS_SIZE;i++){
       dbias += lcl_data[i];
    }
  */  
    barrier(CLK_LOCAL_MEM_FENCE);
 
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //DONE WITH DSCALE REDUCTION	
    if(ygid==0) delta_bias[xgid] = lcl_data[0];//dbias;
}


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormBwdSpatialSavedDX(
					const __global _FLOAT 	*x_in, 
					const __global _FLOAT 	*dy_in, 
					__global _FLOAT 	*dx_out,
					const __global _FLOAT 	*bnScale, 
					__global _FLOAT	 	*delta_scale,
					__global _FLOAT	 	*delta_bias, 
					const __global _FLOAT	*savedMean, 
					const __global _FLOAT	*savedInvVariance){



    int xgid = get_global_id(0);
    int ygid = get_global_id(1);
    int cidx = MIO_BN_HW*xgid;
    unsigned int index;
    _FLOAT mean, invVar;
    _FLOAT elemStd, xhat;
    _FLOAT scale, dscale, dbias;
    _FLOAT tmp1, tmp2, tmp3;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    mean   = savedMean[xgid];//load stashed mean
    invVar = savedInvVariance[xgid];//load stashed inverse variance
    scale  = bnScale[xgid];
    dscale = delta_scale[xgid];
    dbias  = delta_bias[xgid];
 
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// Group level reduction
	//Need to reduce over all elements in NxHxW
	//move across the sections of an image in the mini_batch stack
	//#pragma unroll
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd = x_in[index] - mean;// (x_i - mean)
            xhat 	= elemStd*invVar; //recalculating this again...
            tmp1 	= mad(NHW,dy_in[index],-dbias);
            tmp2 	= -xhat*dscale;
            tmp3 	= (scale*invVar)/NHW;
            dx_out[index] = tmp3*(tmp2+tmp1);//DEBUG
        }
    }
}

//============================================================



#endif




