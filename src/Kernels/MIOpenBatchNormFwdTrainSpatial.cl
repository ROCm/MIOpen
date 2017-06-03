

#define _FLOAT					float
#define _FLOAT2					float2
#define _FLOAT4					float4
#define _FLOAT8					float8

#ifndef FLT_MAX
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 256
#endif

#ifndef MIO_BN_LDS_NSIZE
#define MIO_BN_LDS_NSIZE 256
#endif

#ifndef MIO_BN_LDS_HWSIZE
#define MIO_BN_LDS_HWSIZE 256
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

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
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
#define MIO_BN_VARIANT 4
#endif

#define UNUSED __attribute__((__unused__))


static inline void ReduceKernel(__local _FLOAT * lcl_mem, unsigned int sum_stride, unsigned  int unit_id, unsigned int unit_len){
    _FLOAT sum = 0;
    unsigned int lcl_offset = unit_id * unit_len;
    
    #pragma unroll
    for(unsigned int i = 0; i < unit_len; i += sum_stride){
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}



#if (MIO_BN_VARIANT==0)


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatial(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT       * __restrict scale,
                        __constant _FLOAT       * __restrict bias,
                        float                   INHW,
        #if (MIO_RUNNING_RESULT == 1)    
                        double                  expAvgFactor,
			__global _FLOAT	        * __restrict resultRunningMean,
                        __global _FLOAT         * __restrict resultRunningVariance,
        #endif
                        double                  epsilon
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
			, __global _FLOAT       * __restrict resultSaveMean
                        , __global _FLOAT       * __restrict resultSaveInvVariance
        #endif
){

    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT variance   = 0.;
    _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    _FLOAT elemStd    = 0.;
    _FLOAT pvscale, pvbias, tmp;

    __local _FLOAT lcl_indata[MIO_BN_LDS_HWSIZE][MIO_BN_LDS_NSIZE];
    __local _FLOAT2 lcl_scalebias;
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    //unsigned int idx = cidx+ygid;
    
    if(ylid==0){
        lcl_scalebias[0] = scale[xgid];
        lcl_scalebias[1] = bias[xgid];
    }
       
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
                mean += lcl_indata[hw][ylid] = in[index];
        }  
    }
    
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean =  as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    mean *= INHW;

    if(ygid<MIO_BN_N){
        #pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++){
            elemStd = (lcl_indata[hw][ylid] - mean);
            variance = mad(elemStd,elemStd,variance);
        }  
    }    

    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp       = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp      = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    variance *= INHW;

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    pvscale = lcl_scalebias[0];
    pvbias  = lcl_scalebias[1];

    barrier(CLK_LOCAL_MEM_FENCE);
       
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            inhat =  (lcl_indata[hw][ylid]-mean)*invVariance;      
            out[index] = mad(pvscale, inhat, pvbias);
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0){
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[xgid] = mean;
            resultSaveInvVariance[xgid] = invVariance;
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[xgid];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm




#elif (MIO_BN_VARIANT==1)



__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatial(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT   * __restrict scale,
                        __constant _FLOAT   * __restrict bias,
                        float                  INHW,
        #if (MIO_RUNNING_RESULT == 1)    
                        double                  expAvgFactor,
			__global _FLOAT	        * __restrict resultRunningMean,
                        __global _FLOAT         * __restrict resultRunningVariance,
        #endif
                        double                  epsilon
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
			, __global _FLOAT       * __restrict resultSaveMean
                        , __global _FLOAT       * __restrict resultSaveInvVariance
        #endif
){

    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT tmp        = 0.;
    _FLOAT variance   = 0.;
    _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    _FLOAT elemStd    = 0.;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT2 lcl_scalebias;
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int idx = cidx+ygid;
    
    if(ylid==0){
        lcl_scalebias[0] = scale[xgid];
        lcl_scalebias[1] = bias[xgid];
    }
       
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
                mean += lcl_indata[n][ylid] = in[index];
        }  
    }
    
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean =  as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    mean *= INHW;

    if(ygid<MIO_BN_HW){    
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            elemStd = (lcl_indata[n][ylid] - mean);
            variance = mad(elemStd,elemStd,variance);
        }  
    }    

    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp       = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp      = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    variance *= INHW;

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    pvscale = lcl_scalebias[0];
    pvbias  = lcl_scalebias[1];

    barrier(CLK_LOCAL_MEM_FENCE);
       
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
            inhat =  (lcl_indata[n][ylid]-mean)*invVariance;      
            out[index] = mad(pvscale, inhat, pvbias);
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0){
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[xgid] = mean;
            resultSaveInvVariance[xgid] = invVariance;
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[xgid];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm




#elif (MIO_BN_VARIANT==2)

// This kernel implied that the input data does not fit into LDS, but the image size is
// smaller than 64 pixels

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatial(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT   * __restrict scale,
                        __constant _FLOAT   * __restrict bias,
                        float                  INHW,
                #if (MIO_RUNNING_RESULT == 1)    
                        double                  expAvgFactor,
			__global _FLOAT	        * __restrict resultRunningMean,
                        __global _FLOAT         * __restrict resultRunningVariance,
                #endif
                        double                  epsilon
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
			, __global _FLOAT       * __restrict resultSaveMean
                        , __global _FLOAT       * __restrict resultSaveInvVariance
                #endif
){

    //SPATIAL
    _FLOAT tmp        = 0.;
    _FLOAT mean       = 0.;
    _FLOAT variance   = 0.;
    _FLOAT invVariance, inhat, elemStd;
    _FLOAT pvscale, pvbias;
    
    __local _FLOAT2 lcl_scalebias;
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int idx  = cidx+ygid;
    
    if(ylid==0){
        lcl_scalebias[0] = scale[xgid];
        lcl_scalebias[1] = bias[xgid];
    }
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            mean += in[index];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    
        
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean =  as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    mean *= INHW;

    if(ygid<MIO_BN_HW){    
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            elemStd = (in[index] - mean);
            variance = mad(elemStd,elemStd,variance);
        }  
    }    

    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp       = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp      = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    variance *= INHW;

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    if(ygid<MIO_BN_HW){
        
        pvscale = lcl_scalebias[0];
        pvbias  = lcl_scalebias[1];
        
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (in[index]-mean)*invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
            //out[index] = mad(pvt_scale, inhat, pvt_bias);
            out[index] = mad(pvscale, inhat, pvbias);
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0){
        
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[xgid] = mean;
            resultSaveInvVariance[xgid] = invVariance;
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[xgid];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm



#elif (MIO_BN_VARIANT==3)


// This kernel implies the image is greater than a wavefront, but smaller than 257

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatial(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT   * __restrict scale,
                        __constant _FLOAT   * __restrict bias,
                        float                  INHW,
                #if (MIO_RUNNING_RESULT == 1)    
                        double                  expAvgFactor,
			__global _FLOAT	        * __restrict resultRunningMean,
                        __global _FLOAT         * __restrict resultRunningVariance,
                #endif
                        double                  epsilon
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
			, __global _FLOAT       * __restrict resultSaveMean
                        , __global _FLOAT       * __restrict resultSaveInvVariance
                #endif
){

    //SPATIAL
    _FLOAT tmp = 0.;
    _FLOAT mean       = 0.;
    _FLOAT variance   = 0.;
    _FLOAT invVariance, inhat, elemStd;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT2 lcl_scalebias;
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int idx = cidx+ygid;
    
    if(ylid==0){
        lcl_scalebias[0] = scale[xgid];
        lcl_scalebias[1] = bias[xgid];
    }
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            mean += in[index];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);    
    if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[ylid];
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean =  as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    if(ylid == 0) lcl_data[0] = mean*INHW;       
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];

    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            elemStd = (in[index] - mean);
            variance = mad(elemStd,elemStd,variance);
        }  
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[ylid];
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    if(ylid == 0) lcl_data[0] = mad(variance,INHW,(_FLOAT) epsilon);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0];
    
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance);
    //DONE WITH variance
 
    if(ygid<MIO_BN_HW){
        
        pvscale = lcl_scalebias[0];
        pvbias  = lcl_scalebias[1];
        
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (in[index]-mean)*invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
            //out[index] = mad(pvt_scale, inhat, pvt_bias);
            out[index] = mad(pvscale, inhat, pvbias);
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0){
        
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[xgid] = mean;
            resultSaveInvVariance[xgid] = invVariance;
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[xgid];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm

#elif (MIO_BN_VARIANT==4)


// This kernel implies the image is greater than a wavefront, but smaller than 257

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatial(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT   * __restrict scale,
                        __constant _FLOAT   * __restrict bias,
                        float                  INHW,
                #if (MIO_RUNNING_RESULT == 1)    
                        double                  expAvgFactor,
			__global _FLOAT	        * __restrict resultRunningMean,
                        __global _FLOAT         * __restrict resultRunningVariance,
                #endif
                        double                  epsilon
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
			, __global _FLOAT       * __restrict resultSaveMean
                        , __global _FLOAT       * __restrict resultSaveInvVariance
                #endif
){

    //SPATIAL
    _FLOAT tmp = 0.;
    _FLOAT mean       = 0.;
    _FLOAT variance   = 0.;
    _FLOAT invVariance, inhat, elemStd;
    _FLOAT pvscale, pvbias;
    
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT2 lcl_scalebias;
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int idx = cidx+ygid;
    
    if(ylid==0){
        lcl_scalebias[0] = scale[xgid];
        lcl_scalebias[1] = bias[xgid];
    }
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            mean += lcl_indata[n][ylid] = in[index];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);    
    if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[ylid];
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean =  as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    if(ylid == 0) lcl_data[0] = mean*INHW;       
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];

    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            elemStd = lcl_indata[n][ylid] = (lcl_indata[n][ylid] - mean);
            variance = mad(elemStd,elemStd,variance);
        }  
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[ylid];
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    if(ylid == 0) lcl_data[0] = mad(variance,INHW,(_FLOAT) epsilon);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0];
    
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance);
    //DONE WITH variance
 
    if(ygid<MIO_BN_HW){
        
        pvscale = lcl_scalebias[0];
        pvbias  = lcl_scalebias[1];
        
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (in[index]-mean)*invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
            //out[index] = mad(pvt_scale, inhat, pvt_bias);
            out[index] = mad(pvscale, inhat, pvbias);
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0){
        
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[xgid] = mean;
            resultSaveInvVariance[xgid] = invVariance;
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[xgid];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm



#elif (MIO_BN_VARIANT==5)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialNorm(
			const __global _FLOAT 	* __restrict in, 
			__global _FLOAT		* __restrict out, 
			const __global _FLOAT 	* __restrict scale, 
			const __global _FLOAT	* __restrict bias
){

    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_bias   = 0.;
    
    __local _FLOAT lcl_mean, lcl_ivar, lcl_scale, lcl_bias;

    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int meanstashindex = cidx+ygrp_sz*ygrp_id+1;
    unsigned int varstashindex  = cidx+ygrp_sz*ygrp_id+3;
    
    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    if(get_local_id(1)==0){
        lcl_scale   = scale[xgid];
        lcl_bias    = bias[xgid];
        lcl_mean    = out[meanstashindex];//load stashed mean
        lcl_ivar    = out[varstashindex];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_HW){
        mean = lcl_mean;
        invVariance = lcl_ivar;
        pvt_scale = lcl_scale;
        pvt_bias = lcl_bias;
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + cidx + ygid;
            inhat = (in[index]-mean)*invVariance;
            // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        }//end for(n)
    }//end if(inImgIndex)
}//end spatial norm





__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialFinalVariance(
                        __global _FLOAT		* __restrict varbuff
                        ,float                  INHW
#if (MIO_RUNNING_RESULT == 1)
                        ,double                  expAvgFactor
                        ,__global _FLOAT         * __restrict resultRunningVariance
#endif
                       , double                  epsilon
#if (MIO_SAVE_MEAN_VARIANCE == 1)
                        , __global _FLOAT       * __restrict resultSaveInvVariance
#endif
){



  //SPATIAL
    __private _FLOAT variance   = 0.;
    __private _FLOAT invVariance= 0.;

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int varstashindex=cidx+ygrp_sz*ygrp_id+3;


    for(int gn = 0; gn<yngrps; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int varindex   = cidx + ygrp_sz*offset + 2;
        if(offset < yngrps){//modified to span larger number of groups
            variance += varbuff[varindex];//load per group variance
        }
    }
    
    #if (MIO_BN_NGRPS > 64)
        __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

        _FLOAT tmp = 0.;
        lcl_data[ylid] = variance;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
        barrier(CLK_LOCAL_MEM_FENCE);
        if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
        barrier(CLK_LOCAL_MEM_FENCE);

        variance = lcl_data[ylid];
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
        variance += tmp;
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
        variance += tmp;
        variance *= INHW;
        invVariance = rsqrt(variance + epsilon);    
        if(ylid==63){
            varbuff[varstashindex] = invVariance;//stash mean
        }
        #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
            if(get_global_id(1) == 63){
                //Save mean and calculate and save running mean
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
                    resultSaveInvVariance[xgid] = invVariance;
                #endif

                #if (MIO_RUNNING_RESULT == 1)
                    // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                    // right:: (1 - p)*(b/b-1)*var(n) = (1 - p)*adjust = -p*adjust + adjust
                    // var(n+1) = (p* var(n-1)) +  (-p*adjust + adjust)
                    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;
                    const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*(NHW/(NHW - 1));
                    tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
                    resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
                #endif
            }
        #endif

    #elif(MIO_BN_NGRPS > 16)

        _FLOAT tmp = 0.;
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
        variance += tmp;
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
        variance += tmp;
        variance *= INHW;
        invVariance = rsqrt(variance + epsilon);    
        if(ylid==63){
            varbuff[varstashindex] = invVariance;//stash mean
        }
        #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
            if(get_global_id(1) == 63){
                //Save mean and calculate and save running mean
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
                    resultSaveInvVariance[xgid] = invVariance;
                #endif

                #if (MIO_RUNNING_RESULT == 1)
                    // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                    // right:: (1 - p)*(b/b-1)*var(n) = (1 - p)*adjust = -p*adjust + adjust
                    // var(n+1) = (p* var(n-1)) +  (-p*adjust + adjust)
                    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;
                    const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*(NHW/(NHW - 1));
                    tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
                    resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
                #endif
            }
        #endif
    #else
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
        variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));
        variance *= INHW;
        invVariance = rsqrt(variance + epsilon);    
        if(ylid==0){
            varbuff[varstashindex] = invVariance;//stash mean
        }
        #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
            if(get_global_id(1) == 0){
                //Save mean and calculate and save running mean
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
                    resultSaveInvVariance[xgid] = invVariance;
                #endif

                #if (MIO_RUNNING_RESULT == 1)
                    // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                    // right:: (1 - p)*(b/b-1)*var(n) = (1 - p)*adjust = -p*adjust + adjust
                    // var(n+1) = (p* var(n-1)) +  (-p*adjust + adjust)
                    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;
                    const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*(NHW/(NHW - 1));
                    const _FLOAT tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
                    resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], (_FLOAT)tmp);
                #endif
            }
        #endif
    #endif

}



__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialVariance(
                                    const __global _FLOAT 	* __restrict in, /* x input */
                                    __global _FLOAT		* __restrict meanvarbuff
){
    
    //SPATIAL
    _FLOAT variance   = 0.;
    _FLOAT mean ,elemStd;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_mean;
    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int index;   
 
    unsigned int meanstashindex = cidx + ygrp_sz*ygrp_id + 1;
    
    if(ylid==0){
        lcl_mean = meanvarbuff[meanstashindex];//load stashed mean
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_HW){
        mean = lcl_mean;
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd = (in[index] - mean);
            variance += elemStd*elemStd;
        }  
    }
      
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);    
    _FLOAT tmp = 0.;
    if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[ylid];
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    
    if(ylid==63){
	unsigned int varindex = cidx + ygrp_sz*ygrp_id + 2;
        meanvarbuff[varindex] = variance;//pre-stage for group reduction
    }
}//end spatial variance




__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialFinalMean(
                                        __global _FLOAT		* __restrict meanvarbuff
                                        ,float                  INHW
#if (MIO_RUNNING_RESULT == 1)
					,double			expAvgFactor /* input momentum */                                            
					, __global _FLOAT	* __restrict resultRunningMean /*input and output*/
#endif
#if (MIO_SAVE_MEAN_VARIANCE == 1)
					, __global _FLOAT	* __restrict resultSaveMean /*output only*/
#endif
){

    __private _FLOAT mean = 0.;

    unsigned int ylid    = get_local_id(1); 
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);

    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx    = xgid*MIO_BN_HW;
    
    for(int gn = 0; gn<yngrps; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int meanindex   = cidx + ygrp_sz*offset;
        if(offset < yngrps){//modify to span larger number of groups
            mean += meanvarbuff[meanindex];
        }
    }    
    
    #if (MIO_BN_NGRPS > 64)
        __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

	lcl_data[ylid] = mean;
	barrier(CLK_LOCAL_MEM_FENCE);
        _FLOAT tmp = 0.;
        if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
        barrier(CLK_LOCAL_MEM_FENCE);
        if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
        barrier(CLK_LOCAL_MEM_FENCE);

        mean = lcl_data[ylid];
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
        mean += tmp;
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
        mean += tmp;
        mean *=INHW;
        if(ylid==63){	
            unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
            meanvarbuff[meanstashindex] = mean;//stash mean
    //        printf("Final mean: %f\n" ,mean);
        }
        
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
            unsigned int ygid = get_global_id(1);
            if(ygid == 63){
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
                    resultSaveMean[xgid] = mean;
                #endif

                #if (MIO_RUNNING_RESULT == 1)
                    _FLOAT pvt_runMean      = resultRunningMean[xgid];
                    _FLOAT pvt_newRunMean   = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
                    resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
                #endif
            }
        #endif
    #elif(MIO_BN_NGRPS > 16)
        _FLOAT tmp = 0.;
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
        mean += tmp;
        tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
        mean += tmp;
        mean *=INHW;
        if(ylid==63){	
            unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
            meanvarbuff[meanstashindex] = mean;//stash mean
      //      printf("Final mean: %f\n" ,mean);
        }
                //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
            unsigned int ygid = get_global_id(1);
            if(ygid == 63){
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
                    resultSaveMean[xgid] = mean;
                #endif

                #if (MIO_RUNNING_RESULT == 1)
                    _FLOAT pvt_runMean      = resultRunningMean[xgid];
                    _FLOAT pvt_newRunMean   = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
                    resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
                #endif
            }
        #endif
    #else

        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));
        mean *=INHW;
        if(ylid==0){	
            unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
            meanvarbuff[meanstashindex] = mean;//stash mean
           //printf("Final mean: %f, INHW: %f\n" ,mean, INHW);
        }    
        
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
            unsigned int ygid = get_global_id(1);
            if(ygid == 0){
                #if (MIO_SAVE_MEAN_VARIANCE == 1)
                    resultSaveMean[xgid] = mean;
                #endif

                #if (MIO_RUNNING_RESULT == 1)
                    _FLOAT pvt_runMean      = resultRunningMean[xgid];
                    _FLOAT pvt_newRunMean   = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
                    resultRunningMean[xgid] = mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
                #endif
            }
        #endif
    #endif
}
   


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialMean(const __global _FLOAT    * __restrict in,  
                                           __global _FLOAT          * __restrict meanbuff){

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    //printf("HEllllllllllllllllllllllloooooo\n");
    unsigned int ylid       = get_local_id(1); 
    unsigned int ygrp_id    = get_group_id(1);
    unsigned int xgid       = get_global_id(0);
    unsigned int ygid       = get_global_id(1);
    unsigned int ygrp_sz    = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    _FLOAT mean =0.;

    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            //if(index >= MIO_BN_CHW*MIO_BN_N) printf("ERROR in index: %d\n",index);
            mean += in[index];
        }  
    }
 
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);    
    _FLOAT tmp = 0.;
    if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[ylid];
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    if(ylid==63){
        unsigned int meanindex = cidx+ygrp_sz*ygrp_id;//making assumption of n=0 here
        //if(meanindex >= MIO_BN_CHW*MIO_BN_N) printf("ERROR in mean index: %d\n",meanindex);
        meanbuff[meanindex] = mean;
    }
 
    
}//end spatial mean kernel














#endif

