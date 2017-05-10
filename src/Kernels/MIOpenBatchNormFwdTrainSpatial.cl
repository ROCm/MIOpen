

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
#define MIO_BN_VARIANT 0
#endif



/*
__attribute__((always_inline))
static inline void ReduceKernel(__local _FLOAT * lcl_mem, int sum_stride, int unit_id, int unit_len){
    _FLOAT sum = 0;
    int lcl_offset = unit_id * unit_len;
    
    #pragma unroll
    for(int i = 0; i < unit_len ; i += sum_stride){
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

#if (MIO_BN_VARIANT==0)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleNorm(
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
    _FLOAT variance   = 0.;
    _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    _FLOAT elemStd    = 0.;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT2 lcl_scalebias;
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    //unsigned int idx = cidx+ygid;
    
    lcl_scalebias[0] = scale[xgid];
    lcl_scalebias[1] = bias[xgid];

    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
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
    mean = lcl_data[0]*INHW;///NHW;
    
    if(ygid<MIO_BN_N){
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
            elemStd = (in[index] - mean);
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
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    //DONE WITH variance
 
    if(ygid<MIO_BN_N){
        
        pvscale = lcl_scalebias[0];
        pvbias  = lcl_scalebias[1];
        
        #pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++){
            index = ygid*MIO_BN_CHW + cidx + hw;
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
            const _FLOAT tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm

#elif (MIO_BN_VARIANT==1)


/*
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleVecNorm(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT   * __restrict scale,
                        __constant _FLOAT   * __restrict bias,
                        float                  INHW,
                        double                  expAvgFactor,
#if (MIO_RUNNING_RESULT == 1)          
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
    _FLOAT4 vmean     = 0.;
    
    _FLOAT variance   = 0.;
    _FLOAT4 vvariance = 0.;
    
    //_FLOAT invVariance= 0.;
    _FLOAT4 vinvVariance= 0.;
    
    _FLOAT4 vinhat      = 0.;
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_bias   = 0.;
    _FLOAT elemStd    = expAvgFactor;

    __local _FLOAT lcl_data[4][MIO_BN_LDS_SIZE];
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    //unsigned int ygid = get_global_id(1);
    
    unsigned int tsubgroup = ylid/64;
    unsigned int sub_ylid  = (ylid%(get_local_size(1)>>2));
    
    unsigned int channel = 4*xgid+tsubgroup;
    unsigned int cidx = MIO_BN_HW*channel;
    unsigned int idx = cidx + sub_ylid;

    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            mean += in[index];
        }  
    }
    lcl_data[tsubgroup][sub_ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sub_ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data[tsubgroup], 1, sub_ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data[tsubgroup], 4, sub_ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid == 0) ReduceKernel(lcl_data[tsubgroup], 16, sub_ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    vmean[tsubgroup] = lcl_data[tsubgroup][0]*INHW;
    //barrier(CLK_LOCAL_MEM_FENCE);
    //vmean = vmean*INHW;
    
    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
           // if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in is %d > || = %d in variance\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
           // else{
                elemStd = (in[index] - vmean[tsubgroup]);
                variance = mad(elemStd,elemStd,variance);
          //  }
        }  
    }
    lcl_data[tsubgroup][sub_ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(sub_ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data[tsubgroup], 1, sub_ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data[tsubgroup], 4, sub_ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid == 0) ReduceKernel(lcl_data[tsubgroup], 16, sub_ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    vvariance[tsubgroup] = lcl_data[tsubgroup][0]*INHW;
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    vinvVariance = rsqrt(vvariance + (float)epsilon);
    //DONE WITH variance
    if(channel < MIO_BN_C){
        pvt_scale   = scale[channel];
        pvt_bias    = bias[channel];
    }
    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
         //   if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in is %d > || = %d in norm\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
         //   else{
                // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
                vinhat = (in[index]-vmean[tsubgroup])*vinvVariance[tsubgroup];
                // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
         //       printf("index: %d, n: %d, tsubgroup: %d, ylid: %d, sub_ylid: %d, xgid: %d, cidx: %d, channel: %d, idx: %d\n", index, n, tsubgroup, ylid, sub_ylid, xgid, cidx, channel, idx);
                out[index] = mad(pvt_scale, vinhat[tsubgroup], pvt_bias);
       //     }
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(sub_ylid == 0){
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[channel] = vmean[tsubgroup];
            resultSaveInvVariance[channel] = vinvVariance[tsubgroup];
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[channel];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[channel] = mad(vmean[tsubgroup], (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            _FLOAT4 adjust = (MIO_BN_NHW == 1) ? vvariance : vvariance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            _FLOAT4 tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[channel] = mad((_FLOAT)expAvgFactor,resultRunningVariance[channel], tmp[tsubgroup]);
        #endif    
    }
    #endif
}//end spatial norm
*/

/*
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleVecNorm(
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
    _FLOAT4 vmean     = 0.;
    
    _FLOAT variance   = 0.;
    _FLOAT4 vvariance = 0.;
    
    //_FLOAT invVariance= 0.;
    _FLOAT4 vinvVariance= 0.;
    
    _FLOAT4 vinhat      = 0.;
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_bias   = 0.;
    _FLOAT elemStd    = 0.;

    __local _FLOAT lcl_indata[4][MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_data[4][MIO_BN_LDS_HWSIZE];
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    
    unsigned int tsubgroup = ylid/64;
    unsigned int sub_ylid  = (ylid%(get_local_size(1)>>2));
    
    unsigned int channel = 4*xgid+tsubgroup;
    unsigned int cidx = MIO_BN_HW*channel;
    unsigned int idx = cidx + sub_ylid;

    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            lcl_indata[tsubgroup][n][sub_ylid] = in[index];
            mean += lcl_indata[tsubgroup][n][sub_ylid];
        }  
    }
    lcl_data[tsubgroup][sub_ylid] = mean;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_data[tsubgroup], 1, sub_ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_data[tsubgroup], 4, sub_ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid == 0) ReduceKernel(lcl_data[tsubgroup], 16, sub_ylid, MIO_BN_LDS_HWSIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
     
    //#pragma unroll
   // for(int hw = 0; hw < MIO_BN_HW; hw++){
   //     vmean[tsubgroup] += lcl_data[tsubgroup][hw];
  //  }
   // vmean = vmean*INHW;
    //barrier(CLK_LOCAL_MEM_FENCE);
    vmean[tsubgroup] = lcl_data[tsubgroup][0]*INHW;
    //barrier(CLK_LOCAL_MEM_FENCE);
    //vmean = vmean*INHW;
    
    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
           // if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in is %d > || = %d in variance\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
           // else{
                elemStd = (lcl_indata[tsubgroup][n][sub_ylid] - vmean[tsubgroup]);
                variance = mad(elemStd,elemStd,variance);
          //  }
        }  
    }
    lcl_data[tsubgroup][sub_ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_data[tsubgroup], 1, sub_ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_data[tsubgroup], 4, sub_ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid == 0) ReduceKernel(lcl_data[tsubgroup], 16, sub_ylid, MIO_BN_LDS_HWSIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    vvariance[tsubgroup] = lcl_data[tsubgroup][0]*INHW;
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    vinvVariance = rsqrt(vvariance + (float)epsilon);
    //DONE WITH variance
    if(channel < MIO_BN_C){
        pvt_scale   = scale[channel];
        pvt_bias    = bias[channel];
    }
    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
         //   if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in is %d > || = %d in norm\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
         //   else{
                // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
                vinhat = (lcl_indata[tsubgroup][n][sub_ylid]-vmean[tsubgroup])*vinvVariance[tsubgroup];
                // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
         //       printf("index: %d, n: %d, tsubgroup: %d, ylid: %d, sub_ylid: %d, xgid: %d, cidx: %d, channel: %d, idx: %d\n", index, n, tsubgroup, ylid, sub_ylid, xgid, cidx, channel, idx);
                out[index] = mad(pvt_scale, vinhat[tsubgroup], pvt_bias);
       //     }
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(sub_ylid == 0){
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[channel] = vmean[tsubgroup];
            resultSaveInvVariance[channel] = vinvVariance[tsubgroup];
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[channel];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[channel] = mad(vmean[tsubgroup], (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            _FLOAT4 adjust = (MIO_BN_NHW == 1) ? vvariance : vvariance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            _FLOAT4 tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[channel] = mad((_FLOAT)expAvgFactor,resultRunningVariance[channel], tmp[tsubgroup]);
        #endif    
    }
    #endif
}//end spatial norm
*/




/*



__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleVecNorm(
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
    _FLOAT2 vmean     = 0.;
    
    _FLOAT variance   = 0.;
    _FLOAT2 vvariance = 0.;
    
    //_FLOAT invVariance= 0.;
    _FLOAT2 vinvVariance= 0.;
    
    _FLOAT2 vinhat      = 0.;
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_bias   = 0.;
    _FLOAT elemStd    = 0.;

    __local _FLOAT lcl_indata[2][MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_data[2][MIO_BN_LDS_HWSIZE];
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    
    unsigned int tsubgroup = ylid/128;
    unsigned int sub_ylid  = (ylid%(get_local_size(1)>>1));
    
    unsigned int channel = 2*xgid+tsubgroup;
    unsigned int cidx = MIO_BN_HW*channel;
    unsigned int idx = cidx + sub_ylid;

    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            lcl_indata[tsubgroup][n][sub_ylid] = in[index];
            mean += lcl_indata[tsubgroup][n][sub_ylid];
        }  
    }
    lcl_data[tsubgroup][sub_ylid] = mean;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_data[tsubgroup], 1, sub_ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_data[tsubgroup], 4, sub_ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid == 0) ReduceKernel(lcl_data[tsubgroup], 16, sub_ylid, MIO_BN_LDS_HWSIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
     
    //#pragma unroll
   // for(int hw = 0; hw < MIO_BN_HW; hw++){
   //     vmean[tsubgroup] += lcl_data[tsubgroup][hw];
  //  }
   // vmean = vmean*INHW;
    //barrier(CLK_LOCAL_MEM_FENCE);
    vmean[tsubgroup] = lcl_data[tsubgroup][0]*INHW;
    //barrier(CLK_LOCAL_MEM_FENCE);
    //vmean = vmean*INHW;
    
    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
           // if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in is %d > || = %d in variance\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
           // else{
                elemStd = (lcl_indata[tsubgroup][n][sub_ylid] - vmean[tsubgroup]);
                variance = mad(elemStd,elemStd,variance);
          //  }
        }  
    }
    lcl_data[tsubgroup][sub_ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_data[tsubgroup], 1, sub_ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_data[tsubgroup], 4, sub_ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(sub_ylid == 0) ReduceKernel(lcl_data[tsubgroup], 16, sub_ylid, MIO_BN_LDS_HWSIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    vvariance[tsubgroup] = lcl_data[tsubgroup][0]*INHW;
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    vinvVariance = rsqrt(vvariance + (float)epsilon);
    //DONE WITH variance
    if(channel < MIO_BN_C){
        pvt_scale   = scale[channel];
        pvt_bias    = bias[channel];
    }
    if(sub_ylid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
         //   if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in is %d > || = %d in norm\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
         //   else{
                // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
                vinhat = (lcl_indata[tsubgroup][n][sub_ylid]-vmean[tsubgroup])*vinvVariance[tsubgroup];
                // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
         //       printf("index: %d, n: %d, tsubgroup: %d, ylid: %d, sub_ylid: %d, xgid: %d, cidx: %d, channel: %d, idx: %d\n", index, n, tsubgroup, ylid, sub_ylid, xgid, cidx, channel, idx);
                out[index] = mad(pvt_scale, vinhat[tsubgroup], pvt_bias);
       //     }
        }//end for
    }//end if

    #if (MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(sub_ylid == 0){
        //Save mean and calculate and save running mean
        #if (MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[channel] = vmean[tsubgroup];
            resultSaveInvVariance[channel] = vinvVariance[tsubgroup];
        #endif

        #if (MIO_RUNNING_RESULT == 1)
            _FLOAT pvt_runMean = resultRunningMean[channel];
            _FLOAT pvt_newRunMean = mad((_FLOAT)-expAvgFactor,pvt_runMean, pvt_runMean);//tmp = oldRunMean*(1-factor)
            resultRunningMean[channel] = mad(vmean[tsubgroup], (_FLOAT)expAvgFactor, pvt_newRunMean);//newMean*factor + tmp
            _FLOAT4 adjust = (MIO_BN_NHW == 1) ? vvariance : vvariance*((_FLOAT)MIO_BN_NHW/(_FLOAT)(MIO_BN_NHW - 1.0));
            _FLOAT4 tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[channel] = mad((_FLOAT)expAvgFactor,resultRunningVariance[channel], tmp[tsubgroup]);
        #endif    
    }
    #endif
}//end spatial norm

*/


/*
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleLDSNorm(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT   * __restrict scale,
                        __constant _FLOAT   * __restrict bias,
                        double                  INHW,
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
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_bias   = 0.;
    _FLOAT elemStd    = expAvgFactor;

    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_data[MIO_BN_LDS_HWSIZE];
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int idx = cidx+ygid;

    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            lcl_indata[n][ylid] = in[index];
            mean += lcl_indata[n][ylid];
        }  
    }
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_HWSIZE);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0]*INHW;///NHW;
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + idx;
            elemStd = (lcl_indata[n][ylid] - mean);
            variance = mad(elemStd,elemStd,variance);
        }  
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_HWSIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0]*INHW;
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    //DONE WITH variance
 
    pvt_scale   = scale[xgid];
    pvt_bias    = bias[xgid];
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (lcl_indata[n][ylid]-mean)*invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
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
            const _FLOAT tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm
*/



__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleLDSNorm(
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
    _FLOAT variance   = 0.;
    _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    _FLOAT elemStd    = 0.;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_reduc[MIO_BN_LDS_HWSIZE];
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
            lcl_indata[n][ylid] = in[index];
            mean += lcl_indata[n][ylid];
        }  
    }
    
    lcl_reduc[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_reduc, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_reduc, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_reduc, 16, ylid, MIO_BN_LDS_HWSIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_reduc[0]*INHW;///NHW;
    
    #pragma unroll
    for(int n = 0; n < MIO_BN_N; n++){
        elemStd = (lcl_indata[n][ylid] - mean);
        variance = mad(elemStd,elemStd,variance);
    }  
    lcl_reduc[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_HWSIZE >> 2)) ReduceKernel(lcl_reduc, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_HWSIZE >> 4)) ReduceKernel(lcl_reduc, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_reduc, 16, ylid, MIO_BN_LDS_HWSIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_reduc[0]*INHW;

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    
    //DONE WITH variance
    if(ygid<MIO_BN_HW){
        
        pvscale = lcl_scalebias[0];
        pvbias  = lcl_scalebias[1];
        
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (lcl_indata[n][ylid]-mean)*invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
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
            const _FLOAT tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm




#elif (MIO_BN_VARIANT==2)



__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleNorm(
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
            index = n*MIO_BN_CHW + idx;
            elemStd = (in[index] - mean);
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
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
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
            const _FLOAT tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
            resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], tmp);
        #endif    
    }
    #endif
}//end spatial norm




#elif (MIO_BN_VARIANT==3)

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

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int varstashindex=cidx+ygrp_sz*ygrp_id+3;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    lcl_data[ylid] = 0.;//zero out local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    #pragma unroll
    for(int gn = 0; gn<MIO_BN_NGRPS; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int varindex   = cidx + ygrp_sz*offset + 2;
        if(offset < yngrps){//modify to span larger number of groups
            lcl_data[ylid] += varbuff[varindex];//load per group variance
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    #pragma unroll
    for(unsigned int i =0;i<MIO_BN_LDS_SIZE;i++){
        variance += lcl_data[i];
    }
    variance /= NHW;

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    //DONE WITH variance
    //stash inverse variance

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
                const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance*(NHW/(NHW - 1));
                const _FLOAT tmp = mad((_FLOAT)-expAvgFactor,adjust,adjust);
                resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor,resultRunningVariance[xgid], (_FLOAT)tmp);
            #endif
        }
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

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2)) ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4)) ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0) ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
      
    if(ylid==0){
	unsigned int varindex = cidx + ygrp_sz*ygrp_id + 2;
        meanvarbuff[varindex] = lcl_data[0];//pre-stage for group reduction
    }
}//end spatial variance




__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialFinalMean(
                                        __global _FLOAT		* __restrict meanvarbuff
#if (MIO_RUNNING_RESULT == 1)
					,double			expAvgFactor /* input momentum */                                            
					, __global _FLOAT	* __restrict resultRunningMean /*input and output*/
#endif
#if (MIO_SAVE_MEAN_VARIANCE == 1)
					, __global _FLOAT	* __restrict resultSaveMean /*output only*/
#endif
){

    __private _FLOAT mean = 0.;
	
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid    = get_local_id(1); 
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);

    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx    = xgid*MIO_BN_HW;
        
    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;
    
    mean = 0.;
    lcl_data[ylid] = 0.0;//zero out local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int gn = 0; gn<MIO_BN_NGRPS; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int meanindex  = cidx + ygrp_sz*offset;
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

    unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
    meanvarbuff[meanstashindex] = mean;//stash mean
}
   


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialMean(const __global _FLOAT    * __restrict in,  
                                           __global _FLOAT          * __restrict meanbuff){

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

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


#endif

//====================== END SPATIAL ========================
