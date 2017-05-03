

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



static inline void ReduceKernel(__local _FLOAT * lcl_mem, int sum_stride, int unit_id, int unit_len){
    _FLOAT sum = 0;
    int lcl_offset = unit_id * unit_len;
    
    #pragma unroll
    for(int i = 0; i < unit_len && (lcl_offset+i)<MIO_BN_LDS_SIZE; i += sum_stride){
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}




__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdTrainSpatialSingleNorm(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        __constant _FLOAT   * __restrict scale,
                        __constant _FLOAT   * __restrict bias,
                        double                  INHW,
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
    _FLOAT variance   = 0.;
    _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_bias   = 0.;
    _FLOAT elemStd    = expAvgFactor;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;

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
    mean = lcl_data[0]*INHW;///NHW;
    
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
    variance = lcl_data[0]*INHW;
   
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    //DONE WITH variance
 
    pvt_scale   = scale[xgid];
    pvt_bias    = bias[xgid];
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + cidx + ygid;
            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (in[index]-mean)*invVariance;
            // #5 Gamma and Beta adjust
            //y_i = gamma*x_hat + beta
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

    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    
    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    pvt_scale   = scale[xgid];
    pvt_bias    = bias[xgid];
    
    unsigned int meanstashindex = cidx+ygrp_sz*ygrp_id+1;
    unsigned int varstashindex  = cidx+ygrp_sz*ygrp_id+3;
    
    if(meanstashindex >= MIO_BN_CHW*MIO_BN_N) printf("meanstashindex in norm is %d > || = %d\n", meanstashindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
    else mean        = out[meanstashindex];//load stashed mean
    
    if(varstashindex >= MIO_BN_CHW*MIO_BN_N) printf("varstashindex in norm is %d > || = %d\n", varstashindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
    else invVariance = out[varstashindex];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + cidx + ygid;
            if(index >= MIO_BN_CHW*MIO_BN_N) printf("Output index is %d > || = %d\n", index, MIO_BN_CHW*MIO_BN_N);
            else{//TODO: DLOWELL debug
                inhat = (in[index]-mean)*invVariance;
                // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
                out[index] = mad(pvt_scale, inhat, pvt_bias);
            }            
        }//end for(n)
    }//end if(inImgIndex)
}//end spatial norm











__kernel void BatchNormFwdTrainSpatialFinalVariance(
                        __global _FLOAT		* __restrict varbuff,
                        double                  expAvgFactor,
#if (MIO_RUNNING_RESULT == 1)
                        __global _FLOAT         * __restrict resultRunningVariance,
#endif
                        double                  epsilon
#if (MIO_SAVE_MEAN_VARIANCE == 1)
                        , __global _FLOAT       * __restrict resultSaveInvVariance
#endif
){



  //SPATIAL
    __private _FLOAT variance   = 0.;
    __private _FLOAT invVariance= expAvgFactor;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx = xgid*MIO_BN_HW;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    lcl_data[ylid] = 0.;//zero out local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int gn = 0; gn<yngrps; gn++){//TODO: span multiple, is this right?
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int varindex   = cidx + ygrp_sz*offset + 2;
        if(offset < yngrps){//modify to span larger number of groups
            if(varindex >= MIO_BN_CHW*MIO_BN_N) printf("varindex in final variance is %d > || = %d\n", varindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
            else lcl_data[ylid] += varbuff[varindex];//load per group variance
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

    unsigned int varstashindex=cidx+ygrp_sz*ygrp_id+3;
    if(varstashindex >= MIO_BN_CHW*MIO_BN_N) printf("varstashindex in final variance is %d > || = %d\n", varstashindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
    else varbuff[varstashindex] = invVariance;//stash mean

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




__kernel void BatchNormFwdTrainSpatialVariance(
                                    const __global _FLOAT 	* __restrict in, /* x input */
                                    __global _FLOAT		* __restrict meanvarbuff
){
    
    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT variance   = 0.;
    _FLOAT elemStd    = 0.;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int index;   
 
    unsigned int meanstashindex = cidx + ygrp_sz*ygrp_id + 1;
    if(meanstashindex >= MIO_BN_CHW*MIO_BN_N) printf("meanstashindex in variance is %d > || = %d\n", meanstashindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
    else mean = meanvarbuff[meanstashindex];//load stashed mean

    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in variance is %d > || = %d\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
            else elemStd = (in[index] - mean);
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
        if(varindex >= MIO_BN_CHW*MIO_BN_N) printf("varindex in variance is %d > || = %d\n", varindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
	else meanvarbuff[varindex] = lcl_data[0];//pre-stage for group reduction
    }
}//end spatial variance





__kernel void BatchNormFwdTrainSpatialFinalMean(
                                        __global _FLOAT		* __restrict meanvarbuff, 
					double			expAvgFactor /* input momentum */                                            
#if (MIO_RUNNING_RESULT == 1)
					, __global _FLOAT	* __restrict resultRunningMean /*input and output*/
#endif
#if (MIO_SAVE_MEAN_VARIANCE == 1)
					, __global _FLOAT	* __restrict resultSaveMean /*output only*/
#endif
){

    __private _FLOAT mean = expAvgFactor;
	
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
    for(int gn = 0; gn<yngrps; gn++){//TODO: span multiple, is this right?
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int meanindex  = cidx + ygrp_sz*offset;
        if(offset < yngrps){//modify to span larger number of groups
            if(meanindex >= MIO_BN_CHW*MIO_BN_N) printf("meanindex in final mean is %d > || = %d\n", meanindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
            else lcl_data[ylid] += meanvarbuff[meanindex];
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
    if(meanstashindex >= MIO_BN_CHW*MIO_BN_N) printf("meanstashindex in final mean is %d > || = %d\n", meanstashindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
    else meanvarbuff[meanstashindex] = mean;//stash mean
}
   



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
            if(index >= MIO_BN_CHW*MIO_BN_N) printf("index in mean is %d > || = %d\n", index, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
            else mean += in[index];
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
        if(meanindex >= MIO_BN_CHW*MIO_BN_N) printf("meanindex in mean is %d > || = %d\n", meanindex, MIO_BN_CHW*MIO_BN_N);//TODO: DLOWELL debug
        else meanbuff[meanindex] = lcl_data[0];//pre-stage for group reduction
    }
}//end spatial mean kernel




//====================== END SPATIAL ========================
