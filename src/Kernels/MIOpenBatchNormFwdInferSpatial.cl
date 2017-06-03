

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

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
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

#if(MIO_BN_VARIANT == 0)


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdInferSpatialEst(
					const __global _FLOAT 	* __restrict in, /* x input */
					__global _FLOAT         * __restrict out, /* y output */
					const __global _FLOAT	* __restrict estimatedMean,
					const __global _FLOAT	* __restrict estimatedVariance,
					const __global _FLOAT	* __restrict scale,
					const __global _FLOAT	* __restrict bias,
					double 			epsilon){

    int xgid = get_global_id(0);
    int ygid = get_global_id(1);

    local _FLOAT lmean;
    local _FLOAT lvar;
    local _FLOAT lscale;
    local _FLOAT lbias;

    unsigned int cidx = xgid*MIO_BN_HW;
    unsigned int index;

    _FLOAT mean, variance, invVariance;
    _FLOAT inhat;
    _FLOAT pscale, pbias;

    if(get_local_id(1) == 0){
        lmean   = estimatedMean[xgid];
        lvar    = estimatedVariance[xgid];
        lscale  = scale[xgid];// dims 1xCx1x1
        lbias   = bias[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mean      = lmean;
    variance  = lvar; 
    pscale    = lscale;
    pbias     = lbias;
    invVariance = rsqrt(fabs(variance + epsilon));

    //move across the sections of the image mini_batch stack  

    if(ygid < MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++) {
            index = n*MIO_BN_CHW + cidx + ygid;
            inhat = (in[index] - mean)*invVariance;
            out[index] = mad(pscale, inhat, pbias);// y_i = gamma*x_hat + beta
        }//end for(img_offset) 
    }
}//end spatial norm


//=========================================================

//=== SPATIAL NO SAVED DATA ===============================

#elif(MIO_BN_VARIANT == 1)


__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))
__kernel void BatchNormFwdInferSpatialSingleNorm(
                        const __global _FLOAT   * __restrict in,
                        __global _FLOAT         * __restrict out,
                        const __global _FLOAT   * __restrict scale,
                        const __global _FLOAT   * __restrict bias,
                        double                  epsilon,
                        double                  INHW
){

    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT variance   = 0.;
    _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    _FLOAT pvt_scale  = 0.;
    _FLOAT pvt_bias   = 0.;
    _FLOAT elemStd    = 0.;

    
    local _FLOAT lscale;
    local _FLOAT lbias;
    
  
    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid*MIO_BN_HW;

    if(ylid == 0){
        lscale   = scale[xgid];
        lbias    = bias[xgid];
        
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            mean += in[index];
        }  
    }

#if (MIO_BN_GRP1 > 64)
    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp = 0.;
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
    if(ylid == 63) lcl_data[0] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];
    
#elif (MIO_BN_GRP1 > 16)
    _FLOAT tmp = 0.;
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean =  as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    
#else
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean =  as_float(__builtin_amdgcn_readlane(as_int(mean), 0));
#endif

    //if(ygid==0) printf("premean: %f\n" ,mean);
    mean *= INHW;
    //if(ygid==0) printf("postmean: %f\n" ,mean);
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){
            index = n*MIO_BN_CHW + cidx + ygid;
            elemStd = (in[index] - mean);
            variance += elemStd*elemStd;
        }  
    }

    
#if (MIO_BN_GRP1 > 64)
    tmp = 0.;
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128) lcl_data[ylid] += lcl_data[ylid+128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64) lcl_data[ylid] += lcl_data[ylid+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[ylid];
    barrier(CLK_LOCAL_MEM_FENCE);
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    if(ylid == 63) lcl_data[0] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0];
    
#elif (MIO_BN_GRP1 > 16)
    tmp = 0.;
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp   = as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 63));    
    
#else
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 0));
#endif
    //if(ygid==0) printf("pre variance: %f\n" ,variance);
    variance *= INHW;
    //if(ygid==0) printf("post variance: %f, inhw: %f\n" ,variance,INHW);
    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    // #4 apply the normalization
    // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    if(ygid<MIO_BN_HW){
        
        pvt_scale   = lscale;
        pvt_bias    = lbias;        
        
        #pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + cidx + ygid;
            inhat = (in[index]-mean)*invVariance;
            // #5 Gamma and Beta adjust
            //y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        }//end for
    }//end if

}//end spatial norm



#elif (MIO_BN_VARIANT == 2)




__kernel void BatchNormFwdInferSpatialNorm(
			const __global _FLOAT 	* __restrict in, 
			__global _FLOAT		* __restrict out, 
			const __global _FLOAT 	* __restrict scale, 
			const __global _FLOAT	* __restrict bias  
){

    //SPATIAL
    __private _FLOAT mean       = 0.;
    __private _FLOAT invVariance= 0.;
    _FLOAT inhat      = 0.;
    __private _FLOAT pvt_scale  = 0.;
    __private _FLOAT pvt_bias   = 0.;

    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    
    // #4 apply the normalization
    // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    pvt_scale   = scale[xgid];
    pvt_bias    = bias[xgid];
    
    unsigned int meanstashindex = cidx+ygrp_sz*ygrp_id+1;
    unsigned int varstashindex = cidx+ygrp_sz*ygrp_id+3;
    
    mean        = out[meanstashindex];//load stashed mean
    invVariance = out[varstashindex];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
//  if(ygid==0) printf("mean: %f, ivar: %f\n", mean, invVariance); 
    if(ygid<MIO_BN_HW){
        #pragma unroll
        for(int n = 0; n < MIO_BN_N; n++){//apply normalization
            index = n*MIO_BN_CHW + cidx + ygid;
            inhat = (in[index]-mean)*invVariance;
            out[index] =  mad(pvt_scale, inhat, pvt_bias);// #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
        }//end for(n)
    }//end if(inImgIndex)
}//end spatial norm



__kernel void BatchNormFwdInferSpatialFinalVariance(
                        __global _FLOAT		* __restrict varbuff,
                        double                  epsilon){

  //SPATIAL
    _FLOAT variance   = 0.;
    _FLOAT invVariance= 0.;

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx = xgid*MIO_BN_HW;

    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;

    for(int gn = 0; gn<yngrps; gn++){
        unsigned int offset     = gn*ygrp_sz+ylid;
        unsigned int varindex   = cidx + ygrp_sz*offset + 2;
        if(offset < yngrps){//modified to span larger number of groups
            variance += varbuff[varindex];//load per group variance
        }
    }
#if (MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    local _FLOAT lvar;

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
    if(ylid==63) lvar =  variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvar;
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
    variance =  as_float(__builtin_amdgcn_readlane(as_int(variance), 63));

#else
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float( __builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));

#endif
  //if(ylid==0) printf("variance: %f\n",  variance);

    variance /= NHW;

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    //DONE WITH variance

    if(ylid ==0){
        unsigned int varstashindex=cidx+ygrp_sz*ygrp_id+3;
        varbuff[varstashindex] = invVariance;//stash mean
    }
}




__kernel void BatchNormFwdInferSpatialVariance(
                                    const __global _FLOAT 	* __restrict in, 
                                    __global _FLOAT		* __restrict meanvarbuff){
    
    //SPATIAL
    _FLOAT mean       = 0.;
    _FLOAT elemStd    = 0.;
    _FLOAT variance = 0.;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index, ncIdx;
    unsigned int cidx = xgid*MIO_BN_HW;
    
    unsigned int meanstashindex = cidx + ygrp_sz*ygrp_id + 1;
    mean = meanvarbuff[meanstashindex];//load stashed mean
        
    if(ygid<MIO_BN_HW){
	#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++){
            ncIdx = n*MIO_BN_CHW + cidx;
            index = ncIdx + ygid;
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


__kernel void BatchNormFwdInferSpatialFinalMean(
					__global _FLOAT			* __restrict meanvarbuff){

     _FLOAT mean = 0.;

    unsigned int ylid = get_local_id(1); 
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygrp_sz = get_local_size(1);  
    unsigned int yngrps  = get_num_groups(1);
    unsigned int cidx = xgid*MIO_BN_HW;
    _FLOAT NHW  = (_FLOAT) MIO_BN_NHW;
    
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
        if(ylid==63){	
            unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
            meanvarbuff[meanstashindex] = mean/NHW;//stash mean
    //        printf("Final mean: %f\n" ,mean);
        }
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
        if(ylid==63){	
            unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
            meanvarbuff[meanstashindex] = mean/NHW;//stash mean
      //      printf("Final mean: %f\n" ,mean);
        }
    #else
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
        mean += as_float( __builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));
        if(ylid==0){	
            unsigned int meanstashindex=cidx+ygrp_sz*ygrp_id+1;
            meanvarbuff[meanstashindex] = mean/NHW;//stash mean
        //    printf("Final mean: %f\n" ,mean);
        }
    #endif

}
   



__kernel void BatchNormFwdInferSpatialMean(const __global _FLOAT    * __restrict in,  
                                           __global _FLOAT          * __restrict meanbuff){

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    _FLOAT mean = 0.;
    unsigned int ylid       = get_local_id(1); 
    unsigned int ygrp_id    = get_group_id(1);
    unsigned int xgid       = get_global_id(0);
    unsigned int ygid       = get_global_id(1);
    unsigned int ygrp_sz    = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid*MIO_BN_HW;
    
    //move across the sections of the image mini_batch stack 
    if(ygid<MIO_BN_HW){

    	#pragma unroll
	for(unsigned int n = 0; n < MIO_BN_N; n++){
       	    index = n*MIO_BN_CHW + cidx + ygid;
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
         meanbuff[meanindex] = mean;//pre-stage for group reduction
//printf("Init mean: %f\n",mean);
    }
}//end spatial mean kernel



//====================================================

#endif
