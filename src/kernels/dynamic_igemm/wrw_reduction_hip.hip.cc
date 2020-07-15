#include <hip/hip_runtime.h>

extern "C"
__global__ __launch_bounds__(256,2)
void wrw_reduction(float* output, float* input, int out_length, int in_stride, int n_groups)
{
    float4 vec_in;
    float4 vec_out;
    int i_len, i_groups;
    
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    int offset = bid * out_length * 256 + tid * out_length;
    
    float* local_in = input + offset;
    float* local_out = output + offset;
    
    for (i_len = 0; i_len < out_length; i_len += 4)
    {
        vec_out = (float4)0;
        for (i_groups = 0; i_groups < n_groups; i_groups++)
        {
            vec_in = *(float4* )(local_in + i_len + in_stride * i_groups);
            vec_out += vec_in;
        }
        *(float4 *)(local_out + i_len) = vec_out;
    }
}
