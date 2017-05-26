__kernel void ScaleTensor(global MIOPEN_TYPE * __restrict dst, MIOPEN_ALPHA_TYPE alpha, long num_elems)
{
    uint gid = get_global_id(0);
    if(gid < num_elems) {
        dst[gid] *= alpha;
    }
}

__kernel void SetTensor(global MIOPEN_TYPE * __restrict dst, MIOPEN_ALPHA_TYPE alpha, long num_elems)
{
    uint gid = get_global_id(0);
    if(gid < num_elems) {
        dst[gid] = alpha;
    }
}

