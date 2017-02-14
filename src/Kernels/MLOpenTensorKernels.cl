/* Only works for NCHW
 */

__kernel void AddTensor(global float *a, 
        const int a_n, const int a_c, const int a_h, const int a_w,
        global float *c,
        const int c_n, const int c_c, const int c_h, const int c_w,
        const int n_not_ones, const int work_per_wg)
{   
    int gid = get_group_id(0);
    int lid = get_local_id(0);

#if N_NOT_ONES == 1 && FIRST_N == 1
    float add = a[gid];
    while(lid < work_per_wg) {

        for(int i = 0; i < c_n; i++) {
            c[i*c_c*c_h*c_w + gid*c_h*c_w + lid] += add;
        }

        lid += 256;
    }
#endif // N_NOT_ONES
}


