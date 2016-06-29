kernel 
void hello_world_kernel (global const float *a,
						global const float *b,
						global float *c,
						const int sz) {

	 const size_t tid = get_global_id(0);

	 if(tid < sz) {
		 c[tid] =  a[tid] + b[tid];
	 }
}
