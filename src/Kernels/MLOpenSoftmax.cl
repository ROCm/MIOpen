kernel void SoftmaxForward(global float *y, const int c, const long sz) 
{
#if 1
	local float l_helper[64];
	local float channel_helper[1];
	
	int n = get_group_id(0);
	int lid = get_local_id(0);

	for(n = get_group_id(0); n < sz; n += get_num_groups(0)) {
		channel_helper[0] = -FLT_MAX;
		l_helper[lid] = -FLT_MAX;

		float t_helper = -FLT_MAX;

		// Compute max per channel
		for(int i = lid; i < c; i += get_local_size(0)) {
			t_helper = max(y[n*c + i], t_helper);
		}

		l_helper[lid] = t_helper;
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int i = (get_local_size(0)>>1); i > 0; i >>= 1) {
			if(lid < i) {
				l_helper[lid] = max(l_helper[lid], l_helper[lid+i]);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(lid == 0) {
			channel_helper[0] = max(l_helper[0], channel_helper[0]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// subtract channel_max from each value
		for(int i = lid; i < c; i += get_local_size(0)) {
			y[n*c + i] -= channel_helper[0];

			// exponent of each value
			y[n*c + i] = exp(y[n*c+i]);
		}

		// compute sum per channel
		t_helper = 0.;
		channel_helper[0] = 0.;
		for(int i = lid; i < c; i += get_local_size(0)) {
			t_helper += y[n*c + i];
		}

		l_helper[lid] = t_helper;
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int i = (get_local_size(0)>>1); i > 0; i >>= 1) {
			if(lid < i) {
				l_helper[lid] += l_helper[lid+i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(lid == 0) {
			channel_helper[0] += l_helper[0];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// normalize each value in the channel by the sum
		for(int i = lid; i < c; i += get_local_size(0)) {
			y[n*c + i] /= channel_helper[0];
		}
	}
#else

	local float l_helper[64];
	local float channel_helper[4];

	int n = get_group_id(0);
	int lid = get_local_id(0);

	int batch_lid = lid & 3;
	int batch = lid >> 2;

	for(n = get_group_id(0); n < sz; n += get_num_groups(0)*4) {
		if(batch_lid == 0)
			channel_helper[batch] = -FLT_MAX;

		l_helper[lid] = -FLT_MAX;

		float t_helper = -FLT_MAX;

		// Compute max per channel
		for(int i = batch_lid; i < c; i += 16 /*get_local_size(0)*/) {
			t_helper = max(y[(n*4+batch)*c + i], t_helper);
		}

		l_helper[lid] = t_helper;
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int i = (16>>1); i > 0; i >>= 1) {
			if(batch_lid < i) {
				l_helper[lid] = max(l_helper[lid], l_helper[lid+i]);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(batch_lid == 0) {
			channel_helper[batch] = max(l_helper[batch*16], channel_helper[batch]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// subtract channel_max from each value
		for(int i = batch_lid; i < c; i += 16/*get_local_size(0)*/) {
			y[(n*4+batch)*c + i] -= channel_helper[batch];

			// exponent of each value
			y[(n*4+batch)*c + i] = exp(y[(n*4+batch)*c+i]);
		}

		// compute sum per channel
		t_helper = 0.;
		if(batch_lid == 0)
			channel_helper[batch] = 0.;
		for(int i = batch_lid; i < c; i += 16/*get_local_size(0)*/) {
			t_helper += y[(n*4+batch)*c + i];
		}

		l_helper[lid] = t_helper;
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int i = (16>>1); i > 0; i >>= 1) {
			if(batch_lid < i) {
				l_helper[lid] += l_helper[lid+i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(batch_lid == 0) {
			channel_helper[batch] += l_helper[batch*16];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// normalize each value in the channel by the sum
		for(int i = batch_lid; i < c; i += 16/*get_local_size(0)*/) {
			y[(n*4+batch)*c + i] /= channel_helper[batch];
		}
	}
#endif
}	
