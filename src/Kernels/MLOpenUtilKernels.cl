#if 0
kernel void Im2Col(global float *im, size_t im_offset,
		const int h, const int w,
		const int wei_h, const int wei_w,
		const int out_h, const int out_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		global float *col)
{
	int tid = get_global_id(0);
	int col_row = tid / (out_h * out_w);

	int im_x = col_row % wei_w;
	int im_y = (col_row / wei_w) % wei_h;
	int im_c = col_row / (wei_w * wei_h);

	int out_x = tid % out_w;
	int out_y = (tid / out_w) % out_h;

	int im_off_h = out_y * stride_h - pad_h + im_y;
	int im_off_w = out_x * stride_w - pad_w + im_x;

	global float *im_off = (global float *)&im[im_offset];

	if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w) {
		col[col_row*out_h*out_w + out_y*out_w + out_x] = im_off[im_c*h*w + im_off_h*w + im_off_w];
	}
	else {
		col[col_row*out_h*out_w + out_y*out_w + out_x] = 0.;
	}
}
#endif
#if 1
kernel void Im2Col(global float *im, size_t im_offset,
		const int h, const int w,
		const int wei_h, const int wei_w,
		const int out_h, const int out_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		global float *col)
{
#define THREADS_PER_CH (256 / NUM_CH_PER_WG)

	global float *im_off = im + im_offset;
	int lid = get_local_id(0);
	int gid = get_group_id(0);

#if NUM_IM_BLKS == 1 && STRIDE_GT_1 == 0 

	// Load image into LDS
	local float local_im[256];

	int witem_ch = lid / THREADS_PER_CH;
	if(lid < NUM_CH_PER_WG*h*w)
		local_im[lid] = im_off[(gid*NUM_CH_PER_WG)*h*w + lid]; 
	barrier(CLK_LOCAL_MEM_FENCE);

	// where will each thread to col
	int witem_ch_offset = witem_ch*h*w;

	if(lid % THREADS_PER_CH < out_h*out_w) {
		int inner_lid = lid % THREADS_PER_CH;
		int out_x = inner_lid % out_w;
		int out_y = inner_lid / out_w;
		
		int col_x = out_y * out_w + out_x;
		int col_y = (gid*NUM_CH_PER_WG+witem_ch) * out_h * out_w * wei_h * wei_w;

		for(int y = 0; y < wei_h; y++) {
			for(int x = 0; x < wei_w; x++) {
				int im_off_h = out_y * stride_h - pad_h + y;
				int im_off_w = out_x * stride_w - pad_w + x;
				if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w)
					col[col_y + col_x + (y*wei_w+x)*out_h*out_w] = local_im[witem_ch_offset + (im_off_h)*h + im_off_w];	
				else
					col[col_y + col_x + (y*wei_w+x)*out_h*out_w] = 0;
			}
		}
	}

#else // NUM_IM_BLKS > 1 || STRIDE_GT_1 1

	local float local_im[LOCAL_MEM_SIZE];

	int witem_ch = lid / THREADS_PER_CH;
	int wg_ch = gid / NUM_IM_BLKS;

	int im_x = ((gid % NUM_IM_BLKS) % NUM_IM_BLKS_X) * 16;
	int im_y = ((gid % NUM_IM_BLKS) / NUM_IM_BLKS_X) * 16;
	
	int out_cols_wg = im_x + 16 <= out_w ? 16 : out_w - im_x;
	int out_rows_wg = im_y + 16 <= out_h ? 16 : out_h - im_y;

	int im_cols_wg = 16*stride_w+ wei_w;
	int inner_lid = lid;

	while (inner_lid < LOCAL_MEM_SIZE) {
		int row_to_use = inner_lid / im_cols_wg;
		int col_to_use = inner_lid % im_cols_wg;
		int lm_offset = row_to_use*im_cols_wg + col_to_use;
		if(im_y + row_to_use >= pad_h && im_y + row_to_use < h+pad_h && im_x + col_to_use >= pad_w && im_x + col_to_use < w+pad_w) {
			int im_off_h = im_y + row_to_use - pad_h;
			int im_off_w = im_x + col_to_use - pad_w;
			local_im[lm_offset] = im_off[wg_ch*h*w + im_off_h*w + im_off_w];
		}
		else
			local_im[lm_offset] = 0;

		inner_lid += 256;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid < out_cols_wg * out_rows_wg) {
		int out_x = lid % out_cols_wg;
		int out_y = lid / out_cols_wg;

		int col_x = (im_y + out_y) * out_w + im_x + out_x;
		int col_y = (gid / NUM_IM_BLKS) * out_h * out_w * wei_h * wei_w;

		for(int y = 0; y < wei_h; y++) {
			for(int x = 0; x < wei_w; x++) {
				int im_off_h = out_y*stride_h + y;
				int im_off_w = out_x*stride_w + x;
				col[col_y + col_x + (y*wei_w+x)*out_h*out_w] = local_im[(im_off_h)*im_cols_wg + im_off_w];	
			}
		}

	}
#endif
}
#endif
