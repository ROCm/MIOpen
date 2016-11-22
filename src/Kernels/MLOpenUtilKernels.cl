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

	global float *im_off = im + im_offset;

	if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w) {
		col[col_row*out_h*out_w + out_y*out_w + out_x] = im_off[im_c*h*w + im_off_h*w + im_off_w];
	}
	else {
		col[col_row*out_h*out_w + out_y*out_w + out_x] = 0.;
	}
}
