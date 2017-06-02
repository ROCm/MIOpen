#ifndef GUARD_MIOPEN_UTIL_DRIVER_HPP
#define GUARD_MIOPEN_UTIL_DRIVER_HPP

template<typename T>
void Im2ColCPU(	std::vector<T> &in, const size_t in_offset,
		const int in_c, const int in_h, const int in_w,
		const int wei_h, const int wei_w,
		const int out_h, const int out_w,
		const int pad_h, const int pad_w,
		const int v, const int u,
		std::vector<T> &col) 
{
	int col_m = in_c * wei_h * wei_w;
	
	auto in_iter = in.begin() + in_offset;

	for(int n = 0; n < col_m; n++) {
		int x = n % wei_w;
		int y = (n / wei_w) % wei_h;
		int ch = n / (wei_w * wei_h);

		for(int h = 0; h < out_h; h++) {
			for(int w = 0; w < out_w; w++) {
				int in_off_h = h * v - pad_h + y;
				int in_off_w = w * u - pad_w + x;

				if(in_off_h >= 0 && in_off_h < in_h && in_off_w >= 0 && in_off_w < in_w)
					col[n*out_h*out_w + h*out_w + w] = in_iter[ch*in_h*in_w + in_off_h*in_w + in_off_w];
				else
					col[n*out_h*out_w + h*out_w + w] = 0;
			}
		}
	}
}


template<typename T>
void Col2ImCPU(std::vector<T> data_col, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, std::vector<T> data_im) {
	memset(data_im, 0, sizeof(T) * height * width * channels);
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	height_col = (height_col < 0) ? 1 : height_col;
	width_col = (width_col < 0) ? 1 : width_col;
	int channels_col = channels * ksize * ksize;
	for (int c = 0; c < channels_col; ++c) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;
		for (int h = 0; h < height_col; ++h) {
			for (int w = 0; w < width_col; ++w) {
				int h_pad = h * stride - pad + h_offset;
				int w_pad = w * stride - pad + w_offset;
				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
				{
					data_im[(c_im * height + h_pad) * width + w_pad] +=
						data_col[(c * height_col + h) * width_col + w];
				}
			}
		}
	}
}

#endif // GUARD_MIOPEN_UTIL_DRIVER_HPP
