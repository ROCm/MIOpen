/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 *******************************************************************************/

nbytes = 4 / vec_size
// weights
weights_per_filter = wei_w * wei_h
.if weights_layout == 0 // KCHW
	.set filter_c_stride, nbytes * weights_per_filter
	.set filter_k_stride, nbytes * weights_per_filter * input_channels / group_counts
.else // CKHW
	.set filter_c_stride, nbytes * weights_per_filter * output_channels / group_counts
	.set filter_k_stride, nbytes * weights_per_filter
.endif
filters_size = nbytes * weights_per_filter * input_channels * output_channels / group_counts

// input/output
img_hw = img_h * img_w
img_hw_vec = (img_hw + vec_size - 1) / vec_size
odd_hw = (img_h * img_w) % vec_size
out_w = (img_w + 2 * pad_w - wei_w) / stride_w + 1;
out_h = (img_h + 2 * pad_h - wei_h) / stride_h + 1;
input_line_size = nbytes * img_w

input_c_stride = input_line_size * img_h
input_feature_map_size = input_c_stride

input_n_stride = input_feature_map_size * input_channels
input_stack_size = input_n_stride

output_line_size = nbytes * out_w

output_k_stride = output_line_size * out_h
output_feature_map_size = output_k_stride

output_n_stride = output_feature_map_size * output_channels
output_stack_size = output_n_stride
