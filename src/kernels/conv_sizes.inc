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

// weights
weights_per_filter = wei_w * wei_h
.if weights_layout == 0 // KCHW
	.set filter_c_stride, 4 * weights_per_filter
	.set filter_k_stride, 4 * weights_per_filter * input_channels
.else // CKHW
	.set filter_c_stride, 4 * weights_per_filter * output_channels
	.set filter_k_stride, 4 * weights_per_filter
.endif
filters_size = 4 * weights_per_filter * input_channels * output_channels

// input/output
out_w = (img_w + 2 * pad_w - wei_w) / stride_w + 1;
out_h = (img_h + 2 * pad_h - wei_h) / stride_h + 1;
input_line_size = 4 * img_w
input_feature_map_size = input_line_size * img_h
input_stack_size = input_feature_map_size * input_channels
output_line_size = 4 * out_w
output_feature_map_size = output_line_size * out_h
output_stack_size = output_feature_map_size * output_channels
