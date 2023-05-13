/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include "conv_common.hpp"
#include "get_handle.hpp"

template <class T>
struct conv2d_driver : conv_driver<T>
{
    conv2d_driver() : conv_driver<T>()
    {
        this->add(this->input_dims, "input");
        this->add(this->weight_tensor_dims, "weights");
        this->add(this->batch_size,
                  "batch_size",
                  this->generate_data_limited(this->get_batch_sizes(), 1));
        this->add(this->input_channels,
                  "input_channels",
                  this->generate_data_limited(this->get_input_channels(), 1, {32}));
        this->add(this->output_channels,
                  "output_channels",
                  this->generate_data_limited(this->get_output_channels(), 1, {64}));
        this->add(this->spatial_dim_elements,
                  "spatial_dim_elements",
                  this->generate_data_limited(this->get_2d_spatial_dims(), 1, {28, 28}));
        this->add(this->filter_dims,
                  "filter_dims",
                  this->generate_data_limited(this->get_2d_filter_dims(), 2, {3, 3}));
        this->add(this->pads_strides_dilations,
                  "pads_strides_dilations",
                  this->generate_data_limited(this->get_2d_pads_strides_dilations(), 2));
        this->add(this->trans_output_pads,
                  "trans_output_pads",
                  this->generate_data(this->get_2d_trans_output_pads()));
        this->add(this->in_layout, "in_layout", this->generate_data({"NCHW"}));
        this->add(this->fil_layout, "fil_layout", this->generate_data({"NCHW"}));
        this->add(this->out_layout, "out_layout", this->generate_data({"NCHW"}));
        this->add(this->deterministic, "deterministic", this->generate_data({false}));
        this->add(this->tensor_vect, "tensor_vect", this->generate_data({0}));
        this->add(this->vector_length, "vector_length", this->generate_data({1}));
        // Only valid for int8 input and weights
        this->add(this->output_type, "output_type", this->generate_data({"int32"}));
        this->add(this->int8_vectorize, "int8_vectorize", this->generate_data({false}));
    }
};

class MyTestSuite : public testing::TestWithParam<std::string> {};

TEST_P(MyTestSuite, MyTest)
{
  std::cout << "Example Test Param: " << GetParam() << std::endl;
  const auto& handle = get_handle();
  //std::cout << handle.GetDeviceName() << std::endl;
  if(miopen::StartsWith(handle.GetDeviceName(),"gfx906")){
      std::cout << "gfx906" << std::endl;
      GTEST_SKIP();
  }

  const auto param = GetParam();
  std::stringstream ss(param);
  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  std::vector<std::string> tokens(begin, end);
  std::vector<const char*> ptrs;
   
  for (std::string const& str : tokens) {
      ptrs.push_back(str.data());
      //std::cout << str.data() << std::endl;
  }
  //std::cout << "PTR SIZE: " << ptrs.size() << std::endl;

  test_drive<conv2d_driver>(ptrs.size(), ptrs.data());


}

INSTANTIATE_TEST_SUITE_P(MyGroup, MyTestSuite,
                         testing::Values("--disable-validation --verbose --input  1 8  10  10  --weights 8 8 3 3     --pads_strides_dilations 0 0 1 1 1 1 ",
                         "--disable-validation --verbose --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1"));
