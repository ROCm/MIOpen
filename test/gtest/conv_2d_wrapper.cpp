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
#include <tuple>

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
/*
set(MIOPEN_EMBED_TEST_ARG ${MIOPEN_TEST_FLOAT_ARG} --disable-validation --verbose)
# WORKAROUND for issue #874
set(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1  MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0)
set(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1  MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0)
set(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1W MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0)
# WORKAROUND for issue #1008
set(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2 MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0)
*/ 

/*
env vars:
add_custom_test(test_conv_embed_db TEST_PERF_DB_RECORD_NOT_FOUND GFX908_DISABLED GFX90A_DISABLED
*/


                                                                   
TEST_P(MyTestSuite, MyTest)                                           
{                                                                     

  //SKIP_UNLESS_COMPOSABLEKERNEL
	//SKIP_UNLESS_ALL???
  //TEST_PERF_DB_RECORD_NOT_FOUND check


  std::unordered_map<std::string, std::string> env_vars = {};
  env_vars["MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2"] = "0"; 
  env_vars["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1" = "0"; //F
  env_vars["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1" = "0"; //W, FW
  env_vars["GFX908_DISABLED" = "1"; 
  env_vars["GFX90A_DISABLED" = "1"; 
  env_vars["GFX94X_ENABLED" = "1"; 
  env_vars["GFX103X_ENABLED" = "1"; 
  env_vars["GFX110X_ENABLED" = "1"; 
  env_vars["HALF_ENABLED" = "1"; 
  env_vars["INT8_ENABLED" = "1"; 
  env_vars["OCL_DISABLED" = "1"; 

  //IMPLICITGEMM_MLIR_ENV_BASE
  env_vars["MIOPEN_FIND_MODE" = "normal"];
	env_vars["MIOPEN_DEBUG_FIND_ONLY_SOLVER"= "ConvMlirIgemmFwd"];
	env_vars["MIOPEN_DEBUG_FIND_ONLY_SOLVER"= "ConvMlirIgemmBwd"];
	env_vars["MIOPEN_DEBUG_FIND_ONLY_SOLVER"= "ConvMlirIgemmWrW"];
  env_vars["FIND_MODE" = "normal"]; //IMPLICITGEMM_MLIR_ENV_BASE
	env_vars["MIOPEN_DEBUG_FIND_ONLY_SOLVER" = "ConvMlirIgemmFwdXdlops"];
	env_vars["MIOPEN_DEBUG_FIND_ONLY_SOLVER" = "ConvMlirIgemmBwdXdlops"];
	env_vars["MIOPEN_DEBUG_FIND_ONLY_SOLVER" = "ConvMlirIgemmWrWXdlops"];

  /*
  env_vars["MIOPEN_DEBUG_" = ""; 
  */
  std::unordered_map<std::string, std::string> xtra_args = {};
  xtra_args["MIOPEN_TEST_FLOAT_ARG"] = "--disable-validation --verbose"

  //MIOPEN_TEST_FLOAT_ARGS + 
	xtra_args["TEST_CONV_VERBOSE_F" = "--verbose --disable-backward-data --disable-backward-weights"];
	xtra_args["TEST_CONV_VERBOSE_B" = "--verbose --disable-forward --disable-backward-weights"];
	xtra_args["TEST_CONV_VERBOSE_W" = "--verbose --disable-forward --disable-backward-data"];
  xtra_args["MIOPEN_TEST_FLOAT_ARG" = "--disable-forward --disable-backward-data"];

  std::cout << "Example Test Param: " << GetParam() << std::endl;     
  const auto& handle = get_handle();                                  
  //std::cout << handle.GetDeviceName() << std::endl;                 
  if(miopen::StartsWith(handle.GetDeviceName(),"gfx906")){
      std::cout << "gfx906" << std::endl;
      GTEST_SKIP();
  }

  const auto param = GetParam();
  for (auto& elem : param)
    std::cout << elem << ", ";
  const auto t_size = std::tuple_size<decltype(param)>::value;
  std::cout "tuple size: " << t_size << std::endl;
  const auto test_cmd = std::get<t_size-1>(param);
  std::cout "param: " << test_cmd << std::endl;

  /*
  for elem in param[0,n-2]
		if elem in env_vars.keys set env var
    if elem in xtra_args prepend xtra cmd to driver
    if elem=="TEST_PERF_DB_RECORD_NOT_FOUND" check output
  */


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
                         testing::Values(
														std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1", "--disable-validation --verbose --input  1 8  10  10  --weights 8 8 3 3     --pads_strides_dilations 0 0 1 1 1 1 "),
                         		std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "--disable-validation --verbose --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1")
												));


//set following env vars prior to running test
//add_custom_test(test_conv_embed_db TEST_PERF_DB_RECORD_NOT_FOUND GFX908_DISABLED GFX90A_DISABLED
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", ${MIOPEN_EMBED_TEST_ARG} --input 128 128 28 28 --weights 128 128 3 3 --pads_strides_dilations 1 1 1 1 1 1
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 128 28 28 --weights 512 128 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 2048 7 7 --weights 512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 256 14 14 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 256 14 14 --weights 256 256 3 3 --pads_strides_dilations 1 1 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 128 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 256 56 56 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", ${MIOPEN_EMBED_TEST_ARG} --input 128 3 230 230   --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 1024 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 128 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 512 7 7   --weights 2048 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 512 7 7   --weights 512 512 3 3  --pads_strides_dilations 1 1 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 256 64 1 1  --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"  ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 64 64 1 1  --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1"   ${MIOPEN_EMBED_TEST_ARG} --input 128 64 56 56 --weights 64 64 3 3   --pads_strides_dilations 1 1 1 1 1 1"),



/*
set(IMPLICITGEMM_MLIR_ENV_BASE MIOPEN_FIND_MODE=normal)
set(IMPLICITGEMM_MLIR_ENV_F ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwd)
set(IMPLICITGEMM_MLIR_ENV_B ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwd)
set(IMPLICITGEMM_MLIR_ENV_W ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrW)

set(TEST_CONV_VERBOSE_F ${MIOPEN_TEST_FLOAT_ARG} --verbose --disable-backward-data --disable-backward-weights)
set(TEST_CONV_VERBOSE_B ${MIOPEN_TEST_FLOAT_ARG} --verbose --disable-forward --disable-backward-weights)
set(TEST_CONV_VERBOSE_W ${MIOPEN_TEST_FLOAT_ARG} --verbose --disable-forward --disable-backward-data)

add_custom_test(test_pooling2d_asymmetric SKIP_UNLESS_ALL HALF_ENABLED GFX94X_ENABLED GFX103X_ENABLED GFX110X_ENABLED
    COMMAND $<TARGET_FILE:test_pooling2d> ${MIOPEN_TEST_FLOAT_ARG} --all --dataset 1 --limit 0 ${MIOPEN_TEST_FLAGS_ARGS}
)

add_custom_test(test_conv_igemm_mlir_fwd SKIP_UNLESS_ALL HALF_ENABLED INT8_ENABLED SKIP_UNLESS_MLIR GFX900_DISABLED GFX908_DISABLED GFX90A_DISABLED GFX103X_ENABLED


std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_F} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4")



//set following env vars prior to running test
//add_custom_test(test_conv_igemm_mlir_bwd_wrw SKIP_UNLESS_ALL HALF_ENABLED SKIP_UNLESS_MLIR GFX900_DISABLED GFX908_DISABLED GFX90A_DISABLED GFX103X_ENABLED
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_B} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),

std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
std::make_tuple("${IMPLICITGEMM_MLIR_ENV_W} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 256 1024 14 14 --weights 1024 32   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 32")





set(IMPLICITGEMM_MLIR_ENV_F_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwdXdlops)
set(IMPLICITGEMM_MLIR_ENV_B_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwdXdlops)
set(IMPLICITGEMM_MLIR_ENV_W_XDLOPS ${IMPLICITGEMM_MLIR_ENV_BASE} MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrWXdlops)

add_custom_test(test_conv_igemm_mlir_xdlops_fwd SKIP_UNLESS_ALL HALF_ENABLED INT8_ENABLED SKIP_UNLESS_MLIR GFX900_DISABLED GFX906_DISABLED
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_F_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_F} --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4
)

add_custom_test(test_conv_hip_igemm_xdlops SKIP_UNLESS_ALL OCL_DISABLED HALF_DISABLED FLOAT_DISABLED INT8_ENABLED GFX900_DISABLED GFX906_DISABLED GFX94X_ENABLED SKIP_UNLESS_COMPOSABLEKERNEL
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 256 128  28 28 --weights 128  128  3 3 ${MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT8} --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 128 512  7  7  --weights 512  512  3 3 ${MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT8} --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 128 64   56 56 --weights 64   64   1 1 ${MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT8} --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 256 256  56 56 --weights 256  64   1 1 ${MIOPEN_TEST_CONV_INT8_OUTPUT_TYPE_INT8} --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 256 128  28 28 --weights 128  128  3 3 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 128 512  7  7  --weights 512  512  3 3 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 128 64   56 56 --weights 64   64   1 1 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 256 256  56 56 --weights 256  64   1 1 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 256 128  28 28 --weights 128  128  3 3 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 128 512  7  7  --weights 512  512  3 3 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 128 64   56 56 --weights 64   64   1 1 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-backward-data --disable-backward-weights --verbose --input 256 256  56 56 --weights 256  64   1 1 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 256 128  28 28 --weights 128  128  3 3 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 128 512  7  7  --weights 512  512  3 3 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 128 64   56 56 --weights 64   64   1 1 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 256 256  56 56 --weights 256  64   1 1 --output_type fp32 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 256 128  28 28 --weights 128  128  3 3 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 128 512  7  7  --weights 512  512  3 3 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 1 1 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 128 64   56 56 --weights 64   64   1 1 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
    COMMAND $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --disable-forward --disable-backward-weights --verbose --input 256 256  56 56 --weights 256  64   1 1 --output_type fp16 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --pads_strides_dilations 0 0 1 1 1 1
)

add_custom_test(test_conv_igemm_mlir_xdlops_bwd_wrw SKIP_UNLESS_ALL HALF_ENABLED SKIP_UNLESS_MLIR GFX900_DISABLED GFX906_DISABLED
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_B_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_B} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC

    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1
    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 256 1024 14 14 --weights 1024 32   1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 32

    COMMAND ${IMPLICITGEMM_MLIR_ENV_W_XDLOPS} $<TARGET_FILE:test_conv2d> ${TEST_CONV_VERBOSE_W} --input 64 1024 14 14 --weights 1024 1024  1 1 --pads_strides_dilations 0 0 1 1 1 1
)

set(IMPLICITGEMM_TESTING_ENV
 MIOPEN_DEBUG_CONV_WINOGRAD=0
 MIOPEN_DEBUG_CONV_FFT=0
 MIOPEN_DEBUG_CONV_DIRECT=0
 MIOPEN_DEBUG_CONV_GEMM=0
 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
)

if(WORKAROUND_ISSUE_936 AND MIOPEN_TEST_HALF)
    SET(SAVE_IMPLICITGEMM_TESTING_ENV ${IMPLICITGEMM_TESTING_ENV})
    LIST(APPEND IMPLICITGEMM_TESTING_ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 MIOPEN_FIND_MODE=normal)
    SET(SAVE_MIOPEN_TEST_FLOAT_ARG ${MIOPEN_TEST_FLOAT_ARG})
    LIST(APPEND MIOPEN_TEST_FLOAT_ARG --disable-forward --disable-backward-data)
    #Afther fix need to remove '| grep -v "No suitable algorithm was found to execute the required convolution"'
endif()

add_custom_test(test_conv_for_implicit_gemm SKIP_UNLESS_ALL BF16_ENABLED HALF_ENABLED GFX94X_ENABLED GFX103X_ENABLED
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  16  28  28  --weights 192 16  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  16  14  14  --weights 160 16  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  16   7   7  --weights 128 16  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  16  55  55  --weights 96  16  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  16  28  28  --weights 64  16  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  16  14  14  --weights 32  16  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  32  28  28  --weights 192 32  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  32  14  14  --weights 160 32  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  32  7   7   --weights 128 32  3 3 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  32  55  55  --weights 96  32  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  32  28  28  --weights 64  32  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  32  14  14  --weights 32  32  1 7 --pads_strides_dilations 0 0 2 2 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  64  56  56  --weights 256 64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  64  56  56  --weights 64  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  64  73  73  --weights 80  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  64  56  56  --weights 64  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  128 55  55  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  128 28  28  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  128 14  14  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64  128  7   7  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16   64 56  56  --weights 256  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16   64 56  56  --weights 64   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16   64 73  73  --weights 80   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16   64 56  56  --weights 64   64 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16  128 55  55  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16  128 28  28  --weights 16  128 1 1 --pads_strides_dilations 0 0 1 1 1 1 | grep -v "No suitable algorithm was found to execute the required convolution"
# COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16  128     14  14  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16  128      7   7  --weights   16  128     1   1   --pads_strides_dilations    0   0   1   1   1   1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	128	55	55	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	128	28	28	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	128	14	14	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	128	 7	 7	--weights	16  128		1	1	--pads_strides_dilations	0	0	2	2	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	128	    28	28	--weights	512	128	    1	1	--pads_strides_dilations	0	0	1	1	1	1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	160	    73	73	--weights	64	160	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	192	    35	35	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	192	    35	35	--weights	48	192	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	192	    35	35	--weights	64	192	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	192	28	28	--weights	16	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	192	28	28	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	192	28	28	--weights	64	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	192	28	28	--weights	96	192	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	    35	35	--weights	48	256	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	    35	35	--weights	64	256	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	    56	56	--weights	128	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	    56	56	--weights	512	256	    1	1	--pads_strides_dilations	0	0	2	2	1	1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	    56	56	--weights	64	256	    1	1	--pads_strides_dilations	0	0	1	1	1	1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	28	28	--weights	128	256	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	28	28	--weights	32	256	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	256	28	28	--weights	64	256	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	288	    35	35	--weights	48	288	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	288	    35	35	--weights	64	288	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	384	    35	35	--weights	192	384	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	384	    35	35	--weights	64	384	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	384	    35	35	--weights	96	384	1	1	--pads_strides_dilations	0	0	1	1	1	1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	480	14	14	--weights	16	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	480	14	14	--weights	192	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	480	14	14	--weights	64	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	480	14	14	--weights	96	480	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	    28	28	--weights	128	512	    1	1	--pads_strides_dilations	0	0	1	1	1	1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	    28	28	--weights	256	512	    1	1	--pads_strides_dilations	0	0	2	2	1	1 | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	14	14	--weights	112	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	14	14	--weights	128	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	14	14	--weights	144	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	14	14	--weights	160	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	14	14	--weights	24	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	14	14	--weights	32	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose	--input	64	512	14	14	--weights	64	512	1	1	--pads_strides_dilations	0	0	1	1	1	1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 128  832    7  7  --weights   32  832  1   1   --pads_strides_dilations    0   0   1   1   1   1      | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 128  832    7  7  --weights   192  832  1   1   --pads_strides_dilations    0   0   1   1   1   1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 128  832    7  7  --weights   128  832  1   1   --pads_strides_dilations    0   0   1   1   1   1     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 128  832    7  7  --weights   32  832  1   1   --pads_strides_dilations    0   0   1   1   2   2      | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 128  832    7  7  --weights   192  832  1   1   --pads_strides_dilations    0   0   1   1   2   2     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 128  832    7  7  --weights   128  832  1   1   --pads_strides_dilations    0   0   1   1   2   2     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND ${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 16  2048    7  7  --weights   192  2048 1   1   --pads_strides_dilations    0   0   1   1   2   2     | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64	 32	28 28 --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 8    16 14 14 --weights   32   16   1   1   --pads_strides_dilations	1   1	1   1	1   1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64	 32	14 14 --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64	 32	7 7   --weights   192  32   3	3   --pads_strides_dilations	1   1	2   2	1   1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64	 32	28 28 --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64	 32	14 14 --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1         | grep -v "No suitable algorithm was found to execute the required convolution"
COMMAND	${IMPLICITGEMM_TESTING_ENV} $<TARGET_FILE:test_conv2d> ${MIOPEN_TEST_FLOAT_ARG} --verbose   --input 64	 32	7 7   --weights   192  32   3	3   --pads_strides_dilations	2   2	2   2	1   1         | grep -v "No suitable algorithm was found to execute the required convolution"
)

if(WORKAROUND_ISSUE_936 AND MIOPEN_TEST_HALF)
    SET(IMPLICITGEMM_TESTING_ENV ${SAVE_IMPLICITGEMM_TESTING_ENV})
    SET(MIOPEN_TEST_FLOAT_ARG ${SAVE_MIOPEN_TEST_FLOAT_ARG})
endif()

add_custom_test(test_conv_group SKIP_UNLESS_ALL GFX94X_ENABLED GFX103X_ENABLED GFX110X_ENABLED
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
#
# Workaround for "Memory access fault by GPU node" during "HIP Release All" - WrW disabled.
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32 --disable-backward-weights
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	4	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	32	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	4	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	16	24	240	--weights	32	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	16
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	32	12	120	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	64	6	60	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	54	54	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	128	27	27	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	64	112	112	--weights	128	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	2
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	9	224	224	--weights	63	3	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3
#
# Workaround for "Memory access fault by GPU node" during "FP32 gfx908 Hip Release All subset" - WrW disabled.
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	64	112	112	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4 --disable-backward-weights
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	3	224	224	--weights	63	1	7	7	--pads_strides_dilations	3	3	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	192	28	28	--weights	32	12	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	832	7	7	--weights	128	52	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	192	28	28	--weights	32	24	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	16	832	7	7	--weights	128	104	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	11	23	161	700	--weights	46	1	7	7	--pads_strides_dilations	1	1	2	2	1	1	--group-count	23
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	7
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3
COMMAND	$<TARGET_FILE:test_conv2d>	--verbose	--input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	3
)
*/
