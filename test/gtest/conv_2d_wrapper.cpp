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

class Conv2dSuite : public testing::TestWithParam<std::tuple<std::vector<std::string>, std::string>>
{
};
TEST_P(Conv2dSuite, MyTest)
{
#if MIOPEN_EMBED_DB

    const auto GFX908_DISABLED = std::getenv("GFX908_DISABLED");
    const auto GFX90A_DISABLED = std::getenv("GFX90A_DISABLED");
    if(GFX908_DISABLED && GFX90A_DISABLED)
    {
        const auto& handle = get_handle();
        if(miopen::StartsWith(handle.GetDeviceName(), "gfx906"))
        {
            GTEST_SKIP();
        }

        auto param    = GetParam();
        auto env_vars = std::get<0>(param);
        for(auto& elem : env_vars)
            putenv(elem.data());

        auto cmd = std::get<1>(param);
        std::cout << cmd << std::endl;

        std::stringstream ss(cmd);
        std::istream_iterator<std::string> begin(ss);
        std::istream_iterator<std::string> end;
        std::vector<std::string> tokens(begin, end);
        std::vector<const char*> ptrs;

        for(std::string const& str : tokens)
            ptrs.push_back(str.data());

        testing::internal::CaptureStderr();
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

INSTANTIATE_TEST_SUITE_P(
    Conv2dGroup,
    Conv2dSuite,
    testing::Values(
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 1024 14 14 --weights 2048 i"
                "1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 1024 14 14 --weights 256 "
                "1024 1 1 --pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 1024 14 14 --weights 512 1024 1 1 "
                "--pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 128 28 28 --weights 128 128 3 3 "
                "--pads_strides_dilations 1 1 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 1024 14 14 --weights 512 1024 1 1 "
                "--pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 128 28 28 --weights 512 128 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 2048 7 7 --weights 512 2048 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 256 14 14 --weights 1024 256 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 256 14 14 --weights 256 256 3 3 "
                "--pads_strides_dilations 1 1 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 256 56 56 --weights 128 256 1 1 "
                "--pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 256 56 56 --weights 512 256 1 1 "
                "--pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 256 56 56 --weights 64 256 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 3 230 230   --weights 64 3 7 7 "
                "--pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 512 28 28 --weights 1024 512 1 1 "
                "--pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 512 28 28 --weights 128 512 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 512 28 28 --weights 256 512 1 1 "
                "--pads_strides_dilations 0 0 2 2 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 512 7 7   --weights 2048 512 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 512 7 7   --weights 512 512 3 3 "
                "--pads_strides_dilations 1 1 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 64 56 56 --weights 256 64 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 64 56 56 --weights 64 64 1 1 "
                "--pads_strides_dilations 0 0 1 1 1 1"),
        std::make_tuple<std::vector<std::string>, std::string>(
            {"MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0",
             "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1=0"},
            std::string(std::getenv("MIOPEN_TEST_FLOAT_ARG")) +
                " --disable-validation --verbose --input 128 64 56 56 --weights 64 64 3 3 "
                "--pads_strides_dilations 1 1 1 1 1 1")));
