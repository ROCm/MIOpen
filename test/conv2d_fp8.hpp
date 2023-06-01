/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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
#pragma once

#include "conv_common.hpp"
#include <miopen/conv/problem_description.hpp>
#include <miopen/mlo_internal.hpp>

template <typename T>
struct convfp8_driver : conv_driver<T>
{
    using conv_driver<T>::filter;
    using conv_driver<T>::get_spatial_dim;
    using conv_driver<T>::filter_dims;
    using conv_driver<T>::cmode_lookup;
    using conv_driver<T>::conv_mode;
    using conv_driver<T>::pmode_lookup;
    using conv_driver<T>::pad_mode;
    using conv_driver<T>::groupCount;
    using conv_driver<T>::input_dims;
    using conv_driver<T>::input;
    using conv_driver<T>::batch_size;
    using conv_driver<T>::input_channels;
    using conv_driver<T>::spatial_dim_elements;
    using conv_driver<T>::weight_tensor_dims;
    using conv_driver<T>::weights;
    using conv_driver<T>::output_channels;
    using conv_driver<T>::in_layout;
    using conv_driver<T>::fil_layout;
    using conv_driver<T>::out_layout;
    using conv_driver<T>::pads_strides_dilations;
    using conv_driver<T>::trans_output_pads;
    using conv_driver<T>::show_command;
    using conv_driver<T>::gen_float;
    using conv_driver<T>::enable_fdb;
    using conv_driver<T>::do_forward;
    using conv_driver<T>::search;
    using conv_driver<T>::do_backward_data;
    using conv_driver<T>::do_backward_weights;
    std::string in_cast_type;
    std::string out_cast_type;
    std::string wei_cast_type;
    std::string fp8_rounding_mode;
    std::string fp8_rounding_seed;

    convfp8_driver() : conv_driver<T>()
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
        this->add(this->in_cast_type, "in_cast_type");
        this->add(this->wei_cast_type, "wei_cast_type");
        this->add(this->out_cast_type, "out_cast_type");
        this->add(this->fp8_rounding_mode, "fp8_rounding_mode");
        this->add(this->fp8_rounding_seed, "fp8_rounding_seed");
    }

    miopenDataType_t GetCastType(const std::string& str_)
    {
        const auto str = miopen::ToUpper(str_);
        if(str == "FP8")
            return miopenFloat8;
        if(str == "BFP8")
            return miopenBFloat8;
        std::cerr << "FAILED: Invalid cast type supplied: " << str << std::endl;
        exit(-1);
    }

    void setup_descriptors()
    {
        if(!this->input_dims.empty())
            filter.spatialDim = get_spatial_dim();
        else
            filter.spatialDim = filter_dims.size();

        filter.mode             = cmode_lookup[miopen::ToUpper(conv_mode)];
        filter.paddingMode      = pmode_lookup[miopen::ToUpper(pad_mode)];
        std::size_t spatial_dim = filter.GetSpatialDimension();
        filter.group_count      = std::max(static_cast<int>(groupCount), 1);

        if(!this->fp8_rounding_mode.empty())
        {
            const auto mode = miopen::ToUpper(fp8_rounding_mode);
            if(mode == "STANDARD")
                filter.attribute.fp8rounding_mode.rounding_mode = miopenF8RoundingModeStandard;
            else if(mode == "STOCHASTIC")
                filter.attribute.fp8rounding_mode.rounding_mode = miopenF8RoundingModeStochastic;
            else
            {
                std::cout << "FAILED: Invalid rounding mode supplied" << std::endl;
                exit(-1);
            }
            if(!this->fp8_rounding_seed.empty())
            {
                if(filter.attribute.fp8rounding_mode.rounding_mode ==
                   miopenF8RoundingModeStochastic)
                {
                    const auto seed = std::stol(fp8_rounding_seed);
                    filter.attribute.fp8rounding_mode.SetSeed(seed);
                }
            }
        }

        if(!input_dims.empty())
        {
            input          = tensor<T>{input_dims}.generate(tensor_elem_gen_integer{17});
            batch_size     = input_dims.at(0);
            input_channels = input_dims.at(1);
            std::copy(input_dims.begin() + 2, input_dims.end(), spatial_dim_elements.begin());
        }
        else if(spatial_dim == 2)
        {
            input = tensor<T>{batch_size,
                              input_channels,
                              spatial_dim_elements.at(0),
                              spatial_dim_elements.at(1)}
                        .generate(tensor_elem_gen_integer{17});
        }
        else if(spatial_dim == 3)
        {
            input = tensor<T>{batch_size,
                              input_channels,
                              spatial_dim_elements.at(0),
                              spatial_dim_elements.at(1),
                              spatial_dim_elements.at(2)}
                        .generate(tensor_elem_gen_integer{17});
        }
        if(!this->in_cast_type.empty())
            input.desc.SetCastType(GetCastType(in_cast_type));

        if(!weight_tensor_dims.empty())
        {
            weights         = tensor<T>{weight_tensor_dims}.generate(tensor_elem_gen_integer{17});
            output_channels = weight_tensor_dims.at(0);
        }
        else if(spatial_dim == 2)
        {
            weights = tensor<T>{output_channels,
                                input_channels / filter.group_count,
                                filter_dims.at(0),
                                filter_dims.at(1)}
                          .generate(tensor_elem_gen_integer{17});
        }
        else if(spatial_dim == 3)
        {
            weights = tensor<T>{output_channels,
                                input_channels / filter.group_count,
                                filter_dims.at(0),
                                filter_dims.at(1),
                                filter_dims.at(2)}
                          .generate(tensor_elem_gen_integer{17});
        }
        if(!this->wei_cast_type.empty())
            weights.desc.SetCastType(GetCastType(wei_cast_type));

        if(input.desc.GetSize() != in_layout.size() ||
           weights.desc.GetSize() != fil_layout.size() || input.desc.GetSize() != out_layout.size())
        {
            std::cerr << "FAILED: layout not match dimension size!" << std::endl;
            return;
        }

        // reconstruct tensor descriptor(desc) when layout is not the default NCHW layout.
        // by default, this member is constructed when conv2d/3d is constructed (see
        // test_driver::add())
        // but this requires the dimensions come from commandline, which is hard for non-NCHW layout
        if(in_layout != "NCHW" && in_layout != "NCDHW")
        {
            const std::vector<std::size_t> dim_lens = input.desc.GetLengths();
            std::vector<std::size_t> dim_strides;
            miopen::tensor_layout_to_strides(
                dim_lens,
                miopen::tensor_layout_get_default(input.desc.GetSize()),
                in_layout,
                dim_strides);
            input.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
        }
        if(fil_layout != "NCHW" && fil_layout != "NCDHW")
        {
            const std::vector<std::size_t> dim_lens = weights.desc.GetLengths();
            std::vector<std::size_t> dim_strides;
            miopen::tensor_layout_to_strides(
                dim_lens,
                miopen::tensor_layout_get_default(weights.desc.GetSize()),
                fil_layout,
                dim_strides);
            weights.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
        }

        if(input.desc.GetSize() != 2 + spatial_dim || weights.desc.GetSize() != 2 + spatial_dim ||
           pads_strides_dilations.size() != 3 * spatial_dim ||
           trans_output_pads.size() != spatial_dim)
        {
            std::cerr << "FAILED: dimension is wrong!" << std::endl;
            return;
        }

        filter.pads.resize(spatial_dim);
        filter.strides.resize(spatial_dim);
        filter.dilations.resize(spatial_dim);
        filter.trans_output_pads.resize(spatial_dim);

        std::copy_n(pads_strides_dilations.begin(), spatial_dim, filter.pads.begin());
        std::copy_n(
            pads_strides_dilations.begin() + spatial_dim, spatial_dim, filter.strides.begin());
        std::copy_n(pads_strides_dilations.begin() + 2 * spatial_dim,
                    spatial_dim,
                    filter.dilations.begin());
        std::copy_n(trans_output_pads.begin(), spatial_dim, filter.trans_output_pads.begin());
    }
    bool check_applicability()
    {
        std::size_t spatial_dim = filter.GetSpatialDimension();
        if(input.desc.GetSize() != in_layout.size() ||
           weights.desc.GetSize() != fil_layout.size() || input.desc.GetSize() != out_layout.size())
        {
            std::cerr << "FAILED: layout not match dimension size!" << std::endl;
            return false;
        }
        if(input.desc.GetSize() != 2 + spatial_dim || weights.desc.GetSize() != 2 + spatial_dim ||
           pads_strides_dilations.size() != 3 * spatial_dim ||
           trans_output_pads.size() != spatial_dim)
        {
            std::cerr << "FAILED: dimension is wrong!" << std::endl;
            return false;
        }
        bool is_int8 = (input.desc.GetType() == miopenInt8 || input.desc.GetType() == miopenInt8x4);

        // lack of transposeConv or groupConv for int8 type
        if(is_int8 && (filter.mode == miopenTranspose || filter.group_count > 1))
        {
            show_command();
            std::cout << "MIOpen doesn't support int8 type transpose or group convolution."
                      << std::endl;
            return false;
        }
        bool is_bfloat16 =
            (input.desc.GetType() == miopenBFloat16 && weights.desc.GetType() == miopenBFloat16);

        // bfloat16 is not supported for conv3d
        if(is_bfloat16 && !(filter.spatialDim == 2))
            return false;

        return true;
    }
#if 0
    std::size_t workspace_size()
    {
        auto& handle = get_handle();
        const auto ctx = ExecutionContext{&handle}.DetectRocm();
        const auto problem = ConvProblemDescription{
            input.desc, weights.desc, rout.desc
        }
        auto output  = get_output_tensor(filter, input, weights, out_layout);
        size_t total_mem;
        bool is_int8 = (input.desc.GetType() == miopenInt8 || input.desc.GetType() == miopenInt8x4);
        if(is_int8)
        {
            auto output_int8 = get_output_tensor<T, float>(filter, input, weights, out_layout);
            size_t workspace_size =
                filter.ForwardGetWorkSpaceSize(handle, weights.desc, input.desc, output_int8.desc);

            // 4x because assume type is miopenInt8x4
            total_mem = input.desc.GetNumBytes() + 4 * input.desc.GetNumBytes() +
                        weights.desc.GetNumBytes() + 4 * weights.desc.GetNumBytes() +
                        output_int8.desc.GetNumBytes() + 4 * sizeof(char) * workspace_size;
        }
        else
        {
            size_t workspace_size_1 =
                filter.mode == miopenTranspose
                    ? filter.ForwardGetWorkSpaceSize(handle, weights.desc, output.desc, input.desc)
                    : filter.BackwardDataGetWorkSpaceSize(
                          handle, weights.desc, output.desc, input.desc);

            size_t workspace_size_2 =
                filter.mode == miopenTranspose
                    ? filter.BackwardDataGetWorkSpaceSize(
                          handle, weights.desc, input.desc, output.desc)
                    : filter.ForwardGetWorkSpaceSize(handle, weights.desc, input.desc, output.desc);

            size_t workspace_size_3 = filter.BackwardWeightsGetWorkSpaceSize(
                handle,
                filter.mode == miopenTranspose ? input.desc : output.desc,
                filter.mode == miopenTranspose ? output.desc : input.desc,
                weights.desc);

            std::vector<size_t> workspace_sizes = {
                workspace_size_1, workspace_size_2, workspace_size_3};
            size_t workspace_size =
                *std::max_element(workspace_sizes.begin(), workspace_sizes.end());

            total_mem = input.desc.GetNumBytes() + weights.desc.GetNumBytes() +
                        output.desc.GetNumBytes() +
                        sizeof(char) * workspace_size; // estimate based on backward pass
        }
        return total_mem;
    }
#endif
    template <typename U, typename V>
    struct Fp8Cast
    {
        uint64_t seed = 1234;
        bool is_stoch = false;
        V operator()(U x)
        {
            if(is_stoch)
            {
                auto tmp = float8(
                    static_cast<float>(x), miopen_f8::hip_f8_rounding_mode::stochastic, seed);
                return static_cast<V>(tmp);
            }
            else
            {
                auto tmp = float8(static_cast<float>(x));
                return static_cast<V>(tmp);
            }
        }
    };
    void run()
    {
        setup_descriptors();
        if(!check_applicability())
            return;
        std::vector<std::size_t> in_spatial_len(input.desc.GetLengths().begin() + 2,
                                                input.desc.GetLengths().end());
        std::vector<std::size_t> wei_spatial_len(weights.desc.GetLengths().begin() + 2,
                                                 weights.desc.GetLengths().end());

        std::size_t spatial_dim = filter.GetSpatialDimension();
        std::size_t in_c_len    = input.desc.GetLengths()[1];
        std::size_t wei_k_len   = weights.desc.GetLengths()[0];
        std::size_t wei_c_len   = weights.desc.GetLengths()[1];
        bool is_int8 = (input.desc.GetType() == miopenInt8 || input.desc.GetType() == miopenInt8x4);

        if(((filter.mode == miopenTranspose) &&
            ((filter.group_count == 1 && in_c_len == wei_k_len) ||
             (filter.group_count > 1 && wei_k_len % filter.group_count == 0))) ||
           ((filter.mode == miopenConvolution) &&
            ((filter.group_count == 1 && in_c_len == wei_c_len) ||
             (filter.group_count > 1 && in_c_len % wei_c_len == 0))))
        {
            if(filter.mode == miopenConvolution &&
               (miopen::all_of(filter.GetConvDilations(), [](auto v) { return v == 1; }) ||
                miopen::all_of(wei_spatial_len, [](auto v) { return v == 1; })))
            {
                if(filter.paddingMode == miopenPaddingSame)
                {
                    if(miopen::any_of(filter.GetConvStrides(), [](auto v) { return v == 0; }))
                    {
                        return;
                    }

                    std::vector<std::size_t> pads_(spatial_dim);
                    std::vector<std::ptrdiff_t> out_spatial_len(spatial_dim);

                    for(std::size_t i = 0; i < spatial_dim; ++i)
                    {
                        pads_[i] =
                            (in_spatial_len[i] % filter.GetConvStrides()[i] == 0)
                                ? (std::max(
                                      static_cast<std::ptrdiff_t>(wei_spatial_len[i]) -
                                          static_cast<std::ptrdiff_t>(filter.GetConvStrides()[i]),
                                      static_cast<std::ptrdiff_t>(0)))
                                : (std::max(static_cast<std::ptrdiff_t>(wei_spatial_len[i]) -
                                                static_cast<std::ptrdiff_t>(
                                                    in_spatial_len[i] % filter.GetConvStrides()[i]),
                                            static_cast<std::ptrdiff_t>(0)));

                        filter.pads[i] = pads_[i] / 2;

                        out_spatial_len[i] = miopen::integer_division_ceil(
                            in_spatial_len[i], filter.GetConvStrides()[i]);
                    }

                    if(miopen::any_of(out_spatial_len, [](auto v) { return v <= 0; }))
                    {
                        return;
                    }
                }
                else if(filter.paddingMode == miopenPaddingValid)
                {
                    if(miopen::any_of(filter.GetConvStrides(), [](auto v) { return v == 0; }))
                        return;

                    std::vector<ptrdiff_t> out_spatial_len(spatial_dim);

                    for(std::size_t i = 0; i < spatial_dim; ++i)
                    {
                        filter.pads[i] = 0;

                        out_spatial_len[i] = miopen::integer_division_ceil(
                            static_cast<std::ptrdiff_t>(in_spatial_len[i]) -
                                static_cast<std::ptrdiff_t>(wei_spatial_len[i]) + 1,
                            filter.GetConvStrides()[i]);
                    }

                    if(miopen::any_of(out_spatial_len, [](auto v) { return v <= 0; }))
                    {
                        return;
                    }
                }
            }
            if(filter.mode == miopenTranspose)
            {
                for(std::size_t i = 0; i < spatial_dim; ++i)
                {
                    filter.pads[i] = filter.GetConvStrides()[i] - 1;
                }
            }

            if(((filter.mode == miopenTranspose) &&
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(0))) ||
                 (filter.group_count > 1 &&
                  (weights.desc.GetLengths().at(0) % filter.group_count == 0)))) ||
               ((filter.mode == miopenConvolution) &&
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1))) ||
                 (filter.group_count > 1 &&
                  (input.desc.GetLengths().at(1) % weights.desc.GetLengths().at(1) == 0)))))
            {
                auto output = get_output_tensor(filter, input, weights, out_layout);
                if(!this->out_cast_type.empty())
                    output.desc.SetCastType(GetCastType(this->out_cast_type));

                auto gen_fp8_value = [=](auto...) {
                    const auto tmp = float8(scalar_gen_random_float{-0.5, 0.5}());
                    return static_cast<float>(tmp);
                };
                const auto problem =
                    miopen::conv::ProblemDescription(input.desc,
                                                     weights.desc,
                                                     output.desc,
                                                     filter,
                                                     miopen::conv::Direction::Forward);
                const auto ctx = [&] {
                    auto& handle = get_handle();
                    auto tmp     = miopen::ConvolutionContext{&handle};
                    tmp.DetectRocm();
                    problem.SetupFloats(tmp);
                    return tmp;
                }();

                bool skip_forward = is_int8 && !IsGemmAplicable(ctx, problem);
                if(skip_forward)
                {
                    show_command();
                    std::cout << "This config in int8 type is not supported." << std::endl;
                    return;
                }

                bool skip_backward_data    = is_int8;
                bool skip_backward_weights = is_int8;

                // bwd53 kernel (large images supported) doesnt support stride !=1 and dilation and
                // pad.
                if(filter.GetSpatialDimension() == 2 && in_spatial_len[1] >= 2048 &&
                   ((filter.GetConvStrides()[0] != 1) || (filter.GetConvStrides()[1] != 1) ||
                    (filter.GetConvDilations()[0] != 1) || (filter.GetConvDilations()[1] != 1) ||
                    (filter.GetConvPads()[1] != 0) || (filter.GetConvPads()[0] != 0)))
                {
                    return;
                }
#if 0
                size_t total_mem  = workspace_size();
                size_t device_mem = get_handle().GetGlobalMemorySize();

                if(total_mem >= device_mem)
                {
                    show_command();
                    std::cout << "Config requires " << total_mem
                              << " Bytes to write all necessary tensors to GPU. GPU has "
                              << device_mem << " Bytes of memory." << std::endl;
                    return;
                }
#endif
                conv_stats stats;

                using Tacc    = float;
                Tacc init_val = std::numeric_limits<Tacc>::signaling_NaN();
                using Tout    = T;
                using FI      = Fp8Cast<T, T>;
                using FW      = Fp8Cast<T, T>;
                using FO      = Fp8Cast<T, T>;
                const auto is_stoch =
                    filter.attribute.fp8rounding_mode.Get() == miopenF8RoundingModeStochastic;
                uint64_t seed = 0;
                if(is_stoch)
                    seed = filter.attribute.fp8rounding_mode.GetSeed();
                if(do_forward && !skip_forward)
                {
                    input.generate(gen_fp8_value);
                    weights.generate(gen_fp8_value);
                    conv_driver<T>::verify_eps(
                        verify_forward_conv_fp8<ConvApi::Immediate, T, Tacc, Tout, FI, FW>{
                            input,
                            weights,
                            output,
                            filter,
                            stats,
                            0,
                            search,
                            init_val /*init*/,
                            FI{seed, is_stoch} /*in cast*/,
                            FW{seed, is_stoch} /* weight cast*/});
                }

                if(do_backward_data && !skip_backward_data)
                {
                    // input.generate(gen_fp8_value);
                    output.generate(gen_fp8_value);
                    weights.generate(gen_fp8_value);
                    conv_driver<T>::verify_eps(
                        verify_backward_data_conv_fp8<ConvApi::Immediate, T, Tacc, Tout, FW, FO>{
                            input,
                            weights,
                            output,
                            filter,
                            stats,
                            0,
                            search,
                            init_val,
                            FW{seed, is_stoch},
                            FI{seed, is_stoch}});
                }

                if(do_backward_weights && !skip_backward_weights)
                {
                    output.generate(gen_fp8_value);

                    conv_driver<T>::verify_eps(
                        verify_backward_weights_conv_fp8<ConvApi::Immediate, T, Tacc, Tout, FI, FO>{
                            input,
                            weights,
                            output,
                            filter,
                            stats,
                            0,
                            search,
                            init_val,
                            FI{seed, is_stoch},
                            FO{seed, is_stoch}});
                }
            }
        }
    }
};
