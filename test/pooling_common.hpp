/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_TEST_POOLING_COMMON_HPP
#define GUARD_MIOPEN_TEST_POOLING_COMMON_HPP

#include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/pooling.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/tensor.hpp>
#include <utility>

// #include "network_data.hpp"
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "cpu_conv.hpp"

#define TEST_PADDING_MODE 0
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static int num_uint16_case = 0;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static int num_uint32_case = 0;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static int num_uint32_case_imgidx = 0;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static int num_uint64_case = 0;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static int num_uint64_case_imgidx = 0;

template <class T>
tensor<T> get_output_tensor(const miopen::PoolingDescriptor& filter, const tensor<T>& input)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc)};
}

template <class T>
struct pooling_operators
{
    miopen::PoolingDescriptor filter;
    pooling_operators(miopen::PoolingDescriptor f) : filter(f) {}

    double start() const
    {
        if(filter.GetMode() == miopenPoolingMax)
            return std::numeric_limits<T>::lowest();
        else
            return 0.0;
    }

    double operator()(double x, double y) const
    {
        if(filter.GetMode() == miopenPoolingMax)
        {
            double m = std::max(x, y);
            return (m);
        }
        else
            return x + y;
    }

    double final(double x, double y)
    {
        if(filter.GetMode() == miopenPoolingMax)
        {
            return (x);
        }
        else
            return x / y;
    }
};

template <int SptDim>
struct verify_forward_pooling
{
    template <class T, class Index>
    tensor<T>
    cpu(const tensor<T>& input, const miopen::PoolingDescriptor& filter, std::vector<Index>&) const
    {
        auto out = get_output_tensor(filter, input);

        std::array<int, SptDim> in_dim{};
        std::copy_n(input.desc.GetLengths().begin() + 2, SptDim, in_dim.begin());
        std::array<int, SptDim> strides{};
        std::copy_n(filter.GetStrides().begin(), SptDim, strides.begin());
        std::array<int, SptDim> pads{};
        std::copy_n(filter.GetPads().begin(), SptDim, pads.begin());
        std::array<int, SptDim> kers{};
        std::copy_n(filter.GetLengths().begin(), SptDim, kers.begin());
        auto op = pooling_operators<T>{filter};

        int b_n = out.desc.GetLengths()[0];
        int k_n = out.desc.GetLengths()[1];
        std::array<int, SptDim> out_spatial_len{};
        std::copy_n(out.desc.GetLengths().begin() + 2, SptDim, out_spatial_len.begin());

        auto par_ford_out =
            miopen::unpacker(miopen::prepender(par_ford, b_n, k_n))(out_spatial_len);

        par_ford_out([&](int o, int w, auto... out_spatial_id_pack) {
            auto out_spatial_id = make_array(out_spatial_id_pack...);

            std::array<int, SptDim> start_idx{};
            std::array<int, SptDim> win_sz{};
            for(int i = 0; i < SptDim; ++i)
            {
                start_idx[i] = out_spatial_id[i] * strides[i] - pads[i];
                int end_idx  = start_idx[i] + kers[i];
                end_idx      = std::min(end_idx, in_dim[i]);
                start_idx[i] = std::max(start_idx[i], 0);
                win_sz[i]    = end_idx - start_idx[i];
                win_sz[i]    = std::max(win_sz[i], 1);
            }

            int pool_size =
                filter.GetMode() == miopenPoolingAverageInclusive
                    ? std::accumulate(kers.begin(), kers.end(), 1, std::multiplies<int>())
                    : std::accumulate(win_sz.begin(), win_sz.end(), 1, std::multiplies<int>());

            double acc = op.start();
            miopen::unpacker(ford)(win_sz)([&](auto... in_spatial_id_pack) {
                auto in_spatial_id = make_array(in_spatial_id_pack...);
                std::array<std::size_t, SptDim + 2> idx{};
                idx[0] = o;
                idx[1] = w;

                bool in_cmp_idx = true;
                for(int i = 0; i < SptDim; ++i)
                {
                    idx[i + 2] = start_idx[i] + in_spatial_id[i];
                    in_cmp_idx &= (in_dim[i] > idx[i + 2]);
                }

                if(in_cmp_idx)
                {
                    acc = op(acc, input(idx));
                }
            });
            out(o, w, out_spatial_id_pack...) = T(op.final(acc, pool_size));
        });
        return out;
    }

    template <class T, class Index>
    tensor<T> gpu(const tensor<T>& input,
                  const miopen::PoolingDescriptor& filter,
                  std::vector<Index>& indices) const
    {
        auto&& handle = get_handle();
        auto out      = get_output_tensor(filter, input);
        indices.resize(out.data.size(), 0);

        auto in_dev        = handle.Write(input.data);
        auto out_dev       = handle.Create<T>(out.data.size());
        auto workspace_dev = handle.Write(indices);

        float alpha = 1, beta = 0;
        filter.Forward(handle,
                       &alpha,
                       input.desc,
                       in_dev.get(),
                       &beta,
                       out.desc,
                       out_dev.get(),
                       true,
                       workspace_dev.get(),
                       indices.size() * sizeof(Index));

        indices  = handle.Read<Index>(workspace_dev, indices.size());
        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template <class T, class Index>
    void fail(float,
              const tensor<T>& input,
              const miopen::PoolingDescriptor& filter,
              const std::vector<Index>&) const
    {
        std::cout << "Forward pooling: ";
        if(filter.GetMode() == miopenPoolingAverage)
            std::cout << "Average";
        else if(filter.GetMode() == miopenPoolingAverageInclusive)
            std::cout << "AverageInclusive";
        else
            std::cout << "Max";
        std::cout << std::endl;
        std::cout << "Lengths: ";
        miopen::LogRange(std::cout, filter.GetLengths(), ", ") << std::endl;
        std::cout << "Pads: ";
        miopen::LogRange(std::cout, filter.GetPads(), ", ") << std::endl;
        std::cout << "Strides: ";
        miopen::LogRange(std::cout, filter.GetStrides(), ", ") << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Output tensor: " << filter.GetForwardOutputTensor(input.desc).ToString()
                  << std::endl;
    }
};

template <int SptDim>
struct verify_backward_pooling
{
    template <class T, class Index>
    tensor<T> cpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices,
                  bool use_global_index,
                  bool verify_index) const
    {
        auto dinput = input;
        std::vector<T> din_vec(input.desc.GetElementSpace(), T(0));
        CHECK(dout.desc == out.desc);
        std::array<int, SptDim + 2> in_dim{};
        std::copy_n(input.desc.GetLengths().begin(), SptDim + 2, in_dim.begin());
        std::array<int, SptDim + 2> in_str{};
        std::copy_n(input.desc.GetStrides().begin(), SptDim + 2, in_str.begin());
        std::array<int, SptDim> strides{};
        std::copy_n(filter.GetStrides().begin(), SptDim, strides.begin());
        std::array<int, SptDim> pads{};
        std::copy_n(filter.GetPads().begin(), SptDim, pads.begin());
        std::array<int, SptDim> kers{};
        std::copy_n(filter.GetLengths().begin(), SptDim, kers.begin());
        auto ford_ker = miopen::unpacker(ford)(kers);

        int out_n = out.desc.GetLengths()[0];
        int out_c = out.desc.GetLengths()[1];
        std::array<int, SptDim> out_spatial_len{};
        std::copy_n(out.desc.GetLengths().begin() + 2, SptDim, out_spatial_len.begin());
        auto ford_out = miopen::unpacker(ford)(out_spatial_len);

        par_ford(out_n, out_c)([&](int o, int w) {
            if(filter.GetMode() == miopenPoolingMax)
            {
                ford_out([&](auto... out_spatial_id_pack) {
                    auto mx_idx = indices.at(dout.desc.GetIndex(o, w, out_spatial_id_pack...));
                    std::array<std::size_t, SptDim + 2> idx{};
                    bool in_cmp_idx = true;
                    if(use_global_index)
                    {
                        for(int i = 0; i < SptDim; i++)
                        {
                            std::size_t mx_idx_dim = mx_idx;
                            mx_idx_dim /= std::accumulate(in_dim.begin() + i + 3,
                                                          in_dim.end(),
                                                          1,
                                                          std::multiplies<std::size_t>());
                            mx_idx_dim %= in_dim[i + 2];
                            idx[i + 2] = mx_idx_dim;
                        }
                    }
                    else
                    {
                        auto out_spatial_id = make_array(out_spatial_id_pack...);

                        for(int i = 0; i < SptDim; i++)
                        {
                            int mx_idx_dim = mx_idx;
                            mx_idx_dim /= std::accumulate(
                                kers.begin() + i + 1, kers.end(), 1, std::multiplies<int>());
                            mx_idx_dim %= kers[i];

                            mx_idx_dim += (out_spatial_id[i] * strides[i] - pads[i]);
                            in_cmp_idx &= (in_dim[i + 2] > mx_idx_dim && mx_idx_dim >= 0);

                            idx[i + 2] = std::size_t(mx_idx_dim);
                        }
                    }

                    if(in_cmp_idx)
                    {
                        idx[0] = o;
                        idx[1] = w;
                        if(verify_index)
                        {
                            CHECK(
                                miopen::float_equal(input(idx), out(o, w, out_spatial_id_pack...)));
                        }
                        std::size_t din_idx = 0;
                        for(int i = 0; i < SptDim + 2; i++)
                        {
                            din_idx += idx[i] * in_str[i];
                        }
                        din_vec.at(din_idx) += dout(o, w, out_spatial_id_pack...);
                    }
                });
            }
            else
            {
                ford_out([&](auto... out_spatial_id_pack) {
                    auto out_spatial_id = make_array(out_spatial_id_pack...);

                    std::array<int, SptDim> start_idx{};
                    std::array<int, SptDim> win_sz{};
                    for(int i = 0; i < SptDim; ++i)
                    {
                        start_idx[i] = out_spatial_id[i] * strides[i] - pads[i];
                        int end_idx  = start_idx[i] + kers[i];
                        end_idx      = std::min(end_idx, in_dim[i + 2]);
                        win_sz[i]    = end_idx - std::max(start_idx[i], 0);
                        win_sz[i]    = std::max(win_sz[i], 1);
                    }

                    int pool_size =
                        filter.GetMode() == miopenPoolingAverageInclusive
                            ? std::accumulate(kers.begin(), kers.end(), 1, std::multiplies<int>())
                            : std::accumulate(
                                  win_sz.begin(), win_sz.end(), 1, std::multiplies<int>());

                    ford_ker([&](auto... ker_id_pack) {
                        auto ker_id = make_array(ker_id_pack...);

                        bool in_cmp_idx = true;
                        std::array<int, SptDim + 2> in_idx{};
                        in_idx[0] = o;
                        in_idx[1] = w;
                        for(int i = 0; i < SptDim; ++i)
                        {
                            in_idx[i + 2] = start_idx[i] + ker_id[i];
                            in_cmp_idx &= (in_dim[i + 2] > in_idx[i + 2] && in_idx[i + 2] >= 0);
                        }

                        if(in_cmp_idx)
                        {
                            std::size_t din_idx = 0;
                            for(int i = 0; i < SptDim + 2; i++)
                            {
                                din_idx += in_idx[i] * in_str[i];
                            }

                            din_vec.at(din_idx) += dout(o, w, out_spatial_id_pack...) / pool_size;
                        }
                    });
                });
            }
        });

        miopen::unpacker(ford)(in_dim)([&](auto... in_id_pack) {
            auto in_id          = make_array(in_id_pack...);
            std::size_t din_idx = 0;
            for(int i = 0; i < SptDim + 2; i++)
            {
                din_idx += in_id[i] * in_str[i];
            }
            dinput(in_id_pack...) = din_vec.at(din_idx);
        });
        return dinput;
    }

    template <class T, class Index>
    tensor<T> gpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices,
                  bool,
                  bool) const
    {
        auto&& handle = get_handle();
        auto dinput   = input;

        auto in_dev   = handle.Write(input.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(out.data);
        auto din_dev  = handle.Create<T>(dinput.data.size());

        // std::vector<char> workspace(filter.GetWorkSpaceSize(out.desc));
        // auto workspace_dev = handle.Write(workspace);
        auto workspace_dev = handle.Write(indices);

        float alpha = 1, beta = 0;
        filter.Backward(handle,
                        &alpha,
                        // y
                        out.desc,
                        out_dev.get(),
                        // dy
                        dout.desc,
                        dout_dev.get(),
                        // x
                        input.desc,
                        in_dev.get(),
                        &beta,
                        // dx
                        dinput.desc,
                        din_dev.get(),
                        workspace_dev.get());

        dinput.data = handle.Read<T>(din_dev, dinput.data.size());
        return dinput;
    }

    template <class T, class Index>
    void fail(float,
              const tensor<T>& input,
              const tensor<T>&,
              const tensor<T>& out,
              const miopen::PoolingDescriptor& filter,
              const std::vector<Index>&,
              bool,
              bool) const
    {
        std::cout << "Backward pooling: ";
        if(filter.GetMode() == miopenPoolingAverage)
            std::cout << "Average";
        else if(filter.GetMode() == miopenPoolingAverageInclusive)
            std::cout << "AverageInclusive";
        else
            std::cout << "Max";
        std::cout << std::endl;
        std::cout << "Lengths: ";
        miopen::LogRange(std::cout, filter.GetLengths(), ", ") << std::endl;
        std::cout << "Pads: ";
        miopen::LogRange(std::cout, filter.GetPads(), ", ") << std::endl;
        std::cout << "Strides: ";
        miopen::LogRange(std::cout, filter.GetStrides(), ", ") << std::endl;
        std::cout << "Output tensor: " << out.desc.ToString() << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct pooling_driver : test_driver
{
    miopen::PoolingDescriptor filter;
    std::vector<int> in_shape;
    std::vector<int> lens;
    std::vector<int> pads;
    std::vector<int> strides;
    std::string index_type;
    std::string mode;
#if TEST_PADDING_MODE == 1
    std::string pmode;
#endif
    int verify_indices{};
    int wsidx{};
    std::unordered_map<std::string, miopenIndexType_t> index_type_lookup = {
        {miopen::ToUpper("miopenIndexUint8"), miopenIndexUint8},
        {miopen::ToUpper("miopenIndexUint16"), miopenIndexUint16},
        {miopen::ToUpper("miopenIndexUint32"), miopenIndexUint32},
        {miopen::ToUpper("miopenIndexUint64"), miopenIndexUint64},
    };
    std::unordered_map<std::string, miopenPoolingMode_t> mode_lookup = {
        {"MAX", miopenPoolingMax},
        {"MIOPENPOOLINGMAX", miopenPoolingMax},
        {"AVERAGE", miopenPoolingAverage},
        {"MIOPENPOOLINGAVERAGE", miopenPoolingAverage},
        {"AVERAGEINCLUSIVE", miopenPoolingAverageInclusive},
        {"MIOPENPOOLINGAVERAGEINCLUSIVE", miopenPoolingAverageInclusive},
    };
#if TEST_PADDING_MODE == 1
    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"DEFAULT", miopenPaddingDefault},
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
    };
#endif
    pooling_driver()
    {
        add(index_type,
            "index_type",
            generate_data({"miopenIndexUint8",
                           "miopenIndexUint16",
                           "miopenIndexUint32",
                           "miopenIndexUint64"}));
        add(mode,
            "mode",
            generate_data(
                {"miopenPoolingMax", "miopenPoolingAverage", "miopenPoolingAverageInclusive"}));
#if TEST_PADDING_MODE == 1
        add(pmode, "pmode", generate_data({"default", "same", "valid"}));
#endif
        add(verify_indices, "verify_indices", generate_data({0}));
    }

    template <class Index, int SptDim>
    void run_impl()
    {
        std::vector<Index> indices{};
        auto input = tensor<T>{in_shape}.generate(
            tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17});
        auto out  = verify(verify_forward_pooling<SptDim>{}, input, filter, indices);
        auto dout = out.first;
        dout.generate(tensor_elem_gen_integer{2503});
        verify(verify_backward_pooling<SptDim>{},
               input,
               dout,
               out.first,
               filter,
               indices,
               wsidx != 0,
               static_cast<bool>(this->verify_indices));
    }

    void run()
    {
        auto idx_typ = index_type_lookup.at(miopen::ToUpper(index_type));
        auto idx_sz  = sizeof(uint8_t);
        int spt_dim  = in_shape.size() - 2;
        switch(idx_typ)
        {
        case miopenIndexUint8: {
            // index size too small for 3D image
            if(spt_dim == 3 || (spt_dim == 2 && wsidx == 1))
            {
                return;
            }
            break;
        }
        case miopenIndexUint16: {
            // index size too small for 3D image
            if(spt_dim == 3 || (spt_dim == 2 && wsidx == 1))
            {
                return;
            }

            // test_pooling_test --all only test 5 uint16 cases
            if(num_uint16_case > 5)
            {
                return;
            }
            idx_sz = sizeof(uint16_t);
            ++num_uint16_case;
            break;
        }
        case miopenIndexUint32: {
            // test_pooling_test --all only test 5 uint32 cases
            if(wsidx == 0)
            {
                if(num_uint32_case > 5 || spt_dim == 3)
                    return;

                ++num_uint32_case;
            }
            else
            {
                if(num_uint32_case_imgidx > 5)
                    return;

                ++num_uint32_case_imgidx;
            }

            idx_sz = sizeof(uint32_t);
            break;
        }
        case miopenIndexUint64: {
            if(wsidx == 0)
            {
                if(num_uint64_case > 5 || spt_dim == 3)
                    return;

                ++num_uint64_case;
            }
            else
            {
                if(num_uint64_case_imgidx > 5 && spt_dim == 2)
                    return;

                ++num_uint64_case_imgidx;
            }

            idx_sz = sizeof(uint64_t);
            break;
        }
        }

        auto input_desc = miopen::TensorDescriptor(this->type, in_shape.data(), in_shape.size());

        if(spt_dim != 2 && spt_dim != 3)
        {
            return;
        }

        filter = miopen::PoolingDescriptor
        {
            mode_lookup.at(miopen::ToUpper(mode)),
#if TEST_PADDING_MODE == 1
                pmode_lookup.at(miopen::ToUpper(pmode))
#else
                miopenPaddingDefault
#endif
                    ,
                lens, strides, pads
        };

        filter.SetIndexType(idx_typ);
        filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(wsidx));

        for(int i = 0; i < spt_dim; i++)
            if(lens[i] >= (input_desc.GetLengths()[i + 2] + 2 * pads[i]))
            {
                return;
            }

        auto output_desc = filter.GetForwardOutputTensor(input_desc);
        size_t total_mem = 3 * input_desc.GetNumBytes() + output_desc.GetNumBytes() +
                           idx_sz * output_desc.GetElementSize(); // estimate based on backward pass

        size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        std::vector<int> in_dim(input_desc.GetLengths().begin() + 2, input_desc.GetLengths().end());
        std::vector<int> out_dim(spt_dim);
        std::vector<int> ker_dim(filter.GetLengths().begin(), filter.GetLengths().end());
#if TEST_PADDING_MODE == 1
        if(filter.pmode == miopenPaddingSame)
        {
            if(std::any_of(filter.GetStrides().begin(), filter.GetStrides().end(), [](int i) {
                   return i == 0;
               }))
                return;
            for(int i = 0; i < spt_dim; i++)
            {
                filter.pads[i] =
                    ((in_dim[i] % filter.GetStrides()[i] == 0)
                         ? (std::max((ker_dim[i] - filter.GetStrides()[i]), 0))
                         : (std::max((ker_dim[i] - (in_dim[i] % filter.GetStrides()[i])), 0))) /
                    2;

                out_dim[i] = std::ceil(static_cast<double>(in_dim[i]) / filter.strides[i]);
            }

            if(std::any_of(out_dim.begin(), out_dim.end(), [](int i) { return i <= 0; }))
                return;
        }
        else if(filter.pmode == miopenPaddingValid)
        {
            if(std::any_of(filter.GetStrides().begin(), filter.GetStrides().end(), [](int i) {
                   return i == 0;
               }))
                return;
            for(int i = 0; i < spt_dim; i++)
            {
                filter.pads[i] = 0;

                out_dim[i] = std::ceil(static_cast<double>(in_dim[i] - filter.lens[i] + 1) /
                                       filter.strides[i]);
            }

            if(std::any_of(out_dim.begin(), out_dim.end(), [](int i) { return i <= 0; }))
                return;
        }
#endif
        std::vector<int> check_dim(spt_dim);
        for(int i = 0; i < spt_dim; i++)
        {
            check_dim[i] = in_dim[i] + 2 * filter.GetPads()[i] - ker_dim[i];
        }

        if(std::all_of(check_dim.begin(), check_dim.end(), [](int i) { return i > 0; }))
        {
            switch(filter.GetIndexType())
            {
            case miopenIndexUint8: {
                if(spt_dim == 3)
                {
                    run_impl<uint8_t, 3>();
                }
                else
                {
                    run_impl<uint8_t, 2>();
                }
                break;
            }
            case miopenIndexUint16: {
                if(spt_dim == 3)
                {
                    run_impl<uint16_t, 3>();
                }
                else
                {
                    run_impl<uint16_t, 2>();
                }
                break;
            }
            case miopenIndexUint32: {
                if(spt_dim == 3)
                {
                    run_impl<uint32_t, 3>();
                }
                else
                {
                    run_impl<uint32_t, 2>();
                }
                break;
            }
            case miopenIndexUint64: {
                if(spt_dim == 3)
                {
                    run_impl<uint64_t, 3>();
                }
                else
                {
                    run_impl<uint64_t, 2>();
                }
                break;
            }
            }
        }
    }
};

#endif
