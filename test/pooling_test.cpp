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

static int num_uint16_case = 0;
static int num_uint32_case = 0;
static int num_uint64_case = 0;

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

struct verify_forward_pooling
{
    template <class T, class Index>
    tensor<T>
    cpu(const tensor<T>& input, const miopen::PoolingDescriptor& filter, std::vector<Index>&) const
    {
        auto out = get_output_tensor(filter, input);

        int in_h, in_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

        int stride_h, stride_w, pad_h, pad_w, window_h, window_w;
        std::tie(stride_h, stride_w) = miopen::tien<2>(filter.GetStrides());
        std::tie(pad_h, pad_w)       = miopen::tien<2>(filter.GetPads());
        std::tie(window_h, window_w) = miopen::tien<2>(filter.GetLengths());

        auto op = pooling_operators<T>{filter};

        out.par_for_each([&](int o, int w, int i, int j) {
            const int start_x0 = i * stride_h - pad_h;
            const int start_y0 = j * stride_w - pad_w;

            const int hend = std::min(start_x0 + window_h, in_h);
            const int wend = std::min(start_y0 + window_w, in_w);

            const int start_x = std::max(start_x0, 0);
            const int start_y = std::max(start_y0, 0);

            const int w_h = (hend - start_x);
            const int w_w = (wend - start_y);
            int pool_size = std::max(w_h * w_w, 1);
            if(filter.GetMode() == miopenPoolingAverageInclusive)
                pool_size = window_h * window_w;

            double acc = op.start();
            ford(w_h, w_w)([&](int x, int y) {
                const int in_x = start_x + x;
                const int in_y = start_y + y;
                if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                {
                    acc = op(acc, input(o, w, in_x, in_y));
                }
            });
            out(o, w, i, j) = T(op.final(acc, pool_size));
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

struct verify_backward_pooling
{
    template <class T, class Index>
    tensor<T> cpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices) const
    {
        auto dinput = input;
        CHECK(dout.desc == out.desc);
        std::fill(dinput.begin(), dinput.end(), 0.0);

        int in_h, in_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = miopen::tien<4>(dinput.desc.GetLengths());

        int stride_h, stride_w, pad_h, pad_w, window_h, window_w;
        std::tie(stride_h, stride_w) = miopen::tien<2>(filter.GetStrides());
        std::tie(pad_h, pad_w)       = miopen::tien<2>(filter.GetPads());
        std::tie(window_h, window_w) = miopen::tien<2>(filter.GetLengths());

        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(out.desc.GetLengths());

        par_ford(out_n, out_c)([&](int o, int w) {
            if(filter.GetMode() == miopenPoolingMax)
            {
                ford(out_h, out_w)([&](int i, int j) {
                    auto idx   = indices.at(dout.desc.GetIndex(o, w, i, j));
                    auto idx_h = idx / window_w;
                    auto idx_w = idx % window_w;
                    auto in_y  = i * stride_h - pad_h + idx_h;
                    auto in_x  = j * stride_w - pad_w + idx_w;
                    if(in_y >= 0 && in_x >= 0 && in_y < in_h && in_x < in_w)
                    {
                        CHECK(miopen::float_equal(input(o, w, in_y, in_x), out(o, w, i, j)));
                        dinput(o, w, in_y, in_x) += dout(o, w, i, j);
                    }
                });
            }
            else
            {
                ford(out_h, out_w, window_h, window_w)([&](int i, int j, int x, int y) {
                    const int start_x0 = i * stride_h - pad_h;
                    const int start_y0 = j * stride_w - pad_w;

                    const int hend    = std::min(start_x0 + window_h, in_h);
                    const int wend    = std::min(start_y0 + window_w, in_w);
                    const int start_x = std::max(start_x0, 0);
                    const int start_y = std::max(start_y0, 0);
                    const int w_h     = (hend - start_x);
                    const int w_w     = (wend - start_y);
                    int pool_size     = std::max(w_h * w_w, 1);
                    if(filter.GetMode() == miopenPoolingAverageInclusive)
                        pool_size = window_h * window_w;

                    const int in_x = start_x0 + x;
                    const int in_y = start_y0 + y;
                    if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                    {
                        dinput(o, w, in_x, in_y) += dout(o, w, i, j) / pool_size;
                    }
                });
            }
        });
        return dinput;
    }

    template <class T, class Index>
    tensor<T> gpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices) const
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
              const std::vector<Index>&) const
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
    tensor<T> input;
    std::vector<int> lens;
    std::vector<int> pads;
    std::vector<int> strides;
    std::string index_type;
    std::string mode;
    std::string pmode;
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

    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"DEFAULT", miopenPaddingDefault},
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
    };

    pooling_driver()
    {
        add(input,
            "input",
            get_input_tensor(tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17}));
        add(lens, "lens", generate_data({{2, 2}, {3, 3}}));
        add(strides, "strides", generate_data({{2, 2}, {1, 1}}));
        add(pads, "pads", generate_data({{0, 0}, {1, 1}}));
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
        add(pmode, "pmode", generate_data({"default", "same", "valid"}));
    }

    template <class Index>
    void run_impl()
    {
        std::vector<Index> indices{};
        auto out  = verify(verify_forward_pooling{}, input, filter, indices);
        auto dout = out.first;
        dout.generate(tensor_elem_gen_integer{2503});
        verify(verify_backward_pooling{}, input, dout, out.first, filter, indices);
    }

    void run()
    {
        int in_h, in_w, window_h, window_w, out_h, out_w, pad_h, pad_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

        filter = miopen::PoolingDescriptor{mode_lookup.at(miopen::ToUpper(mode)),
                                           pmode_lookup.at(miopen::ToUpper(pmode)),
                                           lens,
                                           strides,
                                           pads};

        filter.SetIndexType(index_type_lookup.at(miopen::ToUpper(index_type)));

        std::tie(window_h, window_w) = miopen::tien<2>(filter.GetLengths());
        if(filter.pmode == miopenPaddingSame)
        {
            if(filter.strides[0] == 0 || filter.strides[1] == 0)
                return;
            auto _pad_w = (in_h % filter.strides[0] == 0)
                              ? (std::max((window_h - filter.strides[0]), 0))
                              : (std::max((window_h - (in_h % filter.strides[0])), 0));
            auto _pad_h = (in_w % filter.strides[1] == 0)
                              ? (std::max((window_w - filter.strides[1]), 0))
                              : (std::max((window_w - (in_w % filter.strides[1])), 0));

            filter.pads[0] = _pad_h / 2;
            filter.pads[1] = _pad_w / 2;

            out_h = std::ceil(static_cast<double>(in_h) / filter.strides[0]);
            out_w = std::ceil(static_cast<double>(in_w) / filter.strides[1]);

            if(out_h <= 0 || out_w <= 0)
                return;
        }
        else if(filter.pmode == miopenPaddingValid)
        {
            if(filter.strides[0] == 0 || filter.strides[1] == 0)
                return;
            filter.pads[0] = 0;
            filter.pads[1] = 0;

            out_h = std::ceil(static_cast<double>(in_h - filter.lens[0] + 1) / filter.strides[0]);
            out_w = std::ceil(static_cast<double>(in_w - filter.lens[1] + 1) / filter.strides[1]);

            if(out_h <= 0 || out_w <= 0)
                return;
        }

        std::tie(pad_h, pad_w) = miopen::tien<2>(filter.GetPads());

        if((window_h < (in_h + 2 * pad_h)) && (window_w < (in_w + 2 * pad_w)))
        {
            switch(filter.GetIndexType())
            {
            case miopenIndexUint8:
            {
                run_impl<uint8_t>();
                break;
            }
            case miopenIndexUint16:
            {
                // test_pooling_test --all only test 10 uint16 cases
                if(num_uint16_case > 5)
                {
                    return;
                }

                ++num_uint16_case;
                run_impl<uint16_t>();
                break;
            }
            case miopenIndexUint32:
            {
                // test_pooling_test --all only test 5 uint32 cases
                if(num_uint32_case > 5)
                {
                    return;
                }

                ++num_uint32_case;
                run_impl<uint32_t>();
                break;
            }
            case miopenIndexUint64:
            {
                // test_pooling_test --all only test 5 uint64 cases
                if(num_uint64_case > 5)
                {
                    return;
                }

                ++num_uint64_case;
                run_impl<uint64_t>();
                break;
            }
            }
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<pooling_driver>(argc, argv); }
