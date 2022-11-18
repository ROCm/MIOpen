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
#include <miopen/handle.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

#include "driver.hpp"
#include "tensor_holder.hpp"

template <class T>
struct check_numerics_base
{
    static const int size = 42;
    miopen::Handle h{};
    miopen::TensorDescriptor desc{miopen_type<T>{}, {size}};
    miopen::Allocator::ManageDataPtr buffer = nullptr;

    check_numerics_base(T val = 0.0)
    {
        std::vector<T> data(size, val);
        buffer = h.Write(data);
    }
};

template <class T>
struct check_numeric_normal : check_numerics_base<T>
{
    check_numeric_normal(T val) : check_numerics_base<T>(val) {}

    void run()
    {
        CHECK(!miopen::checkNumericsImpl(
            this->h, miopen::CheckNumerics::Throw, this->desc, this->buffer.get(), true));
        CHECK(!miopen::checkNumericsImpl(
            this->h, miopen::CheckNumerics::Throw, this->desc, this->buffer.get(), false));

        CHECK(!miopen::checkNumericsImpl(this->h,
                                         miopen::CheckNumerics::Throw |
                                             miopen::CheckNumerics::ComputeStats,
                                         this->desc,
                                         this->buffer.get(),
                                         true));
        CHECK(!miopen::checkNumericsImpl(this->h,
                                         miopen::CheckNumerics::Throw |
                                             miopen::CheckNumerics::ComputeStats,
                                         this->desc,
                                         this->buffer.get(),
                                         false));
    }
};

template <class T>
struct check_numeric_abnormal : check_numerics_base<T>
{
    check_numeric_abnormal(T val) : check_numerics_base<T>(val) {}

    void run()
    {
        CHECK(miopen::checkNumericsImpl(
            this->h, miopen::CheckNumerics::Warn, this->desc, this->buffer.get(), true));
        CHECK(miopen::checkNumericsImpl(
            this->h, miopen::CheckNumerics::Warn, this->desc, this->buffer.get(), false));

        CHECK(throws([&] {
            miopen::checkNumericsImpl(
                this->h, miopen::CheckNumerics::Throw, this->desc, this->buffer.get(), true);
        }));
        CHECK(throws([&] {
            miopen::checkNumericsImpl(
                this->h, miopen::CheckNumerics::Throw, this->desc, this->buffer.get(), false);
        }));

        CHECK(miopen::checkNumericsImpl(this->h,
                                        miopen::CheckNumerics::Warn |
                                            miopen::CheckNumerics::ComputeStats,
                                        this->desc,
                                        this->buffer.get(),
                                        true));
        CHECK(miopen::checkNumericsImpl(this->h,
                                        miopen::CheckNumerics::Warn |
                                            miopen::CheckNumerics::ComputeStats,
                                        this->desc,
                                        this->buffer.get(),
                                        false));

        CHECK(throws([&] {
            miopen::checkNumericsImpl(this->h,
                                      miopen::CheckNumerics::Throw |
                                          miopen::CheckNumerics::ComputeStats,
                                      this->desc,
                                      this->buffer.get(),
                                      true);
        }));
        CHECK(throws([&] {
            miopen::checkNumericsImpl(this->h,
                                      miopen::CheckNumerics::Throw |
                                          miopen::CheckNumerics::ComputeStats,
                                      this->desc,
                                      this->buffer.get(),
                                      false);
        }));
    }
};

template <class T>
struct numeric_0 : check_numeric_normal<T>
{
    numeric_0() : check_numeric_normal<T>(T(0)) {}
};

template <class T>
struct numeric_1 : check_numeric_normal<T>
{
    numeric_1() : check_numeric_normal<T>(T(1)) {}
};

template <class T>
struct numeric_nan : check_numeric_abnormal<T>
{
    numeric_nan() : check_numeric_abnormal<T>(std::numeric_limits<T>::quiet_NaN()) {}
};

template <class T>
struct numeric_inf : check_numeric_abnormal<T>
{
    numeric_inf() : check_numeric_abnormal<T>(std::numeric_limits<T>::infinity()) {}
};

int main(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);
    as.emplace_back("--float");
    for(auto&& arg : as)
    {
        if(arg == "--half")
        {
            run_test<numeric_0<half_float::half>>();
            run_test<numeric_1<half_float::half>>();
            run_test<numeric_nan<half_float::half>>();
            run_test<numeric_inf<half_float::half>>();
            break;
        }
        if(arg == "--bfloat16")
        {
            run_test<numeric_0<bfloat16>>();
            run_test<numeric_1<bfloat16>>();
            run_test<numeric_nan<bfloat16>>();
            run_test<numeric_inf<bfloat16>>();
            break;
        }
        if(arg == "--float")
        {
            run_test<numeric_0<float>>();
            run_test<numeric_1<float>>();
            run_test<numeric_nan<float>>();
            run_test<numeric_inf<float>>();
            break;
        }
    }
}
