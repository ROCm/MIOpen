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

struct check_numerics_base
{
    static const int size = 42;
    miopen::Handle h{};
    miopen::TensorDescriptor desc{miopenFloat, {size}};
    miopen::Allocator::ManageDataPtr buffer = nullptr;

    check_numerics_base(float val = 0.0)
    {
        std::vector<float> data(size, val);
        buffer = h.Write(data);
    }
};

struct check_numeric_normal : check_numerics_base
{
    check_numeric_normal(float val) : check_numerics_base(val) {}

    void run()
    {
        CHECK(
            !miopen::checkNumericsImpl(h, miopen::CheckNumerics::Throw, desc, buffer.get(), true));
        CHECK(
            !miopen::checkNumericsImpl(h, miopen::CheckNumerics::Throw, desc, buffer.get(), false));

        CHECK(!miopen::checkNumericsImpl(h,
                                         miopen::CheckNumerics::Throw |
                                             miopen::CheckNumerics::ComputeStats,
                                         desc,
                                         buffer.get(),
                                         true));
        CHECK(!miopen::checkNumericsImpl(h,
                                         miopen::CheckNumerics::Throw |
                                             miopen::CheckNumerics::ComputeStats,
                                         desc,
                                         buffer.get(),
                                         false));
    }
};

struct check_numeric_abnormal : check_numerics_base
{
    check_numeric_abnormal(float val) : check_numerics_base(val) {}

    void run()
    {
        CHECK(miopen::checkNumericsImpl(h, miopen::CheckNumerics::Warn, desc, buffer.get(), true));
        CHECK(miopen::checkNumericsImpl(h, miopen::CheckNumerics::Warn, desc, buffer.get(), false));

        CHECK(throws([&] {
            miopen::checkNumericsImpl(h, miopen::CheckNumerics::Throw, desc, buffer.get(), true);
        }));
        CHECK(throws([&] {
            miopen::checkNumericsImpl(h, miopen::CheckNumerics::Throw, desc, buffer.get(), false);
        }));

        CHECK(miopen::checkNumericsImpl(h,
                                        miopen::CheckNumerics::Warn |
                                            miopen::CheckNumerics::ComputeStats,
                                        desc,
                                        buffer.get(),
                                        true));
        CHECK(miopen::checkNumericsImpl(h,
                                        miopen::CheckNumerics::Warn |
                                            miopen::CheckNumerics::ComputeStats,
                                        desc,
                                        buffer.get(),
                                        false));

        CHECK(throws([&] {
            miopen::checkNumericsImpl(h,
                                      miopen::CheckNumerics::Throw |
                                          miopen::CheckNumerics::ComputeStats,
                                      desc,
                                      buffer.get(),
                                      true);
        }));
        CHECK(throws([&] {
            miopen::checkNumericsImpl(h,
                                      miopen::CheckNumerics::Throw |
                                          miopen::CheckNumerics::ComputeStats,
                                      desc,
                                      buffer.get(),
                                      false);
        }));
    }
};

struct numeric_0 : check_numeric_normal
{
    numeric_0() : check_numeric_normal(0.0) {}
};

struct numeric_1 : check_numeric_normal
{
    numeric_1() : check_numeric_normal(1.0) {}
};

struct numeric_nan : check_numeric_abnormal
{
    numeric_nan() : check_numeric_abnormal(std::numeric_limits<float>::quiet_NaN()) {}
};

struct numeric_inf : check_numeric_abnormal
{
    numeric_inf() : check_numeric_abnormal(std::numeric_limits<float>::infinity()) {}
};

int main()
{
    run_test<numeric_0>();
    run_test<numeric_1>();
    run_test<numeric_nan>();
    run_test<numeric_inf>();
}
