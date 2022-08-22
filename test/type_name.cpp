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


#include <miopen/type_name.hpp>
#include "test.hpp"

struct global_class
{
    struct inner_class
    {
    };
};

namespace foo {

struct ns_class
{
    struct inner_class
    {
    };
};
} // namespace foo

int main()
{
    EXPECT_EQUAL(miopen::get_type_name<global_class>(), "global_class");
    EXPECT_EQUAL(miopen::get_type_name<global_class::inner_class>(), "global_class::inner_class");
    EXPECT_EQUAL(miopen::get_type_name<foo::ns_class>(), "foo::ns_class");
    EXPECT_EQUAL(miopen::get_type_name<foo::ns_class::inner_class>(), "foo::ns_class::inner_class");
}
