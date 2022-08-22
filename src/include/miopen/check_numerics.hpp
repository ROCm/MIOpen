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

#ifndef GUARD_MIOPEN_CHECK_NUMERICS_HPP
#define GUARD_MIOPEN_CHECK_NUMERICS_HPP

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

struct CheckNumerics
{
    static const int Info         = 0x01; // print results from all checks
    static const int Warn         = 0x02; // print only if abnormal detected
    static const int Throw        = 0x04; // MIOPEN_THROW on abnormal result
    static const int Abort        = 0x08; // abort on abnormal result (to drop into debugger)
    static const int ComputeStats = 0x10; // Print mean/absmean/min/max (slow)
};
bool CheckNumericsEnabled(int bitMask = -1);

bool checkNumericsInput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data);
bool checkNumericsOutput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data);
bool checkNumericsImpl(
    const Handle& handle, int mode, const TensorDescriptor& dDesc, ConstData_t data, bool isInput);
} // namespace miopen

#endif // GUARD_MIOPEN_CHECK_NUMERICS_HPP
