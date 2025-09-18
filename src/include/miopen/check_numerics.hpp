// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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

MIOPEN_INTERNALS_EXPORT bool CheckNumericsEnabled(int bitMask = -1);

MIOPEN_INTERNALS_EXPORT bool
checkNumericsInput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data);
MIOPEN_INTERNALS_EXPORT bool
checkNumericsOutput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data);
MIOPEN_INTERNALS_EXPORT bool checkNumericsImpl(
    const Handle& handle, int mode, const TensorDescriptor& dDesc, ConstData_t data, bool isInput);
} // namespace miopen

#endif // GUARD_MIOPEN_CHECK_NUMERICS_HPP
