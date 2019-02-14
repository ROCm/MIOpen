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
int CheckNumericsEnabled(int bitMask = -1);

bool checkNumericsInput(Handle& handle, const TensorDescriptor& dDesc, ConstData_t data);
bool checkNumericsOutput(Handle& handle, const TensorDescriptor& dDesc, ConstData_t data);
bool checkNumericsImpl(
    Handle& handle, int mode, const TensorDescriptor& dDesc, ConstData_t data, bool isInput);
} // namespace miopen

#endif // GUARD_MIOPEN_CHECK_NUMERICS_HPP
