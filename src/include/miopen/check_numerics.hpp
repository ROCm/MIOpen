#include <miopen/common.hpp>
#include <miopen/util.hpp>

namespace miopen {
    int CheckNumericsEnabled(int bitMask=-1);

    bool checkNumericsInput(Handle &handle, const TensorDescriptor &dDesc, ConstData_t data);
    bool checkNumericsOutput(Handle &handle, const TensorDescriptor &dDesc, Data_t data);
} // namespace miopen
