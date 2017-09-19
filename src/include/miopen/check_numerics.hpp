#include <miopen/common.hpp>
#include <miopen/util.hpp>

namespace miopen {
    int CheckNumericsEnabled(int bitmask=-1);

    bool checkNumericsInput(Handle &handle, const TensorDescriptor &dDesc, ConstData_t d);
    bool checkNumericsOutput(Handle &handle, const TensorDescriptor &dDesc, Data_t d);
};
