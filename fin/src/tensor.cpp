#include "tensor.hpp"

namespace fin {

tensor::randInit(double dataScale, double min, double max)
{
    // different for fwd/wrw and bwd
    for(auto& it : cpuData)
    {
        // it =
    }
}

size_t tensor::size()
{
    // TODO: check that all internal storages have the same size
    return desc.GetElementSize();
}

size_t tensor::elem_size(miopenDataType_t data_type)
{
    switch(data_type)
    {
    case miopenFloat: return sizeof(float);
    case miopenHalf: return sizeof(float16);
    case miopenBFloat16: return sizeof(bfloat16);
    case miopenInt8: return sizeof(int8_t);
    }
}

} // namespace fin
