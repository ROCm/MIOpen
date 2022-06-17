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

size_t tensor::elem_size(miopen::DataType data_type)
{
    switch(data_type)
    {
    case miopen::DataType::Float: return sizeof(float);
    case miopen::DataType::Half: return sizeof(float16);
    case miopen::DataType::BFloat16: return sizeof(bfloat16);
    case miopen::DataType::Int8: return sizeof(int8_t);
    }
}

} // namespace fin
