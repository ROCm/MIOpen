#ifndef GUARD_MIOPEN_GPU_REFERENCE_KERNEL_HPP
#define GUARD_MIOPEN_GPU_REFERENCE_KERNEL_HPP

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct ProblemDescription;

void GPUReferenceConvolutionForward(const Handle& handle,
                                    const ProblemDescription& conv_param,
                                    ConstData_t input_data,
                                    ConstData_t weight_data,
                                    Data_t output_data);
void GPUReferenceConvolutionBackwardData(const Handle& handle,
                                         const ProblemDescription& conv_param,
                                         Data_t input_data,
                                         ConstData_t weight_data,
                                         ConstData_t output_data);
void GPUReferenceConvolutionBackwardWeights(const Handle& handle,
                                            const ProblemDescription& conv_param,
                                            ConstData_t input_data,
                                            Data_t weight_data,
                                            ConstData_t output_data);
} // namespace miopen

#endif // GUARD_MIOPEN_GPU_REFERENCE_KERNEL_HPP
