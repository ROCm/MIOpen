/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/errors.hpp>
#include <miopen/graphapi/convolution.hpp>
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/enginecfg.hpp>
#include <miopen/graphapi/engineheur.hpp>
#include <miopen/graphapi/execution_plan.hpp>
#include <miopen/graphapi/graphapi.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/pointwise.hpp>
#include <miopen/graphapi/reduction.hpp>
#include <miopen/graphapi/reshape.hpp>
#include <miopen/graphapi/rng.hpp>
#include <miopen/graphapi/tensor.hpp>
#include <miopen/graphapi/variant_pack.hpp>
#include <miopen/logger.hpp>
#include <miopen/graphapi/matmul.hpp>

#include <memory>

extern "C" miopenStatus_t
miopenBackendCreateDescriptor(miopenBackendDescriptorType_t descriptorType,
                              miopenBackendDescriptor_t* descriptor)
{
    MIOPEN_LOG_FUNCTION(descriptorType, descriptor);
    return miopen::try_([&] {
        auto& outputDescriptor = miopen::deref(descriptor);

        switch(descriptorType)
        {
        /* This part is a common place of changes of about 25 PRs and merge conflicts arise heavily
         * here. Turn off clang-format to keep each line unique to simplify resolving of conflicts.
         *
         * TODO: Turn on clang-format when active phase of development is finished.
         */
        // clang-format off
        case MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendConvolutionDescriptor(); break;

        case MIOPEN_BACKEND_ENGINE_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendEngineDescriptor(); break;

        case MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendEngineCfgDescriptor(); break;

        case MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendEngineHeurDescriptor(); break;

        case MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendExecutionPlanDescriptor(); break;

        case MIOPEN_BACKEND_MATMUL_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendMatmulDescriptor();
            break;

        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationConvolutionForwardDescriptor(); break;

        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationConvolutionBackwardFilterDescriptor(); break;

        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationConvolutionBackwardDataDescriptor(); break;

        case MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationMatmulDescriptor();
            break;

        case MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationPointwiseDescriptor(); break;

        case MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationReductionDescriptor(); break;

        case MIOPEN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationReshapeDescriptor(); break;

        case MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationRngDescriptor(); break;

        case MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendOperationGraphDescriptor(); break;

        case MIOPEN_BACKEND_POINTWISE_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendPointwiseDescriptor(); break;

        case MIOPEN_BACKEND_REDUCTION_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendReductionDescriptor(); break;

        case MIOPEN_BACKEND_RNG_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendRngDescriptor(); break;

        case MIOPEN_BACKEND_TENSOR_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendTensorDescriptor(); break;

        case MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR:
            outputDescriptor = new miopen::graphapi::BackendVariantPackDescriptor(); break;

        default: MIOPEN_THROW(miopenStatusUnsupportedOp);
            // clang-format on
        }
    });
}

extern "C" miopenStatus_t miopenBackendSetAttribute(miopenBackendDescriptor_t descriptor,
                                                    miopenBackendAttributeName_t attributeName,
                                                    miopenBackendAttributeType_t attributeType,
                                                    int64_t elementCount,
                                                    void* arrayOfElements)
{
    MIOPEN_LOG_FUNCTION(attributeName, attributeType, elementCount);

    if(arrayOfElements == nullptr)
    {
        return miopenStatusBadParm;
    }

    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(descriptor);
            theDescriptor.setAttribute(attributeName, attributeType, elementCount, arrayOfElements);
        },
        false);
}

extern "C" miopenStatus_t miopenBackendFinalize(miopenBackendDescriptor_t descriptor)
{
    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(descriptor);
            theDescriptor.finalize();
        },
        false);
}

extern "C" miopenStatus_t miopenBackendGetAttribute(miopenBackendDescriptor_t descriptor,
                                                    miopenBackendAttributeName_t attributeName,
                                                    miopenBackendAttributeType_t attributeType,
                                                    int64_t requestedElementCount,
                                                    int64_t* elementCount,
                                                    void* arrayOfElements)
{
    if(elementCount == nullptr || arrayOfElements == nullptr)
    {
        return miopenStatusBadParm;
    }

    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(descriptor);
            theDescriptor.getAttribute(
                attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
        },
        false);
}

extern "C" miopenStatus_t miopenBackendExecute(miopenHandle_t handle,
                                               miopenBackendDescriptor_t executionPlan,
                                               miopenBackendDescriptor_t variantPack)
{
    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(executionPlan);
            theDescriptor.execute(handle, variantPack);
        },
        false);
}

extern "C" miopenStatus_t miopenBackendDestroyDescriptor(miopenBackendDescriptor_t descriptor)
{
    return miopen::try_([&] { miopen_destroy_object(descriptor); }, false);
}

template <typename BackendDescriptorType>
static void initializeBackendDescriptor(void* descriptor, std::size_t sizeInBytes)
{
    void* address = descriptor;
    if(std::align(
           alignof(BackendDescriptorType), sizeof(BackendDescriptorType), address, sizeInBytes) !=
           nullptr &&
       address == descriptor)
    {
        new(descriptor) BackendDescriptorType();
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

extern "C" miopenStatus_t miopenBackendInitialize(miopenBackendDescriptor_t descriptor,
                                                  miopenBackendDescriptorType_t descriptorType,
                                                  size_t sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(descriptorType, sizeInBytes);

    if(descriptor == nullptr)
    {
        return miopenStatusBadParm;
    }

    return miopen::try_([&] {
        switch(descriptorType)
        {
        /** This part is a common place of changes of about 25 PRs and merge conflicts arise heavily
         * here. Turn off clang-format to keep each line unique to simplify resolving of conflicts.
         *
         * \todo Turn on clang-format when active phase of development is finished.
         * --Sergei Apr, 2024
         */
        // clang-format off
        case MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendConvolutionDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_ENGINE_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendEngineDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendEngineCfgDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendEngineHeurDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendExecutionPlanDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_MATMUL_DESCRIPTOR:
	    initializeBackendDescriptor<miopen::graphapi::BackendMatmulDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationConvolutionForwardDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationConvolutionBackwardFilterDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationConvolutionBackwardDataDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationMatmulDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationPointwiseDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationReductionDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationReshapeDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationRngDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendOperationGraphDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_POINTWISE_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendPointwiseDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_REDUCTION_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendReductionDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_RNG_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendRngDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_TENSOR_DESCRIPTOR:
            initializeBackendDescriptor<miopen::graphapi::BackendTensorDescriptor>(descriptor, sizeInBytes); break;

        case MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR:
                initializeBackendDescriptor<miopen::graphapi::BackendVariantPackDescriptor>(descriptor, sizeInBytes); break;

        default: MIOPEN_THROW(miopenStatusUnsupportedOp);
            // clang-format on
        }
    });
}

namespace miopen {

namespace graphapi {

BackendDescriptor::~BackendDescriptor() {}

void BackendDescriptor::execute([[maybe_unused]] miopenHandle_t handle,
                                [[maybe_unused]] miopenBackendDescriptor_t variantPack)
{
    MIOPEN_THROW(miopenStatusBadParm);
}

OpNode* BackendDescriptor::getOperation() { return nullptr; }

} // namespace graphapi

} // namespace miopen
