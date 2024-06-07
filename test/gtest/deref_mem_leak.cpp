#include <gtest/gtest.h>
#include <miopen/tensor.hpp>
#include <miopen/miopen.h>

TEST(DerefMemoryLeak, miopenCreateTensorDescriptor)
{
    miopen::TensorDescriptor::leakedInstances = 0;

    auto status = miopenCreateTensorDescriptor(nullptr);
    EXPECT_NE(status, miopenStatusSuccess)
        << "Descriptor creation returned success on a null pointer";

    EXPECT_EQ(miopen::TensorDescriptor::leakedInstances, 0)
        << "Memory management error on a null pointer";
}
