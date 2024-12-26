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

#include <miopen/graphapi/matmul.hpp>
#include <miopen/graphapi/tensor.hpp>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::Matmul;
using miopen::graphapi::OperationMatmul;
using miopen::graphapi::OperationMatmulBuilder;
using miopen::graphapi::Tensor;

using miopen::graphapi::ValidatedValue;

using DescriptorTuple = std::tuple<bool,
                                   ValidatedValue<Matmul*>,
                                   ValidatedValue<Tensor*>,
                                   ValidatedValue<Tensor*>,
                                   ValidatedValue<Tensor*>,
                                   int64_t,
                                   ValidatedValue<Tensor*>,
                                   ValidatedValue<Tensor*>,
                                   ValidatedValue<Tensor*>>;

} // namespace

class CPU_GraphApiOperationMatmulBuilder_NONE : public testing::TestWithParam<DescriptorTuple>
{
protected:
    bool mAttrsValid;
    ValidatedValue<Matmul*> mMatmul;
    ValidatedValue<Tensor*> mA;
    ValidatedValue<Tensor*> mB;
    ValidatedValue<Tensor*> mC;
    int64_t mBatchCount;
    ValidatedValue<Tensor*> mGemmMOverride;
    ValidatedValue<Tensor*> mGemmNOverride;
    ValidatedValue<Tensor*> mGemmKOverride;

    void SetUp() override
    {
        std::tie(mAttrsValid,
                 mMatmul,
                 mA,
                 mB,
                 mC,
                 mBatchCount,
                 mGemmMOverride,
                 mGemmNOverride,
                 mGemmKOverride) = GetParam();
    }

    OperationMatmul build()
    {
        return OperationMatmulBuilder()
            .setA(mA.value)
            .setB(mB.value)
            .setC(mC.value)
            .setBatchCount(mBatchCount)
            .setGemmMOverride(mGemmMOverride.value)
            .setGemmNOverride(mGemmNOverride.value)
            .setGemmKOverride(mGemmKOverride.value)
            .setMatmulDescriptor(mMatmul.value)
            .build();
    }
};

TEST_P(CPU_GraphApiOperationMatmulBuilder_NONE, ValidateAttributes)
{
    if(mAttrsValid)
    {
        EXPECT_NO_THROW({ build(); }) << "Builder failed on valid attributes";
    }
    else
    {
        EXPECT_ANY_THROW({ build(); }) << "Builder failed to detect invalid attributes";
        if(!mMatmul.valid || !mA.valid || !mB.valid || !mC.valid || !mGemmMOverride.valid ||
           !mGemmNOverride.valid || !mGemmKOverride.valid)
        {
            EXPECT_ANY_THROW({ build(); }) << "Builder failed to detect invalid attributes";
        }
    }

    if(mMatmul.valid)
    {
        EXPECT_NO_THROW({ OperationMatmulBuilder().setMatmulDescriptor(mMatmul.value); })
            << "OperationMatmulBuilder::setMatmul failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationMatmulBuilder().setMatmulDescriptor(mMatmul.value); })
            << "OperationMatmulBuilder::setMatmul failed with an invalid attribute";
    }

    if(mA.valid)
    {
        EXPECT_NO_THROW({ OperationMatmulBuilder().setA(mA.value); })
            << "OperationMatmulBuilder::setA failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationMatmulBuilder().setA(mA.value); })
            << "OperationMatmulBuilder::setA failed with an invalid attribute";
    }

    if(mB.valid)
    {
        EXPECT_NO_THROW({ OperationMatmulBuilder().setB(mB.value); })
            << "OperationMatmulBuilder::setB failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationMatmulBuilder().setB(mB.value); })
            << "OperationMatmulBuilder::setB failed with an invalid attribute";
    }

    if(mC.valid)
    {
        EXPECT_NO_THROW({ OperationMatmulBuilder().setC(mC.value); })
            << "OperationMatmulBuilder::setC failed with a valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ OperationMatmulBuilder().setC(mC.value); })
            << "OperationMatmulBuilder::setC failed with an invalid attribute";
    }
}

TEST_P(CPU_GraphApiOperationMatmulBuilder_NONE, MissingSetter)
{
    EXPECT_ANY_THROW({
        OperationMatmulBuilder().setA(mA.value).setB(mB.value).setC(mC.value).build();
    }) << "Builder failed to detect missing setMatmulDescriptor() call";
    EXPECT_ANY_THROW({
        OperationMatmulBuilder()
            .setB(mB.value)
            .setC(mC.value)
            .setMatmulDescriptor(mMatmul.value)
            .build();
    }) << "Builder failed to detect missing setA() call";
    EXPECT_ANY_THROW({
        OperationMatmulBuilder()
            .setA(mA.value)
            .setC(mC.value)
            .setMatmulDescriptor(mMatmul.value)
            .build();
    }) << "Builder failed to detect missing setB() call";
    EXPECT_ANY_THROW({
        OperationMatmulBuilder()
            .setA(mA.value)
            .setB(mB.value)
            .setMatmulDescriptor(mMatmul.value)
            .build();
    }) << "Builder failed to detect missing setC() call";
}

namespace {

using miopen::graphapi::BackendMatmulDescriptor;
using miopen::graphapi::GMockBackendTensorDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

class GMockBackendMatmulDescriptor : public BackendMatmulDescriptor
{
public:
    GMockBackendMatmulDescriptor& operator=(const ValidatedValue<Matmul*>& testCaseMatmul)
    {
        if(!testCaseMatmul.valid)
        {
            return *this;
        }

        auto& theMatmul = *testCaseMatmul.value;

        auto compType = theMatmul.getComputeType();
        setAttribute(MIOPEN_ATTR_MATMUL_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &compType);

        finalize();

        return *this;
    }
};

} // namespace

class CPU_GraphApiOperationMatmul_NONE : public ::testing::TestWithParam<DescriptorTuple>
{
private:
    // Pointers to these are stored in the objects below
    GMockBackendMatmulDescriptor mMatmulDescriptor;
    GMockBackendTensorDescriptor aDesc;
    GMockBackendTensorDescriptor bDesc;
    GMockBackendTensorDescriptor cDesc;
    GMockBackendTensorDescriptor mOverrideDesc;
    GMockBackendTensorDescriptor nOverrideDesc;
    GMockBackendTensorDescriptor kOverrideDesc;

    // Pointers to these are stored in mExecute object below
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mMatmul;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mA;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mB;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mC;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> mOverrideAttr;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> nOverrideAttr;
    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> kOverrideAttr;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        auto [valid, mamtul, a, b, c, count, mOverride, nOverride, kOverride] = GetParam();

        try
        {
            mMatmulDescriptor = mamtul;
            aDesc             = a;
            bDesc             = b;
            cDesc             = c;
            mOverrideDesc     = mOverride;
            nOverrideDesc     = nOverride;
            kOverrideDesc     = kOverride;
        }
        catch(const std::exception& e)
        {
            FAIL() << e.what();
        }

        mMatmul = {mamtul.valid,
                   "MIOPEN_ATTR_OPERATION_MATMUL_DESC",
                   MIOPEN_ATTR_OPERATION_MATMUL_DESC,
                   MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                   MIOPEN_TYPE_CHAR,
                   2,
                   &mMatmulDescriptor};

        mA = {a.valid,
              "MIOPEN_ATTR_OPERATION_MATMUL_ADESC",
              MIOPEN_ATTR_OPERATION_MATMUL_ADESC,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              2,
              &aDesc};

        mB = {b.valid,
              "MIOPEN_ATTR_OPERATION_MATMUL_BDESC",
              MIOPEN_ATTR_OPERATION_MATMUL_BDESC,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              2,
              &bDesc};

        mC = {c.valid,
              "MIOPEN_ATTR_OPERATION_MATMUL_CDESC",
              MIOPEN_ATTR_OPERATION_MATMUL_CDESC,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              2,
              &cDesc};

        mOverrideAttr = {mOverride.valid,
                         "MIOPEN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC",
                         MIOPEN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC,
                         MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                         MIOPEN_TYPE_CHAR,
                         2,
                         &mOverrideDesc};

        nOverrideAttr = {nOverride.valid,
                         "MIOPEN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC",
                         MIOPEN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC,
                         MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                         MIOPEN_TYPE_CHAR,
                         2,
                         &nOverrideDesc};

        kOverrideAttr = {kOverride.valid,
                         "MIOPEN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC",
                         MIOPEN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC,
                         MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                         MIOPEN_TYPE_CHAR,
                         2,
                         &kOverrideDesc};

        mExecute.descriptor.attrsValid = valid;
        mExecute.descriptor.textName   = "MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR";
        mExecute.descriptor.type       = MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
        mExecute.descriptor.attributes = {
            &mMatmul, &mA, &mB, &mC, &mOverrideAttr, &nOverrideAttr, &kOverrideAttr};
    }
};

TEST_P(CPU_GraphApiOperationMatmul_NONE, CFunctions) { mExecute(); }

static Matmul matmul(miopenFloat);
static std::array<ValidatedValue<Matmul*>, 2> anyMatmuls{ValidatedValue<Matmul*>{true, &matmul},
                                                         ValidatedValue<Matmul*>{false, nullptr}};
static std::array<ValidatedValue<Matmul*>, 1> validMatmuls{ValidatedValue<Matmul*>{true, &matmul}};
static std::array<ValidatedValue<Matmul*>, 1> invalidMatmuls{
    ValidatedValue<Matmul*>{false, nullptr}};

static Tensor A(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 1, false);
static Tensor B(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 1, false);
static Tensor C(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 1, false);

static Tensor mOverride(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 1, false);
static Tensor nOverride(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 1, false);
static Tensor kOverride(miopenFloat, {10, 100, 100}, {100 * 100, 100, 1}, 1, false);

static Tensor D(miopenFloat, {10, 100}, {100, 1}, 1, false);
static Tensor E(miopenFloat, {2, 10, 100, 100}, {100 * 100 * 10, 100 * 100, 100, 1}, 1, false);

static Tensor aSmall(miopenFloat, {100}, {1}, 1, false);
static Tensor bSmall(miopenFloat, {100}, {1}, 1, false);
static Tensor cSmall(miopenFloat, {100}, {1}, 1, false);

static std::array<ValidatedValue<Tensor*>, 3> anyAtensors{ValidatedValue<Tensor*>{true, &A},
                                                          ValidatedValue<Tensor*>{false, nullptr},
                                                          ValidatedValue<Tensor*>{false, &aSmall}};
static std::array<ValidatedValue<Tensor*>, 1> validAtensors{ValidatedValue<Tensor*>{true, &A}};
static std::array<ValidatedValue<Tensor*>, 2> invalidAtensors{
    ValidatedValue<Tensor*>{false, nullptr}, ValidatedValue<Tensor*>{false, &aSmall}};

static std::array<ValidatedValue<Tensor*>, 3> anyBtensors{ValidatedValue<Tensor*>{true, &B},
                                                          ValidatedValue<Tensor*>{false, nullptr},
                                                          ValidatedValue<Tensor*>{false, &bSmall}};
static std::array<ValidatedValue<Tensor*>, 1> validBtensors{ValidatedValue<Tensor*>{true, &B}};
static std::array<ValidatedValue<Tensor*>, 2> invalidBtensors{
    ValidatedValue<Tensor*>{false, nullptr}, ValidatedValue<Tensor*>{false, &bSmall}};

static std::array<ValidatedValue<Tensor*>, 3> anyCtensors{ValidatedValue<Tensor*>{true, &C},
                                                          ValidatedValue<Tensor*>{false, nullptr},
                                                          ValidatedValue<Tensor*>{false, &cSmall}};
static std::array<ValidatedValue<Tensor*>, 1> validCtensors{ValidatedValue<Tensor*>{true, &C}};
static std::array<ValidatedValue<Tensor*>, 2> invalidCtensors{
    ValidatedValue<Tensor*>{false, nullptr}, ValidatedValue<Tensor*>{false, &cSmall}};

static std::array<ValidatedValue<Tensor*>, 1> validMOverridetensors{
    ValidatedValue<Tensor*>{true, &mOverride}};

static std::array<ValidatedValue<Tensor*>, 1> validNOverridetensors{
    ValidatedValue<Tensor*>{true, &nOverride}};

static std::array<ValidatedValue<Tensor*>, 1> validKOverridetensors{
    ValidatedValue<Tensor*>{true, &kOverride}};

static std::array<ValidatedValue<Tensor*>, 1> invalidBroadcastTensors{
    ValidatedValue<Tensor*>{true, &D}};

static std::array<ValidatedValue<Tensor*>, 1> validBroadcastTensors{
    ValidatedValue<Tensor*>{true, &E}};

static auto validAttributes = testing::Combine(testing::Values(true),
                                               testing::ValuesIn(validMatmuls),
                                               testing::ValuesIn(validAtensors),
                                               testing::ValuesIn(validBtensors),
                                               testing::ValuesIn(validCtensors),
                                               testing::Values(1),
                                               testing::ValuesIn(validMOverridetensors),
                                               testing::ValuesIn(validNOverridetensors),
                                               testing::ValuesIn(validKOverridetensors));

static auto invalidBroadcasts = testing::Combine(testing::Values(false),
                                                 testing::ValuesIn(validMatmuls),
                                                 testing::ValuesIn(invalidBroadcastTensors),
                                                 testing::ValuesIn(validBtensors),
                                                 testing::ValuesIn(validCtensors),
                                                 testing::Values(1),
                                                 testing::ValuesIn(validMOverridetensors),
                                                 testing::ValuesIn(validNOverridetensors),
                                                 testing::ValuesIn(validKOverridetensors));

static auto validBroadcasts = testing::Combine(testing::Values(true),
                                               testing::ValuesIn(validMatmuls),
                                               testing::ValuesIn(validBroadcastTensors),
                                               testing::ValuesIn(validBtensors),
                                               testing::ValuesIn(validCtensors),
                                               testing::Values(1),
                                               testing::ValuesIn(validMOverridetensors),
                                               testing::ValuesIn(validNOverridetensors),
                                               testing::ValuesIn(validKOverridetensors));

static auto invalidAtLeastMatmuls = testing::Combine(testing::Values(false),
                                                     testing::ValuesIn(invalidMatmuls),
                                                     testing::ValuesIn(anyAtensors),
                                                     testing::ValuesIn(anyBtensors),
                                                     testing::ValuesIn(anyCtensors),
                                                     testing::Values(1),
                                                     testing::ValuesIn(validMOverridetensors),
                                                     testing::ValuesIn(validNOverridetensors),
                                                     testing::ValuesIn(validKOverridetensors));

static auto invalidAtLeastAtensors = testing::Combine(testing::Values(false),
                                                      testing::ValuesIn(anyMatmuls),
                                                      testing::ValuesIn(invalidAtensors),
                                                      testing::ValuesIn(anyBtensors),
                                                      testing::ValuesIn(anyCtensors),
                                                      testing::Values(1),
                                                      testing::ValuesIn(validMOverridetensors),
                                                      testing::ValuesIn(validNOverridetensors),
                                                      testing::ValuesIn(validKOverridetensors));

static auto invalidAtLeastBtensors = testing::Combine(testing::Values(false),
                                                      testing::ValuesIn(anyMatmuls),
                                                      testing::ValuesIn(anyAtensors),
                                                      testing::ValuesIn(invalidBtensors),
                                                      testing::ValuesIn(validCtensors),
                                                      testing::Values(1),
                                                      testing::ValuesIn(validMOverridetensors),
                                                      testing::ValuesIn(validNOverridetensors),
                                                      testing::ValuesIn(validKOverridetensors));

static auto invalidAtLeastCtensors = testing::Combine(testing::Values(false),
                                                      testing::ValuesIn(anyMatmuls),
                                                      testing::ValuesIn(anyAtensors),
                                                      testing::ValuesIn(anyBtensors),
                                                      testing::ValuesIn(invalidCtensors),
                                                      testing::Values(1),
                                                      testing::ValuesIn(validMOverridetensors),
                                                      testing::ValuesIn(validNOverridetensors),
                                                      testing::ValuesIn(validKOverridetensors));

INSTANTIATE_TEST_SUITE_P(UnitVA, CPU_GraphApiOperationMatmulBuilder_NONE, validAttributes);

INSTANTIATE_TEST_SUITE_P(UnitIBr, CPU_GraphApiOperationMatmulBuilder_NONE, invalidBroadcasts);

INSTANTIATE_TEST_SUITE_P(UnitIM, CPU_GraphApiOperationMatmulBuilder_NONE, invalidAtLeastMatmuls);
INSTANTIATE_TEST_SUITE_P(UnitIA, CPU_GraphApiOperationMatmulBuilder_NONE, invalidAtLeastAtensors);
INSTANTIATE_TEST_SUITE_P(UnitIB, CPU_GraphApiOperationMatmulBuilder_NONE, invalidAtLeastBtensors);
INSTANTIATE_TEST_SUITE_P(UnitIC, CPU_GraphApiOperationMatmulBuilder_NONE, invalidAtLeastCtensors);

INSTANTIATE_TEST_SUITE_P(UnitVA, CPU_GraphApiOperationMatmul_NONE, validAttributes);
INSTANTIATE_TEST_SUITE_P(UnitIM, CPU_GraphApiOperationMatmul_NONE, invalidAtLeastMatmuls);
INSTANTIATE_TEST_SUITE_P(UnitIA, CPU_GraphApiOperationMatmul_NONE, invalidAtLeastAtensors);
INSTANTIATE_TEST_SUITE_P(UnitIB, CPU_GraphApiOperationMatmul_NONE, invalidAtLeastBtensors);
