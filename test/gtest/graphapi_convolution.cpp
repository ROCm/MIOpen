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
#include <miopen/algorithm.hpp>
#include <miopen/miopen.h>
#include <miopen/graphapi/convolution.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using GraphApiConvolutionTuple = std::tuple<bool,
                                            miopenDataType_t,
                                            miopenConvolutionMode_t,
                                            int64_t,
                                            std::vector<int64_t>,
                                            std::vector<int64_t>,
                                            std::vector<int64_t>,
                                            std::vector<int64_t>>;

class CPU_GraphApiConvolution_NONE : public testing::TestWithParam<GraphApiConvolutionTuple>
{
protected:
    void SetUp() override
    {
        std::tie(attrsValid,
                 compType,
                 mode,
                 spatialDims,
                 dilations,
                 filterStrides,
                 prePaddings,
                 postPaddings) = GetParam();
    }
    miopenDataType_t compType;
    miopenConvolutionMode_t mode;
    int64_t spatialDims;
    std::vector<int64_t> dilations;
    std::vector<int64_t> filterStrides;
    std::vector<int64_t> prePaddings;
    std::vector<int64_t> postPaddings;
    bool attrsValid;
};

TEST_P(CPU_GraphApiConvolution_NONE, BuilderValidateAttributes)
{
    bool thrown = false;
    try
    {
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }
    catch(...)
    {
        thrown = true;
    }
    EXPECT_NE(thrown, attrsValid) << "R-value builder failure";

    thrown = false;
    try
    {
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }
    catch(...)
    {
        thrown = true;
    }
    EXPECT_NE(thrown, attrsValid) << "L-value builder failure";
}

TEST_P(CPU_GraphApiConvolution_NONE, RVBuilderMissingSetter)
{
    EXPECT_ANY_THROW({
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setCompType() call";

    EXPECT_ANY_THROW({
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setCompType(compType)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setMode() call";

    EXPECT_ANY_THROW({
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setCompType(compType)
                        .setMode(mode)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setSpatialDims() call";

    EXPECT_ANY_THROW({
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setDilations() call";

    EXPECT_ANY_THROW({
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setFilterStrides() call";

    EXPECT_ANY_THROW({
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setPrePaddings() call";

    EXPECT_ANY_THROW({
        auto conv = miopen::graphapi::ConvolutionBuilder()
                        .setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setPostPaddings() call";
}

TEST_P(CPU_GraphApiConvolution_NONE, LVBuilderMissingSetter)
{
    EXPECT_ANY_THROW({
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setCompType() call";

    EXPECT_ANY_THROW({
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setCompType(compType)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setMode() call";

    EXPECT_ANY_THROW({
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setCompType(compType)
                        .setMode(mode)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setSpatialDims() call";

    EXPECT_ANY_THROW({
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setDilations() call";

    EXPECT_ANY_THROW({
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setPrePaddings(prePaddings)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setFilterStrides() call";

    EXPECT_ANY_THROW({
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPostPaddings(postPaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setPrePaddings() call";

    EXPECT_ANY_THROW({
        miopen::graphapi::ConvolutionBuilder builder;
        auto conv = builder.setCompType(compType)
                        .setMode(mode)
                        .setSpatialDims(spatialDims)
                        .setDilations(dilations)
                        .setFilterStrides(filterStrides)
                        .setPrePaddings(prePaddings)
                        .build();
    }) << "Builder validated attributes despite missing "
          "graphapi::ConvolutionBuilder::setPostPaddings() call";
}

TEST_P(CPU_GraphApiConvolution_NONE, BuilderCopyValues)
{
    auto srcDilations     = dilations;
    auto srcFilterStrides = filterStrides;
    auto srcPrePaddings   = prePaddings;
    auto srcPostPaddings  = postPaddings;

    auto srcDilationsAddress     = srcDilations.data();
    auto srcFilterStridesAddress = srcFilterStrides.data();
    auto srcPrePaddingsAddress   = srcPrePaddings.data();
    auto srcPostPaddingsAddress  = srcPostPaddings.data();

    bool thrown = false;
    miopen::graphapi::Convolution conv;
    try
    {
        miopen::graphapi::ConvolutionBuilder builder;
        conv = builder.setCompType(compType)
                   .setMode(mode)
                   .setSpatialDims(spatialDims)
                   .setDilations(srcDilations)
                   .setFilterStrides(srcFilterStrides)
                   .setPrePaddings(srcPrePaddings)
                   .setPostPaddings(srcPostPaddings)
                   .build();
    }
    catch(...)
    {
        thrown = true;
    }
    EXPECT_NE(thrown, attrsValid) << "graphapi::ConvolutionBuilder failure";

    if(!attrsValid)
        return;

    EXPECT_EQ(conv.getCompType(), compType)
        << "graphapi::ConvolutionBuilder::setCompType didn't set parameter correctly";
    EXPECT_EQ(conv.getMode(), mode)
        << "graphapi::ConvolutionBuilder::setMode didn't set parameter correctly";
    EXPECT_EQ(conv.getSpatialDims(), spatialDims)
        << "graphapi::ConvolutionBuilder::setSpatialDims didn't set parameter correctly";
    EXPECT_THAT(conv.getDilations(), testing::ContainerEq(dilations))
        << "graphapi::ConvolutionBuilder::setDilations didn't set parameter correctly";
    EXPECT_THAT(conv.getFilterStrides(), testing::ContainerEq(filterStrides))
        << "graphapi::ConvolutionBuilder::setFilterStrides didn't set parameter correctly";
    EXPECT_THAT(conv.getPrePaddings(), testing::ContainerEq(prePaddings))
        << "graphapi::ConvolutionBuilder::setPrePaddings didn't set parameter correctly";
    EXPECT_THAT(conv.getPostPaddings(), testing::ContainerEq(postPaddings))
        << "graphapi::ConvolutionBuilder::setPostPaddings didn't set parameter correctly";

    EXPECT_NE(conv.getDilations().data(), srcDilationsAddress)
        << "graphapi::ConvolutionBuilder::setDilations unexpectedly moved the parameter";
    EXPECT_NE(conv.getFilterStrides().data(), srcFilterStridesAddress)
        << "graphapi::ConvolutionBuilder::setFilterStrides unexpectedly moved the parameter";
    EXPECT_NE(conv.getPrePaddings().data(), srcPrePaddingsAddress)
        << "graphapi::ConvolutionBuilder::setPrePaddings unexpectedly moved the parameter";
    EXPECT_NE(conv.getPostPaddings().data(), srcPostPaddingsAddress)
        << "graphapi::ConvolutionBuilder::setPostPaddings unexpectedly moved the parameter";
}

TEST_P(CPU_GraphApiConvolution_NONE, BuilderMoveValues)
{
    auto srcDilations     = dilations;
    auto srcFilterStrides = filterStrides;
    auto srcPrePaddings   = prePaddings;
    auto srcPostPaddings  = postPaddings;

    auto srcDilationsAddress     = srcDilations.data();
    auto srcFilterStridesAddress = srcFilterStrides.data();
    auto srcPrePaddingsAddress   = srcPrePaddings.data();
    auto srcPostPaddingsAddress  = srcPostPaddings.data();

    bool thrown = false;
    miopen::graphapi::Convolution conv;
    try
    {
        miopen::graphapi::ConvolutionBuilder builder;
        builder.setCompType(compType);
        builder.setMode(mode);
        builder.setSpatialDims(spatialDims);
        builder.setDilations(std::move(srcDilations));
        builder.setFilterStrides(std::move(srcFilterStrides));
        builder.setPrePaddings(std::move(srcPrePaddings));
        builder.setPostPaddings(std::move(srcPostPaddings));
        conv = std::move(builder).build();
    }
    catch(...)
    {
        thrown = true;
    }
    EXPECT_NE(thrown, attrsValid) << "graphapi::ConvolutionBuilder failure";

    if(!attrsValid)
        return;

    EXPECT_EQ(conv.getCompType(), compType)
        << "graphapi::ConvolutionBuilder::setCompType didn't set the parameter correctly";
    EXPECT_EQ(conv.getMode(), mode)
        << "graphapi::ConvolutionBuilder::setMode didn't set the parameter correctly";
    EXPECT_EQ(conv.getSpatialDims(), spatialDims)
        << "graphapi::ConvolutionBuilder::setSpatialDims didn't set the parameter correctly";
    EXPECT_THAT(conv.getDilations(), testing::ContainerEq(dilations))
        << "graphapi::ConvolutionBuilder::setDilations didn't set the parameter correctly";
    EXPECT_THAT(conv.getFilterStrides(), testing::ContainerEq(filterStrides))
        << "graphapi::ConvolutionBuilder::setFilterStrides didn't set parameter correctly";
    EXPECT_THAT(conv.getPrePaddings(), testing::ContainerEq(prePaddings))
        << "graphapi::ConvolutionBuilder::setPrePaddings didn't set parameter correctly";
    EXPECT_THAT(conv.getPostPaddings(), testing::ContainerEq(postPaddings))
        << "graphapi::ConvolutionBuilder::setPostPaddings didn't set parameter correctly";

    EXPECT_EQ(conv.getDilations().data(), srcDilationsAddress)
        << "graphapi::ConvolutionBuilder::setDilations didn't move the parameter";
    EXPECT_EQ(conv.getFilterStrides().data(), srcFilterStridesAddress)
        << "graphapi::ConvolutionBuilder::setFilterStrides didn't move the parameter";
    EXPECT_EQ(conv.getPrePaddings().data(), srcPrePaddingsAddress)
        << "graphapi::ConvolutionBuilder::setPrePaddings didn't move the parameter";
    EXPECT_EQ(conv.getPostPaddings().data(), srcPostPaddingsAddress)
        << "graphapi::ConvolutionBuilder::setPostPaddings didn't move the parameter";
}

TEST_P(CPU_GraphApiConvolution_NONE, CFunctions)
{
    // clang-format off
    // Create Desctiptor
    miopenBackendDescriptor_t descrConv;
    miopenStatus_t status = miopenBackendCreateDescriptor(MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR, &descrConv);
    ASSERT_EQ(status, miopenStatusSuccess) << "MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR wasn't created";
    ASSERT_NE(descrConv, nullptr) << "A null MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR was created";

    // Finalize before setting attributes
    status = miopenBackendFinalize(descrConv);
    if(status == miopenStatusSuccess)
    {
        miopenBackendDestroyDescriptor(descrConv);
        FAIL() << "MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR was finalized without setting attributes";
    }

    // Set compType
    bool allParamsSet = true;
    char buffer[64] = {0};
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_BOOLEAN, 1, &compType);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was set with invalid type";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 2, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was set with invalid element count";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was set with null array of elements";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &compType);
    if(attrsValid) // implementation may postpone validating values to finalize()
        EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE wasn't set";
    allParamsSet = allParamsSet && (status == miopenStatusSuccess);

    // Set mode
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_BOOLEAN, 1, &mode);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was set with invalid type";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 2, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was set with invalid element count";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 1, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was set with null array of elements";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 1, &mode);
    if(attrsValid) // implementation may postpone validating values to finalize()
        EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE wasn't set";
    allParamsSet = allParamsSet && (status == miopenStatusSuccess);

    // Set spatialDims
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_BOOLEAN, 1, &spatialDims);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was set with invalid type";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 2, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was set with invalid element count";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 1, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was set with null array of elements";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 1, &spatialDims);
    if(attrsValid) // implementation may postpone validating values to finalize()
        EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS wasn't set";
    allParamsSet = allParamsSet && (status == miopenStatusSuccess);

    // Set dilations
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_BOOLEAN, dilations.size(), dilations.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS was set with invalid type";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, 0, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS was set with invalid element count";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, dilations.size(), nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was set with null array of elements";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, dilations.size(), dilations.data());
    if(attrsValid) // implementation may postpone validating values to finalize()
        EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS wasn't set";
    allParamsSet = allParamsSet && (status == miopenStatusSuccess);

    // Set filterStrides
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_BOOLEAN, filterStrides.size(), filterStrides.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was set with invalid type";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, 0, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was set with invalid element count";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, filterStrides.size(), nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was set with null array of elements";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, filterStrides.size(), filterStrides.data());
    if(attrsValid) // implementation may postpone validating values to finalize()
        EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES wasn't set";
    allParamsSet = allParamsSet && (status == miopenStatusSuccess);

    // Set prePaddings
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_BOOLEAN, prePaddings.size(), prePaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was set with invalid type";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, 0, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was set with invalid element count";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, prePaddings.size(), nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was set with null array of elements";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, prePaddings.size(), prePaddings.data());
    if(attrsValid) // implementation may postpone validating values to finalize()
        EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS wasn't set";
    allParamsSet = allParamsSet && (status == miopenStatusSuccess);

    // Set postPaddings
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_BOOLEAN, postPaddings.size(), postPaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was set with invalid type";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, 0, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was set with invalid element count";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, postPaddings.size(), nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was set with null array of elements";
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, postPaddings.size(), postPaddings.data());
    if(attrsValid) // implementation may postpone validating values to finalize()
        EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS wasn't set";
    allParamsSet = allParamsSet && (status == miopenStatusSuccess);

    // Get attibute before finalizing
    miopenDataType_t gotCompType = miopenHalf;
    int64_t elementCount = 0;
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &elementCount, &gotCompType);
    if(status == miopenStatusSuccess)
    {
        miopenBackendDestroyDescriptor(descrConv);
        FAIL() << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was retrieved before finalize()";
    }

    if(!allParamsSet && attrsValid)
    {
        miopenBackendDestroyDescriptor(descrConv);
        FAIL() << "Not all attributes of MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR were set";
    }

    // Finalize
    status = miopenBackendFinalize(descrConv);
    if(attrsValid && status != miopenStatusSuccess)
    {
        miopenBackendDestroyDescriptor(descrConv);
        FAIL() << "MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR wasn't finalized";
    }
    else if(!attrsValid)
    {
        miopenBackendDestroyDescriptor(descrConv);
        ASSERT_NE(status, miopenStatusSuccess) << "MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR was finalized on invalid attributes";
        return; // no need to continue with non-finalized descriptor
    }

    // Set Attributes after finalizing
    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &compType);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was set after finalize()";

    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 1, &mode);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was set after finalize()";

    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 1, &spatialDims);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was set after finalize()";

    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, dilations.size(), dilations.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS was set after finalize()";

    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, filterStrides.size(), filterStrides.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was set after finalize()";

    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, prePaddings.size(), prePaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was set after finalize()";

    status = miopenBackendSetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, postPaddings.size(), postPaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was set after finalize()";

    // Get compType
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_BOOLEAN, 1, &elementCount, &gotCompType);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was retrieved with invalid type";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 2, &elementCount, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was retrieved with invalid element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, nullptr, &gotCompType);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was retrieved with null element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &elementCount, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE was retrieved with null array of elements";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &elementCount, &gotCompType);
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE wasn't retrieved";
    if(status == miopenStatusSuccess)
        EXPECT_EQ(gotCompType, compType) << "MIOPEN_ATTR_CONVOLUTION_COMP_TYPE set and retrieved values differ";

    // Get mode
    miopenConvolutionMode_t gotMode;
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_BOOLEAN, 1, &elementCount, &gotMode);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was retrieved with invalid type";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 2, &elementCount, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was retrieved with invalid element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 1, nullptr, &gotMode);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was retrieved with null element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 1, &elementCount, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE was retrieved with null array of elements";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 1, &elementCount, &gotMode);
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE wasn't retrieved";
    if(status == miopenStatusSuccess)
        EXPECT_EQ(gotMode, mode) << "MIOPEN_ATTR_CONVOLUTION_CONV_MODE set and retrieved values differ";

    // Get spatialDims
    int64_t gotSpatialDims = 0;
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_BOOLEAN, 1, &elementCount, &gotSpatialDims);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was retrieved with invalid type";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 2, &elementCount, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was retrieved with invalid element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 1, nullptr, &gotSpatialDims);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was retrieved with null element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 1, &elementCount, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS was retrieved with null array of elements";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 1, &elementCount, &gotSpatialDims);
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS wasn't retrieved";
    if(status == miopenStatusSuccess)
        EXPECT_EQ(gotSpatialDims, spatialDims) << "MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS set and retrieved values differ";

    // Get dilations
    std::vector<int64_t> gotDilations(dilations.size());
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_BOOLEAN, gotDilations.size(), &elementCount, gotDilations.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS was retrieved with invalid type";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, 0, &elementCount, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS was retrieved with invalid element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, gotDilations.size(), nullptr, gotDilations.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS was retrieved with null element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, gotDilations.size(), &elementCount, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS was retrieved with null array of elements";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, gotDilations.size(), &elementCount, gotDilations.data());
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS wasn't retrieved";
    if(status == miopenStatusSuccess)
        EXPECT_THAT(gotDilations, testing::ContainerEq(dilations)) << "MIOPEN_ATTR_CONVOLUTION_DILATIONS set and retrieved values differ";

    // Get filterStrides
    std::vector<int64_t> gotFilterStrides(filterStrides.size());
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_BOOLEAN, gotFilterStrides.size(), &elementCount, gotFilterStrides.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was retrieved with invalid type";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, 0, &elementCount, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was retrieved with invalid element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, gotFilterStrides.size(), nullptr, gotFilterStrides.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was retrieved with null element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, gotFilterStrides.size(), &elementCount, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES was retrieved with null array of elements";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, gotFilterStrides.size(), &elementCount, gotFilterStrides.data());
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES wasn't retrieved";
    if(status == miopenStatusSuccess)
        EXPECT_THAT(gotFilterStrides, testing::ContainerEq(filterStrides)) << "MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES set and retrieved values differ";

    // Get prePaddings
    std::vector<int64_t> gotPrePaddings(prePaddings.size());
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_BOOLEAN, gotPrePaddings.size(), &elementCount, gotPrePaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was retrieved with invalid type";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, 0, &elementCount, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was retrieved with invalid element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, gotPrePaddings.size(), nullptr, gotPrePaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was retrieved with null element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, gotPrePaddings.size(), &elementCount, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS was retrieved with null array of elements";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, gotPrePaddings.size(), &elementCount, gotPrePaddings.data());
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS wasn't retrieved";
    if(status == miopenStatusSuccess)
        EXPECT_THAT(gotPrePaddings, testing::ContainerEq(prePaddings)) << "MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS set and retrieved values differ";

    // Get postPaddings
    std::vector<int64_t> gotPostPaddings(postPaddings.size());
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_BOOLEAN, gotPostPaddings.size(), &elementCount, gotPostPaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was retrieved with invalid type";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, 0, &elementCount, buffer);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was retrieved with invalid element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, gotPostPaddings.size(), nullptr, gotPostPaddings.data());
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was retrieved with null element count";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, gotPostPaddings.size(), &elementCount, nullptr);
    EXPECT_NE(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS was retrieved with null array of elements";
    status = miopenBackendGetAttribute(descrConv, MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, gotPostPaddings.size(), &elementCount, gotPostPaddings.data());
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS wasn't retrieved";
    if(status == miopenStatusSuccess)
        EXPECT_THAT(gotPostPaddings, testing::ContainerEq(postPaddings)) << "MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS set and retrieved values differ";

    // Destroy description
    status = miopenBackendDestroyDescriptor(descrConv);
    EXPECT_EQ(status, miopenStatusSuccess) << "MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR destroyed with non-success status";
    // clang-format on
}

INSTANTIATE_TEST_SUITE_P(
    Unit,
    CPU_GraphApiConvolution_NONE,
    testing::Values(
        GraphApiConvolutionTuple{
            true, miopenInt8, miopenConvolution, 2, {5, 6}, {20, 21}, {3, 4}, {1, 2}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 3, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 1, 1}, {1, 1}, {0, 0}, {0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 1}, {1, 1, 1}, {0, 0}, {0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 1}, {1, 1}, {0, 0, 0}, {0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 1}, {1, 1}, {0, 0}, {0, 0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 0}, {1, 1}, {0, 0}, {0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 1}, {1, 0}, {0, 0}, {0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 1}, {1, 1}, {-1, 0}, {0, 0}},
        GraphApiConvolutionTuple{
            false, miopenInt8, miopenConvolution, 2, {1, 1}, {1, 1}, {0, 0}, {0, -1}}));

template <typename T>
class CPU_GraphApiOperationConvolutionBuilder_NONE : public testing::Test
{
protected:
    using TestCase = std::tuple<bool,
                                miopen::graphapi::Convolution*,
                                miopen::graphapi::Tensor*,
                                miopen::graphapi::Tensor*,
                                miopen::graphapi::Tensor*,
                                const char*>;

    void SetUp() override
    {
        testCases = {TestCase{true, &this->convolution, &this->x, &this->y, &this->w, ""},
                     TestCase{false, nullptr, &this->x, &this->y, &this->w, "convolution"},
                     TestCase{false, &this->convolution, nullptr, &this->y, &this->w, "X tensor"},
                     TestCase{false, &this->convolution, &this->x, nullptr, &this->w, "Y tensor"},
                     TestCase{false, &this->convolution, &this->x, &this->y, nullptr, "W tensor"}};
    }

    miopen::graphapi::Convolution convolution;
    miopen::graphapi::Tensor x;
    miopen::graphapi::Tensor y;
    miopen::graphapi::Tensor w;
    double dAlpha = 1.0;
    double dBeta  = 0.0;
    float fAlpha  = 1.0f;
    float fBeta   = 0.0f;

    std::array<TestCase, 5> testCases;
};

using GraphApiOperationConvolutionBuilderClasses =
    testing::Types<miopen::graphapi::OperationConvolutionForwardBuilder,
                   miopen::graphapi::OperationConvolutionBackwardDataBuilder,
                   miopen::graphapi::OperationConvolutionBackwardFilterBuilder>;

TYPED_TEST_SUITE(CPU_GraphApiOperationConvolutionBuilder_NONE,
                 GraphApiOperationConvolutionBuilderClasses);

TYPED_TEST(CPU_GraphApiOperationConvolutionBuilder_NONE, ValidateAttributes)
{
    for(auto [attrsValid, convolution, x, y, w, message] : this->testCases)
    {
        if(attrsValid)
        {
            EXPECT_NO_THROW({
                auto op = TypeParam()
                              .setConvolution(convolution)
                              .setX(x)
                              .setY(y)
                              .setW(w)
                              .setAlpha(this->dAlpha)
                              .setBeta(this->dBeta)
                              .build();
            }) << "Builder didn't validate correct attributes";
        }
        else
        {
            EXPECT_ANY_THROW({
                auto op = TypeParam()
                              .setConvolution(convolution)
                              .setX(x)
                              .setY(y)
                              .setW(w)
                              .setAlpha(this->dAlpha)
                              .setBeta(this->dBeta)
                              .build();
            }) << "Builder validated incorrect "
               << message;
        }
    }
}

TYPED_TEST(CPU_GraphApiOperationConvolutionBuilder_NONE, MissingSetter)
{
    for(auto [attrsValid, convolution, x, y, w, message] : this->testCases)
    {
        EXPECT_ANY_THROW({
            auto op = TypeParam()
                          .setX(&this->x)
                          .setY(&this->y)
                          .setW(&this->w)
                          .setAlpha(this->dAlpha)
                          .setBeta(this->dBeta)
                          .build();
        }) << "Builder validated attributes despite missing"
              "setConvolution() call";

        EXPECT_ANY_THROW({
            auto op = TypeParam()
                          .setConvolution(convolution)
                          .setY(y)
                          .setW(w)
                          .setAlpha(this->dAlpha)
                          .setBeta(this->dBeta)
                          .build();
        }) << "Builder validated attributes despite missing"
              "setX() call";

        EXPECT_ANY_THROW({
            auto op = TypeParam()
                          .setConvolution(convolution)
                          .setX(x)
                          .setW(w)
                          .setAlpha(this->dAlpha)
                          .setBeta(this->dBeta)
                          .build();
        }) << "Builder validated attributes despite missing"
              "setY() call";

        EXPECT_ANY_THROW({
            auto op = TypeParam()
                          .setConvolution(convolution)
                          .setX(x)
                          .setY(y)
                          .setAlpha(this->dAlpha)
                          .setBeta(this->dBeta)
                          .build();
        }) << "Builder validated attributes despite missing"
              "setW() call";

        EXPECT_ANY_THROW({
            auto op = TypeParam()
                          .setConvolution(convolution)
                          .setX(x)
                          .setY(y)
                          .setW(w)
                          .setBeta(this->dBeta)
                          .build();
        }) << "Builder validated attributes despite missing"
              "setAlpha() call";

        EXPECT_ANY_THROW({
            auto op = TypeParam()
                          .setConvolution(convolution)
                          .setX(x)
                          .setY(y)
                          .setW(w)
                          .setAlpha(this->dAlpha)
                          .build();
        }) << "Builder validated attributes despite missing"
              "setBeta() call";
    }
}

namespace {

struct GTestDescAttr
{
    struct TestCase
    {
        const char* textName;
        miopenBackendAttributeName_t name;
        miopenBackendAttributeType_t type;
        int64_t count;
        void* data;

        miopenBackendAttributeType_t invalidType;
        void* invalidTypeData;

        int64_t invalidCount;
        void* invalidCountData;

        void* readBuffer;
    };

    GTestDescAttr() = default;

    TestCase testCase() const { return mTestCase; }
    virtual bool testReadBuffer() = 0;

    virtual ~GTestDescAttr() = default;

protected:
    TestCase mTestCase;
};

template <typename ValueType, typename InvalidType>
struct GTestDescAttrValues : GTestDescAttr
{
    GTestDescAttrValues(const char* textName,
                        miopenBackendAttributeName_t name,
                        miopenBackendAttributeType_t type,
                        miopenBackendAttributeType_t invalidType,
                        int64_t invalidCount,
                        std::initializer_list<ValueType> values)
        : mValues(values),
          mInvalidTypeValues(std::max(static_cast<decltype(values.size())>(1), values.size())),
          mInvalidCountValues(std::max(static_cast<decltype(invalidCount)>(1), invalidCount),
                              values.size() > 0 ? *values.begin() : ValueType{}),
          mReadValues(values.size())
    {
        mTestCase.textName = textName;
        mTestCase.name     = name;
        mTestCase.type     = type;
        mTestCase.count    = mValues.size();
        mTestCase.data     = mValues.data();

        mTestCase.invalidType     = invalidType;
        mTestCase.invalidTypeData = mInvalidTypeValues.data();

        mTestCase.invalidCount     = invalidCount;
        mTestCase.invalidCountData = mInvalidCountValues.data();

        mTestCase.readBuffer = mReadValues.data();
    }
    virtual bool testReadBuffer() override
    {
        return std::equal(mValues.begin(), mValues.end(), mReadValues.begin());
    }

private:
    std::vector<ValueType> mValues;
    std::vector<InvalidType> mInvalidTypeValues;
    std::vector<ValueType> mInvalidCountValues;
    std::vector<ValueType> mReadValues;
};

struct GTestDescriptor
{
    const char* textName;
    miopenBackendDescriptorType_t type;
    bool attrsValid;
    std::vector<std::shared_ptr<GTestDescAttr>> attributes;
};

using GTestAttrAlphaDouble = GTestDescAttrValues<double, char>;
using GTestAttrAlphaFloat  = GTestDescAttrValues<float, char>;

struct GTestAttrConv : GTestDescAttrValues<miopenBackendDescriptor_t, char>
{
    GTestAttrConv(const char* textName,
                  miopenBackendAttributeName_t name,
                  bool finalized = true,
                  bool nullValue = false)
        : GTestDescAttrValues<miopenBackendDescriptor_t, char>(
              textName,
              name,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              0,
              std::initializer_list<miopenBackendDescriptor_t>{nullValue ? nullptr : &mConv}),
          mConv(finalized)
    {
    }

private:
    struct Descr : miopen::graphapi::BackendConvolutionDescriptor
    {
        Descr(bool finalized) { mFinalized = finalized; }
    };
    Descr mConv;
};

struct GTestAttrTensor : GTestDescAttrValues<miopenBackendDescriptor_t, char>
{
    GTestAttrTensor(const char* textName,
                    miopenBackendAttributeName_t name,
                    bool finalized = true,
                    bool nullValue = false)
        : GTestDescAttrValues<miopenBackendDescriptor_t, char>(
              textName,
              name,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              0,
              std::initializer_list<miopenBackendDescriptor_t>{nullValue ? nullptr : &mTens}),
          mTens(finalized)
    {
    }

private:
    struct Descr : miopen::graphapi::BackendTensorDescriptor
    {
        Descr(bool finalized) { mFinalized = finalized; }
    };
    Descr mTens;
};

} // namespace

class CPU_GraphApiOperationConvolution_NONE : public testing::TestWithParam<GTestDescriptor>
{
};

TEST_P(CPU_GraphApiOperationConvolution_NONE, CFuntions)
{
    auto [descrTextName, descrType, attrsValid, attributes] = GetParam();

    // Create Desctiptor
    miopenBackendDescriptor_t descr;
    // clang-format off
    miopenStatus_t status = miopenBackendCreateDescriptor(descrType, &descr);
    ASSERT_EQ(status, miopenStatusSuccess) << descrTextName << " wasn't created";
    ASSERT_NE(descr, nullptr) << "A null " << descrTextName << " was created";
    // clang-format on

    // Finalize before setting attributes
    status = miopenBackendFinalize(descr);
    if(status == miopenStatusSuccess)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << descrTextName << " was finalized without setting attributes";
    }

    // Set attributes (should succeed)
    bool anyAttributeFailed = false;
    for(auto& attrPtr : attributes)
    {
        auto [textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->testCase();

        // clang-format off
        status = miopenBackendSetAttribute(descr, name, invalidType, count, invalidTypeData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set with invalid type";

        status = miopenBackendSetAttribute(descr, name, type, invalidCount, invalidCountData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set with invalid element count";

        status = miopenBackendSetAttribute(descr, name, type, count, nullptr);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set with null array of elements";

        status = miopenBackendSetAttribute(descr, name, type, count, data);
        if(attrsValid) // implementation may postpone validating values to finalize()
            EXPECT_EQ(status, miopenStatusSuccess) << textName << " wasn't set";
        // clang-format on

        anyAttributeFailed = anyAttributeFailed || (status != miopenStatusSuccess);
    }

    // Get attibute before finalizing (not a one should succeed)
    bool anyAttributeGot = false;
    for(auto& attrPtr : attributes)
    {
        auto [textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->testCase();

        int64_t elementCount = 0;

        status = miopenBackendGetAttribute(descr, name, type, count, &elementCount, readBuffer);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved before finalize()";

        anyAttributeGot = anyAttributeGot || (status == miopenStatusSuccess);
    }

    // Stop further execution if needed
    if(anyAttributeGot)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << "Some attributes of " << descrTextName << " were retrieved before finalize()";
    }
    if(anyAttributeFailed && attrsValid)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << "Not all attributes of " << descrTextName << " were set";
    }

    // Finalize
    status = miopenBackendFinalize(descr);

    // Stop further execution if finalize() acted incorrectly
    if(attrsValid && status != miopenStatusSuccess)
    {
        miopenBackendDestroyDescriptor(descr);
        FAIL() << descrTextName << " wasn't finalized";
    }
    else if(!attrsValid)
    {
        miopenBackendDestroyDescriptor(descr);
        ASSERT_NE(status, miopenStatusSuccess)
            << descrTextName << " was finalized on invalid attributes";

        // No need to proceed with invalid attributes
        return;
    }

    // Set attributes after finalizing (not a one should succeed)
    bool anyAttributeSet = false;
    for(auto& attrPtr : attributes)
    {
        auto [textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->testCase();

        status = miopenBackendSetAttribute(descr, name, type, count, data);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was set after finalize()";

        anyAttributeSet = anyAttributeSet || (status == miopenStatusSuccess);
    }

    // Stop if an attribute was set
    if(anyAttributeSet)
    {
        miopenBackendDestroyDescriptor(descr);
        ASSERT_NE(status, miopenStatusSuccess)
            << "An attribute of " << descrTextName << " was set after finalize()";
    }

    // Get attributes
    for(auto& attrPtr : attributes)
    {
        auto [textName,
              name,
              type,
              count,
              data,
              invalidType,
              invalidTypeData,
              invalidCount,
              invalidCountData,
              readBuffer] = attrPtr->testCase();

        int64_t elementCount = 0;
        // clang-format off
        status = miopenBackendGetAttribute(descr, name, invalidType, count, &elementCount, invalidTypeData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with invalid type";
        status = miopenBackendGetAttribute(descr, name, type, invalidCount, &elementCount, invalidCountData);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with invalid element count";
        status = miopenBackendGetAttribute(descr, name, type, count, nullptr, readBuffer);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with null element count";
        status = miopenBackendGetAttribute(descr, name, type, count, &elementCount, nullptr);
        EXPECT_NE(status, miopenStatusSuccess) << textName << " was retrieved with null array of elements";
        status = miopenBackendGetAttribute(descr, name, type, count, &elementCount, readBuffer);
        EXPECT_EQ(status, miopenStatusSuccess) << textName << " wasn't retrieved";
        if(status == miopenStatusSuccess)
            EXPECT_TRUE(attrPtr->testReadBuffer()) << textName << " set and retrieved values differ";
        // clang-format on
    }
}

// TODO: Use testing::Combine to make
//       this list concise
INSTANTIATE_TEST_SUITE_P(
    Unit,
    CPU_GraphApiOperationConvolution_NONE,
    testing::Values(

        // Forward valid
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            true,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            true,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                                                   MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                                                   MIOPEN_TYPE_FLOAT,
                                                   MIOPEN_TYPE_CHAR,
                                                   0,
                                                   std::initializer_list<float>{0.1})}},

        // Forward non-finalized attr
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},

        // Forward null attr
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             true,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             true,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               true,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             true,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               true,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                             true,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               true,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},

        // Bwd data valid
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            true,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            true,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},

        // Bwd data non-finalized attr
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               true),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               true),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               true),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               false),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},

        // Bwd data null attr
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             true,
                                             true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             true,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               true,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             true,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               true,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                             MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                             true,
                                             false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               true,
                                               true),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},

        // Bwd filter valid
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            true,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            true,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.9}),
             std::make_shared<GTestAttrAlphaFloat>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_FLOAT,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<float>{0.1})}},

        // Bwd non-finalized attr
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},

        // Bwd null attr
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 true,
                 true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 true,
                 false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               true,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 true,
                 false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               true,
                                               true),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               true,
                                               false),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}},
        GTestDescriptor{
            "MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
            MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            false,
            {std::make_shared<GTestAttrConv>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                 true,
                 false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               true,
                                               false),
             std::make_shared<GTestAttrTensor>("MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                               MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               true,
                                               true),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.9}),
             std::make_shared<GTestAttrAlphaDouble>(
                 "MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                 MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                 MIOPEN_TYPE_DOUBLE,
                 MIOPEN_TYPE_CHAR,
                 0,
                 std::initializer_list<double>{0.1})}}

        ));
