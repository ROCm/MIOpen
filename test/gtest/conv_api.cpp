#include <iostream>
#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <gtest/gtest.h>

void testGetConvolutionSpatialDim(void)
{
    int spatial_dim = 0;
    int pads[]      = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int strides[]   = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int dilations[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for(size_t i = 0; i < 10; i++)
    {
        miopenConvolutionDescriptor_t conv_desc;
        miopenCreateConvolutionDescriptor(&conv_desc);
        miopenInitConvolutionNdDescriptor(
            conv_desc, i, pads, strides, dilations, miopenConvolutionMode_t::miopenConvolution);
        miopenGetConvolutionSpatialDim(conv_desc, &spatial_dim);
        ASSERT_EQ(spatial_dim, i) << "Spatial Dimension does not match at index: " << i
                                  << std::endl;
    }
}

TEST(CPU_ConvApi_NONE, testGetConvolutionSpatialDim) { testGetConvolutionSpatialDim(); }
