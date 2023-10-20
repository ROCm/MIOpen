#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>

#include <conv_fin.hpp>
#include <bn_fin.hpp>

using json = nlohmann::json;

TEST(MemoryLayoutTest, BasicMemLayoutConv)
{
    std::string input_filename = TEST_RESOURCE_DIR "fin_input_find_compile2.json";
    std::ifstream input_file(input_filename);
    if(!input_file)
    {
        EXPECT_FALSE(true) << "ERROR: cannot open test file " << input_filename << std::endl;
    }

    json j;
    input_file >> j;
    input_file.close();
    for(auto& it : j)
    {
        auto command = it;
        if(command["config"]["cmd"] == "conv")
        {
            fin::ConvFin<float, float> tmp(command);
            ASSERT_TRUE(tmp.inputTensor.desc.GetLayout_t() ==
                        miopenTensorLayout_t::miopenTensorNCHW);
            // set the layout from json file
            tmp.GetandSetData();
            ASSERT_TRUE(tmp.inputTensor.desc.GetLayout_t() ==
                        miopenTensorLayout_t::miopenTensorNHWC);
            ASSERT_TRUE(tmp.inputTensor.desc.GetLayout_t() !=
                        miopenTensorLayout_t::miopenTensorNCHW);
        }
    }
}

TEST(MemoryLayoutTest, BasicMemLayoutBatchNorm)
{
    std::string input_filename = TEST_RESOURCE_DIR "fin_input_find_compile.json";
    std::ifstream input_file(input_filename);
    if(!input_file)
    {
        EXPECT_FALSE(true) << "ERROR: cannot open test file " << input_filename << std::endl;
    }

    json j;
    input_file >> j;
    input_file.close();
    for(auto& it : j)
    {
        auto command = it;
        if(command["config"]["cmd"] == "bnorm")
        {
            fin::BNFin<float, float> tmp(command);
            ASSERT_TRUE(tmp.inputTensor.desc.GetLayout_t() ==
                        miopenTensorLayout_t::miopenTensorNCHW);
            try
            {
                tmp.GetandSetData();
            }
            catch(const std::exception& err)
            {
                EXPECT_EQ(err.what(),
                          std::string("Provided memory layout is :" +
                                      std::string(command["config"]["in_layout"]) +
                                      ". Batch norm only support default NCHW"));
            }
        }
    }
}

TEST(MemoryLayoutTest, TestGetMemLayout)
{
    miopenTensorLayout_t nchw_layout = fin::GetMemLayout("NCHW");
    ASSERT_TRUE(nchw_layout == miopenTensorLayout_t::miopenTensorNCHW);

    miopenTensorLayout_t nhwc_layout = fin::GetMemLayout("NHWC");
    ASSERT_TRUE(nhwc_layout == miopenTensorLayout_t::miopenTensorNHWC);

    std::string unknown_layout = "UNKNOWN";
    try
    {
        fin::GetMemLayout(unknown_layout);
    }
    catch(const std::exception& err)
    {
        EXPECT_EQ(err.what(), std::string("Unknown memory layout : " + unknown_layout));
    }
}
