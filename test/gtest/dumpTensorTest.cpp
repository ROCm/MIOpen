#include <iostream>
#include <limits>

#include "../driver.hpp"
#include "miopen/check_numerics.hpp"
#include "miopen/handle.hpp"
#include "../tensor_holder.hpp"

#include <miopen/convolution.hpp>
#include <gtest/gtest.h>

const std::string test_file_name_prefix = "dumptensortest_";
const size_t tensor_size                = 20;
const size_t nan_index                  = 5;

template <class T>
void prettyPrintTensor(const tensor<T>& host_tensor)
{
    for(std::size_t i = 0; i < host_tensor.desc.GetElementSize(); i++)
    {
        std::cerr << host_tensor[i] << ",";
        if((i + 1) % tensor_size == 0)
            std::cerr << "\n";
    }
    std::cerr << "\n";
}

template <class T>
void populateTensor(tensor<T>& host_tensor)
{
    // batch (n) = 1, channels(c) = 1, height(h) = tensor_size, width(w) = tensor_size
    host_tensor = tensor<T>{1, 1, tensor_size, tensor_size}.generate(tensor_elem_gen_integer{
        100}); // populate tensor with randomly generated element from [0-100]
}

template <class T>
void populateWithNAN(std::vector<T>& data)
{
    std::fill(data.begin(), data.end(), std::numeric_limits<T>::quiet_NaN());
}

template <class T>
void compare(const tensor<T>& host_tensor,
             const tensor<T>& tensor_from_file,
             bool compare_nan = false)
{
    ASSERT_EQ(host_tensor.data.size(), tensor_from_file.data.size());

    for(int i = 0; i < host_tensor.data.size(); ++i)
    {
        // if(compare_nan && i == nan_index)
        if(compare_nan)
        {
            EXPECT_TRUE(std::isnan(host_tensor[i])) << "Was expecting nan at index " << i;
            EXPECT_TRUE(std::isnan(tensor_from_file[i])) << "Was expecting nan at index " << i;
        }
        else
        {
            T tolerance = static_cast<T>(10);
            T threshold = std::numeric_limits<T>::epsilon() * tolerance;
            EXPECT_NEAR(host_tensor[i], tensor_from_file[i], threshold)
                << "Vectors host_tensor and tensor_from_file differ at index " << i;
        }
    }
}

template <typename T>
void readBufferFromFile(T* data, size_t dataNumItems, const std::string& fileName)
{
    std::ifstream infile(fileName, std::ios::binary);
    if(infile)
    {
        infile.read(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        infile.close();
    }
    else
    {
        EXPECT_TRUE(false) << "Could not open file : " << fileName;
    }
}

template <class T>
void testDump(const std::string& test_file_name)
{
    tensor<T> host_tensor;
    populateTensor(host_tensor);

    miopen::Handle handle{};

    // copy tensor from host to gpu
    auto gpu_tensor_addr = handle.Write(host_tensor.data);

    miopen::DumpTensorToFileFromDevice(
        handle, host_tensor.desc, gpu_tensor_addr.get(), test_file_name);

    // read back tensor
    tensor<T> tensor_from_file = tensor<T>{1, 1, tensor_size, tensor_size};
    readBufferFromFile<T>(
        tensor_from_file.data.data(), host_tensor.desc.GetElementSpace(), test_file_name);

    compare(host_tensor, tensor_from_file);

    // clean up
    boost::filesystem::remove(test_file_name);
}

template <class T>
void testDumpWithNan(const std::string& test_file_name)
{
    tensor<T> host_tensor;
    populateTensor(host_tensor);

    miopen::Handle handle{};
    // before writing to gpu we set one of the element
    // in the vector to nan.
    // host_tensor.data[nan_index] = std::numeric_limits<T>::quiet_NaN();
    populateWithNAN(host_tensor.data);

    // write the tensor to GPU
    auto gpu_tensor_addr = handle.Write(host_tensor.data);

    miopen::DumpTensorToFileFromDevice(
        handle, host_tensor.desc, gpu_tensor_addr.get(), test_file_name);

    if(miopen::checkNumericsInput(handle, host_tensor.desc, gpu_tensor_addr.get()))
    {
        // read back tensor
        tensor<T> tensor_from_file = tensor<T>{1, 1, tensor_size, tensor_size};
        readBufferFromFile<T>(
            tensor_from_file.data.data(), host_tensor.desc.GetElementSpace(), test_file_name);
        compare(host_tensor, tensor_from_file, true);
    }
    else
    {
        EXPECT_TRUE(false)
            << "Was expecting NAN value in tensor. Current value at host_tensor.data[" << nan_index
            << "] = " << host_tensor.data[nan_index];
    }
    // clean up
    boost::filesystem::remove(test_file_name);
}

TEST(DUMP_TENSOR_TEST, testDump_float) { testDump<float>(test_file_name_prefix + "float.bin"); }

TEST(DUMP_TENSOR_TEST, testDump_NAN_float)
{
    testDumpWithNan<float>(test_file_name_prefix + "nan_float.bin");
}

TEST(DUMP_TENSOR_TEST, testDump_half_float)
{
    testDump<half_float::half>(test_file_name_prefix + "half_float.bin");
}

TEST(DUMP_TENSOR_TEST, testDump_NAN_half_float)
{
    testDumpWithNan<half_float::half>(test_file_name_prefix + "nan_half_float.bin");
}

TEST(DUMP_TENSOR_TEST, testDump_bfloat16)
{
    testDump<bfloat16>(test_file_name_prefix + "bfloat16.bin");
}

TEST(DUMP_TENSOR_TEST, testDump_NAN_bfloat16)
{
    testDumpWithNan<bfloat16>(test_file_name_prefix + "nan_bfloat16.bin");
}
