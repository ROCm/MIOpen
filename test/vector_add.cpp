/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <miopen/miopen.h>
#include "driver.hpp"
#include "verify.hpp" 
#include <miopen/kernel_build_params.hpp>

#define SIZE_TENSOR 256
#define THREADS_PER_BLOCK 256

template <typename T>
bool verify_tensor_ocl_hip(tensor<T>& t_ocl, tensor<T>& t_hip)
{
    EXPECT(t_ocl.data.size() == t_hip.data.size());
    auto idx          = miopen::mismatch_idx(t_ocl.data, t_hip.data,  [](T r1, T r2) { return r1 == r2; });
    bool valid_result = idx >= miopen::range_distance(t_hip);

    if(!valid_result)
    {
        std::cout << "diff at:" << idx << ", OCL:" << t_ocl[idx] << ", HIP:" << t_hip[idx]
                  << std::endl;
    }
    return valid_result;
}

// Generate numbers between 0 and 100
template <typename T>
void rand_tensor(tensor<T>& t, int max = 100, int min = 0) 
{
    // use integer to random.
    for(size_t i = 0; i < t.data.size(); i++)
        t[i] = static_cast<T>(prng::gen_A_to_B(min, max));
}


template <class T>
void vec_add(const tensor<T>& srcA,
                  const tensor<T>& srcB,
                  tensor<T>& dstC)
{
   
    for(size_t i = 0; i < SIZE_TENSOR; i++)
    {
        dstC[i] = srcA[i] + srcB[i];
    }

}

template <class T>
struct verify_vecadd_ocl
{
    tensor<T> srcA;
    tensor<T> srcB;
    tensor<T> dstC;

    verify_vecadd_ocl(const tensor<T>& p_srcA,
                        const tensor<T>& p_srcB,
                        const tensor<T>& p_dstC)
    {
        srcA      = p_srcA;
        srcB      = p_srcB;
        dstC      = p_dstC;

    }

    tensor<T> cpu() const
    {
        auto r = dstC;

        vec_add(srcA, srcB, r);

        return r;
    }

    tensor<T> gpu() const
    {
        auto r        = dstC;
        auto&& handle = get_handle();
        auto srcA_dev  = handle.Write(srcA.data);
        auto srcB_dev  = handle.Write(srcB.data);

        // clear the destination tensor
        for(size_t i = 0; i < r.data.size(); i++)
            r[i] = 0;

        auto dstC_dev  = handle.Write(r.data);


        std::string program_name = "MIOpenVecAddOCL.cl";
        std::string kernel_name  = "vector_add_ocl";

        std::string network_config = "standalone_kernel_vector_add_ocl";

        miopen::KernelBuildParameters options{
        };

        std::string params = options.GenerateFor(miopen::kbp::OpenCL{});

        uint totalElements = SIZE_TENSOR;
        int threadsPerBlock = THREADS_PER_BLOCK;
        int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

        const std::vector<size_t> vgd{blocksPerGrid * threadsPerBlock, 1, 1};
        const std::vector<size_t> vld{threadsPerBlock, 1, 1};

        handle.AddKernel("vector_add_ocl", network_config, program_name, kernel_name, vld, vgd, params)(srcA_dev.get(), srcB_dev.get(), dstC_dev.get(), totalElements);
        r.data = handle.Read<T>(dstC_dev, dstC.data.size());

        return r;
    }

    void fail(float = 0)
    {
        std::cout << "VecAdd OCL Fail: " << std::endl;
        std::cout << "srcA tensor: " << srcA.desc.ToString() << std::endl;
        std::cout << "srcB tensor: " << srcB.desc.ToString() << std::endl;
        std::cout << "dstC tensor: " << dstC.desc.ToString() << std::endl;

        // print the first 5 elements of the tensors
        std::cout << "srcA: ";
        for(size_t i = 0; i < 5; i++)
            std::cout << srcA[i] << " ";
        std::cout << std::endl;

        std::cout << "srcB: ";
        for(size_t i = 0; i < 5; i++)
            std::cout << srcB[i] << " ";
        std::cout << std::endl;

        std::cout << "dstC: ";
        for(size_t i = 0; i < 5; i++)
            std::cout << dstC[i] << " ";
        std::cout << std::endl;
        
    }
};


template <class T>
struct verify_vecadd_hip
{
    tensor<T> srcA;
    tensor<T> srcB;
    tensor<T> dstC;

    verify_vecadd_hip(  const tensor<T>& p_srcA,
                        const tensor<T>& p_srcB,
                        const tensor<T>& p_dstC)
    {
        srcA      = p_srcA;
        srcB      = p_srcB;
        dstC      = p_dstC;

    }

    tensor<T> cpu() const
    {
        auto r = dstC;

        vec_add(srcA, srcB, r);

        return r;
    }

    tensor<T> gpu() const
    {
        auto r        = dstC;
        auto&& handle = get_handle();
        auto srcA_dev  = handle.Write(srcA.data);
        auto srcB_dev  = handle.Write(srcB.data);

        // clear the destination tensor
        for(size_t i = 0; i < r.data.size(); i++)
            r[i] = 0;

        auto dstC_dev  = handle.Write(r.data);

        std::string program_name = "MIOpenVecAdd.cpp";
        std::string kernel_name  = "vector_add_hip";

        std::string network_config = "standalone_kernel_vector_add_hip";

        miopen::KernelBuildParameters options{
        };

        std::string params = options.GenerateFor(miopen::kbp::HIP{});

        uint totalElements = SIZE_TENSOR;
        int threadsPerBlock = THREADS_PER_BLOCK;
        int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

        const std::vector<size_t> vgd{blocksPerGrid * threadsPerBlock, 1, 1};
        const std::vector<size_t> vld{threadsPerBlock, 1, 1};

        handle.AddKernel("vector_add_hip", network_config, program_name, kernel_name, vld, vgd, params)(srcA_dev.get(), srcB_dev.get(), dstC_dev.get(), totalElements);
        r.data = handle.Read<T>(dstC_dev, dstC.data.size());
        
        return r;
    }

    void fail(float = 0)
    {

        std::cout << "VecAdd HIP Fail: " << std::endl;
        std::cout << "srcA tensor: " << srcA.desc.ToString() << std::endl;
        std::cout << "srcB tensor: " << srcB.desc.ToString() << std::endl;
        std::cout << "dstC tensor: " << dstC.desc.ToString() << std::endl;

        // print the first 5 elements of the tensors
        std::cout << "srcA: ";
        for(size_t i = 0; i < 5; i++)
            std::cout << srcA[i] << " ";
        std::cout << std::endl;

        std::cout << "srcB: ";
        for(size_t i = 0; i < 5; i++)
            std::cout << srcB[i] << " ";
        std::cout << std::endl;

        std::cout << "dstC: ";
        for(size_t i = 0; i < 5; i++)
            std::cout << dstC[i] << " ";
        std::cout << std::endl;

    }
};


template <class T>
struct vecAdd_driver : test_driver
{
    

    void run()
    {
        tensor<T> srcA(SIZE_TENSOR);
        tensor<T> srcB(SIZE_TENSOR);
        tensor<T> dstC(SIZE_TENSOR);

        disabled_cache = true;
        
        rand_tensor(srcA);
        rand_tensor(srcB);

        auto&& handle = get_handle();
        handle.EnableProfiling();
        
        std::cout << "Verifying vecAdd OCL vs CPU..." << std::endl;

        verify_equals(verify_vecadd_ocl<T>{srcA, srcB, dstC});

        std::cout << "Done." << std::endl;

        // insert random values into destination tensor
        rand_tensor(dstC);

        std::cout << "Verifying vecAdd HIP vs CPU..." << std::endl;

        verify_equals(verify_vecadd_hip<T>{srcA, srcB, dstC});

        std::cout << "Done." << std::endl;

        std::cout << "Verifying vecAdd OCL vs HIP..." << std::endl;

        auto r_ocl = verify_vecadd_ocl<T>{srcA, srcB, dstC}.gpu();
        auto r_hip = verify_vecadd_hip<T>{srcA, srcB, dstC}.gpu();

        verify_tensor_ocl_hip(r_ocl, r_hip);

        std::cout << "Done." << std::endl;

        
    }
};

int main(int argc, const char* argv[]) { 
    

    // My kernels currently only support float so append --float to argv
    std::vector<const char*> new_argv(argv, argv + argc); 
    new_argv.push_back("--float"); // Add argv
    argc = new_argv.size(); // Update argc
    argv = new_argv.data(); // Update argv
    
    test_drive<vecAdd_driver>(argc, argv); 
    

    
    }
