#include <miopen.h>
#include "test.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/convolution.hpp>
#include <limits>

#include "tensor_holder.hpp"
#include "verify.hpp"
#include "driver.hpp"
#include "get_handle.hpp"

template<class T>
struct tensor_ops_base
{
    tensor<T> a;
    tensor<T> b;
    tensor<T> c;

    void fail(float=0)
    {
        std::cout << "A tensor: " << a.desc.ToString() << std::endl;
        std::cout << "B tensor: " << b.desc.ToString() << std::endl;
        std::cout << "C tensor: " << c.desc.ToString() << std::endl;
    }
};

template<class T>
struct verify_tensor_ops : tensor_ops_base<T>
{
    using tensor_ops_base<T>::a;
    using tensor_ops_base<T>::b;
    using tensor_ops_base<T>::c;

    verify_tensor_ops(const tensor<T>& pa, const tensor<T>& pb)
    {
        a = pa;
        b = pb;
    }

    tensor<T> cpu()
    {
        c = a;
        std::fill(c.begin(), c.end(), 0);
        return c;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();

        c = a;
        std::fill(c.begin(), c.end(), 0);

        auto c_dev = handle.Write(c.data);
        auto a_dev = handle.Write(a.data);
        auto b_dev = handle.Write(b.data);

        int alpha1 = 1, alpha2 = 1, beta = 0;
        
        miopen::OpTensor(handle,
                miopenOpTensorAdd,
                &alpha1,
                a.desc,
                a_dev.get(),
                &alpha2,
                b.desc,
                b_dev.get(),
                &beta,
                c.desc,
                c_dev.get());

        c.data = handle.Read<T>(c_dev, c.data.size());
        return c;
    }
    
    void fail(float=0)
    {
        std::cout << "TensorOp: " << std::endl;
        this->tensor_ops_base<T>::fail();
    }

};

template<class T>
struct tensor_ops_driver : test_driver
{
    tensor<T> a;
    tensor<T> b;

    tensor_ops_driver()
    {
        add(a, "a", generate_tensor(get_tensor_a(), {16, 16, 28, 28})); 
        add(b, "b", generate_tensor(get_tensor_b(), {1, 16, 1, 1})); 
    }

    std::set<std::vector<int>> get_tensor_a()
    {
        std::vector<std::vector<int>> a_dims {
            {32, 3,  16, 16},
            {16, 16, 27, 27} 
        };
        return (std::set<std::vector<int>> (a_dims.begin(), a_dims.end()));
    }

    std::set<std::vector<int>> get_tensor_b()
    {
        std::vector<std::vector<int>> b_dims {
            { 1,  3,  1,  1 },
            { 1,  16, 1,  1 },
            { 1,  1,  1,  16},
            { 1,  1,  16, 1 },
            { 1,  1,  16, 16},
            { 1,  3,  1,  16},
            { 1,  3,  16, 1 },
            { 1,  3,  16, 16},
            { 32, 3,  1,  1 },
            { 32, 16, 1,  1 },
            { 32, 1,  1,  16},
            { 32, 1,  16, 1 },
            { 32, 1,  16, 16},
            { 32, 3,  1,  16},
            { 32, 3,  16, 1 },
            { 32, 3,  16, 16},
            { 2,  4,  6,  8 },
            { 16, 16, 27, 1 },
            { 16, 16, 1,  1 },
            { 16, 1,  1,  1 }
        };
        return (std::set<std::vector<int>> (b_dims.begin(), b_dims.end()));
    }

    void run()
    {
        verify(verify_tensor_ops<T>{a, b});
    }
};

int main(int argc, const char *argv[])
{
    test_drive<tensor_ops_driver<float>>(argc, argv);
}
