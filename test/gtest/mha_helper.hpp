
#include "tensor_holder.hpp"
#include "conv_tensor_gen.hpp"
#include <type_traits>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/layernorm.hpp>

#include "tensor_holder.hpp"
#include "cpu_layernorm.hpp"
#include "get_handle.hpp"
#include "../driver/tensor_driver.hpp"
#include "verify.hpp"
#include <random>
#include <hip_float8.hpp>

namespace test {
namespace cpu {

struct CPUMHATestCase
{
    // represents total number of seq len present in batch
    size_t batch_size;
    size_t sequence_length;
    size_t num_heads;
    size_t problem_dimension;
    float drop_out_rate;

    friend std::ostream& operator<<(std::ostream& os, const CPUMHATestCase& tc)
    {
        return os << "(batch_size: " << tc.batch_size
                  << " num_heads:" << tc.num_heads 
                  << " sequence_length:" << tc.sequence_length
                  << " problem_dimension:" << tc.problem_dimension
                  << " drop_out_rate:" << tc.drop_out_rate << " )";
    }
};

// template <typename U, typename V>
// struct Fp8Cast
// {
//     uint64_t seed = 1234;
//     bool is_stoch = true;
//     V operator()(U x)
//     {
//         if(is_stoch)
//         {
//             auto tmp =
//                 float8(static_cast<float>(x), miopen_f8::hip_f8_rounding_mode::stochastic, seed);
//             return static_cast<V>(tmp);
//         }
//         else
//         {
//             auto tmp = float8(static_cast<float>(x));
//             return static_cast<V>(tmp);
//         }
//     }
// };

// using F8_T = Fp8Cast<float, float>;

template <typename T>
void print4(const tensor<T>& tensor_val, std::string header_msg = "start")
{
    std::cout << "\n================= " << header_msg << " =====================\n";
    size_t i_size = tensor_val.desc.GetLengths()[0];
    size_t j_size = tensor_val.desc.GetLengths()[1];
    size_t k_size = tensor_val.desc.GetLengths()[2];
    size_t l_size = tensor_val.desc.GetLengths()[3];
    std::cout << i_size << "," << j_size << ", " << k_size << "," << l_size << std::endl;
    for(size_t i = 0; i < i_size; ++i)
    {
        for(size_t j = 0; j < j_size; ++j)
        {
            for(size_t k = 0; k < k_size; ++k)
            {
                for(size_t l = 0; l < l_size; ++l)
                {
                    std::cout << std::fixed << std::setprecision(2) << tensor_val(i, j, k, l)
                              << " , ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n=================end=====================\n";
}

template <typename T>
void print3(const tensor<T>& tensor_val, std::string header_msg = "start")
{
    std::cout << "\n================= " << header_msg << " =====================\n";
    size_t i_size = tensor_val.desc.GetLengths()[0];
    size_t j_size = tensor_val.desc.GetLengths()[1];
    size_t k_size = tensor_val.desc.GetLengths()[2];
    std::cout << i_size << "," << j_size << ", " << k_size << std::endl;
    for(size_t i = 0; i < i_size; ++i)
    {
        for(size_t j = 0; j < j_size; ++j)
        {
            for(size_t k = 0; k < k_size; ++k)
            {
                std::cout << std::fixed << std::setprecision(2) << tensor_val(i, j, k) << " , ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n=================end=====================\n";
}

// C_mat = A_mat.dot(B_mat)
// A_mat : 3D
// B_mat : 3D
// C_mat : 4D
template <typename T>
void Dot_3D_3D(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[2];
    assert(k_val == B_mat.desc.GetLengths()[1]);
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, sl_id, k_id)* B_mat(h_id, k_id, dk_id);
        }
        C_mat(b_id, h_id, sl_id, dk_id) = sum;
    });
}

// C_mat = A_mat.dot(transpose(B_mat))
// A_mat : 4D
// B_mat : 4D
// C_mat : 4D
template <typename T>
void Dot_4D_4D_T(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[3]); // since transpose
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, h_id, sl_id, k_id) * B_mat(b_id, h_id, dk_id, k_id);
        }

        C_mat(b_id, h_id, sl_id, dk_id) = sum;
    });
}

// C_mat = A_mat.dot(B_mat)
// A_mat : 4D
// B_mat : 4D
// C_mat : 4D
template <typename T>
void Dot_4D_4D(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[2]);
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, h_id, sl_id, k_id) * B_mat(b_id, h_id, k_id, dk_id);
        }

        C_mat(b_id, h_id, sl_id, dk_id) = sum;
    });
}

template <typename T>
void AddMask5D_2D(tensor<T>& mat_A_val, const tensor<T>& mat_mask)
{
    mat_A_val.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            mat_A_val(b_id, h_id, sl_i_id, sl_j_id) =
                mat_A_val(b_id, h_id, sl_i_id, sl_j_id) + mat_mask(sl_i_id, sl_j_id);
        });
}

// Computes the sum of the row in A_mat
// A_mat : 5D
// rrm_tensor : 5D
template <class T>
void RowReductionMax(const tensor<T>& A_mat, tensor<T>& rrm_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    rrm_tensor.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_id, size_t sl0_id) {
            T max(A_mat(b_id, h_id, sl_id, sl0_id));
            for(size_t id = 0; id < sl_dim; ++id)
            {
                if(A_mat(b_id, h_id, sl_id, id) > max)
                {
                    max = A_mat(b_id, h_id, sl_id, id);
                }
            }
            rrm_tensor(b_id, h_id, sl_id, sl0_id) = max;
        });
}

template <class T>
void Zinv(const tensor<T>& A_mat, tensor<T>& zinv_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    T one(1);
    zinv_tensor.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_id, size_t sl0_id) {
            double sum(0);
            for(size_t id = 0; id < sl_dim; ++id)
            {
                sum += A_mat(b_id, h_id, sl_id, id);
            }
            zinv_tensor(b_id, h_id, sl_id, sl0_id) = one / sum;
        });
}



template <typename T>
void Scale(tensor<T>& mat_val, double scale_factor)
{
    mat_val.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            mat_val(b_id, h_id, sl_i_id, sl_j_id) =
                mat_val(b_id, h_id, sl_i_id, sl_j_id) * scale_factor;
        });
}

// template <typename T>
// void Scalef8(tensor<T>& mat_val, F8_T scale_func, double scale_factor)
// {
//     // assert scale_factor is != 0.0 .. tolerance
//     mat_val.par_for_each(
//         [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
//             mat_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) =
//                 scale_func(mat_val(b_id, sc_id, h_id, sl_i_id, sl_j_id)) * scale_factor;
//         });
// }

template <typename T>
void Sub(tensor<T>& mat_A_val, const tensor<T>& mat_sub)
{
    mat_A_val.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            mat_A_val(b_id, h_id, sl_i_id, sl_j_id) =
                mat_A_val(b_id, h_id, sl_i_id, sl_j_id) -
                mat_sub(b_id, h_id, sl_i_id, 0);
        });
}

template <class T>
void Exponent(tensor<T>& tensor_val)
{
    tensor_val.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            tensor_val(b_id, h_id, sl_i_id, sl_j_id) =
                std::exp(tensor_val(b_id, h_id, sl_i_id, sl_j_id));
        });
}

template <class T>
void ZinvMultiply(tensor<T>& q_dot_k_transpose, const tensor<T>& zinv_tensors)
{
    q_dot_k_transpose.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            q_dot_k_transpose(b_id, h_id, sl_i_id, sl_j_id) =
                q_dot_k_transpose(b_id, h_id, sl_i_id, sl_j_id) *
                zinv_tensors(b_id, h_id, sl_i_id, 0);
        });
}

template <typename T>
void SetupMask(tensor<T>& mask)
{
    mask.par_for_each([&](size_t s_id, size_t p_id) {
        // make anything above diagonal inf
        if(p_id > s_id)
        {
            mask(s_id, p_id) = -std::numeric_limits<T>::infinity();
        }
    });
}

template <typename T>
void Transpose_5D(const tensor<T>& mat_val, tensor<T>& trans_mat)
{
    trans_mat.par_for_each([&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_id, size_t dk_id) {
        trans_mat(b_id, sc_id, h_id, sl_id, dk_id) = mat_val(b_id, sc_id, h_id, dk_id, sl_id);
    });
}

template <class T>
void DropOut(tensor<T>& q_dot_k_transpose, const double& drop_out_rate)
{
    tensor<T> rand_dis(q_dot_k_transpose.desc.GetLengths());
    rand_dis.generate(GenData<T>{});
    q_dot_k_transpose.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            if(rand_dis(b_id, sc_id, h_id, sl_i_id, sl_j_id) < drop_out_rate)
            {
                q_dot_k_transpose(b_id, sc_id, h_id, sl_i_id, sl_j_id) = T(0);
            }
        });
}

}
}
