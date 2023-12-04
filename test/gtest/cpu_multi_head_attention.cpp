#include "cpu_multi_head_attention.hpp"

int main()
{
    using T                           = float;
    size_t num_heads                  = 3;
    size_t problem_dimension          = 6;
    size_t sequence_length            = 7;
    size_t batch_size                 = 2;
    float drop_out_rate               = 0.2;
    std::vector<size_t> mask_val_lens = {sequence_length, sequence_length};
    tensor<T> mask_val({mask_val_lens});
    // std::mutex coutMutex;
    mask_val.par_for_each([&](size_t s_id, size_t p_id) {
        // make anything above diagonal inf
        if(p_id > s_id)
        {
            // std::lock_guard<std::mutex> guard(coutMutex);
            // std::cout << s_id << "," << p_id << std::endl;
            mask_val(s_id, p_id) = -std::numeric_limits<T>::infinity();
        }
    });

    tensor<T> mha;
    std::vector<tensor<T>> set_of_m_tensors;
    std::vector<tensor<T>> set_of_zinv_tensors;

    std::tie(mha, set_of_m_tensors, set_of_zinv_tensors) = test::cpu::multi_head_attention(
        num_heads, problem_dimension, sequence_length, batch_size, mask_val, drop_out_rate);

    test::cpu::print<T>(mha, problem_dimension);
    test::cpu::print<T>(set_of_m_tensors[0], problem_dimension);
    test::cpu::print<T>(set_of_zinv_tensors[0], problem_dimension);
}
