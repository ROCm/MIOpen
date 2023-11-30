#include "cpu_multi_head_attention.hpp"

int main()
{
    size_t num_heads = 3;
    size_t problem_dimension = 6;
    size_t sequence_length = 7;
    size_t batch_size = 2;
    float drop_out_rate = 0.2;
    std::vector<size_t> mask_val_lens = {sequence_length, sequence_length};
    tensor<float> mask_val({mask_val_lens});
    // std::mutex coutMutex;
    mask_val.par_for_each(
        [&](size_t s_id, size_t p_id){
            if(p_id > s_id)
            {
                // std::lock_guard<std::mutex> guard(coutMutex);
                // std::cout << s_id << "," << p_id << std::endl;
                mask_val(s_id, p_id) = 
                    -std::numeric_limits<float>::infinity();
            }
        });
        
    tensor<float> mha = 
            test::cpu::multi_head_attention(num_heads,
                         problem_dimension,
                         sequence_length,
                         batch_size, 
                         mask_val,
                         drop_out_rate);
    
    test::cpu::print<float>(mha, problem_dimension);
}
