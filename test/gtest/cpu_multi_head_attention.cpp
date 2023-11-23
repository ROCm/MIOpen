#include "cpu_multi_head_attention.hpp"

int main()
{
    size_t num_heads = 3;
    size_t problem_dimension = 15;
    size_t sequence_length = 10;
    size_t batch_size = 2;
    tensor<float> mask_val;
    float drop_out_rate = 0.5;

    test::cpu::multi_head_attention(num_heads,
                         problem_dimension,
                         sequence_length,
                         batch_size, 
                         mask_val,
                         drop_out_rate);
}
