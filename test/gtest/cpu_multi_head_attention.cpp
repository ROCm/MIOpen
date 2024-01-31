#include "cpu_multi_head_attention.hpp"
#include <hip_float8.hpp>

struct CPUMHATestDouble : test::cpu::CPUMHATest<double>
{
};

TEST_P(CPUMHATestDouble, CPUMHATestDoubleFw){

};

INSTANTIATE_TEST_SUITE_P(CPUMHATestSet,
                         CPUMHATestDouble,
                         testing::ValuesIn(test::cpu::CPUMHAConfigs()));
