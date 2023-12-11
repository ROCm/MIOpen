#include "cpu_multi_head_attention.hpp"

struct CPUMHATestFloat : test::cpu::CPUMHATest<float>
{
};

TEST_P(CPUMHATestFloat, CPUMHATestFloatFw){

};

INSTANTIATE_TEST_SUITE_P(CPUMHATestSet,
                         CPUMHATestFloat,
                         testing::ValuesIn(test::cpu::CPUMHAConfigs()));
