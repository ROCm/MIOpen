#include "cpu_multi_head_attention.hpp"
#include <hip_float8.hpp>

struct CPUMHATestFloat : test::cpu::CPUMHATest<float>
{
};

TEST_P(CPUMHATestFloat, CPUMHATestFloatFw){

};

INSTANTIATE_TEST_SUITE_P(CPUMHATestSet,
                         CPUMHATestFloat,
                         testing::ValuesIn(test::cpu::CPUMHAConfigs()));
