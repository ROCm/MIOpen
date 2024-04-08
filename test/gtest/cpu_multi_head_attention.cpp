#include "cpu_multi_head_attention.hpp"
#include <hip_float8.hpp>

struct CPUMHATestFloat : test::cpu::CPUMHATest<float, float>
{
};

struct CPUMHATestFloat8 : test::cpu::CPUMHATest<float, float8>
{
};

TEST_P(CPUMHATestFloat, CPUMHATestFloatFw) {}

TEST_P(CPUMHATestFloat8, CPUMHATestFloat8Fw) {}

INSTANTIATE_TEST_SUITE_P(CPUMHATestSet,
                         CPUMHATestFloat,
                         testing::ValuesIn(test::cpu::CPUMHAConfigs()));

INSTANTIATE_TEST_SUITE_P(CPUMHATestSet,
                         CPUMHATestFloat8,
                         testing::ValuesIn(test::cpu::CPUMHAConfigs()));
