#include "cpu_multi_head_attention.hpp"

struct CPU_Mha_FP32 : test::cpu::CPUMHATest<float, float>
{
};

struct CPU_Mha_FP8 : test::cpu::CPUMHATest<float, float8>
{
};

TEST_P(CPU_Mha_FP32, CPUMHATestFloatFw) {}

TEST_P(CPU_Mha_FP8, CPUMHATestFloat8Fw) {}

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_Mha_FP32, testing::ValuesIn(test::cpu::CPUMHAConfigs()));

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_Mha_FP8, testing::ValuesIn(test::cpu::CPUMHAConfigs()));
