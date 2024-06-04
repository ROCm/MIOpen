#include <gtest/gtest_common.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

// For determining if we should run test suite. First ensure that test is supported on the hardware.
// If the MIOPEN_TEST_ALL environment isn't set, then assume we are running standalone outside
// CICD, and include the test. Otherwise, check the provided functor to ensure the environment
// conditions match expected conditions to run this test suite.
template <typename disabled_mask, typename enabled_mask, typename check_functor>
bool ShouldRunTestCase(check_functor&& checkConditions)
{
    return IsTestSupportedForDevMask<disabled_mask, enabled_mask>() &&
           (miopen::IsUnset(MIOPEN_ENV(MIOPEN_TEST_ALL)) || checkConditions());
}
