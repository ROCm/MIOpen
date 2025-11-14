// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/find_controls.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/env.hpp>

#include <cstdlib>
#include <string>

// Mock context for testing
struct MockContext
{
    bool disable_search_enforce = false;
};

class CPU_FindControls_NONE : public ::testing::Test
{
protected:
    void SetUp() override
    {
        original_find_enforce_disable     = miopen::debug::FindEnforceDisable;
        miopen::debug::FindEnforceDisable = false;
    }

    void TearDown() override { miopen::debug::FindEnforceDisable = original_find_enforce_disable; }

    bool original_find_enforce_disable;
    MockContext context;
};

// Test FindEnforceAction basic functionality
TEST_F(CPU_FindControls_NONE, FindEnforceActionValues)
{
    EXPECT_EQ(static_cast<int>(miopen::FindEnforceAction::None), 1);
    EXPECT_EQ(static_cast<int>(miopen::FindEnforceAction::DbUpdate), 2);
    EXPECT_EQ(static_cast<int>(miopen::FindEnforceAction::Search), 3);
    EXPECT_EQ(static_cast<int>(miopen::FindEnforceAction::SearchDbUpdate), 4);
    EXPECT_EQ(static_cast<int>(miopen::FindEnforceAction::DbClean), 5);
}

TEST_F(CPU_FindControls_NONE, FindModeValues)
{
    EXPECT_EQ(static_cast<int>(miopen::FindMode::Values::Normal), 1);
    EXPECT_EQ(static_cast<int>(miopen::FindMode::Values::Fast), 2);
    EXPECT_EQ(static_cast<int>(miopen::FindMode::Values::Hybrid), 3);
    EXPECT_EQ(static_cast<int>(miopen::FindMode::Values::DeprecatedFastHybrid), 4);
    EXPECT_EQ(static_cast<int>(miopen::FindMode::Values::DynamicHybrid), 5);
    EXPECT_EQ(static_cast<int>(miopen::FindMode::Values::TrustVerify), 6);
    EXPECT_EQ(static_cast<int>(miopen::FindMode::Values::TrustVerifyFull), 7);
}

// Test FindEnforce class with explicit constructors
TEST_F(CPU_FindControls_NONE, FindEnforceGetAction)
{
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    EXPECT_EQ(enforce_none.GetAction(), miopen::FindEnforceAction::None);

    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    EXPECT_EQ(enforce_db_update.GetAction(), miopen::FindEnforceAction::DbUpdate);

    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    EXPECT_EQ(enforce_search.GetAction(), miopen::FindEnforceAction::Search);

    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    EXPECT_EQ(enforce_search_db_update.GetAction(), miopen::FindEnforceAction::SearchDbUpdate);

    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);
    EXPECT_EQ(enforce_clean.GetAction(), miopen::FindEnforceAction::DbClean);
}

TEST_F(CPU_FindControls_NONE, FindEnforceIsDbClean)
{
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);

    EXPECT_FALSE(enforce_none.IsDbClean(context));
    EXPECT_FALSE(enforce_db_update.IsDbClean(context));
    EXPECT_FALSE(enforce_search.IsDbClean(context));
    EXPECT_FALSE(enforce_search_db_update.IsDbClean(context));
    EXPECT_TRUE(enforce_clean.IsDbClean(context));
}

TEST_F(CPU_FindControls_NONE, FindEnforceIsSearch)
{
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);

    EXPECT_FALSE(enforce_none.IsSearch(context));
    EXPECT_FALSE(enforce_db_update.IsSearch(context));
    EXPECT_TRUE(enforce_search.IsSearch(context));
    EXPECT_TRUE(enforce_search_db_update.IsSearch(context));
    EXPECT_FALSE(enforce_clean.IsSearch(context));
}

TEST_F(CPU_FindControls_NONE, FindEnforceIsDbUpdate)
{
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);

    EXPECT_FALSE(enforce_none.IsDbUpdate(context));
    EXPECT_TRUE(enforce_db_update.IsDbUpdate(context));
    EXPECT_FALSE(enforce_search.IsDbUpdate(context));
    EXPECT_TRUE(enforce_search_db_update.IsDbUpdate(context));
    EXPECT_FALSE(enforce_clean.IsDbUpdate(context));
}

TEST_F(CPU_FindControls_NONE, FindEnforceIsSomethingEnforced)
{
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);

    EXPECT_FALSE(enforce_none.IsSomethingEnforced(context));
    EXPECT_TRUE(enforce_db_update.IsSomethingEnforced(context));
    EXPECT_TRUE(enforce_search.IsSomethingEnforced(context));
    EXPECT_TRUE(enforce_search_db_update.IsSomethingEnforced(context));
    EXPECT_TRUE(enforce_clean.IsSomethingEnforced(context));
}

TEST_F(CPU_FindControls_NONE, FindEnforceDisabledByContext)
{
    context.disable_search_enforce = true;
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);

    EXPECT_FALSE(enforce_none.IsDbClean(context));
    EXPECT_FALSE(enforce_db_update.IsDbClean(context));
    EXPECT_FALSE(enforce_search.IsDbClean(context));
    EXPECT_FALSE(enforce_search_db_update.IsDbClean(context));
    EXPECT_FALSE(enforce_clean.IsDbClean(context));

    EXPECT_FALSE(enforce_none.IsSearch(context));
    EXPECT_FALSE(enforce_db_update.IsSearch(context));
    EXPECT_FALSE(enforce_search.IsSearch(context));
    EXPECT_FALSE(enforce_search_db_update.IsSearch(context));
    EXPECT_FALSE(enforce_clean.IsSearch(context));

    EXPECT_FALSE(enforce_none.IsDbUpdate(context));
    EXPECT_FALSE(enforce_db_update.IsDbUpdate(context));
    EXPECT_FALSE(enforce_search.IsDbUpdate(context));
    EXPECT_FALSE(enforce_search_db_update.IsDbUpdate(context));
    EXPECT_FALSE(enforce_clean.IsDbUpdate(context));

    EXPECT_FALSE(enforce_none.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_db_update.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_search.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_search_db_update.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_clean.IsSomethingEnforced(context));
}

TEST_F(CPU_FindControls_NONE, FindEnforceDisabledByDebugFlag)
{
    miopen::debug::FindEnforceDisable = true;
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);

    EXPECT_FALSE(enforce_none.IsDbClean(context));
    EXPECT_FALSE(enforce_db_update.IsDbClean(context));
    EXPECT_FALSE(enforce_search.IsDbClean(context));
    EXPECT_FALSE(enforce_search_db_update.IsDbClean(context));
    EXPECT_FALSE(enforce_clean.IsDbClean(context));

    EXPECT_FALSE(enforce_none.IsSearch(context));
    EXPECT_FALSE(enforce_db_update.IsSearch(context));
    EXPECT_FALSE(enforce_search.IsSearch(context));
    EXPECT_FALSE(enforce_search_db_update.IsSearch(context));
    EXPECT_FALSE(enforce_clean.IsSearch(context));

    EXPECT_FALSE(enforce_none.IsDbUpdate(context));
    EXPECT_FALSE(enforce_db_update.IsDbUpdate(context));
    EXPECT_FALSE(enforce_search.IsDbUpdate(context));
    EXPECT_FALSE(enforce_search_db_update.IsDbUpdate(context));
    EXPECT_FALSE(enforce_clean.IsDbUpdate(context));

    EXPECT_FALSE(enforce_none.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_db_update.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_search.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_search_db_update.IsSomethingEnforced(context));
    EXPECT_FALSE(enforce_clean.IsSomethingEnforced(context));
}

// Test FindMode class with manual setting (bypassing environment)
TEST_F(CPU_FindControls_NONE, FindModeManualSetting_NoEnforcement)
{
    // Test all modes work when no enforcement is active
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_TRUE(mode_fast.IsFast(context));
    EXPECT_FALSE(mode_fast.IsHybrid(context));

    EXPECT_TRUE(mode_hybrid.IsHybrid(context));
    EXPECT_FALSE(mode_hybrid.IsFast(context));

    EXPECT_FALSE(mode_normal.IsFast(context));
    EXPECT_FALSE(mode_normal.IsHybrid(context));
}

// Test FindMode specific mode checks
TEST_F(CPU_FindControls_NONE, FindModeIsFast)
{
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_FALSE(mode_normal.IsFast(context));
    EXPECT_TRUE(mode_fast.IsFast(context));
    EXPECT_FALSE(mode_hybrid.IsFast(context));
    EXPECT_FALSE(mode_dynamic_hybrid.IsFast(context));
    EXPECT_FALSE(mode_trust_verify.IsFast(context));
    EXPECT_FALSE(mode_trust_verify_full.IsFast(context));
}

TEST_F(CPU_FindControls_NONE, FindModeIsHybrid)
{
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_FALSE(mode_normal.IsHybrid(context));
    EXPECT_FALSE(mode_fast.IsHybrid(context));
    EXPECT_TRUE(mode_hybrid.IsHybrid(context));
    EXPECT_TRUE(mode_dynamic_hybrid.IsHybrid(context));
    EXPECT_TRUE(mode_trust_verify.IsHybrid(context));
    EXPECT_TRUE(mode_trust_verify_full.IsHybrid(context));
}

TEST_F(CPU_FindControls_NONE, FindModeIsDynamicHybrid)
{
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_FALSE(mode_normal.IsDynamicHybrid(context));
    EXPECT_FALSE(mode_fast.IsDynamicHybrid(context));
    EXPECT_FALSE(mode_hybrid.IsDynamicHybrid(context));
    EXPECT_TRUE(mode_dynamic_hybrid.IsDynamicHybrid(context));
    EXPECT_TRUE(mode_trust_verify.IsDynamicHybrid(context));
    EXPECT_TRUE(mode_trust_verify_full.IsDynamicHybrid(context));
}

#if !MIOPEN_DISABLE_USERDB
TEST_F(CPU_FindControls_NONE, FindModeIsTrustVerify)
{
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_FALSE(mode_normal.IsTrustVerify(context));
    EXPECT_FALSE(mode_fast.IsTrustVerify(context));
    EXPECT_FALSE(mode_hybrid.IsTrustVerify(context));
    EXPECT_FALSE(mode_dynamic_hybrid.IsTrustVerify(context));
    EXPECT_TRUE(mode_trust_verify.IsTrustVerify(context));
    EXPECT_TRUE(mode_trust_verify_full.IsTrustVerify(context));
}
#else
TEST_F(CPU_FindControls_NONE, FindModeIsTrustVerifyDisabled)
{
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_FALSE(mode_normal.IsTrustVerify(context));
    EXPECT_FALSE(mode_fast.IsTrustVerify(context));
    EXPECT_FALSE(mode_hybrid.IsTrustVerify(context));
    EXPECT_FALSE(mode_dynamic_hybrid.IsTrustVerify(context));
    EXPECT_FALSE(mode_trust_verify.IsTrustVerify(context));
    EXPECT_FALSE(mode_trust_verify_full.IsTrustVerify(context));
}
#endif

TEST_F(CPU_FindControls_NONE, FindModeIsExhaustive)
{
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_FALSE(mode_normal.IsExhaustive(context));
    EXPECT_FALSE(mode_fast.IsExhaustive(context));
    EXPECT_FALSE(mode_hybrid.IsExhaustive(context));
    EXPECT_FALSE(mode_dynamic_hybrid.IsExhaustive(context));
    EXPECT_FALSE(mode_trust_verify.IsExhaustive(context));
    EXPECT_TRUE(mode_trust_verify_full.IsExhaustive(context));
}

// Test GetSet functionality
TEST_F(CPU_FindControls_NONE, FindModeGetSet)
{
    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_EQ(mode_normal.Get(), miopen::FindMode::Values::Normal);
    EXPECT_EQ(mode_fast.Get(), miopen::FindMode::Values::Fast);
    EXPECT_EQ(mode_hybrid.Get(), miopen::FindMode::Values::Hybrid);
    EXPECT_EQ(mode_dynamic_hybrid.Get(), miopen::FindMode::Values::DynamicHybrid);
    EXPECT_EQ(mode_trust_verify.Get(), miopen::FindMode::Values::TrustVerify);
    EXPECT_EQ(mode_trust_verify_full.Get(), miopen::FindMode::Values::TrustVerifyFull);
}

// Test constructor
TEST_F(CPU_FindControls_NONE, FindModeConstructor)
{
    // Test default constructor
    miopen::FindMode mode_default;
    // Just ensure it constructs without crashing
}

// Test edge cases
TEST_F(CPU_FindControls_NONE, FindModeEdgeCases)
{
    // Test deprecated fast hybrid mode
    miopen::FindMode mode;
    mode.Set(miopen::FindMode::Values::DeprecatedFastHybrid);

    // DeprecatedFastHybrid should not be considered as Fast or Hybrid in current implementation
    EXPECT_FALSE(mode.IsFast(context));
    EXPECT_FALSE(mode.IsHybrid(context));
}

// Test that stream operators work (compilation test)
TEST_F(CPU_FindControls_NONE, StreamOperators)
{
    miopen::FindEnforce enforce(miopen::FindEnforceAction::Search);
    miopen::FindMode mode;

    std::ostringstream oss1, oss2;
    oss1 << enforce;
    oss2 << mode;

    // Just ensure they don't crash and produce some output
    EXPECT_FALSE(oss1.str().empty());
    EXPECT_FALSE(oss2.str().empty());
}

// Since we can't test environment variable combinations directly,
// test the constructor behavior with whatever environment is set
TEST_F(CPU_FindControls_NONE, ConstructorReadsEnvironment)
{
    // Test that constructors work and read some value
    miopen::FindMode mode_conv; // Default convolution

    // Just verify they construct and have valid enum values
    EXPECT_GE(static_cast<int>(mode_conv.Get()),
              static_cast<int>(miopen::FindMode::Values::Begin_));
    EXPECT_LT(static_cast<int>(mode_conv.Get()), static_cast<int>(miopen::FindMode::Values::End_));
}

// Test the default constructor behavior
TEST_F(CPU_FindControls_NONE, DefaultConstructorBehavior)
{
    miopen::FindEnforce default_enforce; // Reads from environment

    // Verify it constructed and has a valid action value
    EXPECT_GE(static_cast<int>(default_enforce.GetAction()),
              static_cast<int>(miopen::FindEnforceAction::First_));
    EXPECT_LE(static_cast<int>(default_enforce.GetAction()),
              static_cast<int>(miopen::FindEnforceAction::Last_));
}

// Test current environment state (informational)
TEST_F(CPU_FindControls_NONE, CurrentEnvironmentState)
{
    const char* mode_env    = std::getenv("MIOPEN_FIND_MODE");
    const char* enforce_env = std::getenv("MIOPEN_FIND_ENFORCE");

    // This is informational - shows what environment the tests are running with
    if(mode_env != nullptr)
    {
        std::cout << "Test running with MIOPEN_FIND_MODE=" << mode_env << std::endl;
    }
    if(enforce_env != nullptr)
    {
        std::cout << "Test running with MIOPEN_FIND_ENFORCE=" << enforce_env << std::endl;
    }

    // Test what the constructors actually produce with current environment
    miopen::FindMode mode;
    miopen::FindEnforce enforce;

    std::cout << "Constructor produced FindMode: " << static_cast<int>(mode.Get()) << std::endl;
    std::cout << "Constructor produced FindEnforce: " << static_cast<int>(enforce.GetAction())
              << std::endl;

    // Basic sanity checks
    EXPECT_GE(static_cast<int>(mode.Get()), 1);
    EXPECT_LT(static_cast<int>(mode.Get()), 8);
    EXPECT_GE(static_cast<int>(enforce.GetAction()), 1);
    EXPECT_LE(static_cast<int>(enforce.GetAction()), 5);
}

TEST_F(CPU_FindControls_NONE, IsValidCombination_Combinations)
{
    miopen::FindEnforce enforce_none(miopen::FindEnforceAction::None);
    miopen::FindEnforce enforce_db_update(miopen::FindEnforceAction::DbUpdate);
    miopen::FindEnforce enforce_search(miopen::FindEnforceAction::Search);
    miopen::FindEnforce enforce_search_db_update(miopen::FindEnforceAction::SearchDbUpdate);
    miopen::FindEnforce enforce_clean(miopen::FindEnforceAction::DbClean);

    miopen::FindMode mode_normal;
    mode_normal.Set(miopen::FindMode::Values::Normal);
    miopen::FindMode mode_fast;
    mode_fast.Set(miopen::FindMode::Values::Fast);
    miopen::FindMode mode_hybrid;
    mode_hybrid.Set(miopen::FindMode::Values::Hybrid);
    miopen::FindMode mode_dynamic_hybrid;
    mode_dynamic_hybrid.Set(miopen::FindMode::Values::DynamicHybrid);
    miopen::FindMode mode_trust_verify;
    mode_trust_verify.Set(miopen::FindMode::Values::TrustVerify);
    miopen::FindMode mode_trust_verify_full;
    mode_trust_verify_full.Set(miopen::FindMode::Values::TrustVerifyFull);

    EXPECT_TRUE(IsValidCombination(enforce_none, mode_normal));
    EXPECT_TRUE(IsValidCombination(enforce_db_update, mode_normal));
    EXPECT_TRUE(IsValidCombination(enforce_search, mode_normal));
    EXPECT_TRUE(IsValidCombination(enforce_search_db_update, mode_normal));
    EXPECT_TRUE(IsValidCombination(enforce_clean, mode_normal));

    EXPECT_TRUE(IsValidCombination(enforce_none, mode_fast));
    EXPECT_FALSE(IsValidCombination(enforce_db_update, mode_fast));
    EXPECT_TRUE(IsValidCombination(enforce_search, mode_fast));
    EXPECT_FALSE(IsValidCombination(enforce_search_db_update, mode_fast));
    EXPECT_FALSE(IsValidCombination(enforce_clean, mode_fast));

    EXPECT_TRUE(IsValidCombination(enforce_none, mode_hybrid));
    EXPECT_FALSE(IsValidCombination(enforce_db_update, mode_hybrid));
    EXPECT_TRUE(IsValidCombination(enforce_search, mode_hybrid));
    EXPECT_FALSE(IsValidCombination(enforce_search_db_update, mode_hybrid));
    EXPECT_FALSE(IsValidCombination(enforce_clean, mode_hybrid));

    EXPECT_TRUE(IsValidCombination(enforce_none, mode_dynamic_hybrid));
    EXPECT_FALSE(IsValidCombination(enforce_db_update, mode_dynamic_hybrid));
    EXPECT_TRUE(IsValidCombination(enforce_search, mode_dynamic_hybrid));
    EXPECT_FALSE(IsValidCombination(enforce_search_db_update, mode_dynamic_hybrid));
    EXPECT_FALSE(IsValidCombination(enforce_clean, mode_dynamic_hybrid));

    EXPECT_TRUE(IsValidCombination(enforce_none, mode_trust_verify));
    EXPECT_FALSE(IsValidCombination(enforce_db_update, mode_trust_verify));
    EXPECT_TRUE(IsValidCombination(enforce_search, mode_trust_verify));
    EXPECT_FALSE(IsValidCombination(enforce_search_db_update, mode_trust_verify));
    EXPECT_FALSE(IsValidCombination(enforce_clean, mode_trust_verify));

    EXPECT_TRUE(IsValidCombination(enforce_none, mode_trust_verify_full));
    EXPECT_FALSE(IsValidCombination(enforce_db_update, mode_trust_verify_full));
    EXPECT_TRUE(IsValidCombination(enforce_search, mode_trust_verify_full));
    EXPECT_FALSE(IsValidCombination(enforce_search_db_update, mode_trust_verify_full));
    EXPECT_FALSE(IsValidCombination(enforce_clean, mode_trust_verify_full));
}
