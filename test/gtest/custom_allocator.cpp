// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test.hpp"
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

struct GPU_Allocator_FP32 : public ::testing::TestWithParam<int>
{
    void SetUp() override { buffer = h.Create(size); }

    miopen::Handle h{};
    static constexpr int size = 42;
    miopen::Allocator::ManageDataPtr buffer;
};

constexpr int GPU_Allocator_FP32::size;

} // namespace

TEST_P(GPU_Allocator_FP32, CustomAllocator)
{
    h.SetAllocator(
        +[](void*, std::size_t n) -> void* {
            EXPECT_EQ(n, size);
            throw "Called allocator"; // NOLINT
        },
        nullptr,
        nullptr);
    miopen::Allocator::ManageDataPtr p = nullptr;
    // NOLINTNEXTLINE (bugprone-assignment-in-if-condition)
    EXPECT_TRUE(throws([&]() { p = h.Create(size); }));
}

TEST_P(GPU_Allocator_FP32, NullAllocator)
{
    h.SetAllocator(
        +[](void*, std::size_t n) -> void* {
            EXPECT_EQ(n, size);
            return nullptr;
        },
        nullptr,
        nullptr);
    miopen::Allocator::ManageDataPtr p = nullptr;
    // NOLINTNEXTLINE (bugprone-assignment-in-if-condition)
    EXPECT_TRUE(throws([&]() { p = h.Create(size); }));
}

TEST_P(GPU_Allocator_FP32, Deallocator)
{
    h.SetAllocator(
        +[](void* ctx, std::size_t n) -> void* {
            EXPECT_EQ(n, size);
            return reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx)->get();
        },
        +[](void* ctx, void* data) {
            auto b = reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx);
            EXPECT_EQ(data, b->get());
            *b = nullptr;
        },
        &buffer);
    miopen::Allocator::ManageDataPtr p = h.Create(size);
    EXPECT_EQ(p.get(), buffer.get());
    p = nullptr;
    EXPECT_EQ(p, nullptr);
    EXPECT_EQ(buffer, nullptr);
}

TEST_P(GPU_Allocator_FP32, Deallocator2)
{
    h.SetAllocator(
        +[](void* ctx, std::size_t n) -> void* {
            EXPECT_EQ(n, size);
            return reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx)->get();
        },
        +[](void* ctx, void* data) {
            auto b = reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx);
            EXPECT_EQ(data, b->get());
        },
        &buffer);
    miopen::Allocator::ManageDataPtr p = h.Create(size);
    EXPECT_EQ(p.get(), buffer.get());
    p = nullptr;
    EXPECT_EQ(p, nullptr);
    EXPECT_NE(buffer, nullptr);
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Allocator_FP32, testing::ValuesIn({0}));
