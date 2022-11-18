/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "test.hpp"
#include <miopen/handle.hpp>
#include <miopen/miopen.h>

struct allocator_fixture
{
    miopen::Handle h{};
    static const int size = 42;
    miopen::Allocator::ManageDataPtr buffer;
    allocator_fixture() { buffer = h.Create(size); }
};

struct test_allocator : allocator_fixture
{
    void run()
    {
        h.SetAllocator(
            +[](void*, std::size_t n) -> void* {
                CHECK(n == size);
                throw "Called allocator"; // NOLINT
            },
            nullptr,
            nullptr);
        miopen::Allocator::ManageDataPtr p = nullptr;
        CHECK(throws([&] { p = h.Create(size); }));
    }
};

struct test_null_allocator : allocator_fixture
{
    void run()
    {
        h.SetAllocator(
            +[](void*, std::size_t n) -> void* {
                CHECK(n == size);
                return nullptr;
            },
            nullptr,
            nullptr);
        miopen::Allocator::ManageDataPtr p = nullptr;
        CHECK(throws([&] { p = h.Create(size); }));
    }
};

struct test_deallocator : allocator_fixture
{
    void run()
    {
        h.SetAllocator(
            +[](void* ctx, std::size_t n) -> void* {
                CHECK(n == size);
                return reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx)->get();
            },
            +[](void* ctx, void* data) {
                auto b = reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx);
                CHECK(data == b->get());
                *b = nullptr;
            },
            &buffer);
        miopen::Allocator::ManageDataPtr p = h.Create(size);
        CHECK(p.get() == buffer.get());
        p = nullptr;
        CHECK(p == nullptr);
        CHECK(buffer == nullptr);
    }
};

struct test_deallocator2 : allocator_fixture
{
    void run()
    {
        h.SetAllocator(
            +[](void* ctx, std::size_t n) -> void* {
                CHECK(n == size);
                return reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx)->get();
            },
            +[](void* ctx, void* data) {
                auto b = reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx);
                CHECK(data == b->get());
            },
            &buffer);
        miopen::Allocator::ManageDataPtr p = h.Create(size);
        CHECK(p.get() == buffer.get());
        p = nullptr;
        CHECK(p == nullptr);
        CHECK(buffer != nullptr);
    }
};

int main()
{
    run_test<test_allocator>();
    run_test<test_null_allocator>();
    run_test<test_deallocator>();
    run_test<test_deallocator2>();
}
