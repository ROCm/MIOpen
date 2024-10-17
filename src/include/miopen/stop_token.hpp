/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#pragma once

#ifdef __cpp_lib_jthread

#include <stop_token>

namespace miopen {
using nostopstate_t                        = std::nostopstate_t;
constexpr inline nostopstate_t nostopstate = std::nostopstate;
using stop_token                           = std::stop_token;
using stop_source                          = std::stop_source;
} // namespace miopen

#else

#include <atomic>
#include <memory>
#include <utility>

namespace miopen {
struct stop_state final
{
public:
    [[nodiscard]] bool stop_requested() const noexcept
    {
        return value.load(std::memory_order_relaxed);
    }

    bool request_stop() noexcept { return !value.exchange(true, std::memory_order_relaxed); }

private:
    std::atomic<bool> value;
};

struct nostopstate_t final
{
    explicit nostopstate_t() = default;
};

constexpr inline nostopstate_t nostopstate{};

class stop_source;

class stop_token final
{
public:
    friend class stop_source;

    stop_token() noexcept : state() {}
    stop_token(stop_token&& other) noexcept : state(std::exchange(other.state, {})) {}
    stop_token(stop_token const& other) noexcept = default;

    stop_token& operator=(stop_token&& other) noexcept
    {
        stop_token{std::move(other)}.swap(*this);
        return *this;
    }

    stop_token& operator=(stop_token const& other) noexcept
    {
        stop_token{other}.swap(*this);
        return *this;
    }

    [[nodiscard]] bool stop_requested() const noexcept
    {
        const auto locked = state.lock();
        return locked != nullptr && locked->stop_requested();
    }

    [[nodiscard]] bool stop_possible() const noexcept { return !state.expired(); }

    void swap(stop_token& other) noexcept { std::swap(state, other.state); }

private:
    std::weak_ptr<stop_state> state;

    explicit stop_token(std::weak_ptr<stop_state> state_) : state(std::move(state_)) {}
};

class stop_source final
{
public:
    stop_source() : state(std::make_shared<stop_state>()) {}
    stop_source(nostopstate_t) noexcept : state(nullptr) {}
    stop_source(stop_source&& other) noexcept : state(std::exchange(other.state, nullptr)) {}
    stop_source(stop_source const& other) noexcept = default;

    stop_source& operator=(stop_source&& other) noexcept
    {
        stop_source{std::move(other)}.swap(*this);
        return *this;
    }

    stop_source& operator=(stop_source const& other) noexcept
    {
        stop_source{other}.swap(*this);
        return *this;
    }

    [[nodiscard]] stop_token get_token() const noexcept { return stop_token{state}; }

    [[nodiscard]] bool stop_possible() const noexcept { return state != nullptr; }

    [[nodiscard]] bool stop_requested() const noexcept
    {
        return state != nullptr && state->stop_requested();
    }

    bool request_stop() noexcept { return state != nullptr && !state->request_stop(); }

    void swap(stop_source& other) noexcept { std::swap(state, other.state); }

private:
    std::shared_ptr<stop_state> state;
};
} // namespace miopen

#endif
