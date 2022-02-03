/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/errors.hpp>

#include <memory>
#include <typeinfo>
#include <type_traits>
#include <utility>

namespace miopen {

enum class InvokeType
{
    Run,
    Evaluate,
    AutoTune,
};

struct InvokeParams
{
    InvokeType type = InvokeType::Run;
};

struct AnyInvokeParams
{
public:
    AnyInvokeParams() = default;

    template <
        class Actual,
        class = std::enable_if_t<
            !std::is_same<std::remove_reference_t<std::remove_const_t<Actual>>, AnyInvokeParams>{},
            void>>
    AnyInvokeParams(Actual value)
        : impl(std::make_unique<
               Implementation<std::remove_reference_t<std::remove_const_t<Actual>>>>(value))
    {
    }

    AnyInvokeParams(const AnyInvokeParams& other) : impl(other.impl ? other.impl->Copy() : nullptr)
    {
    }

    AnyInvokeParams(AnyInvokeParams&& other) noexcept = default;

    AnyInvokeParams& operator=(AnyInvokeParams other)
    {
        impl.swap(other.impl);
        return *this;
    }

    void SetInvokeType(InvokeType type)
    {
        if(!impl)
            MIOPEN_THROW("Attempt to use empty AnyInvokeParams.");
        impl->SetInvokeType(type);
    }

    InvokeType GetInvokeType() const
    {
        if(!impl)
            MIOPEN_THROW("Attempt to use empty AnyInvokeParams.");
        return impl->GetInvokeType();
    }

    template <class Actual>
    const std::remove_cv_t<Actual>& CastTo() const
    {
        if(!impl)
            MIOPEN_THROW("Attempt to use empty AnyInvokeParams.");
        if(!impl->CanCastTo(typeid(Actual)))
            MIOPEN_THROW("Attempt to cast AnyInvokeParams to invalid type.");
        return *reinterpret_cast<const std::remove_cv_t<Actual>*>(impl->GetRawPtr());
    }

    template <class Actual>
    Actual& CastTo()
    {
        if(!impl)
            MIOPEN_THROW("Attempt to use empty AnyInvokeParams.");
        if(!impl->CanCastTo(typeid(Actual)))
            MIOPEN_THROW("Attempt to cast AnyInvokeParams to invalid type.");
        return *reinterpret_cast<Actual*>(impl->GetRawPtr());
    }

    operator bool() const { return impl != nullptr; }

private:
    struct Interface
    {
    public:
        Interface(const Interface&) = delete;
        Interface(Interface&&)      = delete;
        Interface& operator=(const Interface&) = delete;
        Interface& operator=(Interface&&) = delete;

        virtual ~Interface(){};

        virtual void SetInvokeType(InvokeType type)         = 0;
        virtual InvokeType GetInvokeType() const            = 0;
        virtual bool CanCastTo(const std::type_info&) const = 0;
        virtual void* GetRawPtr()                           = 0;
        virtual std::unique_ptr<Interface> Copy() const     = 0;

    protected:
        Interface() = default;
    };

    template <class Actual>
    struct Implementation : public Interface
    {
    public:
        Implementation(const Actual& actual) : value(actual) {}
        Implementation(Actual&& actual) : value(std::move(actual)) {}

        void SetInvokeType(InvokeType type) override { value.type = type; }
        InvokeType GetInvokeType() const override { return value.type; }
        bool CanCastTo(const std::type_info& type) const override { return typeid(Actual) == type; }
        void* GetRawPtr() override { return &value; }

        std::unique_ptr<Interface> Copy() const override
        {
            return std::make_unique<Implementation<Actual>>(value);
        }

    private:
        Actual value;
    };

    std::unique_ptr<Interface> impl;
};

} // namespace miopen
