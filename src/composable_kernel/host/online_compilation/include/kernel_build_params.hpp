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
#ifndef GUARD_OLC_KERNEL_BUILD_PARAMETERS_HPP_
#define GUARD_OLC_KERNEL_BUILD_PARAMETERS_HPP_

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <string>
#include <vector>

namespace olCompile {

namespace kbp {
struct Option
{
};
} // namespace kbp

enum class ParameterTypes
{
    Define,
    Option,
};

struct KernelBuildParameter
{
    ParameterTypes type;
    std::string name;
    std::string value;
};

class KernelBuildParameters
{
    public:
    struct KBPInit
    {
        friend class KernelBuildParameters;

        KBPInit(const std::string& name, const std::string& value = "")
            : data{ParameterTypes::Define, name, value}
        {
        }

        template <class TValue, class = decltype(std::to_string(std::declval<TValue>()))>
        KBPInit(const std::string& name, const TValue& value) : KBPInit(name, std::to_string(value))
        {
        }

        KBPInit(kbp::Option, const std::string& name, const std::string& value = "")
            : data{ParameterTypes::Option, name, value}
        {
        }

        template <class TValue, class = decltype(std::to_string(std::declval<TValue>()))>
        KBPInit(kbp::Option, const std::string& name, const TValue& value)
            : KBPInit(kbp::Option{}, name, std::to_string(value))
        {
        }

        private:
        KernelBuildParameter data{};
    };

    KernelBuildParameters() = default;
    KernelBuildParameters(const std::initializer_list<KBPInit>& defines_)
    {
        options.reserve(defines_.size());
        for(const auto& define : defines_)
        {
            assert(ValidateUniqueness(define.data.name));
            options.push_back(define.data);
        }
    }

    bool Empty() const { return options.empty(); }

    void Define(const std::string& name, const std::string& value = "")
    {
        assert(ValidateUniqueness(name));
        options.push_back({ParameterTypes::Define, name, value});
    }

    template <class TValue, class = decltype(std::to_string(std::declval<TValue>()))>
    void Define(const std::string& name, const TValue& value)
    {
        Define(name, std::to_string(value));
    }

    KernelBuildParameters& operator<<(const KernelBuildParameters& other)
    {
        std::copy(other.options.begin(), other.options.end(), std::back_inserter(options));
        return *this;
    }

    template <class TFor>
    std::string GenerateFor(TFor&&) const
    {
        return TFor::Generate(options);
    }

    private:
    std::vector<KernelBuildParameter> options = {};

    bool ValidateUniqueness(const std::string& name) const
    {
        const auto eq = [=](const auto& item) { return item.name == name; };
        return std::find_if(options.begin(), options.end(), eq) == options.end();
    }
};

} // namespace olCompile

#endif
