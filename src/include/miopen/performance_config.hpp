/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#ifndef PERFORMANCE_CONFIG_HPP
#define PERFORMANCE_CONFIG_HPP

#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/serializable.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <functional>

namespace miopen {
namespace solver {

struct PerfConfig : Serializable<PerfConfig>
{
    virtual std::string ToString() const;

    template <class Self, class F>
    static void Visit(Self&&, F)
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }
};

template <class Derived>
struct PerfConfigBase : PerfConfig
{
    void Serialize(std::ostream& stream) const final
    {
        char sep = 0;
        Derived::Visit(
            static_cast<const Derived&>(*this),
            std::bind(SerializeField{}, std::ref(stream), std::ref(sep), std::placeholders::_1));
    }

    bool Deserialize(const std::string& s) final
    {
        auto out = static_cast<const Derived&>(*this);
        bool ok  = true;
        std::istringstream ss(s);
        Derived::Visit(
            out,
            std::bind(
                DeserializeField{}, std::ref(ok), std::ref(ss), std::placeholders::_1));

        if(!ok)
            return false;

        static_cast<Derived&>(*this) = out;
        return true;
    }
};

} // namespace solver
} // namespace miopen

#endif // PERFORMANCE_CONFIG_HPP
