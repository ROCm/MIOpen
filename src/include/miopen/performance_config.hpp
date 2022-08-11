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
#include <miopen/serializable.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <functional>

namespace miopen {
namespace solver {

struct PerfConfig
{
    virtual ~PerfConfig() = default;

    virtual void Serialize(std::ostream& stream) const = 0;
    virtual bool Deserialize(const std::string& s)     = 0;
    virtual std::string ToString() const;
};

std::ostream& operator<<(std::ostream& os, const PerfConfig& c);

template <class Derived>
struct PerfConfigBase : PerfConfig
{
    void Serialize(std::ostream& stream) const final
    {
        SerDes<>::Serialize(static_cast<const Derived&>(*this), stream);
    }

    bool Deserialize(const std::string& s) final
    {
        return SerDes<>::Deserialize(static_cast<Derived&>(*this), s);
    }
};

} // namespace solver
} // namespace miopen

#endif // PERFORMANCE_CONFIG_HPP
