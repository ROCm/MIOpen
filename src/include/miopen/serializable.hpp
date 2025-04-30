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

#ifndef GUARD_MLOPEN_SERIALIZABLE_HPP
#define GUARD_MLOPEN_SERIALIZABLE_HPP

#include <ciso646>
#include <miopen/config.h>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>

namespace miopen {
namespace solver {
namespace serialize {

template <class T>
struct Parse
{
    static bool apply(const std::string& s, T& result)
    {
        std::stringstream ss;
        ss.str(s);
        ss >> result;
        return true;
    }
};

template <char Separator = ','>
struct SerDes
{
private:
    struct SerializeField
    {
        template <class T>
        void operator()(std::ostream& stream, char& sep, const T& x) const
        {
            if(sep != 0)
                stream << sep;
            stream << x;
            sep = Separator;
        }
    };

    struct DeserializeField
    {
        template <class T>
        void operator()(bool& ok, std::istream& stream, T& x) const
        {
            if(not ok)
                return;
            std::string part;

            if(!std::getline(stream, part, Separator))
            {
                ok = false;
                return;
            }

            ok = Parse<T>::apply(part, x);
        }
    };

public:
    template <class Self>
    static void Serialize(const Self& self, std::ostream& stream)
    {
        char sep = 0;
        Self::Visit(
            self,
            std::bind(SerializeField{}, std::ref(stream), std::ref(sep), std::placeholders::_1));
    }

    template <class Self>
    static bool Deserialize(Self& self, const std::string& s)
    {
        auto out = self;
        bool ok  = true;
        std::istringstream ss(s);
        Self::Visit(
            out, std::bind(DeserializeField{}, std::ref(ok), std::ref(ss), std::placeholders::_1));

        if(!ok)
            return false;

        self = out;
        return true;
    }
};

} // namespace serialize

template <class Derived>
struct Serializable
{
    void Serialize(std::ostream& stream) const
    {
        serialize::SerDes<>::Serialize(static_cast<const Derived&>(*this), stream);
    }

    bool Deserialize(const std::string& s)
    {
        return serialize::SerDes<>::Deserialize(static_cast<Derived&>(*this), s);
    }
};

} // namespace solver
} // namespace miopen

#endif
