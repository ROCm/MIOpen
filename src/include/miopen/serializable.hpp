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

namespace miopen {
namespace solver {

// TODO: Provide specialization to use stro* functions
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

template <class Derived, char Seperator = ','>
struct Serializable
{
    struct SerializeField
    {
        template <class T>
        void operator()(std::ostream& stream, char& sep, const T& x) const
        {
            if(sep != 0)
                stream << sep;
            stream << x;
            sep = Seperator;
        }
    };

    struct DeserializeField
    {
        template <class T>
        void operator()(bool& ok, std::istream& stream, char sep, T& x) const
        {
            if(not ok)
                return;
            std::string part;

            if(!std::getline(stream, part, sep))
            {
                ok = false;
                return;
            }

            ok = Parse<T>::apply(part, x);
        }
    };
    void Serialize(std::ostream& stream) const
    {
        char sep = 0;
        Derived::Visit(
            static_cast<const Derived&>(*this),
            std::bind(SerializeField{}, std::ref(stream), std::ref(sep), std::placeholders::_1));
    }

    bool Deserialize(const std::string& s)
    {
        auto out = static_cast<const Derived&>(*this);
        bool ok  = true;
        std::istringstream ss(s);
        Derived::Visit(
            out,
            std::bind(
                DeserializeField{}, std::ref(ok), std::ref(ss), Seperator, std::placeholders::_1));

        if(!ok)
            return false;

        static_cast<Derived&>(*this) = out;
        return true;
    }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    bool LegacyDeserialize(const std::string&) { return false; }
#endif
    friend std::ostream& operator<<(std::ostream& os, const Derived& c)
    {
        c.Serialize(os);
        return os;
    }
};

} // namespace solver
} // namespace miopen

#endif
