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

#pragma once

#include <miopen/errors.hpp>
#include <miopen/type_traits.hpp>

#include <cstring>
#include <type_traits>

namespace miopen {

namespace detail {
class BinarySerializationRelatedStream
{
protected:
    template <class Type, class = void>
    struct SerializationCriteria : std::false_type
    {
    };

    template <class Type>
    struct SerializationCriteria<
        Type,
        std::enable_if_t<std::is_pod<Type>{} && !std::is_pointer<Type>{}, void>> : std::true_type
    {
    };
};
} // namespace detail

class BinarySerializationSizeStream final : detail::BinarySerializationRelatedStream
{
public:
    bool IsSerializing() const { return false; }
    bool IsDeserializing() const { return false; }

    template <class Type, std::enable_if_t<SerializationCriteria<Type>{}, bool> = true>
    BinarySerializationSizeStream& operator<<(const Type&)
    {
        size += sizeof(Type);
        return *this;
    }

    std::size_t GetSize() const { return size; }

private:
    std::size_t size;
};

class BinarySerializationStream final : detail::BinarySerializationRelatedStream
{
public:
    bool IsSerializing() const { return true; }
    bool IsDeserializing() const { return false; }

    BinarySerializationStream(char* buffer) : position(buffer) {}

    template <class Type, std::enable_if_t<SerializationCriteria<Type>{}, bool> = true>
    BinarySerializationStream& operator<<(const Type& value)
    {
        std::memcpy(position, &value, sizeof(Type));
        position += sizeof(Type);
        return *this;
    }

private:
    char* position;
};

class BinaryDeserializationStream final : detail::BinarySerializationRelatedStream
{
public:
    bool IsSerializing() const { return false; }
    bool IsDeserializing() const { return true; }

    BinaryDeserializationStream(const char* begin, const char* end_) : position(begin), end(end_) {}

    template <class Type,
              std::enable_if_t<std::is_pod<Type>{} && !std::is_pointer<Type>{}, bool> = true>
    BinaryDeserializationStream& operator<<(Type& value)
    {
        if(position + sizeof(Type) > end)
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Deserialization buffer is incomplete. " +
                             std::to_string(end - position - sizeof(Type)) + " bytes missing.");
        value = *reinterpret_cast<const Type*>(position);
        position += sizeof(Type);
        return *this;
    }

    bool HasFinished() const { return position == end; }

private:
    const char* position;
    const char* end;
};

template <class Stream>
using IsBinarySerializationRelated =
    std::is_base_of<detail::BinarySerializationRelatedStream, Stream>;

template <class Stream,
          class First,
          class Second,
          std::enable_if_t<IsBinarySerializationRelated<Stream>{}, bool> = true>
Stream& operator<<(Stream& stream, std::pair<First, Second>& pair)
{
    stream << pair.first;
    stream << pair.second;
    return stream;
}

namespace detail {

template <class Type>
using size_method_t = decltype(std::declval<Type>().size());

template <class Type>
using resize_method_t = decltype(std::declval<Type>().resize(0));

template <class Type>
using int_index_member_t = decltype(std::declval<Type>()[0]);

template <class Type>
constexpr bool IsLikeVector =
    HasMember<detail::size_method_t, Type>{} && HasMember<detail::resize_method_t, Type>{} &&
    HasMember<detail::int_index_member_t, Type>{};

} // namespace detail

template <class Stream,
          class Type,
          std::enable_if_t<IsBinarySerializationRelated<Stream>{}, bool> = true,
          std::enable_if_t<detail::IsLikeVector<Type>, bool>             = true>
Stream& operator<<(Stream& stream, Type& value)
{
    auto size = value.size();
    stream << size;
    value.resize(size);

    for(auto i = 0; i < size; ++i)
        stream << value[i];

    return stream;
}

namespace detail {

template <class Type>
using reserve_method_t = decltype(std::declval<Type>().reserve(std::declval<std::size_t>()));

template <class Type>
using empty_t = decltype(std::declval<Type>().empty());

template <class Type, class... Parameters>
using emplace_t = decltype(std::declval<Type>().emplace(std::declval<Parameters>()...));

template <template <class...> class Type, class Key, class Value>
constexpr bool IsLikeMap =
    HasMember<detail::empty_t, Type<Key, Value>>{} &&
    HasMember<detail::reserve_method_t, Type<Key, Value>>{} &&
    HasMember<detail::emplace_t, Type<Key, Value>, std::pair<Key, Value>>{} &&
    !IsLikeVector<Type<Key, Value>>;

} // namespace detail

template <class Stream,
          template <class...>
          class Map,
          class Key,
          class Value,
          std::enable_if_t<IsBinarySerializationRelated<Stream>{}, bool> = true,
          std::enable_if_t<detail::IsLikeMap<Map, Key, Value>, bool>     = true>
Stream& operator<<(Stream& stream, Map<Key, Value>& value)
{
    auto size = value.size();
    stream << size;

    if(stream.IsDeserializing())
    {
        value.empty();
        value.reserve(size);
    }

    auto it = value.begin();
    for(auto i = 0; i < size; ++i)
    {
        std::pair<Key, Value> pair;
        if(!stream.IsDeserializing())
            pair = *it;

        stream << pair;

        if(stream.IsDeserializing())
            value.emplace(std::move(pair));
        else
            ++it;
    }

    return stream;
}

} // namespace miopen
