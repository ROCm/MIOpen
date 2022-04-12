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

#ifndef GUARD_MIOPEN_SEQUENCES_HPP_
#define GUARD_MIOPEN_SEQUENCES_HPP_

#include <miopen/rank.hpp>

#include <boost/range/algorithm/find.hpp>

#include <algorithm>
#include <cassert>
#include <tuple>
#include <vector>

namespace miopen {
namespace seq {

/*
Sequence classes describe sets of unique oredered values.
To be a sequence class must follow the concept:

class Sequence
{
    using ValueType = <unspecified>;

    Sequence();

    // returns iterator pointint to the first element of the sequence.
    constexpr <unspecified> begin() const;

    // returns invalid iterator.
    constexpr <unspecified> end() const;

    // returns iterator pointing to element of sequence equal to value or end().
    // this method may not be implemented. boost::find will be used instead than.
    constexpr <unspecified> find(TValue value) const;
}

ValueType may have any constraints depending on specific sequences but it also must have operator==.
It is forbidden to use any of existing sequences with several equal values. It will lead to hangs.
Each of existing sequences which can produce such scenario has assert which will fail with a reason
of failure stated.
Example: sequence 1-2-1 will be looped as:

1-2-1 -> 1-2-1 -> 1-2-1 -> 1-2-1
|          |      |          |

and Next(...) will only return true because there is no way to distinguish if a value provided is
first or second 1. It may work in whatever way for user defined types and sequences but in common
case it simply won't work. It is quite obvious in case of Sequence<int, 1, 2, 1> but it may not be
with something more complex, even like Join<Span<int, 1, 6>, Span<int, 4, 8>>. But it will hang with
any use of Next(...) based loops so any real life like usage will show invalid work of Next(...).

RuleSet is main way to use the system. They represent a set of rules of iterating over a sequence of
value tupples. Affects all the fields speceified. Has several methods:
    bool IsIn(const Container& container) const - check if value of each field in container is in
        respective sequence.
    bool Next(Container& container) const - switch to the next permutation. Returns true when
        container passed had last value of respective sequence in each field. Uses Next methods of
        respective sequences.
    void FillBegin(Container& container) const - fills provided structure with begin() values
        of sequences.
    bool Compare(const Container& left, const Container& right) const - helper method comparing
        specified fields of provided values.
    bool IsEqualToBegin(const Container& container) const - helper method comparing specified
        fields of provided value with begin()s.

Here Container means type fullfiling member definitions. In most cases member definition will look
like &S::x, meaning that Container means S. There are no other constraints for it.

Simpliest way to define a RuleSet is by MakeRuleSet. Example:

    struct S { int x, y; };

    auto Rules()
    {
        return seq::MakeRuleSet(
            std::make_tuple(seq::Span<int, 0, 9>{}, &S::x),
            std::make_tuple(seq::Sequence<int, 1, 3, 7>{}, &S::y));
    }

    int TotalCombinations()
    {
        S s;
        int count = 0;
        Rules().FillBegin(s);

        while (!Rules().Next(s))
            count++;

        return count;
    }

So to make a ruleset you call MakeRuleSet and provide tuples as arguments.
Each tuple should contain a description of a sequence as first (#0) value and a pointer to member.
In the example the ruleset describes sequence with 0,1,2,3,4,5,6,7,8,9 as applicable values for
x field and 1,3,7 for y.

Predefined sequences:
Sequence - set of any values.
Span - span with lower and upper limits.
TwoPowersSpan - span of powers of 2.
Join - join of two or more sequences.
Multiplied - sequence with each member multipled by factor.

One may argue that it is bad to pass a pointer to member as run-time value and it should only be
passed as template argument (as it can be) but there are two main reasons explaining why it is
done like it is done.
1. Expression
    int TestData::*, &TestData::x
compared to
    &TestData::x
is longer and has unrequired and duplicated code.
2. A test was made to show that it has no if any impact on execution time. It is saved as
speedtests/sequences.cpp and may be muilt as target speedtest_sequences and run as
bin/speedtest_sequences --mode std
bin/speedtest_sequences --mode tmpl
for field and template argument stored pointers respectively.

Examples of real life usages may be found in:
    src/solver/conv_asm_1x1u.cpp
    src/solver/conv_asm_3x3u.cpp
    src/solver/conv_asm_dir_BwdWrW1x1.cpp
    src/solver/conv_asm_dir_BwdWrW3x3.cpp

Rule class is not required to use RuleSet via MakeRuleSet or sequences directly mostly it's like a
sequence but operating a specified field of a class and contains several helper methods.

MemberPtr and MakeMemberPtr are used to make a pointer to member for rules.
They are not needed for external usage of rulesets but in case of need it is easy to extend the
system to accept getters/setters or whatever.

Example of extending the system may be found at
    speedtests/sequences.cpp

Existing sequences in this file may be used as an example of creating a custom sequence.
Span and Multiplied are arguably the simpliest.
 */

template <class TRange, class TValue>
auto GenericFindImpl(rank<1>, const TRange& range, const TValue& value)
    -> decltype(range.find(value))
{
    return range.find(value);
}

template <class TRange, class TValue>
auto GenericFindImpl(rank<0>, const TRange& range, const TValue& value)
{
    return boost::find(range, value);
}

template <class TRange, class TValue>
auto GenericFind(const TRange& range, const TValue& value)
{
    return GenericFindImpl(rank<16>{}, range, value);
}

/// The simpliest of sequences provided. It contains int values supplied as template arguments.
template <class TValue, TValue... values>
struct Sequence
{
    using const_iterator = const TValue*;
    using ValueType      = TValue;

    Sequence() { assert(ValidateValues() && "Values must be unique"); }

    constexpr const_iterator begin() const { return data.begin(); }
    constexpr const_iterator end() const { return data.end(); }
    constexpr const_iterator find(const TValue& value) const { return data.data() + find_(value); }

    private:
    static constexpr std::array<int, sizeof...(values)> data = {{values...}};

    static constexpr int ValuesCount() { return sizeof...(values); }

    static constexpr bool ValidateValues()
    {
        for(auto i = 0; i < ValuesCount() - 1; ++i)
            for(auto j = i + 1; j < ValuesCount(); ++j)
                if(data[i] == data[j])
                    return false;

        return true;
    }

    template <int icur, TValue cur, TValue... rest>
    struct Find
    {
        int operator()(const TValue& value) const
        {
            if(value == cur)
                return icur;
            return rest_(value);
        }

        Find<icur + 1, rest...> rest_ = {};
    };

    template <int icur, TValue cur>
    struct Find<icur, cur>
    {
        int operator()(const TValue& value) const
        {
            if(value == cur)
                return icur;
            return icur + 1;
        }
    };

    Find<0, values...> find_ = {};
};

template <class TValue, TValue... values>
constexpr std::array<int, sizeof...(values)>
    Sequence<TValue, values...>::data; // Sometimes can't link without of this line

template <class TValue>
struct SequenceIteratorBase
{
    SequenceIteratorBase() = default;
    SequenceIteratorBase(TValue value_) : value(value_) {}

    TValue operator*() const { return value; }

    protected:
    TValue value = {};
};

template <class TValue, int high>
struct SpanIterator : public SequenceIteratorBase<TValue>
{
    SpanIterator() : SequenceIteratorBase<TValue>(high + 1) {}
    SpanIterator(TValue value_) : SequenceIteratorBase<TValue>(value_) {}

    SpanIterator& operator++()
    {
        this->value = this->value + 1;
        return *this;
    }

    const SpanIterator operator++(int) // NOLINT (readability-const-return-type)
    {
        const auto copy = *this;
        ++*this;
        return copy;
    }

    bool operator==(const SpanIterator& other) const
    {
        return this->value == other.value || (this->value > high && other.value > high);
    }

    bool operator!=(const SpanIterator& other) const { return !(*this == other); }
};

/// A sequence describing a span. It describes [low; high]. Type implementing + and == is required.
template <class TValue, TValue low, TValue high>
struct Span
{
    using const_iterator = SpanIterator<int, high>;
    using ValueType      = TValue;

    constexpr const_iterator begin() const { return {low}; }
    constexpr const_iterator end() const { return {}; }

    constexpr const_iterator find(TValue value) const
    {
        static_assert(low < high, "Span low limit should be actually lower than high.");
        if(value >= low && value <= high)
            return {value};
        return end();
    }
};

template <class TValue, TValue high>
struct TwoPowersSpanIterator : public SequenceIteratorBase<TValue>
{
    TwoPowersSpanIterator(TValue value_) : SequenceIteratorBase<TValue>(value_) {}

    TwoPowersSpanIterator& operator++()
    {
        this->value = this->value * 2;
        return *this;
    }

    const TwoPowersSpanIterator operator++(int) // NOLINT (readability-const-return-type)
    {
        const auto copy = *this;
        ++*this;
        return copy;
    }

    bool operator==(const TwoPowersSpanIterator& other) const
    {
        return this->value == other.value || (this->value > high && other.value > high);
    }

    bool operator!=(const TwoPowersSpanIterator& other) const { return !(*this == other); }
};

/// A sequence containing a span of powers of two. x in [low; high] and x = 2^n. Only integer types
/// are allowed.
template <class TValue, TValue low, TValue high>
struct TwoPowersSpan
{
    using const_iterator = TwoPowersSpanIterator<int, high>;
    using ValueType      = TValue;

    constexpr const_iterator begin() const { return {low}; }
    constexpr const_iterator end() const { return {high + 1}; }

    constexpr const_iterator find(TValue i) const
    {
        static_assert(low < high, "TwoPowersSpan low limit should be actually lower than high.");
        static_assert(IsTwoPower(low), "TwoPowersSpan low limit should be actually power of two.");
        static_assert(IsTwoPower(high),
                      "TwoPowersSpan high limit should be actually power of two.");

        if(IsTwoPower(i) && low <= i && i <= high)
            return {i};

        return end();
    }

    private:
    static constexpr bool IsTwoPower(TValue i) { return ((i - 1) & i) == 0; }
};

template <class TFirst, class... TRest>
struct JoinIterator : public SequenceIteratorBase<typename TFirst::ValueType>
{
    using ValueType = typename TFirst::ValueType;

    JoinIterator() : SequenceIteratorBase<ValueType>(0) {}
    JoinIterator(ValueType value_) : SequenceIteratorBase<ValueType>(value_), finished(false) {}

    JoinIterator& operator++()
    {
        finished = Next<TFirst, TRest...>{}(this->value);
        return *this;
    }

    const JoinIterator operator++(int) // NOLINT (readability-const-return-type)
    {
        const auto copy = *this;
        ++*this;
        return copy;
    }

    operator bool() const { return !finished; }

    bool operator==(const JoinIterator& other) const
    {
        return (this->value == other.value && *this && other) || (!*this && !other);
    }

    bool operator!=(const JoinIterator& other) const { return !(*this == other); }

    private:
    bool finished = true;

    template <class...>
    struct Next
    {
    };

    template <class TCur, class TNext, class... TRest_>
    struct Next<TCur, TNext, TRest_...>
    {
        bool operator()(ValueType& value)
        {
            auto it = GenericFind(cur, value);

            if(it == cur.end())
                return rest(value);

            if(++it == cur.end())
            {
                value = *next.begin();
                return false;
            }

            value = *it;
            return false;
        }

        private:
        TCur cur                    = {};
        TNext next                  = {};
        Next<TNext, TRest_...> rest = {};
    };

    template <class TSeq>
    struct Next<TSeq>
    {
        bool operator()(ValueType& value)
        {
            auto it = GenericFind(seq, value);

            assert(it != seq.end());
            if(++it == seq.end())
                return true;

            value = *it;
            return false;
        }

        private:
        TSeq seq = {};
    };
};

/// A union of one or more sequences.
template <class TFirst, class... TRest>
struct Join
{
    using ValueType      = typename TFirst::ValueType;
    using const_iterator = JoinIterator<TFirst, TRest...>;

    Join() { assert(Validate() && "Values must be unique"); }

    constexpr const_iterator begin() const { return {*first.begin()}; }
    constexpr const_iterator end() const { return {}; }
    constexpr const_iterator find(ValueType value) const { return find_(value); }

    private:
    template <class TCur, class... TRest_>
    struct Find
    {
        const_iterator operator()(ValueType value) const
        {
            auto it = cur(value);
            return it ? it : rest(value);
        }

        private:
        Find<TCur> cur       = {};
        Find<TRest_...> rest = {};
    };

    template <class TSeq>
    struct Find<TSeq>
    {
        const_iterator operator()(ValueType value) const
        {
            auto it = GenericFind(seq, value);

            if(it != seq.end())
                return {*it};

            return {};
        }

        private:
        TSeq seq = {};
    };

    bool Validate() const
    {
        auto cur = begin();
        std::vector<ValueType> values(1, *cur);

        while(++cur != end())
        {
            if(std::find(values.begin(), values.end(), *cur) != values.end())
                return false;

            values.push_back(*cur);
        }

        return true;
    }

    TFirst first                 = {};
    Find<TFirst, TRest...> find_ = {};
};

template <class TInner, typename TInner::ValueType mul>
struct MultipliedIterator
{
    using InnerIterator = typename TInner::const_iterator;
    using ValueType     = typename TInner::ValueType;

    MultipliedIterator(InnerIterator inner_) : inner(inner_) {}

    MultipliedIterator& operator++()
    {
        ++inner;
        return *this;
    }

    const MultipliedIterator operator++(int) // NOLINT (readability-const-return-type)
    {
        const auto copy = *this;
        ++*this;
        return copy;
    }

    ValueType operator*() const { return mul * *inner; }
    bool operator==(const MultipliedIterator& other) const { return inner == other.inner; }
    bool operator!=(const MultipliedIterator& other) const { return !(*this == other); }

    private:
    InnerIterator inner = {};
};

/// A sequence containing values of another sequence multiplied by a constant.
template <class TInner, typename TInner::ValueType mul>
struct Multiplied
{
    using ValueType      = typename TInner::ValueType;
    using const_iterator = MultipliedIterator<TInner, mul>;

    constexpr const_iterator begin() const { return {inner.begin()}; }
    constexpr const_iterator end() const { return {inner.end()}; }
    constexpr const_iterator find(ValueType value) const
    {
        return {GenericFind(inner, value / mul)};
    }

    private:
    TInner inner = {};
};

/// A common declaration for member pointer containing types. Implementations should have
/// a ctor accepting what they pass as MemberPtr template arguments.
template <class...>
struct MemberPtr
{
};

/// An explicit declaration for pointers to fields.
template <class TType, class TContainer>
struct MemberPtr<TType TContainer::*>
{
    using Container = TContainer;

    MemberPtr(TType TContainer::*field_) : field(field_) {}

    const TType& RV(const TContainer& cont) const { return cont.*field; }
    TType& LV(TContainer& cont) const { return cont.*field; }

    private:
    TType TContainer::*field;
};

/// A helper method for declaring MemberPtr's
template <class... TParams>
inline auto MakeMemberPtr(TParams... args)
{
    return MemberPtr<TParams...>(args...);
}

template <class TValue, TValue, TValue...>
struct SeqNextImpl_Sequence
{
};

template <class TValue, TValue first, TValue cur, TValue next, TValue... values>
struct SeqNextImpl_Sequence<TValue, first, cur, next, values...>
{
    bool operator()(TValue& value) const
    {
        if(value == cur)
        {
            value = next;
            return false;
        }

        return rest(value);
    }

    private:
    SeqNextImpl_Sequence<TValue, first, next, values...> rest = {};
};

template <class TValue, TValue first, TValue cur>
struct SeqNextImpl_Sequence<TValue, first, cur>
{
    bool operator()(TValue& value) const
    {
        assert(value == cur);
        value = first;
        return true;
    }
};

template <class TValue, TValue first, TValue... values>
bool SeqNextImpl(rank<1>, const Sequence<TValue, first, values...>&, TValue& value)
{
    return SeqNextImpl_Sequence<TValue, first, first, values...>{}(value);
}

template <class TSequence>
bool SeqNextImpl(rank<0>, const TSequence& seq, typename TSequence::ValueType& value)
{
    auto it = GenericFind(seq, value);

    assert(it != seq.end());
    if(++it == seq.end())
    {
        value = *seq.begin();
        return true;
    }

    value = *it;
    return false;
}

template <class TSequence>
bool SeqNext(const TSequence& seq, typename TSequence::ValueType& value)
{
    return SeqNextImpl(rank<16>{}, seq, value);
}

/// A rule applying a sequence to a specified member of a specified class instance provided.
/// Also contains Compare and IsEqualToBegin helper methods.
template <class TMember, class TSequence>
struct Rule
{
    using Container = typename TMember::Container;

    Rule(const TMember& member_) : member(member_) {}

    bool IsIn(const Container& container) const
    {
        return GenericFind(sequence, member.RV(container)) != sequence.end();
    }

    bool Next(Container& container) const { return SeqNext(sequence, member.LV(container)); }

    void FillBegin(Container& container) const { member.LV(container) = *sequence.begin(); }

    /// Compares provided members of two values.
    bool Compare(const Container& left, const Container& right) const
    {
        return member.RV(left) == member.RV(right);
    }

    /// Check if value provided is equal to the begin value of the sequence.
    bool IsEqualToBegin(const Container& container) const
    {
        return member.RV(container) == *sequence.begin();
    }

    private:
    TMember member;
    TSequence sequence = {};
};

/// Rule creation helper.
template <class TMember, class TSequence>
inline auto MakeRule(const TMember& member, TSequence)
{
    return Rule<TMember, TSequence>(member);
}

/// A type describing a set of rules and applying them to a provided value.
/// Rules must be of type Rule<...>.
/// Also provides Compare and IsEqualToBegin helper methods.
template <class TFirst, class... TRules>
struct RuleSet
{
    using Container = typename TFirst::Container;

    RuleSet(TFirst first, TRules... rules) : impl(first, rules...) {}

    bool IsIn(const Container& container) const { return impl.IsIn(container); }

    bool Next(Container& container) const { return impl.Next(container); }

    void FillBegin(Container& container) const { impl.FillBegin(container); }

    /// Compares all the fields specified in rules.
    bool Compare(const Container& left, const Container& right) const
    {
        return impl.Compare(left, right);
    }

    /// Compares all the fields specified in rules to appropriate begin() values.
    bool IsEqualToBegin(const Container& container) const { return impl.IsEqualToBegin(container); }

    private:
    template <class...>
    struct Impl
    {
    };

    template <class TRule, class... TRest>
    struct Impl<TRule, TRest...>
    {
        Impl(const TRule& rule_, TRest... rest_) : rule(rule_), rest(rest_...) {}

        bool IsIn(const Container& container) const
        {
            return rule.IsIn(container) && rest.IsIn(container);
        }

        bool Next(Container& container) const
        {
            if(!rule.Next(container))
                return false;

            return rest.Next(container);
        }

        void FillBegin(Container& container) const
        {
            rule.FillBegin(container);
            rest.FillBegin(container);
        }

        bool Compare(const Container& left, const Container& right) const
        {
            return rule.Compare(left, right) && rest.Compare(left, right);
        }

        bool IsEqualToBegin(const Container& container) const
        {
            return rule.IsEqualToBegin(container) && rest.IsEqualToBegin(container);
        }

        private:
        TRule rule;
        Impl<TRest...> rest;
    };

    template <class TRule>
    struct Impl<TRule>
    {
        Impl(const TRule& rule_) : rule(rule_) {}

        bool IsIn(const Container& container) const { return rule.IsIn(container); }

        bool Next(Container& container) const { return rule.Next(container); }

        void FillBegin(Container& container) const { rule.FillBegin(container); }

        bool Compare(const Container& left, const Container& right) const
        {
            return rule.Compare(left, right);
        }

        bool IsEqualToBegin(const Container& container) const
        {
            return rule.IsEqualToBegin(container);
        }

        private:
        TRule rule;
    };

    Impl<TFirst, TRules...> impl;
};

namespace detail {
template <class TTuple, std::size_t first, std::size_t... indexes>
auto MakeMemberPtrFromTuple(const TTuple& tuple, std::index_sequence<first, indexes...>)
{
    return MakeMemberPtr(std::get<indexes>(tuple)...);
}

/// A function making MemberPtr from tuples from MakeRuleSet.
template <class TTuple>
auto MakeMemberPtrFromTuple(const TTuple& tuple)
{
    return MakeMemberPtrFromTuple(tuple,
                                  std::make_index_sequence<std::tuple_size<TTuple>::value>{});
}

/// A type making a rule type from a tuple passed to MakeRuleSet.
template <class TTuple>
struct RuleFromTuple
{
    private:
    using Sequence      = typename std::tuple_element<0, TTuple>::type;
    using MemberPtrType = decltype(MakeMemberPtrFromTuple(std::declval<TTuple>()));

    public:
    using Type = Rule<MemberPtrType, Sequence>;
};
} // namespace detail

template <class... TTuples>
auto MakeRuleSet(TTuples... rules)
{
    return RuleSet<typename detail::RuleFromTuple<TTuples>::Type...>(
        MakeRule(detail::MakeMemberPtrFromTuple(rules), std::get<0>(rules))...);
}

} // namespace seq
} // namespace miopen

#endif // GUARD_MIOPEN_SEQUENCES_HPP_
