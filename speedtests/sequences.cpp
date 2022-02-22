#include <miopen/rank.hpp>
#include <miopen/sequences.hpp>

#include <driver.hpp>

#include <boost/range/algorithm/find.hpp>
#include <boost/range/irange.hpp>

#include <array>
#include <chrono>
#include <iostream>

namespace miopen {
namespace seq {
struct TestData
{
    int x;
};

enum class Modes
{
    Standard,
    Template,
    Native,
    Boost,
    Unknown,
};

enum class Sequences
{
    Seq,
    Span,
    IRange,
    Join,
    Unknown,
};

enum class Instances
{
    Single,
    PerCall,
    Static,
    Unknown,
};

template <class TType, class TContainer, TType TContainer::*field>
struct MemberPtr<std::integral_constant<TType TContainer::*, field>>
{
    using Container = TContainer;

    MemberPtr(const std::integral_constant<TType TContainer::*, field>&) {}

    const TType& RV(const TContainer& cont) const { return cont.*field; }
    TType& LV(TContainer& cont) const { return cont.*field; }
};

template <Sequences seq>
struct NativeImpl
{
};

template <>
struct NativeImpl<Sequences::Seq>
{
    bool Next(TestData& v) const
    {
        if(v.x == 1)
        {
            v.x = 2;
            return false;
        }

        if(v.x == 2)
        {
            v.x = 7;
            return false;
        }

        v.x = 1;
        return true;
    }
};

template <>
struct NativeImpl<Sequences::Span>
{
    bool Next(TestData& v) const
    {
        if(v.x == 8)
        {
            v.x = 0;
            return true;
        }

        ++v.x;
        return false;
    }
};

template <>
struct NativeImpl<Sequences::IRange>
{
    bool Next(TestData&) const
    {
        std::cerr << "irange is only for boost" << std::endl;
        std::exit(-1); // NOLINT (concurrency-mt-unsafe)
    }
};

template <>
struct NativeImpl<Sequences::Join>
{
    bool Next(TestData& td) const
    {
        if(td.x < 8)
        {
            ++td.x;
            return false;
        }

        if(td.x == 8)
        {
            td.x = 10;
            return false;
        }

        if(td.x == 10)
        {
            td.x = 11;
            return false;
        }

        if(td.x == 10)
        {
            td.x = 11;
            return false;
        }

        if(td.x == 11)
        {
            td.x = 14;
            return false;
        }

        if(td.x == 14)
        {
            td.x = 15;
            return false;
        }

        td.x = 0;
        return true;
    }
};

template <class TRange, class TValue>
bool Next(const TRange& range, TValue& value)
{
    auto it = GenericFind(range, value);
    assert(it != range.end());
    if(++it == range.end())
    {
        value = *range.begin();
        return true;
    }

    value = *it;
    return false;
}

template <class TValue, TValue... values>
struct BoostSequence
{
    using ValueType      = TValue;
    using const_iterator = const TValue*;
    constexpr const_iterator begin() const { return arr.begin(); }
    constexpr const_iterator end() const { return arr.end(); }

private:
    static constexpr std::size_t count                         = sizeof...(values);
    static constexpr std::array<int, BoostSequence::count> arr = {{values...}};
};

template <class TValue, TValue... values>
constexpr std::array<int, BoostSequence<TValue, values...>::count>
    BoostSequence<TValue, values...>::arr;

template <class Start, class IntegerSequence>
struct span_impl;

template <class Start, class T, T... Ns>
struct span_impl<Start, std::integer_sequence<T, Ns...>>
{
    using ValueType       = T;
    using const_iterator  = const T*;
    T data[sizeof...(Ns)] = {(Ns + Start{})...};

    constexpr const T* begin() const { return data; }

    constexpr const T* end() const { return data + sizeof...(Ns); }

    constexpr const T* find(T value) const
    {
        if(value < Start{} || value > (Start{} + sizeof...(Ns)))
        {
            return end();
        }
        else
        {
            return data + (value - Start{});
        }
    }
};

template <class T, T start, T end>
struct BoostSpan
    : span_impl<std::integral_constant<T, start>, std::make_integer_sequence<T, end - start>>
{
};

template <Sequences seq>
struct BoostImpl
{
};

template <>
struct BoostImpl<Sequences::IRange>
{
    bool Next(TestData& v) const { return miopen::seq::Next(boost::irange(0, 9), v.x); }
};

struct SpeedTestDriver : public test_driver
{
    SpeedTestDriver()
    {
        add(iterations, "iterations");
        add(modeStr, "mode");
        add(sequenceStr, "seq");
        add(instance_str, "inst");
    }

    void run()
    {
        const auto seq = ParseSequence(sequenceStr);
        mode           = ParseMode(modeStr);
        instance       = ParseInstance(instance_str);

        switch(seq)
        {
        case Sequences::Seq: RunCore<Sequences::Seq>(); break;
        case Sequences::Span: RunCore<Sequences::Span>(); break;
        case Sequences::IRange: RunCore<Sequences::IRange>(); break;
        case Sequences::Join: RunCore<Sequences::Join>(); break;
        case Sequences::Unknown:
            std::cerr << "Unknown sequence." << std::endl;
            std::exit(-1); // NOLINT (concurrency-mt-unsafe)
        }
    }

    void show_help()
    {
        test_driver::show_help();
        std::cout << "Permitted modes: nat, std, tmpl, boost" << std::endl;
        std::cout << "Permitted sequences: seq, span, irange (boost), join(nat, std, tmpl)"
                  << std::endl;
        std::cout << "Permitted instances: single, percall, static" << std::endl;
    }

private:
    int iterations           = 10;
    Instances instance       = Instances::Unknown;
    Modes mode               = Modes::Unknown;
    std::string instance_str = "single";
    std::string modeStr      = "std";
    std::string sequenceStr  = "seq";

    static Modes ParseMode(const std::string& str)
    {
        if(str == "std")
            return Modes::Standard;
        if(str == "tmpl")
            return Modes::Template;
        if(str == "nat")
            return Modes::Native;
        if(str == "boost")
            return Modes::Boost;
        return Modes::Unknown;
    }

    static Sequences ParseSequence(const std::string& str)
    {
        if(str == "seq")
            return Sequences::Seq;
        if(str == "span")
            return Sequences::Span;
        if(str == "join")
            return Sequences::Join;
        if(str == "irange")
            return Sequences::IRange;
        return Sequences::Unknown;
    }

    static Instances ParseInstance(const std::string& str)
    {
        if(str == "single")
            return Instances::Single;
        if(str == "percall")
            return Instances::PerCall;
        if(str == "static")
            return Instances::Static;
        return Instances::Unknown;
    }

    template <Sequences seq>
    void RunCore() const
    {
        switch(mode)
        {
        case Modes::Standard: Standard<seq>(); break;
        case Modes::Template: Template<seq>(); break;
        case Modes::Native: Native<seq>(); break;
        case Modes::Boost: Boost<seq>(); break;
        case Modes::Unknown:
            std::cerr << "Unknown mode." << std::endl;
            std::exit(-1); // NOLINT (concurrency-mt-unsafe)
        }
    }

    template <class TType>
    void SaveDeadCode(const TType& value) const
    {
        static const std::string dead_code_saver;

        if(dead_code_saver.data() == nullptr)
        {
            std::cout << value << std::endl;
            std::terminate();
        }
    }

    template <class TRule>
    void PrintCollectionImpl(rank<0>, const TRule&) const
    {
    }

    template <class TRule, class = decltype(&TRule::FillBegin)>
    void PrintCollectionImpl(rank<1>, const TRule& rule) const
    {
        TestData td{1};
        rule.FillBegin(td);
        auto i = 0;

        do
        {
            std::cout << td.x << " ";
        } while(!rule.Next(td) && ++i < 1024);

        std::cout << std::endl;
    }

    template <class TRule>
    void PrintCollection(const TRule& rule) const
    {
        PrintCollectionImpl(rank<16>{}, rule);
    }

    template <class TRuleGetter>
    void TestCore(const TRuleGetter& rule_getter) const
    {
        PrintCollection(rule_getter());

        TestData td{1};

        const auto start = std::chrono::steady_clock::now();

        for(auto i = 0; i < iterations; i++)
            for(auto j = 0; j < 128 * 1024 * 1024; j++)
                rule_getter().Next(td);

        const auto time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count() *
                          .001 * .001;

        std::cout << "Test time: " << time << " seconds" << std::endl;

        SaveDeadCode(td.x); // required in release builds
    }

    template <class TRuleGetter>
    void Test(const TRuleGetter& rule_getter) const
    {
        switch(instance)
        {
        case Instances::PerCall: TestCore(rule_getter); break;
        case Instances::Single: {
            const auto inst = rule_getter();
            TestCore([&inst]() -> const auto& { return inst; });
            break;
        }
        case Instances::Static:
            TestCore([&rule_getter]() -> const auto& {
                static const auto inst = rule_getter();
                return inst;
            });
            break;
        case Instances::Unknown:
            std::cerr << "Unknown instance type" << std::endl;
            std::exit(-1); // NOLINT (concurrency-mt-unsafe)
        }
    }

    template <Sequences seq>
    void Native() const
    {
        Test([]() { return NativeImpl<seq>{}; });
    }

    static auto MakeSeq() { return Sequence<int, 1, 2, 7>{}; }
    static auto MakeBSeq() { return BoostSequence<int, 1, 2, 7>{}; }
    static auto MakeSpan() { return Span<int, 0, 8>{}; }
    static auto MakeBSpan() { return BoostSpan<int, 0, 9>{}; }
    static auto MakeJoin() { return Join<Span<int, 0, 8>, Sequence<int, 10, 11, 14, 15>>{}; }
    static auto TmplMember() { return std::integral_constant<int TestData::*, &TestData::x>{}; }

    template <class... TParams>
    static auto RS(TParams... args)
    {
        return MakeRuleSet(std::make_tuple(args...));
    }

    template <Sequences seq>
    void Standard() const
    {
        switch(seq)
        {
        case Sequences::Seq: Test([]() { return RS(MakeSeq(), &TestData::x); }); break;
        case Sequences::Span: Test([]() { return RS(MakeSpan(), &TestData::x); }); break;
        case Sequences::Join: Test([]() { return RS(MakeJoin(), &TestData::x); }); break;
        case Sequences::IRange:
            std::cerr << "irange is only for boost" << std::endl;
            std::exit(-1); // NOLINT (concurrency-mt-unsafe)
        }
    }

    template <Sequences seq>
    void Template() const
    {
        switch(seq)
        {
        case Sequences::Seq: Test([]() { return RS(MakeSeq(), TmplMember()); }); break;
        case Sequences::Span: Test([]() { return RS(MakeSpan(), TmplMember()); }); break;
        case Sequences::Join: Test([]() { return RS(MakeJoin(), TmplMember()); }); break;
        case Sequences::IRange:
            std::cerr << "irange is only for boost" << std::endl;
            std::exit(-1); // NOLINT (concurrency-mt-unsafe)
        }
    }

    template <Sequences seq>
    void Boost() const
    {
        switch(seq)
        {
        case Sequences::Seq: Test([]() { return RS(MakeBSeq(), &TestData::x); }); break;
        case Sequences::Span: Test([]() { return RS(MakeBSpan(), &TestData::x); }); break;
        case Sequences::Join:
            std::cerr << "join is only for nat/std/tmpl" << std::endl;
            std::exit(-1); // NOLINT (concurrency-mt-unsafe)
        case Sequences::IRange: Test([]() { return BoostImpl<Sequences::IRange>{}; }); break;
        }
    }
};
} // namespace seq
} // namespace miopen

int main(int argc, const char* argv[])
{
    test_drive<miopen::seq::SpeedTestDriver>(argc, argv);
    return 0;
}
