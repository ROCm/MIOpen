#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "miopen/data_entry.hpp"

namespace miopen
{
    namespace tests
    {

        struct TestData
        {
            int x;
            int y;

            void Serialize(std::ostream& s) const
            {
                static const auto sep = ',';
                s << x
                    << sep << y;
            }

            bool Deserialize(const std::string& s)
            {
                static const auto sep = ',';
                TestData t;
                std::istringstream ss(s);

                auto success =
                    DeserializeField(ss, &t.x, sep) &&
                    DeserializeField(ss, &t.y, sep);

                if (!success)
                    return false;

                *this = t;
                return true;
            }

            bool operator ==(const TestData& other) { return x == other.x && y == other.y; }

        private:
            inline static bool DeserializeField(std::istream& from, int* ret, char separator)
            {
                std::string part;

                if (!std::getline(from, part, separator))
                    return false;

                const auto start = part.c_str();
                char* end;
                auto value = std::strtol(start, &end, 10);

                if (start == end)
                    return false;

                *ret = value;
                return true;
            }
        };

        class TestBase
        {
        public:
            virtual const char* Name() const = 0;
            virtual int Total() const = 0;
            virtual int Failed() const = 0;
            virtual void Run() = 0;
            virtual ~TestBase() {}
            TestBase() noexcept {}
            TestBase(const TestBase&) {}
        };

        class Test : public TestBase
        {
        public:
            inline int Total() const override { return _total; }
            inline int Failed() const override { return _failures.size(); }
            inline const std::vector<std::string>& Failures() const { return _failures; }

        protected:
            template<class TValue>
            inline void AssertEqual(const std::string& name, TValue left, TValue right) { AssertTrue(name, left == right); }

            inline void AssertTrue(const std::string& name, bool condition)
            {
                if (!condition)
                    _failures.push_back(name);

                _total++;
            }

        private:
            int _total = 0;
            std::vector<std::string> _failures;
        };

        class TestBatch : public TestBase
        {
        public:
            inline int Total() const override { return _total; }
            inline int Failed() const override { return _failures.size(); }
            inline const char* Name() const override { return _name; }
            inline const char*& Name() { return _name; }

            inline void Run() override
            {
                auto total = 0;

                for (Test& test : _subtests)
                {
                    test.Run();
                    std::cout << test.Name() << " tests failed: " << test.Failed() << '/' << test.Total() << std::endl;
                    total += test.Total();

                    for (auto& fail : test.Failures())
                        _failures.push_back(std::string(test.Name()) + "." + fail);
                }

                std::cout << "Total for " << Name() << ": " << _failures.size() << '/' << total << std::endl;

                if (_failures.size() > 0)
                {
                    std::cout << "Failures:" << std::endl;

                    for (const auto& fail : _failures)
                        std::cout << fail << std::endl;
                }
            }

            inline void Add(Test& test) { _subtests.push_back(test); }

        private:
            int _total;
            const char* _name;
            std::vector<std::reference_wrapper<Test>> _subtests;
            std::vector<std::string> _failures;
        };

        namespace detail
        {
            class DbRecordReadTest : public Test
            {
            public:
                inline const char* Name() const override { return "DbRecord.Read"; }

                inline void Run() override
                {
                    auto path = "/tmp/dbread.test.temp.pdb";

                    TestData key{ 1, 2, };
                    TestData value0{ 3, 4, };
                    TestData value1{ 5, 6, };
                    auto id0 = "0";
                    auto id1 = "1";

                    std::ostringstream ss_vals;
                    ss_vals << key.x << ',' << key.y << '='
                        << id1 << ':' << value1.x << ',' << value1.y << ';'
                        << id0 << ':' << value0.x << ',' << value0.y;

                    std::ofstream(path) << ss_vals.str() << std::endl;

                    TestData read0, read1;

                    {
                        DbRecord record(path, key);

                        AssertTrue("Read0", record.Load(id0, read0));
                        AssertTrue("Read1", record.Load(id1, read1));
                    }

                    std::remove(path);

                    AssertEqual("Equal0", value0, read0);
                    AssertEqual("Equal1", value1, read1);
                }
            };

            class DbRecordWriteTest : public Test
            {
            public:
                inline const char* Name() const override { return "DbRecord.Write"; }

                inline void Run() override
                {
                    auto path = "/tmp/dbread.test.temp.pdb";

                    TestData key{ 1, 2, };
                    TestData value0{ 3, 4, };
                    TestData value1{ 5, 6, };
                    auto id0 = "0";
                    auto id1 = "1";

                    std::ostringstream ss_vals;
                    ss_vals << key.x << ',' << key.y << '='
                        << id1 << ':' << value1.x << ',' << value1.y << ';'
                        << id0 << ':' << value0.x << ',' << value0.y;

                    (void)std::ofstream(path);

                    {
                        DbRecord record(path, key);

                        AssertTrue("Write0", record.Save(id0, value0));
                        AssertTrue("Write1", record.Save(id1, value1));
                    }

                    std::string read;
                    AssertTrue("GetLine", std::getline(std::ifstream(path), read).good());
                    AssertEqual("Equal", read, ss_vals.str());

                    std::remove(path);
                }
            };
        }

        TestBase& DbRecordTest()
        {
            static auto test = ([]()
            {
                TestBatch ret;
                static detail::DbRecordReadTest read;
                static detail::DbRecordWriteTest write;

                ret.Name() = "DbRecord";
                ret.Add(read);
                ret.Add(write);

                return ret;
            })();

            return test;
        }

    } // namespace miopen
} // namespace tests

int main()
{
    auto& test = miopen::tests::DbRecordTest();
    test.Run();
    return test.Failed();
}
