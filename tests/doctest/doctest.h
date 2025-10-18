#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace doctest
{
using TestFunc = std::function<void()>;

inline std::vector<TestFunc>& registry()
{
    static std::vector<TestFunc> tests;
    return tests;
}

struct TestRegistrar
{
    explicit TestRegistrar(TestFunc func)
    {
        registry().push_back(std::move(func));
    }
};

inline int run_all()
{
    int failures = 0;
    for (const auto& test : registry())
    {
        try
        {
            test();
        }
        catch (const std::exception& e)
        {
            std::cerr << "[doctest] test threw std::exception: " << e.what() << '\n';
            ++failures;
        }
        catch (...)
        {
            std::cerr << "[doctest] test threw unknown exception\n";
            ++failures;
        }
    }
    return failures;
}
} // namespace doctest

#define DOCTEST_CONCAT_IMPL(x, y) x##y
#define DOCTEST_CONCAT(x, y) DOCTEST_CONCAT_IMPL(x, y)

#define DOCTEST_TEST_CASE_INTERNAL(func)                                                        \
    static void func();                                                                         \
    static const doctest::TestRegistrar DOCTEST_CONCAT(func, _registrar)(func);                 \
    static void func()

#define DOCTEST_CHECK(expr)                                                                     \
    do                                                                                          \
    {                                                                                           \
        if (!(expr))                                                                            \
        {                                                                                       \
            std::cerr << "[doctest] check failed: " #expr " @ " << __FILE__ << ':' << __LINE__  \
                      << '\n';                                                                  \
            throw std::runtime_error("doctest check failed");                                   \
        }                                                                                       \
    } while (false)

#define TEST_CASE(desc) DOCTEST_TEST_CASE_INTERNAL(DOCTEST_CONCAT(doctest_tc_, __LINE__))
#define CHECK(expr) DOCTEST_CHECK(expr)
#define CHECK_FALSE(expr) DOCTEST_CHECK(!(expr))

namespace doctest
{
class Approx
{
public:
    explicit Approx(double value) : m_value(value) {}

    Approx& epsilon(double eps)
    {
        m_epsilon = eps;
        return *this;
    }

    template <typename T>
    friend bool operator==(T lhs, const Approx& rhs)
    {
        return rhs.compare(static_cast<double>(lhs));
    }

    template <typename T>
    friend bool operator==(const Approx& lhs, T rhs)
    {
        return lhs.compare(static_cast<double>(rhs));
    }

    template <typename T>
    friend bool operator!=(T lhs, const Approx& rhs)
    {
        return !(lhs == rhs);
    }

    template <typename T>
    friend bool operator!=(const Approx& lhs, T rhs)
    {
        return !(lhs == rhs);
    }

private:
    bool compare(double other) const
    {
        const double scale = std::max(1.0, std::fabs(m_value));
        return std::fabs(other - m_value) <= m_epsilon * scale;
    }

    double m_value;
    double m_epsilon = 1e-6;
};
} // namespace doctest

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
int main()
{
    const int failures = doctest::run_all();
    if (failures != 0)
    {
        std::cerr << "[doctest] " << failures << " test(s) failed\n";
    }
    return failures;
}
#endif
