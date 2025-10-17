#pragma once

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
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

#define DOCTEST_TEST_CASE(name)                                                                 \
    static void name();                                                                         \
    static const doctest::TestRegistrar name##_registrar(name);                                 \
    static void name()

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
