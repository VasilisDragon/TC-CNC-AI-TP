#pragma once

#include "doctest/doctest.h"

#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace tests_core
{

struct RegisteredTest
{
    std::string name;
    std::vector<std::string> tags;
    doctest::TestFunc func;
};

inline std::vector<RegisteredTest>& registry()
{
    static std::vector<RegisteredTest> tests;
    return tests;
}

inline std::vector<std::string> parseTags(const char* tagsRaw)
{
    std::vector<std::string> parsed;
    if (tagsRaw == nullptr)
    {
        return parsed;
    }

    std::string buffer;
    const std::string tags(tagsRaw);
    auto flush = [&]() {
        if (!buffer.empty())
        {
            parsed.push_back(buffer);
            buffer.clear();
        }
    };

    for (char ch : tags)
    {
        if (std::isspace(static_cast<unsigned char>(ch)) || ch == ',' || ch == ';')
        {
            flush();
        }
        else
        {
            buffer.push_back(ch);
        }
    }
    flush();
    return parsed;
}

inline void registerTest(const char* name, const char* tags, doctest::TestFunc func)
{
    RegisteredTest test;
    test.name = name ? std::string(name) : std::string("unnamed");
    test.tags = parseTags(tags);
    test.func = func;
    registry().push_back(test);
    doctest::registry().push_back(std::move(func));
}

inline const std::vector<RegisteredTest>& allTests()
{
    return registry();
}

} // namespace tests_core

#define TESTS_CORE_TEST_CASE(Name, Tags)                                                     \
    static void Name();                                                                      \
    static const bool Name##_tests_core_registered = []() {                                  \
        ::tests_core::registerTest(#Name, Tags, Name);                                       \
        return true;                                                                         \
    }();                                                                                     \
    static void Name()
