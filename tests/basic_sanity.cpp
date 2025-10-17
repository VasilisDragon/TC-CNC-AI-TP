#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"

#include "tp/Toolpath.h"

DOCTEST_TEST_CASE(basic_toolpath_defaults)
{
    tp::Toolpath toolpath{};
    DOCTEST_CHECK(toolpath.feed == 0.0);
    DOCTEST_CHECK(toolpath.spindle == 0.0);
    DOCTEST_CHECK(toolpath.passes.empty());
}

