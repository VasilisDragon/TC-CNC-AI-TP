#include "tests_core/TestRegistry.h"

#include "doctest/doctest.h"
#include "tp/Toolpath.h"

TESTS_CORE_TEST_CASE(basic_toolpath_defaults, "fast")
{
    tp::Toolpath toolpath{};
    DOCTEST_CHECK(toolpath.feed == 0.0);
    DOCTEST_CHECK(toolpath.spindle == 0.0);
    DOCTEST_CHECK(toolpath.passes.empty());
}
