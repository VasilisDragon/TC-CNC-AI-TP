#include "tests_core/TestsCore.h"

#include "tests_core/TestRegistry.h"

#include <QElapsedTimer>

#include <algorithm>
#include <cctype>
#include <exception>

namespace tests_core
{

namespace
{

bool hasFastTag(const std::vector<std::string>& tags)
{
    return std::any_of(tags.begin(), tags.end(), [](const std::string& tag) {
        if (tag.empty())
        {
            return false;
        }
        std::string lowered(tag);
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return lowered == "fast";
    });
}

QStringList toQStringList(const std::vector<std::string>& tags)
{
    QStringList qtTags;
    for (const auto& tag : tags)
    {
        qtTags << QString::fromStdString(tag);
    }
    return qtTags;
}

} // namespace

RunSummary runTests(const RunOptions& options)
{
    RunSummary summary;
    summary.mode = options.mode;
    const auto& all = allTests();
    summary.discovered = static_cast<int>(all.size());

    if (all.empty())
    {
        return summary;
    }

    QElapsedTimer totalTimer;
    totalTimer.start();

    for (const auto& test : all)
    {
        if (options.mode == RunMode::Fast && !hasFastTag(test.tags))
        {
            summary.skipped++;
            continue;
        }

        TestCaseResult result;
        result.name = QString::fromStdString(test.name);
        result.tags = toQStringList(test.tags);

        QElapsedTimer caseTimer;
        caseTimer.start();

        try
        {
            test.func();
            result.passed = true;
        }
        catch (const std::exception& e)
        {
            result.passed = false;
            result.message = QString::fromLocal8Bit(e.what());
        }
        catch (...)
        {
            result.passed = false;
            result.message = QStringLiteral("Unknown exception");
        }

        result.durationMs = static_cast<double>(caseTimer.elapsed());

        summary.cases.append(result);
        summary.executed++;
        if (!result.passed)
        {
            summary.failed++;
        }
    }

    summary.durationMs = static_cast<double>(totalTimer.elapsed());
    return summary;
}

} // namespace tests_core
