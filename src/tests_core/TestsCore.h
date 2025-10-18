#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

namespace tests_core
{

enum class RunMode
{
    Fast,
    All
};

struct TestCaseResult
{
    QString name;
    QStringList tags;
    bool passed{false};
    double durationMs{0.0};
    QString message;
};

struct RunSummary
{
    RunMode mode{RunMode::Fast};
    int discovered{0};
    int executed{0};
    int skipped{0};
    int failed{0};
    double durationMs{0.0};
    QVector<TestCaseResult> cases;
};

struct RunOptions
{
    RunMode mode{RunMode::Fast};
};

RunSummary runTests(const RunOptions& options);

} // namespace tests_core
