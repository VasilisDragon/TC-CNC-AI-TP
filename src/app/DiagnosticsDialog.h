#pragma once

#if WITH_EMBEDDED_TESTS

#include <QDialog>

#include "tests_core/TestsCore.h"

class QLabel;
class QPushButton;
class QTreeWidget;

namespace app
{

class DiagnosticsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DiagnosticsDialog(QWidget* parent = nullptr);

private slots:
    void runFastTests();
    void runAllTests();
    void openBuildLogs();
    void exportReport();

private:
    void runTests(tests_core::RunMode mode);
    void displaySummary(const tests_core::RunSummary& summary);
    QString buildReportMarkdown(const tests_core::RunSummary& summary) const;
    void updateRunButtons(bool running);

    QPushButton* m_runFastButton{nullptr};
    QPushButton* m_runAllButton{nullptr};
    QPushButton* m_openLogsButton{nullptr};
    QPushButton* m_exportReportButton{nullptr};
    QTreeWidget* m_resultsView{nullptr};
    QLabel* m_summaryLabel{nullptr};
    tests_core::RunSummary m_lastSummary;
};

} // namespace app

#endif // WITH_EMBEDDED_TESTS
