#include "app/DiagnosticsDialog.h"

#if WITH_EMBEDDED_TESTS

#include <QAbstractItemView>
#include <QBoxLayout>
#include <QBrush>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QFileDialog>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QStandardPaths>
#include <QTextStream>
#include <QTreeWidget>
#include <QUrl>
#include <QStringList>

namespace app
{

namespace
{

QString statusText(bool passed)
{
    return passed ? QObject::tr("Passed") : QObject::tr("Failed");
}

QBrush statusBrush(bool passed)
{
    return passed ? QBrush(QColor(25, 130, 75)) : QBrush(QColor(200, 45, 45));
}

QString joinTags(const QStringList& tags)
{
    if (tags.isEmpty())
    {
        return QObject::tr("untagged");
    }
    return tags.join(", ");
}

tests_core::RunOptions toOptions(tests_core::RunMode mode)
{
    tests_core::RunOptions options;
    options.mode = mode;
    return options;
}

} // namespace

DiagnosticsDialog::DiagnosticsDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Diagnostics"));
    resize(720, 520);

    auto* rootLayout = new QVBoxLayout(this);

    auto* buttonRow = new QHBoxLayout();
    m_runFastButton = new QPushButton(tr("Run &Fast Tests"), this);
    m_runAllButton = new QPushButton(tr("Run &All Tests"), this);
    buttonRow->addWidget(m_runFastButton);
    buttonRow->addWidget(m_runAllButton);
    buttonRow->addStretch();
    rootLayout->addLayout(buttonRow);

    m_resultsView = new QTreeWidget(this);
    m_resultsView->setColumnCount(3);
    m_resultsView->setHeaderLabels({tr("Test"), tr("Status"), tr("Duration (ms)")});
    m_resultsView->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    m_resultsView->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_resultsView->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    m_resultsView->setRootIsDecorated(false);
    m_resultsView->setAlternatingRowColors(true);
    m_resultsView->setSelectionMode(QAbstractItemView::NoSelection);
    rootLayout->addWidget(m_resultsView);

    m_summaryLabel = new QLabel(tr("No diagnostics run."), this);
    rootLayout->addWidget(m_summaryLabel);

    auto* bottomRow = new QHBoxLayout();
    m_openLogsButton = new QPushButton(tr("Open Build Logs"), this);
    m_exportReportButton = new QPushButton(tr("Open Diagnostics Report"), this);
    m_exportReportButton->setEnabled(false);
    bottomRow->addWidget(m_openLogsButton);
    bottomRow->addStretch();
    bottomRow->addWidget(m_exportReportButton);
    rootLayout->addLayout(bottomRow);

    connect(m_runFastButton, &QPushButton::clicked, this, &DiagnosticsDialog::runFastTests);
    connect(m_runAllButton, &QPushButton::clicked, this, &DiagnosticsDialog::runAllTests);
    connect(m_openLogsButton, &QPushButton::clicked, this, &DiagnosticsDialog::openBuildLogs);
    connect(m_exportReportButton, &QPushButton::clicked, this, &DiagnosticsDialog::exportReport);
}

void DiagnosticsDialog::runFastTests()
{
    runTests(tests_core::RunMode::Fast);
}

void DiagnosticsDialog::runAllTests()
{
    runTests(tests_core::RunMode::All);
}

void DiagnosticsDialog::runTests(tests_core::RunMode mode)
{
    updateRunButtons(true);
    const tests_core::RunSummary summary = tests_core::runTests(toOptions(mode));
    m_lastSummary = summary;
    displaySummary(m_lastSummary);
    updateRunButtons(false);
}

void DiagnosticsDialog::displaySummary(const tests_core::RunSummary& summary)
{
    m_resultsView->clear();

    for (const auto& result : summary.cases)
    {
        auto* item = new QTreeWidgetItem(m_resultsView);
        item->setText(0, result.name);
        item->setToolTip(0, joinTags(result.tags));
        item->setText(1, statusText(result.passed));
        item->setForeground(1, statusBrush(result.passed));
        item->setText(2, QString::number(result.durationMs, 'f', 2));
    }

    const int passed = summary.executed - summary.failed;
    QString summaryText = tr("%1/%2 tests passed (%3 failed, %4 skipped) in %5 ms")
                              .arg(passed)
                              .arg(summary.executed)
                              .arg(summary.failed)
                              .arg(summary.skipped)
                              .arg(QString::number(summary.durationMs, 'f', 2));
    if (summary.discovered > summary.executed + summary.skipped)
    {
        summaryText.append(tr(" (plus %1 tests not selected)").arg(summary.discovered - summary.executed - summary.skipped));
    }
    m_summaryLabel->setText(summaryText);

    m_exportReportButton->setEnabled(summary.executed > 0);
}

void DiagnosticsDialog::openBuildLogs()
{
    QString logsDir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    if (logsDir.isEmpty())
    {
        logsDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    }
    if (logsDir.isEmpty())
    {
        logsDir = QCoreApplication::applicationDirPath();
    }

    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(logsDir)))
    {
        QMessageBox::warning(this, tr("Diagnostics"), tr("Unable to open log location: %1").arg(logsDir));
    }
}

void DiagnosticsDialog::exportReport()
{
    if (m_lastSummary.executed == 0)
    {
        QMessageBox::information(this, tr("Diagnostics"), tr("Run diagnostics before exporting a report."));
        return;
    }

    const QString defaultName = tr("diagnostics_report_%1.md")
                                    .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));

    const QString destination = QFileDialog::getSaveFileName(this,
                                                             tr("Save Diagnostics Report"),
                                                             defaultName,
                                                             tr("Markdown Files (*.md);;All Files (*.*)"));
    if (destination.isEmpty())
    {
        return;
    }

    QSaveFile file(destination);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        QMessageBox::warning(this, tr("Diagnostics"), tr("Unable to write report to %1").arg(destination));
        return;
    }

    QTextStream stream(&file);
    stream << buildReportMarkdown(m_lastSummary);
    if (!file.commit())
    {
        QMessageBox::warning(this, tr("Diagnostics"), tr("Failed to save diagnostics report."));
    }
}

QString DiagnosticsDialog::buildReportMarkdown(const tests_core::RunSummary& summary) const
{
    QStringList lines;
    lines << QStringLiteral("# Diagnostics Report");
    lines << QStringLiteral("");
    lines << tr("- Generated: %1").arg(QDateTime::currentDateTime().toString(Qt::ISODateWithMs));
    lines << tr("- Mode: %1").arg(summary.mode == tests_core::RunMode::Fast ? tr("fast") : tr("all"));
    lines << tr("- Tests discovered: %1").arg(summary.discovered);
    lines << tr("- Tests executed: %1").arg(summary.executed);
    lines << tr("- Tests failed: %1").arg(summary.failed);
    lines << tr("- Total duration: %1 ms").arg(QString::number(summary.durationMs, 'f', 2));
    lines << QStringLiteral("");
    lines << QStringLiteral("## Results");

    if (summary.cases.isEmpty())
    {
        lines << tr("No tests were executed.");
    }
    else
    {
        for (const auto& result : summary.cases)
        {
            lines << QStringLiteral("### %1").arg(result.name);
            lines << tr("- Status: %1").arg(statusText(result.passed));
            lines << tr("- Duration: %1 ms").arg(QString::number(result.durationMs, 'f', 2));
            lines << tr("- Tags: %1").arg(joinTags(result.tags));
            if (!result.message.isEmpty())
            {
                lines << QStringLiteral("");
                lines << QStringLiteral("```\n%1\n```").arg(result.message);
            }
            lines << QStringLiteral("");
        }
    }

    return lines.join(QLatin1Char('\n'));
}

void DiagnosticsDialog::updateRunButtons(bool running)
{
    m_runFastButton->setEnabled(!running);
    m_runAllButton->setEnabled(!running);
}

} // namespace app

#endif // WITH_EMBEDDED_TESTS
