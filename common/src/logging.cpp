#include "common/logging.h"

#include <QtCore/QDateTime>
#include <QtCore/QTextStream>

#include <cstdlib>

namespace common
{

Q_LOGGING_CATEGORY(appLog, "AIToolpathGenerator")

namespace
{

void outputMessage(QtMsgType type, const QMessageLogContext& context, const QString& message)
{
    Q_UNUSED(context)

    QString level;
    switch (type)
    {
    case QtDebugMsg: level = QStringLiteral("DEBUG"); break;
    case QtInfoMsg: level = QStringLiteral("INFO"); break;
    case QtWarningMsg: level = QStringLiteral("WARN"); break;
    case QtCriticalMsg: level = QStringLiteral("CRITICAL"); break;
    case QtFatalMsg: level = QStringLiteral("FATAL"); break;
    }

    QTextStream stream(stderr);
    stream << '[' << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << "] "
           << level << ": " << message << Qt::endl;

    if (type == QtFatalMsg)
    {
        abort();
    }
}

} // namespace

void initLogging()
{
    qInstallMessageHandler(outputMessage);
}

void logInfo(const QString& message)
{
    qCInfo(appLog).noquote() << message;
}

void logWarning(const QString& message)
{
    qCWarning(appLog).noquote() << message;
}

void logError(const QString& message)
{
    qCCritical(appLog).noquote() << message;
}

} // namespace common
