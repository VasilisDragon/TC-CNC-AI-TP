#pragma once

#include <QtCore/QLoggingCategory>
#include <QtCore/QString>

namespace common
{

Q_DECLARE_LOGGING_CATEGORY(appLog)

void initLogging();
void logInfo(const QString& message);
void logWarning(const QString& message);
void logError(const QString& message);

} // namespace common

