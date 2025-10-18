#pragma once

#include "common/Tool.h"

#include <QVector>
#include <QString>
#include <QStringList>

namespace common
{

class ToolLibrary
{
public:
    bool loadFromJson(const QByteArray& data, QStringList& warnings);
    bool loadFromFile(const QString& filePath, QStringList& warnings);

    [[nodiscard]] const QVector<Tool>& tools() const noexcept { return m_tools; }
    [[nodiscard]] const Tool* toolById(const QString& id) const noexcept;
    [[nodiscard]] int indexOf(const QString& id) const noexcept;

private:
    QVector<Tool> m_tools;
};

} // namespace common

