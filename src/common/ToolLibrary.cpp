#include "common/ToolLibrary.h"

#include <QtCore/QFile>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>

namespace common
{

namespace
{

Tool parseTool(const QJsonObject& obj, QStringList& warnings)
{
    Tool tool;
    tool.id = obj.value(QStringLiteral("id")).toString();
    tool.name = obj.value(QStringLiteral("name")).toString();
    tool.type = obj.value(QStringLiteral("type")).toString();
    tool.notes = obj.value(QStringLiteral("notes")).toString();
    tool.diameterMm = obj.value(QStringLiteral("diameter_mm")).toDouble(0.0);

    if (!tool.isValid())
    {
        warnings.push_back(QStringLiteral("Skipping invalid tool entry: \"%1\"").arg(tool.name.isEmpty() ? tool.id : tool.name));
    }

    return tool;
}

} // namespace

bool ToolLibrary::loadFromJson(const QByteArray& data, QStringList& warnings)
{
    warnings.clear();
    m_tools.clear();

    if (data.isEmpty())
    {
        warnings.push_back(QStringLiteral("Tool library JSON is empty."));
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError)
    {
        warnings.push_back(QStringLiteral("Failed to parse tool library: %1").arg(parseError.errorString()));
        return false;
    }

    const QJsonObject root = doc.object();
    const QJsonArray toolsArray = root.value(QStringLiteral("tools")).toArray();
    if (toolsArray.isEmpty())
    {
        warnings.push_back(QStringLiteral("Tool library contains no tools."));
        return false;
    }

    for (const QJsonValue& value : toolsArray)
    {
        if (!value.isObject())
        {
            warnings.push_back(QStringLiteral("Skipping malformed tool entry (expected object)."));
            continue;
        }

        Tool tool = parseTool(value.toObject(), warnings);
        if (tool.isValid())
        {
            m_tools.push_back(std::move(tool));
        }
    }

    if (m_tools.isEmpty())
    {
        warnings.push_back(QStringLiteral("No valid tools were loaded."));
        return false;
    }

    return true;
}

bool ToolLibrary::loadFromFile(const QString& filePath, QStringList& warnings)
{
    QFile file(filePath);
    if (!file.exists())
    {
        warnings.push_back(QStringLiteral("Tool library file not found: %1").arg(filePath));
        return false;
    }

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        warnings.push_back(QStringLiteral("Unable to open tool library file: %1").arg(filePath));
        return false;
    }

    const QByteArray data = file.readAll();
    return loadFromJson(data, warnings);
}

const Tool* ToolLibrary::toolById(const QString& id) const noexcept
{
    for (const Tool& tool : m_tools)
    {
        if (tool.id.compare(id, Qt::CaseInsensitive) == 0)
        {
            return &tool;
        }
    }
    return nullptr;
}

int ToolLibrary::indexOf(const QString& id) const noexcept
{
    for (int i = 0; i < m_tools.size(); ++i)
    {
        if (m_tools.at(i).id.compare(id, Qt::CaseInsensitive) == 0)
        {
            return i;
        }
    }
    return -1;
}

} // namespace common

