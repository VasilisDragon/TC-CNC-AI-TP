#include "ai/ModelCard.h"

#include "ai/FeatureExtractor.h"

#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QRegularExpression>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QStringView>

namespace
{

QString toQString(const std::filesystem::path& path)
{
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    return QString::fromStdString(path.string());
#endif
}

[[nodiscard]] std::vector<double> toNumberVector(const QJsonArray& array, int expectedCount, QStringView fieldName, QString& error)
{
    if (array.size() != expectedCount)
    {
        error = QStringLiteral("%1 expected %2 entries but found %3.")
                    .arg(fieldName, QString::number(expectedCount), QString::number(array.size()));
        return {};
    }

    std::vector<double> values;
    values.reserve(array.size());
    for (int i = 0; i < array.size(); ++i)
    {
        const QJsonValue value = array.at(i);
        double number = 0.0;
        if (value.isDouble())
        {
            number = value.toDouble();
        }
        else if (value.isString())
        {
            bool ok = false;
            number = value.toString().toDouble(&ok);
            if (!ok)
            {
                error = QStringLiteral("%1[%2] is not a numeric value.")
                            .arg(fieldName, QString::number(i));
                return {};
            }
        }
        else if (value.isNull() || value.isUndefined())
        {
            error = QStringLiteral("%1[%2] is not a numeric value.").arg(fieldName, QString::number(i));
            return {};
        }
        values.push_back(number);
    }
    return values;
}

[[nodiscard]] bool isSha256(QStringView value)
{
    static const QRegularExpression kSha256Regex(QStringLiteral("^[0-9a-fA-F]{64}$"));
    return kSha256Regex.match(value).hasMatch();
}

[[nodiscard]] QString missingCardMessage(const std::filesystem::path& modelPath,
                                         const std::vector<std::filesystem::path>& candidates)
{
    QStringList formatted;
    formatted.reserve(static_cast<int>(candidates.size()));
    for (const auto& candidate : candidates)
    {
        formatted.push_back(QDir::toNativeSeparators(toQString(candidate)));
    }
    return QStringLiteral("Model card not found for %1. Expected %2.")
        .arg(QDir::toNativeSeparators(toQString(modelPath)),
             formatted.join(QStringLiteral(" or ")));
}

} // namespace

namespace ai
{

std::optional<ModelCard> ModelCard::loadForModel(const std::filesystem::path& modelPath,
                                                 Backend backend,
                                                 std::string& errorOut)
{
    errorOut.clear();
    if (modelPath.empty())
    {
        errorOut = "Model path is empty.";
        return std::nullopt;
    }

    std::vector<std::filesystem::path> candidates;
    candidates.reserve(2);
    const std::filesystem::path appended = modelPath.string() + std::string(".model.json");
    candidates.push_back(appended);

    std::filesystem::path base = modelPath;
    base.replace_extension();
    const std::filesystem::path replaced = base.string() + std::string(".model.json");
    if (replaced != appended)
    {
        candidates.push_back(replaced);
    }

    QString lastError;
    for (const auto& candidate : candidates)
    {
        std::string parseError;
        auto card = loadFromPath(candidate, backend, parseError);
        if (card.has_value())
        {
            return card;
        }

        if (QFileInfo::exists(toQString(candidate)))
        {
            errorOut = parseError;
            return std::nullopt;
        }
        if (!parseError.empty())
        {
            lastError = QString::fromStdString(parseError);
        }
    }

    if (!lastError.isEmpty())
    {
        errorOut = lastError.toStdString();
        return std::nullopt;
    }

    errorOut = missingCardMessage(modelPath, candidates).toStdString();
    return std::nullopt;
}

std::optional<ModelCard> ModelCard::loadFromPath(const std::filesystem::path& cardPath,
                                                 Backend backend,
                                                 std::string& errorOut)
{
    errorOut.clear();

    const QString qPath = toQString(cardPath);
    QFileInfo info(qPath);
    if (!info.exists())
    {
        return std::nullopt;
    }

    QFile file(qPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        errorOut = QStringLiteral("Unable to open model card %1: %2.")
                       .arg(QDir::toNativeSeparators(qPath), file.errorString())
                       .toStdString();
        return std::nullopt;
    }

    QJsonParseError parseError;
    const QByteArray contents = file.readAll();
    const QJsonDocument doc = QJsonDocument::fromJson(contents, &parseError);
    if (parseError.error != QJsonParseError::NoError)
    {
        errorOut = QStringLiteral("Model card %1 is not valid JSON: %2.")
                       .arg(QDir::toNativeSeparators(qPath), parseError.errorString())
                       .toStdString();
        return std::nullopt;
    }
    if (!doc.isObject())
    {
        errorOut = QStringLiteral("Model card %1 must contain a JSON object at the root.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    const QJsonObject root = doc.object();

    const QJsonValue schemaValue = root.value(QStringLiteral("schema_version"));
    if (!schemaValue.isString() || schemaValue.toString().trimmed().isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 is missing a valid schema_version string.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    const QJsonValue modelTypeValue = root.value(QStringLiteral("model_type"));
    if (!modelTypeValue.isString() || modelTypeValue.toString().trimmed().isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 is missing a valid model_type string.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }
    const QString modelType = modelTypeValue.toString().trimmed().toLower();

    const QJsonObject featuresObj = root.value(QStringLiteral("features")).toObject();
    if (featuresObj.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 is missing the features block.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    const int featureCountValue = featuresObj.value(QStringLiteral("count")).toInt(-1);
    if (featureCountValue <= 0)
    {
        errorOut = QStringLiteral("Model card %1 must specify a positive features.count.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    const std::size_t expectedFeatureCount = FeatureExtractor::featureCount() + 2;
    if (static_cast<std::size_t>(featureCountValue) != expectedFeatureCount)
    {
        errorOut = QStringLiteral("Model card %1 features.count=%2 does not match expected %3.")
                       .arg(QDir::toNativeSeparators(qPath),
                            QString::number(featureCountValue),
                            QString::number(static_cast<qint64>(expectedFeatureCount)))
                       .toStdString();
        return std::nullopt;
    }

    const QJsonArray namesArray = featuresObj.value(QStringLiteral("names")).toArray();
    if (namesArray.size() != featureCountValue)
    {
        errorOut = QStringLiteral("Model card %1 features.names size (%2) must equal features.count (%3).")
                       .arg(QDir::toNativeSeparators(qPath),
                            QString::number(namesArray.size()),
                            QString::number(featureCountValue))
                       .toStdString();
        return std::nullopt;
    }

    std::vector<std::string> featureNames;
    featureNames.reserve(namesArray.size());
    for (int i = 0; i < namesArray.size(); ++i)
    {
        const QJsonValue value = namesArray.at(i);
        if (!value.isString() || value.toString().trimmed().isEmpty())
        {
            errorOut = QStringLiteral("Model card %1 features.names[%2] must be a non-empty string.")
                           .arg(QDir::toNativeSeparators(qPath), QString::number(i))
                           .toStdString();
            return std::nullopt;
        }
        featureNames.push_back(value.toString().trimmed().toStdString());
    }

    const QJsonObject normalizeObj = featuresObj.value(QStringLiteral("normalize")).toObject();
    if (normalizeObj.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 is missing features.normalize.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    QString normalizationError;
    const std::vector<double> mean = toNumberVector(normalizeObj.value(QStringLiteral("mean")).toArray(),
                                                    featureCountValue,
                                                    QStringLiteral("features.normalize.mean"),
                                                    normalizationError);
    if (!normalizationError.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 %2")
                       .arg(QDir::toNativeSeparators(qPath), normalizationError)
                       .toStdString();
        return std::nullopt;
    }
    const std::vector<double> std = toNumberVector(normalizeObj.value(QStringLiteral("std")).toArray(),
                                                   featureCountValue,
                                                   QStringLiteral("features.normalize.std"),
                                                   normalizationError);
    if (!normalizationError.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 %2")
                       .arg(QDir::toNativeSeparators(qPath), normalizationError)
                       .toStdString();
        return std::nullopt;
    }

    const QJsonObject trainingObj = root.value(QStringLiteral("training")).toObject();
    if (trainingObj.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 is missing the training block.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }
    const QString framework = trainingObj.value(QStringLiteral("framework")).toString().trimmed();
    if (framework.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 training.framework must be a non-empty string.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    const QString frameworkLower = framework.toLower();
    if (backend == Backend::Torch)
    {
        if (!frameworkLower.contains(QStringLiteral("torch")))
        {
            errorOut = QStringLiteral("Model card %1 training.framework \"%2\" does not match Torch backend.")
                           .arg(QDir::toNativeSeparators(qPath), framework)
                           .toStdString();
            return std::nullopt;
        }
        if (modelType != QStringLiteral("torchscript"))
        {
            errorOut = QStringLiteral("Model card %1 model_type \"%2\" must be \"torchscript\" for Torch models.")
                           .arg(QDir::toNativeSeparators(qPath), modelTypeValue.toString())
                           .toStdString();
            return std::nullopt;
        }
    }
    else
    {
        if (!frameworkLower.contains(QStringLiteral("onnx")))
        {
            errorOut = QStringLiteral("Model card %1 training.framework \"%2\" does not match ONNX backend.")
                           .arg(QDir::toNativeSeparators(qPath), framework)
                           .toStdString();
            return std::nullopt;
        }
        if (modelType != QStringLiteral("onnx"))
        {
            errorOut = QStringLiteral("Model card %1 model_type \"%2\" must be \"onnx\" for ONNX models.")
                           .arg(QDir::toNativeSeparators(qPath), modelTypeValue.toString())
                           .toStdString();
            return std::nullopt;
        }
    }

    const QJsonArray versionsArray = trainingObj.value(QStringLiteral("versions")).toArray();
    if (versionsArray.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 training.versions must list at least one entry.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    std::vector<std::string> versions;
    versions.reserve(versionsArray.size());
    for (int i = 0; i < versionsArray.size(); ++i)
    {
        const QJsonValue value = versionsArray.at(i);
        if (!value.isString() || value.toString().trimmed().isEmpty())
        {
            errorOut = QStringLiteral("Model card %1 training.versions[%2] must be a non-empty string.")
                           .arg(QDir::toNativeSeparators(qPath), QString::number(i))
                           .toStdString();
            return std::nullopt;
        }
        versions.push_back(value.toString().trimmed().toStdString());
    }

    const QJsonObject datasetObj = root.value(QStringLiteral("dataset")).toObject();
    if (datasetObj.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 is missing the dataset block.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }
    const QString datasetId = datasetObj.value(QStringLiteral("id")).toString().trimmed();
    if (datasetId.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 dataset.id must be a non-empty string.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }
    const QString sha256 = datasetObj.value(QStringLiteral("sha256")).toString().trimmed();
    if (!isSha256(sha256))
    {
        errorOut = QStringLiteral("Model card %1 dataset.sha256 must be a 64 character hex string.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    const QString createdAt = root.value(QStringLiteral("created_at")).toString().trimmed();
    if (createdAt.isEmpty())
    {
        errorOut = QStringLiteral("Model card %1 created_at must be a non-empty ISO8601 string.")
                       .arg(QDir::toNativeSeparators(qPath))
                       .toStdString();
        return std::nullopt;
    }

    ModelCard card;
    card.path = cardPath;
    card.schemaVersion = schemaValue.toString().toStdString();
    card.modelType = modelTypeValue.toString().toStdString();
    card.featureCount = static_cast<std::size_t>(featureCountValue);
    card.featureNames = std::move(featureNames);
    card.normalization.mean = mean;
    card.normalization.std = std;
    card.training.framework = framework.toStdString();
    card.training.versions = std::move(versions);
    card.dataset.id = datasetId.toStdString();
    card.dataset.sha256 = sha256.toStdString();
    card.createdAt = createdAt.toStdString();

    return card;
}

} // namespace ai
