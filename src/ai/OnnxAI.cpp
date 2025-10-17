#include "ai/OnnxAI.h"

#include "render/Model.h"
#include "tp/ToolpathGenerator.h"

#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QStringList>
#include <QtGui/QVector3D>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{
constexpr double kFallbackAngleDeg = 45.0;

QString toQString(const std::filesystem::path& path)
{
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    return QString::fromStdString(path.string());
#endif
}

std::size_t readInputDimFromSchema(const std::filesystem::path& path)
{
    if (path.empty())
    {
        return 0;
    }

    const QString schemaPath = toQString(path);
    QFile file(schemaPath);
    if (!file.exists())
    {
        return 0;
    }
    if (!file.open(QIODevice::ReadOnly))
    {
        qWarning().noquote() << "OnnxAI: unable to open schema" << schemaPath << "-" << file.errorString();
        return 0;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject())
    {
        return 0;
    }
    const QJsonObject root = doc.object();
    const QJsonObject interface = root.value(QStringLiteral("interface")).toObject();
    const QJsonArray inputs = interface.value(QStringLiteral("inputs")).toArray();
    if (inputs.isEmpty() || !inputs.first().isObject())
    {
        return 0;
    }
    const QJsonObject firstInput = inputs.first().toObject();
    const QJsonValue shapeValue = firstInput.value(QStringLiteral("shape"));
    if (shapeValue.isArray())
    {
        const QJsonArray shapeArray = shapeValue.toArray();
        if (shapeArray.size() >= 2)
        {
            const QJsonValue dimValue = shapeArray.at(1);
            if (dimValue.isDouble())
            {
                return static_cast<std::size_t>(dimValue.toInt());
            }
            if (dimValue.isString())
            {
                bool ok = false;
                const int parsed = dimValue.toString().toInt(&ok);
                if (ok && parsed > 0)
                {
                    return static_cast<std::size_t>(parsed);
                }
            }
        }
    }
    return 0;
}

#ifdef AI_WITH_ONNXRUNTIME
Ort::Env& ortEnv()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CNCTC-Onnx");
    return env;
}

bool hasProvider(const std::vector<std::string>& providers, const char* name)
{
    return std::find(providers.begin(), providers.end(), name) != providers.end();
}
#endif

} // namespace

namespace ai
{

OnnxAI::OnnxAI(std::filesystem::path modelPath)
    : m_modelPath(std::move(modelPath))
{
    if (!m_modelPath.empty())
    {
        loadMetadata();
    }
#ifdef AI_WITH_ONNXRUNTIME
    configureSession();
#else
    if (!m_modelPath.empty())
    {
        qWarning().noquote() << "OnnxAI built without ONNX Runtime support. Running in fallback mode.";
    }
    m_device = "CPU (stub)";
    m_loaded = false;
    m_useCuda = false;
    m_hasCuda = false;
#endif
    m_expectedInputSize = resolveExpectedInputSize();
}

void OnnxAI::setForceCpu(bool forceCpu)
{
    if (m_forceCpu == forceCpu)
    {
        return;
    }
    m_forceCpu = forceCpu;
    configureSession();
}

StrategyDecision OnnxAI::predict(const render::Model& model,
                                 const tp::UserParams& params)
{
    m_lastLatencyMs = 0.0;
    StrategyDecision decision = fallbackDecision(params);
#ifdef AI_WITH_ONNXRUNTIME
    if (!m_loaded || !m_session)
    {
        return decision;
    }

    try
    {
        std::vector<float> features = buildFeatures(model, params);
        std::vector<int64_t> inputShape{1, static_cast<int64_t>(features.size())};

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                                 features.data(),
                                                                 features.size(),
                                                                 inputShape.data(),
                                                                 inputShape.size());

        const char* inputNames[] = {m_inputName.c_str()};

        std::vector<const char*> outputNames;
        if (!m_outputs.logits.empty())
        {
            outputNames.push_back(m_outputs.logits.c_str());
        }
        if (!m_outputs.angle.empty())
        {
            outputNames.push_back(m_outputs.angle.c_str());
        }
        if (!m_outputs.step.empty())
        {
            outputNames.push_back(m_outputs.step.c_str());
        }

        const auto start = std::chrono::steady_clock::now();
        std::vector<Ort::Value> results = m_session->Run(Ort::RunOptions{nullptr},
                                                         inputNames,
                                                         &inputTensor,
                                                         1,
                                                         outputNames.empty() ? nullptr : outputNames.data(),
                                                         outputNames.size());
        const auto end = std::chrono::steady_clock::now();
        m_lastLatencyMs = std::chrono::duration<double, std::milli>(end - start).count();

        std::size_t index = 0;
        auto nextValue = [&](bool hasName) -> Ort::Value* {
            if (!hasName || index >= results.size())
            {
                return nullptr;
            }
            return &results[index++];
        };

        Ort::Value* logitsValue = nextValue(!m_outputs.logits.empty());
        Ort::Value* angleValue = nextValue(!m_outputs.angle.empty());
        Ort::Value* stepValue = nextValue(!m_outputs.step.empty());

        if (logitsValue && logitsValue->IsTensor())
        {
            Ort::TensorTypeAndShapeInfo shapeInfo = logitsValue->GetTensorTypeAndShapeInfo();
            const size_t count = shapeInfo.GetElementCount();
            if (count >= 2)
            {
                const float* data = logitsValue->GetTensorData<float>();
                const double maxLogit = std::max(static_cast<double>(data[0]), static_cast<double>(data[1]));
                const double exp0 = std::exp(static_cast<double>(data[0]) - maxLogit);
                const double exp1 = std::exp(static_cast<double>(data[1]) - maxLogit);
                decision.strat = (exp1 > exp0) ? StrategyDecision::Strategy::Waterline
                                               : StrategyDecision::Strategy::Raster;
            }
        }

        if (angleValue && angleValue->IsTensor())
        {
            Ort::TensorTypeAndShapeInfo info = angleValue->GetTensorTypeAndShapeInfo();
            if (info.GetElementCount() >= 1)
            {
                const float* data = angleValue->GetTensorData<float>();
                decision.rasterAngleDeg = data[0];
            }
        }

        if (stepValue && stepValue->IsTensor())
        {
            Ort::TensorTypeAndShapeInfo info = stepValue->GetTensorTypeAndShapeInfo();
            if (info.GetElementCount() >= 1)
            {
                const float* data = stepValue->GetTensorData<float>();
                const double value = data[0];
                if (value > 0.0)
                {
                    decision.stepOverMM = value;
                }
            }
        }

        if (decision.stepOverMM <= 0.0)
        {
            decision.stepOverMM = params.stepOver;
        }

        m_lastError.clear();
        return decision;
    }
    catch (const Ort::Exception& e)
    {
        m_lastError = e.what();
        qWarning().noquote() << "OnnxAI inference failed:" << QString::fromStdString(m_lastError);
    }
#endif
    return decision;
}

StrategyDecision OnnxAI::fallbackDecision(const tp::UserParams& params) const
{
    StrategyDecision decision;
    decision.strat = StrategyDecision::Strategy::Raster;
    decision.rasterAngleDeg = kFallbackAngleDeg;
    decision.stepOverMM = params.stepOver;
    decision.roughPass = true;
    decision.finishPass = true;
    return decision;
}

void OnnxAI::configureSession()
{
#ifdef AI_WITH_ONNXRUNTIME
    m_loaded = false;
    m_useCuda = false;
    m_device = "CPU";
    m_lastError.clear();
    m_hasCuda = false;
    m_session.reset();

    if (m_modelPath.empty())
    {
        m_expectedInputSize = FeatureExtractor::featureCount() + 2;
        m_warnedFeatureSize = false;
        return;
    }

    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    m_hasCuda = hasProvider(providers, "CUDAExecutionProvider");

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const bool cudaRequested = m_hasCuda && !m_forceCpu;
    if (cudaRequested)
    {
        try
        {
            OrtCUDAProviderOptions cudaOptions;
            std::memset(&cudaOptions, 0, sizeof(cudaOptions));
            options.AppendExecutionProvider_CUDA(cudaOptions);
            m_useCuda = true;
            m_device = "CUDA";
        }
        catch (const Ort::Exception& e)
        {
            qWarning().noquote() << "OnnxAI: CUDA provider unavailable -" << QString::fromStdString(e.what());
            m_useCuda = false;
            m_device = "CPU";
        }
    }

    if (!m_useCuda)
    {
        m_device = (m_forceCpu && m_hasCuda) ? "CPU (forced)" : "CPU";
    }

    try
    {
        m_session = std::make_unique<Ort::Session>(ortEnv(), m_modelPath.c_str(), options);
        m_loaded = true;
        if (!m_useCuda)
        {
            m_device = (m_forceCpu && m_hasCuda) ? "CPU (forced)" : "CPU";
        }
        m_lastError.clear();
    }
    catch (const Ort::Exception& e)
    {
        m_lastError = e.what();
        qWarning().noquote() << "OnnxAI: failed to load" << QString::fromStdString(m_modelPath.u8string())
                             << "-" << QString::fromStdString(m_lastError);
        m_session.reset();
        m_loaded = false;
        m_useCuda = false;
        m_device = (m_forceCpu && m_hasCuda) ? "CPU (forced)" : "CPU";
    }

    m_expectedInputSize = resolveExpectedInputSize();
#else
    m_loaded = false;
    m_useCuda = false;
    m_hasCuda = false;
    m_device = "CPU (stub)";
    m_expectedInputSize = FeatureExtractor::featureCount() + 2;
#endif
    m_warnedFeatureSize = false;
    m_loggedFeaturePreview = false;
}

bool OnnxAI::loadMetadata()
{
    m_metadataPath = m_modelPath;
    m_metadataPath += ".json";

    const QString metaPath = toQString(m_metadataPath);
    QFile file(metaPath);
    if (!file.exists())
    {
        return false;
    }

    if (!file.open(QIODevice::ReadOnly))
    {
        qWarning().noquote() << "OnnxAI: unable to open metadata" << metaPath << "-" << file.errorString();
        return false;
    }

    const QByteArray data = file.readAll();
    const QJsonDocument doc = QJsonDocument::fromJson(data);
    if (!doc.isObject())
    {
        qWarning().noquote() << "OnnxAI: metadata file is not a JSON object" << metaPath;
        return false;
    }

    const QJsonObject root = doc.object();
    if (root.contains(QStringLiteral("input")))
    {
        m_inputName = root.value(QStringLiteral("input")).toString(QStringLiteral("input")).toStdString();
    }

    if (root.contains(QStringLiteral("outputs")) && root.value(QStringLiteral("outputs")).isObject())
    {
        const QJsonObject outputs = root.value(QStringLiteral("outputs")).toObject();
        if (outputs.contains(QStringLiteral("logits")))
        {
            m_outputs.logits = outputs.value(QStringLiteral("logits")).toString(QStringLiteral("logits")).toStdString();
        }
        if (outputs.contains(QStringLiteral("angle")))
        {
            m_outputs.angle = outputs.value(QStringLiteral("angle")).toString(QStringLiteral("angle")).toStdString();
        }
        if (outputs.contains(QStringLiteral("stepover")))
        {
            m_outputs.step = outputs.value(QStringLiteral("stepover")).toString(QStringLiteral("stepover")).toStdString();
        }
    }

    return true;
}

std::size_t OnnxAI::parseExpectedInputSizeFromArtifacts() const
{
    std::vector<std::filesystem::path> candidates;
    if (!m_modelPath.empty())
    {
        candidates.emplace_back(m_modelPath.string() + ".card.json");

        std::filesystem::path baseNoExt = m_modelPath;
        baseNoExt.replace_extension();
        candidates.emplace_back(baseNoExt.string() + ".card.json");
        candidates.emplace_back(baseNoExt.string() + ".onnx.json");

        std::filesystem::path schemaPath = m_modelPath;
        schemaPath += ".onnx.json";
        candidates.emplace_back(schemaPath);
    }

    if (!m_metadataPath.empty())
    {
        candidates.emplace_back(m_metadataPath);
    }

    for (const auto& candidate : candidates)
    {
        if (const auto size = readInputDimFromSchema(candidate); size > 0)
        {
            return size;
        }
    }

    return 0;
}

std::size_t OnnxAI::resolveExpectedInputSize() const
{
    std::size_t expected = parseExpectedInputSizeFromArtifacts();

#ifdef AI_WITH_ONNXRUNTIME
    if (expected == 0 && m_session)
    {
        try
        {
            Ort::TypeInfo typeInfo = m_session->GetInputTypeInfo(0);
            Ort::TensorTypeAndShapeInfo shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = shapeInfo.GetShape();
            if (shape.size() >= 2 && shape[1] > 0)
            {
                expected = static_cast<std::size_t>(shape[1]);
            }
        }
        catch (const Ort::Exception& e)
        {
            qWarning().noquote() << "OnnxAI: unable to query input shape -" << QString::fromStdString(e.what());
        }
    }
#endif

    if (expected == 0)
    {
        expected = FeatureExtractor::featureCount() + 2;
    }

    return expected;
}

std::vector<float> OnnxAI::alignFeatureVector(std::vector<float>&& input) const
{
    if (m_expectedInputSize == 0 || input.size() == m_expectedInputSize)
    {
        return std::move(input);
    }

    if (!m_warnedFeatureSize)
    {
        m_warnedFeatureSize = true;
        const QString action = input.size() < m_expectedInputSize ? QStringLiteral("padding with zeros.")
                                                                  : QStringLiteral("truncating.");
        qWarning().noquote() << "OnnxAI: feature vector size mismatch (expected"
                             << static_cast<int>(m_expectedInputSize) << "received"
                             << static_cast<int>(input.size()) << ") -" << action;
    }

    std::vector<float> adjusted = std::move(input);
    if (adjusted.size() < m_expectedInputSize)
    {
        adjusted.resize(m_expectedInputSize, 0.0f);
    }
    else
    {
        adjusted.resize(m_expectedInputSize);
    }
    return adjusted;
}

void OnnxAI::logFeaturePreview(const std::vector<float>& features) const
{
    if (m_loggedFeaturePreview)
    {
        return;
    }
    m_loggedFeaturePreview = true;

    QStringList previewValues;
    const std::size_t previewCount = std::min<std::size_t>(features.size(), 6);
    for (std::size_t i = 0; i < previewCount; ++i)
    {
        previewValues << QString::number(features[i], 'f', 3);
    }

    qInfo().noquote() << "OnnxAI: feature length" << static_cast<int>(features.size())
                      << "preview [" << previewValues.join(QStringLiteral(", ")) << "]";
}

std::vector<float> OnnxAI::buildFeatures(const render::Model& model,
                                         const tp::UserParams& params) const
{
    const auto global = FeatureExtractor::computeGlobalFeatures(model);
    std::vector<float> features = FeatureExtractor::toVector(global);
    features.reserve(features.size() + 2);
    features.push_back(static_cast<float>(params.stepOver));
    features.push_back(static_cast<float>(params.toolDiameter));
    logFeaturePreview(features);
    return alignFeatureVector(std::move(features));
}

} // namespace ai
