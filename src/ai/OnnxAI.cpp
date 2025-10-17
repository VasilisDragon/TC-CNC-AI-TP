#include "ai/OnnxAI.h"

#include "render/Model.h"
#include "tp/ToolpathGenerator.h"

#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
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
        return;
    }

    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    m_hasCuda = hasProvider(providers, "CUDAExecutionProvider");

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    bool cudaRequested = m_hasCuda && !m_forceCpu;
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
#else
    m_loaded = false;
    m_useCuda = false;
    m_hasCuda = false;
    m_device = "CPU (stub)";
#endif
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

std::vector<float> OnnxAI::buildFeatures(const render::Model& model,
                                         const tp::UserParams& params) const
{
    std::vector<float> features;
    features.reserve(6);

    const auto bounds = model.bounds();
    const QVector3D size = bounds.size();
    features.push_back(size.x());
    features.push_back(size.y());
    features.push_back(size.z());

    const double area = computeSurfaceArea(model);
    features.push_back(static_cast<float>(area));

    features.push_back(static_cast<float>(params.stepOver));
    features.push_back(static_cast<float>(params.toolDiameter));

    return features;
}

double OnnxAI::computeSurfaceArea(const render::Model& model) const
{
    const auto& vertices = model.vertices();
    const auto& indices = model.indices();
    if (vertices.empty() || indices.size() < 3)
    {
        return 0.0;
    }

    double area = 0.0;
    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const auto i0 = indices[i];
        const auto i1 = indices[i + 1];
        const auto i2 = indices[i + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
        {
            continue;
        }

        const QVector3D a = vertices[i0].position;
        const QVector3D b = vertices[i1].position;
        const QVector3D c = vertices[i2].position;
        area += 0.5 * QVector3D::crossProduct(b - a, c - a).length();
    }
    return area;
}

} // namespace ai
