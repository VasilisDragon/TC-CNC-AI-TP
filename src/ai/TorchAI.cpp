#include "ai/TorchAI.h"

#include "render/Model.h"
#include "tp/ToolpathGenerator.h"

#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QStringList>
#include <QtGui/QVector3D>

#include <chrono>
#include <utility>
#include <vector>

#ifdef AI_WITH_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

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
        qWarning().noquote() << "TorchAI: unable to open schema" << schemaPath << "-" << file.errorString();
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

#ifdef AI_WITH_TORCH
void assignTensor(torch::Tensor& target, const c10::IValue& value)
{
    if (value.isTensor())
    {
        target = value.toTensor();
    }
}
#endif

} // namespace

namespace ai
{

TorchAI::TorchAI(std::filesystem::path modelPath)
    : m_modelPath(std::move(modelPath))
{
#ifdef AI_WITH_TORCH
    if (!m_modelPath.empty())
    {
        try
        {
            m_module = torch::jit::load(m_modelPath.string());
            m_module.eval();
            m_loaded = true;
            m_lastError.clear();
        }
        catch (const c10::Error& e)
        {
            m_lastError = e.what_without_backtrace();
            m_loaded = false;
            qWarning().noquote() << "TorchAI: failed to load"
                                 << QString::fromStdString(m_modelPath.u8string())
                                 << "-" << QString::fromStdString(m_lastError);
        }
    }
    else
    {
        m_loaded = false;
    }
    m_hasCuda = torch::cuda::is_available();
#else
    if (!m_modelPath.empty())
    {
        qWarning().noquote() << "TorchAI built without LibTorch support. Running in fallback mode.";
    }
    m_loaded = false;
    m_hasCuda = false;
    m_device = "CPU (stub)";
#endif
    m_expectedInputSize = resolveExpectedInputSize();
    configureDevice();
}

void TorchAI::setForceCpu(bool forceCpu)
{
    if (m_forceCpu == forceCpu)
    {
        return;
    }
    m_forceCpu = forceCpu;
    configureDevice();
}

StrategyDecision TorchAI::predict(const render::Model& model,
                                  const tp::UserParams& params)
{
    m_lastLatencyMs = 0.0;
    StrategyDecision decision = fallbackDecision(params);

    auto featuresOpt = buildFeatures(model, params);
    if (!featuresOpt)
    {
        m_lastError = "Feature extraction produced an invalid descriptor.";
        qWarning().noquote() << "TorchAI: feature extraction failed, falling back to heuristics.";
        return decision;
    }

#ifdef AI_WITH_TORCH
    if (!m_loaded)
    {
        return decision;
    }

    try
    {
        std::vector<float> features = std::move(*featuresOpt);
        torch::Tensor input = torch::from_blob(features.data(),
                                               {1, static_cast<long>(features.size())},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                 .clone();
        if (m_useCuda)
        {
            input = input.to(m_deviceObject);
        }

        torch::NoGradGuard guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input);

        const auto start = std::chrono::steady_clock::now();
        torch::jit::IValue output = m_module.forward(inputs);
        const auto end = std::chrono::steady_clock::now();
        m_lastLatencyMs = std::chrono::duration<double, std::milli>(end - start).count();

        torch::Tensor logits;
        torch::Tensor angleTensor;
        torch::Tensor stepTensor;

        if (output.isGenericDict())
        {
            auto dict = output.toGenericDict();
            const c10::IValue logitsKey(std::string("logits"));
            const c10::IValue angleKey(std::string("angle"));
            const c10::IValue stepKey(std::string("stepover"));
            if (dict.contains(logitsKey))
            {
                assignTensor(logits, dict.at(logitsKey));
            }
            if (dict.contains(angleKey))
            {
                assignTensor(angleTensor, dict.at(angleKey));
            }
            if (dict.contains(stepKey))
            {
                assignTensor(stepTensor, dict.at(stepKey));
            }
        }
        else if (output.isTuple())
        {
            const auto& elements = output.toTuple()->elements();
            if (elements.size() >= 1)
            {
                assignTensor(logits, elements[0]);
            }
            if (elements.size() >= 2)
            {
                assignTensor(angleTensor, elements[1]);
            }
            if (elements.size() >= 3)
            {
                assignTensor(stepTensor, elements[2]);
            }
        }
        else if (output.isList())
        {
            const auto list = output.toList();
            const auto count = list.size();
            if (count >= 1)
            {
                assignTensor(logits, list.get(0));
            }
            if (count >= 2)
            {
                assignTensor(angleTensor, list.get(1));
            }
            if (count >= 3)
            {
                assignTensor(stepTensor, list.get(2));
            }
        }
        else if (output.isTensor())
        {
            auto tensor = output.toTensor().to(torch::kCPU).flatten();
            if (tensor.size(0) >= 4)
            {
                logits = tensor.slice(0, 0, 2);
                angleTensor = tensor.slice(0, 2, 3);
                stepTensor = tensor.slice(0, 3, 4);
            }
        }

        if (logits.defined())
        {
            auto cpuLogits = logits.to(torch::kCPU).flatten();
            if (cpuLogits.size(0) == 2)
            {
                const auto probs = torch::softmax(cpuLogits, 0);
                const int stratIndex = probs.argmax().item<int>();
                decision.strat = stratIndex == 0 ? StrategyDecision::Strategy::Raster
                                                 : StrategyDecision::Strategy::Waterline;
            }
        }

        if (angleTensor.defined())
        {
            decision.rasterAngleDeg = angleTensor.to(torch::kCPU).item<double>();
        }

        if (stepTensor.defined())
        {
            const double proposedStep = stepTensor.to(torch::kCPU).item<double>();
            if (proposedStep > 0.0)
            {
                decision.stepOverMM = proposedStep;
            }
        }

        if (decision.stepOverMM <= 0.0)
        {
            decision.stepOverMM = params.stepOver;
        }

        m_lastError.clear();
        return decision;
    }
    catch (const c10::Error& e)
    {
        m_lastError = e.what_without_backtrace();
        qWarning().noquote() << "TorchAI inference failed:" << QString::fromStdString(m_lastError);
    }
#else
    (void)featuresOpt;
#endif
    return decision;
}

StrategyDecision TorchAI::fallbackDecision(const tp::UserParams& params) const
{
    StrategyDecision decision;
    decision.strat = StrategyDecision::Strategy::Raster;
    decision.rasterAngleDeg = kFallbackAngleDeg;
    decision.stepOverMM = params.stepOver;
    decision.roughPass = true;
    decision.finishPass = true;
    return decision;
}

void TorchAI::configureDevice()
{
#ifdef AI_WITH_TORCH
    m_useCuda = false;

    if (!m_loaded)
    {
        m_deviceObject = torch::Device(torch::kCPU);
        m_device = (m_forceCpu && m_hasCuda) ? "CPU (forced)" : "CPU";
        return;
    }

    const bool wantCuda = !m_forceCpu && m_hasCuda;
    torch::Device desiredDevice = wantCuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::string label = "CPU";
    if (wantCuda)
    {
        label = "CUDA";
    }
    else if (m_forceCpu && m_hasCuda)
    {
        label = "CPU (forced)";
    }

    auto moveToDevice = [&](const torch::Device& device, const std::string& name) -> bool {
        try
        {
            m_module.to(device);
            m_deviceObject = device;
            m_useCuda = device.is_cuda();
            m_device = name;
            m_lastError.clear();
            return true;
        }
        catch (const c10::Error& e)
        {
            m_lastError = e.what_without_backtrace();
            qWarning().noquote() << "TorchAI: device configuration failed -"
                                 << QString::fromStdString(m_lastError);
            return false;
        }
    };

    if (!moveToDevice(desiredDevice, label) && desiredDevice.is_cuda())
    {
        if (!moveToDevice(torch::Device(torch::kCPU), "CPU"))
        {
            m_deviceObject = torch::Device(torch::kCPU);
            m_device = "CPU";
        }
        m_useCuda = false;
    }
#else
    m_device = "CPU (stub)";
    m_useCuda = false;
    m_hasCuda = false;
#endif
    m_loggedFeaturePreview = false;
}

std::size_t TorchAI::parseExpectedInputSizeFromArtifacts() const
{
    if (m_modelPath.empty())
    {
        return 0;
    }

    std::vector<std::filesystem::path> candidates;
    candidates.emplace_back(m_modelPath.string() + ".card.json");

    std::filesystem::path baseNoExt = m_modelPath;
    baseNoExt.replace_extension();
    candidates.emplace_back(baseNoExt.string() + ".card.json");
    candidates.emplace_back(baseNoExt.string() + ".onnx.json");

    std::filesystem::path schemaPath = m_modelPath;
    schemaPath += ".onnx.json";
    candidates.emplace_back(schemaPath);

    for (const auto& candidate : candidates)
    {
        if (const auto size = readInputDimFromSchema(candidate); size > 0)
        {
            return size;
        }
    }

    return 0;
}

std::size_t TorchAI::resolveExpectedInputSize()
{
    std::size_t expected = parseExpectedInputSizeFromArtifacts();

#ifdef AI_WITH_TORCH
    if (expected == 0 && m_loaded)
    {
        try
        {
            for (const auto& named : m_module.named_parameters())
            {
                const torch::Tensor& tensor = named.value;
                if (tensor.dim() == 2)
                {
                    expected = static_cast<std::size_t>(tensor.size(1));
                    break;
                }
            }
        }
        catch (const c10::Error& e)
        {
            qWarning().noquote() << "TorchAI: unable to infer input size from parameters -"
                                 << QString::fromStdString(e.what_without_backtrace());
        }
    }
#endif

    if (expected == 0)
    {
        expected = FeatureExtractor::featureCount() + 2; // default to current feature set
    }

    return expected;
}

std::vector<float> TorchAI::alignFeatureVector(std::vector<float>&& input) const
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
        qWarning().noquote() << "TorchAI: feature vector size mismatch (expected"
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

void TorchAI::logFeaturePreview(const std::vector<float>& features) const
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

    qInfo().noquote() << "TorchAI: feature length" << static_cast<int>(features.size())
                      << "preview [" << previewValues.join(QStringLiteral(", ")) << "]";
}

std::optional<std::vector<float>> TorchAI::buildFeatures(const render::Model& model,
                                                         const tp::UserParams& params) const
{
    const auto global = FeatureExtractor::computeGlobalFeatures(model);
    if (!global.valid)
    {
        return std::nullopt;
    }
    std::vector<float> features = FeatureExtractor::toVector(global);
    features.reserve(features.size() + 2);
    features.push_back(static_cast<float>(params.stepOver));
    features.push_back(static_cast<float>(params.toolDiameter));
    logFeaturePreview(features);
    return std::make_optional(alignFeatureVector(std::move(features)));
}

} // namespace ai
