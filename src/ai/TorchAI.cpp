#include "ai/TorchAI.h"

#include "render/Model.h"
#include "tp/ToolpathGenerator.h"

#include <QtCore/QDebug>
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
#ifdef AI_WITH_TORCH
    if (!m_loaded)
    {
        return decision;
    }

    try
    {
        std::vector<float> features = buildFeatures(model, params);
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
}

std::vector<float> TorchAI::buildFeatures(const render::Model& model,
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

double TorchAI::computeSurfaceArea(const render::Model& model) const
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
