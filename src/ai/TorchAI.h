#pragma once

#include "ai/IPathAI.h"

#include <filesystem>
#include <string>
#include <vector>

#ifdef AI_WITH_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace ai
{

class TorchAI : public IPathAI
{
public:
    explicit TorchAI(std::filesystem::path modelPath);

    StrategyDecision predict(const render::Model& model,
                             const tp::UserParams& params) override;

    void setForceCpu(bool forceCpu);
    [[nodiscard]] bool forceCpu() const noexcept { return m_forceCpu; }
    [[nodiscard]] double lastLatencyMs() const noexcept { return m_lastLatencyMs; }

    [[nodiscard]] const std::filesystem::path& modelPath() const noexcept { return m_modelPath; }
    [[nodiscard]] bool isLoaded() const noexcept { return m_loaded; }
    [[nodiscard]] bool usesCuda() const noexcept { return m_useCuda; }
    [[nodiscard]] std::string device() const noexcept { return m_device; }
    [[nodiscard]] std::string lastError() const noexcept { return m_lastError; }
    [[nodiscard]] bool hasCudaSupport() const noexcept { return m_hasCuda; }

private:
    StrategyDecision fallbackDecision(const tp::UserParams& params) const;
    void configureDevice();
    std::vector<float> buildFeatures(const render::Model& model,
                                     const tp::UserParams& params) const;
    double computeSurfaceArea(const render::Model& model) const;

    std::filesystem::path m_modelPath;
    bool m_loaded{false};
    bool m_forceCpu{false};
    bool m_useCuda{false};
    bool m_hasCuda{false};
    std::string m_device{"CPU"};
    std::string m_lastError;
    double m_lastLatencyMs{0.0};

#ifdef AI_WITH_TORCH
    torch::jit::script::Module m_module;
    torch::Device m_deviceObject{torch::kCPU};
#endif
};

} // namespace ai

