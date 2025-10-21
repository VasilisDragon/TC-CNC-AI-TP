#include "ai/OnnxAI.h"
#include "render/Model.h"
#include "tp/ToolpathGenerator.h"

#include <QtGui/QVector3D>

#include <cmath>
#include <filesystem>
#include <vector>

int main()
{
#ifdef AI_WITH_ONNXRUNTIME
    std::vector<render::Vertex> vertices(3);
    vertices[0].position = QVector3D(0.0f, 0.0f, 0.0f);
    vertices[1].position = QVector3D(1.0f, 0.0f, 0.0f);
    vertices[2].position = QVector3D(0.0f, 1.0f, 0.0f);

    std::vector<render::Model::Index> indices = {0, 1, 2};

    render::Model model;
    model.setMeshData(std::move(vertices), std::move(indices));

    tp::UserParams params;
    params.stepOver = 2.5;

    ai::OnnxAI ai(std::filesystem::path());
    ai.setForceCpu(true);
    const ai::StrategyDecision decision = ai.predict(model, params);
    if (decision.steps.empty())
    {
        return 1;
    }
    if (std::abs(decision.steps.front().stepover - params.stepOver) > 1e-6)
    {
        return 1;
    }
#endif
    return 0;
}
