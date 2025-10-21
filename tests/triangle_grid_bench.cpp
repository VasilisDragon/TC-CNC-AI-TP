
#include "render/Model.h"
#include "tp/TriangleGrid.h"

#include "common/log.h"

#include <QtGui/QVector3D>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <vector>

// Micro-benchmark: TriangleGrid candidate gathering.
// Build: cmake --build <build-dir> --target triangle_grid_bench
// Run:   triangle_grid_bench [iterations]

namespace
{

render::Model makeTestPlate(int gridResolution, double size)
{
    const int samples = gridResolution + 1;
    std::vector<render::Vertex> vertices(static_cast<std::size_t>(samples * samples));
    std::vector<render::Model::Index> indices;
    indices.reserve(static_cast<std::size_t>(gridResolution * gridResolution * 6));

    for (int y = 0; y < samples; ++y)
    {
        for (int x = 0; x < samples; ++x)
        {
            const double px = (static_cast<double>(x) / gridResolution) * size;
            const double py = (static_cast<double>(y) / gridResolution) * size;
            render::Vertex vertex;
            vertex.position = QVector3D(static_cast<float>(px), static_cast<float>(py), 0.0f);
            vertex.normal = QVector3D(0.0f, 0.0f, 1.0f);
            vertices[static_cast<std::size_t>(y * samples + x)] = vertex;
        }
    }

    for (int y = 0; y < gridResolution; ++y)
    {
        for (int x = 0; x < gridResolution; ++x)
        {
            const int base = y * samples + x;
            indices.push_back(static_cast<render::Model::Index>(base));
            indices.push_back(static_cast<render::Model::Index>(base + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples + 1));

            indices.push_back(static_cast<render::Model::Index>(base));
            indices.push_back(static_cast<render::Model::Index>(base + samples + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples));
        }
    }

    render::Model model;
    model.setMeshData(std::move(vertices), std::move(indices));
    return model;
}

} // namespace

int main(int argc, char** argv)
{
    const int iterations = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 200000;

    render::Model model = makeTestPlate(64, 128.0);
    if (!model.isValid())
    {
        LOG_ERR(Render,
                "Triangle grid benchmark failed: generated plate model is invalid. "
                "Verify the mesh parameters and rerun the benchmark.");
        return 1;
    }

    tp::TriangleGrid grid(model, 1.0);
    if (grid.triangleCount() == 0)
    {
        LOG_ERR(Render,
                "Triangle grid benchmark failed: triangle grid is empty. "
                "Check the model geometry or adjust the resolution, then retry.");
        return 1;
    }

    std::vector<std::uint32_t> scratch;
    scratch.reserve(128);

    double accumCandidates = 0.0;

    const auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        const double x = std::fmod(static_cast<double>(i), 128.0);
        const double y = std::fmod(static_cast<double>(i) * 0.61803398875, 128.0);
        grid.gatherCandidatesXY(x, y, 1, scratch);
        accumCandidates += static_cast<double>(scratch.size());
        scratch.clear();
    }

    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
    const double avgCandidates = accumCandidates / static_cast<double>(iterations);

    std::ostringstream summary;
    summary << "Triangle grid benchmark completed: iterations=" << iterations
            << ", elapsed_ms=" << ms
            << ", avg_candidates=" << avgCandidates
            << ". Use the report to tune gather parameters.";
    LOG_INFO(Render, summary.str());

    return 0;
}
