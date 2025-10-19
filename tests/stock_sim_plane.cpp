#include "sim/StockGrid.h"

#include "render/Model.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include <glm/vec3.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace
{

double planeZ(double x, double y)
{
    return 0.08 * x + 0.05 * y;
}

render::Model buildPlaneModel(double width, double depth, int divisions)
{
    render::Model model;

    const int samples = divisions + 1;
    std::vector<render::Vertex> vertices(samples * samples);
    const double stepX = width / static_cast<double>(divisions);
    const double stepY = depth / static_cast<double>(divisions);

    for (int row = 0; row < samples; ++row)
    {
        for (int col = 0; col < samples; ++col)
        {
            const double x = static_cast<double>(col) * stepX;
            const double y = static_cast<double>(row) * stepY;
            const double z = planeZ(x, y);

            render::Vertex vertex;
            vertex.position = QVector3D(static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(z));
            vertex.normal = QVector3D(0.0f, 0.0f, 1.0f);
            vertices[row * samples + col] = vertex;
        }
    }

    std::vector<render::Model::Index> indices;
    indices.reserve(divisions * divisions * 6);
    for (int row = 0; row < divisions; ++row)
    {
        for (int col = 0; col < divisions; ++col)
        {
            const int base = row * samples + col;
            indices.push_back(static_cast<render::Model::Index>(base));
            indices.push_back(static_cast<render::Model::Index>(base + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples));
            indices.push_back(static_cast<render::Model::Index>(base + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples));
        }
    }

    model.setMeshData(std::move(vertices), std::move(indices));
    return model;
}

tp::Toolpath buildPlaneToolpath(double width, double depth, int rows, int cols)
{
    tp::Toolpath toolpath;
    toolpath.feed = 1200.0;
    toolpath.spindle = 12000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.rapidFeed = toolpath.machine.rapidFeed_mm_min;
    toolpath.stock = tp::makeDefaultStock();

    const double stepY = depth / static_cast<double>(rows);
    const double stepX = width / static_cast<double>(cols);

    for (int row = 0; row <= rows; ++row)
    {
        tp::Polyline poly;
        poly.motion = tp::MotionType::Cut;

        const double y = static_cast<double>(row) * stepY;
        for (int col = 0; col <= cols; ++col)
        {
            const double x = static_cast<double>(col) * stepX;
            const double z = planeZ(x, y);
            poly.pts.push_back({glm::vec3(static_cast<float>(x),
                                          static_cast<float>(y),
                                          static_cast<float>(z))});
        }

        if ((row % 2) == 1)
        {
            std::reverse(poly.pts.begin(), poly.pts.end());
        }
        toolpath.passes.push_back(std::move(poly));
    }

    return toolpath;
}

} // namespace

int main()
{
    constexpr double kWidth = 40.0;
    constexpr double kDepth = 30.0;
    constexpr int kDivisions = 24;

    render::Model model = buildPlaneModel(kWidth, kDepth, kDivisions);
    assert(model.isValid());

    tp::Toolpath toolpath = buildPlaneToolpath(kWidth, kDepth, kDivisions, kDivisions * 2);
    assert(!toolpath.empty());

    tp::UserParams params;
    params.toolDiameter = 6.0;
    params.cutterType = tp::UserParams::CutterType::FlatEndmill;

    constexpr double kCellSize = 0.5;
    sim::StockGrid grid(model, kCellSize, 1.5);
    grid.subtractToolpath(toolpath, params);
    sim::StockGridSummary summary = grid.summarize();

    assert(!summary.samples.empty());

    const double tolerance = kCellSize * 1.5 + 1e-3;
    assert(summary.maxError <= tolerance + 1e-6);
    assert(summary.minError >= -1e-6);

    return 0;
}
