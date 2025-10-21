#include "render/Model.h"
#include "tp/GougeChecker.h"
#include "tp/TriangleGrid.h"
#include "tp/heightfield/UniformGrid.h"

#include <QtGui/QVector3D>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

render::Model makeFlatPlate(double size)
{
    render::Model model;

    std::vector<render::Vertex> vertices(4);
    vertices[0].position = QVector3D(0.0f, 0.0f, 0.0f);
    vertices[1].position = QVector3D(static_cast<float>(size), 0.0f, 0.0f);
    vertices[2].position = QVector3D(static_cast<float>(size), static_cast<float>(size), 0.0f);
    vertices[3].position = QVector3D(0.0f, static_cast<float>(size), 0.0f);

    for (auto& v : vertices)
    {
        v.normal = QVector3D(0.0f, 0.0f, 1.0f);
    }

    std::vector<render::Model::Index> indices = {
        0, 1, 2,
        0, 2, 3
    };

    model.setMeshData(std::move(vertices), std::move(indices));
    return model;
}

} // namespace

int main()
{
    constexpr double kSize = 10.0;
    render::Model model = makeFlatPlate(kSize);
    assert(model.isValid());

    tp::TriangleGrid grid(model, 1.0);
    assert(!grid.empty());
    assert(grid.triangleCount() == 2);

    std::vector<std::uint32_t> candidates;
    grid.gatherCandidatesXY(5.0, 5.0, 0, candidates);
    assert(!candidates.empty());

    grid.gatherCandidatesAABB(0.0, 0.0, kSize, kSize, candidates);
    assert(candidates.size() == 2);

    tp::heightfield::UniformGrid uniform(model, 1.0);
    double sampleZ = 0.0;
    assert(uniform.sampleMaxZAtXY(2.5, 2.5, sampleZ));
    assert(std::abs(sampleZ) < 1e-8);

    tp::GougeChecker checker(model);
    tp::GougeChecker::Vec3 testPoint{2.5f, 2.5f, 5.0f};
    const auto surfaceZ = checker.surfaceHeightAt(testPoint);
    assert(surfaceZ);
    assert(std::abs(*surfaceZ) < 1e-6);

    std::vector<tp::GougeChecker::Vec3> path = {
        {0.0f, 0.0f, 5.0f},
        {10.0f, 10.0f, 5.0f}
    };
    tp::GougeParams params;
    params.toolRadius = 1.0;
    const double clearance = checker.minClearanceAlong(path, params);
    assert(std::abs(clearance - 5.0) < 1e-6);

    return 0;
}

