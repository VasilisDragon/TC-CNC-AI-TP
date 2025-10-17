#include "tp/heightfield/HeightField.h"
#include "tp/heightfield/UniformGrid.h"

#include "render/Model.h"

#include <QtGui/QVector3D>

#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>

namespace
{

double planeZ(double x, double y)
{
    return 0.1 * x + 0.2 * y;
}

} // namespace

int main()
{
    render::Model model;

    std::vector<render::Vertex> vertices(4);
    vertices[0].position = QVector3D(0.0f, 0.0f, static_cast<float>(planeZ(0.0, 0.0)));
    vertices[1].position = QVector3D(10.0f, 0.0f, static_cast<float>(planeZ(10.0, 0.0)));
    vertices[2].position = QVector3D(10.0f, 10.0f, static_cast<float>(planeZ(10.0, 10.0)));
    vertices[3].position = QVector3D(0.0f, 10.0f, static_cast<float>(planeZ(0.0, 10.0)));

    // Upward facing normals.
    for (auto& v : vertices)
    {
        v.normal = QVector3D(0.0f, 0.0f, 1.0f);
    }

    std::vector<render::Model::Index> indices = {0, 1, 2, 0, 2, 3};
    model.setMeshData(vertices, indices);

    tp::heightfield::UniformGrid grid(model, 1.0);

    std::atomic<bool> cancel{false};
    tp::heightfield::HeightField heightField;
    tp::heightfield::HeightField::BuildStats stats;
    const bool built = heightField.build(grid, 0.5, cancel, &stats);
    assert(built);
    assert(heightField.isValid());
    assert(stats.validSamples > 0);

    for (double y = 0.0; y <= 10.0; y += 1.0)
    {
        for (double x = 0.0; x <= 10.0; x += 1.0)
        {
            double z = 0.0;
            const bool ok = heightField.interpolate(x, y, z);
            assert(ok);
            const double expected = planeZ(x, y);
            assert(std::abs(z - expected) < 0.05);
        }
    }

    return 0;
}

