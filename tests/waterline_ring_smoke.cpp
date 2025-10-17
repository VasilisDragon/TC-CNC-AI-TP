#include "tp/waterline/ZSlicer.h"

#include "render/Model.h"

#include <QtGui/QVector3D>

#include <atomic>
#include <cassert>
#include <numbers>
#include <vector>

namespace
{

double majorRadius() { return 10.0; }
double minorRadius() { return 2.0; }

} // namespace

int main()
{
    const int slices = 48;
    const int stacks = 24;

    std::vector<render::Vertex> vertices;
    vertices.reserve(static_cast<std::size_t>(slices * stacks));

    for (int i = 0; i < slices; ++i)
    {
        const double u = (static_cast<double>(i) / slices) * 2.0 * std::numbers::pi;
        const double cu = std::cos(u);
        const double su = std::sin(u);

        for (int j = 0; j < stacks; ++j)
        {
            const double v = (static_cast<double>(j) / stacks) * 2.0 * std::numbers::pi;
            const double cv = std::cos(v);
            const double sv = std::sin(v);

            const double r = majorRadius();
            const double t = minorRadius();

            const double x = (r + t * cv) * cu;
            const double y = (r + t * cv) * su;
            const double z = t * sv;

            render::Vertex vert;
            vert.position = QVector3D(static_cast<float>(x),
                                      static_cast<float>(y),
                                      static_cast<float>(z));
            const double nx = cv * cu;
            const double ny = cv * su;
            const double nz = sv;
            vert.normal = QVector3D(static_cast<float>(nx),
                                    static_cast<float>(ny),
                                    static_cast<float>(nz)).normalized();
            vertices.push_back(vert);
        }
    }

    std::vector<render::Model::Index> indices;
    for (int i = 0; i < slices; ++i)
    {
        const int nextI = (i + 1) % slices;
        for (int j = 0; j < stacks; ++j)
        {
            const int nextJ = (j + 1) % stacks;

            const int idx0 = i * stacks + j;
            const int idx1 = nextI * stacks + j;
            const int idx2 = nextI * stacks + nextJ;
            const int idx3 = i * stacks + nextJ;

            indices.push_back(static_cast<render::Model::Index>(idx0));
            indices.push_back(static_cast<render::Model::Index>(idx1));
            indices.push_back(static_cast<render::Model::Index>(idx2));

            indices.push_back(static_cast<render::Model::Index>(idx0));
            indices.push_back(static_cast<render::Model::Index>(idx2));
            indices.push_back(static_cast<render::Model::Index>(idx3));
        }
    }

    render::Model model;
    model.setMeshData(std::move(vertices), std::move(indices));

    tp::waterline::ZSlicer slicer(model, 1e-4);

    const std::vector<double> probeZ = {-1.5, 0.0, 1.5};
    for (double z : probeZ)
    {
        const auto loops = slicer.slice(z, 0.0, false);
        assert(!loops.empty());
        for (const auto& loop : loops)
        {
            assert(loop.size() >= 3);
            assert(std::abs(loop.front().z - z) < 1e-6);
            assert(std::abs(loop.back().z - z) < 1e-6);
            assert(std::abs(loop.front().x - loop.back().x) < 1e-6);
            assert(std::abs(loop.front().y - loop.back().y) < 1e-6);
        }
    }

    return 0;
}
