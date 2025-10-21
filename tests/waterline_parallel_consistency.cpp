#include "render/Model.h"
#include "tp/waterline/ZSlicer.h"

#include <QtGui/QVector3D>

#include <cassert>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

namespace
{

render::Model buildPocket(double width, double depth, double pocketDepth, int divisions)
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
            const bool insidePocket = (x > width * 0.2 && x < width * 0.8 && y > depth * 0.2 && y < depth * 0.8);
            const double z = insidePocket ? -pocketDepth : 0.0;

            render::Vertex vertex;
            vertex.position = QVector3D(static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(z));
            vertex.normal = QVector3D(0.0f, 0.0f, insidePocket ? 1.0f : -1.0f);
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

void compareLoops(const tp::waterline::ZSlicer& slicer,
                  double planeZ,
                  double toolRadius,
                  bool applyOffset,
                  double tolerance)
{
    const auto sequential = slicer.slice(planeZ,
                                         toolRadius,
                                         applyOffset,
                                         tp::waterline::ZSlicer::SliceMode::Sequential);
    const auto parallel = slicer.slice(planeZ,
                                       toolRadius,
                                       applyOffset,
                                       tp::waterline::ZSlicer::SliceMode::Parallel);

    assert(sequential.size() == parallel.size());
    for (std::size_t i = 0; i < sequential.size(); ++i)
    {
        const auto& seqLoop = sequential[i];
        const auto& parLoop = parallel[i];
        assert(seqLoop.size() == parLoop.size());
        for (std::size_t j = 0; j < seqLoop.size(); ++j)
        {
            const glm::dvec3& a = seqLoop[j];
            const glm::dvec3& b = parLoop[j];
            assert(std::abs(a.x - b.x) <= tolerance);
            assert(std::abs(a.y - b.y) <= tolerance);
            assert(std::abs(a.z - b.z) <= tolerance);
        }
    }
}

} // namespace

int main()
{
    render::Model model = buildPocket(60.0, 60.0, 6.0, 28);
    assert(model.isValid());

    tp::waterline::ZSlicer slicer(model, 1e-4);
    const double tolerance = 1e-6;

    compareLoops(slicer, -1.0, 0.0, false, tolerance);
    compareLoops(slicer, -3.0, 0.75, true, tolerance);
    compareLoops(slicer, -5.5, 1.1, true, tolerance);

    return 0;
}
