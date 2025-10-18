#include "tp/waterline/ZSlicer.h"

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace tp::waterline
{

namespace
{
constexpr double kEpsilon = 1e-9;
template <typename Vec>
inline auto lengthSquared(const Vec& v) -> decltype(glm::dot(v, v))
{
    return glm::dot(v, v);
}


struct GridKey
{
    std::int64_t x{0};
    std::int64_t y{0};

    bool operator==(const GridKey& other) const noexcept
    {
        return x == other.x && y == other.y;
    }
};

struct GridHash
{
    std::size_t operator()(const GridKey& key) const noexcept
    {
        return std::hash<std::int64_t>{}(key.x) ^ (std::hash<std::int64_t>{}(key.y) << 1);
    }
};

GridKey quantize(const glm::dvec2& p, double tolerance)
{
    const double scale = (tolerance > kEpsilon) ? (1.0 / tolerance) : 1.0e6;
    return {
        static_cast<std::int64_t>(std::llround(p.x * scale)),
        static_cast<std::int64_t>(std::llround(p.y * scale))
    };
}

double polygonArea(const std::vector<glm::dvec2>& pts)
{
    if (pts.size() < 3)
    {
        return 0.0;
    }

    double area = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        const glm::dvec2& a = pts[i];
        const glm::dvec2& b = pts[(i + 1) % pts.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return 0.5 * area;
}

glm::dvec2 rotateCW(const glm::dvec2& v)
{
    return {v.y, -v.x};
}

glm::dvec2 rotateCCW(const glm::dvec2& v)
{
    return {-v.y, v.x};
}

bool nearlyEqual2D(const glm::dvec2& a, const glm::dvec2& b, double tol)
{
    return lengthSquared(a - b) <= tol * tol;
}

glm::dvec2 normalForEdge(const glm::dvec2& edge, double area)
{
    const glm::dvec2 unit = glm::normalize(edge);
    if (std::abs(area) < kEpsilon)
    {
        return rotateCW(unit);
    }
    return (area > 0.0) ? rotateCW(unit) : rotateCCW(unit);
}

std::vector<glm::dvec2> offsetLoop(const std::vector<glm::dvec2>& loop, double radius, double area)
{
    if (radius <= kEpsilon || loop.size() < 3)
    {
        return loop;
    }

    const std::size_t count = loop.size();
    std::vector<glm::dvec2> offsetPoints;
    offsetPoints.reserve(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        const glm::dvec2& prev = loop[(i + count - 1) % count];
        const glm::dvec2& curr = loop[i];
        const glm::dvec2& next = loop[(i + 1) % count];

        glm::dvec2 vPrev = curr - prev;
        glm::dvec2 vNext = next - curr;
        if (lengthSquared(vPrev) < kEpsilon || lengthSquared(vNext) < kEpsilon)
        {
            offsetPoints.push_back(curr);
            continue;
        }

        vPrev = glm::normalize(vPrev);
        vNext = glm::normalize(vNext);

        glm::dvec2 nPrev = normalForEdge(vPrev, area);
        glm::dvec2 nNext = normalForEdge(vNext, area);

        glm::dvec2 bisector = nPrev + nNext;
        if (lengthSquared(bisector) < kEpsilon)
        {
            bisector = nPrev;
        }
        bisector = glm::normalize(bisector);

        const double denom = glm::dot(bisector, nPrev);
        const double scale = (std::abs(denom) > kEpsilon) ? (radius / denom) : radius;
        offsetPoints.push_back(curr + bisector * scale);
    }

    return offsetPoints;
}

} // namespace

ZSlicer::ZSlicer(const render::Model& model, double toleranceMm)
    : m_tolerance(std::max(1e-6, toleranceMm))
{
    const auto& vertices = model.vertices();
    const auto& indices = model.indices();
    m_triangles.reserve(indices.size() / 3);

    m_minZ = std::numeric_limits<double>::infinity();
    m_maxZ = -std::numeric_limits<double>::infinity();

    for (std::size_t tri = 0; tri + 2 < indices.size(); tri += 3)
    {
        const auto i0 = indices[tri + 0];
        const auto i1 = indices[tri + 1];
        const auto i2 = indices[tri + 2];

        Triangle t;
        t.v0 = glm::dvec3(vertices[i0].position.x(), vertices[i0].position.y(), vertices[i0].position.z());
        t.v1 = glm::dvec3(vertices[i1].position.x(), vertices[i1].position.y(), vertices[i1].position.z());
        t.v2 = glm::dvec3(vertices[i2].position.x(), vertices[i2].position.y(), vertices[i2].position.z());
        t.minZ = std::min({t.v0.z, t.v1.z, t.v2.z});
        t.maxZ = std::max({t.v0.z, t.v1.z, t.v2.z});

        m_minZ = std::min(m_minZ, t.minZ);
        m_maxZ = std::max(m_maxZ, t.maxZ);

        m_triangles.push_back(t);
    }

    if (m_triangles.empty())
    {
        m_minZ = 0.0;
        m_maxZ = 0.0;
    }
}

std::vector<std::vector<glm::dvec3>> ZSlicer::slice(double planeZ,
                                                     double toolRadius,
                                                     bool applyOffsetForFlat) const
{
    struct Segment
    {
        glm::dvec2 a;
        glm::dvec2 b;
        bool used{false};
    };

    std::vector<Segment> segments;
    segments.reserve(256);

    const auto addSegment = [&](const glm::dvec2& a, const glm::dvec2& b) {
        if (lengthSquared(a - b) <= m_tolerance * m_tolerance)
        {
            return;
        }
        segments.push_back({a, b, false});
    };

    for (const Triangle& tri : m_triangles)
    {
        if (planeZ < tri.minZ - m_tolerance || planeZ > tri.maxZ + m_tolerance)
        {
            continue;
        }

        std::vector<glm::dvec3> intersections;
        intersections.reserve(4);

        const std::array<glm::dvec3, 3> verts = {tri.v0, tri.v1, tri.v2};
        const std::array<int, 3> edgesStart = {0, 1, 2};
        const std::array<int, 3> edgesEnd = {1, 2, 0};

        for (std::size_t e = 0; e < 3; ++e)
        {
            const glm::dvec3& v0 = verts[edgesStart[e]];
            const glm::dvec3& v1 = verts[edgesEnd[e]];
            const double d0 = v0.z - planeZ;
            const double d1 = v1.z - planeZ;

            const bool on0 = std::abs(d0) <= m_tolerance;
            const bool on1 = std::abs(d1) <= m_tolerance;

            if (on0 && on1)
            {
                intersections.push_back(v0);
                intersections.push_back(v1);
                continue;
            }
            if (on0)
            {
                intersections.push_back(v0);
                continue;
            }
            if (on1)
            {
                intersections.push_back(v1);
                continue;
            }
            if ((d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0))
            {
                const double t = d0 / (d0 - d1);
                const glm::dvec3 p = v0 + t * (v1 - v0);
                intersections.push_back(p);
            }
        }

        std::vector<glm::dvec3> uniquePoints;
        uniquePoints.reserve(intersections.size());
        for (const glm::dvec3& p : intersections)
        {
            bool duplicate = false;
            for (const glm::dvec3& existing : uniquePoints)
            {
                if (lengthSquared(glm::dvec2(existing) - glm::dvec2(p)) <= m_tolerance * m_tolerance)
                {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate)
            {
                uniquePoints.push_back(p);
            }
        }

        if (uniquePoints.size() < 2)
        {
            continue;
        }
        if (uniquePoints.size() == 2)
        {
            addSegment(glm::dvec2(uniquePoints[0].x, uniquePoints[0].y),
                       glm::dvec2(uniquePoints[1].x, uniquePoints[1].y));
        }
        else
        {
            for (std::size_t i = 0; i + 1 < uniquePoints.size(); i += 2)
            {
                addSegment(glm::dvec2(uniquePoints[i].x, uniquePoints[i].y),
                           glm::dvec2(uniquePoints[i + 1].x, uniquePoints[i + 1].y));
            }
        }
    }

    std::unordered_map<GridKey, std::vector<std::pair<std::size_t, bool>>, GridHash> adjacency;
    adjacency.reserve(segments.size() * 2);

    for (std::size_t i = 0; i < segments.size(); ++i)
    {
        adjacency[quantize(segments[i].a, m_tolerance)].push_back({i, true});
        adjacency[quantize(segments[i].b, m_tolerance)].push_back({i, false});
    }

    std::vector<std::vector<glm::dvec2>> loops2d;
    loops2d.reserve(segments.size() / 3 + 1);

    for (std::size_t i = 0; i < segments.size(); ++i)
    {
        if (segments[i].used)
        {
            continue;
        }

        std::vector<glm::dvec2> loop;
        loop.reserve(32);

        Segment* startSegment = &segments[i];
        startSegment->used = true;
        loop.push_back(startSegment->a);
        loop.push_back(startSegment->b);

        glm::dvec2 currentPoint = startSegment->b;
        GridKey currentKey = quantize(currentPoint, m_tolerance);

        bool closed = false;
        while (true)
        {
            bool found = false;
            auto it = adjacency.find(currentKey);
            if (it != adjacency.end())
            {
                for (const auto& entry : it->second)
                {
                    Segment& seg = segments[entry.first];
                    if (seg.used)
                    {
                        continue;
                    }

                    glm::dvec2 nextPoint;
                    if (entry.second)
                    {
                        if (!nearlyEqual2D(seg.a, currentPoint, m_tolerance))
                        {
                            continue;
                        }
                        nextPoint = seg.b;
                    }
                    else
                    {
                        if (!nearlyEqual2D(seg.b, currentPoint, m_tolerance))
                        {
                            continue;
                        }
                        nextPoint = seg.a;
                    }

                    seg.used = true;
                    loop.push_back(nextPoint);
                    currentPoint = nextPoint;
                    currentKey = quantize(currentPoint, m_tolerance);
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                break;
            }

            if (nearlyEqual2D(currentPoint, loop.front(), m_tolerance))
            {
                closed = true;
                break;
            }
        }

        if (!closed)
        {
            continue;
        }

        if (loop.size() > 2)
        {
            loop.pop_back();
            loops2d.push_back(std::move(loop));
        }
    }

    std::vector<std::vector<glm::dvec3>> loops3d;
    if (loops2d.empty())
    {
        return loops3d;
    }

    struct ScoredLoop
    {
        double area;
        std::vector<glm::dvec2> points;
    };

    std::vector<ScoredLoop> scored;
    scored.reserve(loops2d.size());
    for (auto& loop : loops2d)
    {
        const double area = polygonArea(loop);
        if (std::abs(area) <= kEpsilon)
        {
            continue;
        }

        if (applyOffsetForFlat && toolRadius > kEpsilon)
        {
            loop = offsetLoop(loop, toolRadius, area);
        }
        scored.push_back({polygonArea(loop), std::move(loop)});
    }

    if (scored.empty())
    {
        return loops3d;
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredLoop& a, const ScoredLoop& b) {
        return std::abs(a.area) > std::abs(b.area);
    });

    loops3d.reserve(scored.size());
    for (const auto& entry : scored)
    {
        std::vector<glm::dvec3> loop3d;
        loop3d.reserve(entry.points.size() + 1);
        for (const glm::dvec2& p : entry.points)
        {
            loop3d.push_back({p.x, p.y, planeZ});
        }
        if (!loop3d.empty())
        {
            loop3d.push_back(loop3d.front());
        }
        loops3d.push_back(std::move(loop3d));
    }

    return loops3d;
}

} // namespace tp::waterline
