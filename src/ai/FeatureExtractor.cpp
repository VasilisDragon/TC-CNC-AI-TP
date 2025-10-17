#include "ai/FeatureExtractor.h"

#include <QtCore/QtMath>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ai
{

namespace
{
constexpr float kEpsilon = 1e-6f;

[[nodiscard]] QVector3D normalizeSafe(const QVector3D& v)
{
    const float length = v.length();
    if (length < kEpsilon)
    {
        return QVector3D(0.0f, 0.0f, 0.0f);
    }
    return v / length;
}

[[nodiscard]] float clamp01(float value)
{
    if (value < 0.0f)
    {
        return 0.0f;
    }
    if (value > 1.0f)
    {
        return 1.0f;
    }
    return value;
}

[[nodiscard]] std::size_t slopeBinIndex(float slopeDeg)
{
    for (std::size_t i = 0; i + 1 < FeatureExtractor::kSlopeBinBoundariesDeg.size(); ++i)
    {
        const float low = FeatureExtractor::kSlopeBinBoundariesDeg[i];
        const float high = FeatureExtractor::kSlopeBinBoundariesDeg[i + 1];
        if (slopeDeg >= low && slopeDeg < high)
        {
            return std::min<std::size_t>(i, FeatureExtractor::kSlopeBinCount - 1);
        }
    }
    return FeatureExtractor::kSlopeBinCount - 1;
}
} // namespace

FeatureExtractor::GlobalFeatures FeatureExtractor::computeGlobalFeatures(const render::Model& model)
{
    GlobalFeatures features;

    const auto& vertices = model.vertices();
    const auto& indices = model.indices();
    if (vertices.empty() || indices.size() < 3)
    {
        return features;
    }

    features.bboxExtent = model.bounds().size();

    double surfaceArea = 0.0;
    double enclosedVolume = 0.0;
    std::array<double, kSlopeBinCount> slopeArea{};
    double flatArea = 0.0;
    double steepArea = 0.0;
    std::vector<double> curvatureSamples;
    curvatureSamples.reserve(indices.size());

    float minZ = std::numeric_limits<float>::max();

    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        minZ = std::min(minZ, vertices[i].position.z());
    }

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const auto i0 = indices[i];
        const auto i1 = indices[i + 1];
        const auto i2 = indices[i + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
        {
            continue;
        }

        const QVector3D& p0 = vertices[i0].position;
        const QVector3D& p1 = vertices[i1].position;
        const QVector3D& p2 = vertices[i2].position;

        const QVector3D edge1 = p1 - p0;
        const QVector3D edge2 = p2 - p0;
        const QVector3D cross = QVector3D::crossProduct(edge1, edge2);
        const float triArea = 0.5f * cross.length();
        if (triArea < kEpsilon)
        {
            continue;
        }

        surfaceArea += triArea;

        const QVector3D faceNormal = normalizeSafe(cross);
        const float slopeDeg = qRadiansToDegrees(std::acos(std::clamp(std::abs(faceNormal.z()), 0.0f, 1.0f)));
        const std::size_t bin = slopeBinIndex(slopeDeg);
        slopeArea[bin] += triArea;

        if (slopeDeg < 15.0f)
        {
            flatArea += triArea;
        }
        if (slopeDeg >= 60.0f)
        {
            steepArea += triArea;
        }

        enclosedVolume += static_cast<double>(QVector3D::dotProduct(p0, QVector3D::crossProduct(p1, p2))) / 6.0;

        const QVector3D faceNormalNormalised = faceNormal;
        const auto accumulateCurvature = [&](const render::Vertex& vertex) {
            const QVector3D normal = normalizeSafe(vertex.normal);
            if (normal.length() < kEpsilon)
            {
                return;
            }
            const float cosAngle = std::clamp(QVector3D::dotProduct(normal, faceNormalNormalised), -1.0f, 1.0f);
            const double angle = std::acos(cosAngle);
            curvatureSamples.push_back(angle);
        };

        accumulateCurvature(vertices[i0]);
        accumulateCurvature(vertices[i1]);
        accumulateCurvature(vertices[i2]);
    }

    if (surfaceArea <= std::numeric_limits<double>::epsilon())
    {
        return features;
    }

    for (std::size_t i = 0; i < kSlopeBinCount; ++i)
    {
        features.slopeHistogram[i] = static_cast<float>(slopeArea[i] / surfaceArea);
    }

    features.surfaceArea = static_cast<float>(surfaceArea);
    features.volume = static_cast<float>(std::abs(enclosedVolume));
    features.flatAreaRatio = clamp01(static_cast<float>(flatArea / surfaceArea));
    features.steepAreaRatio = clamp01(static_cast<float>(steepArea / surfaceArea));

    if (!curvatureSamples.empty())
    {
        double sum = 0.0;
        for (double value : curvatureSamples)
        {
            sum += value;
        }
        const double mean = sum / static_cast<double>(curvatureSamples.size());
        double varAccum = 0.0;
        for (double value : curvatureSamples)
        {
            const double diff = value - mean;
            varAccum += diff * diff;
        }
        const double variance = varAccum / static_cast<double>(curvatureSamples.size());
        features.meanCurvature = static_cast<float>(mean);
        features.curvatureVariance = static_cast<float>(variance);
    }
    else
    {
        features.meanCurvature = 0.0f;
        features.curvatureVariance = 0.0f;
    }

    const auto bounds = model.bounds();
    features.pocketDepth = bounds.max.z() - minZ;
    features.pocketDepth = std::max(features.pocketDepth, 0.0f);
    features.valid = true;
    return features;
}

std::vector<float> FeatureExtractor::toVector(const GlobalFeatures& features)
{
    std::vector<float> result;
    result.reserve(featureCount());

    result.push_back(features.bboxExtent.x());
    result.push_back(features.bboxExtent.y());
    result.push_back(features.bboxExtent.z());
    result.push_back(features.surfaceArea);
    result.push_back(features.volume);

    for (float value : features.slopeHistogram)
    {
        result.push_back(value);
    }

    result.push_back(features.meanCurvature);
    result.push_back(features.curvatureVariance);
    result.push_back(features.flatAreaRatio);
    result.push_back(features.steepAreaRatio);
    result.push_back(features.pocketDepth);

    return result;
}

} // namespace ai

