#pragma once

#include "render/Model.h"

#include <QtGui/QVector3D>

#include <array>
#include <vector>

namespace ai
{

class FeatureExtractor
{
public:
    // Bins align with common machining breakpoints: 0-15° for flats, 15-30° for shallow walls, etc., so the
    // planner can map geometry statistics directly to strategy templates.
    static constexpr std::array<float, 6> kSlopeBinBoundariesDeg = {0.0f, 15.0f, 30.0f, 45.0f, 60.0f, 90.1f};
    static constexpr std::size_t kSlopeBinCount = 5;

    struct GlobalFeatures
    {
        QVector3D bboxExtent{0.0f, 0.0f, 0.0f};
        float surfaceArea{0.0f};
        float volume{0.0f};
        std::array<float, kSlopeBinCount> slopeHistogram{};
        float meanCurvature{0.0f};
        float curvatureVariance{0.0f};
        float flatAreaRatio{0.0f};
        float steepAreaRatio{0.0f};
        float pocketDepth{0.0f};
        bool valid{false};
    };

    [[nodiscard]] static GlobalFeatures computeGlobalFeatures(const render::Model& model);
    [[nodiscard]] static std::vector<float> toVector(const GlobalFeatures& features);
    [[nodiscard]] static constexpr std::size_t featureCount()
    {
        return 3 /*bbox*/ + 1 /*area*/ + 1 /*volume*/ + kSlopeBinCount + 1 /*mean curv*/
               + 1 /*var curv*/ + 1 /*flat*/ + 1 /*steep*/ + 1 /*pocket*/;
    }
};

} // namespace ai

