#include "tp/Stock.h"

#include <algorithm>

namespace tp
{

namespace
{
constexpr double kMinDimension = 0.0;
}

void Stock::ensureValid()
{
    sizeXYZ_mm.x = std::max(sizeXYZ_mm.x, kMinDimension);
    sizeXYZ_mm.y = std::max(sizeXYZ_mm.y, kMinDimension);
    sizeXYZ_mm.z = std::max(sizeXYZ_mm.z, kMinDimension);
    margin_mm = std::max(margin_mm, 0.0);
}

Stock makeDefaultStock()
{
    Stock stock;
    stock.shape = Stock::Shape::Block;
    stock.sizeXYZ_mm = glm::dvec3{0.0};
    stock.originXYZ_mm = glm::dvec3{0.0};
    stock.topZ_mm = 0.0;
    stock.margin_mm = 0.0;
    return stock;
}

} // namespace tp

