#pragma once

#include <QtGui/QMatrix4x4>
#include <QtGui/QVector3D>
#include <QtOpenGL/QOpenGLBuffer>
#include <QtOpenGL/QOpenGLVertexArrayObject>
#include <QtOpenGL/QOpenGLShaderProgram>
#include <QtOpenGL/QOpenGLFunctions_3_3_Core>

#include <memory>
#include <vector>

namespace render
{

struct HeatmapPoint
{
    QVector3D position{0.0f, 0.0f, 0.0f};
    QVector3D color{1.0f, 0.0f, 0.0f};
};

class HeatmapOverlay
{
public:
    HeatmapOverlay() = default;
    ~HeatmapOverlay();

    void initialize(QOpenGLFunctions_3_3_Core* functions);

    void updateGeometry(std::vector<HeatmapPoint> points);
    void clear();

    void render(QOpenGLShaderProgram& program,
                const QMatrix4x4& mvp,
                float pointSize,
                float alpha);

    [[nodiscard]] bool isEmpty() const noexcept;

private:
    void uploadIfNeeded();

    QOpenGLFunctions_3_3_Core* m_functions{nullptr};
    std::unique_ptr<QOpenGLBuffer> m_buffer;
    std::unique_ptr<QOpenGLVertexArrayObject> m_vao;

    std::vector<HeatmapPoint> m_points;
    bool m_dirty{false};
    int m_vertexCount{0};
};

} // namespace render
