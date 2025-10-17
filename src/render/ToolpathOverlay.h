#pragma once

#include <QtGui/QMatrix4x4>
#include <QtGui/QVector3D>
#include <QtGui/QOpenGLFunctions_3_3_Core>

#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtGui/QOpenGLShaderProgram>

#include <memory>
#include <vector>

namespace render
{

class ToolpathOverlay
{
public:
    ToolpathOverlay() = default;
    ~ToolpathOverlay();

    void initialize(QOpenGLFunctions_3_3_Core* functions);

    void updateGeometry(std::vector<QVector3D> cutVertices,
                        std::vector<QVector3D> rapidVertices,
                        std::vector<QVector3D> linkVertices = {},
                        std::vector<QVector3D> unsafeVertices = {});
    void clear();

    void render(QOpenGLShaderProgram& program,
                const QMatrix4x4& mvp,
                const QVector3D& cutColor,
                const QVector3D& rapidColor,
                const QVector3D& linkColor,
                const QVector3D& unsafeColor,
                float alpha);

    [[nodiscard]] bool isEmpty() const noexcept;

private:
    void uploadIfNeeded();

    QOpenGLFunctions_3_3_Core* m_functions{nullptr};
    std::unique_ptr<QOpenGLBuffer> m_buffer;
    std::unique_ptr<QOpenGLVertexArrayObject> m_vao;

    std::vector<QVector3D> m_cpuVertices;
    int m_cutVertexCount{0};
    int m_rapidVertexCount{0};
    int m_linkVertexCount{0};
    int m_unsafeVertexCount{0};
    bool m_dirty{false};
};

} // namespace render
