#pragma once

#include "render/CameraController.h"
#include "render/Model.h"
#include "render/SimulationController.h"
#include "render/ToolpathOverlay.h"

#include <QtCore/QElapsedTimer>
#include <QtCore/QString>
#include <QtGui/QMatrix4x4>
#include <QtOpenGL/QOpenGLFunctions_3_3_Core>
#include <QtOpenGLWidgets/QOpenGLWidget>

#include <memory>
#include <vector>

namespace tp
{
struct Toolpath;
}

namespace render
{

class ModelViewerWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public:
    enum class ViewPreset
    {
        Top,
        Front,
        Right,
        Iso
    };

    explicit ModelViewerWidget(QWidget* parent = nullptr);
    ~ModelViewerWidget() override;

    void setModel(std::shared_ptr<Model> model);
    void clearModel();

    void setToolpath(std::shared_ptr<tp::Toolpath> toolpath);
    void setSimulationController(SimulationController* controller);

    void resetCamera();
    void setViewPreset(ViewPreset preset);

    Q_SIGNALS:
    void rendererInfoChanged(const QString& vendor, const QString& renderer, const QString& version);
    void frameStatsUpdated(float fps);
    void cameraChanged();

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void updateMeshBuffers();
    void updateToolpathOverlay();
    void rebuildGridGeometry();
    void rebuildAxesGeometry();
    void rebuildSimulationGlyph();

    std::shared_ptr<Model> m_model;
    std::shared_ptr<tp::Toolpath> m_toolpath;

    std::unique_ptr<QOpenGLShaderProgram> m_meshProgram;
    std::unique_ptr<QOpenGLShaderProgram> m_polylineProgram;

    std::unique_ptr<QOpenGLBuffer> m_vertexBuffer;
    std::unique_ptr<QOpenGLBuffer> m_indexBuffer;
    std::unique_ptr<QOpenGLVertexArrayObject> m_meshVao;

    std::unique_ptr<QOpenGLBuffer> m_gridBuffer;
    std::unique_ptr<QOpenGLVertexArrayObject> m_gridVao;
    std::unique_ptr<QOpenGLBuffer> m_axesBuffer;
    std::unique_ptr<QOpenGLVertexArrayObject> m_axesVao;
    std::unique_ptr<QOpenGLBuffer> m_simVertexBuffer;
    std::unique_ptr<QOpenGLBuffer> m_simIndexBuffer;
    std::unique_ptr<QOpenGLVertexArrayObject> m_simVao;

    int m_vertexCount{0};
    int m_indexCount{0};
    int m_gridVertexCount{0};
    int m_axesVertexCount{0};
    int m_simIndexCount{0};

    ToolpathOverlay m_toolpathOverlay;
    CameraController m_camera;
    SimulationController* m_simulation{nullptr};
    QVector3D m_simPosition;
    bool m_simVisible{false};
    bool m_simRapid{false};
    float m_simRadius{1.0f};
    bool m_rendererReported{false};

    QElapsedTimer m_fpsTimer;
    int m_frameCounter{0};

    bool m_meshBuffersDirty{true};
    bool m_toolpathDirty{false};
};

} // namespace render
