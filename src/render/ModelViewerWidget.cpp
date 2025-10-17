#include "render/ModelViewerWidget.h"

#include "tp/Toolpath.h"

#include <QtCore/QElapsedTimer>
#include <QtCore/QFile>
#include <QtGui/QMouseEvent>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLShader>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtGui/QWheelEvent>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <numbers>
#include <utility>

namespace render
{

namespace
{
constexpr QVector3D kCutColor{0.1f, 0.85f, 0.3f};
constexpr QVector3D kRapidColor{0.9f, 0.8f, 0.1f};
constexpr QVector3D kLinkColor{0.4f, 0.75f, 1.0f};
constexpr QVector3D kUnsafeColor{0.9f, 0.2f, 0.2f};
constexpr float kToolpathAlpha = 0.95f;
}

ModelViewerWidget::ModelViewerWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setMinimumSize(640, 480);
}

ModelViewerWidget::~ModelViewerWidget() = default;

void ModelViewerWidget::setModel(std::shared_ptr<Model> model)
{
    m_model = std::move(model);
    if (m_model && m_model->isValid())
    {
        m_camera.setBounds(m_model->bounds());
    }
    else
    {
        m_camera.setBounds({});
    }

    m_toolpath.reset();
    rebuildGridGeometry();
    m_toolpathOverlay.clear();
    m_toolpathDirty = false;
    m_camera.reset();
    m_meshBuffersDirty = true;
    m_simVisible = false;
    update();
}

void ModelViewerWidget::clearModel()
{
    m_model.reset();
    m_vertexCount = 0;
    m_indexCount = 0;
    m_toolpath.reset();
    m_meshBuffersDirty = true;
    rebuildGridGeometry();
    m_toolpathOverlay.clear();
    m_toolpathDirty = false;
    m_simVisible = false;
    update();
}

void ModelViewerWidget::setToolpath(std::shared_ptr<tp::Toolpath> toolpath)
{
    m_toolpath = std::move(toolpath);
    m_toolpathDirty = true;
    m_simVisible = false;
    update();
}

void ModelViewerWidget::setSimulationController(SimulationController* controller)
{
    if (m_simulation == controller)
    {
        return;
    }

    if (m_simulation)
    {
        disconnect(m_simulation, nullptr, this, nullptr);
    }

    m_simulation = controller;
    m_simVisible = false;

    if (m_simulation)
    {
        connect(m_simulation,
                &SimulationController::positionChanged,
                this,
                [this](const QVector3D& position, bool rapid, bool visible, float radius) {
                    m_simPosition = position;
                    m_simRapid = rapid;
                    m_simVisible = visible;
                    m_simRadius = radius;
                    update();
                });
    }
}

void ModelViewerWidget::resetCamera()
{
    m_camera.reset();
    update();
}

void ModelViewerWidget::setViewPreset(ViewPreset preset)
{
    constexpr float degToRad = static_cast<float>(std::numbers::pi) / 180.0f;

    float yaw = 0.0f;
    float pitch = -0.5f;

    switch (preset)
    {
    case ViewPreset::Top:
        yaw = 45.0f * degToRad;
        pitch = 82.0f * degToRad;
        break;
    case ViewPreset::Front:
        yaw = 90.0f * degToRad;
        pitch = -15.0f * degToRad;
        break;
    case ViewPreset::Right:
        yaw = 0.0f;
        pitch = -15.0f * degToRad;
        break;
    case ViewPreset::Iso:
        yaw = 45.0f * degToRad;
        pitch = -35.0f * degToRad;
        break;
    }

    m_camera.setViewAngles(yaw, pitch);

    float targetDistance = std::max(m_camera.distance(), 5.0f);
    if (m_model && m_model->isValid())
    {
        const auto bounds = m_model->bounds();
        const QVector3D extents = bounds.size();
        const float radius = std::max({extents.x(), extents.y(), extents.z(), 1.0f});
        const float scale = (preset == ViewPreset::Iso) ? 2.4f : (preset == ViewPreset::Top ? 3.0f : 2.0f);
        targetDistance = std::max(radius * scale, 1.0f);
    }
    m_camera.setDistance(targetDistance);

    update();
    emit cameraChanged();
}

void ModelViewerWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);

    if (!m_rendererReported)
    {
        const auto* vendorPtr = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        const auto* rendererPtr = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        const auto* versionPtr = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        const QString vendor = vendorPtr ? QString::fromLatin1(vendorPtr).trimmed() : QStringLiteral("Unknown");
        const QString renderer = rendererPtr ? QString::fromLatin1(rendererPtr).trimmed() : QStringLiteral("Unknown");
        const QString version = versionPtr ? QString::fromLatin1(versionPtr).trimmed() : QStringLiteral("Unknown");
        emit rendererInfoChanged(vendor, renderer, version);
        m_rendererReported = true;
    }

    m_fpsTimer.invalidate();
    m_frameCounter = 0;

    m_meshProgram = std::make_unique<QOpenGLShaderProgram>();
    m_meshProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, QStringLiteral(":/render/shaders/flat.vert"));
    m_meshProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, QStringLiteral(":/render/shaders/flat.frag"));
    m_meshProgram->link();

    m_polylineProgram = std::make_unique<QOpenGLShaderProgram>();
    m_polylineProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, QStringLiteral(":/render/shaders/polyline.vert"));
    m_polylineProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, QStringLiteral(":/render/shaders/polyline.frag"));
    m_polylineProgram->link();

    m_vertexBuffer = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    m_vertexBuffer->create();
    m_indexBuffer = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::IndexBuffer);
    m_indexBuffer->create();
    m_meshVao = std::make_unique<QOpenGLVertexArrayObject>();
    m_meshVao->create();

    m_toolpathOverlay.initialize(this);

    m_gridBuffer = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    m_gridBuffer->create();
    m_gridVao = std::make_unique<QOpenGLVertexArrayObject>();
    m_gridVao->create();

    m_axesBuffer = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    m_axesBuffer->create();
    m_axesVao = std::make_unique<QOpenGLVertexArrayObject>();
    m_axesVao->create();

    m_simVertexBuffer = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    m_simVertexBuffer->create();
    m_simIndexBuffer = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::IndexBuffer);
    m_simIndexBuffer->create();
    m_simVao = std::make_unique<QOpenGLVertexArrayObject>();
    m_simVao->create();

    rebuildGridGeometry();
    rebuildAxesGeometry();
    rebuildSimulationGlyph();
}

void ModelViewerWidget::resizeGL(int width, int height)
{
    m_camera.setViewportSize({width, height});
}

void ModelViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_camera.setViewportSize(size());

    if (m_meshBuffersDirty)
    {
        updateMeshBuffers();
    }

    if (m_toolpathDirty)
    {
        updateToolpathOverlay();
    }

    const QMatrix4x4 modelMatrix;
    const QMatrix4x4& viewMatrix = m_camera.viewMatrix();
    const QMatrix4x4& projMatrix = m_camera.projectionMatrix();
    const QMatrix4x4 mvp = projMatrix * viewMatrix * modelMatrix;

    // Draw grid
    if (m_gridVao && m_gridBuffer && m_gridVertexCount > 0)
    {
        m_polylineProgram->bind();
        m_polylineProgram->setUniformValue("u_mvp", mvp);
        m_polylineProgram->setUniformValue("u_color", QVector3D{0.25f, 0.25f, 0.25f});
        m_polylineProgram->setUniformValue("u_isDashed", 0);
        m_polylineProgram->setUniformValue("u_alpha", 0.6f);
        m_polylineProgram->setUniformValue("u_dashSize", 12.0f);
        m_gridVao->bind();
        glDrawArrays(GL_LINES, 0, m_gridVertexCount);
        m_gridVao->release();
        m_polylineProgram->release();
    }

    // Draw mesh
    if (m_model && m_model->isValid() && m_meshVao && m_indexCount > 0)
    {
        m_meshProgram->bind();
        m_meshProgram->setUniformValue("u_model", modelMatrix);
        m_meshProgram->setUniformValue("u_view", viewMatrix);
        m_meshProgram->setUniformValue("u_projection", projMatrix);
        m_meshProgram->setUniformValue("u_lightDir", QVector3D{0.3f, 0.4f, 0.9f}.normalized());
        m_meshProgram->setUniformValue("u_color", QVector3D{0.55f, 0.65f, 0.8f});

        m_meshVao->bind();
        glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
        m_meshVao->release();

        m_meshProgram->release();
    }

    // Draw toolpath overlay
    if (m_polylineProgram && !m_toolpathOverlay.isEmpty())
    {
        glDisable(GL_DEPTH_TEST);
        m_toolpathOverlay.render(*m_polylineProgram, mvp, kCutColor, kRapidColor, kLinkColor, kUnsafeColor, kToolpathAlpha);
        glEnable(GL_DEPTH_TEST);
    }

    if (m_simVisible && m_simVao && m_simIndexCount > 0)
    {
        QMatrix4x4 simModel;
        simModel.translate(m_simPosition);
        simModel.scale(std::max(0.1f, m_simRadius));

        m_meshProgram->bind();
        m_meshProgram->setUniformValue("u_model", simModel);
        m_meshProgram->setUniformValue("u_view", viewMatrix);
        m_meshProgram->setUniformValue("u_projection", projMatrix);
        m_meshProgram->setUniformValue("u_lightDir", QVector3D{0.3f, 0.4f, 0.9f}.normalized());
        const QVector3D toolColor = m_simRapid ? QVector3D{0.95f, 0.85f, 0.2f} : QVector3D{0.2f, 0.9f, 0.4f};
        m_meshProgram->setUniformValue("u_color", toolColor);

        m_simVao->bind();
        glDrawElements(GL_TRIANGLES, m_simIndexCount, GL_UNSIGNED_INT, nullptr);
        m_simVao->release();
        m_meshProgram->release();
    }

    // Draw axes
    if (m_axesVao && m_axesVertexCount >= 6)
    {
        m_polylineProgram->bind();
        m_polylineProgram->setUniformValue("u_mvp", mvp);
        m_polylineProgram->setUniformValue("u_dashSize", 12.0f);
        m_polylineProgram->setUniformValue("u_alpha", 1.0f);
        m_polylineProgram->setUniformValue("u_isDashed", 0);

        m_axesVao->bind();
        m_polylineProgram->setUniformValue("u_color", QVector3D{1.0f, 0.1f, 0.1f});
        glDrawArrays(GL_LINES, 0, 2);
        m_polylineProgram->setUniformValue("u_color", QVector3D{0.1f, 1.0f, 0.1f});
        glDrawArrays(GL_LINES, 2, 2);
        m_polylineProgram->setUniformValue("u_color", QVector3D{0.1f, 0.4f, 1.0f});
        glDrawArrays(GL_LINES, 4, 2);
        m_axesVao->release();
        m_polylineProgram->release();
    }

    if (!m_fpsTimer.isValid())
    {
        m_fpsTimer.start();
        m_frameCounter = 0;
    }
    ++m_frameCounter;
    const qint64 elapsedMs = m_fpsTimer.elapsed();
    if (elapsedMs >= 1'000)
    {
        const float fps = static_cast<float>(m_frameCounter) * 1'000.0f / std::max<qint64>(elapsedMs, 1);
        emit frameStatsUpdated(fps);
        m_frameCounter = 0;
        m_fpsTimer.restart();
    }
}

void ModelViewerWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        m_camera.beginOrbit(event->pos());
    }
    else if (event->buttons() & Qt::MiddleButton || (event->buttons() & Qt::RightButton))
    {
        m_camera.beginPan(event->pos());
    }
    event->accept();
}

void ModelViewerWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        m_camera.updateOrbit(event->pos());
        update();
    }
    else if (event->buttons() & Qt::MiddleButton || (event->buttons() & Qt::RightButton))
    {
        m_camera.updatePan(event->pos());
        update();
    }
    event->accept();
}

void ModelViewerWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
    {
        m_camera.endOrbit();
    }
    else if (event->button() == Qt::MiddleButton || event->button() == Qt::RightButton)
    {
        m_camera.endPan();
    }
    update();
}

void ModelViewerWidget::wheelEvent(QWheelEvent* event)
{
    constexpr float stepsPerDegree = 1.0f / 120.0f;
    const float numSteps = event->angleDelta().y() * stepsPerDegree;
    m_camera.applyZoom(-numSteps);
    update();
}

void ModelViewerWidget::updateMeshBuffers()
{
    if (!context())
    {
        return;
    }

    m_meshBuffersDirty = false;

    if (!m_model || !m_model->isValid())
    {
        m_vertexCount = 0;
        m_indexCount = 0;

        if (m_meshVao)
        {
            m_meshVao->bind();
            if (m_vertexBuffer)
            {
                m_vertexBuffer->bind();
                m_vertexBuffer->allocate(nullptr, 0);
                m_vertexBuffer->release();
            }
            if (m_indexBuffer)
            {
                m_indexBuffer->bind();
                m_indexBuffer->allocate(nullptr, 0);
                m_indexBuffer->release();
            }
            m_meshVao->release();
        }
        return;
    }

    m_meshVao->bind();

    m_vertexBuffer->bind();
    const auto& vertices = m_model->vertices();
    m_vertexBuffer->allocate(vertices.data(), static_cast<int>(vertices.size() * sizeof(Vertex)));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));

    m_indexBuffer->bind();
    const auto& indices = m_model->indices();
    m_indexBuffer->allocate(indices.data(), static_cast<int>(indices.size() * sizeof(Model::Index)));

    m_vertexCount = static_cast<int>(vertices.size());
    m_indexCount = static_cast<int>(indices.size());

    m_indexBuffer->release();
    m_vertexBuffer->release();
    m_meshVao->release();
}

void ModelViewerWidget::updateToolpathOverlay()
{
    if (!context())
    {
        return;
    }

    m_toolpathDirty = false;

    if (!m_toolpath || m_toolpath->empty())
    {
        m_toolpathOverlay.clear();
        return;
    }

    std::vector<QVector3D> cutSegments;
    std::vector<QVector3D> rapidSegments;
    std::vector<QVector3D> linkSegments;
    std::vector<QVector3D> unsafeSegments;

    const bool hasModel = m_model && m_model->isValid();
    const float minZ = hasModel ? m_model->bounds().min.z() : 0.0f;

    for (const tp::Polyline& poly : m_toolpath->passes)
    {
        if (poly.pts.size() < 2)
        {
            continue;
        }

        for (std::size_t i = 1; i < poly.pts.size(); ++i)
        {
            const tp::Vertex& prev = poly.pts[i - 1];
            const tp::Vertex& curr = poly.pts[i];

            const QVector3D p0(prev.p.x, prev.p.y, prev.p.z);
            const QVector3D p1(curr.p.x, curr.p.y, curr.p.z);

            const bool unsafe = hasModel && (prev.p.z < minZ || curr.p.z < minZ);

            if (unsafe)
            {
                unsafeSegments.push_back(p0);
                unsafeSegments.push_back(p1);
            }
            else
            {
                switch (poly.motion)
                {
                case tp::MotionType::Cut:
                    cutSegments.push_back(p0);
                    cutSegments.push_back(p1);
                    break;
                case tp::MotionType::Rapid:
                    rapidSegments.push_back(p0);
                    rapidSegments.push_back(p1);
                    break;
                case tp::MotionType::Link:
                    linkSegments.push_back(p0);
                    linkSegments.push_back(p1);
                    break;
                }
            }
        }
    }

    m_toolpathOverlay.updateGeometry(std::move(cutSegments),
                                     std::move(rapidSegments),
                                     std::move(linkSegments),
                                     std::move(unsafeSegments));
}

void ModelViewerWidget::rebuildGridGeometry()
{
    if (!context())
    {
        return;
    }

    const common::Bounds bounds = m_model && m_model->isValid() ? m_model->bounds() : common::Bounds{};
    const QVector3D size = bounds.size();
    float extent = std::max({std::abs(size.x()), std::abs(size.y())});
    if (extent < 20.0f)
    {
        extent = 200.0f;
    }
    extent *= 0.6f;

    const int linesPerSide = 10;
    const float spacing = extent / static_cast<float>(linesPerSide);

    std::vector<QVector3D> gridLines;
    gridLines.reserve((linesPerSide * 2 + 1) * 4);
    for (int i = -linesPerSide; i <= linesPerSide; ++i)
    {
        const float offset = static_cast<float>(i) * spacing;
        gridLines.emplace_back(-extent, offset, 0.0f);
        gridLines.emplace_back(extent, offset, 0.0f);
        gridLines.emplace_back(offset, -extent, 0.0f);
        gridLines.emplace_back(offset, extent, 0.0f);
    }

    m_gridVertexCount = static_cast<int>(gridLines.size());

    m_gridVao->bind();
    m_gridBuffer->bind();
    m_gridBuffer->allocate(gridLines.data(), static_cast<int>(gridLines.size() * sizeof(QVector3D)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D), nullptr);
    m_gridBuffer->release();
    m_gridVao->release();
}

void ModelViewerWidget::rebuildAxesGeometry()
{
    if (!context())
    {
        return;
    }

    const float axisLength = 50.0f;
    const std::array<QVector3D, 6> axes = {
        QVector3D{0.0f, 0.0f, 0.0f}, QVector3D{axisLength, 0.0f, 0.0f},
        QVector3D{0.0f, 0.0f, 0.0f}, QVector3D{0.0f, axisLength, 0.0f},
        QVector3D{0.0f, 0.0f, 0.0f}, QVector3D{0.0f, 0.0f, axisLength},
    };

    m_axesVertexCount = static_cast<int>(axes.size());

    m_axesVao->bind();
    m_axesBuffer->bind();
    m_axesBuffer->allocate(axes.data(), static_cast<int>(axes.size() * sizeof(QVector3D)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D), nullptr);
    m_axesBuffer->release();
    m_axesVao->release();
}

void ModelViewerWidget::rebuildSimulationGlyph()
{
    if (!context() || !m_simVertexBuffer || !m_simIndexBuffer || !m_simVao)
    {
        return;
    }

    struct Vertex
    {
        QVector3D position;
        QVector3D normal;
    };

    constexpr int stacks = 12;
    constexpr int slices = 24;

    std::vector<Vertex> vertices;
    vertices.reserve((stacks + 1) * (slices + 1));

    for (int i = 0; i <= stacks; ++i)
    {
        const double v = static_cast<double>(i) / stacks;
        const double phi = v * std::numbers::pi_v<double>;
        const double sinPhi = std::sin(phi);
        const double cosPhi = std::cos(phi);

        for (int j = 0; j <= slices; ++j)
        {
            const double u = static_cast<double>(j) / slices;
            const double theta = u * 2.0 * std::numbers::pi_v<double>;
            const double sinTheta = std::sin(theta);
            const double cosTheta = std::cos(theta);

            QVector3D normal(static_cast<float>(sinPhi * cosTheta),
                             static_cast<float>(sinPhi * sinTheta),
                             static_cast<float>(cosPhi));
            QVector3D position = normal; // unit sphere
            vertices.push_back({position, normal.normalized()});
        }
    }

    std::vector<unsigned int> indices;
    indices.reserve(stacks * slices * 6);
    const int ringSize = slices + 1;
    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            const unsigned int first = static_cast<unsigned int>(i * ringSize + j);
            const unsigned int second = first + static_cast<unsigned int>(ringSize);

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    m_simIndexCount = static_cast<int>(indices.size());

    m_simVao->bind();

    m_simVertexBuffer->bind();
    m_simVertexBuffer->allocate(vertices.data(), static_cast<int>(vertices.size() * sizeof(Vertex)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(sizeof(QVector3D)));
    m_simVertexBuffer->release();

    m_simIndexBuffer->bind();
    m_simIndexBuffer->allocate(indices.data(), static_cast<int>(indices.size() * sizeof(unsigned int)));
    m_simIndexBuffer->release();

    m_simVao->release();
}

} // namespace render



