#include "render/HeatmapOverlay.h"

#include <QtOpenGL/QOpenGLShaderProgram>

namespace render
{

HeatmapOverlay::~HeatmapOverlay() = default;

void HeatmapOverlay::initialize(QOpenGLFunctions_3_3_Core* functions)
{
    m_functions = functions;
    if (!m_buffer)
    {
        m_buffer = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
        m_buffer->create();
    }
    if (!m_vao)
    {
        m_vao = std::make_unique<QOpenGLVertexArrayObject>();
        m_vao->create();
    }
}

void HeatmapOverlay::updateGeometry(std::vector<HeatmapPoint> points)
{
    m_points = std::move(points);
    m_vertexCount = static_cast<int>(m_points.size());
    m_dirty = true;
}

void HeatmapOverlay::clear()
{
    m_points.clear();
    m_vertexCount = 0;
    m_dirty = true;
}

bool HeatmapOverlay::isEmpty() const noexcept
{
    return m_vertexCount == 0;
}

void HeatmapOverlay::uploadIfNeeded()
{
    if (!m_dirty || !m_buffer || !m_vao || !m_functions)
    {
        return;
    }

    std::vector<float> data;
    data.reserve(static_cast<std::size_t>(m_vertexCount) * 6);

    for (const HeatmapPoint& point : m_points)
    {
        data.push_back(point.position.x());
        data.push_back(point.position.y());
        data.push_back(point.position.z());
        data.push_back(point.color.x());
        data.push_back(point.color.y());
        data.push_back(point.color.z());
    }

    m_buffer->bind();
    m_buffer->setUsagePattern(QOpenGLBuffer::DynamicDraw);
    m_buffer->allocate(data.data(), static_cast<int>(data.size() * sizeof(float)));

    m_vao->bind();
    m_functions->glEnableVertexAttribArray(0);
    m_functions->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, nullptr);
    m_functions->glEnableVertexAttribArray(1);
    m_functions->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, reinterpret_cast<const void*>(sizeof(float) * 3));
    m_buffer->release();
    m_vao->release();

    m_dirty = false;
}

void HeatmapOverlay::render(QOpenGLShaderProgram& program,
                            const QMatrix4x4& mvp,
                            float pointSize,
                            float alpha)
{
    if (isEmpty() || !m_functions)
    {
        return;
    }

    uploadIfNeeded();
    if (!m_vao)
    {
        return;
    }

    program.bind();
    program.setUniformValue("u_mvp", mvp);
    program.setUniformValue("u_pointSize", pointSize);
    program.setUniformValue("u_alpha", alpha);

    m_vao->bind();
    m_functions->glDrawArrays(GL_POINTS, 0, m_vertexCount);
    m_vao->release();
    program.release();
}

} // namespace render
