#include "render/ToolpathOverlay.h"

#include <utility>

namespace render
{

ToolpathOverlay::~ToolpathOverlay()
{
    if (m_buffer)
    {
        m_buffer->destroy();
    }
    if (m_vao)
    {
        m_vao->destroy();
    }
}

void ToolpathOverlay::initialize(QOpenGLFunctions_3_3_Core* functions)
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

    m_dirty = true;
}

void ToolpathOverlay::updateGeometry(std::vector<QVector3D> cutVertices,
                                     std::vector<QVector3D> rapidVertices,
                                     std::vector<QVector3D> linkVertices,
                                     std::vector<QVector3D> unsafeVertices)
{
    m_cutVertexCount = static_cast<int>(cutVertices.size());
    m_rapidVertexCount = static_cast<int>(rapidVertices.size());
    m_linkVertexCount = static_cast<int>(linkVertices.size());
    m_unsafeVertexCount = static_cast<int>(unsafeVertices.size());

    m_cpuVertices = std::move(cutVertices);
    m_cpuVertices.insert(m_cpuVertices.end(), rapidVertices.begin(), rapidVertices.end());
    m_cpuVertices.insert(m_cpuVertices.end(), linkVertices.begin(), linkVertices.end());
    m_cpuVertices.insert(m_cpuVertices.end(), unsafeVertices.begin(), unsafeVertices.end());

    m_dirty = true;
}

void ToolpathOverlay::clear()
{
    if (m_cpuVertices.empty() && m_cutVertexCount == 0 && m_rapidVertexCount == 0 && m_linkVertexCount == 0 && m_unsafeVertexCount == 0)
    {
        return;
    }

    m_cpuVertices.clear();
    m_cutVertexCount = 0;
    m_rapidVertexCount = 0;
    m_linkVertexCount = 0;
    m_unsafeVertexCount = 0;
    m_dirty = true;
}

void ToolpathOverlay::render(QOpenGLShaderProgram& program,
                             const QMatrix4x4& mvp,
                             const QVector3D& cutColor,
                             const QVector3D& rapidColor,
                             const QVector3D& linkColor,
                             const QVector3D& unsafeColor,
                             float alpha)
{
    if (!m_functions || !m_vao || !m_buffer)
    {
        return;
    }

    const int totalVertexCount = m_cutVertexCount + m_rapidVertexCount + m_linkVertexCount + m_unsafeVertexCount;
    if (totalVertexCount == 0)
    {
        return;
    }

    uploadIfNeeded();

    program.bind();
    program.setUniformValue("u_mvp", mvp);
    program.setUniformValue("u_alpha", alpha);
    m_vao->bind();

    int offset = 0;
    if (m_cutVertexCount > 0)
    {
        program.setUniformValue("u_color", cutColor);
        program.setUniformValue("u_dashSize", 12.0f);
        program.setUniformValue("u_isDashed", 0);
        m_functions->glDrawArrays(GL_LINES, offset, m_cutVertexCount);
        offset += m_cutVertexCount;
    }

    if (m_rapidVertexCount > 0)
    {
        program.setUniformValue("u_color", rapidColor);
        program.setUniformValue("u_dashSize", 12.0f);
        program.setUniformValue("u_isDashed", 1);
        m_functions->glDrawArrays(GL_LINES, offset, m_rapidVertexCount);
        offset += m_rapidVertexCount;
    }

    if (m_linkVertexCount > 0)
    {
        program.setUniformValue("u_color", linkColor);
        program.setUniformValue("u_dashSize", 6.0f);
        program.setUniformValue("u_isDashed", 1);
        m_functions->glDrawArrays(GL_LINES, offset, m_linkVertexCount);
        offset += m_linkVertexCount;
    }

    if (m_unsafeVertexCount > 0)
    {
        program.setUniformValue("u_color", unsafeColor);
        program.setUniformValue("u_dashSize", 12.0f);
        program.setUniformValue("u_isDashed", 0);
        m_functions->glDrawArrays(GL_LINES, offset, m_unsafeVertexCount);
    }

    m_vao->release();
    program.release();
}

bool ToolpathOverlay::isEmpty() const noexcept
{
    return (m_cutVertexCount + m_rapidVertexCount + m_linkVertexCount + m_unsafeVertexCount) == 0;
}

void ToolpathOverlay::uploadIfNeeded()
{
    if (!m_dirty || !m_buffer || !m_vao)
    {
        return;
    }

    m_vao->bind();
    m_buffer->bind();

    if (!m_cpuVertices.empty())
    {
        m_buffer->allocate(m_cpuVertices.data(), static_cast<int>(m_cpuVertices.size() * sizeof(QVector3D)));
    }
    else
    {
        m_buffer->allocate(nullptr, 0);
    }

    m_functions->glEnableVertexAttribArray(0);
    m_functions->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D), nullptr);

    m_buffer->release();
    m_vao->release();

    m_dirty = false;
}

} // namespace render
