#include "tp/GenerateWorker.h"

#include "ai/IPathAI.h"
#include "render/Model.h"

#include <QtCore/QElapsedTimer>
#include <QtCore/QMetaType>
#include <QtCore/QString>

#include <string>

#include <utility>

namespace tp
{

GenerateWorker::GenerateWorker(std::shared_ptr<render::Model> model,
                               UserParams params,
                               std::unique_ptr<ai::IPathAI> ai,
                               QObject* parent)
    : QThread(parent)
    , m_model(std::move(model))
    , m_params(std::move(params))
    , m_ai(std::move(ai))
{
    qRegisterMetaType<std::shared_ptr<tp::Toolpath>>("std::shared_ptr<tp::Toolpath>");
}

void GenerateWorker::requestCancel()
{
    m_cancelled.store(true, std::memory_order_relaxed);
}

void GenerateWorker::run()
{
    if (!m_model || !m_ai)
    {
        emit error(tr("Generation aborted: no model or AI."));
        return;
    }

    emit progress(0);

    ai::StrategyDecision decision;

    auto progressCallback = [this](int value) {
        emit progress(value);
    };

    std::string bannerMessage;
    Toolpath result = m_generator.generate(*m_model,
                                           m_params,
                                           *m_ai,
                                           m_cancelled,
                                           progressCallback,
                                           &decision,
                                           &bannerMessage);

    if (m_cancelled.load(std::memory_order_relaxed))
    {
        emit error(tr("Toolpath generation cancelled."));
        return;
    }

    emit progress(100);
    if (!bannerMessage.empty())
    {
        emit banner(QString::fromStdString(bannerMessage));
    }
    emit finished(std::make_shared<Toolpath>(std::move(result)), decision);
}

} // namespace tp

