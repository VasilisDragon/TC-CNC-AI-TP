#pragma once

#include <QtCore/QThread>
#include <QtCore/QMetaType>

#include "tp/ToolpathGenerator.h"

#include <atomic>
#include <memory>

namespace ai
{
class IPathAI;
struct StrategyDecision;
}

namespace render
{
class Model;
}

namespace tp
{

class GenerateWorker : public QThread
{
    Q_OBJECT

public:
    GenerateWorker(std::shared_ptr<render::Model> model,
                   UserParams params,
                   std::unique_ptr<ai::IPathAI> ai,
                   QObject* parent = nullptr);

    void requestCancel();

    Q_SIGNALS:
    void progress(int value);
    void finished(std::shared_ptr<tp::Toolpath> toolpath, ai::StrategyDecision decision);
    void error(const QString& message);
    void banner(const QString& message);

protected:
    void run() override;

private:
    std::shared_ptr<render::Model> m_model;
    UserParams m_params;
    std::unique_ptr<ai::IPathAI> m_ai;
    tp::ToolpathGenerator m_generator;
    std::atomic<bool> m_cancelled{false};
};

} // namespace tp

Q_DECLARE_METATYPE(std::shared_ptr<tp::Toolpath>)
