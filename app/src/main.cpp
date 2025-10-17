#include "app/MainWindow.h"
#include "common/logging.h"

#include <QtCore/QCoreApplication>
#include <QtGui/QPalette>
#include <QtWidgets/QApplication>

namespace
{

QPalette buildDarkPalette()
{
    QPalette palette;
    palette.setColor(QPalette::Window, QColor(18, 18, 20));
    palette.setColor(QPalette::WindowText, Qt::white);
    palette.setColor(QPalette::Base, QColor(30, 30, 34));
    palette.setColor(QPalette::AlternateBase, QColor(45, 45, 50));
    palette.setColor(QPalette::ToolTipBase, Qt::white);
    palette.setColor(QPalette::ToolTipText, Qt::white);
    palette.setColor(QPalette::Text, Qt::white);
    palette.setColor(QPalette::Button, QColor(45, 45, 50));
    palette.setColor(QPalette::ButtonText, Qt::white);
    palette.setColor(QPalette::BrightText, Qt::red);
    palette.setColor(QPalette::Highlight, QColor(64, 128, 255));
    palette.setColor(QPalette::HighlightedText, Qt::black);
    palette.setColor(QPalette::PlaceholderText, QColor(180, 180, 180));

    palette.setColor(QPalette::Disabled, QPalette::Text, QColor(110, 110, 110));
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(110, 110, 110));
    return palette;
}

} // namespace

int main(int argc, char* argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("AIToolpathGenerator"));
    app.setOrganizationName(QStringLiteral("AIToolpathGenerator"));
    app.setOrganizationDomain(QStringLiteral("example.ai"));

    QApplication::setStyle(QStringLiteral("Fusion"));
    app.setPalette(buildDarkPalette());

    common::initLogging();
    common::logInfo(QStringLiteral("Application started."));

    app::MainWindow window;
    window.show();

    return QApplication::exec();
}
