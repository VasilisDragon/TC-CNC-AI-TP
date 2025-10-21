#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <utility>

#if __has_include(<QtCore/QLoggingCategory>)
#    include <QtCore/QLoggingCategory>
#    include <QtCore/QString>
#    define CNCTC_HAS_QT_LOGGING 1
#else
#    define CNCTC_HAS_QT_LOGGING 0
#endif

namespace common::log
{

enum class Level
{
    Info,
    Warning,
    Error
};

enum class Category
{
    Io,
    Tp,
    Ai,
    Render
};

constexpr const char* categoryName(Category category)
{
    switch (category)
    {
    case Category::Io: return "io";
    case Category::Tp: return "tp";
    case Category::Ai: return "ai";
    case Category::Render: return "render";
    }
    return "unknown";
}

namespace detail
{

constexpr const char* levelLabel(Level level)
{
    switch (level)
    {
    case Level::Info: return "info";
    case Level::Warning: return "warn";
    case Level::Error: return "error";
    }
    return "unknown";
}

#if CNCTC_HAS_QT_LOGGING
using Message = QString;

inline QLoggingCategory& categoryHandle(Category category)
{
    switch (category)
    {
    case Category::Io: {
        static QLoggingCategory instance("io");
        return instance;
    }
    case Category::Tp: {
        static QLoggingCategory instance("tp");
        return instance;
    }
    case Category::Ai: {
        static QLoggingCategory instance("ai");
        return instance;
    }
    case Category::Render: {
        static QLoggingCategory instance("render");
        return instance;
    }
    }
    static QLoggingCategory fallback("unknown");
    return fallback;
}

inline Message toMessage(const QString& message)
{
    return message;
}

inline Message toMessage(const char* message)
{
    return message ? QString::fromUtf8(message) : QString();
}

inline Message toMessage(const std::string& message)
{
    return QString::fromStdString(message);
}

template <typename T>
Message toMessage(const T& value)
{
    std::ostringstream stream;
    stream << value;
    return QString::fromStdString(stream.str());
}

#else

using Message = std::string;

inline Message toMessage(const char* message)
{
    return message ? std::string(message) : std::string();
}

inline Message toMessage(const std::string& message)
{
    return message;
}

template <typename T>
Message toMessage(const T& value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

#endif

} // namespace detail

#if CNCTC_HAS_QT_LOGGING
inline void write(Level level, Category category, const detail::Message& message)
{
    QLoggingCategory& qtCategory = detail::categoryHandle(category);
    switch (level)
    {
    case Level::Info:
        qCInfo(qtCategory).noquote() << message;
        break;
    case Level::Warning:
        qCWarning(qtCategory).noquote() << message;
        break;
    case Level::Error:
        qCCritical(qtCategory).noquote() << message;
        break;
    }
}
#else
inline void write(Level level, Category category, const detail::Message& message)
{
    std::FILE* stream = (level == Level::Info) ? stdout : stderr;
    std::fprintf(stream, "[%s][%s] %s\n",
                 detail::levelLabel(level),
                 categoryName(category),
                 message.c_str());
}
#endif

template <typename Message>
void log(Level level, Category category, Message&& message)
{
    write(level, category, detail::toMessage(std::forward<Message>(message)));
}

} // namespace common::log

#define LOG_INFO(category, message)                                                           \
    do                                                                                        \
    {                                                                                         \
        ::common::log::log(::common::log::Level::Info, ::common::log::Category::category,     \
                           (message));                                                        \
    } while (false)

#define LOG_WARN(category, message)                                                           \
    do                                                                                        \
    {                                                                                         \
        ::common::log::log(::common::log::Level::Warning, ::common::log::Category::category,  \
                           (message));                                                        \
    } while (false)

#define LOG_ERR(category, message)                                                            \
    do                                                                                        \
    {                                                                                         \
        ::common::log::log(::common::log::Level::Error, ::common::log::Category::category,    \
                           (message));                                                        \
    } while (false)

