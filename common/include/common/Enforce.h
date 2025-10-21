#pragma once

#include <stdexcept>
#include <string>
#if !defined(NDEBUG)
#    include <sstream>
#endif

namespace common::detail
{

#if !defined(NDEBUG)
[[noreturn]] inline void enforceFail(const char* expr,
                                     const char* file,
                                     int line,
                                     const char* message)
{
    std::ostringstream oss;
    oss << "ENFORCE failed: " << (message ? message : "no message")
        << " [" << expr << "] @" << file << ':' << line;
    throw std::runtime_error(oss.str());
}
#endif

} // namespace common::detail

#if defined(NDEBUG)
#    define ENFORCE(expr, message) (void)sizeof(expr)
#else
#    define ENFORCE(expr, message)                                                         \
        do                                                                                \
        {                                                                                 \
            if (!(expr))                                                                  \
            {                                                                             \
                ::common::detail::enforceFail(#expr, __FILE__, __LINE__, (message));      \
            }                                                                             \
        } while (false)
#endif
