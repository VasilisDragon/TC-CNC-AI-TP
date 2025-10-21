#include "tp/TemplateEngine.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using tp::TemplateContext;

[[nodiscard]] std::string_view trimView(std::string_view view)
{
    while (!view.empty() && std::isspace(static_cast<unsigned char>(view.front())))
    {
        view.remove_prefix(1);
    }
    while (!view.empty() && std::isspace(static_cast<unsigned char>(view.back())))
    {
        view.remove_suffix(1);
    }
    return view;
}

[[nodiscard]] std::vector<std::string_view> splitParts(std::string_view view)
{
    std::vector<std::string_view> parts;
    std::size_t pos = 0;
    while (pos < view.size())
    {
        while (pos < view.size() && std::isspace(static_cast<unsigned char>(view[pos])))
        {
            ++pos;
        }
        if (pos >= view.size())
        {
            break;
        }
        std::size_t end = pos;
        while (end < view.size() && !std::isspace(static_cast<unsigned char>(view[end])))
        {
            ++end;
        }
        parts.emplace_back(view.substr(pos, end - pos));
        pos = end;
    }
    return parts;
}

struct RenderResult
{
    std::string body;
    std::string elseBody;
    bool hasElse{false};
};

RenderResult renderUntil(const TemplateContext& context,
                         std::string_view tpl,
                         std::size_t& pos,
                         std::string_view endTag);

void appendToken(std::string& out,
                 const TemplateContext& context,
                 std::string_view tag)
{
    const std::string key(trimView(tag));
    out.append(context.value(key));
}

void appendConditional(std::string& out,
                       const TemplateContext& context,
                       std::string_view directive,
                       std::string_view predicate,
                       std::size_t& pos,
                       std::string_view tpl)
{
    RenderResult nested = renderUntil(context, tpl, pos, directive);
    bool condition = false;

    if (directive == "if")
    {
        const std::string key(trimView(predicate));
        condition = context.truthy(key);
    }
    else if (directive == "unless")
    {
        const std::string key(trimView(predicate));
        condition = !context.truthy(key);
    }
    else if (directive == "ifEq")
    {
        const auto parts = splitParts(trimView(predicate));
        if (parts.size() >= 2)
        {
            const std::string lhs(parts[0]);
            const std::string rhs(parts[1]);
            condition = (context.value(lhs) == rhs);
        }
        else
        {
            condition = false;
        }
    }

    if (condition)
    {
        out.append(nested.body);
    }
    else if (nested.hasElse)
    {
        out.append(nested.elseBody);
    }
}

RenderResult renderUntil(const TemplateContext& context,
                         std::string_view tpl,
                         std::size_t& pos,
                         std::string_view endTag)
{
    RenderResult result;
    std::string* active = &result.body;

    while (pos < tpl.size())
    {
        const std::size_t open = tpl.find("{{", pos);
        if (open == std::string_view::npos)
        {
            active->append(tpl.substr(pos));
            pos = tpl.size();
            break;
        }

        active->append(tpl.substr(pos, open - pos));
        pos = open + 2;

        const std::size_t close = tpl.find("}}", pos);
        if (close == std::string_view::npos)
        {
            active->append("{{");
            active->append(tpl.substr(pos));
            pos = tpl.size();
            break;
        }

        const std::string_view rawTag = tpl.substr(pos, close - pos);
        pos = close + 2;
        const std::string_view trimmed = trimView(rawTag);

        if (!endTag.empty())
        {
            if ((endTag == "if" || endTag == "unless" || endTag == "ifEq") && trimmed == "else")
            {
                result.hasElse = true;
                active = &result.elseBody;
                continue;
            }

            const std::string closing = "/" + std::string(endTag);
            if (trimmed == closing)
            {
                return result;
            }
        }

        if (!trimmed.empty() && trimmed.front() == '#')
        {
            const std::vector<std::string_view> parts = splitParts(trimmed);
            if (parts.empty())
            {
                continue;
            }
            const std::string_view directive = parts.front().substr(1);
            std::string_view predicate;
            if (parts.size() >= 2)
            {
                const std::size_t offset = static_cast<std::size_t>(parts.front().data() - trimmed.data())
                                           + parts.front().size();
                predicate = trimView(trimmed.substr(offset));
            }

            appendConditional(*active, context, directive, predicate, pos, tpl);
            continue;
        }

        if (!trimmed.empty() && trimmed.front() == '/')
        {
            // Unmatched closing tag, skip.
            continue;
        }

        appendToken(*active, context, trimmed);
    }

    return result;
}

} // namespace

namespace tp
{

void TemplateContext::clear()
{
    m_entries.clear();
}

void TemplateContext::set(const std::string& key, std::string value)
{
    const bool truthy = !value.empty();
    set(key, std::move(value), truthy);
}

void TemplateContext::set(const std::string& key, std::string value, bool truthy)
{
    m_entries[key] = Entry{std::move(value), truthy};
}

void TemplateContext::setBool(const std::string& key, bool value)
{
    m_entries[key] = Entry{value ? "1" : std::string{}, value};
}

std::string TemplateContext::value(const std::string& key) const
{
    const auto it = m_entries.find(key);
    if (it == m_entries.end())
    {
        return {};
    }
    return it->second.text;
}

bool TemplateContext::truthy(const std::string& key) const
{
    const auto it = m_entries.find(key);
    if (it == m_entries.end())
    {
        return false;
    }
    return it->second.truthy;
}

std::string TemplateEngine::render(std::string_view tpl, const TemplateContext& context)
{
    std::size_t pos = 0;
    RenderResult result = renderUntil(context, tpl, pos, {});
    return std::move(result.body);
}

} // namespace tp
