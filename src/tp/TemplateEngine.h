#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

namespace tp
{

class TemplateContext
{
public:
    void clear();
    void set(const std::string& key, std::string value);
    void set(const std::string& key, std::string value, bool truthy);
    void setBool(const std::string& key, bool value);

    [[nodiscard]] std::string value(const std::string& key) const;
    [[nodiscard]] bool truthy(const std::string& key) const;

private:
    struct Entry
    {
        std::string text;
        bool truthy{false};
    };

    std::unordered_map<std::string, Entry> m_entries;
};

class TemplateEngine
{
public:
    static std::string render(std::string_view tpl, const TemplateContext& context);
};

} // namespace tp

