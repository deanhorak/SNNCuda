#pragma once

#include "snncuda/declarative/NetworkIR.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace snncuda::declarative {

class FormatParser {
public:
    virtual ~FormatParser() = default;

    [[nodiscard]] virtual bool can_parse(const std::filesystem::path& path) const = 0;
    [[nodiscard]] virtual std::string format_name() const = 0;
    [[nodiscard]] virtual NetworkIR parse(const std::filesystem::path& path) const = 0;
};

class ParserRegistry {
public:
    void register_parser(std::unique_ptr<FormatParser> parser);
    [[nodiscard]] const FormatParser* find_parser(const std::filesystem::path& path) const;
    [[nodiscard]] std::vector<std::string> formats() const;

private:
    std::vector<std::unique_ptr<FormatParser>> parsers_;
};

} // namespace snncuda::declarative

