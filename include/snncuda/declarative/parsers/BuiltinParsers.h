#pragma once

#include "snncuda/declarative/FormatParser.h"

namespace snncuda::declarative {

class NativeJsonParser final : public FormatParser {
public:
    [[nodiscard]] bool can_parse(const std::filesystem::path& path) const override;
    [[nodiscard]] std::string format_name() const override;
    [[nodiscard]] NetworkIR parse(const std::filesystem::path& path) const override;
};

class SonataParser final : public FormatParser {
public:
    [[nodiscard]] bool can_parse(const std::filesystem::path& path) const override;
    [[nodiscard]] std::string format_name() const override;
    [[nodiscard]] NetworkIR parse(const std::filesystem::path& path) const override;
};

class NeuroMLParser final : public FormatParser {
public:
    [[nodiscard]] bool can_parse(const std::filesystem::path& path) const override;
    [[nodiscard]] std::string format_name() const override;
    [[nodiscard]] NetworkIR parse(const std::filesystem::path& path) const override;
};

class HocParser final : public FormatParser {
public:
    [[nodiscard]] bool can_parse(const std::filesystem::path& path) const override;
    [[nodiscard]] std::string format_name() const override;
    [[nodiscard]] NetworkIR parse(const std::filesystem::path& path) const override;
};

} // namespace snncuda::declarative

