#pragma once

#include "snncuda/declarative/FormatParser.h"

namespace snncuda::declarative {

class DeclarativeLoader {
public:
    DeclarativeLoader();

    void register_parser(std::unique_ptr<FormatParser> parser);
    [[nodiscard]] NetworkIR parse_only(const std::filesystem::path& path) const;

private:
    ParserRegistry registry_;
};

} // namespace snncuda::declarative

