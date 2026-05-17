#include "snncuda/declarative/FormatParser.h"

#include <algorithm>

namespace snncuda::declarative {

void ParserRegistry::register_parser(std::unique_ptr<FormatParser> parser) {
    parsers_.push_back(std::move(parser));
}

const FormatParser* ParserRegistry::find_parser(const std::filesystem::path& path) const {
    const auto found = std::find_if(parsers_.begin(), parsers_.end(), [&path](const auto& parser) {
        return parser->can_parse(path);
    });
    return found == parsers_.end() ? nullptr : found->get();
}

std::vector<std::string> ParserRegistry::formats() const {
    std::vector<std::string> result;
    for (const auto& parser : parsers_) {
        result.push_back(parser->format_name());
    }
    return result;
}

} // namespace snncuda::declarative

