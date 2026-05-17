#include "snncuda/declarative/DeclarativeLoader.h"

#include "snncuda/declarative/parsers/BuiltinParsers.h"

#include <stdexcept>

namespace snncuda::declarative {

DeclarativeLoader::DeclarativeLoader() {
    register_parser(std::make_unique<NativeJsonParser>());
    register_parser(std::make_unique<SonataParser>());
    register_parser(std::make_unique<NeuroMLParser>());
    register_parser(std::make_unique<HocParser>());
}

void DeclarativeLoader::register_parser(std::unique_ptr<FormatParser> parser) {
    registry_.register_parser(std::move(parser));
}

NetworkIR DeclarativeLoader::parse_only(const std::filesystem::path& path) const {
    const auto* parser = registry_.find_parser(path);
    if (parser == nullptr) {
        throw std::invalid_argument("No parser registered for " + path.string());
    }
    return parser->parse(path);
}

} // namespace snncuda::declarative

