#pragma once

#include <string_view>

namespace snncuda::backends {

class Backend {
public:
    virtual ~Backend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool available() const noexcept = 0;
};

} // namespace snncuda::backends

