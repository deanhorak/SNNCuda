#pragma once

#include "snncuda/backends/Backend.h"

namespace snncuda::backends {

class CpuBackend final : public Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
};

} // namespace snncuda::backends

