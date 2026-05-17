#pragma once

#include "snncuda/backends/Backend.h"
#include "snncuda/core/Ids.h"
#include "snncuda/declarative/Connectome.h"

#include <cstdint>
#include <string>
#include <vector>

namespace snncuda::backends {

struct CudaPropagationResult {
    bool executed{false};
    std::uint64_t delivered_spikes{0};
    std::uint64_t fired_spikes{0};
    double elapsed_seconds{0.0};
};

class CudaBackend final : public Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] std::string availability_status() const;

    [[nodiscard]] CudaPropagationResult run_resident_propagation(
        const declarative::Connectome& connectome,
        const std::vector<core::NeuronId>& initially_active,
        std::uint32_t max_steps) const;
};

} // namespace snncuda::backends
