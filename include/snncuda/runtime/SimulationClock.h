#pragma once

#include <cstdint>

namespace snncuda::runtime {

class SimulationClock {
public:
    using tick_type = std::uint64_t;

    explicit SimulationClock(double timestep_ms = 1.0);

    [[nodiscard]] tick_type tick() const noexcept;
    [[nodiscard]] double time_ms() const noexcept;
    [[nodiscard]] double timestep_ms() const noexcept;

    void advance();
    void reset() noexcept;

private:
    tick_type tick_{0};
    double timestep_ms_{1.0};
};

} // namespace snncuda::runtime

