#include "snncuda/runtime/SimulationClock.h"

#include <stdexcept>

namespace snncuda::runtime {

SimulationClock::SimulationClock(double timestep_ms)
    : timestep_ms_(timestep_ms) {
    if (timestep_ms_ <= 0.0) {
        throw std::invalid_argument("Simulation timestep must be positive");
    }
}

SimulationClock::tick_type SimulationClock::tick() const noexcept {
    return tick_;
}

double SimulationClock::time_ms() const noexcept {
    return static_cast<double>(tick_) * timestep_ms_;
}

double SimulationClock::timestep_ms() const noexcept {
    return timestep_ms_;
}

void SimulationClock::advance() {
    ++tick_;
}

void SimulationClock::reset() noexcept {
    tick_ = 0;
}

} // namespace snncuda::runtime

