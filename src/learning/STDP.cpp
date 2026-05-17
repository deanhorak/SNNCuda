#include "snncuda/learning/STDP.h"

#include <algorithm>
#include <cmath>

namespace snncuda::learning {

StdpRule::StdpRule(StdpConfig config)
    : config_(config) {
}

void StdpRule::set_enabled(bool enabled) noexcept {
    config_.enabled = enabled;
}

bool StdpRule::enabled() const noexcept {
    return config_.enabled;
}

float StdpRule::weight_delta(const snn::RetrogradeEvent& event) const noexcept {
    if (!config_.enabled) {
        return 0.0F;
    }

    const auto dt = static_cast<float>(
        static_cast<long long>(event.post_tick) - static_cast<long long>(event.pre_tick));

    if (dt > 0.0F) {
        return config_.ltp_amplitude * std::exp(-dt / config_.ltp_tau_ticks);
    }
    if (dt < 0.0F) {
        return -config_.ltd_amplitude * std::exp(dt / config_.ltd_tau_ticks);
    }
    return 0.0F;
}

void StdpRule::apply(snn::Synapse& synapse, const snn::RetrogradeEvent& event) const noexcept {
    const auto maximum = std::min(config_.max_weight, synapse.max_weight);
    synapse.weight = std::clamp(synapse.weight + weight_delta(event), config_.min_weight, maximum);
}

} // namespace snncuda::learning

