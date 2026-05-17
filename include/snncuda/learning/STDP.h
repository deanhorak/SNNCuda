#pragma once

#include "snncuda/snn/SpikeEvent.h"
#include "snncuda/snn/Synapse.h"

namespace snncuda::learning {

struct StdpConfig {
    bool enabled{true};
    float ltp_amplitude{0.01F};
    float ltd_amplitude{0.012F};
    float ltp_tau_ticks{20.0F};
    float ltd_tau_ticks{20.0F};
    float min_weight{0.0F};
    float max_weight{2.0F};
};

class StdpRule {
public:
    explicit StdpRule(StdpConfig config = {});

    void set_enabled(bool enabled) noexcept;
    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] float weight_delta(const snn::RetrogradeEvent& event) const noexcept;
    void apply(snn::Synapse& synapse, const snn::RetrogradeEvent& event) const noexcept;

private:
    StdpConfig config_;
};

} // namespace snncuda::learning

