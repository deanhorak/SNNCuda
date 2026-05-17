#pragma once

#include "snncuda/core/Ids.h"

#include <cstdint>

namespace snncuda::snn {

struct Synapse {
    core::SynapseId id{core::invalid_id};
    core::NeuronId source{core::invalid_id};
    core::NeuronId target{core::invalid_id};
    float weight{1.0F};
    float max_weight{2.0F};
    std::uint32_t delay_ticks{1};
};

} // namespace snncuda::snn

