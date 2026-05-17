#pragma once

#include "snncuda/core/Ids.h"

#include <cstdint>

namespace snncuda::snn {

struct SpikeEvent {
    core::NeuronId source_neuron{core::invalid_id};
    core::NeuronId target_neuron{core::invalid_id};
    core::SynapseId synapse{core::invalid_id};
    std::uint64_t delivery_tick{0};
    float weight{0.0F};
};

struct RetrogradeEvent {
    core::SynapseId synapse{core::invalid_id};
    core::NeuronId pre_neuron{core::invalid_id};
    core::NeuronId post_neuron{core::invalid_id};
    std::uint64_t pre_tick{0};
    std::uint64_t post_tick{0};
};

} // namespace snncuda::snn

