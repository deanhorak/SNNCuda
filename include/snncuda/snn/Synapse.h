#pragma once

#include "snncuda/core/Ids.h"

#include <cstdint>

namespace snncuda::snn {

enum class DendriticCompartment {
    Soma,
    Basal,
    Apical,
    Inhibitory,
};

enum class ReceptorType {
    Ampa,
    Nmda,
    GabaA,
    GabaB,
};

struct Synapse {
    core::SynapseId id{core::invalid_id};
    core::NeuronId source{core::invalid_id};
    core::NeuronId target{core::invalid_id};
    float weight{1.0F};
    float max_weight{2.0F};
    std::uint32_t delay_ticks{1};
    DendriticCompartment compartment{DendriticCompartment::Basal};
    ReceptorType receptor{ReceptorType::Ampa};
    bool plasticity_enabled{true};
};

} // namespace snncuda::snn
