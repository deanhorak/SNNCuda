#pragma once

#include "snncuda/core/Ids.h"
#include "snncuda/declarative/NetworkIR.h"
#include "snncuda/snn/Synapse.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace snncuda::declarative {

struct ConnectomeNeuron {
    core::NeuronId id{core::invalid_id};
    std::string path;
    std::string population;
    NeuronParamsIR params;
};

struct ConnectomeSynapse {
    core::SynapseId id{core::invalid_id};
    core::NeuronId source{core::invalid_id};
    core::NeuronId target{core::invalid_id};
    float weight{1.0F};
    float max_weight{2.0F};
    std::uint32_t delay_ticks{1};
    snn::DendriticCompartment compartment{snn::DendriticCompartment::Basal};
    snn::ReceptorType receptor{snn::ReceptorType::Ampa};
    bool plasticity_enabled{true};
};

struct Connectome {
    std::vector<ConnectomeNeuron> neurons;
    std::vector<ConnectomeSynapse> synapses;
    std::unordered_map<std::string, std::vector<core::NeuronId>> populations;
};

class ConnectomeBuilder {
public:
    [[nodiscard]] Connectome build(const NetworkIR& ir) const;
};

} // namespace snncuda::declarative
