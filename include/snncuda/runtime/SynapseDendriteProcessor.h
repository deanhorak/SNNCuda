#pragma once

#include "snncuda/declarative/Connectome.h"
#include "snncuda/learning/STDP.h"
#include "snncuda/runtime/NeuronStateCache.h"
#include "snncuda/snn/SpikeEvent.h"

#include <cstdint>
#include <optional>

namespace snncuda::runtime {

struct SynapseRuntimeState {
    core::SynapseId id{core::invalid_id};
    core::NeuronId source{core::invalid_id};
    core::NeuronId target{core::invalid_id};
    float weight{1.0F};
    float max_weight{2.0F};
    std::uint32_t delay_ticks{1};
    snn::DendriticCompartment compartment{snn::DendriticCompartment::Basal};
    snn::ReceptorType receptor{snn::ReceptorType::Ampa};
    bool plasticity_enabled{true};
    std::optional<std::uint64_t> last_pre_spike_tick;
    std::optional<std::uint64_t> last_post_spike_tick;
    std::uint64_t pre_spike_count{0};
    std::uint64_t plasticity_update_count{0};
    float last_weight_delta{0.0F};
};

struct SynapseDendriteConfig {
    learning::StdpConfig stdp;
    bool temporal_learning_enabled{true};
};

struct SynapseProcessingResult {
    bool fired{false};
    bool pattern_matched{false};
    float applied_current{0.0F};
    float membrane_potential{0.0F};
};

class SynapseDendriteProcessor {
public:
    explicit SynapseDendriteProcessor(SynapseDendriteConfig config = {});

    [[nodiscard]] SynapseRuntimeState initialize_synapse(
        const declarative::ConnectomeSynapse& synapse) const;

    [[nodiscard]] SynapseProcessingResult process_external_input(
        NeuronState& target,
        const snn::SpikeEvent& event,
        std::uint64_t tick) const;

    [[nodiscard]] SynapseProcessingResult process_synaptic_input(
        NeuronState& target,
        SynapseRuntimeState& synapse,
        const snn::SpikeEvent& event,
        std::uint64_t tick);

    void apply_post_spike_plasticity(
        NeuronState& target,
        SynapseRuntimeState& synapse,
        std::uint64_t tick);

private:
    [[nodiscard]] SynapseProcessingResult process_input(
        NeuronState& target,
        snn::DendriticCompartment compartment,
        snn::ReceptorType receptor,
        float weight,
        std::uint64_t tick) const;

    void decay_compartments(NeuronState& target, std::uint64_t tick) const;
    [[nodiscard]] static float receptor_gain(snn::ReceptorType receptor) noexcept;
    [[nodiscard]] static float receptor_decay_tau(snn::ReceptorType receptor) noexcept;
    [[nodiscard]] static float receptor_sign(snn::ReceptorType receptor) noexcept;
    [[nodiscard]] static NeuronState::CompartmentState& compartment(
        NeuronState& target,
        snn::DendriticCompartment compartment) noexcept;
    [[nodiscard]] static float membrane_from_compartments(const NeuronState& target) noexcept;
    void update_temporal_pattern(NeuronState& target, std::uint64_t tick) const;

    SynapseDendriteConfig config_;
    learning::StdpRule stdp_;
};

} // namespace snncuda::runtime
