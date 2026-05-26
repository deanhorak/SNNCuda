#include "snncuda/runtime/SynapseDendriteProcessor.h"

#include <algorithm>
#include <cmath>

namespace snncuda::runtime {
namespace {

[[nodiscard]] snn::TemporalPattern make_pattern(
    const std::vector<std::uint32_t>& offsets) {
    return snn::TemporalPattern{.spike_offsets = offsets};
}

[[nodiscard]] bool matches_learned_pattern(const NeuronState& target) {
    if (target.temporal_pattern.learned_patterns.empty()
        || target.temporal_pattern.spike_offsets.size() < 2) {
        return false;
    }

    snn::TemporalPatternMatcher matcher({
        .window_ticks = target.pattern_window_ticks,
        .similarity_threshold = target.similarity_threshold,
        .max_reference_patterns = target.max_reference_patterns,
    });
    for (const auto& pattern : target.temporal_pattern.learned_patterns) {
        matcher.learn(pattern);
    }
    return matcher.matches(make_pattern(target.temporal_pattern.spike_offsets));
}

void learn_pattern(NeuronState& target) {
    if (target.temporal_pattern.spike_offsets.size() < 2) {
        return;
    }
    if (target.temporal_pattern.learned_patterns.size() >= target.max_reference_patterns) {
        target.temporal_pattern.learned_patterns.erase(
            target.temporal_pattern.learned_patterns.begin());
    }
    target.temporal_pattern.learned_patterns.push_back(
        make_pattern(target.temporal_pattern.spike_offsets));
}

[[nodiscard]] NeuronState::CompartmentState reset_compartment(
    std::uint64_t tick,
    float decay_tau_ticks) {
    NeuronState::CompartmentState state;
    state.last_update_tick = tick;
    state.decay_tau_ticks = decay_tau_ticks;
    return state;
}

} // namespace

SynapseDendriteProcessor::SynapseDendriteProcessor(SynapseDendriteConfig config)
    : config_(config)
    , stdp_(config.stdp) {
}

SynapseRuntimeState SynapseDendriteProcessor::initialize_synapse(
    const declarative::ConnectomeSynapse& synapse) const {
    SynapseRuntimeState state;
    state.id = synapse.id;
    state.source = synapse.source;
    state.target = synapse.target;
    state.weight = synapse.weight;
    state.max_weight = synapse.max_weight;
    state.delay_ticks = synapse.delay_ticks;
    state.spike_code_offsets = synapse.spike_code_offsets.empty()
        ? std::vector<std::uint32_t>{0}
        : synapse.spike_code_offsets;
    state.compartment = synapse.compartment;
    state.receptor = synapse.receptor;
    state.plasticity_enabled = synapse.plasticity_enabled;
    return state;
}

SynapseProcessingResult SynapseDendriteProcessor::process_external_input(
    NeuronState& target,
    const snn::SpikeEvent& event,
    std::uint64_t tick) const {
    return process_input(
        target,
        snn::DendriticCompartment::Soma,
        snn::ReceptorType::Ampa,
        event.weight,
        tick);
}

SynapseProcessingResult SynapseDendriteProcessor::process_synaptic_input(
    NeuronState& target,
    SynapseRuntimeState& synapse,
    const snn::SpikeEvent& event,
    std::uint64_t tick) {
    (void)event;
    const auto pre_tick = tick >= synapse.delay_ticks ? tick - synapse.delay_ticks : tick;
    if (synapse.plasticity_enabled && synapse.last_post_spike_tick.has_value()) {
        snn::Synapse mutable_synapse{
            .id = synapse.id,
            .source = synapse.source,
            .target = synapse.target,
            .weight = synapse.weight,
            .max_weight = synapse.max_weight,
            .delay_ticks = synapse.delay_ticks,
            .compartment = synapse.compartment,
            .receptor = synapse.receptor,
            .plasticity_enabled = synapse.plasticity_enabled,
        };
        const auto before = mutable_synapse.weight;
        stdp_.apply(
            mutable_synapse,
            {
                .synapse = synapse.id,
                .pre_neuron = synapse.source,
                .post_neuron = synapse.target,
                .pre_tick = pre_tick,
                .post_tick = *synapse.last_post_spike_tick,
            });
        synapse.weight = mutable_synapse.weight;
        synapse.last_weight_delta = synapse.weight - before;
        if (synapse.last_weight_delta != 0.0F) {
            ++synapse.plasticity_update_count;
        }
    }

    synapse.last_pre_spike_tick = pre_tick;
    ++synapse.pre_spike_count;
    auto result = process_input(
        target,
        synapse.compartment,
        synapse.receptor,
        synapse.weight,
        tick);
    update_synapse_code_pattern(synapse, target, tick);
    result.pattern_matched = synapse.last_code_match;
    return result;
}

void SynapseDendriteProcessor::apply_post_spike_plasticity(
    NeuronState& target,
    SynapseRuntimeState& synapse,
    std::uint64_t tick) {
    synapse.last_post_spike_tick = tick;
    if (!synapse.plasticity_enabled || !synapse.last_pre_spike_tick.has_value()) {
        return;
    }

    snn::Synapse mutable_synapse{
        .id = synapse.id,
        .source = synapse.source,
        .target = synapse.target,
        .weight = synapse.weight,
        .max_weight = synapse.max_weight,
        .delay_ticks = synapse.delay_ticks,
        .compartment = synapse.compartment,
        .receptor = synapse.receptor,
        .plasticity_enabled = synapse.plasticity_enabled,
    };
    const auto before = mutable_synapse.weight;
    stdp_.apply(
        mutable_synapse,
        {
            .synapse = synapse.id,
            .pre_neuron = synapse.source,
            .post_neuron = target.id,
            .pre_tick = *synapse.last_pre_spike_tick,
            .post_tick = tick,
        });
    synapse.weight = mutable_synapse.weight;
    synapse.last_weight_delta = synapse.weight - before;
    if (synapse.last_weight_delta != 0.0F) {
        ++synapse.plasticity_update_count;
    }
}

SynapseProcessingResult SynapseDendriteProcessor::process_input(
    NeuronState& target,
    snn::DendriticCompartment dendritic_compartment,
    snn::ReceptorType receptor,
    float weight,
    std::uint64_t tick) const {
    decay_compartments(target, tick);

    const auto applied_current = weight * receptor_gain(receptor) * receptor_sign(receptor);
    auto& target_compartment = compartment(target, dendritic_compartment);
    target_compartment.current += applied_current;
    target_compartment.last_update_tick = tick;
    target_compartment.decay_tau_ticks = receptor_decay_tau(receptor);

    target.last_active_tick = tick;
    update_temporal_pattern(target, tick);
    target.membrane_potential = membrane_from_compartments(target);

    SynapseProcessingResult result{
        .pattern_matched = target.temporal_pattern.last_match,
        .applied_current = applied_current,
        .membrane_potential = target.membrane_potential,
    };

    if (target.membrane_potential >= target.threshold) {
        target.membrane_potential = 0.0F;
        target.soma = reset_compartment(tick, receptor_decay_tau(snn::ReceptorType::Ampa));
        target.basal = reset_compartment(tick, receptor_decay_tau(snn::ReceptorType::Ampa));
        target.apical = reset_compartment(tick, receptor_decay_tau(snn::ReceptorType::Nmda));
        target.inhibitory = reset_compartment(tick, receptor_decay_tau(snn::ReceptorType::GabaA));
        target.last_fired_tick = tick;
        ++target.spike_count;
        result.fired = true;
        result.membrane_potential = target.membrane_potential;
    }

    return result;
}

void SynapseDendriteProcessor::decay_compartments(NeuronState& target, std::uint64_t tick) const {
    auto decay = [tick](NeuronState::CompartmentState& state) {
        if (tick <= state.last_update_tick || state.current == 0.0F) {
            state.last_update_tick = tick;
            return;
        }
        const auto elapsed = static_cast<float>(tick - state.last_update_tick);
        state.current = static_cast<float>(state.current * std::exp(-elapsed / state.decay_tau_ticks));
        state.last_update_tick = tick;
    };

    decay(target.soma);
    decay(target.basal);
    decay(target.apical);
    decay(target.inhibitory);
}

float SynapseDendriteProcessor::receptor_gain(snn::ReceptorType receptor) noexcept {
    switch (receptor) {
    case snn::ReceptorType::Ampa:
        return 1.0F;
    case snn::ReceptorType::Nmda:
        return 0.55F;
    case snn::ReceptorType::GabaA:
        return 1.0F;
    case snn::ReceptorType::GabaB:
        return 0.65F;
    }
    return 1.0F;
}

float SynapseDendriteProcessor::receptor_decay_tau(snn::ReceptorType receptor) noexcept {
    switch (receptor) {
    case snn::ReceptorType::Ampa:
        return 5.0F;
    case snn::ReceptorType::Nmda:
        return 100.0F;
    case snn::ReceptorType::GabaA:
        return 10.0F;
    case snn::ReceptorType::GabaB:
        return 150.0F;
    }
    return 5.0F;
}

float SynapseDendriteProcessor::receptor_sign(snn::ReceptorType receptor) noexcept {
    switch (receptor) {
    case snn::ReceptorType::Ampa:
    case snn::ReceptorType::Nmda:
        return 1.0F;
    case snn::ReceptorType::GabaA:
    case snn::ReceptorType::GabaB:
        return -1.0F;
    }
    return 1.0F;
}

NeuronState::CompartmentState& SynapseDendriteProcessor::compartment(
    NeuronState& target,
    snn::DendriticCompartment dendritic_compartment) noexcept {
    switch (dendritic_compartment) {
    case snn::DendriticCompartment::Soma:
        return target.soma;
    case snn::DendriticCompartment::Basal:
        return target.basal;
    case snn::DendriticCompartment::Apical:
        return target.apical;
    case snn::DendriticCompartment::Inhibitory:
        return target.inhibitory;
    }
    return target.basal;
}

float SynapseDendriteProcessor::membrane_from_compartments(const NeuronState& target) noexcept {
    return target.soma.current
        + target.basal.current
        + target.apical.current
        + target.inhibitory.current;
}

void SynapseDendriteProcessor::update_temporal_pattern(
    NeuronState& target,
    std::uint64_t tick) const {
    if (!config_.temporal_learning_enabled) {
        target.temporal_pattern.last_match = false;
        return;
    }

    auto& pattern = target.temporal_pattern;
    if (pattern.spike_offsets.empty()
        || tick - pattern.window_start_tick > target.pattern_window_ticks) {
        pattern.window_start_tick = tick;
        pattern.spike_offsets.clear();
    }

    pattern.spike_offsets.push_back(static_cast<std::uint32_t>(tick - pattern.window_start_tick));
    pattern.last_match = matches_learned_pattern(target);
    if (pattern.last_match) {
        ++pattern.match_count;
        return;
    }
    learn_pattern(target);
}

void SynapseDendriteProcessor::update_synapse_code_pattern(
    SynapseRuntimeState& synapse,
    const NeuronState& target,
    std::uint64_t tick) const {
    if (!config_.temporal_learning_enabled) {
        synapse.last_code_match = false;
        return;
    }

    if (synapse.code_offsets.empty()
        || tick - synapse.code_window_start_tick > target.pattern_window_ticks) {
        synapse.code_window_start_tick = tick;
        synapse.code_offsets.clear();
    }

    synapse.code_offsets.push_back(static_cast<std::uint32_t>(tick - synapse.code_window_start_tick));
    synapse.last_code_match = false;
    const auto expected_count = std::max<std::size_t>(2, synapse.spike_code_offsets.size());
    if (synapse.code_offsets.size() < expected_count) {
        return;
    }
    if (synapse.code_offsets.size() > expected_count) {
        synapse.code_offsets.erase(
            synapse.code_offsets.begin(),
            synapse.code_offsets.end() - static_cast<std::ptrdiff_t>(expected_count));
    }

    snn::TemporalPatternMatcher matcher({
        .window_ticks = target.pattern_window_ticks,
        .similarity_threshold = target.similarity_threshold,
        .max_reference_patterns = target.max_reference_patterns,
    });
    for (const auto& pattern : synapse.learned_code_patterns) {
        matcher.learn(pattern);
    }

    const auto current = make_pattern(synapse.code_offsets);
    synapse.last_code_match = matcher.matches(current);
    if (synapse.last_code_match) {
        ++synapse.code_match_count;
        return;
    }

    if (synapse.learned_code_patterns.size() >= target.max_reference_patterns) {
        synapse.learned_code_patterns.erase(synapse.learned_code_patterns.begin());
    }
    synapse.learned_code_patterns.push_back(current);
}

} // namespace snncuda::runtime
