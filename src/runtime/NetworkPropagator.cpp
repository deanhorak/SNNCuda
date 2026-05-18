#include "snncuda/runtime/NetworkPropagator.h"

namespace snncuda::runtime {

NetworkPropagator::NetworkPropagator(
    const declarative::Connectome& connectome,
    NeuronStateStore& backing_store,
    std::size_t resident_neuron_capacity,
    std::size_t timing_wheel_slots)
    : connectome_(connectome)
    , backing_store_(backing_store)
    , cache_(resident_neuron_capacity, backing_store_)
    , scheduler_(timing_wheel_slots) {
    for (const auto& neuron : connectome_.neurons) {
        NeuronState state;
        state.id = neuron.id;
        state.threshold = neuron.params.threshold;
        state.pattern_window_ticks = neuron.params.pattern_window_ticks;
        state.similarity_threshold = neuron.params.similarity_threshold;
        state.max_reference_patterns = neuron.params.max_reference_patterns;
        backing_store_.save(state);
    }

    for (const auto& synapse : connectome_.synapses) {
        synapses_[synapse.id] = processor_.initialize_synapse(synapse);
        outgoing_[synapse.source].push_back(synapse.id);
        incoming_[synapse.target].push_back(synapse.id);
    }
}

void NetworkPropagator::inject(core::NeuronId target, std::uint64_t delivery_tick, float weight) {
    scheduler_.schedule({
        .target_neuron = target,
        .delivery_tick = delivery_tick,
        .weight = weight,
    });
}

void NetworkPropagator::run_until(std::uint64_t inclusive_tick) {
    for (std::uint64_t tick = 0; tick <= inclusive_tick; ++tick) {
        process_tick(tick);
    }
}

NeuronState NetworkPropagator::state(core::NeuronId id) {
    return cache_.load_for_spike(id);
}

std::uint64_t NetworkPropagator::spike_count(core::NeuronId id) {
    return state(id).spike_count;
}

std::uint64_t NetworkPropagator::delivered_spike_count() const noexcept {
    return delivered_spikes_;
}

std::uint64_t NetworkPropagator::fired_spike_count() const noexcept {
    return fired_spikes_;
}

std::uint64_t NetworkPropagator::cache_eviction_count() const noexcept {
    return cache_.eviction_count();
}

std::size_t NetworkPropagator::resident_neuron_count() const noexcept {
    return cache_.resident_count();
}

const SynapseRuntimeState* NetworkPropagator::synapse_state(core::SynapseId id) const {
    const auto found = synapses_.find(id);
    return found == synapses_.end() ? nullptr : &found->second;
}

bool NetworkPropagator::idle() const noexcept {
    return scheduler_.empty();
}

void NetworkPropagator::process_tick(std::uint64_t tick) {
    const auto due = scheduler_.pop_due(tick);
    for (const auto& event : due) {
        ++delivered_spikes_;

        auto target = cache_.load_for_spike(event.target_neuron);
        const auto synapse_found = synapses_.find(event.synapse);
        const auto result = synapse_found == synapses_.end()
            ? processor_.process_external_input(target, event, tick)
            : processor_.process_synaptic_input(target, synapse_found->second, event, tick);
        cache_.store_after_compute(target);

        if (!result.fired) {
            continue;
        }
        ++fired_spikes_;

        if (const auto incoming_found = incoming_.find(event.target_neuron);
            incoming_found != incoming_.end()) {
            auto post_state = cache_.load_for_spike(event.target_neuron);
            for (const auto synapse_id : incoming_found->second) {
                processor_.apply_post_spike_plasticity(post_state, synapses_.at(synapse_id), tick);
            }
            cache_.store_after_compute(post_state);
        }

        const auto found = outgoing_.find(event.target_neuron);
        if (found == outgoing_.end()) {
            continue;
        }

        for (const auto synapse_id : found->second) {
            const auto& synapse = synapses_.at(synapse_id);
            scheduler_.schedule({
                .source_neuron = synapse.source,
                .target_neuron = synapse.target,
                .synapse = synapse.id,
                .delivery_tick = tick + synapse.delay_ticks,
                .weight = synapse.weight,
            });
        }
    }
}

} // namespace snncuda::runtime
