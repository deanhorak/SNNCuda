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
    , scheduler_(timing_wheel_slots)
    , executor_(cache_) {
    for (const auto& neuron : connectome_.neurons) {
        backing_store_.save({
            .id = neuron.id,
            .threshold = neuron.params.threshold,
        });
    }

    for (const auto& synapse : connectome_.synapses) {
        outgoing_[synapse.source].push_back(&synapse);
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

bool NetworkPropagator::idle() const noexcept {
    return scheduler_.empty();
}

void NetworkPropagator::process_tick(std::uint64_t tick) {
    const auto due = scheduler_.pop_due(tick);
    for (const auto& event : due) {
        ++delivered_spikes_;
        const auto fired = executor_.process_spike(event, tick);
        if (!fired) {
            continue;
        }
        ++fired_spikes_;

        const auto found = outgoing_.find(event.target_neuron);
        if (found == outgoing_.end()) {
            continue;
        }

        for (const auto* synapse : found->second) {
            scheduler_.schedule({
                .source_neuron = synapse->source,
                .target_neuron = synapse->target,
                .synapse = synapse->id,
                .delivery_tick = tick + synapse->delay_ticks,
                .weight = synapse->weight,
            });
        }
    }
}

} // namespace snncuda::runtime
