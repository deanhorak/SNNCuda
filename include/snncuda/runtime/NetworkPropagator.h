#pragma once

#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NeuronStateCache.h"
#include "snncuda/runtime/SpikeScheduler.h"
#include "snncuda/runtime/SynapseDendriteProcessor.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace snncuda::runtime {

class NetworkPropagator {
public:
    NetworkPropagator(
        const declarative::Connectome& connectome,
        NeuronStateStore& backing_store,
        std::size_t resident_neuron_capacity,
        std::size_t timing_wheel_slots = 4096);

    void inject(core::NeuronId target, std::uint64_t delivery_tick, float weight);
    void run_until(std::uint64_t inclusive_tick);

    [[nodiscard]] NeuronState state(core::NeuronId id);
    [[nodiscard]] std::uint64_t spike_count(core::NeuronId id);
    [[nodiscard]] std::uint64_t delivered_spike_count() const noexcept;
    [[nodiscard]] std::uint64_t fired_spike_count() const noexcept;
    [[nodiscard]] std::uint64_t cache_eviction_count() const noexcept;
    [[nodiscard]] std::size_t resident_neuron_count() const noexcept;
    [[nodiscard]] const SynapseRuntimeState* synapse_state(core::SynapseId id) const;
    [[nodiscard]] bool idle() const noexcept;

private:
    void process_tick(std::uint64_t tick);

    const declarative::Connectome& connectome_;
    NeuronStateStore& backing_store_;
    NeuronStateCache cache_;
    SpikeScheduler scheduler_;
    SynapseDendriteProcessor processor_;
    std::unordered_map<core::SynapseId, SynapseRuntimeState> synapses_;
    std::unordered_map<core::NeuronId, std::vector<core::SynapseId>> outgoing_;
    std::unordered_map<core::NeuronId, std::vector<core::SynapseId>> incoming_;
    std::uint64_t delivered_spikes_{0};
    std::uint64_t fired_spikes_{0};
};

} // namespace snncuda::runtime
