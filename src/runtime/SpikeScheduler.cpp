#include "snncuda/runtime/SpikeScheduler.h"

#include <stdexcept>
#include <utility>

namespace snncuda::runtime {

SpikeScheduler::SpikeScheduler(std::size_t wheel_slots)
    : wheel_(wheel_slots) {
    if (wheel_slots == 0) {
        throw std::invalid_argument("Spike scheduler timing wheel must have at least one slot");
    }
}

void SpikeScheduler::schedule(snn::SpikeEvent event) {
    // Late events are still delivered deterministically at the next scheduler poll.
    if (event.delivery_tick < next_pop_tick_) {
        event.delivery_tick = next_pop_tick_;
    }

    wheel_[slot_for(event.delivery_tick)].push_back(event);
    ++pending_count_;
}

std::vector<snn::SpikeEvent> SpikeScheduler::pop_due(std::uint64_t tick) {
    std::vector<snn::SpikeEvent> due;

    while (next_pop_tick_ <= tick) {
        auto& bucket = wheel_[slot_for(next_pop_tick_)];
        Bucket retained;
        retained.reserve(bucket.size());

        for (auto& event : bucket) {
            if (event.delivery_tick <= tick) {
                due.push_back(std::move(event));
                --pending_count_;
            } else {
                retained.push_back(std::move(event));
            }
        }

        bucket = std::move(retained);
        ++next_pop_tick_;
    }

    return due;
}

bool SpikeScheduler::empty() const noexcept {
    return pending_count_ == 0;
}

std::size_t SpikeScheduler::pending_count() const noexcept {
    return pending_count_;
}

std::size_t SpikeScheduler::wheel_slots() const noexcept {
    return wheel_.size();
}

std::size_t SpikeScheduler::slot_for(std::uint64_t tick) const noexcept {
    return static_cast<std::size_t>(tick % wheel_.size());
}

NeuronExecutionScheduler::NeuronExecutionScheduler(NeuronStateCache& cache)
    : cache_(cache) {
}

bool NeuronExecutionScheduler::process_spike(const snn::SpikeEvent& event, std::uint64_t tick) {
    auto state = cache_.load_for_spike(event.target_neuron);
    state.membrane_potential += event.weight;
    state.last_active_tick = tick;
    bool fired = false;
    if (state.membrane_potential >= state.threshold) {
        state.membrane_potential = 0.0F;
        ++state.spike_count;
        fired = true;
    }
    cache_.store_after_compute(state);
    return fired;
}

} // namespace snncuda::runtime
