#include "snncuda/runtime/NeuronStateCache.h"

namespace snncuda::runtime {

std::optional<NeuronState> MemoryNeuronStateStore::load(core::NeuronId id) {
    const auto found = states_.find(id);
    if (found == states_.end()) {
        return std::nullopt;
    }
    return found->second;
}

void MemoryNeuronStateStore::save(const NeuronState& state) {
    states_[state.id] = state;
}

NeuronStateCache::NeuronStateCache(std::size_t resident_capacity, NeuronStateStore& backing_store)
    : cache_(resident_capacity)
    , backing_store_(backing_store) {
}

NeuronState NeuronStateCache::load_for_spike(core::NeuronId id) {
    if (auto resident = cache_.get(id)) {
        ++load_hits_;
        return *resident;
    }

    ++load_misses_;
    if (auto stored = backing_store_.load(id)) {
        return *stored;
    }

    NeuronState state;
    state.id = id;
    return state;
}

void NeuronStateCache::store_after_compute(NeuronState state) {
    if (auto evicted = cache_.put(state.id, state)) {
        ++evictions_;
        backing_store_.save(evicted->second);
    }
}

std::size_t NeuronStateCache::resident_count() const noexcept {
    return cache_.size();
}

std::uint64_t NeuronStateCache::load_hit_count() const noexcept {
    return load_hits_;
}

std::uint64_t NeuronStateCache::load_miss_count() const noexcept {
    return load_misses_;
}

std::uint64_t NeuronStateCache::eviction_count() const noexcept {
    return evictions_;
}

} // namespace snncuda::runtime
