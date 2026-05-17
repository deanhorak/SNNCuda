#pragma once

#include "snncuda/core/Ids.h"
#include "snncuda/storage/LruCache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace snncuda::runtime {

struct NeuronState {
    core::NeuronId id{core::invalid_id};
    float membrane_potential{0.0F};
    float threshold{1.0F};
    std::uint64_t last_active_tick{0};
    std::uint64_t spike_count{0};
};

class NeuronStateStore {
public:
    virtual ~NeuronStateStore() = default;

    [[nodiscard]] virtual std::optional<NeuronState> load(core::NeuronId id) = 0;
    virtual void save(const NeuronState& state) = 0;
};

class MemoryNeuronStateStore final : public NeuronStateStore {
public:
    [[nodiscard]] std::optional<NeuronState> load(core::NeuronId id) override;
    void save(const NeuronState& state) override;

private:
    std::unordered_map<core::NeuronId, NeuronState> states_;
};

class NeuronStateCache {
public:
    NeuronStateCache(std::size_t resident_capacity, NeuronStateStore& backing_store);

    [[nodiscard]] NeuronState load_for_spike(core::NeuronId id);
    void store_after_compute(NeuronState state);
    [[nodiscard]] std::size_t resident_count() const noexcept;

private:
    storage::LruCache<core::NeuronId, NeuronState> cache_;
    NeuronStateStore& backing_store_;
};

} // namespace snncuda::runtime
