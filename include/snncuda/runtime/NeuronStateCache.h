#pragma once

#include "snncuda/core/Ids.h"
#include "snncuda/snn/TemporalPattern.h"
#include "snncuda/storage/LruCache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace snncuda::runtime {

struct NeuronState {
    core::NeuronId id{core::invalid_id};
    float membrane_potential{0.0F};
    float threshold{1.0F};
    std::uint64_t pattern_window_ticks{500};
    float similarity_threshold{0.93F};
    std::size_t max_reference_patterns{500};
    std::uint64_t last_active_tick{0};
    std::optional<std::uint64_t> last_fired_tick;
    std::uint64_t spike_count{0};

    struct CompartmentState {
        float current{0.0F};
        std::uint64_t last_update_tick{0};
        float decay_tau_ticks{5.0F};
    };

    struct TemporalPatternState {
        std::uint64_t window_start_tick{0};
        std::vector<std::uint32_t> spike_offsets;
        std::vector<snn::TemporalPattern> learned_patterns;
        bool last_match{false};
        std::uint64_t match_count{0};
    };

    CompartmentState soma;
    CompartmentState basal;
    CompartmentState apical;
    CompartmentState inhibitory;
    TemporalPatternState temporal_pattern;
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
    [[nodiscard]] std::uint64_t load_hit_count() const noexcept;
    [[nodiscard]] std::uint64_t load_miss_count() const noexcept;
    [[nodiscard]] std::uint64_t eviction_count() const noexcept;

private:
    storage::LruCache<core::NeuronId, NeuronState> cache_;
    NeuronStateStore& backing_store_;
    std::uint64_t load_hits_{0};
    std::uint64_t load_misses_{0};
    std::uint64_t evictions_{0};
};

} // namespace snncuda::runtime
