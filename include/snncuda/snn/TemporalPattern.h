#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace snncuda::snn {

struct TemporalPatternConfig {
    std::uint64_t window_ticks{500};
    float similarity_threshold{0.93F};
    std::size_t max_reference_patterns{500};
};

struct TemporalPattern {
    std::vector<std::uint32_t> spike_offsets;
};

class TemporalPatternMatcher {
public:
    explicit TemporalPatternMatcher(TemporalPatternConfig config = {});

    [[nodiscard]] float best_similarity(const TemporalPattern& candidate) const;
    [[nodiscard]] bool matches(const TemporalPattern& candidate) const;
    void learn(TemporalPattern pattern);
    [[nodiscard]] std::size_t pattern_count() const noexcept;

private:
    [[nodiscard]] static float cosine_similarity(
        const TemporalPattern& left,
        const TemporalPattern& right);

    TemporalPatternConfig config_;
    std::vector<TemporalPattern> references_;
};

} // namespace snncuda::snn
