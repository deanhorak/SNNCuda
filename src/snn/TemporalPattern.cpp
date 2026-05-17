#include "snncuda/snn/TemporalPattern.h"

#include <algorithm>
#include <cmath>

namespace snncuda::snn {

TemporalPatternMatcher::TemporalPatternMatcher(TemporalPatternConfig config)
    : config_(config) {
}

float TemporalPatternMatcher::best_similarity(const TemporalPattern& candidate) const {
    float best = 0.0F;
    for (const auto& reference : references_) {
        best = std::max(best, cosine_similarity(reference, candidate));
    }
    return best;
}

bool TemporalPatternMatcher::matches(const TemporalPattern& candidate) const {
    return best_similarity(candidate) >= config_.similarity_threshold;
}

void TemporalPatternMatcher::learn(TemporalPattern pattern) {
    if (references_.size() >= config_.max_reference_patterns) {
        references_.erase(references_.begin());
    }
    references_.push_back(std::move(pattern));
}

std::size_t TemporalPatternMatcher::pattern_count() const noexcept {
    return references_.size();
}

float TemporalPatternMatcher::cosine_similarity(
    const TemporalPattern& left,
    const TemporalPattern& right) {
    if (left.spike_offsets.empty() || right.spike_offsets.empty()) {
        return 0.0F;
    }

    const auto count = std::min(left.spike_offsets.size(), right.spike_offsets.size());
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const auto l = static_cast<double>(left.spike_offsets[i]);
        const auto r = static_cast<double>(right.spike_offsets[i]);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }

    if (left_norm == 0.0 || right_norm == 0.0) {
        return 0.0F;
    }

    return static_cast<float>(dot / (std::sqrt(left_norm) * std::sqrt(right_norm)));
}

} // namespace snncuda::snn

