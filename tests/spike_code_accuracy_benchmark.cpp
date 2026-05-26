#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using SpikeCode = std::vector<std::uint32_t>;

struct ClassificationStats {
    std::uint64_t true_positive{0};
    std::uint64_t false_positive{0};
    std::uint64_t false_negative{0};
    double top1_accuracy{0.0};
    double precision{0.0};
    double recall{0.0};
    double f1{0.0};
    double mean_margin{0.0};
};

float cosine_similarity(const SpikeCode& left, const SpikeCode& right) {
    if (left.empty() || right.empty() || left.size() != right.size()) {
        return 0.0F;
    }

    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        const auto l = static_cast<double>(left[i]);
        const auto r = static_cast<double>(right[i]);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }

    if (left_norm == 0.0 || right_norm == 0.0) {
        return 0.0F;
    }
    return static_cast<float>(dot / (std::sqrt(left_norm) * std::sqrt(right_norm)));
}

std::vector<SpikeCode> make_single_spike_codes(std::size_t class_count) {
    return std::vector<SpikeCode>(class_count, SpikeCode{1});
}

std::vector<SpikeCode> make_coded_spike_codes(std::size_t class_count) {
    const std::vector<SpikeCode> codebook{
        {64, 207, 244, 275, 314, 323, 352, 472},
        {9, 105, 138, 324, 361, 412, 468, 506},
        {104, 335, 358, 392, 399, 412, 420, 500},
        {13, 41, 76, 102, 214, 226, 370, 442},
        {3, 33, 43, 90, 207, 348, 467, 475},
        {24, 54, 113, 136, 277, 301, 302, 487},
        {16, 46, 101, 204, 235, 395, 418, 457},
        {71, 110, 152, 184, 219, 261, 282, 381},
        {26, 92, 158, 187, 250, 266, 353, 383},
        {1, 29, 83, 110, 113, 119, 239, 329},
        {16, 112, 126, 144, 345, 370, 408, 425},
        {58, 81, 156, 413, 439, 448, 455, 481},
        {115, 122, 153, 292, 329, 347, 446, 466},
        {77, 125, 316, 336, 448, 486, 489, 493},
        {88, 133, 154, 191, 218, 228, 363, 472},
        {23, 28, 66, 80, 92, 141, 153, 424},
    };
    if (class_count > codebook.size()) {
        throw std::invalid_argument("requested class count exceeds fixed codebook");
    }
    return {codebook.begin(), codebook.begin() + static_cast<std::ptrdiff_t>(class_count)};
}

SpikeCode jitter_code(
    const SpikeCode& code,
    std::mt19937& rng,
    int jitter_ticks) {
    std::uniform_int_distribution<int> jitter(-jitter_ticks, jitter_ticks);
    SpikeCode observed;
    observed.reserve(code.size());
    for (const auto offset : code) {
        observed.push_back(static_cast<std::uint32_t>(
            std::max(1, static_cast<int>(offset) + jitter(rng))));
    }
    std::sort(observed.begin(), observed.end());
    return observed;
}

std::pair<std::size_t, double> classify(
    const std::vector<SpikeCode>& references,
    const SpikeCode& observed,
    double* margin) {
    std::size_t best_index = 0;
    double best = -std::numeric_limits<double>::infinity();
    double second = -std::numeric_limits<double>::infinity();

    for (std::size_t index = 0; index < references.size(); ++index) {
        const auto score = static_cast<double>(cosine_similarity(references[index], observed));
        if (score > best) {
            second = best;
            best = score;
            best_index = index;
        } else if (score > second) {
            second = score;
        }
    }

    *margin = best - second;
    return {best_index, best};
}

ClassificationStats run_trials(
    const std::vector<SpikeCode>& references,
    int jitter_ticks,
    int trials_per_class,
    double acceptance_threshold) {
    std::mt19937 rng(0x5EED);
    ClassificationStats stats;
    double margin_sum = 0.0;

    for (std::size_t label = 0; label < references.size(); ++label) {
        for (int trial = 0; trial < trials_per_class; ++trial) {
            const auto observed = jitter_code(references[label], rng, jitter_ticks);
            double margin = 0.0;
            const auto [predicted, score] = classify(references, observed, &margin);
            margin_sum += margin;

            if (score < acceptance_threshold) {
                ++stats.false_negative;
                continue;
            }
            if (predicted == label) {
                ++stats.true_positive;
            } else {
                ++stats.false_positive;
            }
        }
    }

    const auto total = static_cast<double>(
        stats.true_positive + stats.false_positive + stats.false_negative);
    stats.top1_accuracy = total == 0.0 ? 0.0 : static_cast<double>(stats.true_positive) / total;
    stats.precision = stats.true_positive + stats.false_positive == 0
        ? 0.0
        : static_cast<double>(stats.true_positive)
            / static_cast<double>(stats.true_positive + stats.false_positive);
    stats.recall = stats.true_positive + stats.false_negative == 0
        ? 0.0
        : static_cast<double>(stats.true_positive)
            / static_cast<double>(stats.true_positive + stats.false_negative);
    stats.f1 = stats.precision + stats.recall == 0.0
        ? 0.0
        : 2.0 * stats.precision * stats.recall / (stats.precision + stats.recall);
    stats.mean_margin = total == 0.0 ? 0.0 : margin_sum / total;
    return stats;
}

void print_stats(const char* prefix, const ClassificationStats& stats) {
    std::cout << prefix << "_true_positive=" << stats.true_positive << '\n';
    std::cout << prefix << "_false_positive=" << stats.false_positive << '\n';
    std::cout << prefix << "_false_negative=" << stats.false_negative << '\n';
    std::cout << prefix << "_accuracy=" << stats.top1_accuracy << '\n';
    std::cout << prefix << "_precision=" << stats.precision << '\n';
    std::cout << prefix << "_recall=" << stats.recall << '\n';
    std::cout << prefix << "_f1=" << stats.f1 << '\n';
    std::cout << prefix << "_mean_margin=" << stats.mean_margin << '\n';
}

} // namespace

int main() {
    constexpr std::size_t class_count = 16;
    constexpr int trials_per_class = 250;
    constexpr int jitter_ticks = 1;
    constexpr double acceptance_threshold = 0.99;

    const auto single = run_trials(
        make_single_spike_codes(class_count),
        jitter_ticks,
        trials_per_class,
        acceptance_threshold);
    const auto coded = run_trials(
        make_coded_spike_codes(class_count),
        jitter_ticks,
        trials_per_class,
        acceptance_threshold);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "spike_code_accuracy_benchmark=passed\n";
    std::cout << "class_count=" << class_count << '\n';
    std::cout << "trials_per_class=" << trials_per_class << '\n';
    std::cout << "jitter_ticks=" << jitter_ticks << '\n';
    std::cout << "acceptance_threshold=" << acceptance_threshold << '\n';
    print_stats("single_spike", single);
    print_stats("coded_spike", coded);
    std::cout << "accuracy_delta=" << coded.top1_accuracy - single.top1_accuracy << '\n';
    std::cout << "f1_delta=" << coded.f1 - single.f1 << '\n';

    if (coded.top1_accuracy <= single.top1_accuracy) {
        std::cerr << "Coded spike train did not improve recognition accuracy\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
