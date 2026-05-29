#pragma once

#include "snncuda/backends/Backend.h"
#include "snncuda/core/Ids.h"
#include "snncuda/declarative/Connectome.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace snncuda::backends {

struct CudaDebugMetrics {
    std::uint64_t ticks_processed{0};
    std::uint64_t scheduled_event_requests{0};
    std::uint64_t scheduled_events{0};
    std::uint64_t processed_events{0};
    std::uint64_t dropped_events{0};
    std::uint64_t fired_appends{0};
    std::uint64_t fired_overflow{0};
    std::uint64_t lock_spin_iterations{0};
    std::uint64_t lock_timeouts{0};
    std::uint64_t stdp_updates{0};
    std::uint64_t stdp_ltp{0};
    std::uint64_t stdp_ltd{0};
    std::uint64_t receptor_ampa_events{0};
    std::uint64_t receptor_nmda_events{0};
    std::uint64_t receptor_gaba_a_events{0};
    std::uint64_t receptor_gaba_b_events{0};
    std::uint64_t dendritic_integrations{0};
    std::uint64_t post_plasticity_scans{0};
    std::uint64_t temporal_observations{0};
    std::uint64_t temporal_patterns_learned{0};
    std::uint64_t temporal_matches{0};
    std::uint64_t max_scheduled_events_per_tick{0};
    std::uint64_t max_fired_neurons_per_tick{0};
};

struct CudaPropagationResult {
    bool executed{false};
    std::uint64_t delivered_spikes{0};
    std::uint64_t fired_spikes{0};
    double elapsed_seconds{0.0};
    CudaDebugMetrics debug;
    std::vector<float> final_synapse_weights;
    std::vector<float> final_membrane_potentials;
    std::vector<std::uint64_t> final_neuron_spike_counts;
    std::vector<std::uint64_t> final_temporal_match_counts;
    std::vector<std::uint32_t> final_temporal_learned_pattern_counts;
    std::vector<std::uint64_t> final_synapse_code_match_counts;
    std::vector<std::uint32_t> final_synapse_code_learned_pattern_counts;
};

struct CudaWeightedInput {
    core::NeuronId neuron;
    float weight{1.0F};
    std::uint32_t tick{0};
};

struct CudaInferenceSample {
    std::vector<CudaWeightedInput> inputs;
};

struct CudaInferenceOptions {
    std::uint32_t max_steps{8};
    bool reset_state_between_samples{true};
    std::vector<core::NeuronId> readout_neurons;
};

struct CudaInferenceSampleResult {
    std::vector<std::uint64_t> readout_spike_counts;
    std::vector<float> readout_scores;
    std::uint64_t delivered_spikes{0};
    std::uint64_t fired_spikes{0};
    CudaDebugMetrics debug;
};

struct CudaInferenceBatchResult {
    bool executed{false};
    double elapsed_seconds{0.0};
    std::vector<CudaInferenceSampleResult> samples;
};

class CudaInferenceSession {
public:
    explicit CudaInferenceSession(const declarative::Connectome& connectome);
    ~CudaInferenceSession();

    CudaInferenceSession(const CudaInferenceSession&) = delete;
    CudaInferenceSession& operator=(const CudaInferenceSession&) = delete;
    CudaInferenceSession(CudaInferenceSession&&) noexcept;
    CudaInferenceSession& operator=(CudaInferenceSession&&) noexcept;

    [[nodiscard]] bool available() const noexcept;

    [[nodiscard]] CudaInferenceBatchResult run_batch(
        const std::vector<CudaInferenceSample>& samples,
        const CudaInferenceOptions& options);

    [[nodiscard]] CudaInferenceBatchResult run_feedforward_batch(
        const std::vector<CudaInferenceSample>& samples,
        const CudaInferenceOptions& options);

    void reset_state();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class CudaBackend final : public Backend {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] std::string availability_status() const;

    [[nodiscard]] CudaPropagationResult run_resident_propagation(
        const declarative::Connectome& connectome,
        const std::vector<core::NeuronId>& initially_active,
        std::uint32_t max_steps) const;
};

} // namespace snncuda::backends
