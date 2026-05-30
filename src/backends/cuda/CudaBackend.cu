#include "snncuda/backends/CudaBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime_api.h>

namespace {

constexpr unsigned long long invalid_tick = std::numeric_limits<unsigned long long>::max();
constexpr unsigned long long max_lock_spins = 10000000ULL;
constexpr int temporal_capacity = 8;
constexpr int spike_code_capacity = 8;

enum CudaMetricIndex : int {
    metric_scheduled_event_requests = 0,
    metric_scheduled_events,
    metric_processed_events,
    metric_dropped_events,
    metric_fired_appends,
    metric_fired_overflow,
    metric_lock_spin_iterations,
    metric_lock_timeouts,
    metric_stdp_updates,
    metric_stdp_ltp,
    metric_stdp_ltd,
    metric_receptor_ampa_events,
    metric_receptor_nmda_events,
    metric_receptor_gaba_a_events,
    metric_receptor_gaba_b_events,
    metric_dendritic_integrations,
    metric_post_plasticity_scans,
    metric_temporal_observations,
    metric_temporal_patterns_learned,
    metric_temporal_matches,
    metric_count,
};

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
T* copy_to_device(const std::vector<T>& values) {
    if (values.empty()) {
        return nullptr;
    }

    T* device = nullptr;
    check_cuda(cudaMalloc(&device, values.size() * sizeof(T)), "cudaMalloc");
    check_cuda(
        cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy host-to-device");
    return device;
}

__device__ float device_receptor_gain(int receptor) {
    switch (receptor) {
    case 0:
        return 1.0F;
    case 1:
        return 0.55F;
    case 2:
        return 1.0F;
    case 3:
        return 0.65F;
    default:
        return 1.0F;
    }
}

__device__ float device_receptor_sign(int receptor) {
    return receptor == 2 || receptor == 3 ? -1.0F : 1.0F;
}

__device__ float device_receptor_tau(int receptor) {
    switch (receptor) {
    case 0:
        return 5.0F;
    case 1:
        return 100.0F;
    case 2:
        return 10.0F;
    case 3:
        return 150.0F;
    default:
        return 5.0F;
    }
}

__device__ float device_clamp(float value, float minimum, float maximum) {
    return fminf(fmaxf(value, minimum), maximum);
}

__device__ float device_stdp_delta(unsigned long long pre_tick, unsigned long long post_tick) {
    const auto dt = static_cast<float>(
        static_cast<long long>(post_tick) - static_cast<long long>(pre_tick));
    if (dt > 0.0F) {
        return 0.01F * expf(-dt / 20.0F);
    }
    if (dt < 0.0F) {
        return -0.012F * expf(dt / 20.0F);
    }
    return 0.0F;
}

__device__ void device_apply_stdp(
    int edge,
    unsigned long long pre_tick,
    unsigned long long post_tick,
    float* weights,
    const float* max_weights,
    float* last_weight_delta,
    unsigned long long* plasticity_updates,
    unsigned long long* metrics) {
    const auto before = weights[edge];
    const auto delta = device_stdp_delta(pre_tick, post_tick);
    const auto updated = device_clamp(
        before + delta,
        0.0F,
        fminf(2.0F, max_weights[edge]));
    weights[edge] = updated;
    last_weight_delta[edge] = updated - before;
    if (last_weight_delta[edge] != 0.0F) {
        ++plasticity_updates[edge];
        atomicAdd(&metrics[metric_stdp_updates], 1ULL);
        atomicAdd(
            &metrics[delta > 0.0F ? metric_stdp_ltp : metric_stdp_ltd],
            1ULL);
    }
}

__device__ void device_decay_compartment(
    float& current,
    unsigned long long& last_tick,
    float tau,
    unsigned long long tick) {
    if (tick <= last_tick || current == 0.0F) {
        last_tick = tick;
        return;
    }
    const auto elapsed = static_cast<float>(tick - last_tick);
    current *= expf(-elapsed / tau);
    last_tick = tick;
}

__device__ void device_decay_neuron(
    int neuron,
    unsigned long long tick,
    float* soma,
    float* basal,
    float* apical,
    float* inhibitory,
    unsigned long long* soma_ticks,
    unsigned long long* basal_ticks,
    unsigned long long* apical_ticks,
    unsigned long long* inhibitory_ticks,
    const float* soma_tau,
    const float* basal_tau,
    const float* apical_tau,
    const float* inhibitory_tau) {
    device_decay_compartment(soma[neuron], soma_ticks[neuron], soma_tau[neuron], tick);
    device_decay_compartment(basal[neuron], basal_ticks[neuron], basal_tau[neuron], tick);
    device_decay_compartment(apical[neuron], apical_ticks[neuron], apical_tau[neuron], tick);
    device_decay_compartment(inhibitory[neuron], inhibitory_ticks[neuron], inhibitory_tau[neuron], tick);
}

__device__ float device_membrane(
    int neuron,
    const float* soma,
    const float* basal,
    const float* apical,
    const float* inhibitory) {
    return soma[neuron] + basal[neuron] + apical[neuron] + inhibitory[neuron];
}

__device__ void device_add_current(
    int neuron,
    int compartment,
    float current,
    float tau,
    unsigned long long tick,
    float* soma,
    float* basal,
    float* apical,
    float* inhibitory,
    unsigned long long* soma_ticks,
    unsigned long long* basal_ticks,
    unsigned long long* apical_ticks,
    unsigned long long* inhibitory_ticks,
    float* soma_tau,
    float* basal_tau,
    float* apical_tau,
    float* inhibitory_tau) {
    switch (compartment) {
    case 0:
        soma[neuron] += current;
        soma_ticks[neuron] = tick;
        soma_tau[neuron] = tau;
        return;
    case 2:
        apical[neuron] += current;
        apical_ticks[neuron] = tick;
        apical_tau[neuron] = tau;
        return;
    case 3:
        inhibitory[neuron] += current;
        inhibitory_ticks[neuron] = tick;
        inhibitory_tau[neuron] = tau;
        return;
    default:
        basal[neuron] += current;
        basal_ticks[neuron] = tick;
        basal_tau[neuron] = tau;
        return;
    }
}

__device__ void device_reset_neuron(
    int neuron,
    unsigned long long tick,
    float* membrane,
    float* soma,
    float* basal,
    float* apical,
    float* inhibitory,
    unsigned long long* soma_ticks,
    unsigned long long* basal_ticks,
    unsigned long long* apical_ticks,
    unsigned long long* inhibitory_ticks,
    float* soma_tau,
    float* basal_tau,
    float* apical_tau,
    float* inhibitory_tau) {
    membrane[neuron] = 0.0F;
    soma[neuron] = 0.0F;
    basal[neuron] = 0.0F;
    apical[neuron] = 0.0F;
    inhibitory[neuron] = 0.0F;
    soma_ticks[neuron] = tick;
    basal_ticks[neuron] = tick;
    apical_ticks[neuron] = tick;
    inhibitory_ticks[neuron] = tick;
    soma_tau[neuron] = 5.0F;
    basal_tau[neuron] = 5.0F;
    apical_tau[neuron] = 100.0F;
    inhibitory_tau[neuron] = 10.0F;
}

__device__ float device_temporal_similarity(
    int neuron,
    const unsigned int* offsets,
    const unsigned int* reference,
    int count) {
    double dot = 0.0;
    double observed_norm = 0.0;
    double reference_norm = 0.0;
    const auto base = neuron * temporal_capacity;
    for (int index = 0; index < count; ++index) {
        const auto observed = static_cast<double>(offsets[base + index]);
        const auto expected = static_cast<double>(reference[base + index]);
        dot += observed * expected;
        observed_norm += observed * observed;
        reference_norm += expected * expected;
    }
    if (observed_norm == 0.0 || reference_norm == 0.0) {
        return 0.0F;
    }
    return static_cast<float>(dot / (sqrt(observed_norm) * sqrt(reference_norm)));
}

__device__ void device_update_temporal_pattern(
    int neuron,
    unsigned long long tick,
    const unsigned long long* window_ticks,
    const float* similarity_thresholds,
    const unsigned int* max_patterns,
    unsigned long long* window_start,
    unsigned int* observed_offsets,
    unsigned int* observed_counts,
    unsigned int* reference_offsets,
    unsigned int* reference_counts,
    unsigned int* learned_counts,
    unsigned long long* match_counts,
    int* last_match,
    unsigned long long* metrics) {
    atomicAdd(&metrics[metric_temporal_observations], 1ULL);

    auto& start = window_start[neuron];
    auto& observed_count = observed_counts[neuron];
    if (observed_count == 0 || tick - start > window_ticks[neuron]) {
        start = tick;
        observed_count = 0;
    }

    if (observed_count < temporal_capacity) {
        observed_offsets[neuron * temporal_capacity + observed_count] = static_cast<unsigned int>(tick - start);
        ++observed_count;
    } else {
        for (int index = 1; index < temporal_capacity; ++index) {
            observed_offsets[neuron * temporal_capacity + index - 1] =
                observed_offsets[neuron * temporal_capacity + index];
        }
        observed_offsets[neuron * temporal_capacity + temporal_capacity - 1] =
            static_cast<unsigned int>(tick - start);
        observed_count = temporal_capacity;
    }

    last_match[neuron] = 0;
    if (observed_count < 2) {
        return;
    }

    if (learned_counts[neuron] > 0 && reference_counts[neuron] == observed_count) {
        const auto similarity = device_temporal_similarity(
            neuron,
            observed_offsets,
            reference_offsets,
            static_cast<int>(observed_count));
        if (similarity >= similarity_thresholds[neuron]) {
            last_match[neuron] = 1;
            ++match_counts[neuron];
            atomicAdd(&metrics[metric_temporal_matches], 1ULL);
            return;
        }
    }

    if (max_patterns[neuron] == 0) {
        return;
    }
    const auto base = neuron * temporal_capacity;
    for (int index = 0; index < static_cast<int>(observed_count); ++index) {
        reference_offsets[base + index] = observed_offsets[base + index];
    }
    reference_counts[neuron] = observed_count;
    if (learned_counts[neuron] == 0) {
        atomicAdd(&metrics[metric_temporal_patterns_learned], 1ULL);
    }
    learned_counts[neuron] = 1;
}

__device__ void device_update_synapse_code_pattern(
    int edge,
    unsigned long long tick,
    unsigned int expected_count,
    const unsigned long long* window_ticks,
    const float* similarity_thresholds,
    const unsigned int* max_patterns,
    unsigned long long* window_start,
    unsigned int* observed_offsets,
    unsigned int* observed_counts,
    unsigned int* reference_offsets,
    unsigned int* reference_counts,
    unsigned int* learned_counts,
    unsigned long long* match_counts,
    int* last_match,
    unsigned long long* metrics) {
    atomicAdd(&metrics[metric_temporal_observations], 1ULL);

    expected_count = min(max(expected_count, 2U), static_cast<unsigned int>(temporal_capacity));
    auto& start = window_start[edge];
    auto& observed_count = observed_counts[edge];
    if (observed_count == 0 || tick - start > window_ticks[edge]) {
        start = tick;
        observed_count = 0;
    }

    if (observed_count < temporal_capacity) {
        observed_offsets[edge * temporal_capacity + observed_count] = static_cast<unsigned int>(tick - start);
        ++observed_count;
    } else {
        for (int index = 1; index < temporal_capacity; ++index) {
            observed_offsets[edge * temporal_capacity + index - 1] =
                observed_offsets[edge * temporal_capacity + index];
        }
        observed_offsets[edge * temporal_capacity + temporal_capacity - 1] =
            static_cast<unsigned int>(tick - start);
        observed_count = temporal_capacity;
    }

    last_match[edge] = 0;
    if (observed_count < expected_count) {
        return;
    }
    if (observed_count > expected_count) {
        const auto base = edge * temporal_capacity;
        const auto remove_count = static_cast<int>(observed_count - expected_count);
        for (int index = 0; index < static_cast<int>(expected_count); ++index) {
            observed_offsets[base + index] = observed_offsets[base + index + remove_count];
        }
        observed_count = expected_count;
    }

    if (learned_counts[edge] > 0 && reference_counts[edge] == observed_count) {
        const auto similarity = device_temporal_similarity(
            edge,
            observed_offsets,
            reference_offsets,
            static_cast<int>(observed_count));
        if (similarity >= similarity_thresholds[edge]) {
            last_match[edge] = 1;
            ++match_counts[edge];
            atomicAdd(&metrics[metric_temporal_matches], 1ULL);
            return;
        }
    }

    if (max_patterns[edge] == 0) {
        return;
    }
    const auto base = edge * temporal_capacity;
    for (int index = 0; index < static_cast<int>(observed_count); ++index) {
        reference_offsets[base + index] = observed_offsets[base + index];
    }
    reference_counts[edge] = observed_count;
    if (learned_counts[edge] == 0) {
        atomicAdd(&metrics[metric_temporal_patterns_learned], 1ULL);
    }
    learned_counts[edge] = 1;
}

__global__ void initialize_initial_fired_kernel(
    const int* initial_fired,
    int initial_count,
    const int* incoming_offsets,
    const int* incoming_edges,
    unsigned long long* spike_counts,
    unsigned long long* last_fired_ticks,
    unsigned long long* synapse_last_post_tick) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= initial_count) {
        return;
    }

    const auto neuron = initial_fired[index];
    ++spike_counts[neuron];
    last_fired_ticks[neuron] = 0;
    for (auto edge = incoming_offsets[neuron]; edge < incoming_offsets[neuron + 1]; ++edge) {
        synapse_last_post_tick[incoming_edges[edge]] = 0;
    }
}

__global__ void apply_external_inputs_kernel(
    const int* input_neurons,
    const float* input_weights,
    int input_count,
    unsigned long long tick,
    const int* incoming_offsets,
    const int* incoming_edges,
    const float* thresholds,
    float* membrane,
    float* soma,
    float* basal,
    float* apical,
    float* inhibitory,
    unsigned long long* soma_ticks,
    unsigned long long* basal_ticks,
    unsigned long long* apical_ticks,
    unsigned long long* inhibitory_ticks,
    float* soma_tau,
    float* basal_tau,
    float* apical_tau,
    float* inhibitory_tau,
    unsigned long long* spike_counts,
    unsigned long long* last_fired_ticks,
    unsigned long long* synapse_last_post_tick,
    int* neuron_locks,
    int* fired_flags,
    int* fired,
    int* fired_count,
    int fired_capacity) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= input_count) {
        return;
    }

    const auto neuron = input_neurons[index];
    unsigned long long spins = 0;
    while (atomicCAS(&neuron_locks[neuron], 0, 1) != 0) {
        ++spins;
        if (spins >= max_lock_spins) {
            return;
        }
    }

    device_add_current(
        neuron,
        0,
        input_weights[index],
        5.0F,
        tick,
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);
    membrane[neuron] = device_membrane(neuron, soma, basal, apical, inhibitory);
    if (membrane[neuron] < thresholds[neuron] || atomicExch(&fired_flags[neuron], 1) != 0) {
        atomicExch(&neuron_locks[neuron], 0);
        return;
    }

    device_reset_neuron(
        neuron,
        tick,
        membrane,
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);
    ++spike_counts[neuron];
    last_fired_ticks[neuron] = tick;
    const auto fired_index = atomicAdd(fired_count, 1);
    if (fired_index < fired_capacity) {
        fired[fired_index] = neuron;
    }
    for (auto edge = incoming_offsets[neuron]; edge < incoming_offsets[neuron + 1]; ++edge) {
        synapse_last_post_tick[incoming_edges[edge]] = tick;
    }
    atomicExch(&neuron_locks[neuron], 0);
}

__global__ void mark_fired_flags_kernel(
    const int* fired,
    int fired_count,
    int* fired_flags) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < fired_count) {
        fired_flags[fired[index]] = 1;
    }
}

__global__ void expand_fired_kernel(
    const int* fired,
    int fired_count,
    const int* outgoing_offsets,
    const int* outgoing_edges,
    const unsigned int* synapse_delays,
    const unsigned int* synapse_code_offsets,
    const unsigned int* synapse_code_counts,
    int* scheduled_edges,
    int* scheduled_counts,
    int event_capacity,
    unsigned long long tick,
    unsigned long long max_tick,
    unsigned long long* metrics) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= fired_count) {
        return;
    }

    const auto source = fired[index];
    for (auto edge = outgoing_offsets[source]; edge < outgoing_offsets[source + 1]; ++edge) {
        const auto synapse = outgoing_edges[edge];
        const auto code_count = max(1U, synapse_code_counts[synapse]);
        for (unsigned int code = 0; code < code_count && code < spike_code_capacity; ++code) {
            atomicAdd(&metrics[metric_scheduled_event_requests], 1ULL);
            const auto offset = synapse_code_offsets[synapse * spike_code_capacity + code];
            const auto delivery_tick = tick + synapse_delays[synapse] + offset;
            if (delivery_tick > max_tick) {
                continue;
            }
            const auto count = atomicAdd(&scheduled_counts[delivery_tick], 1);
            if (count < event_capacity) {
                scheduled_edges[delivery_tick * event_capacity + count] = synapse;
                atomicAdd(&metrics[metric_scheduled_events], 1ULL);
            } else {
                atomicAdd(&metrics[metric_dropped_events], 1ULL);
            }
        }
    }
}

__global__ void process_synapse_events_kernel(
    const int* scheduled_edges,
    int event_count,
    int* fired,
    int* fired_count,
    int fired_capacity,
    unsigned long long tick,
    const int* edge_targets,
    const unsigned int* edge_delays,
    const unsigned int* edge_code_counts,
    const int* edge_compartments,
    const int* edge_receptors,
    const int* edge_plasticity,
    const int* incoming_offsets,
    const int* incoming_edges,
    float* edge_weights,
    const float* edge_max_weights,
    unsigned long long* edge_last_pre_tick,
    unsigned long long* edge_last_post_tick,
    unsigned long long* edge_pre_counts,
    unsigned long long* edge_plasticity_updates,
    float* edge_last_weight_delta,
    const float* thresholds,
    float* membrane,
    float* soma,
    float* basal,
    float* apical,
    float* inhibitory,
    unsigned long long* soma_ticks,
    unsigned long long* basal_ticks,
    unsigned long long* apical_ticks,
    unsigned long long* inhibitory_ticks,
    float* soma_tau,
    float* basal_tau,
    float* apical_tau,
    float* inhibitory_tau,
    unsigned long long* spike_counts,
    unsigned long long* last_fired_ticks,
    int* neuron_locks,
    int* fired_flags,
    const unsigned long long* temporal_window_ticks,
    const float* temporal_similarity_thresholds,
    const unsigned int* temporal_max_patterns,
    unsigned long long* temporal_window_start,
    unsigned int* temporal_observed_offsets,
    unsigned int* temporal_observed_counts,
    unsigned int* temporal_reference_offsets,
    unsigned int* temporal_reference_counts,
    unsigned int* temporal_learned_counts,
    unsigned long long* temporal_match_counts,
    int* temporal_last_match,
    const unsigned long long* edge_code_window_ticks,
    const float* edge_code_similarity_thresholds,
    const unsigned int* edge_code_max_patterns,
    unsigned long long* edge_code_window_start,
    unsigned int* edge_code_observed_offsets,
    unsigned int* edge_code_observed_counts,
    unsigned int* edge_code_reference_offsets,
    unsigned int* edge_code_reference_counts,
    unsigned int* edge_code_learned_counts,
    unsigned long long* edge_code_match_counts,
    int* edge_code_last_match,
    unsigned long long* metrics) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= event_count) {
        return;
    }
    atomicAdd(&metrics[metric_processed_events], 1ULL);

    const auto edge = scheduled_edges[index];
    const auto target = edge_targets[edge];

    unsigned long long spins = 0;
    while (atomicCAS(&neuron_locks[target], 0, 1) != 0) {
        ++spins;
        if (spins >= max_lock_spins) {
            atomicAdd(&metrics[metric_lock_spin_iterations], spins);
            atomicAdd(&metrics[metric_lock_timeouts], 1ULL);
            atomicAdd(&metrics[metric_dropped_events], 1ULL);
            return;
        }
    }
    if (spins > 0) {
        atomicAdd(&metrics[metric_lock_spin_iterations], spins);
    }

    const auto pre_tick = tick >= edge_delays[edge] ? tick - edge_delays[edge] : tick;
    if (edge_plasticity[edge] != 0 && edge_last_post_tick[edge] != invalid_tick) {
        device_apply_stdp(
            edge,
            pre_tick,
            edge_last_post_tick[edge],
            edge_weights,
            edge_max_weights,
            edge_last_weight_delta,
            edge_plasticity_updates,
            metrics);
    }

    edge_last_pre_tick[edge] = pre_tick;
    ++edge_pre_counts[edge];

    device_decay_neuron(
        target,
        tick,
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);

    const auto receptor = edge_receptors[edge];
    switch (receptor) {
    case 1:
        atomicAdd(&metrics[metric_receptor_nmda_events], 1ULL);
        break;
    case 2:
        atomicAdd(&metrics[metric_receptor_gaba_a_events], 1ULL);
        break;
    case 3:
        atomicAdd(&metrics[metric_receptor_gaba_b_events], 1ULL);
        break;
    default:
        atomicAdd(&metrics[metric_receptor_ampa_events], 1ULL);
        break;
    }
    const auto current = edge_weights[edge] * device_receptor_gain(receptor) * device_receptor_sign(receptor);
    device_add_current(
        target,
        edge_compartments[edge],
        current,
        device_receptor_tau(receptor),
        tick,
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);
    atomicAdd(&metrics[metric_dendritic_integrations], 1ULL);
    device_update_temporal_pattern(
        target,
        tick,
        temporal_window_ticks,
        temporal_similarity_thresholds,
        temporal_max_patterns,
        temporal_window_start,
        temporal_observed_offsets,
        temporal_observed_counts,
        temporal_reference_offsets,
        temporal_reference_counts,
        temporal_learned_counts,
        temporal_match_counts,
        temporal_last_match,
        metrics);
    device_update_synapse_code_pattern(
        edge,
        tick,
        edge_code_counts[edge],
        edge_code_window_ticks,
        edge_code_similarity_thresholds,
        edge_code_max_patterns,
        edge_code_window_start,
        edge_code_observed_offsets,
        edge_code_observed_counts,
        edge_code_reference_offsets,
        edge_code_reference_counts,
        edge_code_learned_counts,
        edge_code_match_counts,
        edge_code_last_match,
        metrics);

    membrane[target] = device_membrane(target, soma, basal, apical, inhibitory);

    if (membrane[target] >= thresholds[target] && atomicExch(&fired_flags[target], 1) == 0) {
        device_reset_neuron(
            target,
            tick,
            membrane,
            soma,
            basal,
            apical,
            inhibitory,
            soma_ticks,
            basal_ticks,
            apical_ticks,
            inhibitory_ticks,
            soma_tau,
            basal_tau,
            apical_tau,
            inhibitory_tau);

        ++spike_counts[target];
        last_fired_ticks[target] = tick;
        const auto fired_index = atomicAdd(fired_count, 1);
        if (fired_index < fired_capacity) {
            fired[fired_index] = target;
            atomicAdd(&metrics[metric_fired_appends], 1ULL);
        } else {
            atomicAdd(&metrics[metric_fired_overflow], 1ULL);
            atomicAdd(&metrics[metric_dropped_events], 1ULL);
        }

        for (auto incoming = incoming_offsets[target]; incoming < incoming_offsets[target + 1]; ++incoming) {
            const auto incoming_edge = incoming_edges[incoming];
            atomicAdd(&metrics[metric_post_plasticity_scans], 1ULL);
            edge_last_post_tick[incoming_edge] = tick;
            if (edge_plasticity[incoming_edge] != 0
                && edge_last_pre_tick[incoming_edge] != invalid_tick) {
                device_apply_stdp(
                    incoming_edge,
                    edge_last_pre_tick[incoming_edge],
                    tick,
                    edge_weights,
                    edge_max_weights,
                    edge_last_weight_delta,
                    edge_plasticity_updates,
                    metrics);
            }
        }
    }

    atomicExch(&neuron_locks[target], 0);
}

__global__ void apply_batched_external_inputs_kernel(
    int sample_count,
    int tick_count,
    int input_capacity,
    int neuron_count,
    int edge_count,
    int fired_capacity,
    unsigned int tick,
    const int* input_neurons,
    const float* input_weights,
    const int* input_counts,
    const int* incoming_offsets,
    const int* incoming_edges,
    const float* thresholds,
    float* membrane,
    float* soma,
    float* basal,
    float* apical,
    float* inhibitory,
    unsigned long long* soma_ticks,
    unsigned long long* basal_ticks,
    unsigned long long* apical_ticks,
    unsigned long long* inhibitory_ticks,
    float* soma_tau,
    float* basal_tau,
    float* apical_tau,
    float* inhibitory_tau,
    unsigned long long* spike_counts,
    unsigned long long* last_fired_ticks,
    unsigned long long* synapse_last_post_tick,
    int* neuron_locks,
    int* fired_flags,
    int* fired_by_tick,
    int* fired_counts) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * input_capacity;
    if (global >= total) {
        return;
    }

    const auto slot = global % input_capacity;
    const auto sample = global / input_capacity;
    const auto count = input_counts[sample * tick_count + tick];
    if (slot >= count) {
        return;
    }

    const auto neuron = input_neurons[(sample * tick_count + tick) * input_capacity + slot];
    const auto neuron_base = sample * neuron_count;
    const auto edge_base = sample * edge_count;
    const auto neuron_index = neuron_base + neuron;
    unsigned long long spins = 0;
    while (atomicCAS(&neuron_locks[neuron_index], 0, 1) != 0) {
        ++spins;
        if (spins >= max_lock_spins) {
            return;
        }
    }

    device_add_current(
        neuron_index,
        0,
        input_weights[(sample * tick_count + tick) * input_capacity + slot],
        5.0F,
        static_cast<unsigned long long>(tick),
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);
    membrane[neuron_index] = device_membrane(neuron_index, soma, basal, apical, inhibitory);
    if (membrane[neuron_index] < thresholds[neuron] || atomicExch(&fired_flags[neuron_index], 1) != 0) {
        atomicExch(&neuron_locks[neuron_index], 0);
        return;
    }

    device_reset_neuron(
        neuron_index,
        static_cast<unsigned long long>(tick),
        membrane,
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);
    ++spike_counts[neuron_index];
    last_fired_ticks[neuron_index] = static_cast<unsigned long long>(tick);
    const auto fired_index = atomicAdd(&fired_counts[sample * tick_count + tick], 1);
    if (fired_index < fired_capacity) {
        fired_by_tick[(sample * tick_count + tick) * fired_capacity + fired_index] = neuron;
    }
    for (auto edge = incoming_offsets[neuron]; edge < incoming_offsets[neuron + 1]; ++edge) {
        synapse_last_post_tick[edge_base + incoming_edges[edge]] = static_cast<unsigned long long>(tick);
    }
    atomicExch(&neuron_locks[neuron_index], 0);
}

__global__ void expand_batched_fired_kernel(
    int sample_count,
    int tick_count,
    int fired_capacity,
    int event_capacity,
    unsigned int tick,
    unsigned int max_tick,
    const int* fired_by_tick,
    const int* fired_counts,
    const int* outgoing_offsets,
    const int* outgoing_edges,
    const unsigned int* synapse_delays,
    const unsigned int* synapse_code_offsets,
    const unsigned int* synapse_code_counts,
    int* scheduled_edges,
    int* scheduled_counts,
    unsigned long long* metrics) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * fired_capacity;
    if (global >= total) {
        return;
    }

    const auto sample = global / fired_capacity;
    const auto slot = global % fired_capacity;
    const auto fired_count = fired_counts[sample * tick_count + tick];
    if (slot >= fired_count) {
        return;
    }

    auto* sample_metrics = metrics + sample * metric_count;
    const auto source = fired_by_tick[(sample * tick_count + tick) * fired_capacity + slot];
    for (auto edge_index = outgoing_offsets[source]; edge_index < outgoing_offsets[source + 1]; ++edge_index) {
        const auto synapse = outgoing_edges[edge_index];
        const auto code_count = max(1U, synapse_code_counts[synapse]);
        for (unsigned int code = 0; code < code_count && code < spike_code_capacity; ++code) {
            atomicAdd(&sample_metrics[metric_scheduled_event_requests], 1ULL);
            const auto offset = synapse_code_offsets[synapse * spike_code_capacity + code];
            const auto delivery_tick = tick + synapse_delays[synapse] + offset;
            if (delivery_tick > max_tick) {
                continue;
            }
            const auto count = atomicAdd(&scheduled_counts[sample * tick_count + delivery_tick], 1);
            if (count < event_capacity) {
                scheduled_edges[(sample * tick_count + delivery_tick) * event_capacity + count] = synapse;
                atomicAdd(&sample_metrics[metric_scheduled_events], 1ULL);
            } else {
                atomicAdd(&sample_metrics[metric_dropped_events], 1ULL);
            }
        }
    }
}

__global__ void process_batched_synapse_events_kernel(
    int sample_count,
    int tick_count,
    int event_capacity,
    int fired_capacity,
    int neuron_count,
    int edge_count,
    unsigned int tick,
    const int* scheduled_edges,
    const int* scheduled_counts,
    int* fired_by_tick,
    int* fired_counts,
    const int* edge_targets,
    const unsigned int* edge_delays,
    const unsigned int* edge_code_counts,
    const int* edge_compartments,
    const int* edge_receptors,
    const int* edge_plasticity,
    const int* incoming_offsets,
    const int* incoming_edges,
    float* edge_weights,
    const float* edge_max_weights,
    unsigned long long* edge_last_pre_tick,
    unsigned long long* edge_last_post_tick,
    unsigned long long* edge_pre_counts,
    unsigned long long* edge_plasticity_updates,
    float* edge_last_weight_delta,
    const float* thresholds,
    float* membrane,
    float* soma,
    float* basal,
    float* apical,
    float* inhibitory,
    unsigned long long* soma_ticks,
    unsigned long long* basal_ticks,
    unsigned long long* apical_ticks,
    unsigned long long* inhibitory_ticks,
    float* soma_tau,
    float* basal_tau,
    float* apical_tau,
    float* inhibitory_tau,
    unsigned long long* spike_counts,
    unsigned long long* last_fired_ticks,
    int* neuron_locks,
    int* fired_flags,
    const unsigned long long* temporal_window_ticks,
    const float* temporal_similarity_thresholds,
    const unsigned int* temporal_max_patterns,
    unsigned long long* temporal_window_start,
    unsigned int* temporal_observed_offsets,
    unsigned int* temporal_observed_counts,
    unsigned int* temporal_reference_offsets,
    unsigned int* temporal_reference_counts,
    unsigned int* temporal_learned_counts,
    unsigned long long* temporal_match_counts,
    int* temporal_last_match,
    const unsigned long long* edge_code_window_ticks,
    const float* edge_code_similarity_thresholds,
    const unsigned int* edge_code_max_patterns,
    unsigned long long* edge_code_window_start,
    unsigned int* edge_code_observed_offsets,
    unsigned int* edge_code_observed_counts,
    unsigned int* edge_code_reference_offsets,
    unsigned int* edge_code_reference_counts,
    unsigned int* edge_code_learned_counts,
    unsigned long long* edge_code_match_counts,
    int* edge_code_last_match,
    unsigned long long* metrics) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * event_capacity;
    if (global >= total) {
        return;
    }

    const auto sample = global / event_capacity;
    const auto slot = global % event_capacity;
    const auto event_count = scheduled_counts[sample * tick_count + tick];
    if (slot >= event_count || slot >= event_capacity) {
        return;
    }

    auto* sample_metrics = metrics + sample * metric_count;
    atomicAdd(&sample_metrics[metric_processed_events], 1ULL);

    const auto neuron_base = sample * neuron_count;
    const auto edge_base = sample * edge_count;
    const auto temporal_base = sample * neuron_count * temporal_capacity;
    const auto edge_temporal_base = sample * edge_count * temporal_capacity;
    const auto edge = scheduled_edges[(sample * tick_count + tick) * event_capacity + slot];
    const auto target = edge_targets[edge];
    const auto target_index = neuron_base + target;

    unsigned long long spins = 0;
    while (atomicCAS(&neuron_locks[target_index], 0, 1) != 0) {
        ++spins;
        if (spins >= max_lock_spins) {
            atomicAdd(&sample_metrics[metric_lock_spin_iterations], spins);
            atomicAdd(&sample_metrics[metric_lock_timeouts], 1ULL);
            atomicAdd(&sample_metrics[metric_dropped_events], 1ULL);
            return;
        }
    }
    if (spins > 0) {
        atomicAdd(&sample_metrics[metric_lock_spin_iterations], spins);
    }

    const auto pre_tick = tick >= edge_delays[edge] ? tick - edge_delays[edge] : tick;
    if (edge_plasticity[edge] != 0 && edge_last_post_tick[edge_base + edge] != invalid_tick) {
        device_apply_stdp(
            edge,
            pre_tick,
            edge_last_post_tick[edge_base + edge],
            edge_weights + edge_base,
            edge_max_weights,
            edge_last_weight_delta + edge_base,
            edge_plasticity_updates + edge_base,
            sample_metrics);
    }

    edge_last_pre_tick[edge_base + edge] = pre_tick;
    ++edge_pre_counts[edge_base + edge];

    device_decay_neuron(
        target_index,
        tick,
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);

    const auto receptor = edge_receptors[edge];
    switch (receptor) {
    case 1:
        atomicAdd(&sample_metrics[metric_receptor_nmda_events], 1ULL);
        break;
    case 2:
        atomicAdd(&sample_metrics[metric_receptor_gaba_a_events], 1ULL);
        break;
    case 3:
        atomicAdd(&sample_metrics[metric_receptor_gaba_b_events], 1ULL);
        break;
    default:
        atomicAdd(&sample_metrics[metric_receptor_ampa_events], 1ULL);
        break;
    }
    const auto current = edge_weights[edge_base + edge] * device_receptor_gain(receptor) * device_receptor_sign(receptor);
    device_add_current(
        target_index,
        edge_compartments[edge],
        current,
        device_receptor_tau(receptor),
        tick,
        soma,
        basal,
        apical,
        inhibitory,
        soma_ticks,
        basal_ticks,
        apical_ticks,
        inhibitory_ticks,
        soma_tau,
        basal_tau,
        apical_tau,
        inhibitory_tau);
    atomicAdd(&sample_metrics[metric_dendritic_integrations], 1ULL);
    device_update_temporal_pattern(
        target,
        tick,
        temporal_window_ticks,
        temporal_similarity_thresholds,
        temporal_max_patterns,
        temporal_window_start + neuron_base,
        temporal_observed_offsets + temporal_base,
        temporal_observed_counts + neuron_base,
        temporal_reference_offsets + temporal_base,
        temporal_reference_counts + neuron_base,
        temporal_learned_counts + neuron_base,
        temporal_match_counts + neuron_base,
        temporal_last_match + neuron_base,
        sample_metrics);
    device_update_synapse_code_pattern(
        edge,
        tick,
        edge_code_counts[edge],
        edge_code_window_ticks,
        edge_code_similarity_thresholds,
        edge_code_max_patterns,
        edge_code_window_start + edge_base,
        edge_code_observed_offsets + edge_temporal_base,
        edge_code_observed_counts + edge_base,
        edge_code_reference_offsets + edge_temporal_base,
        edge_code_reference_counts + edge_base,
        edge_code_learned_counts + edge_base,
        edge_code_match_counts + edge_base,
        edge_code_last_match + edge_base,
        sample_metrics);

    membrane[target_index] = device_membrane(target_index, soma, basal, apical, inhibitory);

    if (membrane[target_index] >= thresholds[target] && atomicExch(&fired_flags[target_index], 1) == 0) {
        device_reset_neuron(
            target_index,
            tick,
            membrane,
            soma,
            basal,
            apical,
            inhibitory,
            soma_ticks,
            basal_ticks,
            apical_ticks,
            inhibitory_ticks,
            soma_tau,
            basal_tau,
            apical_tau,
            inhibitory_tau);

        ++spike_counts[target_index];
        last_fired_ticks[target_index] = tick;
        const auto fired_index = atomicAdd(&fired_counts[sample * tick_count + tick], 1);
        if (fired_index < fired_capacity) {
            fired_by_tick[(sample * tick_count + tick) * fired_capacity + fired_index] = target;
            atomicAdd(&sample_metrics[metric_fired_appends], 1ULL);
        } else {
            atomicAdd(&sample_metrics[metric_fired_overflow], 1ULL);
            atomicAdd(&sample_metrics[metric_dropped_events], 1ULL);
        }

        for (auto incoming = incoming_offsets[target]; incoming < incoming_offsets[target + 1]; ++incoming) {
            const auto incoming_edge = incoming_edges[incoming];
            const auto incoming_edge_index = edge_base + incoming_edge;
            atomicAdd(&sample_metrics[metric_post_plasticity_scans], 1ULL);
            edge_last_post_tick[incoming_edge_index] = tick;
            if (edge_plasticity[incoming_edge] != 0
                && edge_last_pre_tick[incoming_edge_index] != invalid_tick) {
                device_apply_stdp(
                    incoming_edge,
                    edge_last_pre_tick[incoming_edge_index],
                    tick,
                    edge_weights + edge_base,
                    edge_max_weights,
                    edge_last_weight_delta + edge_base,
                    edge_plasticity_updates + edge_base,
                    sample_metrics);
            }
        }
    }

    atomicExch(&neuron_locks[target_index], 0);
}

__global__ void accumulate_feedforward_readout_kernel(
    int sample_count,
    int input_capacity,
    int readout_count,
    const int* input_neurons,
    const float* input_weights,
    const int* input_counts,
    const int* outgoing_offsets,
    const int* outgoing_edges,
    const int* edge_targets,
    const float* edge_weights,
    const int* dense_to_readout,
    float* readout_scores) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * input_capacity;
    if (global >= total) {
        return;
    }

    const auto sample = global / input_capacity;
    const auto slot = global % input_capacity;
    if (slot >= input_counts[sample]) {
        return;
    }

    const auto source = input_neurons[sample * input_capacity + slot];
    const auto input_weight = input_weights[sample * input_capacity + slot];
    for (auto edge_index = outgoing_offsets[source]; edge_index < outgoing_offsets[source + 1]; ++edge_index) {
        const auto edge = outgoing_edges[edge_index];
        const auto readout = dense_to_readout[edge_targets[edge]];
        if (readout >= 0 && readout < readout_count) {
            atomicAdd(&readout_scores[sample * readout_count + readout], input_weight * edge_weights[edge]);
        }
    }
}

__global__ void threshold_feedforward_readout_kernel(
    int sample_count,
    int readout_count,
    const float* readout_scores,
    const float* readout_thresholds,
    unsigned long long* readout_spike_counts) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * readout_count;
    if (global >= total) {
        return;
    }

    const auto readout = global % readout_count;
    readout_spike_counts[global] = readout_scores[global] >= readout_thresholds[readout] ? 1ULL : 0ULL;
}

__global__ void initialize_recurrent_hidden_events_kernel(
    int sample_count,
    int input_capacity,
    int hidden_count,
    int time_count,
    const int* input_neurons,
    const float* input_weights,
    const int* input_counts,
    const int* dense_to_hidden,
    float* hidden_events) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * input_capacity;
    if (global >= total) {
        return;
    }

    const auto sample = global / input_capacity;
    const auto slot = global % input_capacity;
    if (slot >= input_counts[sample]) {
        return;
    }

    const auto hidden = dense_to_hidden[input_neurons[sample * input_capacity + slot]];
    if (hidden >= 0 && hidden < hidden_count) {
        atomicAdd(
            &hidden_events[(sample * time_count) * hidden_count + hidden],
            input_weights[sample * input_capacity + slot]);
    }
}

__global__ void propagate_hidden_events_to_readouts_kernel(
    int sample_count,
    int hidden_count,
    int readout_count,
    int time_count,
    int step,
    const float* hidden_events,
    const int* hidden_dense_neurons,
    const int* outgoing_offsets,
    const int* outgoing_edges,
    const int* edge_targets,
    const unsigned int* edge_delays,
    const float* edge_weights,
    const int* dense_to_readout,
    float* readout_events) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * hidden_count;
    if (global >= total) {
        return;
    }

    const auto sample = global / hidden_count;
    const auto hidden = global % hidden_count;
    const auto hidden_score = hidden_events[(sample * time_count + step) * hidden_count + hidden];
    if (hidden_score == 0.0F) {
        return;
    }

    const auto source = hidden_dense_neurons[hidden];
    for (auto edge_index = outgoing_offsets[source]; edge_index < outgoing_offsets[source + 1]; ++edge_index) {
        const auto edge = outgoing_edges[edge_index];
        const auto readout = dense_to_readout[edge_targets[edge]];
        const auto delivery_step = step + static_cast<int>(edge_delays[edge]);
        if (readout >= 0 && readout < readout_count && delivery_step < time_count) {
            atomicAdd(
                &readout_events[(sample * time_count + delivery_step) * readout_count + readout],
                hidden_score * edge_weights[edge]);
        }
    }
}

__global__ void select_recurrent_readout_events_kernel(
    int sample_count,
    int readout_count,
    int time_count,
    int step,
    int top_k,
    int use_scores_not_spikes,
    const float* readout_events,
    const float* readout_thresholds,
    float* readout_gates,
    int* selected) {
    const auto sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count) {
        return;
    }

    const auto event_base = (sample * time_count + step) * readout_count;
    const auto gate_base = sample * readout_count;
    if (top_k <= 0) {
        for (int readout = 0; readout < readout_count; ++readout) {
            const auto score = readout_events[event_base + readout];
            if (score >= readout_thresholds[readout]) {
                readout_gates[gate_base + readout] = use_scores_not_spikes != 0 ? score : 1.0F;
                selected[gate_base + readout] = 1;
            }
        }
        return;
    }

    const auto limit = min(top_k, readout_count);
    for (int pass = 0; pass < limit; ++pass) {
        auto best = -1;
        auto best_score = -3.402823466e+38F;
        for (int readout = 0; readout < readout_count; ++readout) {
            const auto score = readout_events[event_base + readout];
            if (selected[gate_base + readout] == 0 && score > best_score) {
                best_score = score;
                best = readout;
            }
        }
        if (best >= 0) {
            selected[gate_base + best] = 1;
            readout_gates[gate_base + best] = use_scores_not_spikes != 0 ? best_score : 1.0F;
        }
    }
}

__global__ void propagate_readout_feedback_events_kernel(
    int sample_count,
    int readout_count,
    int hidden_count,
    int time_count,
    int step,
    float feedback_decay,
    const float* readout_gates,
    const int* readout_dense_neurons,
    const int* outgoing_offsets,
    const int* outgoing_edges,
    const int* edge_targets,
    const unsigned int* edge_delays,
    const float* edge_weights,
    const int* dense_to_hidden,
    float* hidden_events) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * readout_count;
    if (global >= total) {
        return;
    }

    const auto sample = global / readout_count;
    const auto readout = global % readout_count;
    const auto gate = readout_gates[sample * readout_count + readout];
    if (gate == 0.0F) {
        return;
    }

    const auto source = readout_dense_neurons[readout];
    for (auto edge_index = outgoing_offsets[source]; edge_index < outgoing_offsets[source + 1]; ++edge_index) {
        const auto edge = outgoing_edges[edge_index];
        const auto hidden = dense_to_hidden[edge_targets[edge]];
        const auto delivery_step = step + static_cast<int>(edge_delays[edge]);
        if (hidden >= 0 && hidden < hidden_count && delivery_step < time_count) {
            atomicAdd(
                &hidden_events[(sample * time_count + delivery_step) * hidden_count + hidden],
                feedback_decay * gate * edge_weights[edge]);
        }
    }
}

__global__ void apply_topk_hidden_events_kernel(
    int sample_count,
    int hidden_count,
    int time_count,
    int step,
    int top_k,
    float* hidden_events,
    int* selected) {
    const auto sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count || top_k <= 0 || top_k >= hidden_count) {
        return;
    }

    const auto base = (sample * time_count + step) * hidden_count;
    const auto selected_base = sample * hidden_count;
    const auto limit = min(top_k, hidden_count);
    for (int pass = 0; pass < limit; ++pass) {
        auto best = -1;
        auto best_score = -3.402823466e+38F;
        for (int hidden = 0; hidden < hidden_count; ++hidden) {
            const auto score = hidden_events[base + hidden];
            if (selected[selected_base + hidden] == 0 && score > best_score) {
                best_score = score;
                best = hidden;
            }
        }
        if (best >= 0) {
            selected[selected_base + best] = 1;
        }
    }
    for (int hidden = 0; hidden < hidden_count; ++hidden) {
        if (selected[selected_base + hidden] == 0) {
            hidden_events[base + hidden] = 0.0F;
        }
    }
}

__global__ void sum_recurrent_readout_events_kernel(
    int sample_count,
    int readout_count,
    int time_count,
    const float* readout_events,
    float* readout_scores) {
    const auto global = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = sample_count * readout_count;
    if (global >= total) {
        return;
    }

    const auto sample = global / readout_count;
    const auto readout = global % readout_count;
    float sum = 0.0F;
    for (int step = 0; step < time_count; ++step) {
        sum += readout_events[(sample * time_count + step) * readout_count + readout];
    }
    readout_scores[global] = sum;
}

} // namespace

namespace snncuda::backends {

namespace {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_)
        , count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() {
        reset();
    }

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)), "cudaMalloc device buffer");
        }
    }

    void upload(const std::vector<T>& values) {
        allocate(values.size());
        if (!values.empty()) {
            check_cuda(
                cudaMemcpy(ptr_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy host-to-device");
        }
    }

    void copy_from_host(const T* values, std::size_t count, std::size_t offset = 0) {
        if (count == 0) {
            return;
        }
        if (offset + count > count_) {
            throw std::runtime_error("device buffer copy exceeds allocation");
        }
        check_cuda(
            cudaMemcpy(ptr_ + offset, values, count * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device");
    }

    void zero() {
        if (ptr_ != nullptr && count_ > 0) {
            check_cuda(cudaMemset(ptr_, 0, count_ * sizeof(T)), "cudaMemset device buffer");
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    [[nodiscard]] T* get() noexcept {
        return ptr_;
    }
    [[nodiscard]] const T* get() const noexcept {
        return ptr_;
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

std::vector<unsigned long long> copy_u64_from_device(unsigned long long* device, std::size_t count, const char* context) {
    std::vector<unsigned long long> values(count, 0);
    if (count > 0) {
        check_cuda(
            cudaMemcpy(values.data(), device, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
            context);
    }
    return values;
}

CudaDebugMetrics metrics_from_device(
    unsigned long long* device_metrics,
    const CudaDebugMetrics& host_debug) {
    auto debug = host_debug;
    std::vector<unsigned long long> raw_metrics(metric_count, 0);
    check_cuda(
        cudaMemcpy(
            raw_metrics.data(),
            device_metrics,
            raw_metrics.size() * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy debug metrics");
    debug.scheduled_event_requests = raw_metrics[metric_scheduled_event_requests];
    debug.scheduled_events = raw_metrics[metric_scheduled_events];
    debug.processed_events = raw_metrics[metric_processed_events];
    debug.dropped_events += raw_metrics[metric_dropped_events];
    debug.fired_appends = raw_metrics[metric_fired_appends];
    debug.fired_overflow = raw_metrics[metric_fired_overflow];
    debug.lock_spin_iterations = raw_metrics[metric_lock_spin_iterations];
    debug.lock_timeouts = raw_metrics[metric_lock_timeouts];
    debug.stdp_updates = raw_metrics[metric_stdp_updates];
    debug.stdp_ltp = raw_metrics[metric_stdp_ltp];
    debug.stdp_ltd = raw_metrics[metric_stdp_ltd];
    debug.receptor_ampa_events = raw_metrics[metric_receptor_ampa_events];
    debug.receptor_nmda_events = raw_metrics[metric_receptor_nmda_events];
    debug.receptor_gaba_a_events = raw_metrics[metric_receptor_gaba_a_events];
    debug.receptor_gaba_b_events = raw_metrics[metric_receptor_gaba_b_events];
    debug.dendritic_integrations = raw_metrics[metric_dendritic_integrations];
    debug.post_plasticity_scans = raw_metrics[metric_post_plasticity_scans];
    debug.temporal_observations = raw_metrics[metric_temporal_observations];
    debug.temporal_patterns_learned = raw_metrics[metric_temporal_patterns_learned];
    debug.temporal_matches = raw_metrics[metric_temporal_matches];
    return debug;
}

} // namespace

class CudaInferenceSession::Impl {
public:
    explicit Impl(const declarative::Connectome& connectome)
        : available_(cuda_available())
        , neuron_count_(connectome.neurons.size()) {
        if (!available_ || connectome.neurons.empty()) {
            return;
        }
        pack(connectome);
        upload_immutable();
        reset_state();
    }

    [[nodiscard]] bool available() const noexcept {
        return available_;
    }

    void reset_state() {
        mutable_allocated_ = false;
        mutable_state_initialized_ = false;
    }

    [[nodiscard]] CudaInferenceBatchResult run_batch(
        const std::vector<CudaInferenceSample>& samples,
        const CudaInferenceOptions& options) {
        if (!available_) {
            return {};
        }

        CudaInferenceBatchResult batch;
        batch.executed = true;
        batch.samples.reserve(samples.size());
        if (options.reset_state_between_samples && !samples.empty()) {
            return run_stateless_device_batch(samples, options);
        }

        const auto start = std::chrono::steady_clock::now();
        ensure_mutable_capacity(options.max_steps, max_sample_input_count(samples));

        for (const auto& sample : samples) {
            if (options.reset_state_between_samples) {
                reset_mutable_buffers(options.max_steps);
            } else if (!mutable_state_initialized_) {
                reset_mutable_buffers(options.max_steps);
            } else {
                reset_sample_work_buffers();
            }
            batch.samples.push_back(run_one(sample, options));
        }

        const auto end = std::chrono::steady_clock::now();
        batch.elapsed_seconds = std::chrono::duration<double>(end - start).count();
        return batch;
    }

    [[nodiscard]] CudaInferenceBatchResult run_feedforward_batch(
        const std::vector<CudaInferenceSample>& samples,
        const CudaInferenceOptions& options) {
        if (!available_) {
            return {};
        }
        CudaInferenceBatchResult batch;
        batch.executed = true;
        batch.samples.resize(samples.size());
        if (samples.empty() || options.readout_neurons.empty()) {
            return batch;
        }

        const auto start = std::chrono::steady_clock::now();
        const auto sample_count = samples.size();
        const auto readout_count = options.readout_neurons.size();
        const auto input_capacity = max_sample_input_count(samples);
        ensure_feedforward_capacity(sample_count, input_capacity, readout_count);

        std::vector<int> dense_to_readout(neuron_count_, -1);
        std::vector<float> readout_thresholds(readout_count, 1.0F);
        for (std::size_t readout = 0; readout < readout_count; ++readout) {
            const auto found = dense_index_.find(options.readout_neurons[readout]);
            if (found == dense_index_.end()) {
                continue;
            }
            dense_to_readout[static_cast<std::size_t>(found->second)] = static_cast<int>(readout);
            readout_thresholds[readout] = thresholds_[static_cast<std::size_t>(found->second)];
        }

        std::vector<int> input_counts(sample_count, 0);
        std::vector<int> input_neurons(sample_count * feedforward_input_capacity_, 0);
        std::vector<float> input_weights(sample_count * feedforward_input_capacity_, 0.0F);
        for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
            std::unordered_map<int, float> merged_inputs;
            merged_inputs.reserve(samples[sample_index].inputs.size());
            for (const auto& input : samples[sample_index].inputs) {
                const auto found = dense_index_.find(input.neuron);
                if (found != dense_index_.end()) {
                    merged_inputs[found->second] += input.weight;
                }
            }
            for (const auto& [neuron, weight] : merged_inputs) {
                const auto slot = input_counts[sample_index]++;
                if (slot < static_cast<int>(feedforward_input_capacity_)) {
                    const auto offset = sample_index * feedforward_input_capacity_ + static_cast<std::size_t>(slot);
                    input_neurons[offset] = neuron;
                    input_weights[offset] = weight;
                }
            }
            batch.samples[sample_index].delivered_spikes = static_cast<std::uint64_t>(merged_inputs.size());
        }

        feedforward_dense_to_readout_.copy_from_host(dense_to_readout.data(), dense_to_readout.size());
        feedforward_readout_thresholds_.copy_from_host(readout_thresholds.data(), readout_thresholds.size());
        feedforward_input_counts_.copy_from_host(input_counts.data(), input_counts.size());
        feedforward_input_neurons_.copy_from_host(input_neurons.data(), input_neurons.size());
        feedforward_input_weights_.copy_from_host(input_weights.data(), input_weights.size());
        feedforward_scores_.zero();
        feedforward_spike_counts_.zero();

        const auto threads = 256;
        const auto input_total = sample_count * feedforward_input_capacity_;
        if (input_total > 0) {
            const auto blocks = static_cast<int>((input_total + threads - 1) / threads);
            accumulate_feedforward_readout_kernel<<<blocks, threads>>>(
                static_cast<int>(sample_count),
                static_cast<int>(feedforward_input_capacity_),
                static_cast<int>(readout_count),
                feedforward_input_neurons_.get(),
                feedforward_input_weights_.get(),
                feedforward_input_counts_.get(),
                device_offsets_.get(),
                device_outgoing_edges_.get(),
                device_targets_.get(),
                device_initial_weights_.get(),
                feedforward_dense_to_readout_.get(),
                feedforward_scores_.get());
            check_cuda(cudaGetLastError(), "accumulate_feedforward_readout_kernel launch");
        }

        const auto readout_total = sample_count * readout_count;
        if (readout_total > 0) {
            const auto blocks = static_cast<int>((readout_total + threads - 1) / threads);
            threshold_feedforward_readout_kernel<<<blocks, threads>>>(
                static_cast<int>(sample_count),
                static_cast<int>(readout_count),
                feedforward_scores_.get(),
                feedforward_readout_thresholds_.get(),
                feedforward_spike_counts_.get());
            check_cuda(cudaGetLastError(), "threshold_feedforward_readout_kernel launch");
        }
        check_cuda(cudaDeviceSynchronize(), "feedforward readout synchronize");

        std::vector<float> scores(readout_total, 0.0F);
        std::vector<unsigned long long> spike_counts(readout_total, 0);
        check_cuda(
            cudaMemcpy(scores.data(), feedforward_scores_.get(), scores.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy feedforward scores");
        check_cuda(
            cudaMemcpy(
                spike_counts.data(),
                feedforward_spike_counts_.get(),
                spike_counts.size() * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy feedforward spike counts");

        for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
            auto& sample_result = batch.samples[sample_index];
            sample_result.readout_scores.reserve(readout_count);
            sample_result.readout_spike_counts.reserve(readout_count);
            for (std::size_t readout = 0; readout < readout_count; ++readout) {
                const auto offset = sample_index * readout_count + readout;
                sample_result.readout_scores.push_back(scores[offset]);
                sample_result.readout_spike_counts.push_back(static_cast<std::uint64_t>(spike_counts[offset]));
                sample_result.fired_spikes += static_cast<std::uint64_t>(spike_counts[offset]);
            }
            sample_result.debug.processed_events = sample_result.delivered_spikes;
            sample_result.debug.dendritic_integrations = sample_result.delivered_spikes;
        }

        const auto end = std::chrono::steady_clock::now();
        batch.elapsed_seconds = std::chrono::duration<double>(end - start).count();
        return batch;
    }

    [[nodiscard]] CudaInferenceBatchResult run_recurrent_feedforward_batch(
        const std::vector<CudaInferenceSample>& samples,
        const CudaRecurrentFeedforwardOptions& options) {
        if (!available_) {
            return {};
        }
        CudaInferenceBatchResult batch;
        batch.executed = true;
        batch.samples.resize(samples.size());
        if (samples.empty() || options.readout_neurons.empty()) {
            return batch;
        }

        if (options.use_full_spike_timing) {
            CudaInferenceOptions timing_options;
            timing_options.max_steps = options.max_timing_steps == 0
                ? static_cast<std::uint32_t>(
                    static_cast<std::size_t>(max_edge_delay_ticks_)
                        * (1U + 2U * static_cast<std::size_t>(options.iterations))
                    + 1U)
                : options.max_timing_steps;
            timing_options.reset_state_between_samples = true;
            timing_options.readout_neurons = options.readout_neurons;
            return run_batch(samples, timing_options);
        }

        const auto start = std::chrono::steady_clock::now();
        const auto sample_count = samples.size();
        const auto readout_count = options.readout_neurons.size();
        const auto hidden_neurons = resolved_hidden_neurons(options);
        const auto hidden_count = hidden_neurons.size();
        if (hidden_count == 0) {
            return batch;
        }
        const auto input_capacity = max_sample_input_count(samples);
        const auto time_count = static_cast<std::size_t>(max_edge_delay_ticks_)
            * (1U + 2U * static_cast<std::size_t>(options.iterations))
            + 1U;
        ensure_recurrent_capacity(sample_count, input_capacity, hidden_count, readout_count, time_count);

        std::vector<int> dense_to_hidden(neuron_count_, -1);
        std::vector<int> hidden_dense_neurons(hidden_count, 0);
        for (std::size_t hidden = 0; hidden < hidden_count; ++hidden) {
            const auto found = dense_index_.find(hidden_neurons[hidden]);
            if (found == dense_index_.end()) {
                continue;
            }
            dense_to_hidden[static_cast<std::size_t>(found->second)] = static_cast<int>(hidden);
            hidden_dense_neurons[hidden] = found->second;
        }

        std::vector<int> dense_to_readout(neuron_count_, -1);
        std::vector<int> readout_dense_neurons(readout_count, 0);
        std::vector<float> readout_thresholds(readout_count, 1.0F);
        for (std::size_t readout = 0; readout < readout_count; ++readout) {
            const auto found = dense_index_.find(options.readout_neurons[readout]);
            if (found == dense_index_.end()) {
                continue;
            }
            dense_to_readout[static_cast<std::size_t>(found->second)] = static_cast<int>(readout);
            readout_dense_neurons[readout] = found->second;
            readout_thresholds[readout] = thresholds_[static_cast<std::size_t>(found->second)];
        }

        std::vector<int> input_counts(sample_count, 0);
        std::vector<int> input_neurons(sample_count * recurrent_input_capacity_, 0);
        std::vector<float> input_weights(sample_count * recurrent_input_capacity_, 0.0F);
        for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
            std::unordered_map<int, float> merged_inputs;
            merged_inputs.reserve(samples[sample_index].inputs.size());
            for (const auto& input : samples[sample_index].inputs) {
                const auto found = dense_index_.find(input.neuron);
                if (found != dense_index_.end()) {
                    merged_inputs[found->second] += input.weight;
                }
            }
            for (const auto& [neuron, weight] : merged_inputs) {
                const auto slot = input_counts[sample_index]++;
                if (slot < static_cast<int>(recurrent_input_capacity_)) {
                    const auto offset = sample_index * recurrent_input_capacity_ + static_cast<std::size_t>(slot);
                    input_neurons[offset] = neuron;
                    input_weights[offset] = weight;
                }
            }
            batch.samples[sample_index].delivered_spikes = static_cast<std::uint64_t>(merged_inputs.size());
        }

        recurrent_dense_to_hidden_.copy_from_host(dense_to_hidden.data(), dense_to_hidden.size());
        recurrent_hidden_dense_neurons_.copy_from_host(hidden_dense_neurons.data(), hidden_dense_neurons.size());
        recurrent_dense_to_readout_.copy_from_host(dense_to_readout.data(), dense_to_readout.size());
        recurrent_readout_dense_neurons_.copy_from_host(readout_dense_neurons.data(), readout_dense_neurons.size());
        recurrent_readout_thresholds_.copy_from_host(readout_thresholds.data(), readout_thresholds.size());
        recurrent_input_counts_.copy_from_host(input_counts.data(), input_counts.size());
        recurrent_input_neurons_.copy_from_host(input_neurons.data(), input_neurons.size());
        recurrent_input_weights_.copy_from_host(input_weights.data(), input_weights.size());
        recurrent_hidden_events_.zero();
        recurrent_readout_events_.zero();
        recurrent_readout_scores_.zero();
        recurrent_readout_gates_.zero();
        recurrent_readout_spike_counts_.zero();

        const auto threads = 256;
        const auto input_total = sample_count * recurrent_input_capacity_;
        if (input_total > 0) {
            const auto blocks = static_cast<int>((input_total + threads - 1) / threads);
            initialize_recurrent_hidden_events_kernel<<<blocks, threads>>>(
                static_cast<int>(sample_count),
                static_cast<int>(recurrent_input_capacity_),
                static_cast<int>(hidden_count),
                static_cast<int>(time_count),
                recurrent_input_neurons_.get(),
                recurrent_input_weights_.get(),
                recurrent_input_counts_.get(),
                recurrent_dense_to_hidden_.get(),
                recurrent_hidden_events_.get());
            check_cuda(cudaGetLastError(), "initialize_recurrent_hidden_events_kernel launch");
        }

        for (std::uint32_t step = 0; step < time_count; ++step) {
            recurrent_readout_gates_.zero();
            recurrent_readout_selected_.zero();
            recurrent_hidden_selected_.zero();

            if (options.top_k_hidden > 0) {
                apply_topk_hidden_events_kernel<<<
                    static_cast<int>((sample_count + threads - 1) / threads),
                    threads>>>(
                    static_cast<int>(sample_count),
                    static_cast<int>(hidden_count),
                    static_cast<int>(time_count),
                    static_cast<int>(step),
                    static_cast<int>(options.top_k_hidden),
                    recurrent_hidden_events_.get(),
                    recurrent_hidden_selected_.get());
                check_cuda(cudaGetLastError(), "apply_topk_hidden_events_kernel launch");
            }

            propagate_hidden_events_to_readouts_kernel<<<
                static_cast<int>((sample_count * hidden_count + threads - 1) / threads),
                threads>>>(
                static_cast<int>(sample_count),
                static_cast<int>(hidden_count),
                static_cast<int>(readout_count),
                static_cast<int>(time_count),
                static_cast<int>(step),
                recurrent_hidden_events_.get(),
                recurrent_hidden_dense_neurons_.get(),
                device_offsets_.get(),
                device_outgoing_edges_.get(),
                device_targets_.get(),
                device_delays_.get(),
                device_initial_weights_.get(),
                recurrent_dense_to_readout_.get(),
                recurrent_readout_events_.get());
            check_cuda(cudaGetLastError(), "propagate_hidden_events_to_readouts_kernel launch");

            select_recurrent_readout_events_kernel<<<
                static_cast<int>((sample_count + threads - 1) / threads),
                threads>>>(
                static_cast<int>(sample_count),
                static_cast<int>(readout_count),
                static_cast<int>(time_count),
                static_cast<int>(step),
                static_cast<int>(options.top_k_feedback_readouts),
                options.use_scores_not_spikes ? 1 : 0,
                recurrent_readout_events_.get(),
                recurrent_readout_thresholds_.get(),
                recurrent_readout_gates_.get(),
                recurrent_readout_selected_.get());
            check_cuda(cudaGetLastError(), "select_recurrent_readout_events_kernel launch");

            propagate_readout_feedback_events_kernel<<<
                static_cast<int>((sample_count * readout_count + threads - 1) / threads),
                threads>>>(
                static_cast<int>(sample_count),
                static_cast<int>(readout_count),
                static_cast<int>(hidden_count),
                static_cast<int>(time_count),
                static_cast<int>(step),
                options.feedback_decay,
                recurrent_readout_gates_.get(),
                recurrent_readout_dense_neurons_.get(),
                device_offsets_.get(),
                device_outgoing_edges_.get(),
                device_targets_.get(),
                device_delays_.get(),
                device_initial_weights_.get(),
                recurrent_dense_to_hidden_.get(),
                recurrent_hidden_events_.get());
            check_cuda(cudaGetLastError(), "propagate_readout_feedback_events_kernel launch");
        }

        sum_recurrent_readout_events_kernel<<<
            static_cast<int>((sample_count * readout_count + threads - 1) / threads),
            threads>>>(
            static_cast<int>(sample_count),
            static_cast<int>(readout_count),
            static_cast<int>(time_count),
            recurrent_readout_events_.get(),
            recurrent_readout_scores_.get());
        check_cuda(cudaGetLastError(), "sum_recurrent_readout_events_kernel launch");

        threshold_feedforward_readout_kernel<<<
            static_cast<int>((sample_count * readout_count + threads - 1) / threads),
            threads>>>(
            static_cast<int>(sample_count),
            static_cast<int>(readout_count),
            recurrent_readout_scores_.get(),
            recurrent_readout_thresholds_.get(),
            recurrent_readout_spike_counts_.get());
        check_cuda(cudaGetLastError(), "threshold recurrent readout launch");
        check_cuda(cudaDeviceSynchronize(), "recurrent feedforward synchronize");

        const auto readout_total = sample_count * readout_count;
        std::vector<float> scores(readout_total, 0.0F);
        std::vector<unsigned long long> spike_counts(readout_total, 0);
        check_cuda(
            cudaMemcpy(
                scores.data(),
                recurrent_readout_scores_.get(),
                scores.size() * sizeof(float),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy recurrent readout scores");
        check_cuda(
            cudaMemcpy(
                spike_counts.data(),
                recurrent_readout_spike_counts_.get(),
                spike_counts.size() * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy recurrent readout spike counts");

        for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
            auto& sample_result = batch.samples[sample_index];
            sample_result.readout_scores.reserve(readout_count);
            sample_result.readout_spike_counts.reserve(readout_count);
            for (std::size_t readout = 0; readout < readout_count; ++readout) {
                const auto offset = sample_index * readout_count + readout;
                sample_result.readout_scores.push_back(scores[offset]);
                sample_result.readout_spike_counts.push_back(static_cast<std::uint64_t>(spike_counts[offset]));
                sample_result.fired_spikes += static_cast<std::uint64_t>(spike_counts[offset]);
            }
            sample_result.debug.ticks_processed = time_count;
            sample_result.debug.processed_events = sample_result.delivered_spikes;
            sample_result.debug.dendritic_integrations = sample_result.delivered_spikes;
        }

        const auto end = std::chrono::steady_clock::now();
        batch.elapsed_seconds = std::chrono::duration<double>(end - start).count();
        return batch;
    }

private:
    static bool cuda_available() noexcept {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    void pack(const declarative::Connectome& connectome) {
        dense_index_.reserve(connectome.neurons.size());
        thresholds_.assign(connectome.neurons.size(), 1.0F);
        temporal_window_ticks_.assign(connectome.neurons.size(), 500);
        temporal_similarity_thresholds_.assign(connectome.neurons.size(), 0.93F);
        temporal_max_patterns_.assign(connectome.neurons.size(), 1);
        for (std::size_t i = 0; i < connectome.neurons.size(); ++i) {
            dense_index_[connectome.neurons[i].id] = static_cast<int>(i);
            thresholds_[i] = connectome.neurons[i].params.threshold;
            temporal_window_ticks_[i] = connectome.neurons[i].params.pattern_window_ticks;
            temporal_similarity_thresholds_[i] = connectome.neurons[i].params.similarity_threshold;
            temporal_max_patterns_[i] = static_cast<unsigned int>(
                std::min<std::size_t>(connectome.neurons[i].params.max_reference_patterns, 1));
        }

        std::vector<std::vector<int>> outgoing(connectome.neurons.size());
        std::vector<std::vector<int>> incoming(connectome.neurons.size());
        edge_targets_.reserve(connectome.synapses.size());
        edge_delays_.reserve(connectome.synapses.size());
        edge_compartments_.reserve(connectome.synapses.size());
        edge_receptors_.reserve(connectome.synapses.size());
        edge_plasticity_.reserve(connectome.synapses.size());
        edge_code_offsets_.reserve(connectome.synapses.size() * spike_code_capacity);
        edge_code_counts_.reserve(connectome.synapses.size());
        edge_code_window_ticks_.reserve(connectome.synapses.size());
        edge_code_similarity_thresholds_.reserve(connectome.synapses.size());
        edge_code_max_patterns_.reserve(connectome.synapses.size());
        initial_edge_weights_.reserve(connectome.synapses.size());
        edge_max_weights_.reserve(connectome.synapses.size());

        for (const auto& synapse : connectome.synapses) {
            const auto source = dense_index_.find(synapse.source);
            const auto target = dense_index_.find(synapse.target);
            if (source == dense_index_.end() || target == dense_index_.end()) {
                continue;
            }
            const auto edge = static_cast<int>(edge_targets_.size());
            edge_targets_.push_back(target->second);
            edge_delays_.push_back(std::max(1U, synapse.delay_ticks));
            max_edge_delay_ticks_ = std::max(max_edge_delay_ticks_, std::max(1U, synapse.delay_ticks));
            const auto code_count = static_cast<unsigned int>(
                std::min<std::size_t>(
                    synapse.spike_code_offsets.empty() ? 1 : synapse.spike_code_offsets.size(),
                    spike_code_capacity));
            edge_code_counts_.push_back(code_count);
            for (int code = 0; code < spike_code_capacity; ++code) {
                if (code < static_cast<int>(code_count) && !synapse.spike_code_offsets.empty()) {
                    edge_code_offsets_.push_back(synapse.spike_code_offsets[static_cast<std::size_t>(code)]);
                } else {
                    edge_code_offsets_.push_back(0U);
                }
            }
            edge_compartments_.push_back(static_cast<int>(synapse.compartment));
            edge_receptors_.push_back(static_cast<int>(synapse.receptor));
            edge_plasticity_.push_back(synapse.plasticity_enabled ? 1 : 0);
            initial_edge_weights_.push_back(synapse.weight);
            edge_max_weights_.push_back(synapse.max_weight);
            edge_code_window_ticks_.push_back(temporal_window_ticks_[static_cast<std::size_t>(target->second)]);
            edge_code_similarity_thresholds_.push_back(
                temporal_similarity_thresholds_[static_cast<std::size_t>(target->second)]);
            edge_code_max_patterns_.push_back(1U);
            outgoing[source->second].push_back(edge);
            incoming[target->second].push_back(edge);
        }

        outgoing_offsets_.assign(connectome.neurons.size() + 1, 0);
        incoming_offsets_.assign(connectome.neurons.size() + 1, 0);
        for (std::size_t neuron = 0; neuron < outgoing.size(); ++neuron) {
            outgoing_offsets_[neuron] = static_cast<int>(outgoing_edges_.size());
            for (const auto edge : outgoing[neuron]) {
                outgoing_edges_.push_back(edge);
            }
        }
        outgoing_offsets_.back() = static_cast<int>(outgoing_edges_.size());
        for (std::size_t neuron = 0; neuron < incoming.size(); ++neuron) {
            incoming_offsets_[neuron] = static_cast<int>(incoming_edges_.size());
            for (const auto edge : incoming[neuron]) {
                incoming_edges_.push_back(edge);
            }
        }
        incoming_offsets_.back() = static_cast<int>(incoming_edges_.size());
    }

    void upload_immutable() {
        device_offsets_.upload(outgoing_offsets_);
        device_outgoing_edges_.upload(outgoing_edges_);
        device_incoming_offsets_.upload(incoming_offsets_);
        device_incoming_edges_.upload(incoming_edges_);
        device_targets_.upload(edge_targets_);
        device_delays_.upload(edge_delays_);
        device_code_offsets_.upload(edge_code_offsets_);
        device_code_counts_.upload(edge_code_counts_);
        device_edge_code_window_ticks_.upload(edge_code_window_ticks_);
        device_edge_code_similarity_thresholds_.upload(edge_code_similarity_thresholds_);
        device_edge_code_max_patterns_.upload(edge_code_max_patterns_);
        device_compartments_.upload(edge_compartments_);
        device_receptors_.upload(edge_receptors_);
        device_plasticity_.upload(edge_plasticity_);
        device_initial_weights_.upload(initial_edge_weights_);
        device_max_weights_.upload(edge_max_weights_);
        device_thresholds_.upload(thresholds_);
        device_temporal_window_ticks_.upload(temporal_window_ticks_);
        device_temporal_similarity_thresholds_.upload(temporal_similarity_thresholds_);
        device_temporal_max_patterns_.upload(temporal_max_patterns_);
    }

    void ensure_feedforward_capacity(std::size_t sample_count, std::size_t input_capacity, std::size_t readout_count) {
        input_capacity = std::max<std::size_t>(1, input_capacity);
        readout_count = std::max<std::size_t>(1, readout_count);
        if (feedforward_allocated_
            && feedforward_sample_count_ >= sample_count
            && feedforward_input_capacity_ >= input_capacity
            && feedforward_readout_count_ >= readout_count) {
            return;
        }

        feedforward_sample_count_ = sample_count;
        feedforward_input_capacity_ = input_capacity;
        feedforward_readout_count_ = readout_count;
        feedforward_input_neurons_.allocate(sample_count * input_capacity);
        feedforward_input_weights_.allocate(sample_count * input_capacity);
        feedforward_input_counts_.allocate(sample_count);
        feedforward_dense_to_readout_.allocate(neuron_count_);
        feedforward_readout_thresholds_.allocate(readout_count);
        feedforward_scores_.allocate(sample_count * readout_count);
        feedforward_spike_counts_.allocate(sample_count * readout_count);
        feedforward_allocated_ = true;
    }

    std::vector<core::NeuronId> resolved_hidden_neurons(const CudaRecurrentFeedforwardOptions& options) const {
        if (!options.hidden_neurons.empty()) {
            return options.hidden_neurons;
        }

        std::unordered_set<core::NeuronId> readouts(options.readout_neurons.begin(), options.readout_neurons.end());
        std::vector<core::NeuronId> hidden;
        hidden.reserve(dense_index_.size());
        for (const auto& [id, dense] : dense_index_) {
            (void)dense;
            if (readouts.find(id) == readouts.end()) {
                hidden.push_back(id);
            }
        }
        return hidden;
    }

    void ensure_recurrent_capacity(
        std::size_t sample_count,
        std::size_t input_capacity,
        std::size_t hidden_count,
        std::size_t readout_count,
        std::size_t time_count) {
        input_capacity = std::max<std::size_t>(1, input_capacity);
        hidden_count = std::max<std::size_t>(1, hidden_count);
        readout_count = std::max<std::size_t>(1, readout_count);
        time_count = std::max<std::size_t>(1, time_count);
        if (recurrent_allocated_
            && recurrent_sample_count_ >= sample_count
            && recurrent_input_capacity_ >= input_capacity
            && recurrent_hidden_count_ >= hidden_count
            && recurrent_readout_count_ >= readout_count
            && recurrent_time_count_ >= time_count) {
            return;
        }

        recurrent_sample_count_ = sample_count;
        recurrent_input_capacity_ = input_capacity;
        recurrent_hidden_count_ = hidden_count;
        recurrent_readout_count_ = readout_count;
        recurrent_time_count_ = time_count;
        recurrent_input_neurons_.allocate(sample_count * input_capacity);
        recurrent_input_weights_.allocate(sample_count * input_capacity);
        recurrent_input_counts_.allocate(sample_count);
        recurrent_dense_to_hidden_.allocate(neuron_count_);
        recurrent_hidden_dense_neurons_.allocate(hidden_count);
        recurrent_dense_to_readout_.allocate(neuron_count_);
        recurrent_readout_dense_neurons_.allocate(readout_count);
        recurrent_readout_thresholds_.allocate(readout_count);
        recurrent_hidden_events_.allocate(sample_count * time_count * hidden_count);
        recurrent_readout_events_.allocate(sample_count * time_count * readout_count);
        recurrent_hidden_selected_.allocate(sample_count * hidden_count);
        recurrent_readout_scores_.allocate(sample_count * readout_count);
        recurrent_readout_gates_.allocate(sample_count * readout_count);
        recurrent_readout_selected_.allocate(sample_count * readout_count);
        recurrent_readout_spike_counts_.allocate(sample_count * readout_count);
        recurrent_allocated_ = true;
    }

    [[nodiscard]] CudaInferenceBatchResult run_stateless_device_batch(
        const std::vector<CudaInferenceSample>& samples,
        const CudaInferenceOptions& options) {
        CudaInferenceBatchResult batch;
        batch.executed = true;
        batch.samples.resize(samples.size());

        const auto start = std::chrono::steady_clock::now();
        const auto sample_count = samples.size();
        const auto tick_count = static_cast<std::size_t>(options.max_steps) + 1;
        const auto max_inputs = max_sample_input_count(samples);
        ensure_batched_capacity(options.max_steps, sample_count, max_inputs);
        reset_batched_buffers(sample_count, tick_count);

        std::vector<int> host_input_counts(sample_count * tick_count, 0);
        std::vector<int> host_input_neurons(sample_count * tick_count * batched_input_capacity_, 0);
        std::vector<float> host_input_weights(sample_count * tick_count * batched_input_capacity_, 0.0F);
        for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
            std::unordered_map<unsigned long long, float> merged_inputs;
            merged_inputs.reserve(samples[sample_index].inputs.size());
            for (const auto& input : samples[sample_index].inputs) {
                if (input.tick > options.max_steps) {
                    continue;
                }
                const auto found = dense_index_.find(input.neuron);
                if (found == dense_index_.end()) {
                    continue;
                }
                const auto key = (static_cast<unsigned long long>(input.tick) << 32U)
                    | static_cast<unsigned int>(found->second);
                merged_inputs[key] += input.weight;
            }
            for (const auto& [key, weight] : merged_inputs) {
                const auto tick = static_cast<std::size_t>(key >> 32U);
                const auto neuron = static_cast<int>(key & 0xFFFFFFFFULL);
                const auto count_index = sample_index * tick_count + tick;
                const auto slot = host_input_counts[count_index]++;
                if (slot < static_cast<int>(batched_input_capacity_)) {
                    const auto input_index = count_index * batched_input_capacity_ + static_cast<std::size_t>(slot);
                    host_input_neurons[input_index] = neuron;
                    host_input_weights[input_index] = weight;
                }
            }
        }

        batched_input_counts_.copy_from_host(host_input_counts.data(), host_input_counts.size());
        batched_input_neurons_.copy_from_host(host_input_neurons.data(), host_input_neurons.size());
        batched_input_weights_.copy_from_host(host_input_weights.data(), host_input_weights.size());

        const auto threads = 256;
        for (std::uint32_t tick = 0; tick <= options.max_steps; ++tick) {
            batched_fired_flags_.zero();
            const auto input_total = sample_count * batched_input_capacity_;
            if (input_total > 0) {
                const auto blocks = static_cast<int>((input_total + threads - 1) / threads);
                apply_batched_external_inputs_kernel<<<blocks, threads>>>(
                    static_cast<int>(sample_count),
                    static_cast<int>(tick_count),
                    static_cast<int>(batched_input_capacity_),
                    static_cast<int>(neuron_count_),
                    static_cast<int>(edge_targets_.size()),
                    static_cast<int>(batched_fired_capacity_),
                    tick,
                    batched_input_neurons_.get(),
                    batched_input_weights_.get(),
                    batched_input_counts_.get(),
                    device_incoming_offsets_.get(),
                    device_incoming_edges_.get(),
                    device_thresholds_.get(),
                    batched_membrane_.get(),
                    batched_soma_.get(),
                    batched_basal_.get(),
                    batched_apical_.get(),
                    batched_inhibitory_.get(),
                    batched_soma_ticks_.get(),
                    batched_basal_ticks_.get(),
                    batched_apical_ticks_.get(),
                    batched_inhibitory_ticks_.get(),
                    batched_soma_tau_.get(),
                    batched_basal_tau_.get(),
                    batched_apical_tau_.get(),
                    batched_inhibitory_tau_.get(),
                    batched_spike_counts_.get(),
                    batched_last_fired_ticks_.get(),
                    batched_last_post_ticks_.get(),
                    batched_neuron_locks_.get(),
                    batched_fired_flags_.get(),
                    batched_fired_by_tick_.get(),
                    batched_fired_counts_.get());
                check_cuda(cudaGetLastError(), "apply_batched_external_inputs_kernel launch");
                check_cuda(cudaDeviceSynchronize(), "apply_batched_external_inputs_kernel synchronize");
            }

            const auto event_total = sample_count * batched_event_capacity_;
            if (event_total > 0) {
                const auto blocks = static_cast<int>((event_total + threads - 1) / threads);
                process_batched_synapse_events_kernel<<<blocks, threads>>>(
                    static_cast<int>(sample_count),
                    static_cast<int>(tick_count),
                    static_cast<int>(batched_event_capacity_),
                    static_cast<int>(batched_fired_capacity_),
                    static_cast<int>(neuron_count_),
                    static_cast<int>(edge_targets_.size()),
                    tick,
                    batched_scheduled_edges_.get(),
                    batched_scheduled_counts_.get(),
                    batched_fired_by_tick_.get(),
                    batched_fired_counts_.get(),
                    device_targets_.get(),
                    device_delays_.get(),
                    device_code_counts_.get(),
                    device_compartments_.get(),
                    device_receptors_.get(),
                    device_plasticity_.get(),
                    device_incoming_offsets_.get(),
                    device_incoming_edges_.get(),
                    batched_weights_.get(),
                    device_max_weights_.get(),
                    batched_last_pre_ticks_.get(),
                    batched_last_post_ticks_.get(),
                    batched_pre_counts_.get(),
                    batched_plasticity_updates_.get(),
                    batched_last_weight_delta_.get(),
                    device_thresholds_.get(),
                    batched_membrane_.get(),
                    batched_soma_.get(),
                    batched_basal_.get(),
                    batched_apical_.get(),
                    batched_inhibitory_.get(),
                    batched_soma_ticks_.get(),
                    batched_basal_ticks_.get(),
                    batched_apical_ticks_.get(),
                    batched_inhibitory_ticks_.get(),
                    batched_soma_tau_.get(),
                    batched_basal_tau_.get(),
                    batched_apical_tau_.get(),
                    batched_inhibitory_tau_.get(),
                    batched_spike_counts_.get(),
                    batched_last_fired_ticks_.get(),
                    batched_neuron_locks_.get(),
                    batched_fired_flags_.get(),
                    device_temporal_window_ticks_.get(),
                    device_temporal_similarity_thresholds_.get(),
                    device_temporal_max_patterns_.get(),
                    batched_temporal_window_start_.get(),
                    batched_temporal_observed_offsets_.get(),
                    batched_temporal_observed_counts_.get(),
                    batched_temporal_reference_offsets_.get(),
                    batched_temporal_reference_counts_.get(),
                    batched_temporal_learned_counts_.get(),
                    batched_temporal_match_counts_.get(),
                    batched_temporal_last_match_.get(),
                    device_edge_code_window_ticks_.get(),
                    device_edge_code_similarity_thresholds_.get(),
                    device_edge_code_max_patterns_.get(),
                    batched_edge_code_window_start_.get(),
                    batched_edge_code_observed_offsets_.get(),
                    batched_edge_code_observed_counts_.get(),
                    batched_edge_code_reference_offsets_.get(),
                    batched_edge_code_reference_counts_.get(),
                    batched_edge_code_learned_counts_.get(),
                    batched_edge_code_match_counts_.get(),
                    batched_edge_code_last_match_.get(),
                    batched_metrics_.get());
                check_cuda(cudaGetLastError(), "process_batched_synapse_events_kernel launch");
                check_cuda(cudaDeviceSynchronize(), "process_batched_synapse_events_kernel synchronize");
            }

            if (tick < options.max_steps && !edge_targets_.empty()) {
                const auto fired_total = sample_count * batched_fired_capacity_;
                const auto blocks = static_cast<int>((fired_total + threads - 1) / threads);
                expand_batched_fired_kernel<<<blocks, threads>>>(
                    static_cast<int>(sample_count),
                    static_cast<int>(tick_count),
                    static_cast<int>(batched_fired_capacity_),
                    static_cast<int>(batched_event_capacity_),
                    tick,
                    options.max_steps,
                    batched_fired_by_tick_.get(),
                    batched_fired_counts_.get(),
                    device_offsets_.get(),
                    device_outgoing_edges_.get(),
                    device_delays_.get(),
                    device_code_offsets_.get(),
                    device_code_counts_.get(),
                    batched_scheduled_edges_.get(),
                    batched_scheduled_counts_.get(),
                    batched_metrics_.get());
                check_cuda(cudaGetLastError(), "expand_batched_fired_kernel launch");
                check_cuda(cudaDeviceSynchronize(), "expand_batched_fired_kernel synchronize");
            }
        }

        std::vector<int> final_fired_counts(host_input_counts.size(), 0);
        std::vector<int> final_scheduled_counts(sample_count * tick_count, 0);
        std::vector<unsigned long long> spike_counts(sample_count * neuron_count_, 0);
        std::vector<unsigned long long> raw_metrics(sample_count * metric_count, 0);
        check_cuda(
            cudaMemcpy(
                final_fired_counts.data(),
                batched_fired_counts_.get(),
                final_fired_counts.size() * sizeof(int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy batched fired counts");
        check_cuda(
            cudaMemcpy(
                final_scheduled_counts.data(),
                batched_scheduled_counts_.get(),
                final_scheduled_counts.size() * sizeof(int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy batched scheduled counts");
        check_cuda(
            cudaMemcpy(
                spike_counts.data(),
                batched_spike_counts_.get(),
                spike_counts.size() * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy batched spike counts");
        check_cuda(
            cudaMemcpy(
                raw_metrics.data(),
                batched_metrics_.get(),
                raw_metrics.size() * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy batched metrics");

        for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
            auto& sample_result = batch.samples[sample_index];
            CudaDebugMetrics debug;
            debug.ticks_processed = tick_count;
            for (std::size_t tick = 0; tick < tick_count; ++tick) {
                const auto count_index = sample_index * tick_count + tick;
                sample_result.delivered_spikes +=
                    static_cast<std::uint64_t>(std::max(0, host_input_counts[count_index]));
                sample_result.delivered_spikes += static_cast<std::uint64_t>(
                    std::min(std::max(0, final_scheduled_counts[count_index]), static_cast<int>(batched_event_capacity_)));
                sample_result.fired_spikes +=
                    static_cast<std::uint64_t>(std::max(0, final_fired_counts[count_index]));
                debug.max_scheduled_events_per_tick = std::max(
                    debug.max_scheduled_events_per_tick,
                    static_cast<std::uint64_t>(std::max(0, final_scheduled_counts[count_index])));
                debug.max_fired_neurons_per_tick = std::max(
                    debug.max_fired_neurons_per_tick,
                    static_cast<std::uint64_t>(std::max(0, final_fired_counts[count_index])));
            }
            const auto metric_base = sample_index * metric_count;
            debug.scheduled_event_requests = raw_metrics[metric_base + metric_scheduled_event_requests];
            debug.scheduled_events = raw_metrics[metric_base + metric_scheduled_events];
            debug.processed_events = raw_metrics[metric_base + metric_processed_events];
            debug.dropped_events = raw_metrics[metric_base + metric_dropped_events];
            debug.fired_appends = raw_metrics[metric_base + metric_fired_appends];
            debug.fired_overflow = raw_metrics[metric_base + metric_fired_overflow];
            debug.lock_spin_iterations = raw_metrics[metric_base + metric_lock_spin_iterations];
            debug.lock_timeouts = raw_metrics[metric_base + metric_lock_timeouts];
            debug.stdp_updates = raw_metrics[metric_base + metric_stdp_updates];
            debug.stdp_ltp = raw_metrics[metric_base + metric_stdp_ltp];
            debug.stdp_ltd = raw_metrics[metric_base + metric_stdp_ltd];
            debug.receptor_ampa_events = raw_metrics[metric_base + metric_receptor_ampa_events];
            debug.receptor_nmda_events = raw_metrics[metric_base + metric_receptor_nmda_events];
            debug.receptor_gaba_a_events = raw_metrics[metric_base + metric_receptor_gaba_a_events];
            debug.receptor_gaba_b_events = raw_metrics[metric_base + metric_receptor_gaba_b_events];
            debug.dendritic_integrations = raw_metrics[metric_base + metric_dendritic_integrations];
            debug.post_plasticity_scans = raw_metrics[metric_base + metric_post_plasticity_scans];
            debug.temporal_observations = raw_metrics[metric_base + metric_temporal_observations];
            debug.temporal_patterns_learned = raw_metrics[metric_base + metric_temporal_patterns_learned];
            debug.temporal_matches = raw_metrics[metric_base + metric_temporal_matches];
            sample_result.debug = debug;

            sample_result.readout_spike_counts.reserve(options.readout_neurons.size());
            for (const auto id : options.readout_neurons) {
                const auto found = dense_index_.find(id);
                sample_result.readout_spike_counts.push_back(
                    found == dense_index_.end()
                        ? 0ULL
                        : static_cast<std::uint64_t>(
                            spike_counts[sample_index * neuron_count_ + static_cast<std::size_t>(found->second)]));
            }
        }

        const auto end = std::chrono::steady_clock::now();
        batch.elapsed_seconds = std::chrono::duration<double>(end - start).count();
        return batch;
    }

    void ensure_batched_capacity(std::uint32_t max_steps, std::size_t sample_count, std::size_t max_inputs) {
        const auto tick_count = static_cast<std::size_t>(max_steps) + 1;
        const auto event_capacity = std::max<std::size_t>(1, edge_targets_.size());
        const auto fired_capacity = std::max<std::size_t>({1, neuron_count_, max_inputs});
        const auto input_capacity = std::max<std::size_t>(1, max_inputs);
        if (batched_allocated_
            && batched_sample_count_ >= sample_count
            && batched_tick_count_ >= tick_count
            && batched_event_capacity_ >= event_capacity
            && batched_fired_capacity_ >= fired_capacity
            && batched_input_capacity_ >= input_capacity) {
            return;
        }

        batched_sample_count_ = sample_count;
        batched_tick_count_ = tick_count;
        batched_event_capacity_ = event_capacity;
        batched_fired_capacity_ = fired_capacity;
        batched_input_capacity_ = input_capacity;
        const auto edge_count = edge_targets_.size();
        const auto neuron_total = sample_count * neuron_count_;
        const auto edge_total = sample_count * edge_count;

        batched_membrane_.allocate(neuron_total);
        batched_soma_.allocate(neuron_total);
        batched_basal_.allocate(neuron_total);
        batched_apical_.allocate(neuron_total);
        batched_inhibitory_.allocate(neuron_total);
        batched_soma_tau_.allocate(neuron_total);
        batched_basal_tau_.allocate(neuron_total);
        batched_apical_tau_.allocate(neuron_total);
        batched_inhibitory_tau_.allocate(neuron_total);
        batched_soma_ticks_.allocate(neuron_total);
        batched_basal_ticks_.allocate(neuron_total);
        batched_apical_ticks_.allocate(neuron_total);
        batched_inhibitory_ticks_.allocate(neuron_total);
        batched_spike_counts_.allocate(neuron_total);
        batched_last_fired_ticks_.allocate(neuron_total);
        batched_temporal_window_start_.allocate(neuron_total);
        batched_temporal_observed_offsets_.allocate(neuron_total * temporal_capacity);
        batched_temporal_observed_counts_.allocate(neuron_total);
        batched_temporal_reference_offsets_.allocate(neuron_total * temporal_capacity);
        batched_temporal_reference_counts_.allocate(neuron_total);
        batched_temporal_learned_counts_.allocate(neuron_total);
        batched_temporal_match_counts_.allocate(neuron_total);
        batched_temporal_last_match_.allocate(neuron_total);
        batched_weights_.allocate(edge_total);
        batched_last_pre_ticks_.allocate(edge_total);
        batched_last_post_ticks_.allocate(edge_total);
        batched_pre_counts_.allocate(edge_total);
        batched_plasticity_updates_.allocate(edge_total);
        batched_last_weight_delta_.allocate(edge_total);
        batched_edge_code_window_start_.allocate(edge_total);
        batched_edge_code_observed_offsets_.allocate(edge_total * temporal_capacity);
        batched_edge_code_observed_counts_.allocate(edge_total);
        batched_edge_code_reference_offsets_.allocate(edge_total * temporal_capacity);
        batched_edge_code_reference_counts_.allocate(edge_total);
        batched_edge_code_learned_counts_.allocate(edge_total);
        batched_edge_code_match_counts_.allocate(edge_total);
        batched_edge_code_last_match_.allocate(edge_total);
        batched_neuron_locks_.allocate(neuron_total);
        batched_fired_flags_.allocate(neuron_total);
        batched_scheduled_edges_.allocate(sample_count * tick_count * event_capacity);
        batched_scheduled_counts_.allocate(sample_count * tick_count);
        batched_fired_by_tick_.allocate(sample_count * tick_count * fired_capacity);
        batched_fired_counts_.allocate(sample_count * tick_count);
        batched_input_neurons_.allocate(sample_count * tick_count * input_capacity);
        batched_input_weights_.allocate(sample_count * tick_count * input_capacity);
        batched_input_counts_.allocate(sample_count * tick_count);
        batched_metrics_.allocate(sample_count * metric_count);
        batched_allocated_ = true;
    }

    void reset_batched_buffers(std::size_t sample_count, std::size_t tick_count) {
        const auto edge_count = edge_targets_.size();
        const auto neuron_total = sample_count * neuron_count_;
        const auto edge_total = sample_count * edge_count;
        batched_membrane_.zero();
        batched_soma_.zero();
        batched_basal_.zero();
        batched_apical_.zero();
        batched_inhibitory_.zero();
        batched_soma_ticks_.zero();
        batched_basal_ticks_.zero();
        batched_apical_ticks_.zero();
        batched_inhibitory_ticks_.zero();
        batched_spike_counts_.zero();
        batched_temporal_window_start_.zero();
        batched_temporal_observed_offsets_.zero();
        batched_temporal_observed_counts_.zero();
        batched_temporal_reference_offsets_.zero();
        batched_temporal_reference_counts_.zero();
        batched_temporal_learned_counts_.zero();
        batched_temporal_match_counts_.zero();
        batched_temporal_last_match_.zero();
        batched_pre_counts_.zero();
        batched_plasticity_updates_.zero();
        batched_last_weight_delta_.zero();
        batched_edge_code_window_start_.zero();
        batched_edge_code_observed_offsets_.zero();
        batched_edge_code_observed_counts_.zero();
        batched_edge_code_reference_offsets_.zero();
        batched_edge_code_reference_counts_.zero();
        batched_edge_code_learned_counts_.zero();
        batched_edge_code_match_counts_.zero();
        batched_edge_code_last_match_.zero();
        batched_neuron_locks_.zero();
        batched_fired_flags_.zero();
        batched_scheduled_edges_.zero();
        batched_scheduled_counts_.zero();
        batched_fired_by_tick_.zero();
        batched_fired_counts_.zero();
        batched_input_neurons_.zero();
        batched_input_weights_.zero();
        batched_input_counts_.zero();
        batched_metrics_.zero();

        std::vector<float> soma_tau(neuron_total, 5.0F);
        std::vector<float> basal_tau(neuron_total, 5.0F);
        std::vector<float> apical_tau(neuron_total, 100.0F);
        std::vector<float> inhibitory_tau(neuron_total, 10.0F);
        batched_soma_tau_.copy_from_host(soma_tau.data(), soma_tau.size());
        batched_basal_tau_.copy_from_host(basal_tau.data(), basal_tau.size());
        batched_apical_tau_.copy_from_host(apical_tau.data(), apical_tau.size());
        batched_inhibitory_tau_.copy_from_host(inhibitory_tau.data(), inhibitory_tau.size());

        std::vector<unsigned long long> invalid_neuron_ticks(neuron_total, invalid_tick);
        batched_last_fired_ticks_.copy_from_host(invalid_neuron_ticks.data(), invalid_neuron_ticks.size());
        if (edge_total > 0) {
            std::vector<unsigned long long> invalid_edge_ticks(edge_total, invalid_tick);
            std::vector<float> weights(edge_total, 0.0F);
            for (std::size_t sample = 0; sample < sample_count; ++sample) {
                std::copy(
                    initial_edge_weights_.begin(),
                    initial_edge_weights_.end(),
                    weights.begin() + static_cast<std::ptrdiff_t>(sample * edge_count));
            }
            batched_last_pre_ticks_.copy_from_host(invalid_edge_ticks.data(), invalid_edge_ticks.size());
            batched_last_post_ticks_.copy_from_host(invalid_edge_ticks.data(), invalid_edge_ticks.size());
            batched_weights_.copy_from_host(weights.data(), weights.size());
        }
        (void)tick_count;
    }

    std::size_t max_sample_input_count(const std::vector<CudaInferenceSample>& samples) const {
        std::size_t max_count = 1;
        for (const auto& sample : samples) {
            max_count = std::max(max_count, sample.inputs.size());
        }
        return max_count;
    }

    void ensure_mutable_capacity(std::uint32_t max_steps, std::size_t max_inputs) {
        const auto tick_count = static_cast<std::size_t>(max_steps) + 1;
        const auto event_capacity = std::max<std::size_t>(1, edge_targets_.size());
        const auto fired_capacity = std::max<std::size_t>({1, neuron_count_, max_inputs});
        if (mutable_allocated_
            && allocated_tick_count_ >= tick_count
            && allocated_event_capacity_ >= event_capacity
            && allocated_fired_capacity_ >= fired_capacity) {
            return;
        }

        allocated_tick_count_ = tick_count;
        allocated_event_capacity_ = event_capacity;
        allocated_fired_capacity_ = fired_capacity;
        const auto edge_count = edge_targets_.size();

        device_membrane_.allocate(neuron_count_);
        device_soma_.allocate(neuron_count_);
        device_basal_.allocate(neuron_count_);
        device_apical_.allocate(neuron_count_);
        device_inhibitory_.allocate(neuron_count_);
        device_soma_tau_.allocate(neuron_count_);
        device_basal_tau_.allocate(neuron_count_);
        device_apical_tau_.allocate(neuron_count_);
        device_inhibitory_tau_.allocate(neuron_count_);
        device_soma_ticks_.allocate(neuron_count_);
        device_basal_ticks_.allocate(neuron_count_);
        device_apical_ticks_.allocate(neuron_count_);
        device_inhibitory_ticks_.allocate(neuron_count_);
        device_spike_counts_.allocate(neuron_count_);
        device_last_fired_ticks_.allocate(neuron_count_);
        device_temporal_window_start_.allocate(neuron_count_);
        device_temporal_observed_offsets_.allocate(neuron_count_ * temporal_capacity);
        device_temporal_observed_counts_.allocate(neuron_count_);
        device_temporal_reference_offsets_.allocate(neuron_count_ * temporal_capacity);
        device_temporal_reference_counts_.allocate(neuron_count_);
        device_temporal_learned_counts_.allocate(neuron_count_);
        device_temporal_match_counts_.allocate(neuron_count_);
        device_temporal_last_match_.allocate(neuron_count_);
        device_weights_.allocate(edge_count);
        device_last_pre_ticks_.allocate(edge_count);
        device_last_post_ticks_.allocate(edge_count);
        device_pre_counts_.allocate(edge_count);
        device_plasticity_updates_.allocate(edge_count);
        device_last_weight_delta_.allocate(edge_count);
        device_edge_code_window_start_.allocate(edge_count);
        device_edge_code_observed_offsets_.allocate(edge_count * temporal_capacity);
        device_edge_code_observed_counts_.allocate(edge_count);
        device_edge_code_reference_offsets_.allocate(edge_count * temporal_capacity);
        device_edge_code_reference_counts_.allocate(edge_count);
        device_edge_code_learned_counts_.allocate(edge_count);
        device_edge_code_match_counts_.allocate(edge_count);
        device_edge_code_last_match_.allocate(edge_count);
        device_neuron_locks_.allocate(neuron_count_);
        device_fired_flags_.allocate(neuron_count_);
        device_scheduled_edges_.allocate(tick_count * event_capacity);
        device_scheduled_counts_.allocate(tick_count);
        device_fired_by_tick_.allocate(tick_count * fired_capacity);
        device_fired_counts_.allocate(tick_count);
        device_external_neurons_.allocate(max_inputs);
        device_external_weights_.allocate(max_inputs);
        device_metrics_.allocate(metric_count);
        mutable_allocated_ = true;
        mutable_state_initialized_ = false;
    }

    void reset_mutable_buffers(std::uint32_t max_steps) {
        const auto edge_count = edge_targets_.size();
        device_membrane_.zero();
        device_soma_.zero();
        device_basal_.zero();
        device_apical_.zero();
        device_inhibitory_.zero();
        device_soma_ticks_.zero();
        device_basal_ticks_.zero();
        device_apical_ticks_.zero();
        device_inhibitory_ticks_.zero();
        device_spike_counts_.zero();
        device_temporal_window_start_.zero();
        device_temporal_observed_offsets_.zero();
        device_temporal_observed_counts_.zero();
        device_temporal_reference_offsets_.zero();
        device_temporal_reference_counts_.zero();
        device_temporal_learned_counts_.zero();
        device_temporal_match_counts_.zero();
        device_temporal_last_match_.zero();
        device_pre_counts_.zero();
        device_plasticity_updates_.zero();
        device_last_weight_delta_.zero();
        device_edge_code_window_start_.zero();
        device_edge_code_observed_offsets_.zero();
        device_edge_code_observed_counts_.zero();
        device_edge_code_reference_offsets_.zero();
        device_edge_code_reference_counts_.zero();
        device_edge_code_learned_counts_.zero();
        device_edge_code_match_counts_.zero();
        device_edge_code_last_match_.zero();
        device_neuron_locks_.zero();
        device_fired_flags_.zero();
        device_scheduled_counts_.zero();
        device_fired_counts_.zero();
        device_external_neurons_.zero();
        device_external_weights_.zero();
        device_metrics_.zero();

        std::vector<float> soma_tau(neuron_count_, 5.0F);
        std::vector<float> basal_tau(neuron_count_, 5.0F);
        std::vector<float> apical_tau(neuron_count_, 100.0F);
        std::vector<float> inhibitory_tau(neuron_count_, 10.0F);
        device_soma_tau_.copy_from_host(soma_tau.data(), soma_tau.size());
        device_basal_tau_.copy_from_host(basal_tau.data(), basal_tau.size());
        device_apical_tau_.copy_from_host(apical_tau.data(), apical_tau.size());
        device_inhibitory_tau_.copy_from_host(inhibitory_tau.data(), inhibitory_tau.size());

        std::vector<unsigned long long> invalid_neuron_ticks(neuron_count_, invalid_tick);
        device_last_fired_ticks_.copy_from_host(invalid_neuron_ticks.data(), invalid_neuron_ticks.size());
        if (edge_count > 0) {
            std::vector<unsigned long long> invalid_edge_ticks(edge_count, invalid_tick);
            device_last_pre_ticks_.copy_from_host(invalid_edge_ticks.data(), invalid_edge_ticks.size());
            device_last_post_ticks_.copy_from_host(invalid_edge_ticks.data(), invalid_edge_ticks.size());
            device_weights_.copy_from_host(initial_edge_weights_.data(), initial_edge_weights_.size());
        }
        mutable_state_initialized_ = true;
        (void)max_steps;
    }

    void reset_sample_work_buffers() {
        device_neuron_locks_.zero();
        device_fired_flags_.zero();
        device_scheduled_counts_.zero();
        device_fired_counts_.zero();
        device_external_neurons_.zero();
        device_external_weights_.zero();
        device_metrics_.zero();
    }

    std::vector<std::vector<CudaWeightedInput>> dense_inputs_by_tick(
        const CudaInferenceSample& sample,
        std::uint32_t max_steps) const {
        std::vector<std::unordered_map<int, float>> merged(static_cast<std::size_t>(max_steps) + 1);
        for (const auto& input : sample.inputs) {
            if (input.tick > max_steps) {
                continue;
            }
            const auto found = dense_index_.find(input.neuron);
            if (found == dense_index_.end()) {
                continue;
            }
            merged[input.tick][found->second] += input.weight;
        }
        std::vector<std::vector<CudaWeightedInput>> by_tick(static_cast<std::size_t>(max_steps) + 1);
        for (std::size_t tick = 0; tick < merged.size(); ++tick) {
            by_tick[tick].reserve(merged[tick].size());
            for (const auto& [neuron, weight] : merged[tick]) {
                by_tick[tick].push_back({
                    .neuron = static_cast<core::NeuronId>(neuron),
                    .weight = weight,
                    .tick = static_cast<std::uint32_t>(tick),
                });
            }
        }
        return by_tick;
    }

    void apply_external_inputs_for_tick(
        const std::vector<CudaWeightedInput>& inputs,
        std::uint32_t tick) {
        if (inputs.empty()) {
            return;
        }
        std::vector<int> neurons;
        std::vector<float> weights;
        neurons.reserve(inputs.size());
        weights.reserve(inputs.size());
        for (const auto& input : inputs) {
            neurons.push_back(static_cast<int>(input.neuron));
            weights.push_back(input.weight);
        }
        device_external_neurons_.copy_from_host(neurons.data(), neurons.size());
        device_external_weights_.copy_from_host(weights.data(), weights.size());
        const auto threads = 256;
        const auto count = static_cast<int>(inputs.size());
        const auto blocks = (count + threads - 1) / threads;
        apply_external_inputs_kernel<<<blocks, threads>>>(
            device_external_neurons_.get(),
            device_external_weights_.get(),
            count,
            tick,
            device_incoming_offsets_.get(),
            device_incoming_edges_.get(),
            device_thresholds_.get(),
            device_membrane_.get(),
            device_soma_.get(),
            device_basal_.get(),
            device_apical_.get(),
            device_inhibitory_.get(),
            device_soma_ticks_.get(),
            device_basal_ticks_.get(),
            device_apical_ticks_.get(),
            device_inhibitory_ticks_.get(),
            device_soma_tau_.get(),
            device_basal_tau_.get(),
            device_apical_tau_.get(),
            device_inhibitory_tau_.get(),
            device_spike_counts_.get(),
            device_last_fired_ticks_.get(),
            device_last_post_ticks_.get(),
            device_neuron_locks_.get(),
            device_fired_flags_.get(),
            device_fired_by_tick_.get() + static_cast<std::size_t>(tick) * allocated_fired_capacity_,
            device_fired_counts_.get() + tick,
            static_cast<int>(allocated_fired_capacity_));
        check_cuda(cudaGetLastError(), "apply_external_inputs_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "apply_external_inputs_kernel synchronize");
    }

    CudaInferenceSampleResult run_one(
        const CudaInferenceSample& sample,
        const CudaInferenceOptions& options) {
        const auto by_tick = dense_inputs_by_tick(sample, options.max_steps);
        const auto initial_spike_counts = copy_u64_from_device(
            device_spike_counts_.get(),
            neuron_count_,
            "cudaMemcpy initial inference spike counts");

        CudaInferenceSampleResult result;
        result.delivered_spikes = 0;
        for (const auto& inputs : by_tick) {
            result.delivered_spikes += inputs.size();
        }
        result.fired_spikes = 0;
        CudaDebugMetrics debug;
        const auto threads = 256;
        const auto edge_count = edge_targets_.size();

        for (std::uint32_t tick = 0; tick <= options.max_steps; ++tick) {
            ++debug.ticks_processed;
            device_fired_flags_.zero();
            apply_external_inputs_for_tick(by_tick[tick], tick);

            int fired_before = 0;
            check_cuda(
                cudaMemcpy(
                    &fired_before,
                    device_fired_counts_.get() + tick,
                    sizeof(int),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy fired before");
            if (fired_before > 0) {
                result.fired_spikes += static_cast<std::uint64_t>(fired_before);
                const auto mark_blocks = (fired_before + threads - 1) / threads;
                mark_fired_flags_kernel<<<mark_blocks, threads>>>(
                    device_fired_by_tick_.get() + static_cast<std::size_t>(tick) * allocated_fired_capacity_,
                    std::min(fired_before, static_cast<int>(allocated_fired_capacity_)),
                    device_fired_flags_.get());
                check_cuda(cudaGetLastError(), "mark_fired_flags_kernel launch");
                check_cuda(cudaDeviceSynchronize(), "mark_fired_flags_kernel synchronize");
            }

            int event_count = 0;
            check_cuda(
                cudaMemcpy(
                    &event_count,
                    device_scheduled_counts_.get() + tick,
                    sizeof(int),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy event count");
            debug.max_scheduled_events_per_tick = std::max(
                debug.max_scheduled_events_per_tick,
                static_cast<std::uint64_t>(std::max(event_count, 0)));
            if (event_count > 0) {
                const auto clamped_event_count = std::min(event_count, static_cast<int>(allocated_event_capacity_));
                if (event_count > clamped_event_count) {
                    debug.dropped_events += static_cast<std::uint64_t>(event_count - clamped_event_count);
                }
                const auto event_blocks = (clamped_event_count + threads - 1) / threads;
                process_synapse_events_kernel<<<event_blocks, threads>>>(
                    device_scheduled_edges_.get() + static_cast<std::size_t>(tick) * allocated_event_capacity_,
                    clamped_event_count,
                    device_fired_by_tick_.get() + static_cast<std::size_t>(tick) * allocated_fired_capacity_,
                    device_fired_counts_.get() + tick,
                    static_cast<int>(allocated_fired_capacity_),
                    tick,
                    device_targets_.get(),
                    device_delays_.get(),
                    device_code_counts_.get(),
                    device_compartments_.get(),
                    device_receptors_.get(),
                    device_plasticity_.get(),
                    device_incoming_offsets_.get(),
                    device_incoming_edges_.get(),
                    device_weights_.get(),
                    device_max_weights_.get(),
                    device_last_pre_ticks_.get(),
                    device_last_post_ticks_.get(),
                    device_pre_counts_.get(),
                    device_plasticity_updates_.get(),
                    device_last_weight_delta_.get(),
                    device_thresholds_.get(),
                    device_membrane_.get(),
                    device_soma_.get(),
                    device_basal_.get(),
                    device_apical_.get(),
                    device_inhibitory_.get(),
                    device_soma_ticks_.get(),
                    device_basal_ticks_.get(),
                    device_apical_ticks_.get(),
                    device_inhibitory_ticks_.get(),
                    device_soma_tau_.get(),
                    device_basal_tau_.get(),
                    device_apical_tau_.get(),
                    device_inhibitory_tau_.get(),
                    device_spike_counts_.get(),
                    device_last_fired_ticks_.get(),
                    device_neuron_locks_.get(),
                    device_fired_flags_.get(),
                    device_temporal_window_ticks_.get(),
                    device_temporal_similarity_thresholds_.get(),
                    device_temporal_max_patterns_.get(),
                    device_temporal_window_start_.get(),
                    device_temporal_observed_offsets_.get(),
                    device_temporal_observed_counts_.get(),
                    device_temporal_reference_offsets_.get(),
                    device_temporal_reference_counts_.get(),
                    device_temporal_learned_counts_.get(),
                    device_temporal_match_counts_.get(),
                    device_temporal_last_match_.get(),
                    device_edge_code_window_ticks_.get(),
                    device_edge_code_similarity_thresholds_.get(),
                    device_edge_code_max_patterns_.get(),
                    device_edge_code_window_start_.get(),
                    device_edge_code_observed_offsets_.get(),
                    device_edge_code_observed_counts_.get(),
                    device_edge_code_reference_offsets_.get(),
                    device_edge_code_reference_counts_.get(),
                    device_edge_code_learned_counts_.get(),
                    device_edge_code_match_counts_.get(),
                    device_edge_code_last_match_.get(),
                    device_metrics_.get());
                check_cuda(cudaGetLastError(), "process_synapse_events_kernel launch");
                check_cuda(cudaDeviceSynchronize(), "process_synapse_events_kernel synchronize");
                result.delivered_spikes += static_cast<std::uint64_t>(clamped_event_count);
            }

            int fired_after = 0;
            check_cuda(
                cudaMemcpy(
                    &fired_after,
                    device_fired_counts_.get() + tick,
                    sizeof(int),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy fired after");
            debug.max_fired_neurons_per_tick = std::max(
                debug.max_fired_neurons_per_tick,
                static_cast<std::uint64_t>(std::max(fired_after, 0)));
            result.fired_spikes += static_cast<std::uint64_t>(std::max(0, fired_after - fired_before));

            if (fired_after > 0 && tick < options.max_steps && edge_count > 0) {
                const auto expand_blocks = (fired_after + threads - 1) / threads;
                expand_fired_kernel<<<expand_blocks, threads>>>(
                    device_fired_by_tick_.get() + static_cast<std::size_t>(tick) * allocated_fired_capacity_,
                    std::min(fired_after, static_cast<int>(allocated_fired_capacity_)),
                    device_offsets_.get(),
                    device_outgoing_edges_.get(),
                    device_delays_.get(),
                    device_code_offsets_.get(),
                    device_code_counts_.get(),
                    device_scheduled_edges_.get(),
                    device_scheduled_counts_.get(),
                    static_cast<int>(allocated_event_capacity_),
                    tick,
                    options.max_steps,
                    device_metrics_.get());
                check_cuda(cudaGetLastError(), "expand_fired_kernel launch");
                check_cuda(cudaDeviceSynchronize(), "expand_fired_kernel synchronize");
            }
        }

        result.debug = metrics_from_device(device_metrics_.get(), debug);
        const auto spike_counts = copy_u64_from_device(
            device_spike_counts_.get(),
            neuron_count_,
            "cudaMemcpy inference spike counts");
        result.readout_spike_counts.reserve(options.readout_neurons.size());
        for (const auto id : options.readout_neurons) {
            const auto found = dense_index_.find(id);
            if (found == dense_index_.end()) {
                result.readout_spike_counts.push_back(0);
            } else {
                const auto index = static_cast<std::size_t>(found->second);
                result.readout_spike_counts.push_back(
                    static_cast<std::uint64_t>(spike_counts[index] - initial_spike_counts[index]));
            }
        }
        return result;
    }

    bool available_{false};
    bool mutable_allocated_{false};
    bool mutable_state_initialized_{false};
    bool batched_allocated_{false};
    bool feedforward_allocated_{false};
    bool recurrent_allocated_{false};
    std::size_t neuron_count_{0};
    std::size_t allocated_tick_count_{0};
    std::size_t allocated_event_capacity_{0};
    std::size_t allocated_fired_capacity_{0};
    std::size_t batched_sample_count_{0};
    std::size_t batched_tick_count_{0};
    std::size_t batched_event_capacity_{0};
    std::size_t batched_fired_capacity_{0};
    std::size_t batched_input_capacity_{0};
    std::size_t feedforward_sample_count_{0};
    std::size_t feedforward_input_capacity_{0};
    std::size_t feedforward_readout_count_{0};
    std::size_t recurrent_sample_count_{0};
    std::size_t recurrent_input_capacity_{0};
    std::size_t recurrent_hidden_count_{0};
    std::size_t recurrent_readout_count_{0};
    std::size_t recurrent_time_count_{0};
    unsigned int max_edge_delay_ticks_{1};

    std::unordered_map<core::NeuronId, int> dense_index_;
    std::vector<float> thresholds_;
    std::vector<unsigned long long> temporal_window_ticks_;
    std::vector<float> temporal_similarity_thresholds_;
    std::vector<unsigned int> temporal_max_patterns_;
    std::vector<int> outgoing_offsets_;
    std::vector<int> outgoing_edges_;
    std::vector<int> incoming_offsets_;
    std::vector<int> incoming_edges_;
    std::vector<int> edge_targets_;
    std::vector<unsigned int> edge_delays_;
    std::vector<int> edge_compartments_;
    std::vector<int> edge_receptors_;
    std::vector<int> edge_plasticity_;
    std::vector<unsigned int> edge_code_offsets_;
    std::vector<unsigned int> edge_code_counts_;
    std::vector<unsigned long long> edge_code_window_ticks_;
    std::vector<float> edge_code_similarity_thresholds_;
    std::vector<unsigned int> edge_code_max_patterns_;
    std::vector<float> initial_edge_weights_;
    std::vector<float> edge_max_weights_;

    DeviceBuffer<int> device_offsets_;
    DeviceBuffer<int> device_outgoing_edges_;
    DeviceBuffer<int> device_incoming_offsets_;
    DeviceBuffer<int> device_incoming_edges_;
    DeviceBuffer<int> device_targets_;
    DeviceBuffer<unsigned int> device_delays_;
    DeviceBuffer<unsigned int> device_code_offsets_;
    DeviceBuffer<unsigned int> device_code_counts_;
    DeviceBuffer<unsigned long long> device_edge_code_window_ticks_;
    DeviceBuffer<float> device_edge_code_similarity_thresholds_;
    DeviceBuffer<unsigned int> device_edge_code_max_patterns_;
    DeviceBuffer<int> device_compartments_;
    DeviceBuffer<int> device_receptors_;
    DeviceBuffer<int> device_plasticity_;
    DeviceBuffer<float> device_initial_weights_;
    DeviceBuffer<float> device_max_weights_;
    DeviceBuffer<float> device_thresholds_;
    DeviceBuffer<unsigned long long> device_temporal_window_ticks_;
    DeviceBuffer<float> device_temporal_similarity_thresholds_;
    DeviceBuffer<unsigned int> device_temporal_max_patterns_;

    DeviceBuffer<float> device_membrane_;
    DeviceBuffer<float> device_soma_;
    DeviceBuffer<float> device_basal_;
    DeviceBuffer<float> device_apical_;
    DeviceBuffer<float> device_inhibitory_;
    DeviceBuffer<float> device_soma_tau_;
    DeviceBuffer<float> device_basal_tau_;
    DeviceBuffer<float> device_apical_tau_;
    DeviceBuffer<float> device_inhibitory_tau_;
    DeviceBuffer<unsigned long long> device_soma_ticks_;
    DeviceBuffer<unsigned long long> device_basal_ticks_;
    DeviceBuffer<unsigned long long> device_apical_ticks_;
    DeviceBuffer<unsigned long long> device_inhibitory_ticks_;
    DeviceBuffer<unsigned long long> device_spike_counts_;
    DeviceBuffer<unsigned long long> device_last_fired_ticks_;
    DeviceBuffer<float> device_weights_;
    DeviceBuffer<unsigned long long> device_last_pre_ticks_;
    DeviceBuffer<unsigned long long> device_last_post_ticks_;
    DeviceBuffer<unsigned long long> device_pre_counts_;
    DeviceBuffer<unsigned long long> device_plasticity_updates_;
    DeviceBuffer<float> device_last_weight_delta_;
    DeviceBuffer<int> device_neuron_locks_;
    DeviceBuffer<int> device_fired_flags_;
    DeviceBuffer<int> device_scheduled_edges_;
    DeviceBuffer<int> device_scheduled_counts_;
    DeviceBuffer<int> device_fired_by_tick_;
    DeviceBuffer<int> device_fired_counts_;
    DeviceBuffer<int> device_external_neurons_;
    DeviceBuffer<float> device_external_weights_;
    DeviceBuffer<unsigned long long> device_temporal_window_start_;
    DeviceBuffer<unsigned int> device_temporal_observed_offsets_;
    DeviceBuffer<unsigned int> device_temporal_observed_counts_;
    DeviceBuffer<unsigned int> device_temporal_reference_offsets_;
    DeviceBuffer<unsigned int> device_temporal_reference_counts_;
    DeviceBuffer<unsigned int> device_temporal_learned_counts_;
    DeviceBuffer<unsigned long long> device_temporal_match_counts_;
    DeviceBuffer<int> device_temporal_last_match_;
    DeviceBuffer<unsigned long long> device_edge_code_window_start_;
    DeviceBuffer<unsigned int> device_edge_code_observed_offsets_;
    DeviceBuffer<unsigned int> device_edge_code_observed_counts_;
    DeviceBuffer<unsigned int> device_edge_code_reference_offsets_;
    DeviceBuffer<unsigned int> device_edge_code_reference_counts_;
    DeviceBuffer<unsigned int> device_edge_code_learned_counts_;
    DeviceBuffer<unsigned long long> device_edge_code_match_counts_;
    DeviceBuffer<int> device_edge_code_last_match_;
    DeviceBuffer<unsigned long long> device_metrics_;

    DeviceBuffer<float> batched_membrane_;
    DeviceBuffer<float> batched_soma_;
    DeviceBuffer<float> batched_basal_;
    DeviceBuffer<float> batched_apical_;
    DeviceBuffer<float> batched_inhibitory_;
    DeviceBuffer<float> batched_soma_tau_;
    DeviceBuffer<float> batched_basal_tau_;
    DeviceBuffer<float> batched_apical_tau_;
    DeviceBuffer<float> batched_inhibitory_tau_;
    DeviceBuffer<unsigned long long> batched_soma_ticks_;
    DeviceBuffer<unsigned long long> batched_basal_ticks_;
    DeviceBuffer<unsigned long long> batched_apical_ticks_;
    DeviceBuffer<unsigned long long> batched_inhibitory_ticks_;
    DeviceBuffer<unsigned long long> batched_spike_counts_;
    DeviceBuffer<unsigned long long> batched_last_fired_ticks_;
    DeviceBuffer<float> batched_weights_;
    DeviceBuffer<unsigned long long> batched_last_pre_ticks_;
    DeviceBuffer<unsigned long long> batched_last_post_ticks_;
    DeviceBuffer<unsigned long long> batched_pre_counts_;
    DeviceBuffer<unsigned long long> batched_plasticity_updates_;
    DeviceBuffer<float> batched_last_weight_delta_;
    DeviceBuffer<int> batched_neuron_locks_;
    DeviceBuffer<int> batched_fired_flags_;
    DeviceBuffer<int> batched_scheduled_edges_;
    DeviceBuffer<int> batched_scheduled_counts_;
    DeviceBuffer<int> batched_fired_by_tick_;
    DeviceBuffer<int> batched_fired_counts_;
    DeviceBuffer<int> batched_input_neurons_;
    DeviceBuffer<float> batched_input_weights_;
    DeviceBuffer<int> batched_input_counts_;
    DeviceBuffer<unsigned long long> batched_temporal_window_start_;
    DeviceBuffer<unsigned int> batched_temporal_observed_offsets_;
    DeviceBuffer<unsigned int> batched_temporal_observed_counts_;
    DeviceBuffer<unsigned int> batched_temporal_reference_offsets_;
    DeviceBuffer<unsigned int> batched_temporal_reference_counts_;
    DeviceBuffer<unsigned int> batched_temporal_learned_counts_;
    DeviceBuffer<unsigned long long> batched_temporal_match_counts_;
    DeviceBuffer<int> batched_temporal_last_match_;
    DeviceBuffer<unsigned long long> batched_edge_code_window_start_;
    DeviceBuffer<unsigned int> batched_edge_code_observed_offsets_;
    DeviceBuffer<unsigned int> batched_edge_code_observed_counts_;
    DeviceBuffer<unsigned int> batched_edge_code_reference_offsets_;
    DeviceBuffer<unsigned int> batched_edge_code_reference_counts_;
    DeviceBuffer<unsigned int> batched_edge_code_learned_counts_;
    DeviceBuffer<unsigned long long> batched_edge_code_match_counts_;
    DeviceBuffer<int> batched_edge_code_last_match_;
    DeviceBuffer<unsigned long long> batched_metrics_;

    DeviceBuffer<int> feedforward_input_neurons_;
    DeviceBuffer<float> feedforward_input_weights_;
    DeviceBuffer<int> feedforward_input_counts_;
    DeviceBuffer<int> feedforward_dense_to_readout_;
    DeviceBuffer<float> feedforward_readout_thresholds_;
    DeviceBuffer<float> feedforward_scores_;
    DeviceBuffer<unsigned long long> feedforward_spike_counts_;

    DeviceBuffer<int> recurrent_input_neurons_;
    DeviceBuffer<float> recurrent_input_weights_;
    DeviceBuffer<int> recurrent_input_counts_;
    DeviceBuffer<int> recurrent_dense_to_hidden_;
    DeviceBuffer<int> recurrent_hidden_dense_neurons_;
    DeviceBuffer<int> recurrent_dense_to_readout_;
    DeviceBuffer<int> recurrent_readout_dense_neurons_;
    DeviceBuffer<float> recurrent_readout_thresholds_;
    DeviceBuffer<float> recurrent_hidden_events_;
    DeviceBuffer<float> recurrent_readout_events_;
    DeviceBuffer<int> recurrent_hidden_selected_;
    DeviceBuffer<float> recurrent_readout_scores_;
    DeviceBuffer<float> recurrent_readout_gates_;
    DeviceBuffer<int> recurrent_readout_selected_;
    DeviceBuffer<unsigned long long> recurrent_readout_spike_counts_;
};

CudaInferenceSession::CudaInferenceSession(const declarative::Connectome& connectome)
    : impl_(std::make_unique<Impl>(connectome)) {}

CudaInferenceSession::~CudaInferenceSession() = default;
CudaInferenceSession::CudaInferenceSession(CudaInferenceSession&&) noexcept = default;
CudaInferenceSession& CudaInferenceSession::operator=(CudaInferenceSession&&) noexcept = default;

bool CudaInferenceSession::available() const noexcept {
    return impl_ != nullptr && impl_->available();
}

CudaInferenceBatchResult CudaInferenceSession::run_batch(
    const std::vector<CudaInferenceSample>& samples,
    const CudaInferenceOptions& options) {
    return impl_->run_batch(samples, options);
}

CudaInferenceBatchResult CudaInferenceSession::run_feedforward_batch(
    const std::vector<CudaInferenceSample>& samples,
    const CudaInferenceOptions& options) {
    return impl_->run_feedforward_batch(samples, options);
}

CudaInferenceBatchResult CudaInferenceSession::run_recurrent_feedforward_batch(
    const std::vector<CudaInferenceSample>& samples,
    const CudaRecurrentFeedforwardOptions& options) {
    return impl_->run_recurrent_feedforward_batch(samples, options);
}

void CudaInferenceSession::reset_state() {
    impl_->reset_state();
}

std::string_view CudaBackend::name() const noexcept {
    return "cuda";
}

bool CudaBackend::available() const noexcept {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::string CudaBackend::availability_status() const {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        return cudaGetErrorString(status);
    }
    if (device_count == 0) {
        return "no CUDA-capable devices reported by the runtime";
    }
    return std::to_string(device_count) + " CUDA device(s) available";
}

CudaPropagationResult CudaBackend::run_resident_propagation(
    const declarative::Connectome& connectome,
    const std::vector<core::NeuronId>& initially_active,
    std::uint32_t max_steps) const {
    if (!available()) {
        return {};
    }
    if (connectome.neurons.empty() || initially_active.empty()) {
        return {.executed = true};
    }

    std::unordered_map<core::NeuronId, int> dense_index;
    dense_index.reserve(connectome.neurons.size());
    std::vector<float> thresholds(connectome.neurons.size(), 1.0F);
    std::vector<unsigned long long> temporal_window_ticks(connectome.neurons.size(), 500);
    std::vector<float> temporal_similarity_thresholds(connectome.neurons.size(), 0.93F);
    std::vector<unsigned int> temporal_max_patterns(connectome.neurons.size(), 1);
    for (std::size_t i = 0; i < connectome.neurons.size(); ++i) {
        dense_index[connectome.neurons[i].id] = static_cast<int>(i);
        thresholds[i] = connectome.neurons[i].params.threshold;
        temporal_window_ticks[i] = connectome.neurons[i].params.pattern_window_ticks;
        temporal_similarity_thresholds[i] = connectome.neurons[i].params.similarity_threshold;
        temporal_max_patterns[i] = static_cast<unsigned int>(
            std::min<std::size_t>(connectome.neurons[i].params.max_reference_patterns, 1));
    }

    std::vector<std::vector<int>> outgoing(connectome.neurons.size());
    std::vector<std::vector<int>> incoming(connectome.neurons.size());
    std::vector<int> edge_sources;
    std::vector<int> edge_targets;
    std::vector<unsigned int> edge_delays;
    std::vector<int> edge_compartments;
    std::vector<int> edge_receptors;
    std::vector<int> edge_plasticity;
    std::vector<unsigned int> edge_code_offsets;
    std::vector<unsigned int> edge_code_counts;
    std::vector<unsigned long long> edge_code_window_ticks;
    std::vector<float> edge_code_similarity_thresholds;
    std::vector<unsigned int> edge_code_max_patterns;
    std::vector<float> edge_weights;
    std::vector<float> edge_max_weights;
    edge_sources.reserve(connectome.synapses.size());
    edge_targets.reserve(connectome.synapses.size());
    edge_delays.reserve(connectome.synapses.size());
    edge_compartments.reserve(connectome.synapses.size());
    edge_receptors.reserve(connectome.synapses.size());
    edge_plasticity.reserve(connectome.synapses.size());
    edge_code_offsets.reserve(connectome.synapses.size() * spike_code_capacity);
    edge_code_counts.reserve(connectome.synapses.size());
    edge_code_window_ticks.reserve(connectome.synapses.size());
    edge_code_similarity_thresholds.reserve(connectome.synapses.size());
    edge_code_max_patterns.reserve(connectome.synapses.size());
    edge_weights.reserve(connectome.synapses.size());
    edge_max_weights.reserve(connectome.synapses.size());

    for (const auto& synapse : connectome.synapses) {
        const auto source = dense_index.find(synapse.source);
        const auto target = dense_index.find(synapse.target);
        if (source == dense_index.end() || target == dense_index.end()) {
            continue;
        }
        const auto edge = static_cast<int>(edge_targets.size());
        edge_sources.push_back(source->second);
        edge_targets.push_back(target->second);
        edge_delays.push_back(std::max(1U, synapse.delay_ticks));
        const auto code_count = static_cast<unsigned int>(
            std::min<std::size_t>(
                synapse.spike_code_offsets.empty() ? 1 : synapse.spike_code_offsets.size(),
                spike_code_capacity));
        edge_code_counts.push_back(code_count);
        for (int code = 0; code < spike_code_capacity; ++code) {
            if (code < static_cast<int>(code_count) && !synapse.spike_code_offsets.empty()) {
                edge_code_offsets.push_back(synapse.spike_code_offsets[static_cast<std::size_t>(code)]);
            } else {
                edge_code_offsets.push_back(0U);
            }
        }
        edge_compartments.push_back(static_cast<int>(synapse.compartment));
        edge_receptors.push_back(static_cast<int>(synapse.receptor));
        edge_plasticity.push_back(synapse.plasticity_enabled ? 1 : 0);
        edge_weights.push_back(synapse.weight);
        edge_max_weights.push_back(synapse.max_weight);
        edge_code_window_ticks.push_back(temporal_window_ticks[static_cast<std::size_t>(target->second)]);
        edge_code_similarity_thresholds.push_back(
            temporal_similarity_thresholds[static_cast<std::size_t>(target->second)]);
        edge_code_max_patterns.push_back(1U);
        outgoing[source->second].push_back(edge);
        incoming[target->second].push_back(edge);
    }

    std::vector<int> outgoing_offsets(connectome.neurons.size() + 1, 0);
    std::vector<int> outgoing_edges;
    std::vector<int> incoming_offsets(connectome.neurons.size() + 1, 0);
    std::vector<int> incoming_edges;
    for (std::size_t neuron = 0; neuron < outgoing.size(); ++neuron) {
        outgoing_offsets[neuron] = static_cast<int>(outgoing_edges.size());
        for (const auto edge : outgoing[neuron]) {
            outgoing_edges.push_back(edge);
        }
    }
    outgoing_offsets.back() = static_cast<int>(outgoing_edges.size());
    for (std::size_t neuron = 0; neuron < incoming.size(); ++neuron) {
        incoming_offsets[neuron] = static_cast<int>(incoming_edges.size());
        for (const auto edge : incoming[neuron]) {
            incoming_edges.push_back(edge);
        }
    }
    incoming_offsets.back() = static_cast<int>(incoming_edges.size());

    std::vector<int> host_active;
    host_active.reserve(initially_active.size());
    std::unordered_set<core::NeuronId> seen_active;
    for (const auto id : initially_active) {
        if (const auto found = dense_index.find(id);
            found != dense_index.end() && seen_active.insert(id).second) {
            host_active.push_back(found->second);
        }
    }
    if (host_active.empty()) {
        return {.executed = true};
    }

    auto* device_offsets = copy_to_device(outgoing_offsets);
    auto* device_outgoing_edges = copy_to_device(outgoing_edges);
    auto* device_incoming_offsets = copy_to_device(incoming_offsets);
    auto* device_incoming_edges = copy_to_device(incoming_edges);
    auto* device_targets = copy_to_device(edge_targets);
    auto* device_delays = copy_to_device(edge_delays);
    auto* device_code_offsets = copy_to_device(edge_code_offsets);
    auto* device_code_counts = copy_to_device(edge_code_counts);
    auto* device_edge_code_window_ticks = copy_to_device(edge_code_window_ticks);
    auto* device_edge_code_similarity_thresholds = copy_to_device(edge_code_similarity_thresholds);
    auto* device_edge_code_max_patterns = copy_to_device(edge_code_max_patterns);
    auto* device_compartments = copy_to_device(edge_compartments);
    auto* device_receptors = copy_to_device(edge_receptors);
    auto* device_plasticity = copy_to_device(edge_plasticity);
    auto* device_weights = copy_to_device(edge_weights);
    auto* device_max_weights = copy_to_device(edge_max_weights);
    auto* device_thresholds = copy_to_device(thresholds);
    auto* device_temporal_window_ticks = copy_to_device(temporal_window_ticks);
    auto* device_temporal_similarity_thresholds = copy_to_device(temporal_similarity_thresholds);
    auto* device_temporal_max_patterns = copy_to_device(temporal_max_patterns);

    float* device_membrane = nullptr;
    float* device_soma = nullptr;
    float* device_basal = nullptr;
    float* device_apical = nullptr;
    float* device_inhibitory = nullptr;
    float* device_soma_tau = nullptr;
    float* device_basal_tau = nullptr;
    float* device_apical_tau = nullptr;
    float* device_inhibitory_tau = nullptr;
    unsigned long long* device_soma_ticks = nullptr;
    unsigned long long* device_basal_ticks = nullptr;
    unsigned long long* device_apical_ticks = nullptr;
    unsigned long long* device_inhibitory_ticks = nullptr;
    unsigned long long* device_spike_counts = nullptr;
    unsigned long long* device_last_fired_ticks = nullptr;
    unsigned long long* device_last_pre_ticks = nullptr;
    unsigned long long* device_last_post_ticks = nullptr;
    unsigned long long* device_pre_counts = nullptr;
    unsigned long long* device_plasticity_updates = nullptr;
    float* device_last_weight_delta = nullptr;
    int* device_neuron_locks = nullptr;
    int* device_fired_flags = nullptr;
    int* device_scheduled_edges = nullptr;
    int* device_scheduled_counts = nullptr;
    int* device_fired_by_tick = nullptr;
    int* device_fired_counts = nullptr;
    unsigned long long* device_temporal_window_start = nullptr;
    unsigned int* device_temporal_observed_offsets = nullptr;
    unsigned int* device_temporal_observed_counts = nullptr;
    unsigned int* device_temporal_reference_offsets = nullptr;
    unsigned int* device_temporal_reference_counts = nullptr;
    unsigned int* device_temporal_learned_counts = nullptr;
    unsigned long long* device_temporal_match_counts = nullptr;
    int* device_temporal_last_match = nullptr;
    unsigned long long* device_edge_code_window_start = nullptr;
    unsigned int* device_edge_code_observed_offsets = nullptr;
    unsigned int* device_edge_code_observed_counts = nullptr;
    unsigned int* device_edge_code_reference_offsets = nullptr;
    unsigned int* device_edge_code_reference_counts = nullptr;
    unsigned int* device_edge_code_learned_counts = nullptr;
    unsigned long long* device_edge_code_match_counts = nullptr;
    int* device_edge_code_last_match = nullptr;
    unsigned long long* device_metrics = nullptr;

    const auto neuron_count = connectome.neurons.size();
    const auto edge_count = edge_targets.size();
    const auto tick_count = static_cast<std::size_t>(max_steps) + 1;
    const auto event_capacity = std::max<std::size_t>(1, edge_count);
    const auto fired_capacity = std::max<std::size_t>({1, neuron_count, host_active.size()});
    check_cuda(cudaMalloc(&device_membrane, neuron_count * sizeof(float)), "cudaMalloc membrane");
    check_cuda(cudaMemset(device_membrane, 0, neuron_count * sizeof(float)), "cudaMemset membrane");
    check_cuda(cudaMalloc(&device_soma, neuron_count * sizeof(float)), "cudaMalloc soma");
    check_cuda(cudaMalloc(&device_basal, neuron_count * sizeof(float)), "cudaMalloc basal");
    check_cuda(cudaMalloc(&device_apical, neuron_count * sizeof(float)), "cudaMalloc apical");
    check_cuda(cudaMalloc(&device_inhibitory, neuron_count * sizeof(float)), "cudaMalloc inhibitory");
    check_cuda(cudaMemset(device_soma, 0, neuron_count * sizeof(float)), "cudaMemset soma");
    check_cuda(cudaMemset(device_basal, 0, neuron_count * sizeof(float)), "cudaMemset basal");
    check_cuda(cudaMemset(device_apical, 0, neuron_count * sizeof(float)), "cudaMemset apical");
    check_cuda(cudaMemset(device_inhibitory, 0, neuron_count * sizeof(float)), "cudaMemset inhibitory");

    std::vector<float> soma_tau(neuron_count, 5.0F);
    std::vector<float> basal_tau(neuron_count, 5.0F);
    std::vector<float> apical_tau(neuron_count, 100.0F);
    std::vector<float> inhibitory_tau(neuron_count, 10.0F);
    std::vector<unsigned long long> zero_ticks(neuron_count, 0);
    std::vector<unsigned long long> invalid_neuron_ticks(neuron_count, invalid_tick);
    std::vector<unsigned long long> zero_neuron_counts(neuron_count, 0);
    std::vector<unsigned int> zero_neuron_u32(neuron_count, 0);
    std::vector<unsigned int> zero_temporal_offsets(neuron_count * temporal_capacity, 0);
    std::vector<int> zero_neuron_i32(neuron_count, 0);
    std::vector<unsigned long long> invalid_edge_ticks(edge_count, invalid_tick);
    std::vector<unsigned long long> zero_edge_counts(edge_count, 0);
    std::vector<unsigned int> zero_edge_u32(edge_count, 0);
    std::vector<int> zero_edge_i32(edge_count, 0);
    std::vector<unsigned int> zero_edge_temporal_offsets(edge_count * temporal_capacity, 0);
    std::vector<float> zero_edge_deltas(edge_count, 0.0F);

    device_soma_tau = copy_to_device(soma_tau);
    device_basal_tau = copy_to_device(basal_tau);
    device_apical_tau = copy_to_device(apical_tau);
    device_inhibitory_tau = copy_to_device(inhibitory_tau);
    device_soma_ticks = copy_to_device(zero_ticks);
    device_basal_ticks = copy_to_device(zero_ticks);
    device_apical_ticks = copy_to_device(zero_ticks);
    device_inhibitory_ticks = copy_to_device(zero_ticks);
    device_spike_counts = copy_to_device(zero_neuron_counts);
    device_last_fired_ticks = copy_to_device(invalid_neuron_ticks);
    device_temporal_window_start = copy_to_device(zero_neuron_counts);
    device_temporal_observed_offsets = copy_to_device(zero_temporal_offsets);
    device_temporal_observed_counts = copy_to_device(zero_neuron_u32);
    device_temporal_reference_offsets = copy_to_device(zero_temporal_offsets);
    device_temporal_reference_counts = copy_to_device(zero_neuron_u32);
    device_temporal_learned_counts = copy_to_device(zero_neuron_u32);
    device_temporal_match_counts = copy_to_device(zero_neuron_counts);
    device_temporal_last_match = copy_to_device(zero_neuron_i32);
    device_edge_code_window_start = copy_to_device(zero_edge_counts);
    device_edge_code_observed_offsets = copy_to_device(zero_edge_temporal_offsets);
    device_edge_code_observed_counts = copy_to_device(zero_edge_u32);
    device_edge_code_reference_offsets = copy_to_device(zero_edge_temporal_offsets);
    device_edge_code_reference_counts = copy_to_device(zero_edge_u32);
    device_edge_code_learned_counts = copy_to_device(zero_edge_u32);
    device_edge_code_match_counts = copy_to_device(zero_edge_counts);
    device_edge_code_last_match = copy_to_device(zero_edge_i32);
    device_last_pre_ticks = copy_to_device(invalid_edge_ticks);
    device_last_post_ticks = copy_to_device(invalid_edge_ticks);
    device_pre_counts = copy_to_device(zero_edge_counts);
    device_plasticity_updates = copy_to_device(zero_edge_counts);
    device_last_weight_delta = copy_to_device(zero_edge_deltas);

    check_cuda(cudaMalloc(&device_neuron_locks, neuron_count * sizeof(int)), "cudaMalloc neuron locks");
    check_cuda(cudaMemset(device_neuron_locks, 0, neuron_count * sizeof(int)), "cudaMemset neuron locks");
    check_cuda(cudaMalloc(&device_fired_flags, neuron_count * sizeof(int)), "cudaMalloc fired flags");
    check_cuda(
        cudaMalloc(&device_scheduled_edges, tick_count * event_capacity * sizeof(int)),
        "cudaMalloc scheduled edges");
    check_cuda(
        cudaMalloc(&device_scheduled_counts, tick_count * sizeof(int)),
        "cudaMalloc scheduled counts");
    check_cuda(cudaMemset(device_scheduled_counts, 0, tick_count * sizeof(int)), "cudaMemset scheduled counts");
    check_cuda(
        cudaMalloc(&device_fired_by_tick, tick_count * fired_capacity * sizeof(int)),
        "cudaMalloc fired by tick");
    check_cuda(cudaMalloc(&device_fired_counts, tick_count * sizeof(int)), "cudaMalloc fired counts");
    check_cuda(cudaMemset(device_fired_counts, 0, tick_count * sizeof(int)), "cudaMemset fired counts");
    check_cuda(cudaMalloc(&device_metrics, metric_count * sizeof(unsigned long long)), "cudaMalloc metrics");
    check_cuda(cudaMemset(device_metrics, 0, metric_count * sizeof(unsigned long long)), "cudaMemset metrics");
    check_cuda(
        cudaMemcpy(
            device_fired_by_tick,
            host_active.data(),
            host_active.size() * sizeof(int),
            cudaMemcpyHostToDevice),
        "cudaMemcpy initial fired");
    const auto initial_count = static_cast<int>(host_active.size());
    check_cuda(
        cudaMemcpy(device_fired_counts, &initial_count, sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy initial fired count");

    const auto threads = 256;
    if (initial_count > 0) {
        const auto blocks = (initial_count + threads - 1) / threads;
        initialize_initial_fired_kernel<<<blocks, threads>>>(
            device_fired_by_tick,
            initial_count,
            device_incoming_offsets,
            device_incoming_edges,
            device_spike_counts,
            device_last_fired_ticks,
            device_last_post_ticks);
        check_cuda(cudaGetLastError(), "initialize_initial_fired_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "initialize_initial_fired_kernel synchronize");
    }

    std::uint64_t delivered = static_cast<std::uint64_t>(initial_count);
    std::uint64_t fired = static_cast<std::uint64_t>(initial_count);
    CudaDebugMetrics debug;

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t tick = 0; tick <= max_steps; ++tick) {
        ++debug.ticks_processed;
        check_cuda(cudaMemset(device_fired_flags, 0, neuron_count * sizeof(int)), "cudaMemset flags");

        int fired_before = 0;
        check_cuda(
            cudaMemcpy(&fired_before, device_fired_counts + tick, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy fired before");
        if (fired_before > 0) {
            const auto mark_blocks = (fired_before + threads - 1) / threads;
            mark_fired_flags_kernel<<<mark_blocks, threads>>>(
                device_fired_by_tick + static_cast<std::size_t>(tick) * fired_capacity,
                fired_before,
                device_fired_flags);
            check_cuda(cudaGetLastError(), "mark_fired_flags_kernel launch");
            check_cuda(cudaDeviceSynchronize(), "mark_fired_flags_kernel synchronize");
        }

        int event_count = 0;
        check_cuda(
            cudaMemcpy(&event_count, device_scheduled_counts + tick, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy event count");
        debug.max_scheduled_events_per_tick = std::max(
            debug.max_scheduled_events_per_tick,
            static_cast<std::uint64_t>(std::max(event_count, 0)));
        if (event_count > 0) {
            const auto clamped_event_count = std::min(event_count, static_cast<int>(event_capacity));
            if (event_count > clamped_event_count) {
                debug.dropped_events += static_cast<std::uint64_t>(event_count - clamped_event_count);
            }
            const auto event_blocks = (clamped_event_count + threads - 1) / threads;
            process_synapse_events_kernel<<<event_blocks, threads>>>(
                device_scheduled_edges + static_cast<std::size_t>(tick) * event_capacity,
                clamped_event_count,
                device_fired_by_tick + static_cast<std::size_t>(tick) * fired_capacity,
                device_fired_counts + tick,
                static_cast<int>(fired_capacity),
                tick,
                device_targets,
                device_delays,
                device_code_counts,
                device_compartments,
                device_receptors,
                device_plasticity,
                device_incoming_offsets,
                device_incoming_edges,
                device_weights,
                device_max_weights,
                device_last_pre_ticks,
                device_last_post_ticks,
                device_pre_counts,
                device_plasticity_updates,
                device_last_weight_delta,
                device_thresholds,
                device_membrane,
                device_soma,
                device_basal,
                device_apical,
                device_inhibitory,
                device_soma_ticks,
                device_basal_ticks,
                device_apical_ticks,
                device_inhibitory_ticks,
                device_soma_tau,
                device_basal_tau,
                device_apical_tau,
                device_inhibitory_tau,
                device_spike_counts,
                device_last_fired_ticks,
                device_neuron_locks,
                device_fired_flags,
                device_temporal_window_ticks,
                device_temporal_similarity_thresholds,
                device_temporal_max_patterns,
                device_temporal_window_start,
                device_temporal_observed_offsets,
                device_temporal_observed_counts,
                device_temporal_reference_offsets,
                device_temporal_reference_counts,
                device_temporal_learned_counts,
                device_temporal_match_counts,
                device_temporal_last_match,
                device_edge_code_window_ticks,
                device_edge_code_similarity_thresholds,
                device_edge_code_max_patterns,
                device_edge_code_window_start,
                device_edge_code_observed_offsets,
                device_edge_code_observed_counts,
                device_edge_code_reference_offsets,
                device_edge_code_reference_counts,
                device_edge_code_learned_counts,
                device_edge_code_match_counts,
                device_edge_code_last_match,
                device_metrics);
            check_cuda(cudaGetLastError(), "process_synapse_events_kernel launch");
            check_cuda(cudaDeviceSynchronize(), "process_synapse_events_kernel synchronize");
            delivered += static_cast<std::uint64_t>(clamped_event_count);
        }

        int fired_after = 0;
        check_cuda(
            cudaMemcpy(&fired_after, device_fired_counts + tick, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy fired after");
        debug.max_fired_neurons_per_tick = std::max(
            debug.max_fired_neurons_per_tick,
            static_cast<std::uint64_t>(std::max(fired_after, 0)));
        fired += static_cast<std::uint64_t>(std::max(0, fired_after - fired_before));

        if (fired_after > 0 && tick < max_steps && edge_count > 0) {
            const auto expand_blocks = (fired_after + threads - 1) / threads;
            expand_fired_kernel<<<expand_blocks, threads>>>(
                device_fired_by_tick + static_cast<std::size_t>(tick) * fired_capacity,
                std::min(fired_after, static_cast<int>(fired_capacity)),
                device_offsets,
                device_outgoing_edges,
                device_delays,
                device_code_offsets,
                device_code_counts,
                device_scheduled_edges,
                device_scheduled_counts,
                static_cast<int>(event_capacity),
                tick,
                max_steps,
                device_metrics);
            check_cuda(cudaGetLastError(), "expand_fired_kernel launch");
            check_cuda(cudaDeviceSynchronize(), "expand_fired_kernel synchronize");
        }
    }
    const auto end = std::chrono::steady_clock::now();

    std::vector<float> final_weights(edge_count, 0.0F);
    if (edge_count > 0) {
        check_cuda(
            cudaMemcpy(
                final_weights.data(),
                device_weights,
                final_weights.size() * sizeof(float),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy final weights");
    }
    std::vector<float> final_membrane(neuron_count, 0.0F);
    check_cuda(
        cudaMemcpy(
            final_membrane.data(),
            device_membrane,
            final_membrane.size() * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy final membrane");
    std::vector<unsigned long long> final_spikes_raw(neuron_count, 0);
    check_cuda(
        cudaMemcpy(
            final_spikes_raw.data(),
            device_spike_counts,
            final_spikes_raw.size() * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy final spike counts");
    std::vector<std::uint64_t> final_spikes;
    final_spikes.reserve(final_spikes_raw.size());
    for (const auto value : final_spikes_raw) {
        final_spikes.push_back(static_cast<std::uint64_t>(value));
    }
    std::vector<unsigned long long> final_temporal_matches_raw(neuron_count, 0);
    check_cuda(
        cudaMemcpy(
            final_temporal_matches_raw.data(),
            device_temporal_match_counts,
            final_temporal_matches_raw.size() * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy final temporal match counts");
    std::vector<std::uint64_t> final_temporal_matches;
    final_temporal_matches.reserve(final_temporal_matches_raw.size());
    for (const auto value : final_temporal_matches_raw) {
        final_temporal_matches.push_back(static_cast<std::uint64_t>(value));
    }
    std::vector<unsigned int> final_temporal_learned(neuron_count, 0);
    check_cuda(
        cudaMemcpy(
            final_temporal_learned.data(),
            device_temporal_learned_counts,
            final_temporal_learned.size() * sizeof(unsigned int),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy final temporal learned counts");
    std::vector<unsigned long long> final_synapse_code_matches_raw(edge_count, 0);
    if (edge_count > 0) {
        check_cuda(
            cudaMemcpy(
                final_synapse_code_matches_raw.data(),
                device_edge_code_match_counts,
                final_synapse_code_matches_raw.size() * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy final synapse code match counts");
    }
    std::vector<std::uint64_t> final_synapse_code_matches;
    final_synapse_code_matches.reserve(final_synapse_code_matches_raw.size());
    for (const auto value : final_synapse_code_matches_raw) {
        final_synapse_code_matches.push_back(static_cast<std::uint64_t>(value));
    }
    std::vector<unsigned int> final_synapse_code_learned(edge_count, 0);
    if (edge_count > 0) {
        check_cuda(
            cudaMemcpy(
                final_synapse_code_learned.data(),
                device_edge_code_learned_counts,
                final_synapse_code_learned.size() * sizeof(unsigned int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy final synapse code learned counts");
    }

    std::vector<unsigned long long> raw_metrics(metric_count, 0);
    check_cuda(
        cudaMemcpy(
            raw_metrics.data(),
            device_metrics,
            raw_metrics.size() * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy debug metrics");
    debug.scheduled_event_requests = raw_metrics[metric_scheduled_event_requests];
    debug.scheduled_events = raw_metrics[metric_scheduled_events];
    debug.processed_events = raw_metrics[metric_processed_events];
    debug.dropped_events += raw_metrics[metric_dropped_events];
    debug.fired_appends = raw_metrics[metric_fired_appends];
    debug.fired_overflow = raw_metrics[metric_fired_overflow];
    debug.lock_spin_iterations = raw_metrics[metric_lock_spin_iterations];
    debug.lock_timeouts = raw_metrics[metric_lock_timeouts];
    debug.stdp_updates = raw_metrics[metric_stdp_updates];
    debug.stdp_ltp = raw_metrics[metric_stdp_ltp];
    debug.stdp_ltd = raw_metrics[metric_stdp_ltd];
    debug.receptor_ampa_events = raw_metrics[metric_receptor_ampa_events];
    debug.receptor_nmda_events = raw_metrics[metric_receptor_nmda_events];
    debug.receptor_gaba_a_events = raw_metrics[metric_receptor_gaba_a_events];
    debug.receptor_gaba_b_events = raw_metrics[metric_receptor_gaba_b_events];
    debug.dendritic_integrations = raw_metrics[metric_dendritic_integrations];
    debug.post_plasticity_scans = raw_metrics[metric_post_plasticity_scans];
    debug.temporal_observations = raw_metrics[metric_temporal_observations];
    debug.temporal_patterns_learned = raw_metrics[metric_temporal_patterns_learned];
    debug.temporal_matches = raw_metrics[metric_temporal_matches];

    cudaFree(device_offsets);
    cudaFree(device_outgoing_edges);
    cudaFree(device_incoming_offsets);
    cudaFree(device_incoming_edges);
    cudaFree(device_targets);
    cudaFree(device_delays);
    cudaFree(device_code_offsets);
    cudaFree(device_code_counts);
    cudaFree(device_edge_code_window_ticks);
    cudaFree(device_edge_code_similarity_thresholds);
    cudaFree(device_edge_code_max_patterns);
    cudaFree(device_compartments);
    cudaFree(device_receptors);
    cudaFree(device_plasticity);
    cudaFree(device_weights);
    cudaFree(device_max_weights);
    cudaFree(device_thresholds);
    cudaFree(device_temporal_window_ticks);
    cudaFree(device_temporal_similarity_thresholds);
    cudaFree(device_temporal_max_patterns);
    cudaFree(device_membrane);
    cudaFree(device_soma);
    cudaFree(device_basal);
    cudaFree(device_apical);
    cudaFree(device_inhibitory);
    cudaFree(device_soma_tau);
    cudaFree(device_basal_tau);
    cudaFree(device_apical_tau);
    cudaFree(device_inhibitory_tau);
    cudaFree(device_soma_ticks);
    cudaFree(device_basal_ticks);
    cudaFree(device_apical_ticks);
    cudaFree(device_inhibitory_ticks);
    cudaFree(device_spike_counts);
    cudaFree(device_last_fired_ticks);
    cudaFree(device_temporal_window_start);
    cudaFree(device_temporal_observed_offsets);
    cudaFree(device_temporal_observed_counts);
    cudaFree(device_temporal_reference_offsets);
    cudaFree(device_temporal_reference_counts);
    cudaFree(device_temporal_learned_counts);
    cudaFree(device_temporal_match_counts);
    cudaFree(device_temporal_last_match);
    cudaFree(device_edge_code_window_start);
    cudaFree(device_edge_code_observed_offsets);
    cudaFree(device_edge_code_observed_counts);
    cudaFree(device_edge_code_reference_offsets);
    cudaFree(device_edge_code_reference_counts);
    cudaFree(device_edge_code_learned_counts);
    cudaFree(device_edge_code_match_counts);
    cudaFree(device_edge_code_last_match);
    cudaFree(device_last_pre_ticks);
    cudaFree(device_last_post_ticks);
    cudaFree(device_pre_counts);
    cudaFree(device_plasticity_updates);
    cudaFree(device_last_weight_delta);
    cudaFree(device_neuron_locks);
    cudaFree(device_fired_flags);
    cudaFree(device_scheduled_edges);
    cudaFree(device_scheduled_counts);
    cudaFree(device_fired_by_tick);
    cudaFree(device_fired_counts);
    cudaFree(device_metrics);

    return {
        .executed = true,
        .delivered_spikes = delivered,
        .fired_spikes = fired,
        .elapsed_seconds = std::chrono::duration<double>(end - start).count(),
        .debug = debug,
        .final_synapse_weights = std::move(final_weights),
        .final_membrane_potentials = std::move(final_membrane),
        .final_neuron_spike_counts = std::move(final_spikes),
        .final_temporal_match_counts = std::move(final_temporal_matches),
        .final_temporal_learned_pattern_counts = std::move(final_temporal_learned),
        .final_synapse_code_match_counts = std::move(final_synapse_code_matches),
        .final_synapse_code_learned_pattern_counts = std::move(final_synapse_code_learned),
    };
}

} // namespace snncuda::backends
