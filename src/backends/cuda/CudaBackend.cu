#include "snncuda/backends/CudaBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime_api.h>

namespace {

constexpr unsigned long long invalid_tick = std::numeric_limits<unsigned long long>::max();
constexpr unsigned long long max_lock_spins = 10000000ULL;
constexpr int temporal_capacity = 8;

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
        atomicAdd(&metrics[metric_scheduled_event_requests], 1ULL);
        const auto delivery_tick = tick + synapse_delays[synapse];
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

__global__ void process_synapse_events_kernel(
    const int* scheduled_edges,
    int event_count,
    int* fired,
    int* fired_count,
    int fired_capacity,
    unsigned long long tick,
    const int* edge_targets,
    const unsigned int* edge_delays,
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

} // namespace

namespace snncuda::backends {

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
    std::vector<float> edge_weights;
    std::vector<float> edge_max_weights;
    edge_sources.reserve(connectome.synapses.size());
    edge_targets.reserve(connectome.synapses.size());
    edge_delays.reserve(connectome.synapses.size());
    edge_compartments.reserve(connectome.synapses.size());
    edge_receptors.reserve(connectome.synapses.size());
    edge_plasticity.reserve(connectome.synapses.size());
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
        edge_compartments.push_back(static_cast<int>(synapse.compartment));
        edge_receptors.push_back(static_cast<int>(synapse.receptor));
        edge_plasticity.push_back(synapse.plasticity_enabled ? 1 : 0);
        edge_weights.push_back(synapse.weight);
        edge_max_weights.push_back(synapse.max_weight);
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
    };
}

} // namespace snncuda::backends
