#include "snncuda/backends/CudaBackend.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_map>

#include <cuda_runtime_api.h>

namespace {

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

__global__ void propagate_kernel(
    const int* active,
    int active_count,
    const int* outgoing_offsets,
    const int* outgoing_targets,
    const float* outgoing_weights,
    const float* thresholds,
    float* membrane,
    int* fired_flags,
    int* next_active,
    int* next_count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= active_count) {
        return;
    }

    const auto source = active[index];
    for (auto edge = outgoing_offsets[source]; edge < outgoing_offsets[source + 1]; ++edge) {
        const auto target = outgoing_targets[edge];
        const auto weight = outgoing_weights[edge];
        const auto previous = atomicAdd(&membrane[target], weight);
        const auto updated = previous + weight;
        if (previous < thresholds[target] && updated >= thresholds[target]) {
            if (atomicExch(&fired_flags[target], 1) == 0) {
                membrane[target] = 0.0F;
                const auto output_index = atomicAdd(next_count, 1);
                next_active[output_index] = target;
            }
        }
    }
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
    for (std::size_t i = 0; i < connectome.neurons.size(); ++i) {
        dense_index[connectome.neurons[i].id] = static_cast<int>(i);
        thresholds[i] = connectome.neurons[i].params.threshold;
    }

    std::vector<std::vector<std::pair<int, float>>> outgoing(connectome.neurons.size());
    for (const auto& synapse : connectome.synapses) {
        const auto source = dense_index.find(synapse.source);
        const auto target = dense_index.find(synapse.target);
        if (source == dense_index.end() || target == dense_index.end()) {
            continue;
        }
        outgoing[source->second].push_back({target->second, synapse.weight});
    }

    std::vector<int> outgoing_offsets(connectome.neurons.size() + 1, 0);
    std::vector<int> outgoing_targets;
    std::vector<float> outgoing_weights;
    for (std::size_t neuron = 0; neuron < outgoing.size(); ++neuron) {
        outgoing_offsets[neuron] = static_cast<int>(outgoing_targets.size());
        for (const auto& [target, weight] : outgoing[neuron]) {
            outgoing_targets.push_back(target);
            outgoing_weights.push_back(weight);
        }
    }
    outgoing_offsets.back() = static_cast<int>(outgoing_targets.size());

    std::vector<int> host_active;
    host_active.reserve(initially_active.size());
    for (const auto id : initially_active) {
        if (const auto found = dense_index.find(id); found != dense_index.end()) {
            host_active.push_back(found->second);
        }
    }
    if (host_active.empty()) {
        return {.executed = true};
    }

    auto* device_offsets = copy_to_device(outgoing_offsets);
    auto* device_targets = copy_to_device(outgoing_targets);
    auto* device_weights = copy_to_device(outgoing_weights);
    auto* device_thresholds = copy_to_device(thresholds);

    float* device_membrane = nullptr;
    int* device_fired_flags = nullptr;
    int* device_active_a = nullptr;
    int* device_active_b = nullptr;
    int* device_next_count = nullptr;

    const auto neuron_count = connectome.neurons.size();
    check_cuda(cudaMalloc(&device_membrane, neuron_count * sizeof(float)), "cudaMalloc membrane");
    check_cuda(cudaMemset(device_membrane, 0, neuron_count * sizeof(float)), "cudaMemset membrane");
    check_cuda(cudaMalloc(&device_fired_flags, neuron_count * sizeof(int)), "cudaMalloc fired flags");
    check_cuda(cudaMalloc(&device_active_a, neuron_count * sizeof(int)), "cudaMalloc active A");
    check_cuda(cudaMalloc(&device_active_b, neuron_count * sizeof(int)), "cudaMalloc active B");
    check_cuda(cudaMalloc(&device_next_count, sizeof(int)), "cudaMalloc next count");
    check_cuda(
        cudaMemcpy(
            device_active_a,
            host_active.data(),
            host_active.size() * sizeof(int),
            cudaMemcpyHostToDevice),
        "cudaMemcpy active");

    auto active_count = static_cast<int>(host_active.size());
    std::uint64_t delivered = static_cast<std::uint64_t>(active_count);
    std::uint64_t fired = static_cast<std::uint64_t>(active_count);

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t step = 0; step < max_steps && active_count > 0; ++step) {
        check_cuda(cudaMemset(device_fired_flags, 0, neuron_count * sizeof(int)), "cudaMemset flags");
        check_cuda(cudaMemset(device_next_count, 0, sizeof(int)), "cudaMemset next count");

        const auto threads = 256;
        const auto blocks = (active_count + threads - 1) / threads;
        propagate_kernel<<<blocks, threads>>>(
            device_active_a,
            active_count,
            device_offsets,
            device_targets,
            device_weights,
            device_thresholds,
            device_membrane,
            device_fired_flags,
            device_active_b,
            device_next_count);
        check_cuda(cudaGetLastError(), "propagate_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "propagate_kernel synchronize");

        int next_count = 0;
        check_cuda(
            cudaMemcpy(&next_count, device_next_count, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy next count");
        active_count = next_count;
        delivered += static_cast<std::uint64_t>(active_count);
        fired += static_cast<std::uint64_t>(active_count);
        std::swap(device_active_a, device_active_b);
    }
    const auto end = std::chrono::steady_clock::now();

    cudaFree(device_offsets);
    cudaFree(device_targets);
    cudaFree(device_weights);
    cudaFree(device_thresholds);
    cudaFree(device_membrane);
    cudaFree(device_fired_flags);
    cudaFree(device_active_a);
    cudaFree(device_active_b);
    cudaFree(device_next_count);

    return {
        .executed = true,
        .delivered_spikes = delivered,
        .fired_spikes = fired,
        .elapsed_seconds = std::chrono::duration<double>(end - start).count(),
    };
}

} // namespace snncuda::backends
