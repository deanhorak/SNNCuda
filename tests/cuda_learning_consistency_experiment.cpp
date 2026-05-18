#include "snncuda/backends/CudaBackend.h"
#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int training_epochs = 35;
constexpr std::uint32_t epoch_spacing = 100;
constexpr std::uint64_t cue_only_test_tick = training_epochs * epoch_spacing;
constexpr std::uint32_t first_temporal_tick = 3520;
constexpr std::uint32_t second_temporal_tick = 3540;
constexpr std::uint64_t max_tick = second_temporal_tick + 3;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

snncuda::declarative::NetworkIR make_cuda_consistency_ir() {
    using namespace snncuda::declarative;

    NetworkIR ir;
    ir.source_format = "cuda-learning-consistency";
    ir.brain.name = "CudaLearningConsistencyBrain";
    ir.neuron_params["input"] = {.threshold = 1.0F};
    ir.neuron_params["detector"] = {.threshold = 0.5F};
    ir.neuron_params["readout"] = {.threshold = 0.5F};
    ir.neuron_params["temporal"] = {
        .threshold = 10.0F,
        .pattern_window_ticks = 8,
        .similarity_threshold = 0.99F,
        .max_reference_patterns = 1,
    };

    LayerIR layer;
    layer.name = "Layer";

    PopulationIR cue;
    cue.name = "cue";
    cue.count = 1;
    cue.neuron_params = "input";
    layer.populations.push_back(std::move(cue));

    PopulationIR detector;
    detector.name = "detector";
    detector.count = 1;
    detector.neuron_params = "detector";
    layer.populations.push_back(std::move(detector));

    PopulationIR readout;
    readout.name = "readout";
    readout.count = 1;
    readout.neuron_params = "readout";
    layer.populations.push_back(std::move(readout));

    PopulationIR temporal_probe;
    temporal_probe.name = "temporal_probe";
    temporal_probe.count = 1;
    temporal_probe.neuron_params = "temporal";
    layer.populations.push_back(std::move(temporal_probe));

    for (int source = 0; source < 4; ++source) {
        PopulationIR temporal_source;
        temporal_source.name = "temporal_source_" + std::to_string(source);
        temporal_source.count = 1;
        temporal_source.neuron_params = "input";
        layer.populations.push_back(std::move(temporal_source));
    }

    for (int epoch = 0; epoch < training_epochs; ++epoch) {
        PopulationIR teacher;
        teacher.name = "teacher_" + std::to_string(epoch);
        teacher.count = 1;
        teacher.neuron_params = "input";
        layer.populations.push_back(std::move(teacher));
    }

    ColumnIR column;
    column.name = "Column";
    column.layers.push_back(std::move(layer));

    NucleusIR nucleus;
    nucleus.name = "Nucleus";
    nucleus.columns.push_back(std::move(column));

    RegionIR region;
    region.name = "Region";
    region.nuclei.push_back(std::move(nucleus));

    LobeIR lobe;
    lobe.name = "Lobe";
    lobe.regions.push_back(std::move(region));

    HemisphereIR hemisphere;
    hemisphere.name = "Left";
    hemisphere.lobes.push_back(std::move(lobe));
    ir.brain.hemispheres.push_back(std::move(hemisphere));

    ir.projections.push_back({
        .name = "cue_repeat",
        .source = "cue",
        .target = "cue",
        .pattern = "one_to_one",
        .weight = 1.0F,
        .max_weight = 1.0F,
        .delay_ticks = epoch_spacing,
        .compartment = snncuda::snn::DendriticCompartment::Basal,
        .receptor = snncuda::snn::ReceptorType::Ampa,
        .plasticity_enabled = false,
    });
    ir.projections.push_back({
        .name = "cue_to_detector_plastic",
        .source = "cue",
        .target = "detector",
        .pattern = "one_to_one",
        .weight = 0.25F,
        .max_weight = 1.5F,
        .delay_ticks = 1,
        .compartment = snncuda::snn::DendriticCompartment::Basal,
        .receptor = snncuda::snn::ReceptorType::Ampa,
        .plasticity_enabled = true,
    });
    ir.projections.push_back({
        .name = "detector_to_readout",
        .source = "detector",
        .target = "readout",
        .pattern = "one_to_one",
        .weight = 0.75F,
        .max_weight = 1.0F,
        .delay_ticks = 1,
        .compartment = snncuda::snn::DendriticCompartment::Basal,
        .receptor = snncuda::snn::ReceptorType::Ampa,
        .plasticity_enabled = false,
    });

    const std::uint32_t temporal_delays[] = {
        first_temporal_tick,
        first_temporal_tick + 2U,
        second_temporal_tick,
        second_temporal_tick + 2U,
    };
    for (int source = 0; source < 4; ++source) {
        ir.projections.push_back({
            .name = "temporal_source_" + std::to_string(source) + "_to_probe",
            .source = "temporal_source_" + std::to_string(source),
            .target = "temporal_probe",
            .pattern = "one_to_one",
            .weight = 0.1F,
            .max_weight = 0.1F,
            .delay_ticks = temporal_delays[source],
            .compartment = snncuda::snn::DendriticCompartment::Soma,
            .receptor = snncuda::snn::ReceptorType::Ampa,
            .plasticity_enabled = false,
        });
    }

    for (int epoch = 0; epoch < training_epochs; ++epoch) {
        ir.projections.push_back({
            .name = "teacher_" + std::to_string(epoch) + "_to_detector",
            .source = "teacher_" + std::to_string(epoch),
            .target = "detector",
            .pattern = "one_to_one",
            .weight = 1.0F,
            .max_weight = 1.0F,
            .delay_ticks = 2U + static_cast<std::uint32_t>(epoch) * epoch_spacing,
            .compartment = snncuda::snn::DendriticCompartment::Soma,
            .receptor = snncuda::snn::ReceptorType::Ampa,
            .plasticity_enabled = false,
        });
    }

    return ir;
}

struct ConsistencyResult {
    std::uint64_t delivered{0};
    std::uint64_t fired{0};
    std::uint64_t cue_spikes{0};
    std::uint64_t detector_spikes{0};
    std::uint64_t readout_spikes{0};
    std::uint64_t temporal_matches{0};
    std::uint32_t temporal_learned_patterns{0};
    float plastic_weight{0.0F};
};

std::size_t synapse_index(
    const snncuda::declarative::Connectome& connectome,
    snncuda::core::NeuronId source,
    snncuda::core::NeuronId target) {
    for (std::size_t index = 0; index < connectome.synapses.size(); ++index) {
        if (connectome.synapses[index].source == source
            && connectome.synapses[index].target == target) {
            return index;
        }
    }
    throw std::runtime_error("synapse not found");
}

std::vector<snncuda::core::NeuronId> initial_active(
    const snncuda::declarative::Connectome& connectome) {
    std::vector<snncuda::core::NeuronId> active;
    active.push_back(connectome.populations.at("cue").at(0));
    for (int epoch = 0; epoch < training_epochs; ++epoch) {
        active.push_back(connectome.populations.at("teacher_" + std::to_string(epoch)).at(0));
    }
    for (int source = 0; source < 4; ++source) {
        active.push_back(connectome.populations.at("temporal_source_" + std::to_string(source)).at(0));
    }
    return active;
}

ConsistencyResult run_cpu_reference(
    const snncuda::declarative::Connectome& connectome,
    const std::vector<snncuda::core::NeuronId>& active,
    std::size_t plastic_index) {
    const auto cue = connectome.populations.at("cue").at(0);
    const auto detector = connectome.populations.at("detector").at(0);
    const auto readout = connectome.populations.at("readout").at(0);
    const auto temporal_probe = connectome.populations.at("temporal_probe").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 4, 8192);
    for (const auto neuron : active) {
        propagator.inject(neuron, 0, 1.0F);
    }
    propagator.run_until(max_tick);

    const auto* plastic = propagator.synapse_state(connectome.synapses[plastic_index].id);
    require(plastic != nullptr, "CPU plastic synapse should exist");
    const auto temporal_state = propagator.state(temporal_probe);
    return {
        .delivered = propagator.delivered_spike_count(),
        .fired = propagator.fired_spike_count(),
        .cue_spikes = propagator.spike_count(cue),
        .detector_spikes = propagator.spike_count(detector),
        .readout_spikes = propagator.spike_count(readout),
        .temporal_matches = temporal_state.temporal_pattern.match_count,
        .temporal_learned_patterns = static_cast<std::uint32_t>(
            temporal_state.temporal_pattern.learned_patterns.size()),
        .plastic_weight = plastic->weight,
    };
}

} // namespace

int main() {
    const snncuda::backends::CudaBackend cuda;
    if (!cuda.available()) {
        std::cout << "CUDA unavailable, skipping CUDA learning consistency experiment: "
                  << cuda.availability_status() << '\n';
        return EXIT_SUCCESS;
    }

    try {
        const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_cuda_consistency_ir());
        const auto cue = connectome.populations.at("cue").at(0);
        const auto detector = connectome.populations.at("detector").at(0);
        const auto readout = connectome.populations.at("readout").at(0);
        const auto temporal_probe = connectome.populations.at("temporal_probe").at(0);
        const auto plastic_index = synapse_index(connectome, cue, detector);
        const auto active = initial_active(connectome);
        const auto cpu = run_cpu_reference(connectome, active, plastic_index);
        const auto gpu = cuda.run_resident_propagation(connectome, active, static_cast<std::uint32_t>(max_tick));

        require(gpu.executed, "CUDA experiment should execute");
        require(gpu.delivered_spikes == cpu.delivered, "CUDA delivered spikes should match CPU");
        require(gpu.fired_spikes == cpu.fired, "CUDA fired spikes should match CPU");
        require(gpu.final_neuron_spike_counts.at(static_cast<std::size_t>(cue - 1)) == cpu.cue_spikes,
            "CUDA cue spike count should match CPU");
        require(gpu.final_neuron_spike_counts.at(static_cast<std::size_t>(detector - 1)) == cpu.detector_spikes,
            "CUDA detector spike count should match CPU");
        require(gpu.final_neuron_spike_counts.at(static_cast<std::size_t>(readout - 1)) == cpu.readout_spikes,
            "CUDA readout spike count should match CPU");
        require(gpu.final_temporal_match_counts.at(static_cast<std::size_t>(temporal_probe - 1))
                == cpu.temporal_matches,
            "CUDA temporal match count should match CPU");
        require(gpu.final_temporal_learned_pattern_counts.at(static_cast<std::size_t>(temporal_probe - 1))
                == cpu.temporal_learned_patterns,
            "CUDA learned temporal pattern count should match CPU");
        require(cpu.temporal_matches >= 1, "CPU should recognize repeated temporal code");
        require(gpu.debug.temporal_matches >= 1, "CUDA should recognize repeated temporal code");
        require(gpu.final_synapse_weights.at(plastic_index) > 0.5F,
            "CUDA plastic synapse should learn enough for cue-only propagation");
        require(std::fabs(gpu.final_synapse_weights.at(plastic_index) - cpu.plastic_weight) < 0.0001F,
            "CUDA plastic weight should match CPU reference");
        require(gpu.debug.dropped_events == 0, "CUDA experiment should not drop events");
        require(gpu.debug.lock_timeouts == 0, "CUDA experiment should not hit lock timeouts");
        require(gpu.debug.stdp_ltp > gpu.debug.stdp_ltd, "CUDA experiment should be net causal LTP");

        std::cout << "cuda_learning_consistency_experiment=passed\n";
        std::cout << "cpu_delivered_spikes=" << cpu.delivered << '\n';
        std::cout << "cuda_delivered_spikes=" << gpu.delivered_spikes << '\n';
        std::cout << "cpu_fired_spikes=" << cpu.fired << '\n';
        std::cout << "cuda_fired_spikes=" << gpu.fired_spikes << '\n';
        std::cout << "cpu_plastic_weight=" << cpu.plastic_weight << '\n';
        std::cout << "cuda_plastic_weight=" << gpu.final_synapse_weights.at(plastic_index) << '\n';
        std::cout << "cpu_detector_spikes=" << cpu.detector_spikes << '\n';
        std::cout << "cuda_detector_spikes="
                  << gpu.final_neuron_spike_counts.at(static_cast<std::size_t>(detector - 1)) << '\n';
        std::cout << "cpu_readout_spikes=" << cpu.readout_spikes << '\n';
        std::cout << "cuda_readout_spikes="
                  << gpu.final_neuron_spike_counts.at(static_cast<std::size_t>(readout - 1)) << '\n';
        std::cout << "cpu_temporal_matches=" << cpu.temporal_matches << '\n';
        std::cout << "cuda_temporal_matches="
                  << gpu.final_temporal_match_counts.at(static_cast<std::size_t>(temporal_probe - 1)) << '\n';
        std::cout << "cpu_temporal_learned_patterns=" << cpu.temporal_learned_patterns << '\n';
        std::cout << "cuda_temporal_learned_patterns="
                  << gpu.final_temporal_learned_pattern_counts.at(
                         static_cast<std::size_t>(temporal_probe - 1)) << '\n';
        std::cout << "cuda_debug_stdp_ltp=" << gpu.debug.stdp_ltp << '\n';
        std::cout << "cuda_debug_stdp_ltd=" << gpu.debug.stdp_ltd << '\n';
        std::cout << "cuda_debug_temporal_observations=" << gpu.debug.temporal_observations << '\n';
        std::cout << "cuda_debug_temporal_patterns_learned=" << gpu.debug.temporal_patterns_learned << '\n';
        std::cout << "cuda_debug_temporal_matches=" << gpu.debug.temporal_matches << '\n';
        std::cout << "cuda_debug_dropped_events=" << gpu.debug.dropped_events << '\n';
        std::cout << "cuda_debug_lock_timeouts=" << gpu.debug.lock_timeouts << '\n';
    } catch (const std::exception& error) {
        std::cerr << "CUDA learning consistency experiment failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
