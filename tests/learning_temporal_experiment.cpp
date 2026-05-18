#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

snncuda::declarative::NetworkIR make_experiment_ir() {
    using namespace snncuda::declarative;

    NetworkIR ir;
    ir.source_format = "learning-temporal-experiment";
    ir.brain.name = "LearningTemporalBrain";
    ir.neuron_params["input"] = {.threshold = 1.0F};
    ir.neuron_params["detector"] = {
        .threshold = 0.5F,
        .pattern_window_ticks = 12,
        .similarity_threshold = 0.99F,
        .max_reference_patterns = 8,
    };
    ir.neuron_params["readout"] = {.threshold = 0.5F};
    ir.neuron_params["temporal"] = {
        .threshold = 10.0F,
        .pattern_window_ticks = 8,
        .similarity_threshold = 0.99F,
        .max_reference_patterns = 4,
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

    return ir;
}

struct ExperimentResult {
    float initial_weight{0.0F};
    float trained_weight{0.0F};
    std::uint64_t detector_before_training{0};
    std::uint64_t readout_before_training{0};
    std::uint64_t detector_after_training{0};
    std::uint64_t readout_after_training{0};
    std::uint64_t temporal_matches{0};
    std::size_t learned_temporal_patterns{0};
};

ExperimentResult run_experiment() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_experiment_ir());
    const auto cue = connectome.populations.at("cue").at(0);
    const auto detector = connectome.populations.at("detector").at(0);
    const auto readout = connectome.populations.at("readout").at(0);
    const auto temporal_probe = connectome.populations.at("temporal_probe").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 2, 256);

    const auto* plastic_synapse = propagator.synapse_state(1);
    require(plastic_synapse != nullptr, "plastic synapse should exist");
    ExperimentResult result;
    result.initial_weight = plastic_synapse->weight;

    propagator.inject(cue, 0, 1.0F);
    propagator.run_until(3);
    result.detector_before_training = propagator.spike_count(detector);
    result.readout_before_training = propagator.spike_count(readout);
    require(result.detector_before_training == 0, "subthreshold cue should not fire detector before training");
    require(result.readout_before_training == 0, "readout should not fire before detector has learned");

    constexpr int training_epochs = 35;
    constexpr std::uint64_t epoch_spacing = 100;
    constexpr std::uint64_t first_training_tick = 20;
    for (int epoch = 0; epoch < training_epochs; ++epoch) {
        const auto base = first_training_tick + epoch_spacing * static_cast<std::uint64_t>(epoch);
        propagator.inject(cue, base, 1.0F);
        // Teacher pulse follows the cue's synaptic arrival by one tick, creating causal STDP.
        propagator.inject(detector, base + 2, 1.0F);
        propagator.run_until(base + 4);
    }

    plastic_synapse = propagator.synapse_state(1);
    require(plastic_synapse != nullptr, "plastic synapse should remain available");
    result.trained_weight = plastic_synapse->weight;
    require(result.trained_weight > result.initial_weight, "STDP should increase cue-to-detector weight");
    require(result.trained_weight >= 0.5F, "trained synapse should cross detector firing threshold");

    const auto detector_spikes_after_training = propagator.spike_count(detector);
    const auto readout_spikes_after_training = propagator.spike_count(readout);
    const auto test_tick = first_training_tick + epoch_spacing * training_epochs + 20;
    propagator.inject(cue, test_tick, 1.0F);
    propagator.run_until(test_tick + 3);

    result.detector_after_training = propagator.spike_count(detector) - detector_spikes_after_training;
    result.readout_after_training = propagator.spike_count(readout) - readout_spikes_after_training;
    require(result.detector_after_training == 1, "cue-only test should fire detector after learning");
    require(result.readout_after_training == 1, "learned detector spike should propagate to readout");

    propagator.inject(temporal_probe, test_tick + 20, 0.1F);
    propagator.inject(temporal_probe, test_tick + 22, 0.1F);
    propagator.run_until(test_tick + 22);

    auto temporal_state = propagator.state(temporal_probe);
    require(
        temporal_state.temporal_pattern.learned_patterns.size() == 1,
        "first temporal code should be learned");

    propagator.inject(temporal_probe, test_tick + 40, 0.1F);
    propagator.inject(temporal_probe, test_tick + 42, 0.1F);
    propagator.run_until(test_tick + 42);

    temporal_state = propagator.state(temporal_probe);
    result.temporal_matches = temporal_state.temporal_pattern.match_count;
    result.learned_temporal_patterns = temporal_state.temporal_pattern.learned_patterns.size();
    require(result.temporal_matches >= 1, "repeated temporal code should be recognized");

    return result;
}

} // namespace

int main() {
    try {
        const auto result = run_experiment();
        std::cout << "learning_temporal_experiment=passed\n";
        std::cout << "initial_plastic_weight=" << result.initial_weight << '\n';
        std::cout << "trained_plastic_weight=" << result.trained_weight << '\n';
        std::cout << "detector_spikes_before_training=" << result.detector_before_training << '\n';
        std::cout << "readout_spikes_before_training=" << result.readout_before_training << '\n';
        std::cout << "detector_spikes_from_cue_only_after_training="
                  << result.detector_after_training << '\n';
        std::cout << "readout_spikes_from_cue_only_after_training="
                  << result.readout_after_training << '\n';
        std::cout << "temporal_pattern_matches=" << result.temporal_matches << '\n';
        std::cout << "learned_temporal_patterns=" << result.learned_temporal_patterns << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Learning/temporal experiment failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
