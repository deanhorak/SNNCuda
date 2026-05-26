#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Stimulus {
    std::string name;
    std::vector<int> active_inputs;
    int expected_class{0};
};

struct Accuracy {
    int correct{0};
    int total{0};

    [[nodiscard]] double value() const noexcept {
        return total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
    }
};

struct ExperimentResult {
    Accuracy before_training;
    std::vector<Accuracy> epoch_accuracy;
    Accuracy final_accuracy;
    std::uint64_t delivered_spikes{0};
    std::uint64_t fired_spikes{0};
    std::vector<float> class0_weights;
    std::vector<float> class1_weights;
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

snncuda::declarative::NetworkIR make_network_ir() {
    using namespace snncuda::declarative;

    NetworkIR ir;
    ir.source_format = "network-learning-accuracy-experiment";
    ir.brain.name = "LearningAccuracyBrain";
    ir.neuron_params["input"] = {.threshold = 1.0F};
    ir.neuron_params["class"] = {
        .threshold = 0.45F,
        .pattern_window_ticks = 16,
        .similarity_threshold = 0.98F,
        .max_reference_patterns = 4,
    };

    LayerIR layer;
    layer.name = "Layer";

    PopulationIR inputs;
    inputs.name = "input";
    inputs.count = 4;
    inputs.neuron_params = "input";
    layer.populations.push_back(std::move(inputs));

    PopulationIR class0;
    class0.name = "class0";
    class0.count = 1;
    class0.neuron_params = "class";
    layer.populations.push_back(std::move(class0));

    PopulationIR class1;
    class1.name = "class1";
    class1.count = 1;
    class1.neuron_params = "class";
    layer.populations.push_back(std::move(class1));

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
        .name = "input_to_class0",
        .source = "input",
        .target = "class0",
        .pattern = "all_to_all",
        .weight = 0.08F,
        .max_weight = 1.2F,
        .delay_ticks = 1,
        .compartment = snncuda::snn::DendriticCompartment::Basal,
        .receptor = snncuda::snn::ReceptorType::Ampa,
        .plasticity_enabled = true,
    });
    ir.projections.push_back({
        .name = "input_to_class1",
        .source = "input",
        .target = "class1",
        .pattern = "all_to_all",
        .weight = 0.08F,
        .max_weight = 1.2F,
        .delay_ticks = 1,
        .compartment = snncuda::snn::DendriticCompartment::Basal,
        .receptor = snncuda::snn::ReceptorType::Ampa,
        .plasticity_enabled = true,
    });

    return ir;
}

std::vector<Stimulus> training_set() {
    return {
        {.name = "left_pair", .active_inputs = {0, 1}, .expected_class = 0},
        {.name = "right_pair", .active_inputs = {2, 3}, .expected_class = 1},
    };
}

std::vector<Stimulus> test_set() {
    return {
        {.name = "left_pair", .active_inputs = {0, 1}, .expected_class = 0},
        {.name = "left_delayed_feature", .active_inputs = {1, 0}, .expected_class = 0},
        {.name = "left_with_noise", .active_inputs = {0, 1, 2}, .expected_class = 0},
        {.name = "right_pair", .active_inputs = {2, 3}, .expected_class = 1},
        {.name = "right_delayed_feature", .active_inputs = {3, 2}, .expected_class = 1},
        {.name = "right_with_noise", .active_inputs = {1, 2, 3}, .expected_class = 1},
    };
}

std::vector<snncuda::core::NeuronId> class_ids(
    const snncuda::declarative::Connectome& connectome) {
    return {
        connectome.populations.at("class0").at(0),
        connectome.populations.at("class1").at(0),
    };
}

void present_stimulus(
    snncuda::runtime::NetworkPropagator& propagator,
    const snncuda::declarative::Connectome& connectome,
    const Stimulus& stimulus,
    std::uint64_t base_tick,
    bool with_teacher) {
    const auto& inputs = connectome.populations.at("input");
    for (std::size_t order = 0; order < stimulus.active_inputs.size(); ++order) {
        const auto input_index = static_cast<std::size_t>(stimulus.active_inputs[order]);
        propagator.inject(inputs.at(input_index), base_tick + static_cast<std::uint64_t>(order % 2), 1.0F);
    }

    if (with_teacher) {
        const auto classes = class_ids(connectome);
        propagator.inject(classes.at(static_cast<std::size_t>(stimulus.expected_class)), base_tick + 3, 1.0F);
    }
}

int predict(
    snncuda::runtime::NetworkPropagator& propagator,
    const snncuda::declarative::Connectome& connectome,
    const Stimulus& stimulus,
    std::uint64_t base_tick) {
    const auto classes = class_ids(connectome);
    const auto before0 = propagator.spike_count(classes[0]);
    const auto before1 = propagator.spike_count(classes[1]);
    present_stimulus(propagator, connectome, stimulus, base_tick, false);
    propagator.run_until(base_tick + 16);
    const auto after0 = propagator.spike_count(classes[0]) - before0;
    const auto after1 = propagator.spike_count(classes[1]) - before1;

    if (after0 == after1) {
        return -1;
    }
    return after0 > after1 ? 0 : 1;
}

Accuracy evaluate(
    const snncuda::declarative::Connectome& connectome,
    const std::vector<Stimulus>& stimuli) {
    Accuracy accuracy;
    for (const auto& stimulus : stimuli) {
        snncuda::runtime::MemoryNeuronStateStore store;
        snncuda::runtime::NetworkPropagator propagator(connectome, store, 8, 4096);
        std::uint64_t tick = 0;
        const auto predicted = predict(propagator, connectome, stimulus, tick);
        ++accuracy.total;
        if (predicted == stimulus.expected_class) {
            ++accuracy.correct;
        }
    }
    return accuracy;
}

void train_epoch(
    snncuda::runtime::NetworkPropagator& propagator,
    const snncuda::declarative::Connectome& connectome,
    const std::vector<Stimulus>& stimuli,
    std::uint64_t& tick) {
    for (const auto& stimulus : stimuli) {
        present_stimulus(propagator, connectome, stimulus, tick, true);
        propagator.run_until(tick + 16);
        tick += 80;
    }
}

std::vector<float> incoming_weights(
    const snncuda::declarative::Connectome& connectome,
    const snncuda::runtime::NetworkPropagator& propagator,
    snncuda::core::NeuronId target) {
    std::vector<float> weights;
    for (const auto& synapse : connectome.synapses) {
        if (synapse.target != target) {
            continue;
        }
        const auto* state = propagator.synapse_state(synapse.id);
        weights.push_back(state == nullptr ? synapse.weight : state->weight);
    }
    return weights;
}

snncuda::declarative::Connectome frozen_snapshot(
    const snncuda::declarative::Connectome& connectome,
    const snncuda::runtime::NetworkPropagator& propagator) {
    auto snapshot = connectome;
    for (auto& synapse : snapshot.synapses) {
        if (const auto* state = propagator.synapse_state(synapse.id)) {
            synapse.weight = state->weight;
        }
        synapse.plasticity_enabled = false;
    }
    return snapshot;
}

ExperimentResult run_experiment() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_network_ir());
    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 8, 4096);

    const auto train = training_set();
    const auto test = test_set();
    std::uint64_t tick = 0;

    ExperimentResult result;
    auto snapshot = frozen_snapshot(connectome, propagator);
    result.before_training = evaluate(snapshot, test);

    constexpr int epochs = 40;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        train_epoch(propagator, connectome, train, tick);
        snapshot = frozen_snapshot(connectome, propagator);
        result.epoch_accuracy.push_back(evaluate(snapshot, test));
    }

    result.final_accuracy = result.epoch_accuracy.back();
    result.delivered_spikes = propagator.delivered_spike_count();
    result.fired_spikes = propagator.fired_spike_count();
    const auto classes = class_ids(connectome);
    result.class0_weights = incoming_weights(connectome, propagator, classes[0]);
    result.class1_weights = incoming_weights(connectome, propagator, classes[1]);
    return result;
}

void print_weights(const char* prefix, const std::vector<float>& weights) {
    for (std::size_t i = 0; i < weights.size(); ++i) {
        std::cout << prefix << "_input" << i << "_weight=" << weights[i] << '\n';
    }
}

} // namespace

int main() {
    try {
        const auto result = run_experiment();
        std::cout << "network_learning_accuracy_experiment=passed\n";
        std::cout << "stimulus_count=" << test_set().size() << '\n';
        std::cout << "training_epochs=" << result.epoch_accuracy.size() << '\n';
        std::cout << "accuracy_before_training=" << result.before_training.value() << '\n';
        for (std::size_t epoch = 0; epoch < result.epoch_accuracy.size(); ++epoch) {
            std::cout << "accuracy_after_epoch_" << (epoch + 1) << "="
                      << result.epoch_accuracy[epoch].value() << '\n';
        }
        std::cout << "final_accuracy=" << result.final_accuracy.value() << '\n';
        std::cout << "delivered_spikes=" << result.delivered_spikes << '\n';
        std::cout << "fired_spikes=" << result.fired_spikes << '\n';
        print_weights("class0", result.class0_weights);
        print_weights("class1", result.class1_weights);

        require(result.before_training.value() < 0.5, "untrained network should not already solve the task");
        require(result.final_accuracy.value() >= 0.95, "trained network should solve the stimulus set");
    } catch (const std::exception& error) {
        std::cerr << "Network learning accuracy experiment failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
