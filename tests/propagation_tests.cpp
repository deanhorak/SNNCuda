#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

snncuda::declarative::NetworkIR make_three_stage_ir() {
    using namespace snncuda::declarative;

    NetworkIR ir;
    ir.source_format = "test";
    ir.brain.name = "PropagationBrain";
    ir.neuron_params["default"] = {
        .threshold = 1.0F,
        .pattern_window_ticks = 10,
        .similarity_threshold = 0.9F,
        .max_reference_patterns = 4,
    };

    LayerIR layer;
    layer.name = "Layer";
    PopulationIR input;
    input.name = "input";
    input.count = 1;
    input.neuron_params = "default";
    layer.populations.push_back(std::move(input));

    PopulationIR hidden;
    hidden.name = "hidden";
    hidden.count = 1;
    hidden.neuron_params = "default";
    layer.populations.push_back(std::move(hidden));

    PopulationIR output;
    output.name = "output";
    output.count = 1;
    output.neuron_params = "default";
    layer.populations.push_back(std::move(output));

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
        .name = "input_to_hidden",
        .source = "input",
        .target = "hidden",
        .pattern = "one_to_one",
        .weight = 1.0F,
        .max_weight = 2.0F,
        .delay_ticks = 2,
    });
    ir.projections.push_back({
        .name = "hidden_to_output",
        .source = "hidden",
        .target = "output",
        .pattern = "one_to_one",
        .weight = 1.0F,
        .max_weight = 2.0F,
        .delay_ticks = 3,
    });

    return ir;
}

void test_synapse_spike_code_pattern_propagates_and_matches() {
    auto ir = make_three_stage_ir();
    ir.neuron_params["default"].threshold = 1.0F;
    ir.neuron_params["sink"] = {
        .threshold = 100.0F,
        .pattern_window_ticks = 8,
        .similarity_threshold = 0.99F,
        .max_reference_patterns = 4,
    };
    auto& layer = ir.brain.hemispheres[0].lobes[0].regions[0].nuclei[0].columns[0].layers[0];
    layer.populations.clear();
    snncuda::declarative::PopulationIR source_population;
    source_population.name = "source";
    source_population.count = 1;
    source_population.neuron_params = "default";
    layer.populations.push_back(std::move(source_population));
    snncuda::declarative::PopulationIR sink_population;
    sink_population.name = "sink";
    sink_population.count = 1;
    sink_population.neuron_params = "sink";
    layer.populations.push_back(std::move(sink_population));
    ir.projections.clear();
    ir.projections.push_back({
        .name = "source_to_sink",
        .source = "source",
        .target = "sink",
        .pattern = "one_to_one",
        .weight = 0.1F,
        .max_weight = 2.0F,
        .delay_ticks = 1,
        .spike_code_offsets = {0, 2},
    });

    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(ir);
    const auto source = connectome.populations.at("source").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 2, 32);

    propagator.inject(source, 0, 1.0F);
    propagator.inject(source, 20, 1.0F);
    propagator.run_until(25);

    const auto* synapse = propagator.synapse_state(1);
    require(synapse != nullptr, "coded synapse should exist");
    require(synapse->spike_code_offsets.size() == 2, "synapse should preserve two-offset spike code");
    require(synapse->pre_spike_count == 4, "two source firings should deliver two spikes per synapse each");
    require(synapse->learned_code_patterns.size() >= 1, "synapse should learn incoming spike code");
    require(synapse->code_match_count >= 1, "synapse should recognize repeated incoming spike code");
    require(propagator.delivered_spike_count() == 6, "delivered count should include external and coded synaptic spikes");
}

void test_three_stage_spike_propagation() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_three_stage_ir());
    require(connectome.neurons.size() == 3, "test connectome should contain three neurons");
    require(connectome.synapses.size() == 2, "test connectome should contain two synapses");

    const auto input = connectome.populations.at("input").at(0);
    const auto hidden = connectome.populations.at("hidden").at(0);
    const auto output = connectome.populations.at("output").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 1, 8);

    propagator.inject(input, 0, 1.0F);

    propagator.run_until(0);
    require(propagator.spike_count(input) == 1, "input neuron should fire at tick 0");
    require(propagator.spike_count(hidden) == 0, "hidden neuron should not fire before delay");
    require(propagator.spike_count(output) == 0, "output neuron should not fire before upstream delay");

    propagator.run_until(2);
    require(propagator.spike_count(hidden) == 1, "hidden neuron should fire after first synaptic delay");
    require(propagator.spike_count(output) == 0, "output neuron should still wait for second delay");

    propagator.run_until(5);
    require(propagator.spike_count(output) == 1, "output neuron should fire after full chain delay");
    require(propagator.idle(), "propagator should be idle after all scheduled spikes are delivered");

    require(propagator.state(input).membrane_potential == 0.0F, "input should reset after firing");
    require(propagator.state(hidden).membrane_potential == 0.0F, "hidden should reset after firing");
    require(propagator.state(output).membrane_potential == 0.0F, "output should reset after firing");
}

void test_subthreshold_spike_does_not_propagate() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_three_stage_ir());
    const auto input = connectome.populations.at("input").at(0);
    const auto hidden = connectome.populations.at("hidden").at(0);
    const auto output = connectome.populations.at("output").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 2, 8);

    propagator.inject(input, 0, 0.5F);
    propagator.run_until(10);

    require(propagator.spike_count(input) == 0, "subthreshold input should not fire");
    require(propagator.spike_count(hidden) == 0, "subthreshold input should not reach hidden");
    require(propagator.spike_count(output) == 0, "subthreshold input should not reach output");
    require(propagator.state(input).membrane_potential == 0.5F, "subthreshold potential should persist");
    require(propagator.idle(), "propagator should be idle after subthreshold event is processed");
}

} // namespace

int main() {
    try {
        test_three_stage_spike_propagation();
        test_synapse_spike_code_pattern_propagates_and_matches();
        test_subthreshold_spike_does_not_propagate();
    } catch (const std::exception& error) {
        std::cerr << "Propagation test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All propagation tests passed\n";
    return EXIT_SUCCESS;
}
