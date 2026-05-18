#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

#include <cmath>
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

void require_close(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        throw std::runtime_error(message);
    }
}

snncuda::declarative::NetworkIR make_two_neuron_ir(
    float target_threshold,
    float weight,
    snncuda::snn::DendriticCompartment compartment = snncuda::snn::DendriticCompartment::Basal,
    snncuda::snn::ReceptorType receptor = snncuda::snn::ReceptorType::Ampa,
    bool plasticity_enabled = false) {
    using namespace snncuda::declarative;

    NetworkIR ir;
    ir.source_format = "test";
    ir.brain.name = "SynapseDendriteBrain";
    ir.neuron_params["input"] = {.threshold = 1.0F};
    ir.neuron_params["target"] = {
        .threshold = target_threshold,
        .pattern_window_ticks = 10,
        .similarity_threshold = 0.99F,
        .max_reference_patterns = 4,
    };

    LayerIR layer;
    layer.name = "Layer";
    PopulationIR input;
    input.name = "input";
    input.count = 1;
    input.neuron_params = "input";
    layer.populations.push_back(std::move(input));

    PopulationIR target;
    target.name = "target";
    target.count = 1;
    target.neuron_params = "target";
    layer.populations.push_back(std::move(target));

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
        .name = "input_to_target",
        .source = "input",
        .target = "target",
        .pattern = "one_to_one",
        .weight = weight,
        .max_weight = 2.0F,
        .delay_ticks = 1,
        .compartment = compartment,
        .receptor = receptor,
        .plasticity_enabled = plasticity_enabled,
    });
    return ir;
}

snncuda::declarative::NetworkIR make_single_neuron_ir() {
    using namespace snncuda::declarative;

    NetworkIR ir;
    ir.source_format = "test";
    ir.brain.name = "PatternBrain";
    ir.neuron_params["default"] = {
        .threshold = 10.0F,
        .pattern_window_ticks = 10,
        .similarity_threshold = 0.99F,
        .max_reference_patterns = 2,
    };

    LayerIR layer;
    layer.name = "Layer";
    PopulationIR target;
    target.name = "target";
    target.count = 1;
    target.neuron_params = "default";
    layer.populations.push_back(std::move(target));

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
    return ir;
}

void test_spike_enters_dendritic_compartment_before_soma() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(
        make_two_neuron_ir(
            1.0F,
            0.6F,
            snncuda::snn::DendriticCompartment::Apical));
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 8, 8);
    propagator.inject(input, 0, 1.0F);
    propagator.run_until(1);

    const auto target_state = propagator.state(target);
    require(propagator.spike_count(target) == 0, "subthreshold dendritic input should not fire");
    require_close(target_state.soma.current, 0.0F, 0.0001F, "synaptic input should not land directly in soma");
    require_close(target_state.apical.current, 0.6F, 0.0001F, "synaptic input should land in apical dendrite");
    require_close(target_state.membrane_potential, 0.6F, 0.0001F, "membrane should reflect dendritic current");

    const auto* synapse = propagator.synapse_state(1);
    require(synapse != nullptr, "synapse runtime state should be available");
    require(synapse->pre_spike_count == 1, "synapse should record presynaptic spike arrival");
}

void test_inhibitory_receptor_and_decay() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(
        make_two_neuron_ir(
            0.2F,
            0.8F,
            snncuda::snn::DendriticCompartment::Inhibitory,
            snncuda::snn::ReceptorType::GabaA));
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 8, 16);
    propagator.inject(input, 0, 1.0F);
    propagator.run_until(1);

    const auto inhibited = propagator.state(target);
    require(inhibited.inhibitory.current < -0.79F, "GABA input should be negative current");
    require(inhibited.membrane_potential < 0.0F, "inhibitory current should reduce membrane potential");
    require(propagator.spike_count(target) == 0, "inhibitory input should not fire target");

    propagator.inject(target, 11, 0.0F);
    propagator.run_until(11);
    const auto decayed = propagator.state(target);
    require(decayed.inhibitory.current > inhibited.inhibitory.current, "inhibitory current should decay toward zero");
    require(decayed.inhibitory.current < 0.0F, "decayed inhibitory current should remain negative");
}

void test_post_after_pre_potentiates_synapse() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(
        make_two_neuron_ir(
            0.4F,
            0.5F,
            snncuda::snn::DendriticCompartment::Basal,
            snncuda::snn::ReceptorType::Ampa,
            true));
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 8, 8);
    propagator.inject(input, 0, 1.0F);
    propagator.run_until(1);

    require(propagator.spike_count(target) == 1, "target should fire from potentiating input");
    const auto* synapse = propagator.synapse_state(1);
    require(synapse != nullptr, "synapse state should exist");
    require(synapse->weight > 0.5F, "post-after-pre STDP should potentiate the synapse");
    require(synapse->plasticity_update_count == 1, "potentiation should be recorded exactly once");
}

void test_pre_after_post_depresses_synapse() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(
        make_two_neuron_ir(
            1.0F,
            0.5F,
            snncuda::snn::DendriticCompartment::Basal,
            snncuda::snn::ReceptorType::Ampa,
            true));
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 8, 16);
    propagator.inject(target, 0, 1.0F);
    propagator.run_until(0);
    require(propagator.spike_count(target) == 1, "target should fire before presynaptic spike");

    propagator.inject(input, 5, 1.0F);
    propagator.run_until(6);

    const auto* synapse = propagator.synapse_state(1);
    require(synapse != nullptr, "synapse state should exist");
    require(synapse->weight < 0.5F, "pre-after-post STDP should depress the synapse");
    require(propagator.spike_count(target) == 1, "depressed subthreshold input should not refire target");
}

void test_temporal_pattern_state_learns_and_matches() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_single_neuron_ir());
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 4, 64);
    propagator.inject(target, 10, 0.1F);
    propagator.inject(target, 12, 0.1F);
    propagator.run_until(12);

    auto state = propagator.state(target);
    require(state.temporal_pattern.learned_patterns.size() == 1, "first two-spike temporal pattern should be learned");
    require(state.temporal_pattern.match_count == 0, "first pattern should learn before it can match");

    propagator.inject(target, 30, 0.1F);
    propagator.inject(target, 32, 0.1F);
    propagator.run_until(32);

    state = propagator.state(target);
    require(state.temporal_pattern.last_match, "repeated temporal pattern should match learned pattern");
    require(state.temporal_pattern.match_count == 1, "repeated pattern should increment match count");
}

void test_lru_state_paging_under_tiny_resident_cache() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(
        make_two_neuron_ir(0.4F, 0.5F));
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(connectome, store, 1, 8);
    propagator.inject(input, 0, 1.0F);
    propagator.run_until(1);

    require(propagator.spike_count(input) == 1, "input should fire through paged cache");
    require(propagator.spike_count(target) == 1, "target should fire through paged cache");
    require(propagator.resident_neuron_count() == 1, "resident cache should respect capacity");
    require(propagator.cache_eviction_count() > 0, "LRU cache should page out neuron state");
}

} // namespace

int main() {
    try {
        test_spike_enters_dendritic_compartment_before_soma();
        test_inhibitory_receptor_and_decay();
        test_post_after_pre_potentiates_synapse();
        test_pre_after_post_depresses_synapse();
        test_temporal_pattern_state_learns_and_matches();
        test_lru_state_paging_under_tiny_resident_cache();
    } catch (const std::exception& error) {
        std::cerr << "Synapse/dendrite test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All synapse/dendrite tests passed\n";
    return EXIT_SUCCESS;
}
