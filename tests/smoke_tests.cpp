#include "snncuda/core/Version.h"
#include "snncuda/declarative/Connectome.h"
#include "snncuda/declarative/DeclarativeLoader.h"
#include "snncuda/hierarchy/CorticalMicrocircuit.h"
#include "snncuda/learning/STDP.h"
#include "snncuda/runtime/NeuronStateCache.h"
#include "snncuda/runtime/SimulationClock.h"
#include "snncuda/runtime/SpikeScheduler.h"
#include "snncuda/runtime/SynapseDendriteProcessor.h"
#include "snncuda/snn/Neuron.h"
#include "snncuda/snn/TemporalPattern.h"
#include "snncuda/storage/LruCache.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path write_fixture(const std::string& name, const std::string& content) {
    const auto suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path = std::filesystem::temp_directory_path() / (suffix + "_" + name);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write fixture");
    }
    output << content;
    return path;
}

std::filesystem::path make_temp_dir(const std::string& prefix) {
    const auto suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return std::filesystem::temp_directory_path() / (prefix + "_" + suffix);
}

void test_clock() {
    snncuda::runtime::SimulationClock clock{0.25};
    require(clock.tick() == 0, "initial tick should be zero");
    require(clock.time_ms() == 0.0, "initial time should be zero");

    clock.advance();
    clock.advance();

    require(clock.tick() == 2, "clock should advance ticks");
    require(clock.time_ms() == 0.5, "clock should advance time");
}

void test_neuron() {
    snncuda::snn::Neuron neuron{1.0};

    neuron.receive(0.4);
    require(!neuron.should_fire(), "neuron should not fire below threshold");

    neuron.receive(0.7);
    require(neuron.should_fire(), "neuron should fire at threshold");

    neuron.fire();
    require(neuron.spike_count() == 1, "spike count should increment");
    require(neuron.membrane_potential() == 0.0, "fire should reset potential");
}

void test_version() {
    require(!snncuda::core::version().empty(), "version should be defined");
}

void test_hierarchy() {
    snncuda::hierarchy::Hierarchy hierarchy;
    const auto brain = hierarchy.create_root("Brain");
    const auto hemisphere = hierarchy.create_child(
        brain,
        snncuda::hierarchy::NodeKind::Hemisphere,
        "Left Hemisphere");
    const auto lobe = hierarchy.create_child(
        hemisphere,
        snncuda::hierarchy::NodeKind::Lobe,
        "Occipital Lobe");
    const auto region = hierarchy.create_child(lobe, snncuda::hierarchy::NodeKind::Region, "V1");
    const auto nucleus = hierarchy.create_child(
        region,
        snncuda::hierarchy::NodeKind::Nucleus,
        "Stage1");

    const auto column = snncuda::hierarchy::create_canonical_cortical_column(
        hierarchy,
        nucleus,
        "Column0");

    require(column.column != snncuda::core::invalid_id, "column should be created");
    require(hierarchy.find_by_kind(snncuda::hierarchy::NodeKind::Layer).size() == 5,
        "canonical cortical column should create five named layer groups");
    require(hierarchy.path(column.layers[2]) == "Brain/Left Hemisphere/Occipital Lobe/V1/Stage1/Column0/L4",
        "hierarchy path should include layer name");
}

void test_lru_cache() {
    snncuda::storage::LruCache<int, int> cache{2};
    require(!cache.put(1, 10).has_value(), "first insert should not evict");
    require(!cache.put(2, 20).has_value(), "second insert should not evict");
    require(cache.get(1).value() == 10, "get should return value and refresh recency");
    const auto evicted = cache.put(3, 30);
    require(evicted.has_value(), "third insert should evict");
    require(evicted->first == 2, "least recently used entry should evict");
}

void test_neuron_state_cache() {
    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NeuronStateCache cache{1, store};

    snncuda::runtime::NeuronState first;
    first.id = 1;
    first.membrane_potential = 0.5F;
    cache.store_after_compute(first);

    snncuda::runtime::NeuronState second;
    second.id = 2;
    second.membrane_potential = 0.25F;
    cache.store_after_compute(second);

    const auto restored = cache.load_for_spike(1);
    require(restored.id == 1, "evicted neuron should reload from backing store");
    require(restored.membrane_potential == 0.5F, "backing store should preserve state");
}

void test_stdp() {
    snncuda::learning::StdpRule rule;
    snncuda::snn::Synapse synapse{.id = 1, .source = 10, .target = 11, .weight = 1.0F};
    const snncuda::snn::RetrogradeEvent event{
        .synapse = 1,
        .pre_neuron = 10,
        .post_neuron = 11,
        .pre_tick = 10,
        .post_tick = 12,
    };

    rule.apply(synapse, event);
    require(synapse.weight > 1.0F, "causal pre-before-post spike should potentiate");

    rule.set_enabled(false);
    const auto frozen = synapse.weight;
    rule.apply(synapse, event);
    require(synapse.weight == frozen, "disabled STDP should freeze weight");
}

void test_temporal_pattern() {
    snncuda::snn::TemporalPatternMatcher matcher{
        {.window_ticks = 100, .similarity_threshold = 0.95F, .max_reference_patterns = 2}};
    matcher.learn({{1, 2, 3}});

    require(matcher.matches({{1, 2, 3}}), "identical pattern should match");
    require(matcher.pattern_count() == 1, "pattern should be stored");
}

void test_declarative_loader_detection() {
    snncuda::declarative::DeclarativeLoader loader;
    const auto path = write_fixture(
        "snncuda_native_test.snncuda.json",
        R"json({
          "snncuda": {
            "neuron_params": {
              "default": { "threshold": 1.0, "window_size_ms": 20 }
            },
            "network": {
              "name": "Flat",
              "populations": [
                { "name": "input", "count": 2 },
                { "name": "output", "count": 1 }
              ],
              "projections": [
                { "name": "input_to_output", "source": "input", "target": "output", "weight": 0.5 }
              ]
            }
          }
        })json");

    const auto ir = loader.parse_only(path);
    require(ir.source_format == "native-json", "native parser should be selected");
    require(ir.brain.name == "Flat", "native flat network should set brain name");

    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(ir);
    require(connectome.neurons.size() == 3, "native parser should produce neurons");
    require(connectome.synapses.size() == 2, "native parser should produce all-to-all projection");
}

void test_all_declared_parsers_build_connectome() {
    snncuda::declarative::DeclarativeLoader loader;
    const auto sonata = write_fixture(
        "snncuda_test.sonata.json",
        R"json({
          "network_name": "SONATA Fixture",
          "snncuda": {
            "neuron_params": { "default": { "threshold": 1.0 } },
            "network": {
              "name": "SonataFlat",
              "populations": [
                { "name": "a", "count": 1 },
                { "name": "b", "count": 1 }
              ],
              "projections": [
                { "name": "a_to_b", "source": "a", "target": "b", "pattern": "one_to_one" }
              ]
            }
          }
        })json");

    const auto neuroml = write_fixture(
        "snncuda_test.nml",
        R"xml(<neuroml id="fixture">
          <cell id="cell">
            <property tag="snnfw:threshold" value="1.0"/>
            <property tag="snnfw:window_size_ms" value="25"/>
          </cell>
          <network id="net">
            <population id="pre" component="cell" size="1"/>
            <population id="post" component="cell" size="1"/>
            <projection id="pre_to_post" presynapticPopulation="pre" postsynapticPopulation="post" synapse="exc"/>
          </network>
        </neuroml>)xml");

    const auto hoc = write_fixture(
        "snncuda_test.hoc",
        R"hoc(begintemplate Cortical
          proc init() {
            threshold = 1
            window_size_ms = 20
            similarity_threshold = 0.9
          }
        endtemplate Cortical

        for i = 0, 0 { pre_cells.append(new Cortical()) }
        for i = 0, 0 { post_cells.append(new Cortical()) }
        for i = 0, 0 {
          nc = new NetCon(pre_cells.o(i).soma(0.5), post_cells.o(i).syn, 0, 1, 0.75)
        })hoc");

    const auto sonata_connectome = snncuda::declarative::ConnectomeBuilder{}.build(loader.parse_only(sonata));
    require(sonata_connectome.neurons.size() == 2, "SONATA parser should normalize populations");
    require(sonata_connectome.synapses.size() == 1, "SONATA parser should normalize projections");

    const auto neuroml_connectome = snncuda::declarative::ConnectomeBuilder{}.build(loader.parse_only(neuroml));
    require(neuroml_connectome.neurons.size() == 2, "NeuroML parser should normalize populations");
    require(neuroml_connectome.synapses.size() == 1, "NeuroML parser should normalize projections");

    const auto hoc_ir = loader.parse_only(hoc);
    require(hoc_ir.projections.size() == 1, "HOC parser should extract NetCon projections");
    require(hoc_ir.projections[0].source == "pre_cells", "HOC parser should extract NetCon source");
    require(hoc_ir.projections[0].target == "post_cells", "HOC parser should extract NetCon target");
    const auto hoc_connectome = snncuda::declarative::ConnectomeBuilder{}.build(hoc_ir);
    require(hoc_connectome.neurons.size() == 2, "HOC parser should normalize populations");
    require(hoc_connectome.populations.find("pre_cells") != hoc_connectome.populations.end(),
        "HOC connectome should index source population");
    require(hoc_connectome.populations.find("post_cells") != hoc_connectome.populations.end(),
        "HOC connectome should index target population");
    require(hoc_connectome.synapses.size() == 1, "HOC parser should normalize projections");
}

void test_sonata_hdf5_nodes_and_edges() {
#if SNNCUDA_HAS_HDF5
    if (std::system("python3 -c 'import h5py'") != 0) {
        return;
    }

    const auto root = make_temp_dir("snncuda_sonata_hdf5_fixture");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "networks");

    const auto script = root / "create_fixture.py";
    {
        std::ofstream output(script);
        output << R"py(import h5py
import pathlib

root = pathlib.Path(r")py" << root.string() << R"py(")
networks = root / "networks"
with h5py.File(networks / "nodes.h5", "w") as h5:
    g = h5.create_group("/nodes/v1")
    g.create_dataset("node_id", data=[0, 1])
    g.create_dataset("node_type_id", data=[1, 1])
    g = h5.create_group("/nodes/v2")
    g.create_dataset("node_id", data=[2])
    g.create_dataset("node_type_id", data=[2])

with h5py.File(networks / "edges.h5", "w") as h5:
    g = h5.create_group("/edges/v1_to_v2")
    g.create_dataset("source_node_id", data=[0, 1])
    g.create_dataset("target_node_id", data=[2, 2])
    g.create_dataset("weight", data=[0.5, 0.75])
    g.create_dataset("delay", data=[1.0, 2.0])
)py";
    }

    const auto command = "python3 " + script.string();
    require(std::system(command.c_str()) == 0, "failed to create SONATA HDF5 fixture");

    const auto config = root / "circuit_config.json";
    {
        std::ofstream output(config);
        output << R"json({
          "network_name": "HDF5 Fixture",
          "manifest": {
            "$BASE_DIR": ".",
            "$NETWORK_DIR": "$BASE_DIR/networks"
          },
          "networks": {
            "nodes": [
              { "nodes_file": "$NETWORK_DIR/nodes.h5" }
            ],
            "edges": [
              { "edges_file": "$NETWORK_DIR/edges.h5" }
            ]
          },
          "snncuda": {
            "neuron_params": {
              "default": { "threshold": 1.0 }
            }
          }
        })json";
    }

    snncuda::declarative::DeclarativeLoader loader;
    const auto ir = loader.parse_only(config);
    require(ir.source_format == "sonata", "SONATA HDF5 parser should preserve source format");
    require(ir.explicit_connections.size() == 2, "SONATA HDF5 parser should extract explicit edges");

    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(ir);
    require(connectome.neurons.size() == 3, "SONATA HDF5 nodes should become connectome neurons");
    require(connectome.synapses.size() == 2, "SONATA HDF5 edges should become connectome synapses");
    require(connectome.synapses[0].weight == 0.5F, "SONATA HDF5 edge weight should be preserved");
    require(connectome.synapses[1].delay_ticks == 2, "SONATA HDF5 edge delay should be preserved");
#endif
}

void test_spike_scheduler() {
    snncuda::runtime::SpikeScheduler scheduler{4};
    scheduler.schedule({.target_neuron = 10, .delivery_tick = 3, .weight = 1.0F});
    scheduler.schedule({.target_neuron = 11, .delivery_tick = 1, .weight = 1.0F});
    scheduler.schedule({.target_neuron = 12, .delivery_tick = 7, .weight = 1.0F});

    require(scheduler.pending_count() == 3, "scheduler should count pending spikes");

    const auto first = scheduler.pop_due(1);
    require(first.size() == 1, "scheduler should deliver only due early spike");
    require(first[0].target_neuron == 11, "scheduler should deliver correct early target");

    const auto skipped = scheduler.pop_due(6);
    require(skipped.size() == 1, "scheduler should support skipped ticks");
    require(skipped[0].target_neuron == 10, "scheduler should retain future wrapped bucket event");

    const auto wrapped = scheduler.pop_due(7);
    require(wrapped.size() == 1, "scheduler should deliver wrapped future spike");
    require(wrapped[0].target_neuron == 12, "scheduler should deliver correct wrapped target");
    require(scheduler.empty(), "scheduler should be empty after all due spikes");

    scheduler.schedule({.target_neuron = 13, .delivery_tick = 2, .weight = 1.0F});
    const auto late = scheduler.pop_due(8);
    require(late.size() == 1, "late spike should be delivered on next poll");
    require(late[0].delivery_tick == 8, "late spike delivery tick should be clamped");

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NeuronStateCache cache{2, store};
    snncuda::runtime::SynapseDendriteProcessor processor;

    auto state = cache.load_for_spike(42);
    const auto result = processor.process_external_input(
        state,
        {.target_neuron = 42, .delivery_tick = 1, .weight = 1.25F},
        1);
    cache.store_after_compute(state);
    require(result.fired, "external spike processing should report firing");
    state = cache.load_for_spike(42);
    require(state.spike_count == 1, "spike arrival should wake and update target neuron state");
}

} // namespace

int main() {
    try {
        test_clock();
        test_neuron();
        test_version();
        test_hierarchy();
        test_lru_cache();
        test_neuron_state_cache();
        test_stdp();
        test_temporal_pattern();
        test_declarative_loader_detection();
        test_all_declared_parsers_build_connectome();
        test_sonata_hdf5_nodes_and_edges();
        test_spike_scheduler();
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All smoke tests passed\n";
    return EXIT_SUCCESS;
}
