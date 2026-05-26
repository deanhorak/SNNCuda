#include "snncuda/declarative/Connectome.h"
#include "snncuda/declarative/DeclarativeLoader.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using snncuda::declarative::Connectome;
using snncuda::declarative::ConnectomeBuilder;
using snncuda::declarative::DeclarativeLoader;
using snncuda::declarative::NetworkIR;

[[nodiscard]] std::filesystem::path fixture_dir() {
    return SNNCUDA_TEST_FIXTURE_DIR;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path make_temp_dir(const std::string& prefix) {
    const auto suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return std::filesystem::temp_directory_path() / (prefix + "_" + suffix);
}

[[nodiscard]] const auto& first_layer(const NetworkIR& ir) {
    return ir.brain.hemispheres.at(0)
        .lobes.at(0)
        .regions.at(0)
        .nuclei.at(0)
        .columns.at(0)
        .layers.at(0);
}

[[nodiscard]] std::size_t population_size(const Connectome& connectome, const std::string& name) {
    const auto found = connectome.populations.find(name);
    return found == connectome.populations.end() ? 0 : found->second.size();
}

[[nodiscard]] bool has_population_path_suffix(const Connectome& connectome, const std::string& suffix) {
    for (const auto& [path, _] : connectome.populations) {
        if (path.size() >= suffix.size()
            && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

void assert_valid_ir_tree(const NetworkIR& ir, const char* context) {
    const auto errors = ir.validate();
    if (!errors.empty()) {
        throw std::runtime_error(std::string(context) + " validation failed: " + errors.front());
    }

    require(!ir.brain.name.empty(), "IR brain name should be populated");
    require(!ir.brain.hemispheres.empty(), "IR should contain hemispheres");
    require(!ir.brain.hemispheres.front().lobes.empty(), "IR should contain lobes");
    require(!ir.brain.hemispheres.front().lobes.front().regions.empty(), "IR should contain regions");
    require(!ir.brain.hemispheres.front().lobes.front().regions.front().nuclei.empty(),
        "IR should contain nuclei");
    require(!ir.brain.hemispheres.front().lobes.front().regions.front().nuclei.front().columns.empty(),
        "IR should contain columns");
}

void test_native_hierarchy_fixture() {
    DeclarativeLoader loader;
    const auto ir = loader.parse_only(fixture_dir() / "native_hierarchy.snncuda.json");
    assert_valid_ir_tree(ir, "native");

    require(ir.source_format == "native-json", "native fixture should use native parser");
    require(ir.brain.name == "NativeBrain", "native brain name should be preserved");
    require(ir.brain.hemispheres.at(0).name == "Left Hemisphere", "native hemisphere should be preserved");
    require(ir.brain.hemispheres.at(0).lobes.at(0).regions.at(0).nuclei.at(0).columns.size() == 2,
        "native column template should expand into two columns");
    require(ir.projections.size() == 1, "native projection should be parsed");
    require(ir.simulation.cuda_resident_neurons == 128, "native simulation settings should parse");
    require(ir.neuron_params.at("sensory").threshold == 0.75F, "native neuron params should parse");

    const auto connectome = ConnectomeBuilder{}.build(ir);
    require(connectome.neurons.size() == 6, "native connectome should contain all template neurons");
    require(population_size(connectome, "sensory") == 4, "native sensory population count should flatten");
    require(population_size(connectome, "readout") == 2, "native readout population count should flatten");
    require(connectome.synapses.size() == 8, "native wildcard projection should connect all sources/targets");
    require(connectome.synapses.front().weight == 0.6F, "native projection weight should be preserved");
    require(connectome.synapses.front().delay_ticks == 2, "native projection delay should be preserved");
    require(connectome.synapses.front().spike_code_offsets == std::vector<std::uint32_t>({0, 2, 5}),
        "native projection spike code should be preserved");
    require(has_population_path_suffix(connectome, "Orient_0_Freq_3/L4/sensory"),
        "native connectome should retain abstract population path");
    require(has_population_path_suffix(connectome, "Orient_90_Freq_3/L5/readout"),
        "native connectome should retain second template population path");
}

void test_sonata_extension_fixture() {
    DeclarativeLoader loader;
    const auto ir = loader.parse_only(fixture_dir() / "sonata_extension.sonata.json");
    assert_valid_ir_tree(ir, "sonata extension");

    require(ir.source_format == "sonata", "SONATA extension fixture should use SONATA parser");
    require(ir.brain.name == "SonataFlat", "SONATA extension should parse flat network name");
    require(first_layer(ir).populations.size() == 2, "SONATA extension should parse populations");

    const auto connectome = ConnectomeBuilder{}.build(ir);
    require(connectome.neurons.size() == 4, "SONATA extension connectome should contain all neurons");
    require(population_size(connectome, "input") == 2, "SONATA extension input count should flatten");
    require(population_size(connectome, "output") == 2, "SONATA extension output count should flatten");
    require(connectome.synapses.size() == 2, "SONATA one-to-one projection should create two synapses");
    require(connectome.synapses.front().weight == 0.4F, "SONATA projection weight should be preserved");
    require(connectome.synapses.front().delay_ticks == 3, "SONATA projection delay should be preserved");
}

void test_neuroml_fixture() {
    DeclarativeLoader loader;
    const auto ir = loader.parse_only(fixture_dir() / "neuroml_basic.nml");
    assert_valid_ir_tree(ir, "neuroml");

    require(ir.source_format == "neuroml", "NeuroML fixture should use NeuroML parser");
    require(ir.neuron_params.at("excitatory_cell").threshold == 0.9F,
        "NeuroML cell threshold should parse");
    require(ir.neuron_params.at("excitatory_cell").pattern_window_ticks == 30,
        "NeuroML window should parse");
    require(first_layer(ir).populations.size() == 2, "NeuroML populations should parse");
    require(ir.projections.size() == 1, "NeuroML projection should parse");

    const auto connectome = ConnectomeBuilder{}.build(ir);
    require(connectome.neurons.size() == 4, "NeuroML connectome should contain all neurons");
    require(population_size(connectome, "pre") == 2, "NeuroML pre count should flatten");
    require(population_size(connectome, "post") == 2, "NeuroML post count should flatten");
    require(connectome.synapses.size() == 2, "NeuroML one-to-one projection should create two synapses");
}

void test_hoc_fixture() {
    DeclarativeLoader loader;
    const auto ir = loader.parse_only(fixture_dir() / "hoc_basic.hoc");
    assert_valid_ir_tree(ir, "hoc");

    require(ir.source_format == "hoc", "HOC fixture should use HOC parser");
    require(ir.neuron_params.at("CorticalCell").threshold == 0.8F, "HOC template threshold should parse");
    require(ir.neuron_params.at("CorticalCell").pattern_window_ticks == 35,
        "HOC template window should parse");
    require(first_layer(ir).populations.size() == 2, "HOC instantiated lists should become populations");
    require(ir.projections.size() == 1, "HOC NetCon should become projection");
    require(ir.projections.front().weight == 0.7F, "HOC NetCon weight should parse");
    require(ir.projections.front().delay_ticks == 2, "HOC NetCon delay should parse");

    const auto connectome = ConnectomeBuilder{}.build(ir);
    require(connectome.neurons.size() == 4, "HOC connectome should contain all neurons");
    require(population_size(connectome, "pre_cells") == 2, "HOC pre count should flatten");
    require(population_size(connectome, "post_cells") == 2, "HOC post count should flatten");
    require(connectome.synapses.size() == 2, "HOC one-to-one NetCon should create two synapses");
}

void copy_fixture_file(const std::filesystem::path& source, const std::filesystem::path& target) {
    std::ifstream input(source);
    std::ofstream output(target);
    if (!input || !output) {
        throw std::runtime_error("failed to copy fixture");
    }
    output << input.rdbuf();
}

void test_sonata_hdf5_fixture() {
#if SNNCUDA_HAS_HDF5
    if (std::system("python3 -c 'import h5py'") != 0) {
        return;
    }

    const auto root = make_temp_dir("snncuda_committed_sonata_hdf5_fixture");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    copy_fixture_file(fixture_dir() / "sonata_hdf5_circuit_config.json", root / "circuit_config.json");

    const auto command = "python3 " + (fixture_dir() / "create_sonata_hdf5_fixture.py").string()
        + " " + root.string();
    require(std::system(command.c_str()) == 0, "failed to generate committed SONATA HDF5 fixture");

    DeclarativeLoader loader;
    const auto ir = loader.parse_only(root / "circuit_config.json");
    assert_valid_ir_tree(ir, "sonata hdf5");

    require(ir.source_format == "sonata", "SONATA HDF5 fixture should use SONATA parser");
    require(ir.explicit_connections.size() == 2, "SONATA HDF5 explicit edges should parse");
    require(first_layer(ir).populations.size() == 2, "SONATA HDF5 node populations should parse");

    const auto connectome = ConnectomeBuilder{}.build(ir);
    require(connectome.neurons.size() == 4, "SONATA HDF5 connectome should contain all neurons");
    require(population_size(connectome, "v1") == 2, "SONATA HDF5 v1 count should flatten");
    require(population_size(connectome, "v2") == 2, "SONATA HDF5 v2 count should flatten");
    require(connectome.synapses.size() == 2, "SONATA HDF5 explicit edges should flatten");
    require(connectome.synapses[0].source == connectome.populations.at("v1")[0],
        "SONATA HDF5 edge should use source population-local node ids");
    require(connectome.synapses[0].target == connectome.populations.at("v2")[0],
        "SONATA HDF5 edge should use target population-local node ids");
    require(connectome.synapses[1].weight == 0.75F, "SONATA HDF5 second edge weight should parse");
    require(connectome.synapses[1].delay_ticks == 2, "SONATA HDF5 second edge delay should parse");
#endif
}

} // namespace

int main() {
    try {
        test_native_hierarchy_fixture();
        test_sonata_extension_fixture();
        test_neuroml_fixture();
        test_hoc_fixture();
        test_sonata_hdf5_fixture();
    } catch (const std::exception& error) {
        std::cerr << "Parser test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All parser tests passed\n";
    return EXIT_SUCCESS;
}
