#pragma once

#include "snncuda/snn/Synapse.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace snncuda::declarative {

struct NeuronParamsIR {
    float threshold{1.0F};
    std::uint64_t pattern_window_ticks{500};
    float similarity_threshold{0.93F};
    std::size_t max_reference_patterns{500};
};

struct PopulationIR {
    std::string name;
    std::size_t count{0};
    std::string neuron_params;
    std::unordered_map<std::string, std::string> properties;
};

struct LayerIR {
    std::string name;
    std::vector<PopulationIR> populations;
};

struct ColumnIR {
    std::string name;
    std::vector<LayerIR> layers;
};

struct NucleusIR {
    std::string name;
    std::vector<ColumnIR> columns;
};

struct RegionIR {
    std::string name;
    std::vector<NucleusIR> nuclei;
};

struct LobeIR {
    std::string name;
    std::vector<RegionIR> regions;
};

struct HemisphereIR {
    std::string name;
    std::vector<LobeIR> lobes;
};

struct BrainIR {
    std::string name;
    std::vector<HemisphereIR> hemispheres;
};

struct ProjectionIR {
    std::string name;
    std::string source;
    std::string target;
    std::string pattern{"all_to_all"};
    float probability{1.0F};
    float weight{1.0F};
    float max_weight{2.0F};
    std::uint32_t delay_ticks{1};
    std::vector<std::uint32_t> spike_code_offsets{0};
    snn::DendriticCompartment compartment{snn::DendriticCompartment::Basal};
    snn::ReceptorType receptor{snn::ReceptorType::Ampa};
    bool plasticity_enabled{true};
    std::string scope{"global"};
};

struct ExplicitConnectionIR {
    std::string source_population;
    std::size_t source_index{0};
    std::string target_population;
    std::size_t target_index{0};
    float weight{1.0F};
    float max_weight{2.0F};
    std::uint32_t delay_ticks{1};
    std::vector<std::uint32_t> spike_code_offsets{0};
    snn::DendriticCompartment compartment{snn::DendriticCompartment::Basal};
    snn::ReceptorType receptor{snn::ReceptorType::Ampa};
    bool plasticity_enabled{true};
};

struct SimulationIR {
    std::uint64_t timestep_ns{1000000};
    std::size_t cuda_resident_neurons{65536};
    bool stdp_enabled{true};
};

struct NetworkIR {
    std::string source_format;
    BrainIR brain;
    std::unordered_map<std::string, NeuronParamsIR> neuron_params;
    std::vector<ProjectionIR> projections;
    std::vector<ExplicitConnectionIR> explicit_connections;
    SimulationIR simulation;

    [[nodiscard]] std::vector<std::string> validate() const;
};

} // namespace snncuda::declarative
