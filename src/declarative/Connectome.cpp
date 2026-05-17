#include "snncuda/declarative/Connectome.h"

#include <algorithm>
#include <stdexcept>

namespace snncuda::declarative {
namespace {

[[nodiscard]] bool ends_with_path(const std::string& path, const std::string& suffix) {
    return path == suffix || (path.size() > suffix.size()
        && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0
        && path[path.size() - suffix.size() - 1] == '/');
}

void add_population(
    Connectome& connectome,
    core::NeuronId& next_neuron_id,
    const std::string& path,
    const PopulationIR& population,
    const std::unordered_map<std::string, NeuronParamsIR>& params) {
    const auto params_found = params.find(population.neuron_params);
    const auto neuron_params = params_found == params.end() ? NeuronParamsIR{} : params_found->second;
    auto& ids = connectome.populations[path];
    auto& short_ids = connectome.populations[population.name];

    for (std::size_t i = 0; i < population.count; ++i) {
        const auto id = next_neuron_id++;
        ids.push_back(id);
        short_ids.push_back(id);
        connectome.neurons.push_back({
            .id = id,
            .path = path + "/" + std::to_string(i),
            .population = population.name,
            .params = neuron_params,
        });
    }
}

void collect_population_paths(
    Connectome& connectome,
    core::NeuronId& next_neuron_id,
    const NetworkIR& ir,
    const std::string& prefix,
    const LayerIR& layer) {
    for (const auto& population : layer.populations) {
        add_population(
            connectome,
            next_neuron_id,
            prefix + "/" + layer.name + "/" + population.name,
            population,
            ir.neuron_params);
    }
}

[[nodiscard]] std::vector<core::NeuronId> resolve_population(
    const Connectome& connectome,
    const std::string& selector) {
    if (const auto found = connectome.populations.find(selector); found != connectome.populations.end()) {
        return found->second;
    }

    std::vector<core::NeuronId> ids;
    const auto wildcard = selector.find('*');
    if (wildcard == std::string::npos) {
        for (const auto& [path, population] : connectome.populations) {
            if (ends_with_path(path, selector)) {
                ids.insert(ids.end(), population.begin(), population.end());
            }
        }
        return ids;
    }

    const auto prefix = selector.substr(0, wildcard);
    const auto suffix = selector.substr(wildcard + 1);
    for (const auto& [path, population] : connectome.populations) {
        if (path.size() >= prefix.size() + suffix.size()
            && path.rfind(prefix, 0) == 0
            && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            ids.insert(ids.end(), population.begin(), population.end());
        }
    }
    return ids;
}

} // namespace

Connectome ConnectomeBuilder::build(const NetworkIR& ir) const {
    const auto errors = ir.validate();
    if (!errors.empty()) {
        throw std::invalid_argument(errors.front());
    }

    Connectome connectome;
    core::NeuronId next_neuron_id = 1;
    core::SynapseId next_synapse_id = 1;

    for (const auto& hemisphere : ir.brain.hemispheres) {
        for (const auto& lobe : hemisphere.lobes) {
            for (const auto& region : lobe.regions) {
                for (const auto& nucleus : region.nuclei) {
                    for (const auto& column : nucleus.columns) {
                        const auto prefix = ir.brain.name + "/" + hemisphere.name + "/" + lobe.name
                            + "/" + region.name + "/" + nucleus.name + "/" + column.name;
                        for (const auto& layer : column.layers) {
                            collect_population_paths(connectome, next_neuron_id, ir, prefix, layer);
                        }
                    }
                }
            }
        }
    }

    for (const auto& projection : ir.projections) {
        const auto sources = resolve_population(connectome, projection.source);
        const auto targets = resolve_population(connectome, projection.target);
        if (sources.empty() || targets.empty()) {
            continue;
        }

        if (projection.pattern == "one_to_one") {
            const auto count = std::min(sources.size(), targets.size());
            for (std::size_t i = 0; i < count; ++i) {
                connectome.synapses.push_back({
                    .id = next_synapse_id++,
                    .source = sources[i],
                    .target = targets[i],
                    .weight = projection.weight,
                    .max_weight = projection.max_weight,
                    .delay_ticks = projection.delay_ticks,
                });
            }
            continue;
        }

        for (const auto source : sources) {
            for (const auto target : targets) {
                connectome.synapses.push_back({
                    .id = next_synapse_id++,
                    .source = source,
                    .target = target,
                    .weight = projection.weight,
                    .max_weight = projection.max_weight,
                    .delay_ticks = projection.delay_ticks,
                });
            }
        }
    }

    for (const auto& connection : ir.explicit_connections) {
        const auto source_population = resolve_population(connectome, connection.source_population);
        const auto target_population = resolve_population(connectome, connection.target_population);
        if (connection.source_index >= source_population.size()
            || connection.target_index >= target_population.size()) {
            continue;
        }
        connectome.synapses.push_back({
            .id = next_synapse_id++,
            .source = source_population[connection.source_index],
            .target = target_population[connection.target_index],
            .weight = connection.weight,
            .max_weight = connection.max_weight,
            .delay_ticks = connection.delay_ticks,
        });
    }

    return connectome;
}

} // namespace snncuda::declarative
