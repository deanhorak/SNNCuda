#include "snncuda/declarative/NetworkIR.h"

namespace snncuda::declarative {

std::vector<std::string> NetworkIR::validate() const {
    std::vector<std::string> errors;

    if (brain.name.empty()) {
        errors.emplace_back("brain.name is required");
    }
    if (brain.hemispheres.empty()) {
        errors.emplace_back("brain must declare at least one hemisphere");
    }

    for (const auto& [name, params] : neuron_params) {
        if (name.empty()) {
            errors.emplace_back("neuron parameter set name cannot be empty");
        }
        if (params.threshold <= 0.0F) {
            errors.emplace_back("neuron threshold must be positive");
        }
    }

    for (const auto& projection : projections) {
        if (projection.name.empty()) {
            errors.emplace_back("projection.name is required");
        }
        if (projection.source.empty() || projection.target.empty()) {
            errors.emplace_back("projection source and target are required");
        }
    }

    return errors;
}

} // namespace snncuda::declarative

