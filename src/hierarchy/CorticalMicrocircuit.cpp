#include "snncuda/hierarchy/CorticalMicrocircuit.h"

#include <stdexcept>

namespace snncuda::hierarchy {

std::string_view to_string(CorticalLayer layer) noexcept {
    switch (layer) {
    case CorticalLayer::L1:
        return "L1";
    case CorticalLayer::L2_3:
        return "L2/3";
    case CorticalLayer::L4:
        return "L4";
    case CorticalLayer::L5:
        return "L5";
    case CorticalLayer::L6:
        return "L6";
    }
    return "unknown";
}

CorticalColumnIds create_canonical_cortical_column(
    Hierarchy& hierarchy,
    core::ObjectId nucleus_id,
    std::string_view column_name) {
    if (hierarchy.find(nucleus_id) == nullptr) {
        throw std::invalid_argument("Cannot create cortical column without a valid nucleus");
    }

    CorticalColumnIds ids;
    ids.column = hierarchy.create_child(nucleus_id, NodeKind::Column, std::string(column_name));

    const std::array<CorticalLayer, 5> layers{
        CorticalLayer::L1,
        CorticalLayer::L2_3,
        CorticalLayer::L4,
        CorticalLayer::L5,
        CorticalLayer::L6,
    };

    for (std::size_t i = 0; i < layers.size(); ++i) {
        ids.layers[i] = hierarchy.create_child(
            ids.column,
            NodeKind::Layer,
            std::string(to_string(layers[i])));
    }

    return ids;
}

} // namespace snncuda::hierarchy

