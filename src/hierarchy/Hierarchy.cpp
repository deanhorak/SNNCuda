#include "snncuda/hierarchy/Hierarchy.h"

#include <algorithm>
#include <stdexcept>

namespace snncuda::hierarchy {

core::ObjectId Hierarchy::create_root(std::string name) {
    if (!nodes_.empty()) {
        throw std::logic_error("Hierarchy root already exists");
    }

    const auto id = next_id();
    nodes_.push_back({id, core::invalid_id, NodeKind::Brain, std::move(name), {}});
    return id;
}

core::ObjectId Hierarchy::create_child(core::ObjectId parent_id, NodeKind kind, std::string name) {
    auto* parent = find(parent_id);
    if (parent == nullptr) {
        throw std::invalid_argument("Cannot create hierarchy child for unknown parent");
    }

    const auto id = next_id();
    nodes_.push_back({id, parent_id, kind, std::move(name), {}});
    parent = find(parent_id);
    parent->children.push_back(id);
    return id;
}

const HierarchyNode* Hierarchy::find(core::ObjectId id) const noexcept {
    const auto found = std::find_if(nodes_.begin(), nodes_.end(), [id](const auto& node) {
        return node.id == id;
    });
    return found == nodes_.end() ? nullptr : &(*found);
}

HierarchyNode* Hierarchy::find(core::ObjectId id) noexcept {
    const auto found = std::find_if(nodes_.begin(), nodes_.end(), [id](const auto& node) {
        return node.id == id;
    });
    return found == nodes_.end() ? nullptr : &(*found);
}

std::vector<core::ObjectId> Hierarchy::children(core::ObjectId id) const {
    const auto* node = find(id);
    if (node == nullptr) {
        return {};
    }
    return node->children;
}

std::string Hierarchy::path(core::ObjectId id) const {
    std::vector<std::string> names;
    auto current = find(id);
    while (current != nullptr) {
        names.push_back(current->name);
        current = find(current->parent_id);
    }

    std::string result;
    for (auto it = names.rbegin(); it != names.rend(); ++it) {
        if (!result.empty()) {
            result += '/';
        }
        result += *it;
    }
    return result;
}

std::vector<core::ObjectId> Hierarchy::find_by_kind(NodeKind kind) const {
    std::vector<core::ObjectId> ids;
    for (const auto& node : nodes_) {
        if (node.kind == kind) {
            ids.push_back(node.id);
        }
    }
    return ids;
}

core::ObjectId Hierarchy::next_id() noexcept {
    return next_id_++;
}

std::string_view to_string(NodeKind kind) noexcept {
    switch (kind) {
    case NodeKind::Brain:
        return "brain";
    case NodeKind::Hemisphere:
        return "hemisphere";
    case NodeKind::Lobe:
        return "lobe";
    case NodeKind::Region:
        return "region";
    case NodeKind::Nucleus:
        return "nucleus";
    case NodeKind::Column:
        return "column";
    case NodeKind::Layer:
        return "layer";
    case NodeKind::Cluster:
        return "cluster";
    case NodeKind::Neuron:
        return "neuron";
    }
    return "unknown";
}

} // namespace snncuda::hierarchy

