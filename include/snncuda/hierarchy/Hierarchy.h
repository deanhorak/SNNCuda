#pragma once

#include "snncuda/core/Ids.h"

#include <string>
#include <string_view>
#include <vector>

namespace snncuda::hierarchy {

enum class NodeKind {
    Brain,
    Hemisphere,
    Lobe,
    Region,
    Nucleus,
    Column,
    Layer,
    Cluster,
    Neuron,
};

struct HierarchyNode {
    core::ObjectId id{core::invalid_id};
    core::ObjectId parent_id{core::invalid_id};
    NodeKind kind{NodeKind::Brain};
    std::string name;
    std::vector<core::ObjectId> children;
};

class Hierarchy {
public:
    core::ObjectId create_root(std::string name);
    core::ObjectId create_child(core::ObjectId parent_id, NodeKind kind, std::string name);

    [[nodiscard]] const HierarchyNode* find(core::ObjectId id) const noexcept;
    [[nodiscard]] HierarchyNode* find(core::ObjectId id) noexcept;
    [[nodiscard]] std::vector<core::ObjectId> children(core::ObjectId id) const;
    [[nodiscard]] std::string path(core::ObjectId id) const;
    [[nodiscard]] std::vector<core::ObjectId> find_by_kind(NodeKind kind) const;

private:
    [[nodiscard]] core::ObjectId next_id() noexcept;

    core::ObjectId next_id_{1};
    std::vector<HierarchyNode> nodes_;
};

[[nodiscard]] std::string_view to_string(NodeKind kind) noexcept;

} // namespace snncuda::hierarchy

