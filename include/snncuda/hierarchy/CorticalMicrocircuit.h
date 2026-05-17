#pragma once

#include "snncuda/hierarchy/Hierarchy.h"

#include <array>
#include <string_view>

namespace snncuda::hierarchy {

enum class CorticalLayer {
    L1,
    L2_3,
    L4,
    L5,
    L6,
};

struct CorticalColumnIds {
    core::ObjectId column{core::invalid_id};
    std::array<core::ObjectId, 5> layers{};
};

[[nodiscard]] std::string_view to_string(CorticalLayer layer) noexcept;

CorticalColumnIds create_canonical_cortical_column(
    Hierarchy& hierarchy,
    core::ObjectId nucleus_id,
    std::string_view column_name);

} // namespace snncuda::hierarchy

