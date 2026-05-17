#pragma once

#include <cstdint>

namespace snncuda::core {

using ObjectId = std::uint64_t;
using NeuronId = ObjectId;
using SynapseId = ObjectId;
using PopulationId = ObjectId;

constexpr ObjectId invalid_id = 0;

} // namespace snncuda::core

