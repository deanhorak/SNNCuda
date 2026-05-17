#pragma once

#include <string_view>

namespace snncuda::core {

std::string_view version() noexcept;
bool cuda_available() noexcept;

} // namespace snncuda::core

