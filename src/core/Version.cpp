#include "snncuda/core/Version.h"

namespace snncuda::core {

std::string_view version() noexcept {
    return SNNCUDA_VERSION;
}

bool cuda_available() noexcept {
    return SNNCUDA_HAS_CUDA != 0;
}

} // namespace snncuda::core

