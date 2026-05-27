#include "snncuda/c/SNNCudaC.h"

#include "snncuda/core/Version.h"

const char* snncuda_version(void) {
    return snncuda::core::version().data();
}

int snncuda_cuda_available(void) {
    return snncuda::core::cuda_available() ? 1 : 0;
}
