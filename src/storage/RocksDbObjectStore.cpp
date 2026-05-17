#include "snncuda/storage/RocksDbObjectStore.h"

#include <stdexcept>

namespace snncuda::storage {

class RocksDbObjectStore::Impl {
public:
    explicit Impl(std::filesystem::path path)
        : path_(std::move(path)) {
    }

    std::filesystem::path path_;
};

RocksDbObjectStore::RocksDbObjectStore(std::filesystem::path path)
    : impl_(std::make_unique<Impl>(std::move(path))) {
}

RocksDbObjectStore::~RocksDbObjectStore() = default;

void RocksDbObjectStore::put(const StoredObject&) {
    throw std::runtime_error("RocksDB support is not wired yet");
}

std::optional<StoredObject> RocksDbObjectStore::get(core::ObjectId) const {
    throw std::runtime_error("RocksDB support is not wired yet");
}

void RocksDbObjectStore::erase(core::ObjectId) {
    throw std::runtime_error("RocksDB support is not wired yet");
}

void RocksDbObjectStore::flush() {
}

} // namespace snncuda::storage

