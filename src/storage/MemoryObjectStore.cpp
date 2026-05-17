#include "snncuda/storage/MemoryObjectStore.h"

namespace snncuda::storage {

void MemoryObjectStore::put(const StoredObject& object) {
    objects_[object.id] = object;
}

std::optional<StoredObject> MemoryObjectStore::get(core::ObjectId id) const {
    const auto found = objects_.find(id);
    if (found == objects_.end()) {
        return std::nullopt;
    }
    return found->second;
}

void MemoryObjectStore::erase(core::ObjectId id) {
    objects_.erase(id);
}

void MemoryObjectStore::flush() {
}

} // namespace snncuda::storage

