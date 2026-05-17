#pragma once

#include "snncuda/storage/ObjectStore.h"

#include <unordered_map>

namespace snncuda::storage {

class MemoryObjectStore final : public ObjectStore {
public:
    void put(const StoredObject& object) override;
    [[nodiscard]] std::optional<StoredObject> get(core::ObjectId id) const override;
    void erase(core::ObjectId id) override;
    void flush() override;

private:
    std::unordered_map<core::ObjectId, StoredObject> objects_;
};

} // namespace snncuda::storage

