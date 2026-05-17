#pragma once

#include "snncuda/core/Ids.h"

#include <optional>
#include <string>

namespace snncuda::storage {

struct StoredObject {
    core::ObjectId id{core::invalid_id};
    std::string type;
    std::string payload;
};

class ObjectStore {
public:
    virtual ~ObjectStore() = default;

    virtual void put(const StoredObject& object) = 0;
    [[nodiscard]] virtual std::optional<StoredObject> get(core::ObjectId id) const = 0;
    virtual void erase(core::ObjectId id) = 0;
    virtual void flush() = 0;
};

} // namespace snncuda::storage

