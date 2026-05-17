#pragma once

#include "snncuda/storage/ObjectStore.h"

#include <filesystem>
#include <memory>

namespace snncuda::storage {

class RocksDbObjectStore final : public ObjectStore {
public:
    explicit RocksDbObjectStore(std::filesystem::path path);
    ~RocksDbObjectStore() override;

    void put(const StoredObject& object) override;
    [[nodiscard]] std::optional<StoredObject> get(core::ObjectId id) const override;
    void erase(core::ObjectId id) override;
    void flush() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace snncuda::storage

