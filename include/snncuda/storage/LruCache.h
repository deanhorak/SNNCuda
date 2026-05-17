#pragma once

#include <cstddef>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace snncuda::storage {

template <typename Key, typename Value>
class LruCache {
public:
    explicit LruCache(std::size_t capacity)
        : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("LRU cache capacity must be nonzero");
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return entries_.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return capacity_;
    }

    [[nodiscard]] bool contains(const Key& key) const {
        return index_.find(key) != index_.end();
    }

    std::optional<Value> get(const Key& key) {
        auto found = index_.find(key);
        if (found == index_.end()) {
            return std::nullopt;
        }

        entries_.splice(entries_.begin(), entries_, found->second);
        return found->second->second;
    }

    std::optional<std::pair<Key, Value>> put(Key key, Value value) {
        auto found = index_.find(key);
        if (found != index_.end()) {
            found->second->second = std::move(value);
            entries_.splice(entries_.begin(), entries_, found->second);
            return std::nullopt;
        }

        entries_.emplace_front(std::move(key), std::move(value));
        index_[entries_.front().first] = entries_.begin();

        if (entries_.size() <= capacity_) {
            return std::nullopt;
        }

        auto evicted = std::move(entries_.back());
        index_.erase(evicted.first);
        entries_.pop_back();
        return evicted;
    }

private:
    using Entry = std::pair<Key, Value>;
    using List = std::list<Entry>;

    std::size_t capacity_;
    List entries_;
    std::unordered_map<Key, typename List::iterator> index_;
};

} // namespace snncuda::storage

