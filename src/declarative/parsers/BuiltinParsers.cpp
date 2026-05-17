#include "snncuda/declarative/parsers/BuiltinParsers.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>

#if SNNCUDA_HAS_HDF5
#include <hdf5.h>
#endif

namespace snncuda::declarative {
namespace {

class JsonValue {
public:
    using Object = std::unordered_map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;

    JsonValue() = default;
    explicit JsonValue(std::nullptr_t) {}
    explicit JsonValue(bool value)
        : value_(value) {}
    explicit JsonValue(double value)
        : value_(value) {}
    explicit JsonValue(std::string value)
        : value_(std::move(value)) {}
    explicit JsonValue(Object value)
        : value_(std::move(value)) {}
    explicit JsonValue(Array value)
        : value_(std::move(value)) {}

    [[nodiscard]] bool is_object() const noexcept {
        return std::holds_alternative<Object>(value_);
    }

    [[nodiscard]] bool is_array() const noexcept {
        return std::holds_alternative<Array>(value_);
    }

    [[nodiscard]] bool is_string() const noexcept {
        return std::holds_alternative<std::string>(value_);
    }

    [[nodiscard]] bool is_number() const noexcept {
        return std::holds_alternative<double>(value_);
    }

    [[nodiscard]] bool is_bool() const noexcept {
        return std::holds_alternative<bool>(value_);
    }

    [[nodiscard]] const Object& object() const {
        return std::get<Object>(value_);
    }

    [[nodiscard]] const Array& array() const {
        return std::get<Array>(value_);
    }

    [[nodiscard]] const std::string& string() const {
        return std::get<std::string>(value_);
    }

    [[nodiscard]] double number() const {
        return std::get<double>(value_);
    }

    [[nodiscard]] bool boolean() const {
        return std::get<bool>(value_);
    }

    [[nodiscard]] const JsonValue* find(std::string_view key) const {
        if (!is_object()) {
            return nullptr;
        }
        const auto found = object().find(std::string(key));
        return found == object().end() ? nullptr : &found->second;
    }

private:
    std::variant<std::nullptr_t, bool, double, std::string, Object, Array> value_{nullptr};
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text)
        : text_(text) {}

    [[nodiscard]] JsonValue parse() {
        skip_ws();
        auto value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            throw std::runtime_error("Unexpected trailing JSON content");
        }
        return value;
    }

private:
    [[nodiscard]] JsonValue parse_value() {
        skip_ws();
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of JSON");
        }

        const auto c = text_[pos_];
        if (c == '{') {
            return JsonValue(parse_object());
        }
        if (c == '[') {
            return JsonValue(parse_array());
        }
        if (c == '"') {
            return JsonValue(parse_string());
        }
        if (c == 't' || c == 'f') {
            return JsonValue(parse_bool());
        }
        if (c == 'n') {
            consume_literal("null");
            return JsonValue(nullptr);
        }
        return JsonValue(parse_number());
    }

    [[nodiscard]] JsonValue::Object parse_object() {
        expect('{');
        JsonValue::Object object;
        skip_ws();
        if (try_consume('}')) {
            return object;
        }

        while (true) {
            const auto key = parse_string();
            skip_ws();
            expect(':');
            object.emplace(key, parse_value());
            skip_ws();
            if (try_consume('}')) {
                return object;
            }
            expect(',');
        }
    }

    [[nodiscard]] JsonValue::Array parse_array() {
        expect('[');
        JsonValue::Array array;
        skip_ws();
        if (try_consume(']')) {
            return array;
        }

        while (true) {
            array.push_back(parse_value());
            skip_ws();
            if (try_consume(']')) {
                return array;
            }
            expect(',');
        }
    }

    [[nodiscard]] std::string parse_string() {
        expect('"');
        std::string result;
        while (pos_ < text_.size()) {
            const auto c = text_[pos_++];
            if (c == '"') {
                return result;
            }
            if (c != '\\') {
                result.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) {
                throw std::runtime_error("Invalid JSON escape");
            }
            const auto escaped = text_[pos_++];
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                result.push_back(escaped);
                break;
            case 'b':
                result.push_back('\b');
                break;
            case 'f':
                result.push_back('\f');
                break;
            case 'n':
                result.push_back('\n');
                break;
            case 'r':
                result.push_back('\r');
                break;
            case 't':
                result.push_back('\t');
                break;
            default:
                throw std::runtime_error("Unsupported JSON escape");
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    [[nodiscard]] bool parse_bool() {
        if (text_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return true;
        }
        if (text_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return false;
        }
        throw std::runtime_error("Invalid JSON boolean");
    }

    [[nodiscard]] double parse_number() {
        const auto start = pos_;
        if (text_[pos_] == '-') {
            ++pos_;
        }
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }

        return std::stod(std::string(text_.substr(start, pos_ - start)));
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    void expect(char expected) {
        skip_ws();
        if (pos_ >= text_.size() || text_[pos_] != expected) {
            throw std::runtime_error("Unexpected JSON token");
        }
        ++pos_;
    }

    [[nodiscard]] bool try_consume(char expected) {
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    void consume_literal(std::string_view literal) {
        if (text_.substr(pos_, literal.size()) != literal) {
            throw std::runtime_error("Invalid JSON literal");
        }
        pos_ += literal.size();
    }

    std::string_view text_;
    std::size_t pos_{0};
};

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

[[nodiscard]] std::string lowercase_extension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension;
}

[[nodiscard]] bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size()
        && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

[[nodiscard]] std::filesystem::path resolve_manifest_path(
    std::string raw_path,
    const std::unordered_map<std::string, std::string>& manifest,
    const std::filesystem::path& base_dir) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& [key, value] : manifest) {
            const auto pos = raw_path.find(key);
            if (pos != std::string::npos) {
                raw_path.replace(pos, key.size(), value);
                changed = true;
            }
        }
    }

    std::filesystem::path path(raw_path);
    if (path.is_relative()) {
        path = base_dir / path;
    }
    return path.lexically_normal();
}

[[nodiscard]] std::unordered_map<std::string, std::string> parse_manifest(
    const JsonValue* manifest,
    const std::filesystem::path& base_dir) {
    std::unordered_map<std::string, std::string> values;
    if (manifest == nullptr || !manifest->is_object()) {
        values["$BASE_DIR"] = base_dir.string();
        return values;
    }

    for (const auto& [key, value] : manifest->object()) {
        if (value.is_string()) {
            values[key] = value.string();
        }
    }
    values.try_emplace("$BASE_DIR", base_dir.string());

    for (auto& [_, value] : values) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& [key, replacement] : values) {
                const auto pos = value.find(key);
                if (pos != std::string::npos && key != value) {
                    value.replace(pos, key.size(), replacement);
                    changed = true;
                }
            }
        }
    }
    return values;
}

[[nodiscard]] const JsonValue* find_any(const JsonValue& value, std::initializer_list<std::string_view> keys) {
    for (const auto key : keys) {
        if (const auto* found = value.find(key)) {
            return found;
        }
    }
    return nullptr;
}

[[nodiscard]] std::string json_string(const JsonValue* value, std::string fallback = {}) {
    return value != nullptr && value->is_string() ? value->string() : std::move(fallback);
}

[[nodiscard]] float json_float(const JsonValue* value, float fallback) {
    return value != nullptr && value->is_number() ? static_cast<float>(value->number()) : fallback;
}

[[nodiscard]] std::uint64_t json_u64(const JsonValue* value, std::uint64_t fallback) {
    return value != nullptr && value->is_number() ? static_cast<std::uint64_t>(value->number()) : fallback;
}

[[nodiscard]] std::size_t json_size(const JsonValue* value, std::size_t fallback) {
    return value != nullptr && value->is_number() ? static_cast<std::size_t>(value->number()) : fallback;
}

[[nodiscard]] bool json_bool(const JsonValue* value, bool fallback) {
    return value != nullptr && value->is_bool() ? value->boolean() : fallback;
}

[[nodiscard]] NeuronParamsIR parse_neuron_params(const JsonValue& value) {
    NeuronParamsIR params;
    params.threshold = json_float(value.find("threshold"), params.threshold);
    params.pattern_window_ticks = json_u64(
        find_any(value, {"pattern_window_ticks", "window_ticks", "window_size_ms"}),
        params.pattern_window_ticks);
    params.similarity_threshold = json_float(
        value.find("similarity_threshold"),
        params.similarity_threshold);
    params.max_reference_patterns = json_size(
        find_any(value, {"max_reference_patterns", "max_patterns"}),
        params.max_reference_patterns);
    return params;
}

[[nodiscard]] PopulationIR parse_population(const JsonValue& value) {
    PopulationIR population;
    population.name = json_string(value.find("name"), "population");
    population.count = json_size(value.find("count"), json_size(value.find("size"), 0));
    population.neuron_params = json_string(value.find("neuron_params"), "default");
    if (population.neuron_params.empty()) {
        population.neuron_params = "default";
    }
    return population;
}

[[nodiscard]] LayerIR parse_layer(const JsonValue& value) {
    LayerIR layer;
    layer.name = json_string(value.find("name"), "Layer");
    if (const auto* populations = value.find("populations");
        populations != nullptr && populations->is_array()) {
        for (const auto& population : populations->array()) {
            layer.populations.push_back(parse_population(population));
        }
    }
    return layer;
}

[[nodiscard]] std::string trim_number(double value) {
    auto text = std::to_string(value);
    while (!text.empty() && text.back() == '0') {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.') {
        text.pop_back();
    }
    return text;
}

[[nodiscard]] std::vector<ColumnIR> parse_columns(const JsonValue& nucleus) {
    std::vector<ColumnIR> columns;
    if (const auto* explicit_columns = nucleus.find("columns");
        explicit_columns != nullptr && explicit_columns->is_array()) {
        for (const auto& column_value : explicit_columns->array()) {
            ColumnIR column;
            column.name = json_string(column_value.find("name"), "Column");
            if (const auto* layers = column_value.find("layers"); layers != nullptr && layers->is_array()) {
                for (const auto& layer : layers->array()) {
                    column.layers.push_back(parse_layer(layer));
                }
            }
            columns.push_back(std::move(column));
        }
    }

    if (const auto* templ = nucleus.find("column_template");
        templ != nullptr && templ->is_object()) {
        std::vector<double> orientations{0.0};
        std::vector<double> frequencies{0.0};
        if (const auto* values = templ->find("orientations"); values != nullptr && values->is_array()) {
            orientations.clear();
            for (const auto& item : values->array()) {
                if (item.is_number()) {
                    orientations.push_back(item.number());
                }
            }
        }
        if (const auto* values = templ->find("frequencies"); values != nullptr && values->is_array()) {
            frequencies.clear();
            for (const auto& item : values->array()) {
                if (item.is_number()) {
                    frequencies.push_back(item.number());
                }
            }
        }

        const auto pattern = json_string(templ->find("naming_pattern"), "Orient_{orientation}_Freq_{frequency}");
        for (const auto orientation : orientations) {
            for (const auto frequency : frequencies) {
                auto name = pattern;
                const auto orientation_text = trim_number(orientation);
                const auto frequency_text = trim_number(frequency);
                if (const auto pos = name.find("{orientation}"); pos != std::string::npos) {
                    name.replace(pos, 13, orientation_text);
                }
                if (const auto pos = name.find("{frequency}"); pos != std::string::npos) {
                    name.replace(pos, 11, frequency_text);
                }

                ColumnIR column;
                column.name = name;
                if (const auto* layers = templ->find("layers"); layers != nullptr && layers->is_array()) {
                    for (const auto& layer : layers->array()) {
                        column.layers.push_back(parse_layer(layer));
                    }
                }
                columns.push_back(std::move(column));
            }
        }
    }

    return columns;
}

[[nodiscard]] BrainIR parse_brain(const JsonValue* brain_value, std::string fallback_name) {
    BrainIR brain;
    brain.name = std::move(fallback_name);
    if (brain_value == nullptr || !brain_value->is_object()) {
        brain.hemispheres.push_back({"Left", {}});
        return brain;
    }

    brain.name = json_string(brain_value->find("name"), brain.name.empty() ? "Brain" : brain.name);
    if (const auto* hemispheres = brain_value->find("hemispheres");
        hemispheres != nullptr && hemispheres->is_array()) {
        for (const auto& hemisphere_value : hemispheres->array()) {
            HemisphereIR hemisphere;
            hemisphere.name = json_string(hemisphere_value.find("name"), "Hemisphere");
            if (const auto* lobes = hemisphere_value.find("lobes"); lobes != nullptr && lobes->is_array()) {
                for (const auto& lobe_value : lobes->array()) {
                    LobeIR lobe;
                    lobe.name = json_string(lobe_value.find("name"), "Lobe");
                    if (const auto* regions = lobe_value.find("regions");
                        regions != nullptr && regions->is_array()) {
                        for (const auto& region_value : regions->array()) {
                            RegionIR region;
                            region.name = json_string(region_value.find("name"), "Region");
                            if (const auto* nuclei = region_value.find("nuclei");
                                nuclei != nullptr && nuclei->is_array()) {
                                for (const auto& nucleus_value : nuclei->array()) {
                                    NucleusIR nucleus;
                                    nucleus.name = json_string(nucleus_value.find("name"), "Nucleus");
                                    nucleus.columns = parse_columns(nucleus_value);
                                    region.nuclei.push_back(std::move(nucleus));
                                }
                            }
                            lobe.regions.push_back(std::move(region));
                        }
                    }
                    hemisphere.lobes.push_back(std::move(lobe));
                }
            }
            brain.hemispheres.push_back(std::move(hemisphere));
        }
    }

    if (brain.hemispheres.empty()) {
        brain.hemispheres.push_back({"Left", {}});
    }
    return brain;
}

void parse_neuron_param_map(const JsonValue* value, NetworkIR& ir) {
    if (value == nullptr || !value->is_object()) {
        ir.neuron_params["default"] = {};
        return;
    }

    for (const auto& [name, params] : value->object()) {
        if (params.is_object()) {
            ir.neuron_params[name] = parse_neuron_params(params);
        }
    }
    if (ir.neuron_params.empty()) {
        ir.neuron_params["default"] = {};
    }
}

[[nodiscard]] ProjectionIR parse_projection(const JsonValue& value) {
    ProjectionIR projection;
    projection.name = json_string(value.find("name"), "projection");
    projection.source = json_string(value.find("source"));
    projection.target = json_string(value.find("target"));
    projection.pattern = json_string(value.find("pattern"), projection.pattern);
    projection.probability = json_float(value.find("probability"), projection.probability);
    projection.weight = json_float(value.find("weight"), projection.weight);
    projection.max_weight = json_float(value.find("max_weight"), projection.max_weight);
    projection.delay_ticks = static_cast<std::uint32_t>(
        json_u64(find_any(value, {"delay_ticks", "delay_ms", "delay"}), projection.delay_ticks));
    projection.scope = json_string(value.find("scope"), projection.scope);
    return projection;
}

void parse_projection_array(const JsonValue* value, NetworkIR& ir) {
    if (value == nullptr || !value->is_array()) {
        return;
    }
    for (const auto& projection : value->array()) {
        if (projection.is_object()) {
            ir.projections.push_back(parse_projection(projection));
        }
    }
}

void parse_simulation(const JsonValue* value, NetworkIR& ir) {
    if (value == nullptr || !value->is_object()) {
        return;
    }
    ir.simulation.cuda_resident_neurons = json_size(
        find_any(*value, {"cuda_resident_neurons", "resident_neurons"}),
        ir.simulation.cuda_resident_neurons);
    ir.simulation.stdp_enabled = json_bool(value->find("stdp_enabled"), ir.simulation.stdp_enabled);
    if (const auto* stdp = value->find("stdp"); stdp != nullptr && stdp->is_object()) {
        ir.simulation.stdp_enabled = json_bool(stdp->find("enabled"), ir.simulation.stdp_enabled);
    }
}

[[nodiscard]] std::vector<std::filesystem::path> sonata_files(
    const JsonValue& root,
    std::string_view section,
    std::string_view field,
    const std::filesystem::path& base_dir) {
    std::vector<std::filesystem::path> paths;
    const auto manifest = parse_manifest(root.find("manifest"), base_dir);
    const auto* networks = root.find("networks");
    if (networks == nullptr || !networks->is_object()) {
        return paths;
    }
    const auto* entries = networks->find(section);
    if (entries == nullptr || !entries->is_array()) {
        return paths;
    }

    for (const auto& entry : entries->array()) {
        if (!entry.is_object()) {
            continue;
        }
        const auto raw = json_string(entry.find(field));
        if (!raw.empty()) {
            paths.push_back(resolve_manifest_path(raw, manifest, base_dir));
        }
    }
    return paths;
}

#if SNNCUDA_HAS_HDF5
struct H5Handle {
    hid_t id{-1};
    herr_t (*closer)(hid_t){nullptr};

    H5Handle() = default;
    H5Handle(hid_t value, herr_t (*close_fn)(hid_t))
        : id(value)
        , closer(close_fn) {
    }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept
        : id(std::exchange(other.id, -1))
        , closer(std::exchange(other.closer, nullptr)) {
    }
    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) {
            close();
            id = std::exchange(other.id, -1);
            closer = std::exchange(other.closer, nullptr);
        }
        return *this;
    }
    ~H5Handle() {
        close();
    }

    [[nodiscard]] bool valid() const noexcept {
        return id >= 0;
    }

    void close() noexcept {
        if (valid() && closer != nullptr) {
            closer(id);
        }
        id = -1;
        closer = nullptr;
    }
};

[[nodiscard]] bool h5_exists(hid_t parent, const std::string& path) {
    return H5Lexists(parent, path.c_str(), H5P_DEFAULT) > 0;
}

[[nodiscard]] std::vector<std::string> h5_child_names(hid_t parent, const std::string& path) {
    std::vector<std::string> names;
    if (!h5_exists(parent, path)) {
        return names;
    }
    H5Handle group(H5Gopen2(parent, path.c_str(), H5P_DEFAULT), H5Gclose);
    if (!group.valid()) {
        return names;
    }

    const auto callback = [](hid_t, const char* name, const H5L_info_t*, void* data) -> herr_t {
        auto* output = static_cast<std::vector<std::string>*>(data);
        output->push_back(name);
        return 0;
    };

    hsize_t index = 0;
    H5Literate(group.id, H5_INDEX_NAME, H5_ITER_INC, &index, callback, &names);
    return names;
}

template <typename T>
[[nodiscard]] std::vector<T> h5_read_1d(hid_t file, const std::string& path, hid_t native_type) {
    if (!h5_exists(file, path)) {
        return {};
    }
    H5Handle dataset(H5Dopen2(file, path.c_str(), H5P_DEFAULT), H5Dclose);
    if (!dataset.valid()) {
        return {};
    }
    H5Handle space(H5Dget_space(dataset.id), H5Sclose);
    if (!space.valid() || H5Sget_simple_extent_ndims(space.id) != 1) {
        return {};
    }

    hsize_t dims[1]{0};
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    std::vector<T> values(static_cast<std::size_t>(dims[0]));
    if (!values.empty()) {
        H5Dread(dataset.id, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    }
    return values;
}

using SonataNodeLookup = std::unordered_map<std::string, std::unordered_map<long long, std::size_t>>;

void load_sonata_nodes_hdf5(
    const std::vector<std::filesystem::path>& node_files,
    NetworkIR& ir,
    SonataNodeLookup& node_lookup) {
    if (node_files.empty()) {
        return;
    }

    if (ir.brain.hemispheres.empty()) {
        ir.brain = parse_brain(nullptr, ir.brain.name.empty() ? "SONATA Network" : ir.brain.name);
    }

    auto& hemisphere = ir.brain.hemispheres.front();
    if (hemisphere.lobes.empty()) {
        LobeIR lobe;
        lobe.name = "Lobe";
        hemisphere.lobes.push_back(std::move(lobe));
    }
    auto& lobe = hemisphere.lobes.front();
    if (lobe.regions.empty()) {
        RegionIR region;
        region.name = "Region";
        lobe.regions.push_back(std::move(region));
    }
    auto& region = lobe.regions.front();
    if (region.nuclei.empty()) {
        NucleusIR nucleus;
        nucleus.name = "SONATA";
        region.nuclei.push_back(std::move(nucleus));
    }
    auto& nucleus = region.nuclei.front();
    if (nucleus.columns.empty()) {
        ColumnIR column;
        column.name = "Column";
        LayerIR layer;
        layer.name = "Layer";
        column.layers.push_back(std::move(layer));
        nucleus.columns.push_back(std::move(column));
    }
    auto& layer = nucleus.columns.front().layers.front();

    for (const auto& path : node_files) {
        H5Handle file(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
        if (!file.valid()) {
            throw std::runtime_error("Unable to open SONATA nodes file " + path.string());
        }

        for (const auto& population_name : h5_child_names(file.id, "/nodes")) {
            const auto group = "/nodes/" + population_name;
            auto node_ids = h5_read_1d<long long>(file.id, group + "/node_id", H5T_NATIVE_LLONG);
            if (node_ids.empty()) {
                auto node_type_ids = h5_read_1d<int>(file.id, group + "/node_type_id", H5T_NATIVE_INT);
                node_ids.reserve(node_type_ids.size());
                for (std::size_t i = 0; i < node_type_ids.size(); ++i) {
                    node_ids.push_back(static_cast<long long>(i));
                }
            }

            PopulationIR population;
            population.name = population_name;
            population.count = node_ids.size();
            population.neuron_params = "default";
            const auto existing = layer.populations.size();
            layer.populations.push_back(std::move(population));
            for (std::size_t i = 0; i < node_ids.size(); ++i) {
                node_lookup[layer.populations[existing].name][node_ids[i]] = i;
            }
        }
    }
}

[[nodiscard]] std::pair<std::string, std::string> infer_edge_populations(
    const std::string& edge_population,
    const SonataNodeLookup& node_lookup) {
    const auto separator = edge_population.find("_to_");
    if (separator != std::string::npos) {
        const auto source = edge_population.substr(0, separator);
        const auto target = edge_population.substr(separator + 4);
        if (node_lookup.find(source) != node_lookup.end()
            && node_lookup.find(target) != node_lookup.end()) {
            return {source, target};
        }
    }

    if (node_lookup.size() == 1) {
        const auto& name = node_lookup.begin()->first;
        return {name, name};
    }

    if (node_lookup.size() >= 2) {
        auto it = node_lookup.begin();
        const auto first = it->first;
        ++it;
        return {first, it->first};
    }

    return {};
}

void load_sonata_edges_hdf5(
    const std::vector<std::filesystem::path>& edge_files,
    NetworkIR& ir,
    const SonataNodeLookup& node_lookup) {
    for (const auto& path : edge_files) {
        H5Handle file(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
        if (!file.valid()) {
            throw std::runtime_error("Unable to open SONATA edges file " + path.string());
        }

        for (const auto& population_name : h5_child_names(file.id, "/edges")) {
            const auto group = "/edges/" + population_name;
            const auto [source_population, target_population] = infer_edge_populations(population_name, node_lookup);
            if (source_population.empty() || target_population.empty()) {
                continue;
            }
            const auto& source_nodes = node_lookup.at(source_population);
            const auto& target_nodes = node_lookup.at(target_population);
            const auto sources = h5_read_1d<long long>(file.id, group + "/source_node_id", H5T_NATIVE_LLONG);
            const auto targets = h5_read_1d<long long>(file.id, group + "/target_node_id", H5T_NATIVE_LLONG);
            const auto weights = h5_read_1d<float>(file.id, group + "/weight", H5T_NATIVE_FLOAT);
            const auto delays = h5_read_1d<float>(file.id, group + "/delay", H5T_NATIVE_FLOAT);
            const auto count = std::min(sources.size(), targets.size());

            for (std::size_t i = 0; i < count; ++i) {
                const auto source = source_nodes.find(sources[i]);
                const auto target = target_nodes.find(targets[i]);
                if (source == source_nodes.end() || target == target_nodes.end()) {
                    continue;
                }
                ir.explicit_connections.push_back({
                    .source_population = source_population,
                    .source_index = source->second,
                    .target_population = target_population,
                    .target_index = target->second,
                    .weight = i < weights.size() ? weights[i] : 1.0F,
                    .max_weight = 2.0F,
                    .delay_ticks = static_cast<std::uint32_t>(i < delays.size() ? std::max(1.0F, delays[i]) : 1.0F),
                });
            }
        }
    }
}
#endif

void parse_flat_network(const JsonValue* value, NetworkIR& ir) {
    if (value == nullptr || !value->is_object() || value->find("populations") == nullptr) {
        return;
    }

    ir.brain.name = json_string(value->find("name"), "Network");
    ColumnIR column;
    column.name = "Column";
    LayerIR layer;
    layer.name = "Layer";
    if (const auto* populations = value->find("populations"); populations != nullptr && populations->is_array()) {
        for (const auto& population : populations->array()) {
            layer.populations.push_back(parse_population(population));
        }
    }
    column.layers.push_back(std::move(layer));
    NucleusIR nucleus{.name = "Nucleus", .columns = {std::move(column)}};
    RegionIR region{.name = "Region", .nuclei = {std::move(nucleus)}};
    LobeIR lobe{.name = "Lobe", .regions = {std::move(region)}};
    HemisphereIR hemisphere{.name = "Left", .lobes = {std::move(lobe)}};
    ir.brain.hemispheres = {std::move(hemisphere)};
}

[[nodiscard]] NetworkIR parse_native_json_document(const JsonValue& root, std::string format) {
    const auto* document = &root;
    if (const auto* snncuda = root.find("snncuda")) {
        document = snncuda;
    } else if (const auto* snnframe = root.find("snnframe")) {
        document = snnframe;
    }

    NetworkIR ir;
    ir.source_format = std::move(format);
    parse_neuron_param_map(document->find("neuron_params"), ir);
    ir.brain = parse_brain(document->find("brain"), json_string(document->find("network_name"), "Brain"));
    parse_flat_network(document->find("network"), ir);
    parse_projection_array(document->find("projections"), ir);
    if (const auto* network = document->find("network"); network != nullptr && network->is_object()) {
        parse_projection_array(network->find("projections"), ir);
    }
    parse_simulation(document->find("simulation"), ir);
    return ir;
}

[[nodiscard]] std::string xml_attr(const std::string& text, std::string_view attr) {
    const std::regex pattern(std::string(attr) + "=\"([^\"]*)\"");
    std::smatch match;
    return std::regex_search(text, match, pattern) ? match[1].str() : std::string{};
}

[[nodiscard]] std::string property_value(const std::string& block, std::string_view tag) {
    const std::regex pattern(
        "<property\\s+[^>]*tag=\"" + std::string(tag)
        + "\"[^>]*value=\"([^\"]*)\"[^>]*/?>");
    std::smatch match;
    return std::regex_search(block, match, pattern) ? match[1].str() : std::string{};
}

[[nodiscard]] std::vector<std::string> split_arguments(std::string_view text) {
    std::vector<std::string> args;
    std::string current;
    int depth = 0;
    for (const auto c : text) {
        if (c == '(') {
            ++depth;
            current.push_back(c);
            continue;
        }
        if (c == ')') {
            --depth;
            current.push_back(c);
            continue;
        }
        if (c == ',' && depth == 0) {
            args.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty()) {
        args.push_back(current);
    }
    for (auto& arg : args) {
        const auto first = arg.find_first_not_of(" \t\r\n");
        const auto last = arg.find_last_not_of(" \t\r\n");
        arg = first == std::string::npos ? std::string{} : arg.substr(first, last - first + 1);
    }
    return args;
}

[[nodiscard]] std::string infer_hoc_list_name(const std::string& expression) {
    const std::regex list_pattern(R"((\w+_cells)\.o\(i\))");
    std::smatch match;
    return std::regex_search(expression, match, list_pattern) ? match[1].str() : std::string{};
}

} // namespace

bool NativeJsonParser::can_parse(const std::filesystem::path& path) const {
    return ends_with(path.filename().string(), ".snnf.json")
        || ends_with(path.filename().string(), ".snncuda.json");
}

std::string NativeJsonParser::format_name() const {
    return "native-json";
}

NetworkIR NativeJsonParser::parse(const std::filesystem::path& path) const {
    return parse_native_json_document(JsonParser(read_file(path)).parse(), format_name());
}

bool SonataParser::can_parse(const std::filesystem::path& path) const {
    const auto name = path.filename().string();
    return name == "circuit_config.json" || ends_with(name, ".sonata.json");
}

std::string SonataParser::format_name() const {
    return "sonata";
}

NetworkIR SonataParser::parse(const std::filesystem::path& path) const {
    auto root = JsonParser(read_file(path)).parse();
    NetworkIR ir;
    ir.source_format = format_name();
    if (const auto* snncuda = root.find("snncuda")) {
        ir = parse_native_json_document(*snncuda, format_name());
    } else if (const auto* snnframe = root.find("snnframe")) {
        ir = parse_native_json_document(*snnframe, format_name());
    } else {
        ir.brain = parse_brain(nullptr, json_string(root.find("network_name"), "SONATA Network"));
        ir.neuron_params["default"] = {};
    }
    ir.source_format = format_name();

    const auto base_dir = path.parent_path().empty() ? std::filesystem::current_path() : path.parent_path();
    const auto node_files = sonata_files(root, "nodes", "nodes_file", base_dir);
    const auto edge_files = sonata_files(root, "edges", "edges_file", base_dir);
    if (!node_files.empty() || !edge_files.empty()) {
#if SNNCUDA_HAS_HDF5
        SonataNodeLookup node_lookup;
        load_sonata_nodes_hdf5(node_files, ir, node_lookup);
        load_sonata_edges_hdf5(edge_files, ir, node_lookup);
#else
        throw std::runtime_error("SONATA HDF5 files were declared, but SNNCuda was built without HDF5 support");
#endif
    }

    return ir;
}

bool NeuroMLParser::can_parse(const std::filesystem::path& path) const {
    const auto extension = lowercase_extension(path);
    return extension == ".nml" || extension == ".neuroml";
}

std::string NeuroMLParser::format_name() const {
    return "neuroml";
}

NetworkIR NeuroMLParser::parse(const std::filesystem::path& path) const {
    const auto text = read_file(path);
    NetworkIR ir;
    ir.source_format = format_name();
    ir.brain.name = "NeuroML Network";

    const std::regex cell_pattern(
        "<cell\\s+[^>]*id=\"([^\"]*)\"[^>]*>([\\s\\S]*?)</cell>",
        std::regex_constants::icase);
    for (std::sregex_iterator it(text.begin(), text.end(), cell_pattern), end; it != end; ++it) {
        const auto id = (*it)[1].str();
        const auto block = (*it)[2].str();
        NeuronParamsIR params;
        if (const auto value = property_value(block, "snnfw:threshold"); !value.empty()) {
            params.threshold = std::stof(value);
        }
        if (const auto value = property_value(block, "snnfw:window_size_ms"); !value.empty()) {
            params.pattern_window_ticks = static_cast<std::uint64_t>(std::stoull(value));
        }
        if (const auto value = property_value(block, "snnfw:similarity_threshold"); !value.empty()) {
            params.similarity_threshold = std::stof(value);
        }
        if (const auto value = property_value(block, "snnfw:max_reference_patterns"); !value.empty()) {
            params.max_reference_patterns = static_cast<std::size_t>(std::stoull(value));
        }
        ir.neuron_params[id] = params;
    }
    if (ir.neuron_params.empty()) {
        ir.neuron_params["default"] = {};
    }

    ColumnIR column;
    column.name = "Column";
    std::unordered_map<std::string, std::size_t> layer_index;
    const std::regex population_pattern(R"(<population\b[^>]*>)", std::regex_constants::icase);
    for (std::sregex_iterator it(text.begin(), text.end(), population_pattern), end; it != end; ++it) {
        const auto tag = (*it)[0].str();
        const auto id = xml_attr(tag, "id");
        const auto component = xml_attr(tag, "component");
        const auto size_text = xml_attr(tag, "size");
        PopulationIR population;
        population.name = id.empty() ? "population" : id;
        population.neuron_params = component.empty() ? "default" : component;
        population.count = size_text.empty() ? 0 : static_cast<std::size_t>(std::stoull(size_text));

        const auto layer_name = property_value(tag, "snnfw:layer").empty()
            ? std::string("Layer")
            : property_value(tag, "snnfw:layer");
        auto found = layer_index.find(layer_name);
        if (found == layer_index.end()) {
            found = layer_index.emplace(layer_name, column.layers.size()).first;
            LayerIR layer;
            layer.name = layer_name;
            column.layers.push_back(std::move(layer));
        }
        column.layers[found->second].populations.push_back(std::move(population));
    }

    const std::regex projection_pattern(R"(<projection\b[^>]*>)", std::regex_constants::icase);
    for (std::sregex_iterator it(text.begin(), text.end(), projection_pattern), end; it != end; ++it) {
        const auto tag = (*it)[0].str();
        ProjectionIR projection;
        projection.name = xml_attr(tag, "id");
        projection.source = xml_attr(tag, "presynapticPopulation");
        projection.target = xml_attr(tag, "postsynapticPopulation");
        projection.pattern = "one_to_one";
        ir.projections.push_back(std::move(projection));
    }

    NucleusIR nucleus{.name = "Network", .columns = {std::move(column)}};
    RegionIR region{.name = "Region", .nuclei = {std::move(nucleus)}};
    LobeIR lobe{.name = "Lobe", .regions = {std::move(region)}};
    HemisphereIR hemisphere{.name = "Left", .lobes = {std::move(lobe)}};
    ir.brain.hemispheres = {std::move(hemisphere)};
    return ir;
}

bool HocParser::can_parse(const std::filesystem::path& path) const {
    return lowercase_extension(path) == ".hoc";
}

std::string HocParser::format_name() const {
    return "hoc";
}

NetworkIR HocParser::parse(const std::filesystem::path& path) const {
    const auto text = read_file(path);
    NetworkIR ir;
    ir.source_format = format_name();
    ir.brain.name = "HOC Network";

    const std::regex template_pattern(
        R"(begintemplate\s+(\w+)([\s\S]*?)endtemplate)",
        std::regex_constants::icase);
    for (std::sregex_iterator it(text.begin(), text.end(), template_pattern), end; it != end; ++it) {
        const auto name = (*it)[1].str();
        const auto block = (*it)[2].str();
        NeuronParamsIR params;
        std::smatch match;
        const std::regex threshold_pattern(R"(threshold\s*=\s*([0-9.]+))");
        if (std::regex_search(block, match, threshold_pattern)) {
            params.threshold = std::stof(match[1].str());
        }
        const std::regex window_pattern(R"(window_size_ms\s*=\s*([0-9.]+))");
        if (std::regex_search(block, match, window_pattern)) {
            params.pattern_window_ticks = static_cast<std::uint64_t>(std::stoull(match[1].str()));
        }
        const std::regex similarity_pattern(R"(similarity_threshold\s*=\s*([0-9.]+))");
        if (std::regex_search(block, match, similarity_pattern)) {
            params.similarity_threshold = std::stof(match[1].str());
        }
        ir.neuron_params[name] = params;
    }
    if (ir.neuron_params.empty()) {
        ir.neuron_params["default"] = {};
    }

    LayerIR layer;
    layer.name = "Layer";
    const std::regex append_pattern(
        R"(for\s+\w+\s*=\s*0\s*,\s*(\d+)\s*\{\s*(\w+)\.append\(new\s+(\w+)\(\)\))");
    for (std::sregex_iterator it(text.begin(), text.end(), append_pattern), end; it != end; ++it) {
        PopulationIR population;
        population.name = (*it)[2].str();
        population.count = static_cast<std::size_t>(std::stoull((*it)[1].str())) + 1;
        population.neuron_params = (*it)[3].str();
        layer.populations.push_back(std::move(population));
    }

    std::size_t search_from = 0;
    while (true) {
        const auto start = text.find("NetCon(", search_from);
        if (start == std::string::npos) {
            break;
        }
        const auto args_start = start + std::string("NetCon(").size();
        int depth = 1;
        auto end = args_start;
        while (end < text.size() && depth > 0) {
            if (text[end] == '(') {
                ++depth;
            } else if (text[end] == ')') {
                --depth;
            }
            ++end;
        }
        search_from = end;
        if (depth != 0 || end <= args_start) {
            continue;
        }

        const auto args = split_arguments(std::string_view(text).substr(args_start, end - args_start - 1));
        if (args.size() < 5) {
            continue;
        }

        ProjectionIR projection;
        projection.source = infer_hoc_list_name(args[0]);
        projection.target = infer_hoc_list_name(args[1]);
        if (projection.source.empty() || projection.target.empty()) {
            continue;
        }
        projection.name = projection.source + "_to_" + projection.target;
        projection.pattern = "one_to_one";
        projection.delay_ticks = static_cast<std::uint32_t>(std::stoul(args[3]));
        projection.weight = std::stof(args[4]);
        ir.projections.push_back(std::move(projection));
    }

    if (ir.projections.empty()) {
        const std::regex same_line_netcon(
            R"((\w+_cells).*?,\s*(\w+_cells).*?,\s*[^,]+,\s*([0-9.]+),\s*([0-9.]+)\))");
        for (std::sregex_iterator it(text.begin(), text.end(), same_line_netcon), end; it != end; ++it) {
            ProjectionIR projection;
            projection.source = (*it)[1].str();
            projection.target = (*it)[2].str();
            projection.name = projection.source + "_to_" + projection.target;
            projection.pattern = "one_to_one";
            projection.delay_ticks = static_cast<std::uint32_t>(std::stoul((*it)[3].str()));
            projection.weight = std::stof((*it)[4].str());
            ir.projections.push_back(std::move(projection));
        }
    }

    ColumnIR column{.name = "Column", .layers = {std::move(layer)}};
    NucleusIR nucleus{.name = "Network", .columns = {std::move(column)}};
    RegionIR region{.name = "Region", .nuclei = {std::move(nucleus)}};
    LobeIR lobe{.name = "Lobe", .regions = {std::move(region)}};
    HemisphereIR hemisphere{.name = "Left", .lobes = {std::move(lobe)}};
    ir.brain.hemispheres = {std::move(hemisphere)};
    return ir;
}

} // namespace snncuda::declarative
