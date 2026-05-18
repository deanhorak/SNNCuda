#include "snncuda/backends/CudaBackend.h"
#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace {

using snncuda::declarative::ConnectomeBuilder;
using snncuda::declarative::NetworkIR;

NetworkIR make_benchmark_ir() {
    using namespace snncuda::declarative;

    constexpr int hemispheres = 2;
    constexpr int lobes_per_hemisphere = 2;
    constexpr int regions_per_lobe = 3;
    constexpr int nuclei_per_region = 3;
    constexpr int columns_per_nucleus = 8;
    constexpr int layers_per_column = 6;
    constexpr int neurons_per_population = 16;

    NetworkIR ir;
    ir.source_format = "benchmark";
    ir.brain.name = "BenchmarkBrain";
    ir.neuron_params["default"] = {
        .threshold = 1.0F,
        .pattern_window_ticks = 16,
        .similarity_threshold = 0.9F,
        .max_reference_patterns = 8,
    };

    for (int h = 0; h < hemispheres; ++h) {
        HemisphereIR hemisphere;
        hemisphere.name = "Hemisphere" + std::to_string(h);
        for (int lobe_index = 0; lobe_index < lobes_per_hemisphere; ++lobe_index) {
            LobeIR lobe;
            lobe.name = "Lobe" + std::to_string(lobe_index);
            for (int region_index = 0; region_index < regions_per_lobe; ++region_index) {
                RegionIR region;
                region.name = "Region" + std::to_string(region_index);
                for (int nucleus_index = 0; nucleus_index < nuclei_per_region; ++nucleus_index) {
                    NucleusIR nucleus;
                    nucleus.name = "Nucleus" + std::to_string(nucleus_index);
                    for (int column_index = 0; column_index < columns_per_nucleus; ++column_index) {
                        ColumnIR column;
                        column.name = "Column" + std::to_string(column_index);
                        for (int layer_index = 0; layer_index < layers_per_column; ++layer_index) {
                            LayerIR layer;
                            layer.name = "L" + std::to_string(layer_index);

                            PopulationIR population;
                            population.name = "H" + std::to_string(h)
                                + "_Lo" + std::to_string(lobe_index)
                                + "_R" + std::to_string(region_index)
                                + "_N" + std::to_string(nucleus_index)
                                + "_C" + std::to_string(column_index)
                                + "_L" + std::to_string(layer_index);
                            population.count = neurons_per_population;
                            population.neuron_params = "default";
                            layer.populations.push_back(std::move(population));
                            column.layers.push_back(std::move(layer));
                        }
                        nucleus.columns.push_back(std::move(column));
                    }
                    region.nuclei.push_back(std::move(nucleus));
                }
                lobe.regions.push_back(std::move(region));
            }
            hemisphere.lobes.push_back(std::move(lobe));
        }
        ir.brain.hemispheres.push_back(std::move(hemisphere));
    }

    for (const auto& hemisphere : ir.brain.hemispheres) {
        for (const auto& lobe : hemisphere.lobes) {
            for (const auto& region : lobe.regions) {
                for (const auto& nucleus : region.nuclei) {
                    for (const auto& column : nucleus.columns) {
                        for (std::size_t layer = 0; layer + 1 < column.layers.size(); ++layer) {
                            ir.projections.push_back({
                                .name = column.layers[layer].populations[0].name + "_to_"
                                    + column.layers[layer + 1].populations[0].name,
                                .source = column.layers[layer].populations[0].name,
                                .target = column.layers[layer + 1].populations[0].name,
                                .pattern = "one_to_one",
                                .weight = 1.0F,
                                .max_weight = 2.0F,
                                .delay_ticks = 1,
                            });
                        }
                    }
                }
            }
        }
    }

    return ir;
}

} // namespace

int main() {
    const auto ir = make_benchmark_ir();
    const auto connectome = ConnectomeBuilder{}.build(ir);

    std::vector<snncuda::core::NeuronId> initially_active;
    initially_active.reserve(connectome.neurons.size());
    std::unordered_set<snncuda::core::NeuronId> seen_initially_active;
    for (const auto& [name, ids] : connectome.populations) {
        if (name.size() < 3 || name.substr(name.size() - 3) != "_L0") {
            continue;
        }
        for (const auto id : ids) {
            if (seen_initially_active.insert(id).second) {
                initially_active.push_back(id);
            }
        }
    }

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator propagator(
        connectome,
        store,
        4096,
        1024);

    for (const auto id : initially_active) {
        propagator.inject(id, 0, 1.0F);
    }

    const auto start = std::chrono::steady_clock::now();
    propagator.run_until(6);
    const auto end = std::chrono::steady_clock::now();
    const auto seconds = std::chrono::duration<double>(end - start).count();
    const auto delivered = propagator.delivered_spike_count();
    const auto fired = propagator.fired_spike_count();
    const auto spikes_per_second = static_cast<double>(delivered) / seconds;

    std::cout << "backend_compiled_cuda=" << (SNNCUDA_HAS_CUDA ? "yes" : "no") << '\n';
    std::cout << "hdf5_enabled=" << (SNNCUDA_HAS_HDF5 ? "yes" : "no") << '\n';
    std::cout << "neurons=" << connectome.neurons.size() << '\n';
    std::cout << "synapses=" << connectome.synapses.size() << '\n';
    std::cout << "injected_spikes=" << initially_active.size() << '\n';
    std::cout << "cpu_delivered_spikes=" << delivered << '\n';
    std::cout << "cpu_fired_spikes=" << fired << '\n';
    std::cout << "cpu_elapsed_seconds=" << std::fixed << std::setprecision(6) << seconds << '\n';
    std::cout << "cpu_delivered_spikes_per_second=" << std::fixed << std::setprecision(2)
              << spikes_per_second << '\n';

#if SNNCUDA_HAS_CUDA
    const snncuda::backends::CudaBackend cuda;
    std::cout << "cuda_availability_status=" << cuda.availability_status() << '\n';
    const auto cuda_result = cuda.run_resident_propagation(connectome, initially_active, 6);
    if (cuda_result.executed) {
        const auto cuda_spikes_per_second = static_cast<double>(cuda_result.delivered_spikes)
            / cuda_result.elapsed_seconds;
        std::cout << "cuda_executed=yes\n";
        std::cout << "cuda_delivered_spikes=" << cuda_result.delivered_spikes << '\n';
        std::cout << "cuda_fired_spikes=" << cuda_result.fired_spikes << '\n';
        std::cout << "cuda_elapsed_seconds=" << std::fixed << std::setprecision(6)
                  << cuda_result.elapsed_seconds << '\n';
        std::cout << "cuda_delivered_spikes_per_second=" << std::fixed << std::setprecision(2)
                  << cuda_spikes_per_second << '\n';
        std::cout << "cuda_vs_cpu_speedup=" << std::fixed << std::setprecision(2)
                  << (cuda_spikes_per_second / spikes_per_second) << '\n';
        std::cout << "cuda_debug_ticks_processed=" << cuda_result.debug.ticks_processed << '\n';
        std::cout << "cuda_debug_scheduled_event_requests="
                  << cuda_result.debug.scheduled_event_requests << '\n';
        std::cout << "cuda_debug_scheduled_events=" << cuda_result.debug.scheduled_events << '\n';
        std::cout << "cuda_debug_processed_events=" << cuda_result.debug.processed_events << '\n';
        std::cout << "cuda_debug_dropped_events=" << cuda_result.debug.dropped_events << '\n';
        std::cout << "cuda_debug_fired_appends=" << cuda_result.debug.fired_appends << '\n';
        std::cout << "cuda_debug_fired_overflow=" << cuda_result.debug.fired_overflow << '\n';
        std::cout << "cuda_debug_lock_spin_iterations="
                  << cuda_result.debug.lock_spin_iterations << '\n';
        std::cout << "cuda_debug_lock_timeouts=" << cuda_result.debug.lock_timeouts << '\n';
        std::cout << "cuda_debug_stdp_updates=" << cuda_result.debug.stdp_updates << '\n';
        std::cout << "cuda_debug_stdp_ltp=" << cuda_result.debug.stdp_ltp << '\n';
        std::cout << "cuda_debug_stdp_ltd=" << cuda_result.debug.stdp_ltd << '\n';
        std::cout << "cuda_debug_receptor_ampa_events="
                  << cuda_result.debug.receptor_ampa_events << '\n';
        std::cout << "cuda_debug_receptor_nmda_events="
                  << cuda_result.debug.receptor_nmda_events << '\n';
        std::cout << "cuda_debug_receptor_gaba_a_events="
                  << cuda_result.debug.receptor_gaba_a_events << '\n';
        std::cout << "cuda_debug_receptor_gaba_b_events="
                  << cuda_result.debug.receptor_gaba_b_events << '\n';
        std::cout << "cuda_debug_dendritic_integrations="
                  << cuda_result.debug.dendritic_integrations << '\n';
        std::cout << "cuda_debug_post_plasticity_scans="
                  << cuda_result.debug.post_plasticity_scans << '\n';
        std::cout << "cuda_debug_temporal_observations="
                  << cuda_result.debug.temporal_observations << '\n';
        std::cout << "cuda_debug_temporal_patterns_learned="
                  << cuda_result.debug.temporal_patterns_learned << '\n';
        std::cout << "cuda_debug_temporal_matches="
                  << cuda_result.debug.temporal_matches << '\n';
        std::cout << "cuda_debug_max_scheduled_events_per_tick="
                  << cuda_result.debug.max_scheduled_events_per_tick << '\n';
        std::cout << "cuda_debug_max_fired_neurons_per_tick="
                  << cuda_result.debug.max_fired_neurons_per_tick << '\n';
        return delivered == cuda_result.delivered_spikes && fired == cuda_result.fired_spikes
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
    }
    std::cout << "cuda_executed=no\n";
#else
    std::cout << "cuda_executed=no\n";
#endif

    return delivered == fired && delivered > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
