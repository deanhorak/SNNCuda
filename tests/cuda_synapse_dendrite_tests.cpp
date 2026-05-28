#include "snncuda/backends/CudaBackend.h"
#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

snncuda::declarative::NetworkIR make_ltp_ir(
    snncuda::snn::DendriticCompartment compartment = snncuda::snn::DendriticCompartment::Basal,
    snncuda::snn::ReceptorType receptor = snncuda::snn::ReceptorType::Ampa) {
    using namespace snncuda::declarative;

    NetworkIR ir;
    ir.source_format = "cuda-test";
    ir.brain.name = "CudaLtpBrain";
    ir.neuron_params["input"] = {.threshold = 1.0F};
    ir.neuron_params["target"] = {.threshold = 0.4F};

    LayerIR layer;
    layer.name = "Layer";
    PopulationIR input;
    input.name = "input";
    input.count = 1;
    input.neuron_params = "input";
    layer.populations.push_back(std::move(input));

    PopulationIR target;
    target.name = "target";
    target.count = 1;
    target.neuron_params = "target";
    layer.populations.push_back(std::move(target));

    ColumnIR column;
    column.name = "Column";
    column.layers.push_back(std::move(layer));

    NucleusIR nucleus;
    nucleus.name = "Nucleus";
    nucleus.columns.push_back(std::move(column));

    RegionIR region;
    region.name = "Region";
    region.nuclei.push_back(std::move(nucleus));

    LobeIR lobe;
    lobe.name = "Lobe";
    lobe.regions.push_back(std::move(region));

    HemisphereIR hemisphere;
    hemisphere.name = "Left";
    hemisphere.lobes.push_back(std::move(lobe));
    ir.brain.hemispheres.push_back(std::move(hemisphere));

    ir.projections.push_back({
        .name = "input_to_target",
        .source = "input",
        .target = "target",
        .pattern = "one_to_one",
        .weight = 0.5F,
        .max_weight = 2.0F,
        .delay_ticks = 1,
        .compartment = compartment,
        .receptor = receptor,
        .plasticity_enabled = true,
    });
    return ir;
}

snncuda::declarative::NetworkIR make_ltd_ir() {
    using namespace snncuda::declarative;

    auto ir = make_ltp_ir();
    ir.brain.name = "CudaLtdBrain";
    ir.neuron_params["target"].threshold = 1.0F;
    auto& layer = ir.brain.hemispheres[0].lobes[0].regions[0].nuclei[0].columns[0].layers[0];
    PopulationIR trigger;
    trigger.name = "trigger";
    trigger.count = 1;
    trigger.neuron_params = "input";
    layer.populations.push_back(std::move(trigger));

    ir.projections.clear();
    ir.projections.push_back({
        .name = "trigger_to_input",
        .source = "trigger",
        .target = "input",
        .pattern = "one_to_one",
        .weight = 1.0F,
        .max_weight = 2.0F,
        .delay_ticks = 5,
        .compartment = snncuda::snn::DendriticCompartment::Basal,
        .receptor = snncuda::snn::ReceptorType::Ampa,
        .plasticity_enabled = false,
    });
    ir.projections.push_back({
        .name = "input_to_target",
        .source = "input",
        .target = "target",
        .pattern = "one_to_one",
        .weight = 0.5F,
        .max_weight = 2.0F,
        .delay_ticks = 1,
        .compartment = snncuda::snn::DendriticCompartment::Basal,
        .receptor = snncuda::snn::ReceptorType::Ampa,
        .plasticity_enabled = true,
    });
    return ir;
}

void run_cpu(
    const std::vector<snncuda::core::NeuronId>& initially_active,
    std::uint64_t max_tick,
    snncuda::runtime::NetworkPropagator& propagator) {
    for (const auto neuron : initially_active) {
        propagator.inject(neuron, 0, 1.0F);
    }
    propagator.run_until(max_tick);
}

void test_cuda_ltp_matches_cpu_counts_and_potentiates() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_ltp_ir());
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator cpu(connectome, store, 8, 8);
    run_cpu({input}, 1, cpu);

    const snncuda::backends::CudaBackend cuda;
    const auto gpu = cuda.run_resident_propagation(connectome, {input}, 1);

    require(gpu.executed, "CUDA LTP test should execute");
    require(gpu.delivered_spikes == cpu.delivered_spike_count(), "CUDA LTP delivered count should match CPU");
    require(gpu.fired_spikes == cpu.fired_spike_count(), "CUDA LTP fired count should match CPU");
    require(gpu.final_synapse_weights.size() == 1, "CUDA LTP should return synapse weights");
    require(gpu.final_synapse_weights[0] > 0.5F, "CUDA post-after-pre STDP should potentiate");
    require(cpu.synapse_state(1)->weight > 0.5F, "CPU post-after-pre STDP should potentiate");
    require(gpu.final_neuron_spike_counts.size() == 2, "CUDA LTP should return neuron spike counts");
    require(gpu.final_neuron_spike_counts[0] == 1, "CUDA input neuron should fire once");
    require(gpu.final_neuron_spike_counts[1] == 1, "CUDA target neuron should fire once");
    require(cpu.spike_count(target) == 1, "CPU target neuron should fire once");
}

void test_cuda_inhibitory_receptor_matches_cpu_nonfire() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(
        make_ltp_ir(
            snncuda::snn::DendriticCompartment::Inhibitory,
            snncuda::snn::ReceptorType::GabaA));
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator cpu(connectome, store, 8, 8);
    run_cpu({input}, 1, cpu);

    const snncuda::backends::CudaBackend cuda;
    const auto gpu = cuda.run_resident_propagation(connectome, {input}, 1);

    require(gpu.executed, "CUDA inhibitory test should execute");
    require(gpu.delivered_spikes == cpu.delivered_spike_count(), "CUDA inhibitory delivered count should match CPU");
    require(gpu.fired_spikes == cpu.fired_spike_count(), "CUDA inhibitory fired count should match CPU");
    require(cpu.spike_count(target) == 0, "CPU inhibitory target should not fire");
    require(gpu.final_neuron_spike_counts[1] == 0, "CUDA inhibitory target should not fire");
    require(gpu.final_membrane_potentials[1] < 0.0F, "CUDA inhibitory membrane should be negative");
}

void test_cuda_ltd_matches_cpu_direction() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_ltd_ir());
    const auto trigger = connectome.populations.at("trigger").at(0);
    const auto target = connectome.populations.at("target").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator cpu(connectome, store, 8, 16);
    run_cpu({target, trigger}, 6, cpu);

    const snncuda::backends::CudaBackend cuda;
    const auto gpu = cuda.run_resident_propagation(connectome, {target, trigger}, 6);

    require(gpu.executed, "CUDA LTD test should execute");
    require(gpu.delivered_spikes == cpu.delivered_spike_count(), "CUDA LTD delivered count should match CPU");
    require(gpu.fired_spikes == cpu.fired_spike_count(), "CUDA LTD fired count should match CPU");
    require(gpu.final_synapse_weights.size() == 2, "CUDA LTD should return two synapse weights");
    require(gpu.final_synapse_weights[1] < 0.5F, "CUDA pre-after-post STDP should depress");
    require(cpu.synapse_state(2)->weight < 0.5F, "CPU pre-after-post STDP should depress");
}

void test_cuda_coded_synapse_expansion_matches_cpu() {
    auto ir = make_ltp_ir();
    ir.neuron_params["target"].threshold = 100.0F;
    ir.neuron_params["target"].pattern_window_ticks = 8;
    ir.neuron_params["target"].similarity_threshold = 0.99F;
    ir.neuron_params["target"].max_reference_patterns = 4;
    ir.projections[0].weight = 0.1F;
    ir.projections[0].spike_code_offsets = {0, 2, 3};
    ir.projections.push_back({
        .name = "input_retrigger",
        .source = "input",
        .target = "input",
        .pattern = "one_to_one",
        .weight = 1.0F,
        .max_weight = 2.0F,
        .delay_ticks = 20,
        .plasticity_enabled = false,
    });
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(ir);
    const auto input = connectome.populations.at("input").at(0);

    snncuda::runtime::MemoryNeuronStateStore store;
    snncuda::runtime::NetworkPropagator cpu(connectome, store, 8, 16);
    run_cpu({input}, 25, cpu);

    const snncuda::backends::CudaBackend cuda;
    const auto gpu = cuda.run_resident_propagation(connectome, {input}, 25);

    require(gpu.executed, "CUDA coded synapse test should execute");
    require(cpu.delivered_spike_count() == 8, "CPU should process external plus repeated coded synaptic spikes");
    require(gpu.delivered_spikes == cpu.delivered_spike_count(), "CUDA coded synapse delivered count should match CPU");
    require(gpu.fired_spikes == cpu.fired_spike_count(), "CUDA coded synapse fired count should match CPU");
    require(gpu.debug.scheduled_event_requests == 8, "CUDA should request every emitted code spike and retrigger spike");
    require(gpu.debug.scheduled_events == 7, "CUDA should schedule in-window emitted code spikes and retrigger spike");
    require(cpu.synapse_state(1)->code_match_count >= 1, "CPU synapse should recognize repeated spike code");
    require(gpu.final_synapse_code_match_counts.size() == 2, "CUDA should return synapse code match counts");
    require(gpu.final_synapse_code_match_counts[0] >= 1, "CUDA synapse should recognize repeated spike code");
}

void test_cuda_inference_session_matches_repeated_single_sample_cuda() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_ltp_ir());
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    const snncuda::backends::CudaBackend cuda;
    const auto first = cuda.run_resident_propagation(connectome, {input}, 1);
    const auto second = cuda.run_resident_propagation(connectome, {input}, 1);

    snncuda::backends::CudaInferenceSession session(connectome);
    snncuda::backends::CudaInferenceOptions options;
    options.max_steps = 1;
    options.readout_neurons = {target};
    const auto batch = session.run_batch(
        {
            {.inputs = {{.neuron = input}}},
            {.inputs = {{.neuron = input}}},
        },
        options);

    require(batch.executed, "CUDA inference session should execute");
    require(batch.samples.size() == 2, "CUDA inference session should return each sample");
    require(batch.samples[0].delivered_spikes == first.delivered_spikes, "batch delivered count should match single CUDA");
    require(batch.samples[0].fired_spikes == first.fired_spikes, "batch fired count should match single CUDA");
    require(batch.samples[0].readout_spike_counts.size() == 1, "batch should return requested readout");
    require(
        batch.samples[0].readout_spike_counts[0] == first.final_neuron_spike_counts[1],
        "batch readout spike count should match single CUDA");
    require(batch.samples[1].delivered_spikes == second.delivered_spikes, "second batch delivered count should match");
    require(batch.samples[1].fired_spikes == second.fired_spikes, "second batch fired count should match");
    require(
        batch.samples[1].readout_spike_counts[0] == second.final_neuron_spike_counts[1],
        "second batch readout spike count should match");
}

void test_cuda_inference_session_reuses_setup_for_many_samples() {
    const auto connectome = snncuda::declarative::ConnectomeBuilder{}.build(make_ltp_ir());
    const auto input = connectome.populations.at("input").at(0);
    const auto target = connectome.populations.at("target").at(0);

    std::vector<snncuda::backends::CudaInferenceSample> samples;
    samples.reserve(100);
    for (int i = 0; i < 100; ++i) {
        samples.push_back({.inputs = {{.neuron = input}}});
    }

    const snncuda::backends::CudaBackend cuda;
    const auto repeated_start = std::chrono::steady_clock::now();
    for (const auto& sample : samples) {
        std::vector<snncuda::core::NeuronId> active;
        active.reserve(sample.inputs.size());
        for (const auto& input_sample : sample.inputs) {
            active.push_back(input_sample.neuron);
        }
        const auto result = cuda.run_resident_propagation(connectome, active, 1);
        require(result.executed, "repeated CUDA inference should execute");
    }
    const auto repeated_end = std::chrono::steady_clock::now();
    const auto repeated_seconds = std::chrono::duration<double>(repeated_end - repeated_start).count();

    snncuda::backends::CudaInferenceSession session(connectome);
    snncuda::backends::CudaInferenceOptions options;
    options.max_steps = 1;
    options.readout_neurons = {target};
    const auto batch = session.run_batch(samples, options);

    require(batch.executed, "persistent CUDA inference should execute");
    require(batch.samples.size() == samples.size(), "persistent CUDA inference should return all samples");
    require(batch.elapsed_seconds > 0.0, "persistent CUDA inference should report elapsed time");
    require(
        batch.elapsed_seconds < repeated_seconds,
        "persistent CUDA inference should avoid repeated per-sample setup overhead");

    std::cout << "CUDA repeated setup inference 100 samples: " << repeated_seconds
              << "s, persistent batch: " << batch.elapsed_seconds << "s\n";
}

} // namespace

int main() {
    const snncuda::backends::CudaBackend cuda;
    if (!cuda.available()) {
        std::cout << "CUDA unavailable, skipping CUDA synapse/dendrite tests: "
                  << cuda.availability_status() << '\n';
        return EXIT_SUCCESS;
    }

    try {
        test_cuda_ltp_matches_cpu_counts_and_potentiates();
        test_cuda_inhibitory_receptor_matches_cpu_nonfire();
        test_cuda_ltd_matches_cpu_direction();
        test_cuda_coded_synapse_expansion_matches_cpu();
        test_cuda_inference_session_matches_repeated_single_sample_cuda();
        test_cuda_inference_session_reuses_setup_for_many_samples();
    } catch (const std::exception& error) {
        std::cerr << "CUDA synapse/dendrite test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All CUDA synapse/dendrite tests passed\n";
    return EXIT_SUCCESS;
}
