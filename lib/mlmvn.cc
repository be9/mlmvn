#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "mlmvn.h"

using std::vector;

klogic::mlmvn::mlmvn(const vector<int> &sizes, const vector<int> &k_values)
    : calculator(*this)
{
    assert(sizes.size() == k_values.size() + 1);

    max_layer_size = -1;
    input_size  = sizes[0];
    output_size = sizes[sizes.size() - 1];

    // Starting from 1 here because first element of sizes is inputs count
    for (int layer = 1; layer < sizes.size(); ++layer) {
        // k value for current layer
        int k = k_values[layer - 1];

        // inputs count for current layer
        int ninputs = sizes[layer - 1];

        // current layer size
        int size = sizes[layer];

        if (size > max_layer_size)
            max_layer_size = size;

        vector<mvn> layer_neurons(size);

        for (int i = 0; i < size; ++i)
            layer_neurons[i] = mvn(k, ninputs);

        neurons.push_back(layer_neurons);
        errors.push_back(cvector(size));
    }
}

void klogic::mlmvn::learn(const klogic::cvector &X, const klogic::cvector &errs,
                          double learning_rate)
{
    // Calculate errors for all neurons (backward pass)
    calculate_errors(errs);

    // dump_errors();

    // Make a forward pass with error correction
    calculator.start(X);

    do {
        int layer = calculator.current_layer();

        // Divide by |z| for all layers except output
        bool variable_rate = layer < layers_count() - 1;

        vector<mvn>   &layer_neurons = neurons[layer];
        cvector const &layer_errors  = errors[layer];

        // Input for current layer
        cvector::const_iterator input_begin = calculator.input_begin(),
                                input_end   = calculator.input_end();

        // Learn current layer of neurons
//#pragma omp parallel
        {
//#pragma omp for
            for (int k = 0; k < layer_neurons.size(); ++k) {
                layer_neurons[k].learn(input_begin, input_end, layer_errors[k],
                                       learning_rate, variable_rate);
            }
        }
    } while (!calculator.step());

    // dump();
}

void klogic::mlmvn::calculate_errors(const klogic::cvector &errs)
{
    assert(errs.size() == output_size);

    int j = neurons.size() - 1;

    cvector::iterator q = errors[j].begin();
    double s_m = s_j(j);

    // Use (4.121) to calculate errors for output layer
    for (cvector::const_iterator i = errs.begin(); i != errs.end(); ++i, ++q) {
        *q = *i / s_m;
    }

    // Now use (4.122)
    // \delta_{k,j} = (1/s_{j})
    //                \sum_{i=1}^{N_{j+1}} \delta_{i,j+1} (w_k^{i,j+1})^{-1}
    for (--j; j >= 0; --j) {
        cvector             &layer_errors       = errors[j];
        const cvector       &next_layer_errors  = errors[j+1];
        const vector<mvn>   &next_layer_neurons = neurons[j+1];

        int next_layer_size = next_layer_errors.size();
        double layer_s_j = s_j(j);

        // NOTE can be parallelized
#pragma omp parallel
        {
#pragma omp for
            for (int k = 0; k < layer_errors.size(); ++k) {
                cmplx sum(0);

                for (int i = 0; i < next_layer_size; ++i)
                    sum += next_layer_errors[i] / next_layer_neurons[i].weight_for_input(k);

                layer_errors[k] = sum / layer_s_j;
            }
        }
    }
}

void klogic::mlmvn::dump() const
{
    for (int layer = 0; layer < layers_count(); ++layer) {
        const vector<mvn> &layer_neurons = neurons[layer];

        for (int k = 0; k < layer_neurons.size(); ++k) {
            std::cerr << "MVN[" << (k+1) << ',' << (layer+1) << "]: " << layer_neurons[k].weights_vector() << std::endl;
        }
    }
}

void klogic::mlmvn::dump_errors() const
{
    for (int layer = 0; layer < layers_count(); ++layer) {
        const cvector &layer_errs = errors[layer];

        for (int k = 0; k < layer_errs.size(); ++k) {
            std::cerr << "Delta[" << (k+1) << ',' << (layer+1) << "] = " << layer_errs[k] << std::endl;
        }
    }
}

void klogic::mlmvn::get_stats(size_t &n_weights, size_t &n_neurons) const
{
    n_weights = 0, n_neurons = 0;

    size_t prev_layer_size = input_size;

    for (int layer = 0; layer < layers_count(); ++layer) {
        size_t layer_size = neurons[layer].size();

        n_weights += layer_size * (prev_layer_size + 1);
        n_neurons += layer_size;

        prev_layer_size = layer_size;
    }
}

void klogic::mlmvn::export_neurons(klogic::cvector &all_weights, std::vector<int> &k_values) const
{
    all_weights.clear();
    k_values.clear();

    // calculate number of weights and reserve this size
    size_t n_weights, n_neurons;

    get_stats(n_weights, n_neurons);

    all_weights.reserve(n_weights);
    k_values.reserve(n_neurons);

    for (int layer = 0; layer < layers_count(); ++layer) {
        const vector<mvn> &layer_neurons = neurons[layer];

        for (int k = 0; k < layer_neurons.size(); ++k) {
            // Neuron [(k+1), (layer+1)]
            const cvector &weights(layer_neurons[k].weights_vector());

            all_weights.insert(all_weights.end(), weights.begin(), weights.end());
            k_values.push_back(layer_neurons[k].k_value());
        }
    }

    // Sanity check
    assert(all_weights.size() == n_weights);
    assert(k_values.size() == n_neurons);
}

void klogic::mlmvn::load_neurons(const klogic::cvector &all_weights, const std::vector<int> &k_values)
{
    size_t n_weights, n_neurons;

    get_stats(n_weights, n_neurons);

    if (all_weights.size() != n_weights || k_values.size() != n_neurons)
        throw std::runtime_error("klogic::mlmvn::load_neurons(): size mismatch");

    klogic::cvector::const_iterator  it_w = all_weights.begin();
    std::vector<int>::const_iterator it_k = k_values.begin();

    for (int layer = 0; layer < layers_count(); ++layer) {
        vector<mvn> &layer_neurons = neurons[layer];

        for (int k = 0; k < layer_neurons.size(); ++k) {
            // Neuron [(k+1), (layer+1)]
            size_t w = layer_neurons[k].weights_vector().size();
            mvn neuron(*it_k, w - 1);

            copy(it_w, it_w + w, neuron.weights_vector().begin());

            layer_neurons[k] = neuron;

            it_w += w;
            ++it_k;
        }
    }

    assert(it_w == all_weights.end());
    assert(it_k == k_values.end());
}

/*
 * mlmvn_forward_base
 */

klogic::mlmvn_forward_base::mlmvn_forward_base(const klogic::mlmvn &_net)
    : net(_net)
{
}

void klogic::mlmvn_forward_base::start(const klogic::cvector &X, cvector::iterator _out)
{
    layer1.resize(net.max_layer_size);

    // Allocate memory for 1 or 2 layers
    if (net.layers_count() > 1)
        layer2.resize(layer1.size());

    from = &X;
    to = &layer1;
    from_size = X.size();

    out = _out;
    use_out = true;
    layer = 0;
}

void klogic::mlmvn_forward_base::start(const klogic::cvector &X)
{
    start(X, layer1.end());
    use_out = false;
}

bool klogic::mlmvn_forward_base::step()
{
    // Don't do anything if we've already passed all layers
    if (layer >= net.layers_count())
        return true;

    // Use *from as input to layer neurons
    cvector::const_iterator from_beg = from->begin();
    cvector::const_iterator from_end = from_beg + from_size;

    // use *to or out as output
    cvector::iterator j = (layer == net.layers_count() - 1 && use_out)
        ? out : to->begin();

    const vector<mvn> &layer_neurons = net.neurons[layer];

    for (vector<mvn>::const_iterator i = layer_neurons.begin(); i < layer_neurons.end(); ++i)
        *j++ = i->output(from_beg, from_end);

    // Set up "from" and "to" for the next layer
    from_size = layer_neurons.size();

    if (layer == 0) {
        // for the first layer output is always in layer1
        from = &layer1;
        to = &layer2;
    } else if (from == &layer1) {
        from = &layer2;
        to = &layer1;
    } else {
        from = &layer1;
        to = &layer2;
    }

    return ++layer >= net.layers_count();
}
