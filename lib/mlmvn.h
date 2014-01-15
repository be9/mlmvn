// MLMVN implementation
#pragma once

#include "mvn.h"

namespace klogic {
    class mlmvn;

    // Separate class for calculating mlmvn output allows to parallelize, but
    // is stingy about memory allocation. It uses max. two vectors to support
    // computations for networks containing arbitrary number of layers
    class mlmvn_forward_base {
    public:
        mlmvn_forward_base(const mlmvn &_net);

        // Layer-by-layer calculation interface
        // Start calculation X -> out
        void start(const klogic::cvector &X, cvector::iterator out);

        // Start calculation for X not meaning to get result
        void start(const klogic::cvector &X);

        // Get current layer, i.e. neurons which process incoming data and
        // output results to the next layer
        int current_layer() const { return layer; }

        // Make calculations for current layer and step to the next one.
        // Returns true if result is already available.
        bool step();

        // Input for current layer
        cvector::const_iterator input_begin() const {
            return from->begin();
        }

        cvector::const_iterator input_end() const {
            return from->begin() + from_size;
        }

    protected:
        // original values passed to start are saved here
        cvector::iterator out;
        const mlmvn &net;

        bool use_out;

        // Two internal vectors storing intermediate results
        cvector layer1, layer2;

        // `from` always points to a vector which contains input for current
        // layer. At start, it points to X but then its value cycles between
        // &layer1 and &layer2
        const cvector *from;

        // `to` points to a vector which receives output from current layer. It
        // also cycles between &layer1 and &layer2 but it always "opposite" to
        // `from`. If `to` points to layer1, `from` points to layer2 and vice versa.
        cvector *to;

        // This value corresponds to input size available in `*from`
        size_t from_size;

        // Current layer number. Input layer is not counted, so layer == 0
        // means first hidden layer or output layer if network has no hidden
        // layers
        int layer;
    };

    //--------------------------------------------------------------

    class mlmvn {
        friend class mlmvn_forward;
        friend class mlmvn_forward_base;
    public:
        typedef cvector desired_type;

        // Construct an MLMVN. sizes is the following:
        // Number of inputs, hidden layer 1 size, ...,
        // hidden layer M size, output layer size
        mlmvn(const std::vector<int> &sizes,
              const std::vector<int> &k_values);

        // Returns total layer count in network (hidden + one output)
        size_t layers_count() const { return neurons.size(); }

        size_t output_layer_size() const { return output_size; }

        // Correct weights
        void learn(const cvector &X, const cvector &error,
            double learning_rate = 1.0);

        // i-th neuron in j-th layer
        mvn &neuron(int i, int j) {
            return neurons[j][i];
        }

        // Net output. Use with care since it allocates memory on each run
        cvector output(const cvector &X) const;

        void dump() const;
        void dump_errors() const;

    protected:
        // Calculate errors for all neurons given
        // output layer errors
        void calculate_errors(const cvector &errs);

        std::vector<std::vector<mvn> >   neurons;
        std::vector<std::vector<cmplx> > errors;

        int s_j(int j) {
            return (j <= 0) ? 1 : 1 + neurons[j-1].size();
        }

        int max_layer_size;
        size_t input_size, output_size;

        mlmvn_forward_base calculator;
    };

    //--------------------------------------------------------------

    // Higher level version of mlmvn_forward_base
    class mlmvn_forward : protected mlmvn_forward_base {
    public:
        mlmvn_forward(const mlmvn &net)
            : mlmvn_forward_base(net) {}

        // Calculate network output for input vector X and return it as a vector
        cvector output(const cvector &X) {
            cvector result(net.output_size);

            output(X, result.begin());

            return result;
        }

        // Calculate network output for input vector X and put it to out
        void output(const cvector &X, cvector::iterator out) {
            start(X, out);

            while (!step())
                ;
        }
    };

    //--------------------------------------------------------------

    inline cvector mlmvn::output(const cvector &X) const {
        return mlmvn_forward(*this).output(X);
    }
};
