// Single MVN implementation
#pragma once

#include "klogic.h"

namespace klogic {

    class mvn {
    public:
        typedef cmplx desired_type;

        // Create mvn in k-valued logic with N inputs.
        // This counts for N+1 weights, including bias.
        mvn(int k, int N);
        mvn() : k(-1) {}

        // Applies activation function to weighted sum
        cmplx output(const cvector &X) const {
            return activation(k, weighted_sum(X.begin(), X.end()));
        }

        cmplx output(cvector::const_iterator xbeg, cvector::const_iterator xend) const {
            return activation(k, weighted_sum(xbeg, xend));
        }

        // Returns true if this neuron is discrete
        bool is_discrete() const { return k > 0; }

        // Returns k
        int k_value() const { return k; }

        // Uses Error-Correction Learning Rule (3.92) to
        // change weights. If variable_rate is true,
        // additional division by |z| is done per (3.94)
        void learn(cvector::const_iterator Xbeg, cvector::const_iterator Xend,
                   const cmplx &error, double learning_rate = 1.0,
                   bool variable_rate = false);

        void learn(const cvector &X, const cmplx &error, 
                   double learning_rate = 1.0, bool variable_rate = false) {
            learn(X.begin(), X.end(), error, learning_rate, variable_rate);
        }

        // Returns weights
        const cvector &weights_vector() const { return weights; }
        cvector &weights_vector()             { return weights; }

        // Returns weight which corresponds to i-th input of this neuron
        cmplx weight_for_input(int i) const {
            assert(i >= 0 && i < weights.size() - 1);

            // 0th weight is bias
            return weights[i+1];
        }

    protected:
        cvector weights;
        int k;
        /**************/
        // Calculates w_0+w_1*i_1+....+w_N*i_N
        cmplx weighted_sum(cvector::const_iterator xbeg, cvector::const_iterator xend) const;
    };
}
