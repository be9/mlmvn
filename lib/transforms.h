#pragma once

#include "klogic.h"

namespace klogic {
    namespace learning {
        template<typename Desired> struct sample;   // Forward declaration
    }

    ////////////////////////////

    namespace transform {
        // Discrete (one value)
        template<int K>
        cmplx discrete(int n) {
            assert(n >= 0 && n < K);

            return epsilon(n, K);
        }

        // Discrete (vector)
        template<int K>
        cvector discrete(const std::vector<int> &values) {
            cvector result(values.size());

            std::transform(values.begin(), values.end(),
                result.begin(), 
                static_cast<cmplx (*)(int)>(&discrete<K>));

            return result;
        }

        // Sample
        template<int K, typename Desired, typename Input>
        learning::sample<Desired> discrete(const std::vector<int> &values, const Input &desired) {
            return learning::sample<Desired>(discrete<K>(values), discrete<K>(desired));
        }

        //////

        // Continuous (0..2pi)
        inline cmplx continuous(double x) {
            assert(x >= 0 && x < TWOPI);

            return std::polar(1.0, x);
        }

        // Continuous. Vector (0..2pi)
        inline cvector continuous(std::vector<double> xs) {
            cvector result(xs.size());

            std::transform(xs.begin(), xs.end(),
                result.begin(), 
                static_cast<cmplx (*)(double)>(&continuous));

            return result;
        }

        template<typename Desired, typename Input>
        learning::sample<Desired> continuous(const std::vector<double> &values, const Input &desired) {
            return learning::sample<Desired>(continuous(values), continuous(desired));
        }
    }
}