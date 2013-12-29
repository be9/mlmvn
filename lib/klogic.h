#pragma once

#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cassert>
#include <vector>

namespace klogic {
    typedef std::complex<double> cmplx;
    typedef std::vector<cmplx> cvector;

    const double TWOPI = 2 * M_PI;

    // nth power of kth complex root of unity 
    inline cmplx epsilon(unsigned n, int k) {
        assert(k > 0 && n >= 0 && n < k);

        double phase = (TWOPI * n) / k;

        return std::polar(1.0, phase);
    }

    // phase in [0..2pi) range
    inline double phase(const cmplx &z) {
        double phi = std::arg(z);

        return phi < 0 ? phi + TWOPI : phi;
    }

    // sector number in k-valued logic
    inline int sector_number(int k, const cmplx &z) {
        return int(std::floor((phase(z) / TWOPI) * k) + 0.5);
    }

    // complex activation function in k-valued logic.
    // if k=0 uses continuous activation function.
    inline cmplx activation(int k, const cmplx &z) {
        if (k == 0)
            return z / std::abs(z);
        else {
            return epsilon(sector_number(k, z), k);
        }
    }

    // ------------------

    template<typename Stream, typename T>
    Stream &operator<<(Stream& os, const std::vector<T> &v) {
        os << '[';
        for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i) {
            os << *i;
        }
        return os << ']';
    }
}