// #include <iostream>
#include "mvn.h"
#include <cstdlib>

using namespace std;

klogic::mvn::mvn(int k, int N)
    : weights(N + 1)
{
    assert(k >= 0 && N >= 0);
    this->k = k;

    // Randomize weights
    for (cvector::iterator i = weights.begin(); i != weights.end(); ++i)
        *i = cmplx(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
}

//-------------------------------------------------------------------------

klogic::cmplx klogic::mvn::weighted_sum(
    klogic::cvector::const_iterator Xbeg,
    klogic::cvector::const_iterator Xend) const
{
    assert(weights.size() == Xend - Xbeg + 1);

    cvector::const_iterator w = weights.begin(), x = Xbeg;
    cmplx z = *w++;         // bias

    while (w != weights.end())
        z += (*w++) * (*x++);   // pairwise multiply and summate

    return z;
}

//-------------------------------------------------------------------------
void klogic::mvn::learn(cvector::const_iterator Xbeg,
                        cvector::const_iterator Xend,
                        const cmplx &error, double learning_rate, bool variable_rate)
{
    assert(weights.size() == Xend - Xbeg + 1);

    cmplx factor = error * learning_rate / (double)weights.size(); // division by N+1

    if (variable_rate)
        factor /= std::abs(weighted_sum(Xbeg, Xend));

    cvector::iterator       w = weights.begin();
    cvector::const_iterator x = Xbeg;

    // cout << "learn(): error=" << error << " factor=" << factor << "weights before: " << weights_vector() << endl;

    *w++ += factor;         // change bias

    for (; w != weights.end(); w++, x++) {
        *w += factor * std::conj(*x);
    }

    // cout << "learn(): weights after: " << weights_vector() << endl;

}
