/*
 * Implement Post function (max(x1, x2)) in 3-valued logic
 */

#include <iostream>
#include "mvn.h"
#include "learning.h"
#include "transforms.h"

#define K 3

const int learning_samples[][3] = {
    { 0, 0, 0 },
    { 0, 1, 1 },
    { 0, 2, 2 },
    { 1, 0, 1 },
    { 1, 1, 1 },
    { 1, 2, 2 },
    { 2, 0, 2 },
    { 2, 1, 2 },
    { 2, 2, 2 }
};

const int nsamples = 9;

int main()
{
    using namespace std;
    using namespace klogic;
    using namespace klogic::learning;
    using namespace klogic::transform;

    mvn neuron(K, 2);
    teacher<mvn> teacher(neuron);

    for (int i = 0; i < nsamples; ++i) {
        const int *inp_beg = &learning_samples[i][0], *inp_end = inp_beg + 2;
        vector<int> input(inp_beg, inp_end);
        int desired = inp_beg[2];

        teacher.add_sample(discrete<K, cmplx>(input, desired));
    }

    // cout << "Samples:" << endl << teacher.samples() << endl;

    cout << "Before hits: " << teacher.hits<single_discrete_match<K> >() << endl;

    for (int step = 1;; ++step) {
        // cout << "MVN weights:" << endl << neuron.weights_vector() << endl;

        teacher.learn_run();

        int hits = teacher.hits<single_discrete_match<K> >();

        cout << "After step " << step << " hits: " << hits << endl;

        if (hits == teacher.samples_count())
            break;
    }

    cout << "MVN weights:" << endl << neuron.weights_vector() << endl;

    return 0;
}