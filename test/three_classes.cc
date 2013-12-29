#include <iostream>
#include <cmath>
#include "mlmvn.h"
#include "learning.h"
#include "transforms.h"

using namespace std;
using namespace klogic;

const double learning_samples[][3] = {
    { 4.23, 2.10, 0.76 },
    { 5.34, 1.24, 2.56 },
    { 2.10, 0.00, 5.35 }
};

const int nsamples = 3;

////////////////////////////

void set_initial_weights(mlmvn &net)
{
	cvector &weights0 = net.neuron(0, 0).weights_vector();
	weights0[0] = cmplx(0.23, -0.38);
	weights0[1] = cmplx(0.19, -0.46);
	weights0[2] = cmplx(0.36, -0.33);

	cvector &weights1 = net.neuron(1, 0).weights_vector();
	weights1[0] = cmplx(0.23, -0.38);
	weights1[1] = cmplx(0.19, -0.46);
	weights1[2] = cmplx(0.36, -0.33);

	cvector &weights2 = net.neuron(0, 1).weights_vector();
	weights2[0] = cmplx(0.23, -0.38);
	weights2[1] = cmplx(0.19, -0.46);
	weights2[2] = cmplx(0.36, -0.33);
}

////////////////////////////

void add_samples(learning::teacher<mlmvn> &teacher) 
{
    for (int i = 0; i < nsamples; ++i) {
        const double *inp_beg = &learning_samples[i][0], *inp_end = inp_beg + 2;
        vector<double> input(inp_beg, inp_end);
        vector<double> desired(inp_beg + 2, inp_beg + 3);

        teacher.add_sample(transform::continuous<cvector>(input, desired));
    }
}

////////////////////////////

const double TOLERANCE = 0.05, MSE_TOLERANCE = TOLERANCE * TOLERANCE;

class PhaseSubtractor {
public:
	double phase_difference(const learning::sample<cvector> &sample, const cvector &actual) const {
		assert(actual.size() == 1);

		double phase_actual  = phase(actual[0]);
		double phase_desired = phase(sample.desired[0]);

		return fabs(phase_desired - phase_actual);
	}
};

class TolerancePicker : protected PhaseSubtractor {
public:
	bool operator()(const learning::sample<cvector> &sample, const cvector &actual) const {
		double err = phase_difference(sample, actual);

		cerr << "Output=" << actual[0] << " PhaseError=" << err << endl;

		return err > TOLERANCE;
	}
};

class SingleSquareError : protected PhaseSubtractor {
public:
	double operator()(const learning::sample<cvector> &sample, const cvector &actual) const {
		double err = phase_difference(sample, actual);

		return err * err;
	}	
};

int main()
{
	vector<int> sizes(3), k_values(2);

	sizes[0] = 2;
	sizes[1] = 2;
	sizes[2] = 1;

	k_values[0] = k_values[1] = 0;

	mlmvn net(sizes, k_values);
	set_initial_weights(net);

	learning::teacher<mlmvn> teacher(net);
	add_samples(teacher);

	double current_mse = teacher.mse<SingleSquareError>();

	cout << "Before everything: MSE = " << current_mse << endl;

	int epoch = 1;

	bool prev_bad = true, current_bad = true;

	do {
		prev_bad = current_bad;

		cout << "\n\n ============== EPOCH " << epoch << " ==============" << endl;

	    teacher.learn_run<TolerancePicker>();

		current_mse = teacher.mse<SingleSquareError>();

	    cout << "Current MSE = " << current_mse << endl;

	    current_bad = current_mse >= MSE_TOLERANCE;

	    epoch++;
	} while (prev_bad || current_bad);

	return 0;
}