#pragma once

#include <algorithm>
#include <vector>
#include <cassert>
#include "mvn.h"

namespace klogic {
    namespace learning {
        template<typename Desired>
        struct sample {
            typedef Desired desired_type; 

            cvector input;
            Desired desired;

            sample(const cvector &i, const Desired &d) : input(i), desired(d) {}        
        };

        // ------------------

        template<typename Desired>
        class learn_error {};

        template<>
        class learn_error<cmplx> {
        public:
            cmplx operator()(const cmplx &output, const cmplx &sample) const {
                return sample - output;
            }
        };

        template<>
        class learn_error<cvector> {
        public:
            cvector operator()(const cvector &output, const cvector &sample) const {
                assert(output.size() == sample.size());
                cvector errors(output.size());

                for (int i = 0; i < output.size(); ++i)
                    errors[i] = sample[i] - output[i];

                return errors;
            }
        };

        // ------------------

        template <typename Sample, typename Desired = typename Sample::desired_type>
        class learn_always {
        public:
            bool operator()(Sample const &, Desired const &) const { return true; }
        };

        // ------------------

        template<typename Learner,          // mvn or mlmvn
                 typename Sample     = sample<typename Learner::desired_type>,
                 typename LearnError = learn_error <typename Sample::desired_type> >
        class teacher {
        public:
            teacher(Learner &_learner, 
                    const std::vector<Sample> &samples = std::vector<Sample>())
                : learner(_learner), _samples(samples)
                {}

            // Add sample to the set
            void add_sample(const Sample &sample) {
                _samples.push_back(sample);
            }

            // Learning set
            const std::vector<Sample> samples() const { return _samples; }

            // Learning set size
            int samples_count() const { return samples().size(); }

            // How well learner matches the learning set
            template <class Match>
            int hits(Match const &match = Match()) const {
                int count = 0;

                for (typename std::vector<Sample>::const_iterator i = _samples.begin();
                        i != _samples.end(); ++i) {

                    if (match(learner.output(i->input), i->desired))
                        ++count;
                }

                return count;
            }

            // Make a run against set. picker instance is used to skip some set items
            template <typename SamplePicker>
            void learn_run(SamplePicker const &picker = SamplePicker()) {
                for (typename std::vector<Sample>::const_iterator i = _samples.begin();
                        i != _samples.end(); ++i) {

                    typename Sample::desired_type actual = learner.output(i->input);

                    if (picker(*i, actual))
                        learner.learn(i->input, learn_error(actual, i->desired));
                }
            }

            // Run against whole set
            void learn_run() {
                learn_run<learn_always<Sample> >();
            }

            // Calculate MSE for all samples
            template <typename SquareError>
            double mse(SquareError const &sq_err = SquareError()) {
                double acc_error = 0.0;

                for (typename std::vector<Sample>::const_iterator i = _samples.begin();
                        i != _samples.end(); ++i) {

                    typename Sample::desired_type actual = learner.output(i->input);

                    acc_error += sq_err(*i, actual);
                }

                return acc_error / samples_count();
            }

        private:
            std::vector<Sample> _samples;
            Learner &learner;
            LearnError learn_error;
        };

        // ------------------

        template<typename Stream, typename Desired>
        Stream &operator<<(Stream& os, const sample<Desired> &sample) {
            return os << "Input: " << sample.input << " Desired: " << sample.desired << std::endl;
        }

        // ------------------
        template <int K>
        class single_discrete_match {
        public:
            bool operator()(const cmplx &output, const cmplx &sample) const {
                return sector_number(K, output) == sector_number(K, sample);
            }
        };
    }
}