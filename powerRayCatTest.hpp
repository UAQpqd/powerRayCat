//
// Created by dev on 1/13/2018.
//

#ifndef POWERRAYCAT_POWERRAYCATTEST_HPP
#define POWERRAYCAT_POWERRAYCATTEST_HPP

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "synthSignal/Signal.hpp"
#include "minusDarwin/Solver.hpp"

namespace PowerRayCatTest {
    class SyntheticWaveform : public ::testing::Test {
    public:
        SynthSignal::Signal *signal;
        MinusDarwin::Solver *solver;

        void SetUp() override {
            const float time = 1.0f;
            const size_t sps = 8000;
            const float a = 100.0f;
            const float omega = 2.0f * (float) M_PI * 60.0f;
            const float omegaMin = omega * 0.95f;
            const float omegaMax = omega * 1.05f;
            const float phiMax = 2.0f * (float) M_PI;
            const float phi = 0.0f;

            signal = new SynthSignal::Signal();
            auto wf =
                    new SynthSignal::SineWaveform(
                            a, omega, phi);
            SynthSignal::Interpolation interpolation;
            interpolation.addPoint(0.0f, 1.0f);
            interpolation.addPoint(10.0f, 1.0f);
            signal->addEvent(wf, interpolation);
            signal->gen(time, sps);
            std::vector<float> signalData(signal->lastGen->begin(),
                                          signal->lastGen->end());
            const float sumOfSquares = std::accumulate(
                    signalData.begin(), signalData.end(), 0.0f,
                    [](float accum, float val) { return accum + val * val; });
            MinusDarwin::SolverParameterSet solverParameterSet = {
                    2, 2400, 40, MinusDarwin::GoalFunction::EpsilonReached,
                    MinusDarwin::CrossoverMode::Random, 1,
                    0.0005f, 0.7f, 0.7f
            };
            std::function<float(std::vector < float > )> fitError =
                    [signalData, sumOfSquares, sps, a,
                            omegaMin, omegaMax, phiMax](std::vector<float> v) -> float {
                        float error = 0.0f;
                        for (size_t p = 0; p < signalData.size(); p++) {
                            float t = (float) p / (float) sps;
                            float realOmega = omegaMin + v.at(0) * (omegaMax - omegaMin);
                            float realPhi = v.at(1) * phiMax;
                            float estimated =
                                    a * sin(realOmega * t + realPhi);
                            error += pow(estimated - signalData.at(p), 2.0f);
                        }
                        return error / sumOfSquares;
                    };
            solver = new MinusDarwin::Solver(
                    solverParameterSet, fitError, bc::system::default_device()
            );
        }

        void TearDown() override {
            delete signal->lastGen;
            delete signal;
        }
    };
}
#endif //POWERRAYCAT_POWERRAYCATTEST_HPP
