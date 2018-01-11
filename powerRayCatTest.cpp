//
// Created by dev on 1/10/2018.
//
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "synthSignal/Signal.hpp"
#include "FrequencyPhaseEstimator.hpp"

namespace PowerRayCatTest {
    class SyntheticWaveform : public ::testing::Test {
    public:
        SynthSignal::Signal *signal;

        void SetUp() override {
            signal = new SynthSignal::Signal();
            auto wf =
                    new SynthSignal::SineWaveform(
                            1.0f, 2.0f * M_PI * 6.0f, 0.0f);
            SynthSignal::Interpolation interpolation;
            interpolation.addPoint(0.0f, 1.0f);
            interpolation.addPoint(10.0f, 1.0f);
            signal->addEvent(wf, interpolation);
            signal->gen(10.0f, 8000);
        }

        void TearDown() override {
            delete signal->lastGen;
            delete signal;
        }
    };

    TEST_F(SyntheticWaveform,OpenCLAreAgentsSorted) {
        PowerRayCat::FittingParameters fittingParameters;
        PowerRayCat::FrequencyPhaseEstimator estimator(
                signal->lastGen,fittingParameters
        );
        estimator.run(true);
        ASSERT_TRUE(std::is_sorted(
                estimator.populations.at(0).begin(),
                estimator.populations.at(0).end(),
                PowerRayCat::compareAgentsByScore
        ));
    }

    TEST_F(SyntheticWaveform,AreAgentsSorted) {
        PowerRayCat::FittingParameters fittingParameters;
        PowerRayCat::FrequencyPhaseEstimator estimator(
                signal->lastGen,fittingParameters
        );
        estimator.run(false);
        ASSERT_TRUE(std::is_sorted(
                estimator.populations.at(0).begin(),
                estimator.populations.at(0).end(),
                PowerRayCat::compareAgentsByScore
        ));
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}