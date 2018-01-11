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
        PowerRayCat::FittingParameters fittingParameters = {
                1800,
                80,
                PowerRayCat::GoalFunction::MaxGenerations,
                1,
                0.05f,
                0.7f,
                0.7f,
                1.0f,
                2.0f*(float)M_PI*1.0f,
                0.05f,
                8000
        };

        void SetUp() override {
            signal = new SynthSignal::Signal();
            const float omega = 2.0f * M_PI * 6.0f;
            auto wf =
                    new SynthSignal::SineWaveform(
                            1.0f, omega , 0.0f);
            SynthSignal::Interpolation interpolation;
            interpolation.addPoint(0.0f, 1.0f);
            interpolation.addPoint(10.0f, 1.0f);
            signal->addEvent(wf, interpolation);
            signal->gen(10.0f, 8000);
            fittingParameters.idealAmplitude = 1.0f;
            fittingParameters.idealOmega = omega;
            fittingParameters.omegaPercentage = 0.05f;
        }

        void TearDown() override {
            delete signal->lastGen;
            delete signal;
        }
    };

    TEST_F(SyntheticWaveform,DISABLED_OpenCLAreAgentsSorted) {
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


    TEST_F(SyntheticWaveform,DISABLED_AreAgentsSorted) {
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

    TEST_F(SyntheticWaveform,OpenCLIsFittingGoalReached) {
        PowerRayCat::FrequencyPhaseEstimator estimator(
                signal->lastGen,fittingParameters
        );
        auto bestAgent = estimator.run(true);
        float percentage =
                bestAgent.score/estimator.signalSquaredSum;
        ASSERT_LE(percentage,0.000005f);
    }

    TEST_F(SyntheticWaveform,IsFittingGoalReached) {
        PowerRayCat::FrequencyPhaseEstimator estimator(
                signal->lastGen,fittingParameters
        );
        auto bestAgent = estimator.run(false);
        float percentage =
                bestAgent.score/estimator.signalSquaredSum;
        ASSERT_LE(percentage,0.000005f);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}