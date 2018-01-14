//
// Created by dev on 1/10/2018.
//
#include "powerRayCatTest.hpp"
namespace PowerRayCatTest {
    TEST_F(SyntheticWaveform,IsEpsilonReachedUsingRandom) {
        MinusDarwin::Agent result = solver->run(true);
        ASSERT_LE(solver->evaluateAgent(result),0.0005f);
    }
    TEST_F(SyntheticWaveform,IsEpsilonReachedUsingBest) {
        solver->sParams.mode = MinusDarwin::CrossoverMode::Best;
        MinusDarwin::Agent result = solver->run(true);
        ASSERT_LE(solver->evaluateAgent(result),0.0005f);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}