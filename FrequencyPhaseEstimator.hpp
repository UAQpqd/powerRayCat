//
// Created by dev on 1/10/2018.
//

#ifndef POWERRAYCAT_FREQUENCYPHASEESTIMATOR_HPP
#define POWERRAYCAT_FREQUENCYPHASEESTIMATOR_HPP

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <boost/compute.hpp>

namespace bc = boost::compute;

namespace PowerRayCat {
    struct Agent { float omega, phi, score; };
    bool compareAgentsByScore(PowerRayCat::Agent a, PowerRayCat::Agent b);
    enum class GoalFunction { MaxGenerations, EpsilonReached };
    struct FittingParameters {
        size_t popSize = 1800;
        size_t maxGenerations = 80;
        GoalFunction goalFunction = GoalFunction::MaxGenerations;
        float epsilon = 0.05f;
        float crossOverProbability = 0.7f;
        float differentialFactor = 0.7f;
        float idealAmplitude = 1.0f;
        float idealOmega = 2.0f*(float)M_PI*1.0f;
        float omegaPercentage = 0.05f;
        size_t sps = 8000;
    };
    typedef std::vector<Agent> Population;
    class FrequencyPhaseEstimator {
    public:
        FrequencyPhaseEstimator(
                std::vector<float> *inData,
                FittingParameters inParameters) :
                data(inData),
                parameters(inParameters)
                { signalSquaredSum = calcSignalSquaredSum(); };
        float calcSignalSquaredSum();
        Agent run(bool openCL = true);
        std::vector<float> *data;
        FittingParameters parameters;
        Agent lastResult = { 0.0f, 0.0f, 1.0f };
        float signalSquaredSum = 0.0f;
        std::vector<Population> populations;
    private:
        Population *calcPopScores(
                Population &X,
                const size_t popSize,
                std::vector<float> *data,
                const size_t dataSize,
                const size_t sps,
                const float amplitude,
                const float omegaMin,
                const float omegaMax,
                const float phiMax);
    };
}

BOOST_COMPUTE_ADAPT_STRUCT(PowerRayCat::Agent, Agent, (omega,phi,score));
#endif //POWERRAYCAT_FREQUENCYPHASEESTIMATOR_HPP
