//
// Created by dev on 1/10/2018.
//


#include "FrequencyPhaseEstimator.hpp"

BOOST_COMPUTE_FUNCTION(bool, clCompareAgentsByScore,
                       (PowerRayCat::Agent
                               a, PowerRayCat::Agent
                               b),
                       {
                           return a.score < b.score;
                       });

bool PowerRayCat::compareAgentsByScore(PowerRayCat::Agent a, PowerRayCat::Agent b) {
    return a.score < b.score;
}

PowerRayCat::Agent PowerRayCat::FrequencyPhaseEstimator::run(bool openCL) {

    //Create random initial population
    Population X(parameters.popSize, Agent());

    Population Y(parameters.popSize, Agent());

    populations.clear();
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> unifrd(0, 1);
    std::uniform_int_distribution<size_t> unifid(0, 1);
    std::vector<CrossoverRandoms> cvRands(parameters.popSize);


    signalSquaredSum = calcSignalSquaredSum();

    bc::device dev;
    bc::context ctx;
    bc::command_queue queue;
    bc::program program;

    bc::kernel clCalcPopScores;
    bc::kernel clCreateAuxPopulation;
    bc::vector<Agent> clX, clY;
    bc::vector<float> clData;
    bc::vector<size_t> clNeighbours;
    bc::vector<CrossoverRandoms> clCvRands;
    bc::vector<size_t> clDeltas;


    std::for_each(X.begin(), X.end(), [&eng, &unifrd](Agent &a) {
        a.omega = unifrd(eng);
        a.phi = unifrd(eng);
    });
    std::for_each(cvRands.begin(), cvRands.end(),
                  [&eng, &unifrd, &unifid](CrossoverRandoms &c) {
                      c.delta = unifid(eng);
                      c.omega = unifrd(eng);
                      c.phi = unifrd(eng);
                  });
    Population bestAgentsByGeneration;
    size_t bestAgentId;
    //Calculate score of initial population
    if (openCL) {
        dev = bc::system::default_device();
        ctx = bc::context(dev);
        queue = bc::command_queue(ctx, dev);
        program = bc::program::create_with_source_file("../program.cl", ctx);
        program.build();
        clCalcPopScores = bc::kernel(program, "calcPopulationScores");
        clCreateAuxPopulation = bc::kernel(program, "createAuxPopulation");

        clData = bc::vector<float>(data->begin(), data->end(), queue);
        clX = bc::vector<Agent>(X.begin(), X.end(), queue);
        clY = bc::vector<Agent>(parameters.popSize, ctx);
        clNeighbours = bc::vector<size_t>(kNeighboursSize, ctx);
        clCvRands =
                bc::vector<CrossoverRandoms>(cvRands.begin(),cvRands.end(),queue);

        clCalcPopScores.set_args(
                clX,
                parameters.popSize,
                clData,
                data->size(),
                parameters.sps,
                parameters.idealAmplitude,
                parameters.idealOmega * (1.0f - parameters.omegaPercentage),
                parameters.idealOmega * (1.0f + parameters.omegaPercentage),
                2.0f * (float) M_PI
        );
        queue.enqueue_1d_range_kernel(
                clCalcPopScores, 0, parameters.popSize, 0);
        auto bestAgentIt =
                bc::max_element(clX.begin(), clX.end(), clCompareAgentsByScore, queue);
        size_t bestAgentId = bestAgentIt.get_index();
        bc::copy(clX.begin(), clX.end(), X.begin(), queue);
        populations.emplace_back(X);
        lastResult = X.at(bestAgentId);
    } else {
        calcPopScores(
                X,
                parameters.popSize,
                data,
                data->size(),
                parameters.sps,
                parameters.idealAmplitude,
                parameters.idealOmega * (1.0f - parameters.omegaPercentage),
                parameters.idealOmega * (1.0f + parameters.omegaPercentage),
                2.0f * (float) M_PI);
        auto bestAgentIt =
                std::max_element(X.begin(), X.end(), compareAgentsByScore);
        size_t bestAgentId = bestAgentIt - X.begin();
        populations.emplace_back(X);
    }

    bool epsilonReached = isEpsilonReachedTrue(X);
    for (unsigned int g = 0; g < parameters.maxGenerations && !epsilonReached; g++) {
        auto neighbours = calculateNeighbours(bestAgentId);
        if (openCL) {
            bc::copy(neighbours.begin(), neighbours.end(),
                     clNeighbours.begin(), queue);
            clCreateAuxPopulation.set_args(
                    clX,
                    clY,
                    clNeighbours,
                    parameters.modeDepth,
                    clCvRands,
                    parameters.crossOverProbability,
                    parameters.differentialFactor
            );
            queue.enqueue_1d_range_kernel(
                    clCreateAuxPopulation, 0, parameters.popSize, 0);
        } else {
            createAuxPopulation(X, Y, cvRands);
        }
        epsilonReached = isEpsilonReachedTrue(X);
    }

    lastResult = X.at(bestAgentId);
    return lastResult;
}

bool PowerRayCat::FrequencyPhaseEstimator::isEpsilonReachedTrue(const Population &X) {
    return parameters.goalFunction == GoalFunction::EpsilonReached &&
           parameters.epsilon >= X.at(0).score / signalSquaredSum;
}

float PowerRayCat::FrequencyPhaseEstimator::calcSignalSquaredSum() {
    float ss = std::accumulate(data->begin(), data->end(), 0.0f,
                               [](float accum, float val) { return accum + val * val; });
    return ss;
}

PowerRayCat::Population *
PowerRayCat::FrequencyPhaseEstimator::calcPopScores(PowerRayCat::Population &X, const size_t popSize,
                                                    std::vector<float> *data, const size_t dataSize, const size_t sps,
                                                    const float amplitude, const float omegaMin,
                                                    const float omegaMax, const float phiMax) {
    for (auto &agent : X) {
        float score = 0.0f;
        float omegaSpan = omegaMax - omegaMin;
        float realOmega = omegaMin + agent.omega * omegaSpan;
        float realPhi = agent.phi * phiMax;
        float t = 0.0f;
        for (unsigned int p = 0; p < dataSize; p++) {
            t = (float) p / (float) sps;
            score +=
                    pow(amplitude * sin(realOmega * t + realPhi) - data->at(p), 2.0f);
        }
        agent.score = score;
    }
    return &X;
}

std::vector<size_t> PowerRayCat::FrequencyPhaseEstimator::calculateNeighbours(const size_t &bestAgentId) {
    std::vector<size_t> neighbours(kNeighboursSize);
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<size_t> unifid(0, 1);
    for (size_t agentId = 0; agentId < parameters.popSize; agentId++) {
        size_t offset = agentId * (1 + 2 * parameters.modeDepth);
        size_t endOffset = (agentId + 1) * (1 + 2 * parameters.modeDepth);
        size_t n = 0;
        if (parameters.goalFunction == GoalFunction::EpsilonReached) {
            neighbours.at(offset) = bestAgentId;
            n = 1;
        }
        for (; n < 1 + parameters.modeDepth * 2; n++) {
            int selected;
            do {
                selected = unifid(eng);
            } while (
                    selected == agentId ||
                    std::find(
                            neighbours.begin() + offset,
                            neighbours.begin() + endOffset,
                            selected) != neighbours.begin() + endOffset);
            neighbours.at(offset + n) = selected;
        }
    }
    return neighbours;
}
#define RANDOMPARAMBYID(cvRandoms,agentId,paramId) ( \
    (paramId)==0? \
    (cvRandoms)[(agentId)].omega : \
    (cvRandoms)[(agentId)].phi)
#define NEIGHBOURPARAMBYID(population,neighbours,neighbourId,paramId) ( \
    (paramId)==0? \
    (population)[(neighbours)[1+(neighbourId)]].omega : \
    (population)[(neighbours)[1+(neighbourId)]].phi)
#define AGENTPARAMBYID(agentPtr,paramId) ( \
    (paramId)==0? \
    (agentPtr)->omega : \
    (agentPtr)->phi)
#define AGENTPARAMPTRBYID(agentPtr,paramId) ( \
    (paramId)==0? \
    &(agentPtr)->omega : \
    &(agentPtr)->phi)

void PowerRayCat::FrequencyPhaseEstimator::createAuxPopulation(std::vector<PowerRayCat::Agent> &X,
                                                               std::vector<PowerRayCat::Agent> &Y,
                                                               std::vector<PowerRayCat::CrossoverRandoms> &cvRands) {
    for (unsigned int k = 0; k < 2; k++) {
        std::vector<float *> srcParams = {&X.at(k).omega, &X.at(k).phi};
        std::vector<float *> dstParams = {&Y.at(k).omega, &Y.at(k).phi};
        float delta = cvRands.at(k).delta;
        for (unsigned int paramId = 0; paramId < 2; paramId++) {
            if (cvRands.at(offset) < parameters.crossOverProbability || delta == paramId) {
                float c =
                        paramId == 0 ? X.at(NEIGHBOURIDX(k, 0)).omega : X.at(NEIGHBOURIDX(k, 0)).phi;
                float cDiff = 0.0f;
                for (unsigned int n = 1; n < 2 * modeDepth; n++) {
                    if (n % 2 == 0)
                        cDiff += paramId == 0 ? X.at(NEIGHBOURIDX(k, n)).omega : X.at(NEIGHBOURIDX(k, n)).phi;
                    else cDiff -= paramId == 0 ? X.at(NEIGHBOURIDX(k, n)).omega : X.at(NEIGHBOURIDX(k, n)).phi;
                }
                *dstParams.at(paramId) = std::min(1.0f, std::max(0.0f,
                                                                 c +
                                                                 configuration.parameters.params.differentialFactor *
                                                                 cDiff));
            } else {
                *dstParams.at(paramId) = *srcParams.at(paramId);
            }
        }

    }
}

