//
// Created by dev on 1/10/2018.
//


#include "FrequencyPhaseEstimator.hpp"
BOOST_COMPUTE_FUNCTION(bool, clCompareAgentsByScore,
                       (PowerRayCat::Agent a, PowerRayCat::Agent b),
                       {
                           return a.score > b.score;
                       });
bool PowerRayCat::compareAgentsByScore(PowerRayCat::Agent a, PowerRayCat::Agent b) {
    return a.score > b.score;
}
PowerRayCat::Agent PowerRayCat::FrequencyPhaseEstimator::run(bool openCL) {

    //Create random initial population
    Population X(parameters.popSize,Agent());
    populations.clear();
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> unifrd;

    bc::device dev = bc::system::default_device();
    bc::context ctx(dev);
    bc::command_queue queue(ctx,dev);
    bc::program program =
            bc::program::create_with_source_file("../program.cl",ctx);
    program.build();
    bc::kernel clCalcPopScores(program,"calcPopulationScores");



    std::for_each(X.begin(),X.end(),[&eng,&unifrd](Agent &a){
        a.omega = unifrd(eng);
        a.phi = unifrd(eng);
    });
    Population bestAgentsByGeneration;
    //Calculate score of initial population
    if(openCL)
    {
        bc::vector<float> clData(data->begin(),data->end(),queue);
        bc::vector<Agent> clX(X.begin(),X.end(),queue);
        clCalcPopScores.set_args(
                clX,
                parameters.popSize,
                clData,
                data->size(),
                parameters.sps,
                parameters.idealAmplitude,
                parameters.idealOmega*(1.0f-parameters.omegaPercentage),
                parameters.idealOmega*(1.0f+parameters.omegaPercentage),
                2.0f*(float)M_PI
        );
        queue.enqueue_1d_range_kernel(
                clCalcPopScores,0,parameters.popSize,0);
        bc::sort(clX.begin(),clX.end(),clCompareAgentsByScore,queue);
        bc::copy(clX.begin(),clX.end(),X.begin(),queue);
        populations.emplace_back(X);
    }
    else {
        calcPopScores(
                X,
                parameters.popSize,
                data,
                data->size(),
                parameters.sps,
                parameters.idealAmplitude,
                parameters.idealOmega*(1.0f-parameters.omegaPercentage),
                parameters.idealOmega*(1.0f+parameters.omegaPercentage),
                2.0f*(float)M_PI);
        std::sort(X.begin(),X.end(),compareAgentsByScore);
        populations.emplace_back(X);
    }


    return *X.begin();
}

float PowerRayCat::FrequencyPhaseEstimator::calcSignalSquaredSum() {
    float ss = std::accumulate(data->begin(),data->end(),0.0f,
                    [](float accum, float val) { return accum+val*val; });

    return ss;
}

PowerRayCat::Population *
PowerRayCat::FrequencyPhaseEstimator::calcPopScores(PowerRayCat::Population &X, const size_t popSize,
                                                    std::vector<float> *data, const size_t dataSize, const size_t sps,
                                                    const float amplitude, const float omegaMin,
                                                    const float omegaMax, const float phiMax) {
    for(auto &agent : X) {
        float score = 0.0f;
        float omegaSpan = omegaMax-omegaMin;
        float realOmega = omegaMin+agent.omega*omegaSpan;
        float realPhi = agent.phi*phiMax;
        float t = 0.0f;
        for(unsigned int p=0;p<dataSize;p++) {
            t = (float)p/(float)sps;
            score +=
                    pow(amplitude*sin(realOmega*t+realPhi)-data->at(p),2.0f);
        }
        agent.score = score;
    }
    return &X;
}

