typedef struct { float omega; float phi; float score; } Agent;
typedef struct { size_t delta; float omega; float phi; } CrossoverRandoms;
__kernel void calcPopulationScores(
	__global Agent *population,
	const size_t popSize,
	__global const float *data,
	const size_t dataSize,
	const size_t sps,
	const float amplitude,
	const float omegaMin,
	const float omegaMax,
	const float phiMax) {
	unsigned int k = get_global_id(0);
	if(k>=popSize) return;
	__global const Agent *agent = &population[k];
	//The agent must calculate its score
	float score = 0.0f;
	float omegaSpan = omegaMax-omegaMin;
	float realOmega = omegaMin+agent->omega*omegaSpan;
	float realPhi = agent->phi*phiMax;
	float t = 0.0f;
	for(unsigned int p=0;p<dataSize;p++) {
        t = (float)p/(float)sps;
        score +=
            pow(amplitude*sin(realOmega*t+realPhi)-data[p],2.0f);
	}
	population[k].score = score;
	return;
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

__kernel void createAuxPopulation(
	__global const Agent *srcPopulation,
	__global Agent *dstPopulation,
	__global const size_t *neighbors,
	const unsigned int modeDepth,
	__global const CrossoverRandoms *cvRandoms,
	const float crossoverProbability,
	const float differentialFactor) {
	unsigned int k = get_global_id(0);
	__global const Agent *srcAgent = &srcPopulation[k];
	__global Agent *dstAgent = &dstPopulation[k];
	__global const size_t *agentNeighbors = &neighbors[k*(1+2*modeDepth)]; //Neighbors of this agent
	unsigned int delta = cvRandoms[k].delta;
	for(unsigned int paramId = 0; paramId < 2; paramId++) {
	    if(RANDOMPARAMBYID(cvRandoms,k,paramId)<crossoverProbability || delta == paramId) {
	        float diff = 0.0f;
	        for(unsigned int n = 0; n<modeDepth*2;n++) {
	            float paramValue = NEIGHBOURPARAMBYID(srcPopulation,agentNeighbors,n,paramId);
	            if(n%2==0) diff += paramValue;
	            else diff -= paramValue;
	        }
	    }
	    else *(AGENTPARAMPTRBYID(dstAgent,paramId)) =
	        AGENTPARAMBYID(srcAgent,paramId);
	}
	return;
}