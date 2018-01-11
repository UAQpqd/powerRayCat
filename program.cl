typedef struct { float omega; float phi; float score; } Agent;

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