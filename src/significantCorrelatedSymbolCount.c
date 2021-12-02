#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "binFormat.h"
#include "binIndex.h"
#include "utils.h"

typedef struct countEntry_t {
	index_t id;
	int count;
	int positive;
	int negative;
} countEntry_t;

static int fillCountEntry(countEntry_t* results,const index_t symbol,const double correlation,const int factor) {
	results[symbol].count += factor;
	if(correlation > 0.0L) {
		results[symbol].positive += factor;
		return 1;
	}else{
		results[symbol].negative += factor;
		return -1;
	}
}

typedef struct fillCountProcParams_t {
	binIndex* lookup;
	countEntry_t* results;
	int* presenceVector;
	int result;
} fillCountProcParams_t;

static int fillCountProc(char* line,size_t size,void* params) {
	fillCountProcParams_t *vparams=(fillCountProcParams_t*)params;

	index_t indexCount=binIndexGetCount(vparams->lookup);

	char s1[size];
	char s2[size];
	double correlation;

	if(sscanf(line,"%s\t%s\t%lf",s1,s2,&correlation)==3) {
		index_t s1Id=binIndexLookup2(vparams->lookup,s1);
		index_t s2Id=binIndexLookup2(vparams->lookup,s2);
		if((s1Id<indexCount)&&(s2Id<indexCount)) {
			if(vparams->presenceVector[s1Id]^vparams->presenceVector[s2Id]) {
				index_t cIndex=(vparams->presenceVector[s1Id])?s2Id:s1Id;
				fillCountEntry(vparams->results,cIndex,correlation,1);
			}else if(vparams->presenceVector[s1Id]&&vparams->presenceVector[s2Id]){
				fillCountEntry(vparams->results,s1Id,correlation,1);
				fillCountEntry(vparams->results,s2Id,correlation,1);
			}
			vparams->result++;
		}else{
			fprintf(stderr, "significant correlation count : mapping not found [%s : %i ][ %s : %i ]\n", s1,s1Id,s2,s2Id);
		}
	}
	return 1;
}

static int fillCount(countEntry_t* results,const binIndex* lookup,const int* presenceVector,FILE* input) {
	fillCountProcParams_t params=(fillCountProcParams_t){
		lookup,
		results,
		presenceVector,
		0
	};
	int res=readlineLoop(input,fillCountProc,(void*)&params);
	return params.result;
}

int main(int argc, char *argv[]) {
	binIndex* symbolsIndex = binIndexLoadRev(argv[1]); // symbols
	const int symbolsCount=binIndexGetCount(symbolsIndex);
	int result=0;
	int presenceVector[symbolsCount];
	memset(presenceVector,0,sizeof(int)*symbolsCount);
	result=fillPresenceVector(presenceVector,symbolsIndex,stdin);
	fprintf(stderr,"%i symbols \n",result);

	fprintf(stdout, "%i\n",result ); //output symobols count

	countEntry_t results[symbolsCount];
	for(int i=0;i<symbolsCount;i++) {
		results[i]=(countEntry_t){i,0,0,0};
	}
	FILE* input=fopen(argv[2],"r");
	assert(input!=NULL);
	fillCount(results,symbolsIndex,presenceVector,input);
	fclose(input);

	for(int i=0;i<symbolsCount;i++) {
		fprintf(stdout, "%s\t%i\t%i\t%i\t%i\n", 
			binIndexReverseLookup(symbolsIndex,results[i].id), 
			results[i].count,
			results[i].positive,
			results[i].negative,
			presenceVector[i]
			);
	}
	binIndexFree(symbolsIndex);
}
