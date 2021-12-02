#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "binFormat.h"
#include "binIndex.h"
#include "utils.h"

typedef struct readGcClassStatsFileProcParams_t {
	double* results;
	binIndex* motifIndex;
	int result;
} readGcClassStatsFileProcParams_t;

static int readGcClassStatsFileProc(char* line,size_t size,void* params) {
	readGcClassStatsFileProcParams_t* vparams=(readGcClassStatsFileProcParams_t*)params;
	int motifCount = binIndexGetCount(vparams->motifIndex);
	char motifItem[size];
	double c1=0.0L;
	double c2=0.0L;
	if(sscanf(line,"%s\t%lf%lf",motifItem,&c1,&c2)==3) {
		index_t motif=binIndexLookup2(vparams->motifIndex,motifItem);
		if(motif<motifCount) {
			vparams->results[motif]=c1;
			vparams->results[motif+motifCount]=c2;
			vparams->result++;
		}else{
			fprintf(stderr, "motif not found [%s]\n",motifItem );
		}
	}else{
		fprintf(stderr, "bad format [%s]\n", line);
	}
	return 1;
}

static int readGcClassStatsFile(double* results,binIndex* motifIndex,FILE* input) {
	readGcClassStatsFileProcParams_t params=(readGcClassStatsFileProcParams_t) {
		results,
		motifIndex,
		0
	};
	int res=readlineLoop(input,readGcClassStatsFileProc,(void*)&params);
	assert(res==params.result);
	return res;
}

typedef struct readPwm2tfsProcParams_t {
	char** tfs;
	const binIndex* motifIndex;
	int result;
}readPwm2tfsProcParams_t;

static int readPwm2tfsProc(char* line,size_t size,void* params) {
	readPwm2tfsProcParams_t* vparams=(readPwm2tfsProcParams_t*) params;
	int motifCount = binIndexGetCount(vparams->motifIndex);
	char motifItem[size];
	char tfs[size];
	if(sscanf(line,"%s\t%s",motifItem,tfs)==2) {
		index_t motif=binIndexLookup2(vparams->motifIndex,motifItem);
		if(motif<motifCount) {
			vparams->tfs[motif]=strdup(tfs);
			vparams->result++;
		}else{
			fprintf(stderr, "motif not found [%s]\n",motifItem );
		}
	}
	return 1;
}

static int readPwm2tfs(char** tfs,binIndex* motifIndex,FILE* input) {
	readPwm2tfsProcParams_t params=(readPwm2tfsProcParams_t) {
		tfs,
		motifIndex,
		0
	};
	int res=readlineLoop(input,readPwm2tfsProc,(void*)&params);
	return res;
}

typedef struct LineAddDataProcParams_t {
	const double* gcClassStats;
	const binIndex* motifIndex;
	const char** tfs;
	int result;
}LineAddDataProcParams_t;

static int LineAddDataProc(char* line,size_t size,void* params) {
	LineAddDataProcParams_t* vparams=(LineAddDataProcParams_t*)params;
	int motifCount = binIndexGetCount(vparams->motifIndex);
	char motifItem[size];
	int count=0;
	if(sscanf(line,"%s\t%i",motifItem,&count)==2) {
		index_t motif=binIndexLookup2(vparams->motifIndex,motifItem);
		if(motif<motifCount) {
			fprintf(stdout, "%s\t%s\t%i\t%lf\t%lf\n", 
				motifItem,
				vparams->tfs[motif],
				count,
				vparams->gcClassStats[motif],
				vparams->gcClassStats[motif+motifCount]);
			vparams->result++;
		}else{
			fprintf(stderr, "motif not found [%s]\n", motifItem);
		}
	}else{
		// the line that are not correctly formated should be the class counts
		fprintf(stdout,"%s\n",line);
	}
	return 1;
}

int main(int argc, char *argv[]) {
	binIndex* motifIndex = binIndexLoadRev(argv[1]);
	int motifCount = binIndexGetCount(motifIndex);

	double gcClassStats[motifCount*2];
	FILE* input;
	input=fopen(argv[2],"r"); // gcClassStats Probabilities
	assert(input!=NULL);
	int res=readGcClassStatsFile(gcClassStats,motifIndex,input);
	assert(res==motifCount);
	fclose(input);

	char** tfs=(char**)malloc(sizeof(char*)*motifCount);
	for(int i=0;i<motifCount;i++) {
		tfs[i]=NULL;
	}
	input=fopen(argv[3],"r"); // pwm2tfs name
	assert(input!=NULL);
	res=readPwm2tfs(tfs,motifIndex,input);
	assert(input!=NULL);
	fclose(input);	

	LineAddDataProcParams_t params=(LineAddDataProcParams_t) {
		gcClassStats,
		motifIndex,
		tfs,
		0
	};
	res=readlineLoop(stdin,LineAddDataProc,(void*)&params);

	for(int i=0;i<motifCount;i++) {
		if(tfs[i]!=NULL) {
			free(tfs[i]);
			tfs[i]=NULL;
		}
	}
	free(tfs);
	binIndexFree(motifIndex);
}
