#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "binFormat.h"
#include "binIndex.h"
#include "utils.h"

typedef struct degreeEntry_t {
	int count;
	int positive;
	int negative;
	index_t id;
}degreeEntry_t;

typedef struct readDegreesProcParams_t {
	binIndex* lookup;
	degreeEntry_t* results;
	int count;
}readDegreesProcParams_t;

void setEntry(degreeEntry_t* results,const index_t sid,const int count,const int positive,const int negative) {
	results[sid].id=sid;
	results[sid].count=count;
	results[sid].positive=positive;
	results[sid].negative=negative;
}

int readDegreesProc(char* line,size_t size,void* params) {
	readDegreesProcParams_t* vparams=(readDegreesProcParams_t*) params;
	char symbol[size];
	int count=0;
	int positive=0;
	int negative=0;

	if(sscanf(line,"%s\t%i\t%i\t%i",symbol,&count,&positive,&negative)==4) {
		index_t sid=binIndexLookup2(vparams->lookup,symbol);
		if(sid<binIndexGetCount(vparams->lookup)) {
			setEntry(vparams->results,sid,count,positive,negative);
			vparams->count++;
		}else{
			fprintf(stderr, "addDegree : symbol not mapped [%s]\n",symbol );
		}
	}else{
		fprintf(stderr, "addDegree : skipping [%s]\n", line);
	}
	return 1;
}

int printLineProc(char* line,size_t size,void* params) {
	readDegreesProcParams_t* vparams=(readDegreesProcParams_t*) params;
	char symbol[size];
	int count=0;
	int positive=0;
	int negative=0;
	if(sscanf(line,"%s\t%i\t%i\t%i",symbol,&count,&positive,&negative)==4) {
		index_t sid=binIndexLookup2(vparams->lookup,symbol);
		if(sid<binIndexGetCount(vparams->lookup)) {
			fprintf(stdout, "%s\t%i\t%i\t%i\n", 
				line,
				vparams->results[sid].count,
				vparams->results[sid].positive,
				vparams->results[sid].negative
				);
			vparams->count++;
		}else{
			fprintf(stderr, "addDegree : symbol not mapped [%s]\n",symbol );
		}
	}else{
		fprintf(stderr, "skipping [%s]\n", line);
		fprintf(stdout, "%s\n", line);
	}
	return 1;
}

int main(int argc, char *argv[]) {
	binIndex* symbolsIndex = binIndexLoadRev(argv[1]); // full symbols list
	const int symbolsCount=binIndexGetCount(symbolsIndex);

	degreeEntry_t results[symbolsCount]; // degrees 
	for(int i=0;i<symbolsCount;i++) {
		// initialize structure 
		results[i]=(degreeEntry_t){0,0,0,i};
	}

	readDegreesProcParams_t params=(readDegreesProcParams_t) {
		symbolsIndex,
		results,
		0
	};

	FILE* input=fopen(argv[2],"r"); // degree files
	assert(input!=NULL);
	int res=readlineLoop(input,readDegreesProc,(void*)&params);
	fprintf(stderr, "%i degrees loaded (%i lines) (%i expected)\n",params.count,res,symbolsCount );
	//assert(params.count==symbolsCount);
	fclose(input);

	params.count=0;

	res=readlineLoop(stdin,printLineProc,(void*)&params); // print line with associated degrees
	//assert(params.count==symbolsCount);
	binIndexFree(symbolsIndex);
	return(EXIT_SUCCESS); 
}
