#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "binIndex.h"
#include <Rmath.h>
#include <stdbool.h>
#include <string.h>
#include "utils.h"

typedef struct readCountParam_t
{
	binIndex* index;
	int* vector;
} readCountParam_t;

static int readGoCountProc(char* line,size_t size,void* params) {
	readCountParam_t *vparams=(readCountParam_t*)params;
	int indexCount=binIndexGetCount(vparams->index);
	char name[size];
	int count=0;
	if(sscanf(line,"%s\t%i",name,&count)==2) {
		int index=binIndexLookup2(vparams->index,name);
		if(index<indexCount) {
			vparams->vector[index]=count;
		}else{
			fprintf(stderr, "[%s] is not in the index\n",name );

		}
	}
	return 1;
}

static int readGoCount(int* vector,binIndex* index,FILE* input) {
	readCountParam_t params={
		index,
		vector
	};
	int res=readlineLoop(input,readGoCountProc,(void*)&params);
	return res;
}

static void showUsage(char *exeName) {
	fprintf(stderr, "Usage : \n\t%s goIndex totalGoIdsCount inputGenesCount totalGenesCount referenceGoCount < inputGoCount\n", exeName);
}

int main(int argc,char** argv) {
	if(argc < 6) {
		showUsage(argv[0]);
		abort();
	}
	char *goIndexFilename=argv[1];
	int totalGoIds=atoi(argv[2]);
	int sizeInputGenes=atoi(argv[3]);
	int sizeGenomeWithGo=atoi(argv[4]);
	char *goCountReferenceFilename=argv[5];
	
	assert((totalGoIds*sizeInputGenes*sizeGenomeWithGo)!=0);
	
	binIndex* goIndex=binIndexLoadRev(goIndexFilename);
	int goCount=binIndexGetCount(goIndex);
	fprintf(stderr, "%s\n", goCountReferenceFilename);
	FILE *goCountReference=fopen(goCountReferenceFilename,"r");
	assert(goCountReference!=NULL);
	int goCountReferenceVector[goCount];
	memset(goCountReferenceVector,0,sizeof(int)*goCount);
	int count=readGoCount(goCountReferenceVector,goIndex,goCountReference);
	fclose(goCountReference);

	int goCountInputVector[goCount];
	memset(goCountInputVector,0,sizeof(int)*goCount);
	count=readGoCount(goCountInputVector,goIndex,stdin);

	for(int i=0;i<goCount;i++) {
		int inputCount=goCountInputVector[i];
		if(inputCount>0) {
			int inputValue=goCountInputVector[i];
			int reference=goCountReferenceVector[i];
			double pValue=phyper(
					inputCount-1.0, //x
					reference, // NR
					sizeGenomeWithGo-reference, //NB
					sizeInputGenes, // n
					false,
					false);
			double expected = 1.0L*reference*sizeInputGenes/sizeGenomeWithGo;
			double foldEnrichment = 1.0L*inputValue/expected;
			double pValueCorrected = pValue*totalGoIds;
			if(pValueCorrected > 1.0L) {
				pValueCorrected=1.0L;
			}
			fprintf(stdout,"%s\t%i\t%.3lf\t%.3lf\t%.3lf\t%.3lf\n",
					binIndexReverseLookup(goIndex,i),
					inputCount,
					expected,
					foldEnrichment,
					pValue,
					pValueCorrected
					);
		}
	}
}
