#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "binIndex.h"
#include "fillPresenceVectorMap.h"

static void showUsage(char *exeName) {
	fprintf(stderr, "Usage : \n\t%s goIndex symbolIndex symbol2GoMapping < inputSymbolsList\n", exeName);
}

int main(int argc,char** argv) {
	if(argc<4) {
		showUsage(argv[0]);
		abort();
	}
	const char* goIndexFilename=argv[1];
	const char* symbolIndexFilename=argv[2];
	const char* symbol2goFilename=argv[3];
	//const char* goCountFileName=argv[4];

	binIndex* goIndex=binIndexLoadRev(goIndexFilename);
	binIndex* symbolIndex=binIndexLoadRev(symbolIndexFilename);

	int goCount=binIndexGetCount(goIndex);
	int symbolCount=binIndexGetCount(symbolIndex);

	int symbolPresenceVector[symbolCount];
	memset(symbolPresenceVector,0,sizeof(int)*symbolCount);
	int count=fillPresenceVector(symbolPresenceVector,symbolIndex,stdin);
	
	int goCountVector[goCount];
	memset(goCountVector,0,sizeof(int)*goCount);
	FILE* symbol2go=fopen(symbol2goFilename,"r");
	assert(symbol2go!=NULL);
	count=fillPresenceVectorMap(goCountVector,symbolIndex,goIndex,symbolPresenceVector,symbol2go);
	fclose(symbol2go);

	for(int i=0;i<goCount;i++) {
		//char* name=binIndexReverseLookup(goIndex,i);
		if(goCountVector[i]>0) {
			fprintf(stdout,"%s\t%i\n",
					binIndexReverseLookup(goIndex,i),
					goCountVector[i]
					);
		}
	}
}
