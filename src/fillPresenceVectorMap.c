#include "fillPresenceVectorMap.h"

typedef struct fillPresenceVectorMapProcParams_t {
	int result;
	const binIndex* leftLookupIndex;
	const binIndex* rightLookupIndex;
	const int* inputVector;
	int* resultVector;
}fillPresenceVectorMapProcParams_t;

static int fillPresenceVectorMapProc(char* line,size_t size,void* params) {
	fillPresenceVectorMapProcParams_t* vparams=(fillPresenceVectorMapProcParams_t*)params;
	index_t lCount=binIndexGetCount(vparams->leftLookupIndex);
	index_t rCount=binIndexGetCount(vparams->rightLookupIndex);
	char leftItem[size];
	char rightItem[size];
	if(sscanf(line,"%s\t%s",leftItem,rightItem)==2) {
		index_t left=binIndexLookup2(vparams->leftLookupIndex,leftItem);
		if(left<lCount) {
			if(vparams->inputVector[left]>0) {
				index_t right=binIndexLookup2(vparams->rightLookupIndex,rightItem);
				if(right<rCount) {
					vparams->resultVector[right]++;
					vparams->result++;
				}else{
					fprintf(stderr, "not found right : [%s]\n", rightItem);
				}
			}
		}else{
			fprintf(stderr, "not found left : [%s]\n", leftItem);
		}		
	}
	return 1;
}

extern int fillPresenceVectorMap(int* resultVector,const binIndex* leftLookupIndex,const binIndex* rightLookupIndex,const int* inputVector,FILE* mapInput) {
	fillPresenceVectorMapProcParams_t params=(fillPresenceVectorMapProcParams_t) {
		0,
		leftLookupIndex,
		rightLookupIndex,
		inputVector,
		resultVector
	};
	int res=readlineLoop(mapInput,fillPresenceVectorMapProc,(void*)&params);
	return params.result;
}
