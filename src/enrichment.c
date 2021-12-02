#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "binFormat.h"
#include "binIndex.h"
#include "utils.h"

static int fillPresenceVectorMapProc(char* line,size_t size,void* params);
static int fillPresenceVectorMap(int*,const binIndex*,const binIndex*,const int*,FILE*);

static int readGCclassProc(char* line,size_t size,void* params);
static int readGCclass(int* class1Count,int* class2Count,const binIndex* refSeqIdx,const int* refseqPresenceVector,FILE* input);

void showUsage(char* programName);

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

static int fillPresenceVectorMap(int* resultVector,const binIndex* leftLookupIndex,const binIndex* rightLookupIndex,const int* inputVector,FILE* mapInput) {
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

typedef struct readGCclassProcParams_t {
	int result;
	int class1Count;
	int class2Count;
	const binIndex* refSeqIdx;
	const int* refseqPresenceVector;

}readGCclassProcParams_t;

static int readGCclassProc(char* line,size_t size,void* params) {
	readGCclassProcParams_t* vparams=(readGCclassProcParams_t*)params;
	index_t count=binIndexGetCount(vparams->refSeqIdx);
	char refseqItem[size];
	int classValue;
	if(sscanf(line,"%s\t%i",refseqItem,&classValue)) {
		index_t refseq=binIndexLookup2(vparams->refSeqIdx,refseqItem);
		if(refseq<count) {
			if(vparams->refseqPresenceVector[refseq]>0) {
				switch(classValue) {
					case 1:
						vparams->class1Count++;
						vparams->result++;
						break;
					case 2:
						vparams->class2Count++;
						vparams->result++;
						break;
					default:
						fprintf(stderr,"error format (should end with 1 or 2): %s",line);
				}
			}
		}
	}else{
		fprintf(stderr,"not found %s\n",refseqItem);
	}
	return 1;
}

static int readGCclass(int* class1Count,int* class2Count,const binIndex* refSeqIdx,const int* refseqPresenceVector,FILE* input) {
	readGCclassProcParams_t params=(readGCclassProcParams_t) {
		0,
		0,
		0,
		refSeqIdx,
		refseqPresenceVector
	};
	int res=readlineLoop(input,readGCclassProc,(void*)&params);
	*class1Count=params.class1Count;
	*class2Count=params.class2Count;
	return params.result;
}	

void showUsage(char* programName) {
	fprintf(stderr, "%s [refseq list] [motifs list] [gcClassFile list] [input] < [refseq to motif mapping] > [output]\n", programName);

}

int main(int argc, char *argv[]) {
	fprintf(stderr,"version 20150928\n");
	assert(!isatty(fileno(stdin)));
	//fprintf(stderr, "%i\n", argc);
	if(argc!=5) {
		showUsage(argv[0]);
		abort();
	}
	//assert(argc==4);
	binIndex* refSeqIdx = binIndexLoadRev(argv[1]);
	binIndex* motifIdx = binIndexLoadRev(argv[2]);
	char *gcClassFile = argv[3];
	char *inputFile = argv[4];

	int motifCount = binIndexGetCount(motifIdx);
	int refSeqCount = binIndexGetCount(refSeqIdx);

	fprintf(stderr, "%i motifs in index\n", motifCount);
	fprintf(stderr, "%i refseqs in index\n", refSeqCount);
	int refseqPresenceVector[refSeqCount];
	memset(refseqPresenceVector,0,sizeof(int)*refSeqCount);
	// read refseq input file
	FILE* input=fopen(inputFile,"r");
	assert(input!=NULL);
	int lines=fillPresenceVector(refseqPresenceVector,refSeqIdx,input);
	int count=0;
	for(int i=0;i<refSeqCount;i++) {
		if(refseqPresenceVector[i]==1) {
			count++;
		}
	}
	fclose(input);
	fprintf(stderr, "%i refseqs in the list ( %i unique ones )\n", lines,count);

	// read presences table
	// get count of presences in input
	int motifPresenceVector[motifCount];
	memset(motifPresenceVector,0,sizeof(int)*motifCount);

	lines=fillPresenceVectorMap(motifPresenceVector,refSeqIdx,motifIdx,refseqPresenceVector,stdin);
	fprintf(stderr, "%i pairs in mapping\n", lines);

	// read GC Class file
	int input_cl1_count=0;
	int input_cl2_count=0;

	// get ratio of presences in genome wide set 
	// and estimate p values of enrichment
	input=fopen(gcClassFile,"r");
	assert(input!=NULL);
	int debugPresenceVector[refSeqCount];
	for(int i=0;i<refSeqCount;i++) {debugPresenceVector[i] = 0;}
	lines=readGCclass(&input_cl1_count,&input_cl2_count,refSeqIdx,refseqPresenceVector,input);
	fclose(input);
	fprintf(stderr, "%i/%i gc class\n", lines);
	//assert(lines==100);
	binIndexFree(refSeqIdx);

	// output results
	fprintf(stdout, "%i\n%i\n", input_cl1_count, input_cl2_count);
	for(index_t i=0;i<motifCount;i++) {
		fprintf(stdout,"%s\t%i\n",binIndexReverseLookup(motifIdx,i) , motifPresenceVector[i]);
	}
	binIndexFree(motifIdx);
	return(EXIT_SUCCESS);
}

