#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "binFormat.h"
#include "binIndex.h"

int readGcClassFile(int* gcClass,int* class1Count,int* class2Count,FILE* input,const binIndex *refSeqIdx) {
	int result=0;
	size_t size=0;
	char* line=NULL;

	int refSeqCount = binIndexGetCount(refSeqIdx);
	*class1Count=0;
	*class2Count=0;

	while(getline(&line,&size,input)!=-1) {
		
		for(int i=0;i<size;i++) {
			if(line[i]=='\t') {
				line[i]=0;
				index_t key=binIndexLookup2(refSeqIdx,line);
				if(key < refSeqCount) {
					int value=(line[i+1]=='2')?2:1;
					if(line[i+1]=='2') {
						(*class2Count)++;
					}else{
						(*class1Count)++;
					}

					gcClass[key]=value;
				}else{
					fprintf(stderr, "key not found [%s]\n", line);
				}
				result++;
				break;

			}
		}
		
		

		// Reinit vars for next getline
		free(line);
		line=NULL;
		size=0;
	}

	return result;
}

int readMap(int* output,FILE* input,const int* gcClass,const binIndex *refSeqIdx,const binIndex *motifIdx) {
	int result=0;
	size_t size=0;
	char* line=NULL;

	int motifCount = binIndexGetCount(motifIdx);
	int refSeqCount = binIndexGetCount(refSeqIdx);

	memset(output,0,sizeof(int)*motifCount*2);

	while(getline(&line,&size,input)!=-1) {
		int refseq=0;
		int motif=0;

		char* right=NULL;
		for(int i=0;i<size;i++) {
			char c=line[i];
			if(right==NULL) {
				if(c=='\t') {
					right=line+i+1;
					line[i]=0;
					refseq=binIndexLookup2(refSeqIdx,line);
				}
			}else{
				if(c=='\n') {
					line[i]=0;
					motif=binIndexLookup2(motifIdx,right);
					if((refseq<refSeqCount)&&(motif<motifCount)) {
						output[motif+motifCount*(gcClass[refseq]==2)]++;
					}
				}
			}
		}

		result++;

		// Reinit vars for next getline
		free(line);
		line=NULL;
		size=0;
	}
	return result;
}

int main(int argc, char *argv[]) {
	assert(!isatty(fileno(stdin)));

	const binIndex* refSeqIdx = binIndexLoadRev(argv[1]);
	const binIndex* motifIdx = binIndexLoadRev(argv[2]);

	index_t motifCount = binIndexGetCount(motifIdx);
	index_t refSeqCount = binIndexGetCount(refSeqIdx);

	int gcClass[refSeqCount];
	int class1Count=0;
	int class2Count=0;
	FILE* input;
	int lines;

	input=fopen(argv[3],"r");
	lines=readGcClassFile(gcClass,&class1Count,&class2Count,input,refSeqIdx);
	fclose(input);
	fprintf(stderr, "%i class read (%i expected)\n", lines,refSeqCount);
	assert(lines==refSeqCount);

	int counts[motifCount*2];


	lines=readMap(counts,stdin,gcClass,refSeqIdx,motifIdx);

	for(index_t i=0;i<motifCount;i++) {
		fprintf(stdout, "%s\t%1.18llf\t%1.18llf\n",binIndexReverseLookup(motifIdx,i),1.0L*counts[i]/class1Count,1.0L*counts[i+motifCount]/class2Count);
	}

	return(EXIT_SUCCESS);
}
