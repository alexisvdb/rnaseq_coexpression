#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "binIndex.h"

typedef struct symbolScoreInfo_t {
	index_t symbol;
	index_t probe;
	score_t score;
} symbolScoreInfo_t;

void ntrim(char* input,int length) {
	for(int i=length-1;i>0;i--) {
		if(input[i]=='\n') {
			input[i]=0;			
		}
	}
}

void readMap(index_t *map, const binIndex *probeIndex, const binIndex *symbolIndex, const char* probeSymbolMapFileName) {
	FILE* input=fopen(probeSymbolMapFileName,"rb");
	assert(input!=NULL);

	char* line=NULL; //getline
	size_t size=0;

	while(getline(&line,&size,input)!=-1) {
		//char* probeKey=NULL;
		//char* symbolKey=NULL;
		int split=0;
		for(unsigned int i=0;i<size;i++) {
			if((line[i]=='\t')||(line[i]=='\n')) {
				if(line[i]=='\t') {
					split=i+1;
				}
				line[i]=0;
			}
		}
		char* probeKey=strndup(line,size);
		char* symbolKey=strndup(line+split,size);
		//fprintf(stderr, "%s\t%s\n",probeKey,symbolKey );
		index_t probeIndexPos=binIndexLookup(probeIndex,probeKey);
		index_t symbolIndexPos=binIndexLookup(symbolIndex,symbolKey);

		free(probeKey);
		free(symbolKey);

		map[probeIndexPos]=symbolIndexPos;

		size=0;
		free(line);
	}

	fclose(input);	
}

void readScoresFile(symbolScoreInfo_t *scores,const index_t *map,const char* filename,const int probeCount,const int minMax) {
	score_t read[probeCount];
	FILE* input=fopen(filename,"rb");
	assert(input!=NULL);
	fread(&read[0],sizeof(score_t),probeCount,input);
	fclose(input);
	for(int i=0;i<probeCount;i++) {
		int mapped=map[i];
		if(mapped!=0xFFFF) {
			if(isnan(scores[mapped].score)) {
				scores[mapped].score=read[i];
				scores[mapped].probe=i;
			}else{
				if(minMax?(read[i]>scores[mapped].score):(read[i]<scores[mapped].score)) {
					scores[mapped].score=read[i];
					scores[mapped].probe=i;
				}
			}
		}else{
		}
	}
}

void readScores(symbolScoreInfo_t *scores,const index_t *map,FILE* input,const int probeCount,const int minMax) {
	char* line=NULL; //getline
	size_t size=0;
	while(getline(&line,&size,input)!=-1) {
		ntrim(line,size);
		fprintf(stderr, "loading scores for %s ... " , line);
		readScoresFile(&scores[0],map,line,probeCount,minMax);
		fprintf(stderr, "done\n" );
		size=0;
		free(line);		
	}
}

int comparatorAsc(const void* pa,const void* pb) {
	symbolScoreInfo_t *a=(symbolScoreInfo_t*) pa;
	symbolScoreInfo_t *b=(symbolScoreInfo_t*) pb;

	if(a->score==b->score) {return 0;}
	return (a->score < b->score)? -1 :1;
}

int comparatorDsc(const void* pa,const void* pb) {
	return -1*comparatorAsc(pa,pb);
}

int main(int argc,char** argv) {
	assert(!isatty(fileno(stdin)));
	const char *probeIndexFileName=argv[1];
	const char *symbolIndexFileName=argv[2];
	const char *probeSymbolMapFileName=argv[3];
	const int  row=atoi(argv[4]);
	const int minMax=(strcmp(argv[5],"max")==0);
	fprintf(stderr, "loading probe index ... " );
	binIndex *probeIndex=binIndexLoadRev(probeIndexFileName);
	fprintf(stderr, "done\n" );
	fprintf(stderr, "loading symbol index ... " );
	binIndex *symbolIndex=binIndexLoadRev(symbolIndexFileName);
	fprintf(stderr, "done\n" );
	
	const int probeCount=binIndexGetCount(probeIndex);
	const int symbolCount=binIndexGetCount(symbolIndex);

	index_t probeSymbolMap[probeCount];
	for(int i=0;i<probeCount;i++) {
		probeSymbolMap[i]=0xFFFF;
	}

	fprintf(stderr, "loading probe <=> symbol mapping ... " );
	readMap(&probeSymbolMap[0],probeIndex,symbolIndex,probeSymbolMapFileName);
	fprintf(stderr, "done\n" );

	//float scores[symbolCount];
	symbolScoreInfo_t scores[symbolCount];
	for(int i=0;i<symbolCount;i++) {
		scores[i].symbol=i;
		scores[i].probe=probeCount+1;
		scores[i].score=NAN;
	}
	fprintf(stderr, "loading scores ... \n" );
	readScores(&scores[0],probeSymbolMap,stdin,probeCount,minMax);
	fprintf(stderr, "done\n" );

	qsort(&scores[0],symbolCount,sizeof(symbolScoreInfo_t),(minMax)?comparatorDsc:comparatorAsc);
	//if(row==0) {row=symbolCount;};
	int limit=0;
	for(int i=0;i<symbolCount;i++) {
		if(!isnan(scores[i].score)&&(scores[i].score!=1.0)) {
			if((row!=0)&&(limit++>=row)) {
				break;
			}
			fprintf(stdout,"%s\t%s\t%0.6f\n",
				binIndexReverseLookup(symbolIndex,scores[i].symbol),
				binIndexReverseLookup(probeIndex,scores[i].probe),
				scores[i].score
			);
		}
	}
	return(EXIT_SUCCESS);
}
