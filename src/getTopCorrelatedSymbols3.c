#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>


// Data structure storing the top symbols
typedef struct symbolScoreInfo_t {
	index_t symbol; // 
	index_t probe;
	index_t sourceProbe;
	score_t score;
} symbolScoreInfo_t;

void ntrim(char* input,int length) {
	for(int i=length-1;i>0;i--) {
		if(input[i]=='\n') {
			input[i]=0;			
		}
	}
}

index_t* readMap(const binIndex *probeIndex, const binIndex *symbolIndex,const char* probeSymbolMapFileName) {

	int read=0;
	int capacity=16;

	index_t* result=malloc(capacity*2*sizeof(index_t));
	assert(result!=NULL);

	FILE* input=fopen(probeSymbolMapFileName,"rb");
	assert(input!=NULL);

	char* line=NULL; //getline
	size_t size=0;

	while(getline(&line,&size,input)!=-1) {
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

		result[read*2]=probeKey;
		result[read*2+1]=symbolKey;
		read++;
		if(read==capacity) {
			capacity += 16;
			result=realloc(result,capacity*2*sizeof(index_t));
			assert(result!=NULL);
		}
		size=0;
		free(line);
	}

	fclose(input);

	result=realloc(result,read*2*sizeof(index_t));
	assert(result!=NULL);
	fprintf(stderr,"%i lines read\n",read);
	return result;
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

void readScoresFile(symbolScoreInfo_t *scores,const index_t *map,const char* filename,const int probeCount,const int minMax, const index_t probe) {
	score_t read[probeCount];
	FILE* input=fopen(filename,"rb");
	assert(input!=NULL);
	fread(&read[0],sizeof(score_t),probeCount,input);
	fclose(input);
	for(int i=0;i<probeCount;i++) {
		int mapped=map[i];
		if(mapped!=0xFFFF) {
			//if(isnan(scores[mapped].score)) {
			if(scores[mapped].probe > probeCount) {
				scores[mapped].score=read[i];
				scores[mapped].probe=i;
				scores[mapped].sourceProbe=probe;
			}else{
				if(minMax?(read[i]>scores[mapped].score):(read[i]<scores[mapped].score)) {
					scores[mapped].score=read[i];
					scores[mapped].probe=i;
					scores[mapped].sourceProbe=probe;
				}
			}
		}else{
		}
	}
}

void readScores(symbolScoreInfo_t *scores,const index_t *map,FILE* input,const int probeCount,const int minMax,binIndex *probeIndex) {
	char* line=NULL; //getline
	size_t size=0;
	while(getline(&line,&size,input)!=0) {
		//ntrim(line,size);
		char* probe=line;
		char* file=NULL;
		for(int i=0;i<size;i++) {
			if(line[i]=='\t') {
				line[i]=0;
				file=line+i+1;
				//fprintf(stderr,"filename : %s\n",file);
			}
			if((line[i]=='\n')||(line[i]=='\r')) {
				//fprintf(stderr,"triming endline\n");
				line[i]=0;
			}
		}
		if(file==NULL) {
			break;
		}
		assert(file!=NULL);
		/*for(int i=0;i<size;i++) {
			if(file[i]==0) {break;}
			if(file[i]=='\n') {file[i]=0;}
		}*/
		file=strndup(file,size);
		probe=strndup(probe,size);
		fprintf(stderr, "loading scores for %s : %s ... " , probe ,  file);
		readScoresFile(&scores[0],map,file,probeCount,minMax,binIndexLookup(probeIndex,probe));
		fprintf(stderr, "done\n" );
		size=0;
		free(probe);
		free(file);
		free(line);		
	}
}

int comparatorAsc(const void* pa,const void* pb) {
	symbolScoreInfo_t *a=(symbolScoreInfo_t*) pa;
	symbolScoreInfo_t *b=(symbolScoreInfo_t*) pb;

	//if(a->score==b->score) {return 0;}
	//return (a->score < b->score)? -1 :1;
	if(isnan(a->score)||isnan(b->score)) {
		return 1;
	} 
	int result=(a->score > b->score) - (a->score < b->score);
	//fprintf(stderr, "%0.6f\t%0.6f\t%i\n",a->score,b->score,result );
	return result;
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
	const int probeCount=binIndexGetCount(probeIndex);
	fprintf(stderr, "done %i\n" ,probeCount);

	fprintf(stderr, "loading symbol index ... " );
	binIndex *symbolIndex=binIndexLoadRev(symbolIndexFileName);
	const int symbolCount=binIndexGetCount(symbolIndex);
	fprintf(stderr, "done %i\n" ,symbolCount);
	

	index_t probeSymbolMap[probeCount];
	for(int i=0;i<probeCount;i++) {
		probeSymbolMap[i]=0xFFFF;
	}

	fprintf(stderr, "loading probe <=> symbol mapping ... " );
	readMap(&probeSymbolMap[0],probeIndex,symbolIndex,probeSymbolMapFileName);
	fprintf(stderr, "done\n" );

	//float scores[symbolCount];
	symbolScoreInfo_t scores[symbolCount]; // storage for top symbol
	for(int i=0;i<symbolCount;i++) {
		scores[i].symbol=i;
		scores[i].probe=probeCount+1;
		scores[i].score=10.0;
	}
	fprintf(stderr, "loading scores ... \n" );
	readScores(scores,probeSymbolMap,stdin,probeCount,minMax,probeIndex);
	fprintf(stderr, "done\n" );

	qsort(scores,symbolCount,sizeof(symbolScoreInfo_t),(minMax)?comparatorDsc:comparatorAsc);
	//if(row==0) {row=symbolCount;};
	int limit=0;
	for(int i=0;i<symbolCount;i++) {
		if(!isnan(scores[i].score)&&(scores[i].score!=10.0)) {
			if((row!=0)&&(limit++>=row)) {
				break;
			}
			fprintf(stdout,"%s\t%s\t%s\t%0.6f\t%i\n",
				binIndexReverseLookup(symbolIndex,scores[i].symbol),
				binIndexReverseLookup(probeIndex,scores[i].probe),
				binIndexReverseLookup(probeIndex,scores[i].sourceProbe),
				scores[i].score,
				i
			);
		}
	}
	return(EXIT_SUCCESS);
}

