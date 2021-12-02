#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "binIndex.h"

static void readMap(index_t *map, const binIndex *probeIndex, const binIndex *symbolIndex, const char* probeSymbolMapFileName) {
	FILE* input=fopen(probeSymbolMapFileName,"rb");
	assert(input!=NULL);

	char* line=NULL; //getline
	size_t size=0;

	while(getline(&line,&size,input)!=-1) {
		//char* probeKey=NULL;
		//char* symbolKey=NULL;
		int split=-1;
		for(unsigned int i=0;i<size;i++) {
			if((line[i]=='\t')||(line[i]=='\n')) {
				if(line[i]=='\t') {
					split=i+1;
				}
				line[i]=0;
			}
		}
		if(split!=-1) {
			char* probeKey=strndup(line,size);
			char* symbolKey=strndup(line+split,size);
			//fprintf(stderr, "%s\t%s\n",probeKey,symbolKey );
			index_t probeIndexPos=binIndexLookup(probeIndex,probeKey);
			index_t symbolIndexPos=binIndexLookup(symbolIndex,symbolKey);

			free(probeKey);
			free(symbolKey);
			map[probeIndexPos]=symbolIndexPos;
		}else{
			fprintf(stderr,"invalid line [%s]\n",line);
		}

		size=0;
		free(line);
		line=NULL;
	}

	fclose(input);	
}

static index_t* readMap2(const binIndex *probeIndex,const binIndex *symbolIndex,const char* probeSymbolMapFileName,int *readsize) {
	FILE* input=fopen(probeSymbolMapFileName,"rb");
	assert(input!=NULL);
	char* line=NULL;
	size_t size=0;

	int capacity=10;
	int growth=10;
	int read=0;

	index_t *result=calloc(capacity*2,sizeof(index_t));
	assert(result!=NULL);
	while(getline(&line,&size,input)!=-1) {
		int split=-1;
		for(unsigned int i=0;i<size;i++) {
			if(line[i]=='\t') {
				if(split!=-1) {
					fprintf(stderr, "overwriting split %i => %i\n",split,i+1 );
				}
				split=i+1;
				if(split>16) {
					fprintf(stderr, "%s [%i]", line, line[i]);
				}
				line[i]=0;
			}
			if(line[i]=='\n') {
				line[i]=0;				
				break;
			}
		}
		if(split!=-1) {
			char* probeKey=strndup(line,size);
			char* symbolKey=strndup(line+split,size);
			//fprintf(stderr, "%s\t%s\t%i\t%i\n",probeKey,symbolKey,split,size );
			index_t probeIndexPos=binIndexLookup(probeIndex,probeKey);
			index_t symbolIndexPos=binIndexLookup(symbolIndex,symbolKey);

			free(probeKey);
			free(symbolKey);
			//map[probeIndexPos]=symbolIndexPos;
			result[read*2]=probeIndexPos;
			result[read*2+1]=symbolIndexPos;

			read++;
			if(read==capacity){
				capacity += growth;
				result=realloc(result,capacity*2*sizeof(index_t));
			}
			size=0;
		}else{
			fprintf(stderr,"invalid line [%s]\n",line);
		}

		free(line);
		line=NULL;
	}

	fprintf(stderr,"%i lines read\n",read);
	*readsize = read;
	result=realloc(result,read*2*sizeof(index_t));
	return result;
}

static size_t getIndex(const size_t x,const size_t y,const size_t n) {
	size_t k = ( n * ( n - 1 ) / 2 ) - ( ( n - x ) * ( n - x - 1 ) / 2 ) + y - x - 1;
	return k;
}

int main(int argc, char** argv) {
	const char* probeIndexFile=argv[1];
	const char* symbolIndexFile=argv[2];
	const char* probe2symbolFile=argv[3];

	binIndex *probeIndex=binIndexLoadRev(probeIndexFile);
	binIndex *symbolIndex=binIndexLoadRev(symbolIndexFile);

	const int probeCount=binIndexGetCount(probeIndex);
	fprintf(stderr,"%i probes\n",probeCount);
	const int symbolCount=binIndexGetCount(symbolIndex);
	fprintf(stderr,"%i symbols\n",symbolCount);

	int mapCount=0;
	index_t* map=readMap2(probeIndex,symbolIndex,probe2symbolFile,&mapCount);
	int probePresence[probeCount];
	for(int i=0;i<probeCount;i++) {
		probePresence[i]=-1;
	}
	for(int i=0;i<mapCount;i++) {
		if(probePresence[map[i*2]]==-1) {
			probePresence[map[i*2]]=i;
		}
	}
	int invalidProbe=0;
	int validProbe=0;
	for(int i=0;i<probeCount;i++) {
		if(probePresence[i]==-1) {
			invalidProbe++;
		}else{
			validProbe++;
		}
	}
	fprintf(stderr,"checking map\n");
	index_t t=0;
	for(int i=0;i<mapCount;i++) {
		assert(map[i*2]>=t);
		t=map[i*2];
	}

	int symbolPresence[symbolCount];
	for(int i=0;i<symbolCount;i++) {
		symbolPresence[i]=0;
	}
	int invalidSymbol=0;
	int validSymbol=0;
	for(int i=0;i<mapCount;i++) {
		symbolPresence[map[i*2+1]]=1;
	}
	for(int i=0;i<symbolCount;i++) {
		if(symbolPresence[i]==1) {
			validSymbol++;
		}else{
			invalidSymbol++;
		}
	}
	fprintf(stderr,"%i unmapped symbol - %i mapped symbol - %i total\n",invalidSymbol,validSymbol,symbolCount);
	fprintf(stderr,"%i unmapped probe - %i mapped probe - %i total\n",invalidProbe,validProbe,probeCount);
	fprintf(stderr,"done\n");
	size_t probeScoreCount=((size_t)probeCount)*((size_t)probeCount-1)/2;
	fprintf(stderr,"mem: %zu\texpected: %zu\tprobeCount: %i\n",probeScoreCount*sizeof(float),probeScoreCount,probeCount);
	float* probeScore=malloc(probeScoreCount*sizeof(float));//calloc(probeScoreCount,sizeof(float));
	assert(probeScore!=NULL);
	
	size_t read=fread(probeScore,sizeof(float),probeScoreCount,stdin);
	
	/*
	size_t read=0;
	float val=0.0;
	while(fread(&val,sizeof(float),1,stdin)==1){
		probeScore[read]=val;
		read++;
	}
	*/
	fprintf(stderr,"read: %zu\texpected: %zu\n",read,probeScoreCount);
	assert(read==probeScoreCount);
	size_t symbolScoreCount=(symbolCount)*(symbolCount-1)/2;
	float* symbolScore=calloc(symbolScoreCount,sizeof(float));
	for(int i=0;i<symbolScoreCount;i++) {
		symbolScore[i]=-100.0;
	}
	size_t maxS=0;
	size_t maxP=0;
	for(int i=0;i<symbolCount;i++) {
		symbolPresence[i]=0;
	}
	//size_t pos=0;
	fprintf(stderr,"remaping ... \n");
	for(int i=0;i<probeCount;i++) {
		if(probePresence[i]!=-1) {
			for(int j=i+1;j<probeCount;j++) {
				if(probePresence[j]!=-1) {
					size_t pos=getIndex(i,j,probeCount);
					maxP=(maxP>pos)?maxP:pos;
					if(pos%(probeScoreCount/100)==1) {
						fprintf(stderr,"%zu%\r",pos*100/probeScoreCount);
					}
					float value=probeScore[pos];
					for(int k=probePresence[i];k<mapCount;k++) {
						if(map[k*2]==i) {
							index_t s1=map[k*2+1];
							symbolPresence[s1]=1;
							for(int l=probePresence[j];l<mapCount;l++) {
								if(map[l*2]==j) {
									index_t s2=map[l*2+1];
									symbolPresence[s2]=1;
									if(s1!=s2) {
										size_t sIndex=0;
										if(s1<s2) {
											sIndex=getIndex(s1,s2,symbolCount);
										}else{
											sIndex=getIndex(s2,s1,symbolCount);
										}
										maxS=(maxS>sIndex)?maxS:sIndex;
										if(value>symbolScore[sIndex]) {
											symbolScore[sIndex]=value;
										}
									}
								}
								if(map[l*2]>j) {
									break;
								}
							}
						}
						if(map[k*2]>i) {
							break;
						}
					}
				}
			}
		}
	}

	invalidSymbol=0;
	validSymbol=0;
	for(int i=0;i<symbolCount;i++) {
		if(symbolPresence[i]==1) {
			validSymbol++;
		}else{
			fprintf(stderr,"uncovered : %s\n",binIndexReverseLookup(symbolIndex,i));
			invalidSymbol++;
		}
	}
	fprintf(stderr,"%i uncovered symbol - %i covered symbol - %i total\n",invalidSymbol,validSymbol,symbolCount);
	
	fprintf(stderr,"done \n %zu / %zu \n %zu / %zu \n",maxP,probeScoreCount,maxS,symbolScoreCount);
	fprintf(stderr,"checking\n");
	int valid=0;
	int invalid=0;
	for(size_t i=0;i<symbolScoreCount;i++) {
		float value=symbolScore[i];
		if((value>1.0)||(value<-1.0)) {
			//fprintf(stderr,"error @%zu == %f\n",i,value);
			invalid++;
		}else{
			valid++;
		}
	}
	fprintf(stderr,"%i (%f%) - %i (%f%) / %zu\n",valid,100.0*valid/symbolScoreCount,invalid,100.0*invalid/symbolScoreCount,symbolScoreCount);
	fprintf(stderr,"writing\n");
	size_t written=fwrite(symbolScore,sizeof(float),symbolScoreCount,stdout);
	assert(written==symbolScoreCount);
	/*
	   free(probeScore);
	   free(symbolScoreCount);
	   binIndexFree(probeIndex);
	   binIndexFree(symbolIndex);
	 */
	return EXIT_SUCCESS;
}
