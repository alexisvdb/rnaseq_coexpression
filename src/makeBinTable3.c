#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>
#include <time.h>

#include "binIndex.h"
#include "binFormat.h"

#ifndef MAX_PATH
#define MAX_PATH 1024
#endif

int readRecord(scoreRecord record,void* extra) {
	void** params=(void**)extra;
	score_t *data=(score_t*)params[0];
	int* index=(int*)params[1];
	index_t* query=(index_t*)params[2];
	int* rowLength = (int*) params[3];
	index_t row=(record.a== *query)?record.b:record.a;
	
	data[row * *rowLength + *index]=record.score;
	
	return TRUE;
}

char* getFilename(const char* outputDir,const char* name) {
	char *ext=".bin";
	int combinedLength=strnlen(outputDir,MAX_PATH)+strnlen(name,MAX_PATH)+strnlen(ext,MAX_PATH)+5;
	char* path=(char*)malloc(combinedLength);
	snprintf(path,combinedLength,"%s/%s%s",outputDir,name,ext);
	return path;
}

void writeData(const score_t* data,const int length,const char* outputDir,const char* probeName) {
	char *path=getFilename(outputDir,probeName);
	assert(path!=NULL);
	FILE* output=fopen(path,"w");
	assert(output!=NULL);
	assert(fwrite(data,sizeof(score_t),length,output)==length);
	fclose(output);
	free(path);
}

int fileExists(const char* fname) {
	return(access( fname, F_OK ) != -1);
}
//Args

// index outputdir input1..n

int main(int argc, char *argv[]) {
	assert(argc>2);
	const char* indexFile=argv[1];
	const char* outputDir=argv[2];
	const int inputsOffset=3;
	const int inputsCount=argc-inputsOffset;
	
	binIndex* index=binIndexLoadRev(indexFile);
	
	//FILE *inputs[inputsCount];
	int inputs[inputsCount];
	for(int i=0;i<inputsCount;i++) {
		inputs[i]=-1;
		char* input=argv[i+inputsOffset];
		/*inputs[i]=fopen(input,"r");
		assert(inputs[i]!=NULL);
		if(inputs[i]==NULL) {
			fprintf(stderr,"error opening [ %s ]",input);
			abort();
		}
		*/
		inputs[i]=open64(input,O_RDONLY);
		assert(inputs[i]!=-1);
		if(inputs[i]==-1) {
			fprintf(stderr,"error opening [ %s ]",input);
			abort();
		}
	}
	int probesCount=binIndexGetCount(index);
	int dataLength=probesCount*inputsCount;
	score_t data[dataLength];

	for(int i=0;i<probesCount;i++) {
		const char* probeName=binIndexReverseLookup(index,i);
		fprintf(stderr,"%s ( %i/ %i ) ...",probeName,i,probesCount);
		char* outputFilename=getFilename(outputDir,probeName);
		if(!fileExists(outputFilename)) {
			fprintf(stderr," init ... ");
			for(int j=0;j<dataLength;j++) {
				data[j]=1.0;
			}
			fprintf(stderr," reading ... ");
			for(int j=0;j<inputsCount;j++) {
				int input=inputs[j];
				assert(input!=-1);
				//fseek(input,0,SEEK_SET);
				//printf("%i %i\n",i,j);
				void* params[4]={&data[0],&j,&i,&inputsCount};
				walkBinFileQueryFd(input,i,probesCount,readRecord,&params);
			}
			fprintf(stderr," writing ... ");
			writeData(&data[0],dataLength,outputDir,probeName);
			fprintf(stderr," done\n");
		}else{
			fprintf(stderr,"skipped\n");
		}
	}

	
	for(int i=0;i<inputsCount;i++) {
		assert(inputs[i]!=-1);
		close(inputs[i]);
	}
	return(EXIT_SUCCESS);
}

