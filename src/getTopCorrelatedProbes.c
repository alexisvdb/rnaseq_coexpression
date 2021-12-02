#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

float* readScores(const char*,size_t*);

typedef struct chain chain;

#define BLOCK_SIZE	1024
struct chain
{
	float block[BLOCK_SIZE];
	size_t read;
	size_t offset;
	chain* next;
};

float* readScores_(const char* path,size_t* sizeRead) {
	chain* blocks=NULL;
	FILE* input=fopen(path,"rb");
	if(input==NULL) {
		fprintf(stderr, "error reading [%s]\n", path);
		abort();
	}
	//assert(input!=NULL);
	size_t offset=0;

	while(true) {
		chain* element=malloc(sizeof(chain));
		assert(element!=NULL);

		element->next=blocks;
		element->offset=offset;
		element->read=fread(element->block,sizeof(float),(size_t)BLOCK_SIZE,input);

		if(element->read==0) {
			free(element);
			break;
		}

		offset += element->read;

		blocks = element;

		if(element->read!=BLOCK_SIZE) {
			break;
		}
	}
	fclose(input);

	if(blocks==NULL) {
		*sizeRead=0;
		return NULL;
	}

	
	// calculate data size
	chain* current=blocks;
	size_t size=0;
	while(current!=NULL) {
		size += current->read;
		current = current->next;
	}
	float* result=malloc(size*sizeof(float));
	assert(result!=NULL);

	// copy results
	current=blocks;
	while(current!=NULL) {
		memcpy(&result[current->offset],current->block,current->read);
		current = current->next;
	}

	//free data
	while(blocks!=NULL) {
		chain* element=blocks;
		blocks=blocks->next;
		free(element);
	}

	fprintf(stderr, "%s\t%zu\n", path,size);
	*sizeRead=size;
	return result;
}

float* readScores(const char* path,size_t* sizeRead) {
	FILE* input=fopen(path,"rb");
	if(input==NULL) {
		fprintf(stderr, "error reading [%s]\n", path);
		abort();
	}
	size_t capacity=1024;
	size_t read=0;
	float* result=malloc(capacity*sizeof(float));
	assert(result!=NULL);
	size_t r=0;

	while((r=fread(&result[read],sizeof(float),1024,stdin))!=0) {
		fprintf(stderr, "%zu\n", read);
		if(r<1024) {
			read+=r;
			break;
		}else{
			read+=r;
			capacity+=1024;
			result=realloc(result,capacity*sizeof(float));
			assert(result!=NULL);
		}
	}
	fclose(input);
	result=realloc(result,read*sizeof(float));
	assert(result!=NULL);
	*sizeRead=read;
	return result;
}


enum modeValue {
	modeMin,
	modeMax,
	modeAbs
};
typedef enum modeValue modeValue;

int main(int argc,char* argv[]) {
	char* modeParam=argv[1];
	assert(argc>=2);
	modeValue mode=modeMax;
	if(strcmp(modeParam,"min")==0) {
		mode=modeMin;
	}
	if(strcmp(modeParam,"abs")==0) {
		mode=modeAbs;
	}

	char* line=NULL; //getline
	size_t size=0;

	size_t scoreCount=0;
	float* result=NULL;
	int* origin=NULL;
	int originIndex=0;
	while(getline(&line,&size,stdin)>0) {
		assert(line!=NULL);
		for(int i=size-1;i>=0;i--) {
			if(line[i]=='\n') {
				line[i]=0;
				break;
			}
		}
		size_t read=0;
		float* input=readScores(line,&read);
		assert(input!=NULL);

		if(result==NULL) {
			result=input;
			scoreCount=read;
			origin=calloc(scoreCount,sizeof(int));
		}else{
			assert(scoreCount==read);
			for(size_t i=0;i<scoreCount;i++) {
				if(
					( ( mode == modeMax ) && ( input[i] > result[i] ) ) ||
					( ( mode == modeMin ) && ( input[i] < result[i] ) ) ||
					( ( mode == modeAbs ) && ( fabs(input[i]) > fabs(result[i]) ) ) 
					) {
					result[i]=input[i];
					origin[i]=originIndex;
				}
			}
			free(input);
		}
		originIndex++;
		size=0;
		free(line);
	}
	assert(result!=NULL);
	assert(origin!=NULL);
	fprintf(stderr, "%zu\n", scoreCount);
	for(size_t i=0;i<scoreCount;i++) {
		fprintf(stdout, "%0.6f\t%i\n", result[i],origin[i]);
	}
	free(result);
	return EXIT_SUCCESS;
}
