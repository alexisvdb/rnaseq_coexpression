#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include "expressionReader.h"

const char* separator="\t";

expressionRecord* expressionRecordCreate(const int cols) {
	expressionRecord* result=malloc(sizeof(expressionRecord));
	assert(result!=NULL);
	result->key=NULL;
	size_t size=sizeof(double)*cols;
	result->samples=(double*)malloc(size);
	return(result);
}

void readExpressionRecord2(expressionRecord *target,char* line,const int cols) {
	char* key=strtok(line,separator);
	if(target->key!=NULL) {
		if(strlen(target->key)>=strlen(key)) {
			strncpy(target->key,key,strlen(target->key));
		}else{
			free(target->key);
			target->key=strdup(key);
		}
	}else{
		target->key=strdup(key);
	}
	for(int i=0;i<cols;i++) {
		char* next=strtok(NULL,separator);
		assert(next!=NULL);
		target->samples[i]=atof(next);
	}
}

int readBinaryExpressionNoAbort(expressionRecord *target,const int cols,FILE* input) {
	uint16_t id;
	if(fread(&id,sizeof(uint16_t),1,input)!=1) {
		return 0;
	}
	if(fread(target->samples,sizeof(double),cols,input)!=cols) {
		return 0;
	}
	target->nkey=id;
	return 1;
}

expressionRecord* readBinaryExpressionNew(const int cols,FILE* input) {
	expressionRecord* result=expressionRecordCreate(cols);
	if(readBinaryExpressionNoAbort(result,cols,input)==1) {
		return result;
	}
	freeExpressionRecord(result);
	return NULL;
}
void readBinaryExpression(expressionRecord *target,const int cols,FILE* input) {
	uint16_t id;
	assert(fread(&id,sizeof(uint16_t),1,input)==1);
	target->nkey=id;
	assert(fread(target->samples,sizeof(double),cols,input)==cols);
}

void writeBinaryExpression(expressionRecord *source,const int cols,FILE* output) {
	uint16_t id=source->nkey;
	assert(fwrite(&id,sizeof(uint16_t),1,output)==1);
	assert(fwrite(source->samples,sizeof(double),cols,output)==cols);
}

double* readArray(const int cols) {
	double* result=(double*)calloc(cols,sizeof(double));
	assert(result!=NULL);

	for(int c=0;c<cols;c++) {
		char* next=strtok(NULL,separator);
		assert(next!=NULL);
		//printf("%s\n",next);
		result[c]=atof(next);
	}
	return result;
}

expressionRecord* readExpressionRecord(char* line,const int cols) {
	expressionRecord* result=malloc(sizeof(expressionRecord));
	//puts(line);
	char* key=strtok(line,separator);
	result->key=strdup(key);
	assert(result->key!=NULL);
	result->samples=readArray(cols);
	return(result);
}

void freeExpressionRecord(expressionRecord* item) {
	free(item->key);
	free(item->samples);
	free(item);
}

double* arrayDup(const double* source,const int cols) {
	size_t size=sizeof(double)*cols;
	double* result=(double*)malloc(size);
	memcpy(result,source,size);
	return result;
}

expressionRecord* expressionRecordDup(const expressionRecord* item,const int cols) {
	expressionRecord* result=malloc(sizeof(expressionRecord));
	assert(result!=NULL);
	result->key=strdup(item->key);
	result->samples=arrayDup(item->samples,cols);
	assert(result->key!=NULL);
	result->nkey=item->nkey;
	return result;
}
