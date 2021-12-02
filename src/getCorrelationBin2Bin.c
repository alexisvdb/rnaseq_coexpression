#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include "pearsonsCorrelation.h"
#include "expressionReader.h"
#include "binIndex.h"
#include "binFormat.h"

uint16_t readCols(FILE* input) {
	uint16_t result=0;
	assert(fread(&result,sizeof(uint16_t),1,input)==1);
	return(result);
}

typedef struct _linkNode linkNode_t;

struct _linkNode {
	linkNode_t* next;
	expressionRecord* record;
};

linkNode_t* createNode(expressionRecord* data) {
	linkNode_t *result=(linkNode_t*)malloc(sizeof(linkNode_t));
	result->next=NULL;
	result->record=data;
	return result;
}

linkNode_t* readInput(const int cols,FILE *input) {
	linkNode_t *result=NULL;
	linkNode_t *current=NULL;
	expressionRecord* record;

	while((record=readBinaryExpressionNew(cols,input))!=NULL) {
		linkNode_t *node=createNode(record);
		if(result==NULL) {
			result=node;
		}else{
			current->next=node;
		}
		current=node;
	}
	return result;
}

void processOutput(int a,int b,double score) {
	scoreRecord score_record={a,b,score};
	assert(fwrite(&score_record,sizeof(scoreRecord),1,stdout)==1);
}

void processPair(expressionRecord *a,expressionRecord *b,const int cols) {
	double correlation=getPearsonsCorrelation(cols,a->samples,b->samples);
	processOutput(a->nkey,b->nkey,correlation);
}

void processInput(linkNode_t* input,const int cols) {
	linkNode_t * current=input;
	while(current!=NULL) {
		linkNode_t *child=current->next;
		while(child!=NULL) {
			processPair(current->record,child->record,cols);
			child=child->next;
		}
		linkNode_t* next=current->next;
		freeExpressionRecord(current->record);
		free(current);
		current=next;
	}
}

int main(int argc, char *argv[]) {
	assert(!isatty(fileno(stdin)));
	assert(!isatty(fileno(stdout)));
	int cols=0; // number of header columns
	//expressionRecord* reference=NULL; // reference

	cols=readCols(stdin);
	assert(cols>0);
	linkNode_t *list=readInput(cols,stdin);
	processInput(list,cols);	
	return(EXIT_SUCCESS);
}

