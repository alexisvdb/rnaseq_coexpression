#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <unistd.h>
#include "expressionReader.h"
#include "binIndex.h"

int countSeparator(const char* input,char needle) {
	int i=0;
	int result=0;
	while(input[i]!='\0') {
		if(input[i]==needle) {
			result++;
		}
		i++;
	}
	return result;
}

int main(int argc,char** argv) {
	assert(!isatty(fileno(stdin)));
	assert(!isatty(fileno(stdout)));
	
	binIndex* index=binIndexLoad(argv[1]);
	size_t size=0;
	char* line=NULL;
	size_t count=0;
	expressionRecord *record=NULL;
	while(getline(&line,&size,stdin)!=-1) {
		assert(size>0);
		if(count==0) {
			count=countSeparator(line,'\t')+1;
			uint16_t cols=count;
			assert(fwrite(&cols,sizeof(uint16_t),1,stdout));
			record=expressionRecordCreate(count);
		}else{
			readExpressionRecord2(record,line,count);
			record->nkey=binIndexLookup(index,record->key);
			writeBinaryExpression(record,count,stdout);
			/*
			uint16_t key=record->nkey;
			assert(fwrite(&key,sizeof(uint16_t),1,stdout)==1);
			assert(fwrite(record->samples,sizeof(double)*count,1,stdout)==1);
			*/
		}
		free(line);
		size=0;
		line=NULL;
	}
	free(record);
}
