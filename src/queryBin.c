#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "binFormat.h"
#include "binIndex.h"

int printRecord(scoreRecord record,void* extra) {
	binIndex* index=extra;
	printf("%s\t%s\t%f\n",binIndexReverseLookup(index,record.a),binIndexReverseLookup(index,record.b),record.score);
	return TRUE;
}

int main(int argc, char *argv[]) {
	//assert(!isatty(fileno(stdin)));
	const char* indexFile=argv[1];
	const char* inputFile=argv[2];
	
	const char* probe1=argv[3];
	
	binIndex* index=binIndexLoadRev(indexFile);
	FILE* input=fopen(inputFile,"r");
	assert(input!=NULL);
	index_t idx1=0;
	if(strcmp(probe1,"-")!=0) {
		idx1=binIndexLookup(index,probe1);
	}

	int count=binIndexGetCount(index);	
	walkBinFileQuery(input,idx1,count,printRecord,index);
	
	/*
	scoreRecord record;
	
	size_t size=sizeof(scoreRecord);
	long offset=0;
	for(int i=0;i<idx1;i++) {
		int local=idx1-1-i*2;
		long address=offset+local;	
		fseek(input,address*size,SEEK_SET);
		assert(fread(&record,sizeof(scoreRecord),1,input)==1);
		//printf("%i\t%i\t%i\t%f\n",idx1,arecord.a,arecord.b,arecord.score);
		printf("%s\t%s\t%f\n",binIndexReverseLookup(index,record.a),binIndexReverseLookup(index,record.b),record.score);
		offset+=count-i;
	}
	
	
	
	//printf("----\n");
	fseek(input,(offset-idx1)*size,SEEK_SET);


	for(int i=idx1+1;i<count;i++) {
		assert(fread(&record,sizeof(scoreRecord),1,input)==1);
		printf("%s\t%s\t%f\n",binIndexReverseLookup(index,record.a),binIndexReverseLookup(index,record.b),record.score);
	}
	
	*/
	
	/*
	while(!feof(input) && (fread(&arecord,sizeof(record),1,input)==1)) {
		
		if(
			((wildcard1==1) || (idx1==arecord.a) || (idx1==arecord.b)) &&
			((wildcard2==1) || (idx2==arecord.a) || (idx2==arecord.b))
		) {
			printf("%s\t%s\t%f\n",binIndexReverseLookup(index,arecord.a),binIndexReverseLookup(index,arecord.b),arecord.score);
		}
	}
	*/
	return(EXIT_SUCCESS);
}

