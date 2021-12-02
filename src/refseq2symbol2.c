#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "binFormat.h"
#include "binIndex.h"
#include "utils.h"

int printMapped(char* line,size_t size,void* param) {
	binIndex* refseqIndex=(binIndex*)param;
	index_t refseqsCount=binIndexGetCount(refseqIndex);
	char symbol[size];
	char refseq[size];
	if(sscanf(line,"%s\t%s",symbol,refseq)==2) {
		index_t refseqId=binIndexLookup2(refseqIndex,refseq);
		// for each refseq in the input ( stored in index )
		if(refseqId<refseqsCount) {
			// print the corresponding symbol
			fprintf(stdout,"%s\n",symbol);
		}
	}
	return 1;
}

int main(int argc, char *argv[]) {
	assert(!isatty(fileno(stdin)));
	// store input refseqs list in an index
	binIndex* refseqIndex=binIndexLoadStream(stdin);
	int count=binIndexGetCount(refseqIndex);
	fprintf(stderr, "%i refseqs in input\n", count);

	// opening symbol <-> refseq mapping file
	FILE* input=fopen(argv[1],"r");
	readlineLoop(input,printMapped,(void*)refseqIndex);
	fclose(input);

	//free memory
	binIndexFree(refseqIndex);
	return(EXIT_SUCCESS); 
}