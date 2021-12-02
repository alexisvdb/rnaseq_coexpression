#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "binIndex.h"
#include "binFormat.h"

const int cellTypeCount=24;

typedef struct {
	index_t index;
	score_t score;
} sortRecord;


int cmpFuncAsc(const void *a,const void *b) {
	sortRecord *va=(sortRecord*)a;
	sortRecord *vb=(sortRecord*)b;
	return (va->score - vb->score)>0.0?1:-1;;
}

int cmpFuncDsc(const void *a,const void *b) {
	sortRecord *va=(sortRecord*)a;
	sortRecord *vb=(sortRecord*)b;

	return (vb->score - va->score)>0.0?1:-1;;
}

int main(int argc, char *argv[]) {
	if(isatty(fileno(stdin))) {
		fputs("Usage : printTable2 indexFile colsort limit < inputFile\n\
\tindexFile\t: probes.txt\n\
\tcolsort\t: integer 1-24 , positive for ascending sort and negative for descending sort, 0 (zero) for no sorting\n\
\tlimit\t: integer\n",stderr);
		abort();
	}
	assert(!isatty(fileno(stdin)));
	// file_location indexfile colsort
	char * indexFile=argv[1];
	//char * inputFile=argv[2];
	int colsort=atoi(argv[2]);
	int limit=atoi(argv[3]);


	fprintf(stderr, "load index [%s]\n",indexFile);
	binIndex* index=binIndexLoadRev(indexFile);
	int count=binIndexGetCount(index);

	int size=cellTypeCount*count;
	score_t data[size];
	fprintf(stderr, "load data\n");
	assert(fread(&data[0],size*sizeof(score_t),1,stdin)==1);

	sortRecord sortData[count];
	for(int i=0;i<count;i++) {
		sortData[i].index=i;
		if(colsort!=0) {
			sortData[i].score=data[i*cellTypeCount+(abs(colsort)-1)];
		}
	}

	if(colsort!=0) {
		fprintf(stderr, "sort data\n");
		qsort(sortData,count,sizeof(sortRecord),(colsort>0)?cmpFuncAsc:cmpFuncDsc);
		fprintf(stderr, "done\n");
	}
	
	int lines=0;

	for(int i=0;i<count;i++) {
		if((limit!=0)&&(lines++>limit)) {
			break;
		}
		int current_index=sortData[i].index;
		if(sortData[i].score!=1.0) {
			printf("%s",binIndexReverseLookup(index,current_index));
			for(int j=0;j<cellTypeCount;j++) {
				printf("\t%.3f",data[current_index*cellTypeCount+j]);
			}
			printf("\n");
		}
	}
}

