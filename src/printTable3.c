#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "binIndex.h"
#include "binFormat.h"
#include "utils/SingleLinkedList.h"

typedef struct scoreIndexpair {
	int index;
	float score;
} scoreIndexpair;

static int sortCmp(void* pa,void* pb) {
	scoreIndexpair* a=(scoreIndexpair*)pa;
	scoreIndexpair* b=(scoreIndexpair*)pb;

	if(a->score == b->score) {
		return 0;
	}
	return (a->score > b->score)?1:-1;
}

static int sortDesc(void* pa,void* pb) {
	return -1*sortCmp(pa,pb);
}

static int sortAsc(void* pa,void* pb) {
	return sortCmp(pa,pb);
}

int main(int argc,char* argv[]) {
	assert(argc==5);
	assert(!isatty(fileno(stdin)));
	char* probeIndexFile=argv[1];
	int sortCol=atoi(argv[2]);
	int sortOrder=atoi(argv[3]);
	int rowCount=atoi(argv[4]);
	//char* colList=argv[4];

	//fprintf(stderr, "load index [%s]\n",probeIndexFile);
	binIndex* index=binIndexLoadRev(probeIndexFile);
	int count=binIndexGetCount(index);

	singleLinkedList* colsList=singleLinkedList_init();
	char* line=NULL;
	size_t size=0;
	ssize_t read;
	while((read = getline(&line,&size,stdin))!= -1) {
		for(int i=0;i<read;i++) {
			if(line[i]=='\n') {
				line[i]=0;
			}
		}
		singleLinkedList_push(colsList,strdup(line));
		free(line);
		line=NULL;
		size=0;
	}

	int colCount=singleLinkedList_count(colsList);
	char* colsFiles[colCount];
	for(int i=colCount-1;i>=0;i--) {
		colsFiles[i]=singleLinkedList_pop(colsList);
		//fprintf(stderr, "%0i\t%s\n", i, colsFiles[i]);
	}
	assert(colCount>0);

	char* sortColFilename=colsFiles[sortCol];
	scoreIndexpair order[count];
	//fprintf(stderr, "%s\n", sortColFilename);
	FILE* inputStream=fopen(sortColFilename,"rb");
	assert(inputStream!=NULL);
	for(int i=0;i<count;i++) {
		float score=0.0;
		read=fread(&score,sizeof(float),1,inputStream);
		assert(read==1);
		order[i].index=i;
		order[i].score=score;
	}
	fclose(inputStream);
	if(sortOrder!=0) {
		qsort(order,count,sizeof(scoreIndexpair),(sortOrder<0)?sortDesc:sortAsc);
	}
	if(rowCount==0) {
		rowCount=count;
	}
	if(rowCount>count) {
		rowCount=count;
	}
	float result[colCount*rowCount];
	for(int i=0;i<rowCount;i++) {
		result[i*colCount+sortCol]=order[i].score;
	}

	for(int j=0;j<colCount;j++) {
		if(j!=sortCol) {
			char* colFilename=colsFiles[j];
			inputStream=fopen(colFilename,"rb");
			assert(inputStream!=NULL);
			for(int i=0;i<rowCount;i++) {
				float score=0.0;

				fseek(inputStream,order[i].index*sizeof(float),SEEK_SET);
				read=fread(&score,sizeof(float),1,inputStream);
				assert(read==1);
				result[i*colCount+j]=score;
			}
		}
	}
	for(int i=0;i<rowCount;i++) {
		if(result[i*colCount]!=1.0) {
			printf("%s",binIndexReverseLookup(index,order[i].index));
			for(int j=0;j<colCount;j++) {
				//if(j>0) {
					printf("\t");
				//}
				printf("%0.6f", result[i*colCount+j]);
			}
			printf("\n");
		}
	}
	/*
	for(int i=0;i<colCount*rowCount;i++) {
		if((i!=0)&&(i%colCount==0)) {
			printf("\n");
		}else{
			if(i!=0) {
				printf(",");
			}
		}
		printf("%0.6f",result[i]);
	}
	*/
	return EXIT_SUCCESS;
}
