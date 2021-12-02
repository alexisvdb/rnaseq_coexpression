#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include "pearsonCorrelationBatch.h"

static void showUsage(const char* exeName) {
	fprintf(stderr,
			"Usage:\n"
			"\t%s < INPUT_EXPRESSION_FILE > OUTPUT_FILE\n\n"
		   ,exeName);
}

static double* readInput(int* out_cols,int* out_rows,FILE* input,char*** out_names) {
	int cols=0;
	int rows=0;

	char **names;

	const int increment=10000;

	char* line=NULL;
	size_t size=0;

	double* result=NULL;
	int capacity=0;
	const char* separator="\t";

	while(getline(&line,&size,input)!=-1) {
		if(cols==0){
			for(int i=0;i<size;i++) {
				if(line[i]==separator[0]) {
					cols++;
				}
			}
			assert(cols!=0);
			cols++;
			capacity=1;
			result=malloc(sizeof(double)*capacity*increment*cols);
			names=malloc(sizeof(char*)*capacity*increment);
			assert(result!=NULL);
			assert(names!=NULL);
		}else{
			char* name=strtok(line,separator);
			assert(name!=NULL);
			names[rows]=strdup(name);
			//fprintf(stderr, "%s\n", names[rows]);
			for(int i=0;i<cols;i++) {
				char* value=strtok(NULL,separator);
				assert(value!=NULL);
				result[rows*cols+i]=atof(value);
			}
			rows++;

			if(rows>(capacity*increment/100*90)) {
				capacity++;
				result=realloc(result,sizeof(double)*capacity*increment*cols);
				names=realloc(names,sizeof(char*)*capacity*increment);
				assert(result!=NULL);
				assert(names!=NULL);
			}
		}
		free(line);
		line=NULL;
		size=0;
	}
	*out_rows=rows;
	*out_cols=cols;

	result=realloc(result,sizeof(double)*rows*cols);
	names=realloc(names,sizeof(char*)*rows);
	assert(result!=NULL);
	assert(names!=NULL);

	*out_names=names;
	return result;
}
typedef struct callBackarg_t {
	char* output;
	char **names;

}callBackarg_t;

extern int probeBatchCallBack(int probeIndex,float* input,int count,void* extra) {
	callBackarg_t *arg=extra;
	//fprintf(stdout, "%s\n", arg->names[probeIndex]);
	char* path=alloca(2000);
	sprintf(path,"%s/%s.bin",arg->output,arg->names[probeIndex]);
	//fprintf(stdout, "%s\t=> %s\n", arg->names[probeIndex],path);
	FILE* output=fopen(path,"w");
	assert(output!=NULL);
	size_t res=fwrite(input,sizeof(float),count,output);
	//fprintf(stderr, "%i / %i\n",res,count );
	assert(res==count);
	fclose(output);
	//abort();
	return count;
	//fprintf(stderr, "%i\t", probeIndex);
}

int main(int argc,char** argv) {
	if(argc<2) {
		showUsage(argv[0]);
		abort();
	}
	int rows=0;
	int cols=0;
	char ***names=malloc(sizeof(void*));
	fprintf(stderr,"reading input ... ");
	double* input=readInput(&cols,&rows,stdin,names);
	assert(input!=NULL);
	int size=rows*cols*sizeof(double);
	char* sizeSuffix="b";
	if(size > (4*1024*1024)) {
		size /= 1024*1024;
		sizeSuffix = "mb";
	}else if(size > (4*1024)) {
		size /= 1024;
		sizeSuffix = "kb";
	}
	fprintf(stderr,"done ( %i cols, %i rows ,%i%s)\n",cols,rows,size,sizeSuffix);
	/*for(int i=0;i<rows;i++) {
		fprintf(stderr, "%s\n", (*names)[i]);
	}*/
	//abort();
	fprintf(stderr,"processing data ...");
	callBackarg_t arg={argv[1],*names};
	int count=batchPearsonCorrelationAll(input,cols,rows,probeBatchCallBack,&arg);
	fprintf(stderr,"done (%i values calculated) \n",count);
	return(EXIT_SUCCESS);
}
