#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

#include "utils.h"
#include <Rmath.h>

int getSymbolCount(FILE* input) {
	char* line=NULL;
	size_t size=0;
	if(getline(&line,&size,input)!=-1) {
		int result=atoi(line);
		free(line);
		return result;
	}else{
		fprintf(stderr, "expecting symbol count on first line\n");
		abort();
	}
	return -1;
}
typedef struct printLinePvalueProcParams_t {
	const int filteredSymbolCount;
	const int fullSymbolCount;
	int result;
} printLinePvalueProcParams_t;

static double getPValue(const int count,const int base,const int present,const int filteredSymbolCount,const int fullSymbolCount) {
	double x=count-1;
	double n=filteredSymbolCount - present;
	double p=1.0L*base/(1.0L*fullSymbolCount);
	//fprintf(stderr, "%lg\t%lg\t%lg\n", x,n,p);
	return pbinom(x,n,p,0,0);
}

static int printLinePvalueProc(char* line,size_t size,void* params) {
	printLinePvalueProcParams_t* vparams=(printLinePvalueProcParams_t*)params;
	char symbol[size];

	int count=0;
	int positive=0;
	int negative=0;

	int present=0;

	int baseCount=0;
	int basePositive=0;
	int baseNegative=0;

	if(sscanf(line,"%s\t%i\t%i\t%i\t%i\t%i\t%i\t%i",symbol,&count,&positive,&negative,&present,&baseCount,&basePositive,&baseNegative)==8) {
		double pValueCount=0.0;
		double pValuePositive=0.0;
		double pValueNegative=0.0;

		//pValueCount=pbinom(count-1,vparams->filteredSymbolCount - present, 1.0L*baseCount/vparams->fullSymbolCount,0,0);
		//pValuePositive=pbinom(positive-1,vparams->filteredSymbolCount - present, 1.0L*basePositive/vparams->fullSymbolCount,0,0);
		//pValueNegative=pbinom(negative-1,vparams->filteredSymbolCount - present, 1.0L*baseNegative*1.0L/vparams->fullSymbolCount,0,0);
		pValueCount=getPValue(count,baseCount,present,vparams->filteredSymbolCount,vparams->fullSymbolCount);
		pValuePositive=getPValue(positive,basePositive,present,vparams->filteredSymbolCount,vparams->fullSymbolCount);
		pValueNegative=getPValue(negative,baseNegative,present,vparams->filteredSymbolCount,vparams->fullSymbolCount);
		fprintf(stdout, "%s\t%i\t%i\t%i\t%lg\t%lg\t%lg\n", 
			symbol,
			count,
			positive,
			negative,
			pValueCount,
			pValuePositive,
			pValueNegative
			);
		vparams->result++;
	}else{
		fprintf(stderr, "invalid line format [%s]\n", line);
	}
	return 1;
}

int main(int argc, char *argv[]) {
	assert(!isatty(fileno(stdin)));
	printLinePvalueProcParams_t params=(printLinePvalueProcParams_t) {
		getSymbolCount(stdin), //filteredSymbolCount
		atoi(argv[1]), //fullSymbolCount
		0
	};
	int res=readlineLoop(stdin,printLinePvalueProc,(void*)&params);
	fprintf(stderr, "%i record processed (%i line reads)\n",params.result,res);
}
