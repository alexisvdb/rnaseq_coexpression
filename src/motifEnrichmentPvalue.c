#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

#include <Rmath.h>
#include "utils.h"

int getCount(FILE* input) {
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

typedef struct printLinePvalueProcParam_t{
	const double c1Count;
	const double c2Count;
	const double correctionFactor;
	const double threshold;
	int result;
}printLinePvalueProcParam_t;

static int doubleCompare(const void* a,const void* b) {
	const double* da=(double*)a;
	const double* db=(double*)b;
	return (*da > *db) - (*da < *db);
}

static double max(double x,double y) {
	return (x>y)?x:y;
}

static double logxpy(double x,double y) {
	return max(x,y)+log1p(exp(-1.0L*fabs(x-y)));
}

static double logxmy(double x,double y) {
	return x+log1p(-1.0L*exp(y-x));
}
static double toDouble(int value) {
	return 1.0L*value;
}
static double getPvalue(const int value,const double correctionFactor,const double c1Count,const double c2Count,const double c1,const double c2) {
	double pvalue=1.0L;
	// only calculate p values if the number of presences in the input is higher than 0
	if(value>0) {
		double p[value];
		int count=0;
		for(int i=0;i<value;i++) {
			// changes of getting 0, 1, 2, etc hits in part 2
			double p2=dbinom(toDouble(i),c2Count,c2,true);
			// changes of getting < x, ... <2, <1 hits in part 1
			double p1=pbinom(toDouble(value-1-i),c1Count,c1,true,true);

			// remove impossible cases (in the case there are more observed hits than there are sequences in one of the 2 classes
			if(!(isinf(p1)||(isinf(p2)))) {
				// sum these (equivalend to product, but in log values)
				p[count++]=p1+p2;
			}
		}

		// sort results
		qsort(p,count,sizeof(double),doubleCompare);

		
		double last=p[0];
		for(int i=1;i<count;i++) {
			last=logxpy(last,p[i]);
		}

		if(last>=0) {
			pvalue=0;
		}else{
			pvalue= exp(logxmy(log(1),last));
		}
		if(pvalue<0) {
			// correct those that are negative to 0, this should not be necessary
			pvalue=0;
		}
	}else{
		// if the number of presences is 0, then the p value is automatically 1
		pvalue = 1.0L;
	}
	pvalue *= correctionFactor;
	
	return pvalue;
}

static int printLinePvalueProc(char* line,size_t size,void* params) {
	printLinePvalueProcParam_t* vparams=(printLinePvalueProcParam_t*)params;
	char motif[size];
	char tfs[size];
	int count=0;
	double c1=0.0L;
	double c2=0.0L;

	if(sscanf(line,"%s\t%s\t%i\t%lf\t%lf",motif,tfs,&count,&c1,&c2)==5) {
		double expected = c1*vparams->c1Count + c2*vparams->c2Count;
		double fold=(count/expected);
		if(isnan(fold)) {
			fold=0.0L;
		}
		//fprintf(stdout, "%s\t%s\t%i\t%lf\t%lf\t%lf\t%lf\t%lf\n",motif,tfs,count,c1,vparams->c1Count,c2,vparams->c2Count,expected);
		double pvalue=getPvalue(count,vparams->correctionFactor,vparams->c1Count,vparams->c2Count,c1,c2);
		//if(pvalue<vparams->threshold) {
			fprintf(stdout, "%s\t%s\t%i\t%.1lf\t%.2lf\t%.2g\n", 
				motif,
				tfs,
				count, //signif_presences
				expected, // signif_expected
				fold, // fold_enriched
				pvalue
				);
		//}
		//abort();
	}else{
		fprintf(stderr, "bad format[%s]\n", line);
	}
	return 1;
}

int main(int argc, char *argv[]) {
	//assert(argc==3);
	int correctionFactor=atoi(argv[1]);
	//double threshold=atof(argv[2]);
	//fprintf(stderr, "threshold : %f\n",threshold );
	int c1Count=getCount(stdin);
	int c2Count=getCount(stdin);
	printLinePvalueProcParam_t params=(printLinePvalueProcParam_t) {
		toDouble(c1Count),
		toDouble(c2Count),
		toDouble(correctionFactor),
		//threshold,
		0
	};
	int res=readlineLoop(stdin,printLinePvalueProc,(void*)&params);
}
