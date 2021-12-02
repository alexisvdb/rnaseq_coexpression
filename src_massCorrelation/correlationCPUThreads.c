/**
@file
@author DINH Viet Huy Hubert <dinh@ifrec.osaka-u.ac.jp>
@version 1.0
@section DESCRIPTION

Calculate correlation values of vectors against each others in a list of vectors
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "correlation.h"
#include "threadPool.h"

typedef struct precomputeSumsArg_t {
	const float* input;
	size_t cols;
	size_t row;
	float* preSum;
	float* preS23;
} precomputeSumsArg_t;

static void precomputeSumsWorker(void* arg) {
	precomputeSumsArg_t* psArg=arg;
	float fCols=psArg->cols;
	size_t i=psArg->row;
	size_t base=i*psArg->cols;
	float sum=0.0;
	float sum2=0.0;
	for(size_t j=0;j<psArg->cols;j++) {
		float x=psArg->input[base+j];
		sum += x;
		sum2 += x*x;
	}
	psArg->preSum[i]=sum;
	psArg->preS23[i]=(sum2*fCols) - (sum*sum);
	free(arg);
}

typedef struct calculateCorrelationWorkerArg_t {
	const float* input;
	size_t cols;
	size_t rows;
	float* preSum;
	float* preS23;
	float* output;
	size_t i;
	//size_t j;
	//size_t index;
} calculateCorrelationWorkerArg_t;

static size_t getIndex(const size_t x,const size_t y,const size_t n) {
	size_t k = ( n * ( n - 1 ) / 2 ) - ( ( n - x ) * ( n - x - 1 ) / 2 ) + y - x - 1;
	return k;
}
	
static void calculateCorrelationWorker(void* arg) {
	calculateCorrelationWorkerArg_t* cCArg=arg;
	size_t i=cCArg->i;
	size_t baseX=cCArg->i*cCArg->cols;
	//size_t baseY=cCArg->j*cCArg->cols;
	for(size_t j=i+1;j<cCArg->rows;j++) {
		size_t baseY=j*cCArg->cols;


		float sumXY = 0.0;
		for(size_t k=0;k<cCArg->cols;k++) {
			float a=cCArg->input[baseX+k];
			float b=cCArg->input[baseY+k];
			sumXY += a*b;
		}

		float sumX = cCArg->preSum[cCArg->i];
		float sumY = cCArg->preSum[j];
		float fCols= cCArg->cols;
		float s1 = (sumXY*fCols) - (sumX*sumY);
		float s2= cCArg->preS23[cCArg->i];
		float s3 = cCArg->preS23[j];
		float s4 = sqrtf(s2*s3);
		float correlation = s1/s4;

		//cCArg->output[cCArg->index] = correlation;
		size_t index=getIndex(i,j,cCArg->rows);
		cCArg->output[index] = correlation;
	}
	free(arg);
}

extern float* getCorrelation(const float* vectors,const size_t cols,const size_t rows) {
	/*
	   Correlation calculation derived from the R project implementation
	*/
	
	// creating the thread pool (=queue of tasks)
	tp_threadpool_t *pool;
	pool = tp_init(128);

	float preSum[rows]; /* Precomputed vectors terms sum */
	float preS23[rows]; /* Precomputed S2 S3 values */
	//float fCols= cols;
	fprintf(stderr,"start\n");

	fprintf(stderr,"precomputing sums\n");

	for(size_t i=0;i<rows;i++) { /* precomputing sums */
		precomputeSumsArg_t *arg=malloc(sizeof(precomputeSumsArg_t));
		assert(arg!=NULL);
		*arg=(precomputeSumsArg_t){vectors,cols,i,preSum,preS23};
		tp_enqueue(pool,precomputeSumsWorker,(void*)arg);
	}
	tp_wait(pool);

	fprintf(stderr, "done\n");
	fprintf(stderr, "calculate correlation values\n");

	size_t totalSize=(rows)*(rows-1)/2; /* number of value in the flat vector storage of the triangle matrix*/
	fprintf(stderr,"%zu (%zu)\n",totalSize,totalSize*sizeof(float));
	float *output=malloc(sizeof(float)*totalSize);
	assert(output!=NULL);
	if(output==NULL) {
		fprintf(stderr,"not enough memory\n");
		abort();
		//return(NULL);
	}

	//size_t index=0; /* position set in the vector to write to */
	for(size_t i=0;i<rows;i++) {
		//for(size_t j=i+1;j<rows;j++) {
			calculateCorrelationWorkerArg_t *arg=malloc(sizeof(calculateCorrelationWorkerArg_t));
			assert(arg!=NULL);
			//*arg=(calculateCorrelationWorkerArg_t){vectors,cols,rows,preSum,preS23,output,i,j,index};
			*arg=(calculateCorrelationWorkerArg_t){vectors,cols,rows,preSum,preS23,output,i};
			tp_enqueue(pool,calculateCorrelationWorker,(void*)arg);
			/*
			if(index % (totalSize/100) == 1) {
				fprintf(stderr,"%zu%%\r",100*index/totalSize);
			}*/
			fprintf(stderr,"%zu\r",i);
			//index++;
		//}
	}
	tp_wait(pool);
	tp_destroy(pool);
	fprintf(stderr,"\nend\n");
	return(output);
}

