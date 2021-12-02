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

extern float* getCorrelation(const float* vectors,const size_t cols,const size_t rows) {

	/*
	   Correlation calculation derived from the R project implementation
	*/
	float preSum[rows]; /* Precomputed vectors terms sum */
	float preS23[rows]; /* Precomputed S2 S3 values */
	float fCols= cols;
	fprintf(stderr,"start\n");

	fprintf(stderr,"precomputing sums\n");
	for(size_t i=0;i<rows;i++) { /* precomputing sums */
		size_t base=i*cols;
		float sum=0.0;
		float sum2=0.0;
		for(size_t j=0;j<cols;j++) {
			float x=vectors[base+j];
			sum += x;
			sum2 += x*x;
		}
		preSum[i]=sum;
		preS23[i]=(sum2*fCols) - (sum*sum);
	}

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

	size_t index=0; /* position set in the vector to write to */
	for(size_t i=0;i<rows;i++) {
		size_t baseX=i*cols;
		float sumX = preSum[i];
		float s2=preS23[i];
		for(size_t j=i+1;j<rows;j++) {
			size_t baseY=j*cols;

			float sumXY = 0.0;
			for(size_t k=0;k<cols;k++) {
				float a=vectors[baseX+k];
				float b=vectors[baseY+k];
				sumXY += a*b;
			}
			float sumY = preSum[j];
			float s3 = preS23[j];
			float s1 = (sumXY*fCols) - (sumX*sumY);
			float s4 = sqrtf(s2*s3);
			float correlation = s1/s4;
			assert(index<totalSize);
			if(index % (totalSize/100) == 1) {
				fprintf(stderr,"%zu%%\r",100*index/totalSize);
			}
			output[index++] = correlation;
		}
	}
	fprintf(stderr,"\nend\n");
	return(output);
}

