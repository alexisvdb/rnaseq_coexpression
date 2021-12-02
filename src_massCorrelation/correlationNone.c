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

	size_t totalSize=(rows)*(rows-1)/2; /* number of value in the flat vector storage of the triangle matrix*/
	fprintf(stderr,"%zu (%zu)\n",totalSize,totalSize*sizeof(float));
	float *output=malloc(sizeof(float)*totalSize);
	assert(output!=NULL);
	if(output==NULL) {
		fprintf(stderr,"not enough memory\n");
		abort();
		//return(NULL);
	}
	return(output);
}

