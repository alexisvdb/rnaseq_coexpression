/**
@file
@author DINH Viet Huy Hubert <dinh@ifrec.osaka-u.ac.jp>
@version 1.0
@section DESCRIPTION
matrix approach to calculating correlation values
**/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "correlation.h"

/**
Perform a cross product of the matrix with its transposed self
As defined : \f$ B = A  \times A^\intercal\f$

@param input [in] input matrix
@param cols [in] number of columns in the input matrix
@param rows [in] number of rows in the input matrix
@return result of the cross product
**/
static float* tcrossprod(const float* input, const size_t cols, const size_t rows) {
	size_t outputSize=(rows)*(rows-1)/2;

	float* output=calloc(outputSize,sizeof(float));
	assert(output!=NULL);

	size_t position=0;
	for(size_t i=0;i<rows;i++) {
		size_t basei=i*cols;
		for(size_t j=i+1;j<rows;j++) {
			size_t basej=j*cols;
			float value=0.0;
			for(size_t k=0;k<cols;k++) {
				value += input[basei+k]*input[basej+k];
			}

			assert(position<outputSize);
			if(position % (outputSize/100) == 1) {
				fprintf(stderr,"%zu%%\r",100*position/outputSize);
			}
			
			output[position++] = value;
		}
	}
	return output;
}

/**
Normalize the input matrix row by row on GPU
Substract each row by its mean value
\f$A=A-\bar A\f$
Divide each row by its std deviation 

@param input [in][out] matrix to perform operations on
@param cols [in] columns count
@param rows [in] rows count
\f$A=A / \sqrt{\sum_{rows} {A^2} }\f$
**/
static void normalizeRows(float* input, const size_t cols, const size_t rows) {
	for(size_t i=0;i<rows;i++) {
		size_t base=i*cols;
		float value=0.0;//mean
		for(size_t j=0;j<cols;j++) {
			value += input[base+j];
		}
		value /= cols;

		for(size_t j=0;j<cols;j++) {
			input[base+j] -= value;
		}

		value=0.0;
		for(size_t j=0;j<cols;j++) {
			float x=input[base+j];
			value += x*x;
		}
		value=sqrtf(value);

		for(size_t j=0;j<cols;j++) {
			input[base+j] /= value;
		}
	}
}

extern float* getCorrelation(const float* vectors,const size_t cols,const size_t rows) {
	size_t inputSize=cols*rows*sizeof(float);
	float* input=malloc(inputSize);
	assert(input!=NULL);
	memcpy(input,vectors,inputSize);
	normalizeRows(input,cols,rows);
	float* output=tcrossprod(input,cols,rows);
	return output;
}

