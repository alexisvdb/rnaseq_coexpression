/**
@file
@author DINH Viet Huy Hubert <dinh@ifrec.osaka-u.ac.jp>
@version 1.0
@section DESCRIPTION
GPU implementation of the matric approach to calculating correlation values
**/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "correlation.h"

extern "C" {

	/**
	Get The linear index if an upper triangle matrix

	@param x [in] Horizontal position in the matrix ( row number )
	@param y [in] Vertical position in the matrix ( col number )
	@param matrix dimension

	@return linear position in a flat representation (!D) of the triangle
	**/
	__device__ inline int getIndex(const size_t x,const size_t y,const size_t n) {
		long long int k = ( n * ( n - 1 ) / 2 ) - ( ( n - x ) * ( n - x - 1 ) / 2 ) + y - x - 1;
		return k;
	}


	/**
	Perform a cross product of the matrix with its transposed self
	for one element of the ouput matrix as run on GPU
	As defined : \f$ B = A  \times A^\intercal\f$

	@param input [in] input matrix
	@param cols [in] number of columns in the input matrix
	@param rows [in] number of rows in the input matrix
	@param output [out] output matrix
	**/
	__global__ void tcrossprodSingle(const float* input,const size_t cols,const size_t rows,float* output) {
		size_t i = blockIdx.x*blockDim.x+threadIdx.x; // current row
		size_t j = blockIdx.y*blockDim.y+threadIdx.y; // current column
		if((i<rows)&&(j<rows)&&(i<j)) { // check if the row and col are within the triangle
			size_t basei=i*cols; // base input offset
			size_t basej=j*cols; // base input offset
			size_t k=getIndex(i,j,rows); // output offset
		
			float value=0.0;
			for(size_t c=0;c<cols;c++) {
				value += input[basei+c]*input[basej+c];
			}
			output[k] = value;
		}
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
	__global__ void normalizeRowsSingle(float* input,const size_t cols,const size_t rows) {
		const size_t row=blockIdx.x*blockDim.x+ threadIdx.x;
		if(row<rows) {
			const size_t base=row*cols;
			float value=0.0;
			for(size_t i=0;i<cols;i++) {
				value += input[base+i];
			}
			value /= cols;
		
			for(size_t i=0;i<cols;i++) {
				input[base+i] -= value;
			}
			value=0.0;
			for(size_t i=0;i<cols;i++) {
				float x= input[base+i];
				value += x*x;
			}
			value=sqrtf(value);
			for(size_t i=0;i<cols;i++) {
				input[base+i] /= value;
			}
		}
	}

	extern float* getCorrelation(const float* vectors,const size_t cols,const size_t rows) {
		size_t inputSize=cols*rows*sizeof(float);
		float* input=(float*)malloc(inputSize);
		float* output=NULL;

		fprintf(stderr,"normalize input\n");

			float *dev_input;
			
			cudaMalloc(&dev_input,inputSize);
			cudaMemcpy(dev_input,vectors,inputSize,cudaMemcpyHostToDevice);

			normalizeRowsSingle<<<rows/1024+1,1024>>>(dev_input,cols,rows);
			//fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError()));
			float *dev_output;
			size_t outputSize=(rows)*(rows-1)/2*sizeof(float);
			cudaMalloc(&dev_output,outputSize);
			//cudaMemset(dev_output,0,outputSize);
			cudaDeviceSynchronize();

			
			size_t blockDim=32;
			size_t blockCount=rows/blockDim+1;
			//dim3 matCount(rows/32+1,rows/32+1);
			//dim3 matDim(32,32);
			tcrossprodSingle<<<dim3(blockCount,blockCount),dim3(blockDim,blockDim)>>>(dev_input,cols,rows,dev_output);
			//fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError()));
			fprintf(stderr,"waiting for kernels to finish\n");
			cudaDeviceSynchronize();

			fprintf(stderr,"copying memory back to host\n");

			output=(float*)malloc(outputSize);
			assert(output!=NULL);
			cudaMemcpy( output, dev_output, outputSize, cudaMemcpyDeviceToHost );
			cudaDeviceSynchronize();
			cudaFree(dev_output);
			cudaFree(dev_input);

		return output;
	}
}
