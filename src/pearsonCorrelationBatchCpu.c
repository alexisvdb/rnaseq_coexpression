#include <stdio.h>
#include <assert.h>
#include "pearsonCorrelationBatch.h"
#include "math.h"

//typedef int probeBatchCallBack_t(int probeIndex,double* input,int count,void* extra);
void getPearsonCorrelationProbe(float* output,double* input,int inputRowNumber,int rowCount,int colCount) {
	int index= blockIdx.x*blockDim.x + threadIdx.x;
	int outputBase=index*rowCount;
	int aBaseIndex=inputRowNumber*colCount;

	//double* aVector=input+aBaseIndex;
	
	double sum_x=0.0L;
	double sum_x2=0.0L;
	for(int i=0;i<colCount;i++) {
		double a=input[aBaseIndex+i];
		sum_x += a;
		sum_x2 += a*a;
	}
	double partial_s2 = sum_x2*colCount;
	for(int i=0;i<rowCount;i++) {
		int bBaseIndex=i*colCount;
		//double* bVector=input+bBaseIndex;
		output[outputBase+i]=0.0L;
		//double sum_x=0.0L;
		//double sum_x2=0.0L;
		double sum_y=0.0L;
		double sum_xy = 0.0L;
		double sum_y2 = 0.0L;

		for(int j=0;j<colCount;j++) {
			double a=input[aBaseIndex+j];
			double b=input[bBaseIndex+j];

			//sum_x += a;
			//sum_x2 += a*a;

			sum_y += b;
			sum_y2 += b*b;
			sum_xy += a*b;
		}
		/*
		double s1= (sum_xy*colCount) - (sum_x*sum_y);
		double s2= (sum_x2*colCount) - (sum_x*sum_x);
		double s3= (sum_y2*colCount) - (sum_y*sum_y);
		double s4= sqrt(s2*s3);
		*/
		double sum_x_sum_y = sum_x*sum_y;

		double s1 = (sum_xy*colCount) - sum_x_sum_y;
		double s2 = partial_s2 - sum_x_sum_y;
		double s3 = (sum_y2*colCount) - sum_x_sum_y;
		double s4 = sqrt(s2*s3);
		output[outputBase+i]=s1/s4;
	}
}

extern int batchPearsonCorrelationAll(double* input,int cols,int rows,probeBatchCallBack_t callback,void* extra) {
	int inputSize=cols*rows*sizeof(double);

	const int stride=512;
	float output[rows*stride];
	cudaMalloc(&dev_output,rows*sizeof(float)*stride);
	for(int i=0;i<rows;i+=stride) {

		
		//fprintf(stderr, "copy result buffer\n" );
		
		for(int j=0;j<stride;j++) {
			int index=i+j;
			getPearsonCorrelationProbe(dev_output,dev_input,index,rows,cols);
			if(index<rows) {
				/*float outputref[rows];
				getPearsonsCorrelationCpu(outputref,input,i+j,rows,cols);
				for(int k=0;k<rows;k++) {
					//fprintf(stdout, "%lf\t%lf\n", outputref[k],output[k]);
					assert(abs(outputref[k]-output[k])<1e-6);
				}*/
				callback(index,output+j*rows,rows,extra);
			}
		}
	}
	cudaFree(dev_output);
	cudaFree(dev_input);
	return rows*rows;
}

