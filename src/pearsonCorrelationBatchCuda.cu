#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "pearsonCorrelationBatch.h"

#ifdef __cplusplus
extern "C" {
#endif

	//typedef int probeBatchCallBack_t(int probeIndex,double* input,int count,void* extra);
	__global__ void getPearsonCorrelationProbe(float* output,double* input,int inputRowNumber,int rowCount,int colCount) {
		int index= blockIdx.x*blockDim.x + threadIdx.x;
		int outputBase=index*rowCount;
		int aBaseIndex=inputRowNumber*colCount;

		//double* aVector=input+aBaseIndex;
		
		double sum_x=0.0L;
		double sum_x2=0.0L;
		for(int i=0;i<colCount;i++) {
			double a=input[aBaseIndex+i];
			//double a=aVector[i];
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
			//double s2 = (sum_x2*colCount) - sum_x_sum_y;
			double s3 = (sum_y2*colCount) - sum_x_sum_y;
			double s4 = sqrt(s2*s3);
			output[outputBase+i]=s1/s4;
		}
	}
	/*
	static double getPearsonsCorrelation(const int length,const double* x,const double* y) {

		double sum_x=0.0L;
		double sum_y=0.0L;
		double sum_xy=0.0L;
		double sum_x2=0.0L;
		double sum_y2=0.0L;

		for(int i=0;i<length;i++) {
			double a=x[i];
			double b=y[i];
			sum_x += a;
			sum_y += b;
			sum_xy += a*b;
			sum_x2 += a*a;
			sum_y2 += b*b;

		}
		
		double s1= (sum_xy*length) - (sum_x*sum_y);
		double s2= (sum_x2*length) - (sum_x*sum_x);
		double s3= (sum_y2*length) - (sum_y*sum_y);
		double s4= sqrt(s2*s3);
		double result=s1/s4;

		return(result);
	}

	static void getPearsonsCorrelationCpu(float* output,double* input,int inputRowNumber,int rowCount,int colCount) {
		int outputBase=0;
		int aBaseIndex=inputRowNumber*colCount;

		//double* aVector=input+aBaseIndex;

		for(int i=0;i<rowCount;i++) {
			int bBaseIndex=i*colCount;
			//double* bVector=input+bBaseIndex;

			output[outputBase+i]=0.0L;

			double sum_x=0.0L;
			double sum_x2=0.0L;
			double sum_y=0.0L;
			double sum_xy = 0.0L;
			double sum_y2 = 0.0L;

			for(int j=0;j<colCount;j++) {
				double a=input[j+aBaseIndex];
				double b=input[j+bBaseIndex];
				//fprintf(stdout, "%i\t%lf\t%lf\n",j, a,b);
				sum_x += a;
				sum_x2 += a*a;

				sum_y += b;
				sum_y2 += b*b;
				sum_xy += a*b;
			}
			double s1= (sum_xy*colCount) - (sum_x*sum_y);
			double s2= (sum_x2*colCount) - (sum_x*sum_x);
			double s3= (sum_y2*colCount) - (sum_y*sum_y);
			double s4= sqrt(s2*s3);

			output[outputBase+i]=s1/s4;
			//fprintf(stdout,"----- : %lf\n",output[outputBase+i]);
		}		
	}
	*/
	extern int batchPearsonCorrelationAll(double* input,int cols,int rows,probeBatchCallBack_t callback,void* extra) {
		int inputSize=cols*rows*sizeof(double);

		double *dev_input;
		cudaMalloc(&dev_input,inputSize);
		cudaMemcpy(dev_input,input,inputSize,cudaMemcpyHostToDevice);

		//fprintf(stderr, "%i\n", rows);
		const int stride=512;
		float output[rows*stride];
		float *dev_output;
		cudaMalloc(&dev_output,rows*sizeof(float)*stride);
		for(int i=0;i<rows;i+=stride) {
			//fprintf(stderr, "calling kernel\n" );
			getPearsonCorrelationProbe<<<16,32>>>(dev_output,dev_input,i,rows,cols);
			//fprintf(stderr, "copy result buffer\n" );
			cudaMemcpy( output, dev_output, rows*sizeof(float)*stride, cudaMemcpyDeviceToHost );
			for(int j=0;j<stride;j++) {
				int index=i+j;
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

#ifdef __cplusplus
}
#endif

