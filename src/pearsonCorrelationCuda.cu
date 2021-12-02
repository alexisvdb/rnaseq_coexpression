#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "pearsonsCorrelation.h"

__global__ void getPearsonsCorrelation(const int* l,double* result,const double* x,const double* y) {
	double sum_x=0.0L;
	double sum_y=0.0L;
	double sum_xy=0.0L;
	double sum_x2=0.0L;
	double sum_y2=0.0L;
	int length= *l;
	for(int i=0;i< length;i++) {
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
	*result=s1/s4;
}

extern "C" double getPearsonsCorrelation(const int length,const double* x,const double* y) {
	int* dev_length;
	size_t size=length*sizeof(double);
	size_t size1=sizeof(double);
	double correlation;

	double *dev_a,*dev_b,*dev_result;

	cudaMalloc(&dev_a,size);
	cudaMalloc(&dev_b,size);
	cudaMalloc(&dev_result,size1);
	cudaMalloc(&dev_length,sizeof(int));

	cudaMemcpy( dev_a, x, size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, y, size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_length, &length, sizeof(int), cudaMemcpyHostToDevice );

	getPearsonsCorrelation<<<1,1>>>(dev_length,dev_result,dev_a,dev_b);

	cudaMemcpy( &correlation, &dev_result, size, cudaMemcpyDeviceToHost );

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_length);
	cudaFree(dev_result);

	return correlation;
}

