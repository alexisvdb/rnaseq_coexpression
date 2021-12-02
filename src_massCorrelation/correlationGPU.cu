#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
	__device__ size_t getIndex(const size_t x,const size_t y,const size_t n) {
		size_t k = ( n * ( n - 1 ) / 2 ) - ( ( n - x ) * ( n - x - 1 ) / 2 ) + y - x - 1;
		return k;
	}
	__device__ void getPos(const size_t k,const size_t n,size_t* x, size_t* y) {
		size_t i = n - 2 - floor(sqrtf(-8*k + 4*n*(n-1)-7)/2.0 - 0.5);
		size_t j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
		*x = i;
		*y = j;
	}

	__global__ void makePresumSingle(const float* input,const size_t cols,const size_t rows,float* sum,float* s23) {
		size_t row = blockIdx.x*blockDim.x+threadIdx.x;
		if(row<rows) {
			size_t base=row*cols;
			float csum=0.0;
			float csum2=0.0;
			for (size_t i=0;i<cols;i++) {
				float x=input[base+i];
				csum += x;
				csum2 += x*x;
			}
			float cs23=(csum2*cols) - (csum*csum);
			sum[row] = csum;
			s23[row] = cs23;
		}
	}

	__global__ void makeCorrelation2(const float* input,const size_t cols,const size_t rows,float* sum,float* s23,float* output) {
		size_t i = blockIdx.x*blockDim.x+threadIdx.x;
		size_t j = blockIdx.y*blockDim.y+threadIdx.y;

		extern __shared__ float cache[];
		if(
			(i<rows) &&
			(j<rows) &&
			(j>i)
			) {
			size_t basei=i*cols;
			size_t basej=j*cols;
			if(threadIdx.x==0) {
				for(size_t k=0;k<cols;k++) {
					cache[k+(threadIdx.y+blockDim.x)*cols]=input[basej+k];
				}
			}
			if(threadIdx.y==0) {
				for(size_t k=0;k<cols;k++) {
					cache[k+(threadIdx.x)*cols]=input[basei+k];
				}
			}
			/*
			for(size_t k=0;k<cols;k++) {
				cache[k+threadIdx.x*cols]=input[basei+k];
				cache[k+(threadIdx.y+8)*cols]=input[basej+k];
			}
			*/
			__syncthreads();
			float sumXY = 0.0;
			for(size_t k=0;k<cols;k++) {
				float a=cache[k+threadIdx.x*cols];
				float b=cache[k+(threadIdx.y+blockDim.x)*cols];
				sumXY += a*b;
			}
			float sumX = sum[i];
			float sumY = sum[j];
			float s1= (sumXY*cols) - (sumX*sumY) ;
			float s2=s23[i];
			float s3=s23[j];
			float s4= sqrtf(s2*s3);
			float correlation=s1/s4;
			size_t k=getIndex(i,j,rows);
			output[k]=correlation;
		}
	}

	__global__ void makeCorrelation(const float* input,const size_t cols,const size_t rows,float* sum,float* s23,float* output) {
		size_t i = blockIdx.x*blockDim.x+threadIdx.x;
		size_t j = blockIdx.y*blockDim.y+threadIdx.y;
		if(
			(i<rows) &&
			(j<rows) &&
			(j>i)
			) {
			size_t basei=i*cols;
			size_t basej=j*cols;
			float sumXY = 0.0;
			for(size_t k=0;k<cols;k++) {
				float a=input[basei+k];
				float b=input[basej+k];
				sumXY += a*b;
			}
			float sumX = sum[i];
			float sumY = sum[j];
			float s1= (sumXY*cols) - (sumX*sumY) ;
			float s2=s23[i];
			float s3=s23[j];
			float s4= sqrtf(s2*s3);
			float correlation=s1/s4;
			size_t k=getIndex(i,j,rows);
			output[k]=correlation;
		}
	}

	static float* getDevInput(const float* input,const size_t cols,const size_t rows) {
		float* output;
		size_t inputSize=rows*cols*sizeof(float);
		cudaMalloc(&output,inputSize);
		cudaMemcpy(output,input,inputSize,cudaMemcpyHostToDevice);
		return output;
	}

	static float* getDevOutput(const size_t rows) {
		float* output;
		size_t outputSize=(rows)*(rows-1)/2*sizeof(float);
		cudaMalloc(&output,outputSize);
		cudaMemset(output,0,outputSize);
		return output;
	}

	static float* getOutput(float* input,const size_t rows) {
		size_t outputSize=(rows)*(rows-1)/2*sizeof(float);
		float* output=(float*)malloc(outputSize);
		assert(output!=NULL);
		cudaMemcpy(output,input,outputSize,cudaMemcpyDeviceToHost);
		return output;
	}

	extern float* getCorrelation(const float* vectors,const size_t cols,const size_t rows) {
		float* dev_sum;
		cudaMalloc(&dev_sum,rows*sizeof(float));
		float* dev_s23;
		cudaMalloc(&dev_s23,rows*sizeof(float));

		float* dev_input=getDevInput(vectors,cols,rows);
		

		fprintf(stderr,"start\n");

		fprintf(stderr,"precomputing sums\n");
		makePresumSingle<<<rows/1024+1,1024>>>(dev_input,cols,rows,dev_sum,dev_s23);
		fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError()));
		float* dev_output=getDevOutput(rows);
		cudaDeviceSynchronize();
		fprintf(stderr, "done\n");

		
		if(1){
			size_t blockDim=32;
			size_t blockCount=rows/blockDim+1;
			makeCorrelation<<<dim3(blockCount,blockCount),dim3(blockDim,blockDim)>>>(dev_input,cols,rows,dev_sum,dev_s23,dev_output);
			fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError()));
		}else{
			int blocksize=32;
			size_t shared=cols*2*blocksize*sizeof(float);
			while(shared>48*1024) {
				blocksize--;
				shared=cols*2*blocksize*sizeof(float);
			}
			fprintf(stderr," %zu %zu %08x\n",shared,blocksize,shared);
			//cudaFuncSetCacheConfig("makeCorrelation2", cudaFuncCachePreferShared);
			cudaDeviceSetCacheConfig( cudaFuncCachePreferShared);
			fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError()));
			makeCorrelation2<<<dim3(rows/blocksize+1,rows/blocksize+1),dim3(blocksize,blocksize),shared>>>(dev_input,cols,rows,dev_sum,dev_s23,dev_output);
			fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError()));
		}
		cudaDeviceSynchronize();
		fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError()));
		fprintf(stderr,"end\n");

		float* output=getOutput(dev_output,rows);
		cudaDeviceSynchronize();
		cudaFree(dev_output);
		cudaFree(dev_input);
		return(output);
	}

}

