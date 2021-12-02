#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <unistd.h>
#include <errno.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

static size_t getIndex(const size_t x,const size_t y,const size_t n) {
	if(y<x) return getIndex(y,x,n);
	size_t k = ( n * ( n - 1 ) / 2 ) - ( ( n - x ) * ( n - x - 1 ) / 2 ) + y - x - 1;
	return k;
}
	
int main(int argc,char** argv) {
	const size_t rows=atoi(argv[1]);
	//const char *path=argv[2];
	
	assert((SIZE_MAX/rows*2) > (rows-1));
	size_t totalSize=(rows)*(rows-1)/2; /* number of value in the flat vector storage of the triangle matrix*/
	
	// allocate the memory even though we are in main, because this is potentially quite large
	float* input=malloc(totalSize*sizeof(float));
	assert(input!=NULL);
	
	//size_t read=fread(input,sizeof(float),totalSize,stdin);
	size_t readCount=0;
	float value;
	// read in the binary file (fread) one float at the time
	while(fread(&value,sizeof(float),1,stdin)==1) {
		assert(readCount<totalSize);
		input[readCount]=value;
		readCount++;
	}
	fprintf(stderr,"%zu / %zu\n",readCount,totalSize);
	
	// check if the number of read items is really the same as the expected number
	// this will result in an error if the file is shorter than expected
	assert(readCount==totalSize);

	float output[rows];
	for(size_t i=0;i<rows;i++) {
		fprintf(stderr,"%zu / %zu \r",i,rows);
		for(size_t j=0;j<rows;j++) {
			if(i==j) {
				output[j]=1.0;	
			}else{
				size_t offset=getIndex(j,i,rows);
				if(offset>=totalSize) {
					fprintf(stderr,"%zu %zu %zu %zu %zu\n",i,j,rows,offset,totalSize);
					abort();
				}
				assert(offset<totalSize);
				output[j]=input[offset];
			}
		}
		char name[200];
		sprintf(name,"%zu.bin",i);
		FILE* out=fopen(name,"w");
		if(out==NULL) {
			fprintf(stderr,"%x %i %s\n",errno,errno,name);
		}
		assert(out!=NULL);
		
		// print the output in binary to the file (1 file per probe)
		size_t written=fwrite(output,sizeof(float),rows,out);
		assert(written==rows);
		fclose(out);
		sync();
	}
	fprintf(stderr,"\n");
}
