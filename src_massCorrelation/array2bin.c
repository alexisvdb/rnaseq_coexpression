#include <stdlib.h>
#include <stdio.h>
#include "arrayIO.h"
#include <assert.h>

int main(int argc,char** argv) {
	size_t cols=0;
	size_t rows=0;
	float* input=readInputVectors(stdin,&cols,&rows);
	size_t out=0;
	out=fwrite(&cols,sizeof(size_t),1,stdout);
	assert(out==1);
	out=fwrite(&rows,sizeof(size_t),1,stdout);
	assert(out==1);
	size_t all=cols*rows;
	out=fwrite(input,sizeof(float),all,stdout);
	fprintf(stderr,"%i\t%i\n",out,all);
	assert(out==all);
}

