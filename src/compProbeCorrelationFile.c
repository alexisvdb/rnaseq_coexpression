#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

int main(int argc,char** argv) {
	const char* fileA=argv[1];
	const char* fileB=argv[2];
	fprintf(stdout, "%s %s\n",fileA,fileB );
	struct stat statData;
	off_t fileSize=0;
	assert(stat(fileA,&statData)!=-1);
	fileSize = statData.st_size;
	assert(stat(fileB,&statData)!=-1);
	if(statData.st_size != fileSize ) {
		fprintf(stderr, "file size differs (%s : %u, %s : %u)\n",fileA,fileSize,fileB,statData.st_size );
		return EXIT_FAILURE;
	}
	FILE* inputA=fopen(fileA,"r");
	FILE* inputB=fopen(fileB,"r");
	assert(inputA!=NULL);
	assert(inputB!=NULL);
	float a=0.0;
	float b=0.0;
	const float epsilon=1e-6;
	int errorsCount=0;
	int index=0;
	while(fread(&a,sizeof(float),1,inputA)==1) {
		assert(fread(&b,sizeof(float),1,inputB)==1);
		if(fabsf(a-b)>epsilon) {
			fprintf(stdout, "%i %f %f\n",index, a,b);
			errorsCount++;
		}
		index++;
	}
	if(errorsCount!=0) {
		fprintf(stdout, "%i error(s)\n", errorsCount);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}