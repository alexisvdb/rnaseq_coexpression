#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifndef MAX_PATH
#define MAX_PATH	1024
#endif

off_t getFileSize(const char* path) {
	struct stat data;
	int ret=stat(path,&data);
	if(ret!=0) {
		fprintf(stderr,"error reading file [%s][%x]",path,errno);
		return 0;
	}
	return data.st_size;
}

float* readProbe(const char* path,size_t* length) {
	off_t size=getFileSize(path);
	fprintf(stderr, "[%s][%zu]\n", path,size);
	if(size==0) {
		return NULL;
	}
	FILE* input = fopen(path,"r");
	assert(input!=NULL);
	*length=size/sizeof(float);
	float* result=malloc(size);
	int ret=fread(result,size,1,input);
	assert(ret==1);
	fclose(input);
	return result;
}

int main(int argc, char* argv[]) {
	assert(argc>1);
	bool modeMax=(strcmp(argv[1],"max")==0);
	
	char path[MAX_PATH];
	float* result=NULL;
	int* origin=NULL;
	size_t length=0;
	int n=1;
	while(fgets(path,MAX_PATH,stdin)!=NULL) {
		for(int i=0;i<1024;i++) {
			if(path[i]=='\n') {
				path[i]=0;
				break;
			}
		}
		if(strlen(path)>0) {
			if(result==NULL) {
				result=readProbe(path,&length);
				if(result!=NULL) {
					origin=calloc(length,sizeof(int));
				}
			}else{
				size_t readLength=0;
				float* read=readProbe(path,&readLength);
				if(read!=NULL) {
					assert(length==readLength);
					for(size_t i=0;i<readLength;i++) {
						if(modeMax) {
							if(read[i]>result[i]) {
								origin[i]=n;
								result[i]=read[i];
							}
						}else{
							if(read[i]>result[i]) {
								origin[i]=n;
								result[i]=read[i];
							}
						}
					}
					free(read);
				}
				n++;
			}
		}
	}
	if(result==NULL) {
		return EXIT_FAILURE;
	}
	assert(result!=NULL);
	assert(origin!=NULL);
	for(size_t i=0;i<length;i++) {
		fprintf(stdout,"%0.6f\t%i\n",result[i],origin[i]);
	}
	free(result);
	free(origin);
	return EXIT_SUCCESS;
}

