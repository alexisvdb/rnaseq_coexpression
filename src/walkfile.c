#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include "binIndex.h"
#include "binFormat.h"


int main(int argc, char *argv[]) {
	char* indexFile=argv[1];
	binIndex* index=binIndexLoad(indexFile);
	int count=binIndexGetCount(index);
	int i,j;
	for(i=0;i<count-1;i++) {
		for(j=i+1;j<count;j++) {
			scoreRecord record;
			if(fread(&record,sizeof(scoreRecord),1,stdin)==1) {
				if((record.a!=i)||(record.b!=j)) {
					fprintf(stderr,"%i:%i %i:%i\n",i,record.a,j,record.b);
					return(EXIT_FAILURE);
				}else{
					printf("%i %i\n",i,j);
				}
			}else{
				fprintf(stderr,"truncated\n");
				return(EXIT_FAILURE);
			}
		}
	}
	fprintf(stderr,"successfully ended\n");
	return(EXIT_SUCCESS);
}
