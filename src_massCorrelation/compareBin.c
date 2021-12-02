/*
@file
@author DINH Viet Huy Hubert <dinh@ifrec.osaka-u.ac.jp>
@version 1.0
@section DESCRIPTION
compare 2 binary file containing float, 
print an error when the values differ from an arbitrary threshold (10e-4)
stop when a number of errors reach another arbitrary value (10)

@TODO
- user friendly I/O error handling
- user defined threshold (argument)
- user defined error limit
**/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main(int argc,char** argv) {
	if(argc<3) {
		fprintf(stderr, 
			"Usage : \n"
			"%s file1 file2\n", argv[0]);
	}
	FILE *file1=fopen(argv[1],"r");
	assert(file1!=NULL);
	FILE *file2=fopen(argv[2],"r");
	assert(file2!=NULL);
	float v1=0.0;
	float v2=0.0;
	int errorCount=0;
	while((!feof(file1)) && (!feof(file2))) {
		off_t offset=ftell(file1); /* Current offset */
		size_t read=fread(&v1,sizeof(float),1,file1);
		int end=0;
		if(read==0) {
			if(feof(file1)) {
				fprintf(stderr, "end of [%s]\n", argv[1]);
			}else{
				fprintf(stderr, "error reading [%s]\n", argv[1]);
			}
			end=1;
		}
		read=fread(&v2,sizeof(float),1,file2);
		if(read==0) {
			if(feof(file2)) {
				fprintf(stderr, "end of [%s]\n", argv[2]);
			}else{
				fprintf(stderr, "error reading [%s]\n", argv[2]);
			}
			end=1;
		}
		if(end==0) {
			if(fabs(v1-v2)>10e-4) { /* diff threshold here */
				fprintf(stderr,"%f %f %jd\n",v1,v2,offset);
				errorCount++;
			}
		}else{
			break;
		}
		if(errorCount>10) { /* error limit here*/
			fprintf(stderr, "too many error (abort)\n" );
			abort();
		}
	}
	fclose(file1);
	fclose(file2);
	return(EXIT_SUCCESS);
}
