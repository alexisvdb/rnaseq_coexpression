#include <stdlib.h>
#include <string.h>
#include "utils.h"

// Trim newline char at the end of the string 
extern int trimCR(char* subject,const size_t maxLength) {
	for(size_t i=0;i<maxLength;i++) { // loop from the start of the string (scanning from the end of the string is a possible optimisation )
		if(subject[i]=='\n') { // if the current char is a newline one (\n)
			subject[i]=0; // replace it with null char \0
			return i; // return current position
		}
	}
	return -1; // not found return -1
}

// read 
extern int readlineLoop(FILE* input,readlineCallbackProc_t readlineCallbackProc,void* params) {
	char* line=NULL;
	size_t size=0;
	int result=0;
	while(getline(&line,&size,input)!=-1) {
		int res=trimCR(line,size); // trim  newline character 
		if(strlen(line)>0) {
			readlineCallbackProc(line,size,params); // call user function
			result++; 
		}
		// Reinit vars for next getline
		free(line);
		line=NULL;
		size=0;
	}
	return result; // return read lines
}

