/**
@file
@author DINH Viet Huy Hubert <dinh@ifrec.osaka-u.ac.jp>
@version 1.1
@section DESCRIPTION

entry point for processing tab separated data to correlation values

**/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include "correlation.h"
#include "arrayIO.h"

static void die(const char*);
/**
 * print error message and exit program
 *
 * @param [in] message
 * a string containing the message to print before ending the program
 */
static void die(const char* message) {
	fprintf(stderr,"%s\n",message);
	abort();
}

static void showUsage(char* exeName) {
	fprintf(stderr, 
		"Usage :\n"
		"%s < inputFile.txt > output.bin\n"
		"%s -i inputfile.txt -o outputFile.txt\n"
		"arguments could be mixed with pipe style calling"
		, exeName,exeName);
	exit(0);
}

int main(int argc,char** argv) {
	FILE *input=stdin;
	FILE *output=stdout;
	char* inputPath=NULL;
	char* outputPath=NULL;

	int c;
	opterr=0;
	
	// this just processes the input parameters
	while((c = getopt(argc,argv,"i:o:h"))!= -1) {
		switch(c) {
			case 'i':
				inputPath=strdup(optarg);
				input=fopen(inputPath,"r");
				if(input==NULL) {
					fprintf(stderr, "error reading [%s]\n", inputPath);
					abort();
				}
				break;
			case 'o':
				outputPath=strdup(optarg);
				output=fopen(outputPath,"w");
				if(output==NULL) {
					fprintf(stderr, "error writing [%s]\n", outputPath);
					abort();
				}
				break;
			case 'h':
				showUsage(argv[0]);
				break;
			default:
				showUsage(argv[0]);
				break;

		}
	}

	if(isatty(fileno(output))||isatty(fileno(input))) {
		showUsage(argv[0]);
	}
	//free(malloc(1024));
	fprintf(stderr,"reading input\n");
	size_t cols=0;
	size_t rows=0;
	float* inputData=readInputVectors(input,&cols,&rows);

	if(inputPath!=NULL) {
		free(inputPath);
		inputPath=NULL;
		fclose(input);
	}

	fprintf(stderr,"read %zu cols ,%zu rows\n",cols,rows);
	//abort();
	float* outputData=getCorrelation(inputData,cols,rows);
	free(inputData);

	fprintf(stderr,"writing output data\n");
	
	// writing the correlations into a binary output file
	// unlike printf fwrite prints the data as it is in memory (so not human readable) 
	fwrite(outputData,sizeof(float),(rows*(rows-1)/2),output);
	free(outputData);
	
	if(outputPath!=NULL) {
		free(outputPath);
		outputPath=NULL;
		fclose(output);
	}
	fprintf(stderr,"done\n");
	return(EXIT_SUCCESS);
}

